"""Jobs: the frontend side (JobAPI writes job rows), the backend side (JobWorker claims and
runs them), and how a start job ends once its engine is gone.
"""

import os
import threading
import time
import traceback
import uuid

import modal

from fes import bench
from fes import scheduler
from fes import workers
from fes.catalog import Catalog
from fes.store import HEARTBEAT_S, Store

# start_engine jobs: queued -> running -> serving -> timeout | stopped | failed
# bench jobs:        queued -> running -> succeeded | failed | cancelled
TERMINAL = ("succeeded", "failed", "cancelled", "interrupted", "timeout", "stopped")
STREAM_END = TERMINAL + ("serving",)  # a serving job's startup story is complete
EXIT_TIMEOUT, EXIT_KILLED = 124, 137  # Modal sandbox exit codes: lifetime cap vs terminate()
KINDS = ("start_engine", "bench")


# ================================================================ engine end states

def engine_end_status(returncode: int | None, expires_at: float | None, stop_requested: bool) -> str:
    """The exit code is authoritative; the clock is only a fallback when there is no code."""
    if stop_requested or returncode == EXIT_KILLED:
        return "stopped"
    if returncode == EXIT_TIMEOUT:
        return "timeout"
    if returncode is None and expires_at and time.time() >= expires_at - 30:
        return "timeout"
    return "failed"


def finish_engine_job(store: "Store", job_id: str | None, status: str, **info) -> None:
    """Close a serving start_engine job once its engine is gone."""
    if not job_id:
        return
    job = store.get(job_id)
    if not job or job["status"] in TERMINAL:
        return
    store.append(job_id, "engine_exited", {"status": status, **info})
    store.update(job_id, status=status, ended=time.time(),
                 **({"error": f"engine exited unexpectedly (code {info.get('returncode')})"} if status == "failed" else {}))
    store.append(job_id, status, info)


def stop_engine(cat: Catalog, store: "Store", eid: str, by: str) -> None:
    """Stop one engine. Engines sharing a sandbox (GPU groups) stop one at a time: the others
    keep serving, and the sandbox (and the job) ends with the last one."""
    e = cat.engines().get(eid)
    if not e:
        return
    siblings = [k for k, o in cat.engines().items() if k != eid and o["sandbox_id"] == e["sandbox_id"]]
    if siblings and e.get("port"):
        workers.stop_group(e["sandbox_id"], e["port"])
        cat.remove_engine(eid)
        store.append(e["job_id"], "engine_stopped", {"engine_id": eid, "by": by, "still_serving": siblings})
        return
    workers.stop_engine(e["sandbox_id"])
    cat.remove_engine(eid)
    finish_engine_job(store, e.get("job_id"), "stopped", engine_id=eid, by=by)


def terminate_job_sandboxes(store: Store, cat: Catalog, jid: str, known: set[str] = frozenset()) -> list[str]:
    """Terminate every sandbox a job created (from its event log), except a serving engine,
    which belongs to the engine registry once the job has handed it over."""
    ids = set(known) | {e["sandbox_id"] for e in store.events(jid, types=("sandbox",)) if e.get("sandbox_id")}
    engines = {e["sandbox_id"] for e in cat.engines().values()}
    killed = []
    for sid in ids:
        if sid in engines and store.get(jid)["status"] in ("succeeded", "serving"):
            continue
        try:
            workers.stop_engine(sid)
            killed.append(sid)
        except Exception:
            pass
    return killed


# ================================================================ jobs

class JobAPI:
    """Frontend side of the queue: validates and writes job rows; never runs anything."""

    def __init__(self, store: Store, cat: Catalog):
        self.store, self.cat = store, cat

    def submit(self, kind: str, params: dict) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r}; expected one of {KINDS}")
        if kind == "bench":
            unknown = [a for a in params.get("approaches", []) if a not in bench.APPROACHES]
            if unknown:
                raise ValueError(f"unknown approaches {unknown}")
        if kind == "start_engine":
            scheduler.Request.from_dict(params)  # reject bad params now, not when a worker picks it up
        job = self.store.create(kind, params)
        self.store.append(job["id"], "queued", {"position": self.store.queue_position(job["id"])})
        return job

    def cancel(self, jid: str) -> dict:
        job = self.store.get(jid)
        if not job:
            raise KeyError(jid)
        if job["status"] in TERMINAL:
            return job
        if job["status"] == "serving":  # nothing left to cancel: stop the engine(s) it started
            res = job["result"] or {}
            eids = list(res.get("engines") or [res.get("engine_id")])
            e = next((self.cat.engines()[x] for x in eids if x in self.cat.engines()), None)
            if e:
                workers.stop_engine(e["sandbox_id"])
            for x in eids:
                self.cat.remove_engine(x)
            finish_engine_job(self.store, jid, "stopped", engine_id=eids[0] if eids else None, by="cancel")
            return self.store.get(jid)
        if self.store.cancel_if_queued(jid):
            self.store.append(jid, "cancelled", {"before_start": True})
            return self.store.get(jid)
        # Running: flag it for the worker, and terminate its sandboxes now so no GPU keeps
        # billing while the worker notices (their ids are already in the event log).
        self.store.request_cancel(jid)
        killed = terminate_job_sandboxes(self.store, self.cat, jid)
        self.store.append(jid, "cancel_requested", {"terminated_sandboxes": killed})
        return self.store.get(jid)


class JobWorker:
    """Backend side of the queue: claims queued jobs from the database and runs them.

    `slots` threads (default 1: one GPU job at a time) each loop claim -> run. While a job
    runs, a monitor thread heartbeats it and turns the database cancel flag into the job's
    in-process cancel event. start_engine jobs don't end when the engine is ready: they
    stay `serving` until the engine is stopped, times out or dies (see finish_engine_job).
    """

    def __init__(self, store: Store, cat: Catalog, slots: int = 1, poll_s: float = 1.0):
        self.store, self.cat, self.poll_s = store, cat, poll_s
        self.id = f"{os.uname().nodename}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self.running: dict[str, threading.Event] = {}  # job id -> cancel event
        self.sandboxes: dict[str, set[str]] = {}       # job id -> sandbox ids it created
        self.store.register_worker(self.id)
        for i in range(slots):
            threading.Thread(target=self._loop, name=f"gpu-worker-{i}", daemon=True).start()
        threading.Thread(target=self._monitor, name="job-monitor", daemon=True).start()

    def _loop(self):
        while True:
            job = self.store.claim(self.id)
            if job is None:
                with self.store.new_event:  # woken early by a same-process submit
                    self.store.new_event.wait(timeout=self.poll_s)
                continue
            self._execute(job)

    def _monitor(self):
        """Heartbeat running jobs; propagate cancel requests made through the database."""
        last_beat = 0.0
        while True:
            for jid, ev in list(self.running.items()):
                if not ev.is_set() and self.store.cancel_requested(jid):
                    ev.set()
            if time.time() - last_beat >= HEARTBEAT_S:
                rec = getattr(self, "reconciler", None)
                self.store.heartbeat(list(self.running), self.id,
                                     {"running": list(self.running), "reconciler": rec.last if rec else None})
                last_beat = time.time()
            time.sleep(1)

    def _execute(self, job: dict):
        """Run one claimed job and map how it ended onto a terminal status."""
        jid = job["id"]
        cancelled = self.running.setdefault(jid, threading.Event())
        self.store.append(jid, "started", {"worker": self.id})

        def emit(type_, **data):  # every progress event lands in the job's event log
            if type_ == "sandbox" and data.get("sandbox_id"):
                self.sandboxes.setdefault(jid, set()).add(data["sandbox_id"])
            self.store.append(jid, type_, data)

        def cleanup():
            terminate_job_sandboxes(self.store, self.cat, jid, self.sandboxes.get(jid, set()))

        try:
            if job["cancel_requested"]:
                raise scheduler.Cancelled()
            result = self._run(job, emit, cancelled)
            if cancelled.is_set():
                raise scheduler.Cancelled()
            if job["kind"] == "start_engine":
                # Not over yet: the job now tracks its engine until timeout / stop / failure.
                self.store.update(jid, status="serving", result=result)
                self.store.append(jid, "serving", {"engine_id": result["engine_id"], "url": result["url"]})
            else:
                self.store.update(jid, status="succeeded", result=result, ended=time.time())
                self.store.append(jid, "succeeded", {})
        except scheduler.Cancelled:
            cleanup()
            self.store.update(jid, status="cancelled", ended=time.time())
            self.store.append(jid, "cancelled", {})
        except Exception as e:
            if cancelled.is_set():
                self.store.update(jid, status="cancelled", ended=time.time())
                self.store.append(jid, "cancelled", {})
            elif isinstance(e, workers.EngineTimeout):
                cleanup()
                self.store.update(jid, status="timeout", error=str(e), ended=time.time())
                self.store.append(jid, "timeout", {"error": str(e), "before_ready": True})
            else:
                cleanup()
                self.store.update(jid, status="failed", error=str(e)[-4000:], ended=time.time())
                self.store.append(jid, "error", {"error": str(e)[-4000:], "trace": traceback.format_exc()[-4000:]})
        finally:
            self.running.pop(jid, None)
            self.sandboxes.pop(jid, None)

    def _run(self, job: dict, emit, cancelled: threading.Event):
        """Dispatch on job kind; the return value becomes the job's `result`."""
        p = job["params"]
        if job["kind"] == "start_engine":
            req = scheduler.Request.from_dict(p)
            plan = scheduler.plan(req, self.cat)
            rec = scheduler.execute(plan, self.cat, emit=emit, cancelled=cancelled, job_id=job["id"],
                                    engine_id=f"{time.strftime('%Y%m%d-%H%M%S')}-{req.policy}")
            return {"engine_id": rec["engine_id"], "url": rec["url"], "engines": rec["engines"], "error_s": rec["error_s"],
                    "request_to_ready_s": rec["actual"]["request_to_ready_s"],
                    "startup_s": rec["actual"]["startup_s"], "scheduling_wait_s": rec["actual"]["scheduling_wait_s"],
                    "predicted_ready_s": round(rec["plan"]["predicted_ready_s"], 1)}
        if job["kind"] == "bench":
            rows = bench.run_matrix(p.get("approaches", ["baseline", "jit_cache"]), int(p.get("reps", 1)),
                                    p.get("scenario", "fresh_start"), serving=p.get("serving", True),
                                    emit=emit, cancelled=cancelled)
            return {"trials": rows}
        raise ValueError(job["kind"])

    # ---- recovery
    def reap_stale(self) -> list[str]:
        """Running jobs whose worker stopped heartbeating: mark them interrupted, don't
        resume, and terminate their sandboxes. Runs at startup and on every reconciler tick,
        so a crashed worker's jobs are cleaned up even while other workers keep going."""
        reaped = []
        for job in self.store.stale_running():
            if job["id"] in self.running:
                continue  # ours and alive (heartbeat just lagging)
            killed = terminate_job_sandboxes(self.store, self.cat, job["id"])
            self.store.update(job["id"], status="interrupted", ended=time.time(),
                              error=f"worker {job.get('worker') or '?'} stopped heartbeating while this job was running")
            self.store.append(job["id"], "interrupted", {"terminated_sandboxes": killed, "worker": job.get("worker")})
            reaped.append(job["id"])
        return reaped

    def reattach_serving(self):
        """Start jobs whose engine may still be up are re-attached to the engine registry;
        if the engine is gone, its exit code decides timeout / stopped / failed."""
        live_engines = self.cat.engines()
        for job in self.store.list(limit=500):
            eid = (job["result"] or {}).get("engine_id")
            if not (job["kind"] == "start_engine" and job["status"] in ("serving", "succeeded") and eid):
                continue
            if any(x in live_engines for x in ((job["result"] or {}).get("engines") or [eid])):
                if job["status"] != "serving":
                    self.store.update(job["id"], status="serving", ended=None)
                continue
            sids = [e["sandbox_id"] for e in self.store.events(job["id"], types=("sandbox",)) if e.get("role") == "engine"]
            rc = None
            if sids:
                try:
                    rc = modal.Sandbox.from_id(sids[-1]).poll()
                except Exception:
                    pass
            if rc is None and sids:
                continue  # still running but not registered; leave it for an operator
            self.store.update(job["id"], status="serving")  # so finish_engine_job applies
            ttl = int(job["params"].get("ttl_s", 1800))
            finish_engine_job(self.store, job["id"], engine_end_status(rc, (job["started"] or 0) + ttl, False),
                              engine_id=eid, returncode=rc, by="recovery")
