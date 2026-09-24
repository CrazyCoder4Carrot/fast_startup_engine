"""Reconciler: keeps the engine registry, the job table and the artifact catalog true to what is on Modal."""

import threading
import time

import modal

from fes import scheduler, workers
from fes.catalog import Catalog
from fes.jobs import JobWorker, engine_end_status, finish_engine_job
from fes.store import Store

CATALOG_SYNC_EVERY = 10  # ticks (30 s each): check the catalog's volume copies every ~5 min


class Reconciler(threading.Thread):
    """Keep the engine registry true to what is actually running on Modal."""

    def __init__(self, cat: Catalog, store: Store, interval_s: int = 30, worker: "JobWorker | None" = None):
        super().__init__(daemon=True, name="reconciler")
        self.cat, self.store, self.interval, self.worker = cat, store, interval_s, worker
        self.last = {"at": None, "removed": [], "expired": []}
        self.ticks, self.catalog_removed = 0, []

    def tick(self):
        """One pass: drop engines whose sandbox has exited (the exit code says why), and stop
        engines past their lifetime. Either way the owning start job is closed."""
        removed, expired = [], []
        now = time.time()
        for eid, e in self.cat.engines().items():
            try:
                rc = modal.Sandbox.from_id(e["sandbox_id"]).poll()
            except Exception:
                rc = -1
            if rc is not None:
                self.cat.remove_engine(eid)
                status = engine_end_status(rc, e.get("expires_at"), stop_requested=False)
                finish_engine_job(self.store, e.get("job_id"), status, engine_id=eid, returncode=rc, by="reconciler")
                removed.append(eid)
            elif e.get("expires_at") and now > e["expires_at"]:
                try:
                    workers.stop_engine(e["sandbox_id"])
                except Exception:
                    pass  # already stopped with a sibling group
                self.cat.remove_engine(eid)
                finish_engine_job(self.store, e.get("job_id"), "timeout", engine_id=eid, by="reconciler")
                expired.append(eid)
        reaped = self.worker.reap_stale() if self.worker else []
        self.ticks += 1
        if self.ticks % CATALOG_SYNC_EVERY == 1:  # first tick, then every ~5 min
            self.catalog_removed = scheduler.sync_catalog(self.cat)
        self.last = {"at": now, "removed": removed, "expired": expired, "reaped": reaped,
                     "catalog_removed": self.catalog_removed}

    def run(self):
        while True:
            try:
                self.tick()
            except Exception as e:
                self.last = {"at": time.time(), "error": str(e)}
            time.sleep(self.interval)
