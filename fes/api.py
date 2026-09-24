"""HTTP API and pages of the control plane (Flask). Reads and writes the databases only; jobs
are run by JobWorker (in this process with --role all, or another one with --role worker).

Pages   /  /engines  /jobs  /playground  /catalog  /compare  /analysis
Jobs    GET  /api/jobs?limit=50            POST /api/jobs {kind, params}
        GET  /api/jobs/<id>                POST /api/jobs/<id>/cancel
        GET  /api/jobs/<id>/events?after=N GET  /api/jobs/<id>/stream   (Server-Sent Events)
        kinds: start_engine {model, tp, gpu, groups, policy, ttl_s, graph_sizes: pow2|full,
                             prefetch, import_prefetch, modelexpress, extra_args}
               bench        {approaches, reps, scenario, serving}
Engines GET  /api/engines                  POST /api/engines/<id>/stop
        POST /api/engines/<id>/generate {prompt, mode: chat|raw, max_tokens, temperature}
Plan    GET  /api/plan?model=&tp=&gpu=&groups=&graph_sizes=&modelexpress=&policy=&ttl=&extra=
Catalog GET  /api/catalog                  POST /api/catalog/reconcile
Results GET  /api/compare   /api/runs   /api/run/<id>   /api/run/<id>/log/<n>   /api/architecture
Health  GET  /api/health    (queue counts, live workers, last reconciler pass)
"""

import importlib
import json
import time

import modal
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from fes import dashboard as ds  # dashboard parsing helpers, for /api/compare
from fes import scheduler
from fes.catalog import Catalog
from fes.jobs import STREAM_END, JobAPI, stop_engine
from fes.paths import WEB
from fes.store import Store

STATIC = str(WEB)


# ================================================================ Modal links

_volume_urls: dict[str, str | None] = {}


_last_sync = {"at": 0.0, "removed": []}


def sync_catalog(cat, max_age_s: float = 30) -> list[dict]:
    """Check the catalog's volume copies against Modal (at most every `max_age_s`), so a volume
    deleted by hand disappears from the Catalog page. Also forgets cached dashboard links."""
    if time.time() - _last_sync["at"] >= max_age_s:
        _last_sync["removed"] = scheduler.sync_catalog(cat)
        _last_sync["at"] = time.time()
        _volume_urls.clear()
    return _last_sync["removed"]


def volume_url(name: str) -> str | None:
    """The volume's page on the Modal dashboard (looked up once per name; None if it doesn't exist)."""
    if name not in _volume_urls:
        try:
            _volume_urls[name] = modal.Volume.from_name(name).get_dashboard_url()
        except Exception:
            return None  # not cached: the volume may simply not exist yet
    return _volume_urls[name]


# ================================================================ engine proxy

_served_names: dict[str, str] = {}


def _engine_call(url: str, path: str, body: dict | None = None, timeout: float = 180) -> dict:
    """POST (or GET, without a body) JSON to an engine through its tunnel URL."""
    import urllib.request
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(engine: dict, prompt: str, mode: str = "chat", max_tokens: int = 256,
             temperature: float = 0.7) -> dict:
    """Forward one prompt to a serving engine (browsers can't call it directly: no CORS)."""
    url = engine["url"]
    t0 = time.time()
    if mode == "chat":
        if url not in _served_names:  # the OpenAI-style API wants the served model name
            _served_names[url] = _engine_call(url, "/v1/models", timeout=30)["data"][0]["id"]
        r = _engine_call(url, "/v1/chat/completions", {
            "model": _served_names[url], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature})
        text = r["choices"][0]["message"]["content"]
        usage = r.get("usage", {})
        out_tokens = usage.get("completion_tokens")
    else:
        r = _engine_call(url, "/generate", {"text": prompt, "sampling_params": {
            "max_new_tokens": max_tokens, "temperature": temperature}})
        text = r["text"]
        meta = r.get("meta_info", {})
        usage = {"prompt_tokens": meta.get("prompt_tokens"), "completion_tokens": meta.get("completion_tokens")}
        out_tokens = meta.get("completion_tokens")
    dt = time.time() - t0
    return {"text": text, "mode": mode, "latency_s": round(dt, 3), "usage": usage,
            "tok_per_s": round(out_tokens / dt, 1) if out_tokens and dt else None}


# ================================================================ HTTP

PAGES = {"/": "scheduler.html", "/engines": "scheduler.html", "/jobs": "jobs.html", "/playground": "playground.html",
         "/catalog": "catalog.html", "/compare": "compare.html", "/analysis": "index.html"}
MILESTONES = ("phase", "ready", "trial_done", "trial_failed", "error", "plan")


def create_app(store: Store, jobs: JobAPI, cat: Catalog) -> Flask:
    """The API and pages. Routes and responses are the same as before the move to Flask, so the
    pages are unchanged."""
    app = Flask(__name__, static_folder=None)
    started_at = time.time()

    def job_view(job: dict) -> dict:
        """A job plus its recent milestone events (not the full log), for the job detail view."""
        return {**job, "milestones": store.events(job["id"], types=MILESTONES)[-40:]}

    # ---- errors are JSON, like every other response
    @app.errorhandler(KeyError)
    def _not_found(e):
        return jsonify(error=f"not found: {e}"), 404

    @app.errorhandler(ValueError)
    def _bad_request(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(Exception)
    def _server_error(e):
        code = getattr(e, "code", 500)  # werkzeug HTTP errors (404 for unknown routes, 405, ...)
        return jsonify(error=str(e) if code == 500 else getattr(e, "description", str(e))), code if isinstance(code, int) else 500

    @app.after_request
    def _no_store(resp):
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # ---- pages and assets
    for path, page in PAGES.items():
        app.add_url_rule(path, f"page_{page}_{path}", lambda page=page: send_from_directory(STATIC, page))

    @app.get("/viz.js")
    @app.get("/viz.css")
    def assets():
        return send_from_directory(STATIC, request.path.lstrip("/"))

    # ---- reads
    @app.get("/api/health")
    def health():
        recent = store.list(limit=200)
        ws = store.workers()
        return jsonify(ok=True, uptime_s=round(time.time() - started_at),
                       queued=sum(j["status"] == "queued" for j in recent),
                       running=sum(j["status"] == "running" for j in recent),
                       engines=len(cat.engines()), workers=ws,
                       reconciler=next((w.get("reconciler") for w in ws if w.get("reconciler")), {"at": None}))

    @app.get("/api/plan")
    def plan():
        q = request.args
        req = scheduler.Request.from_dict({"policy": q.get("policy", "late_claim"), "ttl_s": q.get("ttl", 900),
                                           "extra_args": q.get("extra", ""),
                                           **{k: q[k] for k in ("model", "tp", "gpu", "graph_sizes", "groups", "modelexpress") if q.get(k)}})
        return jsonify(scheduler.plan(req, cat).to_dict())

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify([job_view(j) for j in store.list(int(request.args.get("limit", 50)))])

    @app.get("/api/jobs/<jid>")
    def get_job(jid):
        j = store.get(jid)
        return jsonify(job_view(j)) if j else (jsonify(error="no such job"), 404)

    @app.get("/api/jobs/<jid>/events")
    def job_events(jid):
        return jsonify(store.events(jid, after=int(request.args.get("after", 0))))

    @app.get("/api/jobs/<jid>/stream")
    def job_stream(jid):
        """Server-sent events: replay a job's events after `after`, then follow live until it
        ends. A keepalive every 15 s stops proxies and browsers from dropping an idle stream."""
        if not store.get(jid):
            return jsonify(error="no such job"), 404
        after = int(request.args.get("after", request.headers.get("Last-Event-ID") or 0))

        def events():
            nonlocal after
            last_beat = time.time()
            while True:
                evs = store.events(jid, after=after)
                for e in evs:
                    yield f"id: {e['seq']}\ndata: {json.dumps(e, default=str)}\n\n"
                    after = e["seq"]
                job = store.get(jid)
                if job["status"] in STREAM_END and not evs:
                    yield f"event: end\ndata: {json.dumps({'status': job['status']})}\n\n"
                    return
                if time.time() - last_beat > 15:
                    yield ": keepalive\n\n"
                    last_beat = time.time()
                with store.new_event:  # same-process events wake us; a separate worker is polled
                    store.new_event.wait(timeout=1)

        return Response(stream_with_context(events()), mimetype="text/event-stream",
                        headers={"X-Accel-Buffering": "no"})

    @app.get("/api/engines")
    def engines():
        return jsonify(cat.engines())

    @app.get("/api/catalog")
    def catalog():
        from fes.catalog import DEFAULT_STATS
        from fes.common import MODEL_VOLUMES, artifacts_volume_name, model_volume_name
        sync_catalog(cat)
        d = cat.dump()
        models = set(MODEL_VOLUMES) | {e.get("model") for e in d["engines"].values() if e.get("model")} \
            | {a["inputs"].get("model") or a["inputs"].get("model_arch_of") for a in d["artifacts"].values()} - {None}
        volumes = {m: {"weights": model_volume_name(m), "artifacts": artifacts_volume_name(m)} for m in sorted(models)}
        names = {n for v in volumes.values() for n in v.values()}
        names |= {l["uri"].split(":")[0] for a in d["artifacts"].values() for l in a["locations"] if l["tier"] == "volume"}
        return jsonify({**d, "priors": DEFAULT_STATS, "volumes": volumes,
                        "volume_urls": {n: u for n in sorted(names) if (u := volume_url(n))}})

    @app.get("/api/compare")
    def compare():
        return jsonify(importlib.reload(ds).compare_data())  # comparison changes need no restart

    # ---- analysis (experiments, benchmark report, architecture); parsing changes need no restart
    @app.get("/api/runs")
    def runs():
        return jsonify(importlib.reload(ds).list_runs())

    @app.get("/api/run/<run_id>")
    def run(run_id):
        r = importlib.reload(ds).load_run(run_id)
        return jsonify(r) if r else (jsonify(error="not found"), 404)

    @app.get("/api/run/<run_id>/log/<int:n>")
    def run_log(run_id, n):
        if not ds.RUN_ID.match(run_id):
            return jsonify(error="not found"), 404
        return send_from_directory(ds.RESULTS, f"{run_id}/run{n}.log", mimetype="text/plain")

    @app.get("/api/architecture")
    def architecture():
        from fes.catalog import DEFAULT_STATS  # priors, to show learned-vs-prior
        importlib.reload(ds)
        return jsonify(catalog=ds.load_catalog(), scheduler_runs=ds.load_scheduler_runs(), priors=DEFAULT_STATS)

    # ---- writes
    def body() -> dict:
        return request.get_json(silent=True) or {}

    @app.post("/api/jobs")
    def submit():
        b = body()
        return jsonify(jobs.submit(b.get("kind", ""), b.get("params", {}))), 201

    @app.post("/api/jobs/<jid>/cancel")
    def cancel(jid):
        return jsonify(jobs.cancel(jid))

    @app.post("/api/engines/<eid>/generate")
    def engine_generate(eid):
        e = cat.engines().get(eid)
        if not e:
            return jsonify(error=f"unknown engine {eid}"), 404
        b = body()
        prompt = (b.get("prompt") or "").strip()
        if not prompt:
            return jsonify(error="empty prompt"), 400
        try:
            return jsonify(generate(e, prompt, b.get("mode", "chat"), max(1, min(int(b.get("max_tokens", 256)), 4096)),
                                    float(b.get("temperature", 0.7))))
        except Exception as ex:
            return jsonify(error=f"engine request failed: {ex}"), 502

    @app.post("/api/engines/<eid>/stop")
    def engine_stop(eid):
        if eid not in cat.engines():
            return jsonify(error=f"unknown engine {eid}"), 404
        stop_engine(cat, store, eid, by="stop")
        return jsonify(stopped=eid)

    @app.post("/api/catalog/reconcile")
    def reconcile():
        removed = sync_catalog(cat, max_age_s=0)
        return jsonify(removed=removed, found=scheduler.reconcile(cat), artifacts=cat.artifacts())

    return app
