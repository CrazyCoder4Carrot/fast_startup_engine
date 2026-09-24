"""Control-plane entrypoint: starts the API, the worker, or both, sharing one database.

    uv run fes-control-plane                  # both in one process: http://127.0.0.1:8060
    uv run fes-control-plane --role worker    # claims queued jobs and runs them (+ reconciler)
    uv run fes-control-plane --role api       # UI + API only: writes jobs, runs nothing

As two processes, restarting the API never interrupts a running job. This file only wires the
pieces together; each lives in its own module:
    api.py         HTTP API (Flask) and pages; the full route list is there
    db.py          Postgres connection (docker compose up -d; DATABASE_URL)
    store.py       jobs (the queue), events, worker heartbeats
    jobs.py        JobAPI (validates, writes job rows, cancels) and JobWorker (claims a job,
                   runs the scheduler or a benchmark, heartbeats; one GPU job at a time)
    reconciler.py  every 30 s: engine exit codes and lifetimes, jobs of dead workers
    scheduler.py   plan() and execute() for engine starts, run by the worker
    catalog.py     artifacts, engines, learned timings; opened by both roles
"""

import argparse
import threading
from werkzeug.serving import make_server

from fes.api import create_app
from fes.catalog import Catalog
from fes.jobs import JobAPI, JobWorker
from fes.reconciler import Reconciler
from fes.db import DATABASE_URL, redacted
from fes.store import Store


def main():
    """Run the API, the worker, or both (default), all sharing one Postgres database.

    --role api     HTTP API + pages only: writes jobs to the database, runs nothing
    --role worker  claims queued jobs and runs them; also the reconciler (no HTTP)
    --role all     both in one process
    With separate processes, restarting the API never interrupts a running job.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("all", "api", "worker"), default="all")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--gpu-workers", type=int, default=1, help="max concurrent GPU jobs in this worker")
    ap.add_argument("--reconcile-every", type=int, default=30)
    args = ap.parse_args()

    store = Store()
    cat = Catalog()
    if args.role in ("all", "worker"):
        worker = JobWorker(store, cat, slots=args.gpu_workers)
        worker.reattach_serving()
        worker.reaped_at_start = worker.reap_stale()
        reconciler = Reconciler(cat, store, args.reconcile_every, worker=worker)
        worker.reconciler = reconciler
        reconciler.start()
        print(f"worker {worker.id}: {args.gpu_workers} slot(s), queue = {redacted(DATABASE_URL)}"
              + (f", reaped {worker.reaped_at_start}" if worker.reaped_at_start else ""), flush=True)
    if args.role == "worker":
        threading.Event().wait()  # the worker threads do the work
        return
    app = create_app(store, JobAPI(store, cat), cat)
    print(f"control plane API: http://{args.host}:{args.port}  (role {args.role}, Flask, db {redacted(DATABASE_URL)})", flush=True)
    make_server(args.host, args.port, app, threaded=True).serve_forever()


if __name__ == "__main__":
    main()
