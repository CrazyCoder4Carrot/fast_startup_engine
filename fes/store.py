"""Control-plane state store: jobs, their event logs, and workers, in Postgres (fes/db.py).

The jobs table is also the job queue (see Store). Shared by the API and worker processes.
"""

from __future__ import annotations  # Store.list shadows the builtin inside the class body

import json
import os
import threading
import time
import uuid

import psycopg

from fes import db

JOB_COLS = ("id", "kind", "status", "params", "result", "error", "created", "started", "ended",
            "cancel_requested", "worker", "heartbeat")
HEARTBEAT_S = 5      # a worker refreshes its running jobs' heartbeat this often
STALE_AFTER_S = 30   # a running job with an older heartbeat belongs to a dead worker

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, kind TEXT, status TEXT, params TEXT, result TEXT, error TEXT,
        created DOUBLE PRECISION, started DOUBLE PRECISION, ended DOUBLE PRECISION,
        cancel_requested INTEGER NOT NULL DEFAULT 0, worker TEXT, heartbeat DOUBLE PRECISION)""",
    "CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, created)",
    """CREATE TABLE IF NOT EXISTS events (
        job_id TEXT, seq INTEGER, ts DOUBLE PRECISION, type TEXT, data TEXT, PRIMARY KEY (job_id, seq))""",
    """CREATE TABLE IF NOT EXISTS workers (
        id TEXT PRIMARY KEY, host TEXT, pid INTEGER, started DOUBLE PRECISION, heartbeat DOUBLE PRECISION, info TEXT)""",
]


class Store:
    """Jobs + append-only event log in Postgres. This database is also the job queue.

    The API (frontend side) only writes rows: a new job is a row with status 'queued', and
    cancelling sets `cancel_requested`. Workers (backend side, possibly other processes or
    hosts) claim queued rows with FOR UPDATE SKIP LOCKED, heartbeat while running, and watch
    the cancel flag. One connection per Store, guarded by a lock (threads share it).
    """

    def __init__(self, url: str | None = None):
        self.db = db.connect(url)
        self.lock = threading.Lock()
        self.new_event = threading.Condition()  # same-process wakeups; other processes poll
        with self.lock:
            db.create_schema(self.db, SCHEMA)

    def _row(self, r) -> dict:
        d = dict(zip(JOB_COLS, r))
        d["params"] = json.loads(d["params"] or "{}")
        d["result"] = json.loads(d["result"]) if d["result"] else None
        d["cancel_requested"] = bool(d["cancel_requested"])
        return d

    def create(self, kind: str, params: dict) -> dict:
        jid = uuid.uuid4().hex[:10]
        with self.lock:
            self.db.execute("INSERT INTO jobs (id, kind, status, params, created) VALUES (%s, %s, %s, %s, %s)",
                            (jid, kind, "queued", json.dumps(params), time.time()))
        return self.get(jid)

    def update(self, jid: str, **fields) -> None:
        if "result" in fields:
            fields["result"] = json.dumps(fields["result"], default=str)
        cols = ", ".join(f"{k}=%s" for k in fields)
        with self.lock:
            self.db.execute(f"UPDATE jobs SET {cols} WHERE id=%s", (*fields.values(), jid))

    def get(self, jid: str) -> dict | None:
        with self.lock:
            r = self.db.execute(f"SELECT {', '.join(JOB_COLS)} FROM jobs WHERE id=%s", (jid,)).fetchone()
        return self._row(r) if r else None

    def list(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.db.execute(f"SELECT {', '.join(JOB_COLS)} FROM jobs ORDER BY created DESC LIMIT %s",
                                   (limit,)).fetchall()
        return [self._row(r) for r in rows]

    # ---- the queue
    def claim(self, worker_id: str) -> dict | None:
        """Take the oldest queued job, atomically: SKIP LOCKED lets many workers claim at once
        without two ever getting the same row."""
        now = time.time()
        with self.lock:
            r = self.db.execute(
                "UPDATE jobs SET status='running', started=%s, worker=%s, heartbeat=%s "
                "WHERE id=(SELECT id FROM jobs WHERE status='queued' ORDER BY created, id LIMIT 1 FOR UPDATE SKIP LOCKED) "
                "RETURNING id", (now, worker_id, now)).fetchone()
        return self.get(r[0]) if r else None

    def queue_position(self, jid: str) -> int:
        """How many queued jobs are ahead of this one."""
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued' AND created < "
                                   "(SELECT created FROM jobs WHERE id=%s)", (jid,)).fetchone()[0]

    def cancel_if_queued(self, jid: str) -> bool:
        """Cancel a job no worker has claimed yet (guarded, so it can't race a claim)."""
        with self.lock:
            return self.db.execute("UPDATE jobs SET status='cancelled', ended=%s WHERE id=%s AND status='queued'",
                                   (time.time(), jid)).rowcount == 1

    def request_cancel(self, jid: str) -> None:
        """Ask the worker running this job to stop; it checks the flag every second."""
        with self.lock:
            self.db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=%s", (jid,))

    def cancel_requested(self, jid: str) -> bool:
        with self.lock:
            r = self.db.execute("SELECT cancel_requested FROM jobs WHERE id=%s", (jid,)).fetchone()
        return bool(r and r[0])

    def heartbeat(self, jids: list[str], worker_id: str, info: dict) -> None:
        """Refresh the worker's row and the heartbeat of every job it is running."""
        now = time.time()
        with self.lock, self.db.transaction():
            self.db.execute("UPDATE workers SET heartbeat=%s, info=%s WHERE id=%s",
                            (now, json.dumps(info, default=str), worker_id))
            for jid in jids:
                self.db.execute("UPDATE jobs SET heartbeat=%s WHERE id=%s AND worker=%s", (now, jid, worker_id))

    def register_worker(self, worker_id: str) -> None:
        now = time.time()
        with self.lock:
            self.db.execute("INSERT INTO workers VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                            "host=EXCLUDED.host, pid=EXCLUDED.pid, started=EXCLUDED.started, "
                            "heartbeat=EXCLUDED.heartbeat, info=EXCLUDED.info",
                            (worker_id, os.uname().nodename, os.getpid(), now, now, "{}"))

    def workers(self) -> list[dict]:
        """Workers seen in the last minute (older rows are dead processes)."""
        with self.lock:
            rows = self.db.execute("SELECT id, host, pid, started, heartbeat, info FROM workers WHERE heartbeat > %s "
                                   "ORDER BY started", (time.time() - 60,)).fetchall()
        return [{"id": i, "host": h_, "pid": p, "started": s_, "heartbeat": hb, **json.loads(inf or "{}")}
                for i, h_, p, s_, hb, inf in rows]

    def stale_running(self) -> list[dict]:
        """Running jobs whose worker stopped heartbeating (crashed or was killed)."""
        with self.lock:
            ids = [r[0] for r in self.db.execute("SELECT id FROM jobs WHERE status='running' AND "
                                                 "COALESCE(heartbeat, started, 0) < %s",
                                                 (time.time() - STALE_AFTER_S,)).fetchall()]
        return [self.get(j) for j in ids]

    def append(self, jid: str, type_: str, data: dict) -> int:
        """Add an event with the job's next sequence number and wake any SSE streams.

        The API and a worker may append to the same job at once; a clash on (job_id, seq)
        just takes the next number."""
        payload = json.dumps(data, default=str)
        with self.lock:
            for _ in range(20):
                try:
                    seq = self.db.execute(
                        "INSERT INTO events (job_id, seq, ts, type, data) "
                        "SELECT %s, COALESCE(MAX(seq), 0) + 1, %s, %s, %s FROM events WHERE job_id=%s RETURNING seq",
                        (jid, time.time(), type_, payload, jid)).fetchone()[0]
                    break
                except psycopg.errors.UniqueViolation:
                    continue
            else:
                raise RuntimeError(f"could not append an event to job {jid}")
        with self.new_event:
            self.new_event.notify_all()
        return seq

    def events(self, jid: str, after: int = 0, types: tuple | None = None, limit: int = 5000) -> list[dict]:
        """Events after sequence number `after` (for resuming a stream), optionally of some types."""
        q = "SELECT seq, ts, type, data FROM events WHERE job_id=%s AND seq>%s"
        args: list = [jid, after]
        if types:
            q += " AND type = ANY(%s)"
            args.append(list(types))
        q += " ORDER BY seq LIMIT %s"
        args.append(limit)
        with self.lock:
            rows = self.db.execute(q, args).fetchall()
        return [{"seq": s, "ts": ts, "type": t, **json.loads(d)} for s, ts, t, d in rows]
