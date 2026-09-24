"""Artifact catalog: what reusable startup state exists, for which config, and where.

Stored in Postgres (fes/db.py), next to the job queue. Both control-plane roles open it: the
worker records starts (artifacts, engines, learned timings) and the API writes on stop / cancel
/ reconcile. Sandboxes never write it; they report on stdout. Within a process a lock
serializes read-modify-write updates; multi-statement updates run in one transaction. Result
pages open it read-only.

Tables
    artifacts(key, kind, inputs, bytes, last_used)
    artifact_locations(key, tier, uri, validated, created)   one artifact, many copies
    engines(engine_id, status, model, sandbox_id, url, expires_at, info)   info = full record
    stats(name, value)   learned timings (EWMA): shared rows, and per-model rows "<model>|<name>"
"""

import functools
import hashlib
import json
import threading
import time

from fes import db

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS artifacts (
        key TEXT PRIMARY KEY, kind TEXT NOT NULL, inputs TEXT NOT NULL, bytes BIGINT, last_used DOUBLE PRECISION)""",
    """CREATE TABLE IF NOT EXISTS artifact_locations (
        key TEXT NOT NULL REFERENCES artifacts(key) ON DELETE CASCADE, tier TEXT NOT NULL, uri TEXT NOT NULL,
        validated INTEGER NOT NULL DEFAULT 0, created DOUBLE PRECISION NOT NULL, PRIMARY KEY (key, tier, uri))""",
    """CREATE TABLE IF NOT EXISTS engines (
        engine_id TEXT PRIMARY KEY, status TEXT, model TEXT, sandbox_id TEXT, url TEXT, expires_at DOUBLE PRECISION,
        info TEXT NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS stats (name TEXT PRIMARY KEY, value DOUBLE PRECISION NOT NULL)",
]

# Priors from the vanilla baseline (results/20260923-184715, 2xH100, TP=2).
# Every launch refines these with an EWMA; see Catalog.observe().
DEFAULT_STATS = {
    "container_start_s": 14.0,     # sandbox create -> first exec
    "other_s": 107.3,              # imports, NCCL, model init, KV, warmup: ready - weights - graphs
    "graph_s_miss": 75.9,          # CUDA graph capture incl. Triton JIT (cold)
    "graph_s_hit": 33.0,           # prior from the warm run (29.3 s); replaced once measured
    "graph_s_hit_pow2": 9.6,       # power-of-two graph sizes, cache hit (measured 9.4 / 9.9 s, n=2)
    "graph_s_miss_pow2": 27.0,     # pow2 without cache: ~0.36x of full capture incl. JIT (estimate, unmeasured)
    "volume_gbps": 1.07,           # weights: volume -> GPU, cold page cache
    "hf_gbps": 0.54,               # weights: Hugging Face -> volume (CPU worker)
    "restore_gbps": 1.0,           # compile-cache tar: volume -> local disk (prior)
    "pre_load_s": 75.0,            # spawn -> "Load weight begin": the window prefetch can overlap
    "warm_load_s": 13.0,           # weight load with shards already in page cache (12.7 s measured)
    "prefetch_gbps": 4.3,          # weight prefetch read rate (30B on H100 hosts; a CPU sandbox read 1.3)
    "samples": 0,
}
EWMA = 0.5


def _hash(*parts: str) -> str:
    """Short stable key: sha256 over NUL-separated parts, 16 hex chars."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def weights_key(model_id: str, revision: str = "main") -> str:
    """Weights are identified by model and revision only (not by where they are stored)."""
    return "weights-" + _hash(model_id, revision)


def compile_key(sglang: str, gpu_name: str, tp: int, model_config_json: str) -> str:
    """Independent of the weights: a new revision of the same architecture reuses it.

    Must match sglang_engine.cache_key(), which computes it inside the container.
    """
    h = hashlib.sha256()
    for part in (sglang, gpu_name, str(tp), model_config_json):
        h.update(part.encode())
    return h.hexdigest()[:16]


def _locked(fn):
    """Read-modify-write must not interleave between threads (one connection, shared)."""
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    return wrapper


class Catalog:
    """The artifact catalog. Reads return plain dicts in the shapes the scheduler and UI use;
    writes go through @_locked methods. readonly=True for pages that only display it."""

    def __init__(self, url: str | None = None, readonly: bool = False):
        self._lock = threading.RLock()
        self.db = db.connect(url, readonly=readonly)
        if not readonly:
            db.create_schema(self.db, SCHEMA)

    # ---- reads
    @_locked
    def artifacts(self) -> dict:
        """key -> {kind, key, inputs, bytes, locations: [{tier, uri, validated, created}], last_used?}"""
        arts = {}
        for key, kind, inputs, nbytes, last_used in self.db.execute(
                "SELECT key, kind, inputs, bytes, last_used FROM artifacts ORDER BY key").fetchall():
            arts[key] = {"kind": kind, "key": key, "inputs": json.loads(inputs), "bytes": nbytes, "locations": []}
            if last_used is not None:
                arts[key]["last_used"] = last_used
        for key, tier, uri, validated, created in self.db.execute(
                "SELECT key, tier, uri, validated, created FROM artifact_locations ORDER BY created").fetchall():
            if key in arts:
                arts[key]["locations"].append({"tier": tier, "uri": uri, "validated": bool(validated), "created": created})
        return arts

    @_locked
    def engines(self) -> dict:
        """engine id -> its full registry record."""
        return {eid: json.loads(info) for eid, info in self.db.execute("SELECT engine_id, info FROM engines ORDER BY engine_id").fetchall()}

    @_locked
    def stats(self, model: str | None = None) -> dict:
        """Learned timings for `model` (its own rows, `<model>|<name>`), falling back to the
        shared estimates and then to the baseline priors. `model_samples` counts the model's
        own launches (0 = it is being predicted from other models)."""
        rows = dict(self.db.execute("SELECT name, value FROM stats").fetchall())
        shared = {k: v for k, v in rows.items() if "|" not in k}
        own = {k.split("|", 1)[1]: v for k, v in rows.items() if model and k.startswith(model + "|")}
        s = {**DEFAULT_STATS, **shared, **own}
        s["samples"] = int(shared.get("samples", 0))
        s["model_samples"] = int(own.get("samples", 0))
        s["own"] = sorted(k for k in own if k != "samples")  # which estimates are this model's own
        return s

    @_locked
    def per_model_stats(self) -> dict:
        """model -> its own learned timings (for the catalog page)."""
        out: dict = {}
        for k, v in self.db.execute("SELECT name, value FROM stats WHERE position('|' in name) > 0").fetchall():
            m, name = k.split("|", 1)
            out.setdefault(m, {})[name] = v
        return out

    # ---- artifacts
    def find(self, key: str, validated_only: bool = False) -> dict | None:
        """The artifact, or None. validated_only: only if some copy has served traffic."""
        a = self.artifacts().get(key)
        if not a:
            return None
        if validated_only and not any(l["validated"] for l in a["locations"]):
            return None
        return a

    @_locked
    def register(self, kind: str, key: str, inputs: dict, nbytes: int, tier: str, uri: str,
                 validated: bool = False) -> dict:
        """Record that artifact `key` has a copy at (tier, uri). Idempotent: re-registering a
        known copy keeps it, and can only upgrade it to validated. A size of 0 keeps the old one."""
        with self.db.transaction():
            self.db.execute("INSERT INTO artifacts (key, kind, inputs, bytes) VALUES (%s, %s, %s, %s) "
                            "ON CONFLICT (key) DO UPDATE SET bytes = COALESCE(NULLIF(EXCLUDED.bytes, 0), artifacts.bytes)",
                            (key, kind, json.dumps(inputs), nbytes))
            self.db.execute("INSERT INTO artifact_locations (key, tier, uri, validated, created) VALUES (%s, %s, %s, %s, %s) "
                            "ON CONFLICT (key, tier, uri) DO UPDATE SET "
                            "validated = GREATEST(artifact_locations.validated, EXCLUDED.validated)",
                            (key, tier, uri, int(validated), time.time()))
        return self.artifacts()[key]

    @_locked
    def remove(self, key: str) -> None:
        """Forget an artifact and all its copies (the data on the volume is left alone)."""
        self.db.execute("DELETE FROM artifacts WHERE key = %s", (key,))  # its copies cascade

    @_locked
    def remove_location(self, key: str, tier: str, uri: str) -> None:
        """Forget one copy; an artifact with no copies left is forgotten too."""
        with self.db.transaction():
            self.db.execute("DELETE FROM artifact_locations WHERE key = %s AND tier = %s AND uri = %s", (key, tier, uri))
            self.db.execute("DELETE FROM artifacts WHERE key = %s AND NOT EXISTS "
                            "(SELECT 1 FROM artifact_locations WHERE key = %s)", (key, key))

    @_locked
    def validate(self, key: str) -> None:
        """Mark every copy as proven good (an engine just served traffic with it)."""
        with self.db.transaction():
            self.db.execute("UPDATE artifact_locations SET validated = 1 WHERE key = %s", (key,))
            self.db.execute("UPDATE artifacts SET last_used = %s WHERE key = %s", (time.time(), key))

    # ---- engines
    def _write_engine(self, engine_id: str, info: dict) -> None:
        self.db.execute("INSERT INTO engines (engine_id, status, model, sandbox_id, url, expires_at, info) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (engine_id) DO UPDATE SET "
                        "status=EXCLUDED.status, model=EXCLUDED.model, sandbox_id=EXCLUDED.sandbox_id, url=EXCLUDED.url, "
                        "expires_at=EXCLUDED.expires_at, info=EXCLUDED.info",
                        (engine_id, info.get("status"), info.get("model"), info.get("sandbox_id"), info.get("url"),
                         info.get("expires_at"), json.dumps(info)))

    @_locked
    def update_engine(self, engine_id: str, **fields) -> None:
        """Merge fields into an engine's record; unknown engines are ignored."""
        row = self.db.execute("SELECT info FROM engines WHERE engine_id = %s", (engine_id,)).fetchone()
        if row:
            self._write_engine(engine_id, {**json.loads(row[0]), **fields})

    @_locked
    def add_engine(self, engine_id: str, info: dict) -> None:
        self._write_engine(engine_id, info)

    @_locked
    def remove_engine(self, engine_id: str) -> None:
        self.db.execute("DELETE FROM engines WHERE engine_id = %s", (engine_id,))

    # ---- learned timings
    @_locked
    def observe(self, model: str | None = None, **measured: float) -> dict:
        """Fold one launch's measurements into the learned timings (EWMA, weight 0.5): the
        model's own estimates if `model` is given, else the shared ones. A model's first
        measurement replaces the borrowed estimate outright. None means "not measured this
        time" and leaves that estimate unchanged."""
        s = self.stats(model)
        own = model is not None
        first = (s["model_samples"] if own else s["samples"]) == 0
        s.pop("own", None)
        updated = {}
        for k, v in measured.items():
            if v is None:
                continue
            # A model's first measurement of an estimate replaces the borrowed value outright.
            updated[k] = v if first or (own and k not in self.stats(model)["own"]) or k not in s else (1 - EWMA) * s[k] + EWMA * v
        updated["samples"] = (s["model_samples"] if own else s["samples"]) + 1
        prefix = f"{model}|" if own else ""
        with self.db.transaction(), self.db.cursor() as cur:
            cur.executemany("INSERT INTO stats (name, value) VALUES (%s, %s) "
                            "ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value",
                            [(prefix + k, v) for k, v in updated.items()])
        return self.stats(model)

    # ---- export (dashboard, /api/catalog)
    def dump(self) -> dict:
        """Everything, as one JSON-able snapshot."""
        return {"artifacts": self.artifacts(), "engines": self.engines(), "stats": self.stats(),
                "model_stats": self.per_model_stats(), "read_at": time.time()}
