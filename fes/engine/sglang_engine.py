"""SGLang engine library, used inside GPU sandboxes (stdlib only; runs in SGLang's own Python).

Launch SGLang and wait until it serves, parse its startup phases, keep the compile cache
(key, restore, write-back), and pre-read weights and import files into the page cache.
Used by engine_agent.py (production engines) and bench_runner.py (benchmark trials);
uploaded to /root next to them and imported by plain name (`import sglang_engine`).
"""

import hashlib
import json
import os
import re
import subprocess
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = 30000
BASE = f"http://127.0.0.1:{PORT}"

PHASES = [
    ("server_args", r"server_args="),
    ("dist_begin", r"Init torch distributed begin"),
    ("dist_end", r"Init torch distributed end"),
    ("load_begin", r"Load weight begin"),
    ("load_end", r"Load weight end"),
    ("kv_alloc", r"KV Cache is allocated"),
    ("prefill_graph_begin", r"Capture .*prefill CUDA graph begin"),
    ("prefill_graph_end", r"Capture .*prefill CUDA graph end"),
    ("decode_graph_begin", r"Capture .*decode CUDA graph begin"),
    ("decode_graph_end", r"Capture .*decode CUDA graph end"),
    ("uvicorn", r"Uvicorn running on"),
    ("fired_up", r"The server is fired up"),
]

# Paths never counted as startup artifacts.
SKIP_ROOTS = ("/proc", "/sys", "/dev", "/models", "/results", "/artifacts", "/run", "/__modal")
CACHE_HINT = re.compile(r"cache|triton|inductor|flashinfer|deep_?gemm|jit|ComputeCache|torch_compile", re.I)

# launch() sends this once, greedy, to prove the server really serves before reporting ready.
FIRST_REQUEST_PROMPT = "The capital of France is"


def log(msg: str) -> None:
    print(f"[runner {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def post(path: str, body: dict, timeout: float = 300, base: str = BASE) -> dict:
    """POST JSON to the local SGLang server and return its JSON reply."""
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- compile cache

def cache_key(cfg: dict) -> str:
    """Compile-cache key: SGLang version, GPU model, TP and the model's config.json.

    Computed here from the real GPU name; must match catalog.compile_key(), which the
    scheduler computes before a GPU exists.
    """
    with open(os.path.join(cfg["model_path"], "config.json"), "rb") as f:
        model_cfg = f.read()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split("\n")[0].strip()
    h = hashlib.sha256()
    for part in (cfg["sglang"], gpu, str(cfg["tp"]), model_cfg.decode()):
        h.update(part.encode())
    return h.hexdigest()[:16]


def restore_cache(path: str) -> dict:
    """Unpack a compile-cache tar over / (it holds absolute paths under ~/.cache). Miss = no tar."""
    t0 = time.time()
    if not os.path.exists(path):
        log(f"compile cache miss: {path}")
        return {"hit": False, "restore_s": round(time.time() - t0, 2)}
    with tarfile.open(path) as tar:
        tar.extractall("/", filter="fully_trusted")
    size = os.path.getsize(path)
    log(f"compile cache restored: {size / 1e6:.0f} MB in {time.time() - t0:.1f}s")
    return {"hit": True, "restore_s": round(time.time() - t0, 2), "bytes": size}


def save_cache(path: str, dirs: list[str]) -> dict:
    """Tar `dirs` to `path` on the artifacts volume. Written to a temp name and renamed, so a
    concurrent reader never sees a partial tar."""
    t0 = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with tarfile.open(tmp, "w") as tar:
        for d in dirs:
            if os.path.exists(d):
                tar.add(d)
    os.replace(tmp, path)
    subprocess.run(["sync", os.path.dirname(path)], check=False)  # flush the v2 volume
    size = os.path.getsize(path)
    log(f"compile cache written: {size / 1e6:.0f} MB from {dirs} in {time.time() - t0:.1f}s")
    return {"dirs": dirs, "bytes": size, "save_s": round(time.time() - t0, 2)}


def discover_writes(marker: str, min_bytes: int = 1 << 20) -> list[dict]:
    """Directories (depth <= 4) that gained files after `marker` was touched."""
    out = subprocess.run(
        ["find", "/", "-xdev", "-type", "f", "-newer", marker, "-printf", "%s\t%p\n"],
        capture_output=True, text=True,
    ).stdout
    per_dir: dict[str, list[int]] = {}
    for line in out.splitlines():
        try:
            size, path = line.split("\t", 1)
        except ValueError:
            continue
        if path.startswith(SKIP_ROOTS):
            continue
        key = "/" + "/".join(path.strip("/").split("/")[:4])
        if os.path.isfile(key):
            key = os.path.dirname(key)
        agg = per_dir.setdefault(key, [0, 0])
        agg[0] += int(size)
        agg[1] += 1
    rows = [{"dir": d, "bytes": b, "files": n} for d, (b, n) in per_dir.items() if b >= min_bytes]
    return sorted(rows, key=lambda r: -r["bytes"])


def cache_roots(found: list[dict]) -> list[str]:
    """Collapse discovered dirs to their shallowest cache-looking ancestor."""
    roots = []
    for r in found:
        parts = r["dir"].strip("/").split("/")
        for i in range(1, len(parts) + 1):
            if CACHE_HINT.search(parts[i - 1]):
                cand = "/" + "/".join(parts[:i])
                if not any(cand == x or cand.startswith(x + "/") for x in roots):
                    roots = [x for x in roots if not x.startswith(cand + "/")] + [cand]
                break
    return sorted(roots)


# ---------------------------------------------------------------- launch

def launch(cfg: dict, log_path: str) -> tuple[subprocess.Popen, dict, list]:
    """Start `sglang.launch_server` and block until it has served one request.

    Every output line is timestamped relative to spawn and written to `log_path`; progress
    lines are echoed. Returns (process, {marks: phase -> seconds since spawn, first request
    timing, its output}, all timestamped lines). Raises if the server exits first.
    """
    port = cfg.get("port", PORT)
    base = f"http://127.0.0.1:{port}"
    tag = f"[{cfg['label']}] " if cfg.get("label") else ""  # engine group, when several share a host
    cmd = ["python", "-m", "sglang.launch_server", "--model-path", cfg["model_path"],
           "--tp", str(cfg["tp"]), "--host", cfg.get("host", "127.0.0.1"), "--port", str(port),
           *cfg["extra_args"]]
    log("$ " + " ".join(cmd))
    lines: list[tuple[float, str]] = []
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, **cfg.get("env", {})})

    progress = re.compile("|".join(p for _, p in PHASES) + r"|Traceback|Error")

    def reader():
        with open(log_path, "w") as f:
            for line in proc.stdout:
                t = time.time() - t0
                lines.append((t, line.rstrip()))
                f.write(f"{t:9.3f} {line}")
                f.flush()
                if progress.search(line) and "TP1]" not in line:
                    print(f"[{t:7.1f}s] {tag}{line.rstrip()[:200]}", flush=True)

    threading.Thread(target=reader, daemon=True).start()
    marks = {"spawn": 0.0}
    while True:
        if proc.poll() is not None:
            time.sleep(1)
            tail = "\n".join(l for _, l in lines[-30:])
            raise RuntimeError(f"sglang exited with code {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
                if r.status == 200:
                    break
        except OSError:
            pass
        time.sleep(0.25)
    marks["health_ok"] = time.time() - t0
    t1 = time.time()
    first = post("/generate", {"text": FIRST_REQUEST_PROMPT,
                               "sampling_params": {"max_new_tokens": 16, "temperature": 0}}, base=base)
    marks["first_request_done"] = time.time() - t0
    # A phase begins when the first TP rank logs it and ends when the last one does: the
    # slowest rank gates startup (at TP=8, "Load weight end" spans 64 s across ranks).
    for name, pat in PHASES:
        rx = re.compile(pat)
        hits = [t for t, line in lines if rx.search(line)]
        if hits:
            marks[name] = round(hits[-1] if name.endswith("_end") else hits[0], 3)
    return proc, {"marks": dict(sorted(marks.items(), key=lambda kv: kv[1])),
                  "first_request_s": round(time.time() - t1, 3), "output": first["text"]}, lines


# ---------------------------------------------------------------- ModelExpress (optional loader)
# https://github.com/ai-dynamo/modelexpress. Its GPU-to-GPU (NIXL/UCX) transfer can't start on
# Modal (NIXL_ERR_BACKEND, even between engines on one host), so every load falls back to its
# model_streamer loader, which reads the checkpoint from the volume (or the page cache).

MX_SERVER_PORT = 8001
REDIS_PORT = 6379


def modelexpress_args(group_index: int = 0) -> list[str]:
    """SGLang flags that route weight loading through ModelExpress."""
    return ["--load-format", "remote_instance", "--remote-instance-weight-loader-backend", "modelexpress",
            "--modelexpress-config", json.dumps({"transport": "nixl", "url": f"127.0.0.1:{MX_SERVER_PORT}"}),
            "--engine-info-bootstrap-port", str(6789 + group_index)]


def modelexpress_env(model_path: str, group_index: int = 0) -> dict:
    """Per-engine environment for ModelExpress's client (distinct ports per GPU group)."""
    return {"MX_SERVER_ADDRESS": f"127.0.0.1:{MX_SERVER_PORT}", "MX_METADATA_BACKEND": "redis",
            "MX_REDIS_URL": f"redis://127.0.0.1:{REDIS_PORT}", "MX_MODEL_URI": model_path, "MX_MS_DISTRIBUTED": "1",
            "MX_WORKER_GRPC_PORT": str(52000 + 100 * group_index), "MX_METADATA_PORT": str(53000 + 100 * group_index),
            "MODEL_EXPRESS_LOG_LEVEL": "info"}


def _wait_port(port: int, timeout: float = 60) -> None:
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"port {port} not up after {timeout}s")


def start_modelexpress(out: str) -> list[subprocess.Popen]:
    """Start Redis (metadata) and modelexpress-server inside the sandbox; weights never pass
    through them. Returns the processes (they end with the sandbox)."""
    t0 = time.time()
    redis = subprocess.Popen(["redis-server", "--port", str(REDIS_PORT), "--save", "", "--appendonly", "no"],
                             stdout=open(f"{out}/redis.log", "w"), stderr=subprocess.STDOUT)
    _wait_port(REDIS_PORT)
    env = {**os.environ, "MX_METADATA_BACKEND": "redis", "REDIS_URL": f"redis://127.0.0.1:{REDIS_PORT}",
           "MODEL_EXPRESS_SERVER_PORT": str(MX_SERVER_PORT), "MODEL_EXPRESS_SERVER_HOST": "127.0.0.1",
           "MODEL_EXPRESS_SERVER_METRICS_PORT": "0"}
    server = subprocess.Popen(["modelexpress-server"], env=env, stdout=open(f"{out}/mx-server.log", "w"),
                              stderr=subprocess.STDOUT)
    _wait_port(MX_SERVER_PORT)
    log(f"modelexpress: redis + server up in {time.time() - t0:.1f}s")
    return [redis, server]


MX_LOG = re.compile(r"Eligible loaders|Trying strategy|Strategy .*(succeeded|failed)|Loaded .* via|NIXL_ERR|model_streamer|fallback", re.I)


# ---------------------------------------------------------------- prefetch

def usable_cpus() -> int:
    """CPUs this process may run on (the sandbox's allocation, not the host's core count)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


# CPUs the sandbox was *requested* with (set by the engine agent from its cfg). Modal lets a
# sandbox see more (24 visible for a request of 8), but only the request is guaranteed.
CPUS_REQUESTED: int | None = None


def thread_cap(requested: int) -> int:
    """Never more I/O threads than 2x the CPUs: reads inside Modal's sandbox cost CPU too, and
    prefetch runs while SGLang imports (a single-threaded, CPU-bound ~15 s)."""
    cpus = min(CPUS_REQUESTED or usable_cpus(), usable_cpus())
    return max(1, min(requested, 2 * cpus))


def prefetch_weights(model_path: str, workers: int = 16, chunk_mb: int = 16) -> dict:
    """Pull every shard into the page cache: one whole file per thread (the fastest pattern
    measured), at most thread_cap(workers) threads.

    Started beside SGLang's launch so the reads overlap its imports and NCCL init instead of
    waiting behind them. Returns timing once all reads finish.
    """
    files = sorted(os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".safetensors"))
    chunk = chunk_mb << 20
    t0 = time.time()

    def read(path: str) -> int:
        n = 0
        with open(path, "rb", buffering=0) as f:
            while (b := f.read(chunk)):
                n += len(b)
        return n

    workers = thread_cap(workers)
    with ThreadPoolExecutor(workers) as ex:
        total = sum(ex.map(read, files))
    dt = time.time() - t0
    return {"bytes": total, "seconds": round(dt, 1), "gbps": round(total / 1e9 / dt, 2), "files": len(files),
            "threads": workers, "cpus_requested": CPUS_REQUESTED, "cpus_visible": usable_cpus()}


def prefetch_files(manifest: str, workers: int = 64, max_mb: int = 16) -> dict:
    """Read the files in an import manifest, many at a time, to pull them out of Modal's
    lazily fetched image filesystem before SGLang's (sequential) imports reach them.

    Files over `max_mb` are skipped: they are the big CUDA/torch libraries (5.0 of 5.2 GB),
    which are mmapped and paged in as used; reading them whole took 28-41 s on GPU hosts,
    competing with the weight prefetch. The 13.8k Python files are only 262 MB.
    """
    files = json.load(open(manifest))
    cap = max_mb << 20
    t0 = time.time()

    def read(path: str) -> int:
        try:
            if os.path.getsize(path) > cap:
                return 0
            with open(path, "rb", buffering=0) as f:
                n = 0
                while (b := f.read(8 << 20)):
                    n += len(b)
                return n
        except OSError:
            return 0

    workers = thread_cap(workers)
    with ThreadPoolExecutor(workers) as ex:
        total = sum(ex.map(read, files))
    return {"files": len(files), "bytes": total, "seconds": round(time.time() - t0, 2), "threads": workers}
