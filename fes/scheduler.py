"""Scheduler library: decide what to reuse, what to prepare, and when to claim GPUs.

Run by the control plane's worker (jobs.JobWorker) for start_engine jobs; the API calls
plan() alone for the Engines page's live prediction.

    p = plan(Request(...), catalog)                 # pure, free: steps + predicted times
    record = execute(p, catalog, emit, cancelled)   # runs the plan, emits structured events

Policy
    late_claim (default)  all CPU-side prep (weights -> volume) finishes before a GPU is requested
    asap                  claim the GPU immediately; missing weights are fetched on the GPU box

execute() emits events as emit(type, **data):
    plan · prep_started · prep_done · sandbox(role, sandbox_id) · log(line) · phase(name, t)
    · writeback(...) · ready(engine_id, engines, url, startup_s, scheduling_wait_s, ...)
and returns the record.
Every start logs predicted vs actual to results/scheduler/<engine_id>.json and feeds
the measured timings back into the catalog's stats.
"""

import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable

import modal

from fes import workers
from fes.paths import RESULTS
from fes.catalog import Catalog, compile_key, weights_key
from fes.common import (MODEL_ID, SGLANG_VERSION, artifacts_volume, artifacts_volume_name, engine_cpus, engine_memory,
                    model_path, model_volume, model_volume_name)

# What nvidia-smi reports inside the container; part of the compile-cache key.
GPU_NAMES = {"H100": "NVIDIA H100 80GB HBM3", "H200": "NVIDIA H200", "B200": "NVIDIA B200"}
DEFAULT_WEIGHT_BYTES = 61e9  # used only when the size can't be looked up
# Prefetch keeps the whole checkpoint in page cache, so the plan reserves host RAM sized from
# the checkpoint (common.engine_memory: weights x2 + 64 GiB, limit 180 GiB per GPU) and
# prefetches whenever that fits: 61 GB -> 178 GiB on 2 GPUs, 470 GB -> 940 GiB on 8.
PREFETCH = {"workers": 16, "chunk_mb": 16}
# CUDA graph sizes. "pow2" captures 9 decode + 12 prefill shapes instead of SGLang's 94:
# capture 26.8 -> 9.6 s, throughput -2%, per-token latency +0.05 ms, identical output (n=2 each,
# graphs_on vs graphs_pow2). SGLang pads each batch up to the next captured size.
GRAPH_FLAGS = {
    "pow2": ["--cuda-graph-bs-decode", *[str(2 ** i) for i in range(9)],       # 1..256
             "--cuda-graph-bs-prefill", *[str(2 ** i) for i in range(2, 14)]],  # 4..8192 tokens
    "full": [],
}
LOG_DIR = str(RESULTS / "scheduler")

Emit = Callable[..., None]


class Cancelled(Exception):
    """Raised inside execute() once the job's cancel flag is set."""


# Shortest lifetime a model may be started with. The sandbox timeout counts from container
# start and includes startup: 235B needs ~11 min to be ready (weights alone ~8 min) and can
# wait minutes for 8xB200, so 5 min killed it mid weight-load (job c8bd7f4ed5).
DEFAULT_MIN_TTL_S = 5 * 60
MIN_TTL_S = {"Qwen/Qwen3-235B-A22B-Instruct-2507": 15 * 60}


def min_ttl(model: str) -> int:
    """Shortest allowed lifetime for `model`, in seconds."""
    return MIN_TTL_S.get(model, DEFAULT_MIN_TTL_S)


@dataclass
class Request:
    """What the caller asked for: model, hardware and startup options. Built from API params
    by from_dict(), which validates them and applies per-model defaults."""

    model: str = MODEL_ID
    tp: int = 2
    gpu: str = "H100!:2"
    policy: str = "late_claim"
    extra_args: list[str] = field(default_factory=list)
    ttl_s: int = 30 * 60
    prefetch: bool = True  # overlap weight reads with SGLang's imports
    import_prefetch: bool = True  # pre-read the image files SGLang's imports need
    modelexpress: bool = False  # load weights through ModelExpress (its streamer; P2P can't run on Modal)
    groups: int = 1  # engines per host, each on its own `tp` GPUs (2 -> A: GPU 0-1, B: GPU 2-3 at TP=2)
    graph_sizes: str = "pow2"  # "pow2" | "full"; ignored if extra_args already set CUDA-graph sizes

    @classmethod
    def from_dict(cls, d: dict) -> "Request":
        """Lifetimes below the model's minimum are raised to it (see MIN_TTL_S)."""
        extra = d.get("extra_args", d.get("extra", []))
        if isinstance(extra, str):
            extra = extra.split()
        tp, groups = int(d.get("tp", 2)), int(d.get("groups", 1))
        if not 1 <= groups <= 8 // tp:
            raise ValueError(f"groups must be 1..{8 // tp} at TP={tp}")
        gpu = d.get("gpu", "H100!:2")
        if groups > 1:  # the host needs tp GPUs per group
            gpu = f"{gpu.split(':')[0]}:{tp * groups}"
        return cls(model=d.get("model", MODEL_ID), tp=tp, gpu=gpu, groups=groups,
                   policy=d.get("policy", "late_claim"), extra_args=extra, ttl_s=max(int(d.get("ttl_s", d.get("ttl", 1800))), min_ttl(d.get("model", MODEL_ID))),
                   prefetch=str(d.get("prefetch", True)).lower() not in ("0", "false", "no"),
                   import_prefetch=str(d.get("import_prefetch", True)).lower() not in ("0", "false", "no"),
                   modelexpress=str(d.get("modelexpress", False)).lower() in ("1", "true", "yes"),
                   graph_sizes=d.get("graph_sizes", "pow2") if d.get("graph_sizes", "pow2") in GRAPH_FLAGS else "pow2")

    def group_layout(self) -> list[dict]:
        """GPU groups on the host: name, CUDA_VISIBLE_DEVICES, port. Empty for a single engine."""
        if self.groups == 1:
            return []
        return [{"name": chr(ord("a") + i), "devices": ",".join(str(i * self.tp + j) for j in range(self.tp)),
                 "port": 30000 + 10 * i} for i in range(self.groups)]

    def effective_graphs(self) -> str:
        """User-supplied CUDA-graph flags win over the preset."""
        if any(a.startswith(("--cuda-graph-bs", "--cuda-graph-max-bs", "--disable-cuda-graph",
                             "--disable-prefill-cuda-graph", "--disable-decode-cuda-graph")) for a in self.extra_args):
            return "custom"
        return self.graph_sizes

    def engine_args(self) -> list[str]:
        """Extra SGLang flags: the caller's own plus the CUDA-graph preset's."""
        g = self.effective_graphs()
        return self.extra_args + (GRAPH_FLAGS[g] if g in GRAPH_FLAGS else [])


@dataclass
class Step:
    """One planned step with its estimated duration; `where` says whether GPUs are billed."""

    where: str   # "cpu" | "gpu"
    action: str
    est_s: float
    detail: str = ""


@dataclass
class Plan:
    """The scheduler's decision for a request: which artifacts exist, the steps to run, and
    the predicted time to ready (sum of the steps' estimates)."""

    request: Request
    weights_key: str
    weights_bytes: float
    weights_on_volume: bool
    compile_key: str | None
    compile_hit: bool
    steps: list[Step]
    prefetch: bool
    memory_mib: int        # host RAM requested for the GPU sandbox, sized from the checkpoint
    memory_limit_mib: int  # its hard limit (per-GPU cap)
    predicted_cpu_s: float
    predicted_gpu_s: float
    predicted_ready_s: float

    def show(self) -> str:
        """Human-readable plan: artifacts, then a timeline of steps with running totals."""
        r = self.request
        out = [f"plan for {r.model} TP={r.tp} on {r.gpu} (policy={r.policy})",
               f"  weights  {self.weights_key}: {'on volume' if self.weights_on_volume else 'MISSING'} "
               f"({self.weights_bytes / 1e9:.1f} GB)",
               f"  compile  {self.compile_key or '(needs weights config first)'}: "
               f"{'HIT' if self.compile_hit else 'miss -> write back after ready'}",
               f"  prefetch {'on (overlapped with imports)' if self.prefetch else 'off (checkpoint too big for host RAM)'}",
               f"  memory   request {self.memory_mib / 1024:.0f} GiB host RAM ({self.weights_bytes / 1e9:.0f} GB weights x2 + 64 GiB), "
               f"limit {self.memory_limit_mib / 1024:.0f} GiB (180 GiB per GPU)",
               f"  loader   {'ModelExpress (model_streamer; GPU-to-GPU unavailable on Modal)' if r.modelexpress else 'SGLang default'}",
               f"  graphs   {self.request.effective_graphs()}"
               + (" (9 decode + 12 prefill sizes)" if self.request.effective_graphs() == "pow2" else "")]
        t = 0.0
        for s in self.steps:
            out.append(f"  [{s.where}] +{t:6.1f}s  {s.action:<28} ~{s.est_s:6.1f}s  {s.detail}")
            t += s.est_s
        out.append(f"  predicted: ready {self.predicted_ready_s:.0f}s after request, "
                   f"GPU billed {self.predicted_gpu_s:.0f}s, CPU prep {self.predicted_cpu_s:.0f}s")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {**asdict(self), "text": self.show()}


def gpu_name(gpu_spec: str) -> str:
    """Modal GPU spec ("H100!:2") -> the device name the engine reports ("NVIDIA H100 80GB HBM3")."""
    base = gpu_spec.split(":")[0].rstrip("!")
    return GPU_NAMES.get(base, base)


def read_model_config(model_id: str) -> str | None:
    """The model's config.json from its weights volume (an input to the compile key)."""
    try:
        return b"".join(model_volume(model_id, create=False).read_file(f"{model_id}/config.json")).decode()
    except Exception:
        return None


def hub_bytes(model_id: str) -> int | None:
    """Checkpoint size from Hugging Face metadata, for models not seen before."""
    try:
        url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
        with urllib.request.urlopen(url, timeout=10) as r:
            files = json.loads(r.read()).get("siblings", [])
        return sum(f.get("size") or 0 for f in files if f["rfilename"].endswith(".safetensors")) or None
    except Exception:
        return None


def weights_complete(model_id: str) -> tuple[bool, list[str]]:
    """Same completeness rule as the prep worker, checked from here against the volume."""
    try:
        files = {e.path.split(model_id + "/", 1)[-1] for e in model_volume(model_id, create=False).listdir(model_id, recursive=True) if e.type.name == "FILE"}
    except Exception:
        return False, ["(model directory missing)"]
    need = ["config.json"]
    if "model.safetensors.index.json" in files:
        idx = json.loads(b"".join(model_volume(model_id, create=False).read_file(f"{model_id}/model.safetensors.index.json")))
        need += sorted(set(idx["weight_map"].values()))
    elif "model.safetensors" not in files:  # sharded checkpoints always ship an index
        need.append("model.safetensors.index.json")
    missing = [f for f in need if f not in files]
    if "tokenizer.json" not in files and not {"vocab.json", "merges.txt"} <= files:
        missing.append("tokenizer files")
    return not missing, missing


def volume_bytes(vol, path: str) -> int:
    """Total size of the files under `path` on a volume (0 if it doesn't exist)."""
    try:
        return sum(e.size for e in vol.listdir(path, recursive=True) if e.type.name == "FILE")
    except Exception:
        return 0


# ---------------------------------------------------------------- planning

def plan(req: Request, cat: Catalog) -> Plan:
    """Decide how to start `req` from what the catalog holds, and estimate each step.

    Weights missing from the volume become a CPU prep step (late_claim) or a download on the
    GPU box (asap). A compile-cache hit restores in parallel with the launch (0 s on the
    critical path) and uses the cached graph-capture estimate. Host RAM is sized from the
    checkpoint, and weights are prefetched if the checkpoint fits it. Estimates come from the
    catalog's learned timings for this model (shared ones until it has its own). The compile
    key needs the model's config.json, so a model with no weights yet is re-planned after prep.
    """
    st = cat.stats(req.model)
    wkey = weights_key(req.model)
    w = cat.find(wkey)
    on_volume = bool(w and any(l["tier"] == "volume" for l in w["locations"]))
    wbytes = (w or {}).get("bytes") or hub_bytes(req.model) or DEFAULT_WEIGHT_BYTES

    cfg_json = read_model_config(req.model) if on_volume else None
    ckey = compile_key(SGLANG_VERSION, gpu_name(req.gpu), req.tp, cfg_json) if cfg_json else None
    c = cat.find(ckey) if ckey else None
    hit = bool(c and any(l["tier"] == "volume" for l in c["locations"]))

    steps: list[Step] = []
    fetch_s = wbytes / 1e9 / st["hf_gbps"]
    if not on_volume and req.policy == "late_claim":
        steps.append(Step("cpu", "prep: weights -> volume", fetch_s,
                          f"CPU sandbox, no GPU billed; into the model's own volume {model_volume_name(req.model)} (created if new)"))
    steps.append(Step("gpu", "claim GPU + container start", st["container_start_s"]))
    if not on_volume and req.policy == "asap":
        steps.append(Step("gpu", "download weights on GPU box", fetch_s, "GPU idle while downloading"))
    if hit:  # runs beside SGLang's launch, long before the kernels are needed: off the critical path
        steps.append(Step("gpu", "restore compile cache", 0.0,
                          f"{c['bytes'] / 1e6:.0f} MB, in parallel with launch"))
    cold_load_s = wbytes / 1e9 / st["volume_gbps"]
    (memory_mib, memory_limit_mib), fits = engine_memory(wbytes, req.gpu)
    prefetch = req.prefetch and fits
    if prefetch:
        steps.append(Step("gpu", "prefetch weights (parallel)", 0.0,
                          f"{wbytes / 1e9:.0f} GB @ {st['prefetch_gbps']:.1f} GB/s, overlapping the ~{st['pre_load_s']:.0f} s before load begins"))
    steps.append(Step("gpu", "imports/NCCL/init/warmup", st["other_s"]))
    if prefetch:
        # Whatever the prefetch hasn't read by "Load weight begin" is still on the critical path;
        # the rest loads at page-cache speed (warm_load_s was measured on 61 GB over 2 GPUs).
        read_s = wbytes / 1e9 / st["prefetch_gbps"]
        # A model's own measurement if it has one; else scale the 30B reference (61 GB, TP=2).
        warm_s = st["warm_load_s"] if "warm_load_s" in st["own"] else \
            st["warm_load_s"] * (wbytes / DEFAULT_WEIGHT_BYTES) / max(req.tp / 2, 1)
        load_s = max(warm_s, read_s - st["pre_load_s"])
        steps.append(Step("gpu", "load weights (prefetched)", load_s,
                          f"{read_s:.0f} s of reads, {min(read_s, st['pre_load_s']):.0f} s hidden behind startup"))
    else:
        steps.append(Step("gpu", "load weights from volume", cold_load_s, f"@ {st['volume_gbps']:.2f} GB/s"))
    pow2 = req.effective_graphs() == "pow2"
    graph_s = (st["graph_s_hit_pow2"] if hit else st["graph_s_miss_pow2"]) if pow2 else \
        (st["graph_s_hit"] if hit else st["graph_s_miss"])
    steps.append(Step("gpu", "CUDA graphs" + (" pow2" if pow2 else "") + (" (cached kernels)" if hit else " + JIT"),
                      graph_s, ("21 shapes" if pow2 else ("94 shapes" if req.effective_graphs() == "full" else "custom"))
                      + ("" if hit else f"; compile cache written back to {artifacts_volume_name(req.model)} (created if new)")))

    cpu_s = sum(s.est_s for s in steps if s.where == "cpu")
    gpu_s = sum(s.est_s for s in steps if s.where == "gpu")
    return Plan(req, wkey, wbytes, on_volume, ckey, hit, steps, prefetch, memory_mib, memory_limit_mib,
                cpu_s, gpu_s, cpu_s + gpu_s)


# ---------------------------------------------------------------- execution

def execute(p: Plan, cat: Catalog, emit: Emit = lambda *a, **k: None,
            cancelled: threading.Event | None = None, engine_id: str | None = None,
            job_id: str | None = None) -> dict:
    """Run a plan. Raises Cancelled if `cancelled` is set; the caller terminates sandboxes."""
    cancelled = cancelled or threading.Event()

    def check():  # called between steps: cancellation takes effect at the next boundary
        if cancelled.is_set():
            raise Cancelled()

    req = p.request
    engine_id = engine_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{req.policy}"
    t_request = time.time()
    actual: dict = {}
    emit("plan", plan=p.to_dict())

    # The catalog can be stale (e.g. the volume was deleted by hand): check before claiming a GPU.
    if p.weights_on_volume and not weights_complete(req.model)[0]:
        cat.remove(p.weights_key)
        p = plan(req, cat)
        emit("plan", plan=p.to_dict(), replanned=True, reason="weights missing from the volume")

    if not p.weights_on_volume and req.policy == "late_claim":
        emit("prep_started", what="weights", model=req.model)
        r = workers.prepare_weights(req.model, on_sandbox=lambda sid: emit("sandbox", role="prep", sandbox_id=sid))
        check()
        actual["prep"] = r
        emit("prep_done", **r)
        cat.register("weights", p.weights_key, {"model": req.model, "revision": "main"},
                     r["bytes"], "volume", f"{model_volume_name(req.model)}:/{req.model}")
        if not r["already_present"]:
            cat.observe(hf_gbps=r["bytes"] / 1e9 / max(r["seconds"], 1e-3))
        p = plan(req, cat)  # re-plan: the compile key is computable now that config.json exists
        emit("plan", plan=p.to_dict(), replanned=True)

    restore = p.compile_hit
    cfg = {
        "run_id": engine_id, "out": f"/results/engines/{engine_id}", "model": req.model,
        "model_path": model_path(req.model), "tp": req.tp, "sglang": SGLANG_VERSION,
        "extra_args": req.engine_args(), "restore_cache": restore, "write_back": not restore,
        "download_if_missing": req.policy == "asap" and not p.weights_on_volume,
        "prefetch": PREFETCH if p.prefetch else None,
        "import_prefetch": req.import_prefetch,
        "groups": req.group_layout(),
        "cpus": engine_cpus(req.gpu),
        "modelexpress": req.modelexpress,
    }
    check()
    eng = workers.start_engine(cfg, req.gpu, req.ttl_s, tags={"engine": engine_id, "model": req.model},
                               secrets=cfg["download_if_missing"],
                               ports=[g["port"] for g in cfg["groups"]] or None, memory_mib=(p.memory_mib, p.memory_limit_mib))
    t_gpu = time.time()
    emit("sandbox", role="engine", sandbox_id=eng.sb.object_id, engine_id=engine_id)
    eng.wait(expect_writeback=not restore, emit=emit, cancelled=cancelled)
    check()
    ready_at = time.time()
    # Modal scheduling = sandbox create -> its container starts (waiting for a host with free
    # GPUs). Not billed and not our startup: reported on its own and excluded from startup_s.
    st = eng.started or {}
    container_start_s = st.get("container_start_s")
    scheduling_wait_s = None
    if st.get("t") is not None and st.get("uptime_s") is not None:
        scheduling_wait_s = round(min(max(st["t"] - st["uptime_s"] - t_gpu, 0.0), container_start_s or 0.0), 1)

    m = eng.ready["run"]["marks"]
    span = lambda a, b: (m[b] - m[a]) if a in m and b in m else 0.0
    load_s = span("load_begin", "load_end")
    graph_s = span("prefill_graph_begin", "prefill_graph_end") + span("decode_graph_begin", "decode_graph_end")
    other_s = m["first_request_done"] - load_s - graph_s
    prep = eng.ready["prep"]
    cache = prep.get("cache", {})
    wbytes = volume_bytes(model_volume(req.model, create=False), req.model) or p.weights_bytes

    # Feed measurements back into the estimates.
    # A prefetched load reads from page cache, so it says nothing about volume bandwidth.
    # Learned per model (235B's init is ~3x 30B's). Several engines per host is a different
    # workload, so those starts don't feed the single-engine estimates.
    if req.groups == 1:
        cat.observe(
            model=req.model,
            container_start_s=(container_start_s - (scheduling_wait_s or 0.0)) if container_start_s is not None else None,
            other_s=other_s,
            pre_load_s=m.get("load_begin"),
            **({"warm_load_s": load_s} if p.prefetch and (prep.get("prefetch") or {}).get("done_at_s", 1e9) <= m.get("load_begin", 0) + eng.ready.get("pre_launch_s", 0)
               else {} if p.prefetch else {"volume_gbps": wbytes / 1e9 / load_s if load_s else None}),
            **({f"graph_s_{'hit' if cache.get('hit') else 'miss'}{'_pow2' if req.effective_graphs() == 'pow2' else ''}": graph_s}
               if req.effective_graphs() in GRAPH_FLAGS else {}),  # custom graph flags: don't pollute either estimate
            **({"restore_gbps": cache["bytes"] / 1e9 / max(cache["restore_s"], 1e-3)} if cache.get("hit") else {}),
            prefetch_gbps=(prep.get("prefetch") or {}).get("gbps"),
        )
    # Artifacts that just served traffic are proven good.
    cat.register("weights", p.weights_key, {"model": req.model, "revision": "main"}, wbytes,
                 "volume", f"{model_volume_name(req.model)}:/{req.model}", validated=True)
    if cache.get("hit"):
        cat.validate(prep["cache_key"])
    if eng.writeback and eng.writeback.get("saved"):
        wb = eng.writeback
        cat.register("compile", wb["key"], {"sglang": SGLANG_VERSION, "gpu": gpu_name(req.gpu), "tp": req.tp,
                                            "model_arch_of": req.model, "dirs": wb["dirs"]},
                     wb["bytes"], "volume", f"{artifacts_volume_name(req.model)}:{wb['path'].removeprefix('/artifacts')}")

    base = {"sandbox_id": eng.sb.object_id, "model": req.model, "tp": req.tp, "gpu": req.gpu,
            "started": t_request, "ttl_s": req.ttl_s, "expires_at": t_gpu + req.ttl_s, "status": "serving",
            "job_id": job_id}
    engines = {}  # engine id -> url
    if cfg["groups"]:
        # One registry entry per group; they share the sandbox (and its lifetime).
        for g in cfg["groups"]:
            gid = f"{engine_id}-{g['name']}"
            engines[gid] = eng.url_for(g["port"])
            cat.add_engine(gid, {**base, "url": engines[gid], "group": g["name"], "devices": g["devices"],
                                 "port": g["port"], "gpus_used": req.tp})
    else:
        engines[engine_id] = eng.url
        cat.add_engine(engine_id, {**base, "url": engines[engine_id]})
    url = next(iter(engines.values()))
    eng.sb.detach()
    groups_actual = {
        name: {"devices": r["devices"], "port": r["port"], "url": engines[f"{engine_id}-{name}"],
               "ready_since_start_s": r["ready_since_start_s"], "marks": r["marks"]}
        for name, r in (eng.ready.get("groups") or {}).items()}

    actual.update({
        "request_to_ready_s": round(ready_at - t_request, 1),
        "scheduling_wait_s": scheduling_wait_s,
        "startup_s": round(ready_at - t_request - (scheduling_wait_s or 0.0), 1),
        "gpu_billed_to_ready_s": round(ready_at - t_gpu - (scheduling_wait_s or 0.0), 1),
        "container_start_s": container_start_s,
        "pre_launch_s": eng.ready["pre_launch_s"], "load_s": round(load_s, 1), "graph_s": round(graph_s, 1),
        "other_s": round(other_s, 1), "cache": cache, "writeback": eng.writeback, "marks": m,
        "prefetch": prep.get("prefetch"), "import_prefetch": prep.get("import_prefetch"),
        **({"modelexpress": prep.get("modelexpress")} if req.modelexpress else {}),
        **({"groups": groups_actual} if groups_actual else {}),
    })
    record = {"engine_id": next(iter(engines)), "engines": engines, "url": url, "request": asdict(req), "plan": asdict(p), "actual": actual,
              "error_s": round(actual["startup_s"] - p.predicted_ready_s, 1)}
    emit("ready", engine_id=record["engine_id"], engines=engines, url=url, request_to_ready_s=actual["request_to_ready_s"],
         startup_s=actual["startup_s"], scheduling_wait_s=scheduling_wait_s,
         predicted_ready_s=round(p.predicted_ready_s, 1), error_s=record["error_s"],
         gpu_billed_to_ready_s=actual["gpu_billed_to_ready_s"])
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, f"{engine_id}.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


def location_exists(uri: str) -> bool | None:
    """Does a volume copy "<volume>:/<path>" still exist? None if Modal couldn't be asked."""
    volume, _, path = uri.partition(":")
    parent, name = os.path.split(path.strip("/"))
    try:
        entries = modal.Volume.from_name(volume).listdir(parent or "/")
    except modal.exception.NotFoundError:  # the volume, or the directory holding the copy, is gone
        return False
    except Exception:
        return None
    return any(os.path.basename(e.path.rstrip("/")) == name for e in entries)


def sync_catalog(cat: Catalog) -> list[dict]:
    """Drop catalog copies whose volume or path no longer exists (deleted outside the system).
    Copies Modal couldn't be asked about are kept. Returns what was removed."""
    locs = [(key, l["uri"]) for key, a in cat.artifacts().items() for l in a["locations"] if l["tier"] == "volume"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        exists = list(ex.map(lambda kl: location_exists(kl[1]), locs))
    removed = []
    for (key, uri), ok in zip(locs, exists):
        if ok is False:
            cat.remove_location(key, "volume", uri)
            removed.append({"key": key, "uri": uri})
    return removed


def reconcile(cat: Catalog, model: str = MODEL_ID) -> list[dict]:
    """Register artifacts that already exist on the volumes (bootstrap / repair)."""
    found = []
    n = volume_bytes(model_volume(model, create=False), model)
    complete, missing = weights_complete(model)
    if n and complete:
        cat.register("weights", weights_key(model), {"model": model, "revision": "main"}, n,
                     "volume", f"{model_volume_name(model)}:/{model}")
        found.append({"kind": "weights", "key": weights_key(model), "bytes": n})
    elif n:  # partial download: make sure the catalog doesn't advertise it
        cat.remove(weights_key(model))
        found.append({"kind": "weights", "key": weights_key(model), "bytes": n, "incomplete": missing[:10]})
    try:
        entries = artifacts_volume(model, create=False).listdir("compile")
    except Exception:
        entries = []
    for e in entries:
        if e.path.endswith(".tar"):
            key = os.path.basename(e.path)[:-4]
            cat.register("compile", key, {"note": "found on volume by reconcile", "model_arch_of": model}, e.size,
                         "volume", f"{artifacts_volume_name(model)}:/{e.path}")
            found.append({"kind": "compile", "key": key, "bytes": e.size})
    return found
