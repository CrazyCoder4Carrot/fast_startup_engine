"""Benchmark driver: one fresh Modal Sandbox per trial, so every trial is a cold start.

    uv run fes-bench                                   # baseline (+ compile-cache write-back)
    uv run fes-bench -a baseline,jit_cache -n 3        # 3 reps of each, sequential
    uv run fes-bench -a no_cuda_graph --no-serving
    uv run fes-bench --model Qwen/Qwen3-235B-A22B-Instruct-2507 --gpu B200:8 --tp 8
    uv run fes-bench --gpu 'H100!:2' --tp 2 --groups 2   # two engines on one 4xH100 host
    uv run fes-bench --list

Each trial lands in ./results/<run_id>/ (summary.json, run<i>.log per engine, gpu.csv) with
a "bench" block (scenario, approach, rep, serving load, correctness hash), which the
Analysis and Compare pages read.
"""

import argparse
import json
import os
import sys
import threading
import time

import modal

from fes.paths import ENGINE, EXPERIMENTS, RESULTS
from fes.common import (MODEL_ID, MODELS_DIR, RESULTS_DIR, SGLANG_VERSION, artifacts_volume, engine_cpus, engine_memory, hf_secret, model_path, model_volume,
                    results_vol, with_bytecode, with_lazy_imports, without_vision_packages)

GPU = "H100!:2"  # "!" pins H100 so Modal doesn't silently upgrade to H200
TP = 2

MX_ENV = {
    "MX_SERVER_ADDRESS": "127.0.0.1:8001", "MX_METADATA_BACKEND": "redis", "MX_REDIS_URL": "redis://127.0.0.1:6379",
    "MX_MODEL_URI": model_path(MODEL_ID), "MX_MS_DISTRIBUTED": "1", "MODEL_EXPRESS_LOG_LEVEL": "info",
}
MX_FLAGS = ["--load-format", "remote_instance", "--remote-instance-weight-loader-backend", "modelexpress",
            "--modelexpress-config", json.dumps({"transport": "nixl", "url": "127.0.0.1:8001"})]

# Shared libraries over 16 MB that `import sglang.launch_server` maps (measured in the engine image).
_SP = "/opt/sglang/lib/python3.12/site-packages"
BIG_LIBS = [f"{_SP}/{p}" for p in (
    "nvidia/cu13/lib/libcublasLt.so.13", "torch/lib/libtorch_cuda.so", "triton/_C/libtriton.so", "torch/lib/libtorch_cpu.so",
    "nvidia/cu13/lib/libcufft.so.12", "nvidia/nccl/lib/libnccl.so.2", "nvidia/cusparselt/lib/libcusparseLt.so.0",
    "nvidia/cu13/lib/libcusparse.so.12", "nvidia/cu13/lib/libcusolver.so.12", "nvidia/cu13/lib/libcurand.so.10",
    "nvidia/cu13/lib/libnvrtc.so.13", "nvidia/cu13/lib/libnvJitLink.so.13", "nvidia/cu13/lib/libcublas.so.13",
    "xgrammar/libxgrammar_bindings.so", "nvidia/nvshmem/lib/libnvshmem_host.so.3", "torch/lib/libtorch_python.so",
    "scipy.libs/libscipy_openblas-5f890258.so", "numpy.libs/libscipy_openblas64_-fdde5778.so")] + [
    "/usr/lib/x86_64-linux-gnu/libicudata.so.74.2", "/usr/lib/x86_64-linux-gnu/libcodec2.so.1.2"]

APPROACHES = {
    # Vanilla launch. Also writes the compile cache back, so jit_cache has something to restore.
    # Vanilla SGLang: stock image, no cache restore, no prefetch, full graphs, and no raw-read
    # probe (it would warm ~3% of the checkpoint). Write-back runs only after the measurement.
    "baseline": {"extra_args": [], "save_cache": True, "discover": True, "probe_read": False},
    # Restore the compile cache written by an earlier baseline trial before launching.
    "jit_cache": {"extra_args": [], "restore_cache": True, "discover": True},
    # Weight-loading variants (all read the same 61 GB from the model's weights volume).
    # prefetch: parallel reads into the page cache start at t=0, overlapping SGLang's imports.
    "prefetch": {"extra_args": [], "prefetch": {"workers": 16, "chunk_mb": 16}},
    "prefetch_jit": {"extra_args": [], "prefetch": {"workers": 16, "chunk_mb": 16}, "restore_cache": True},
    # Same as the scheduler's current engine: compile cache + prefetch + bytecode baked into the image.
    "prefetch_jit_pyc": {"extra_args": [], "prefetch": {"workers": 16, "chunk_mb": 16}, "restore_cache": True,
                         "bytecode": True},
    "bytecode": {"extra_args": [], "bytecode": True},
    # ModelExpress on Modal: its GPU-to-GPU (NIXL) path can't initialize here, so use its
    # ModelStreamer strategy (concurrent reads, split across TP ranks). Compare with jit_cache.
    "mx_streamer": {"extra_args": MX_FLAGS, "env": MX_ENV, "image": "mx", "mx_infra": True, "restore_cache": True},
    # Lazy imports, on top of the current engine stack (cache + prefetch + bytecode).
    "lazy_imports": {"extra_args": [], "prefetch": {"workers": 16, "chunk_mb": 16}, "restore_cache": True, "image": "lazy"},
    "text_only": {"extra_args": [], "prefetch": {"workers": 16, "chunk_mb": 16}, "restore_cache": True, "image": "text_only"},
    # CUDA graph trade-off, on top of the current engine stack (cache + prefetch + bytecode).
    "graphs_on": {"extra_args": [], "prefetch": {"workers": 16, "chunk_mb": 16}, "restore_cache": True, "bytecode": True},
    "graphs_off": {"extra_args": ["--disable-cuda-graph"], "prefetch": {"workers": 16, "chunk_mb": 16},
                   "restore_cache": True, "bytecode": True},
    # v0.5.20 split --cuda-graph-max-bs into -decode / -prefill (the old name is rejected as ambiguous).
    "graphs_small": {"extra_args": ["--cuda-graph-max-bs-decode", "16"], "prefetch": {"workers": 16, "chunk_mb": 16},
                     "restore_cache": True, "bytecode": True},
    # Full coverage with ~70% fewer shapes: SGLang pads each batch up to the next captured size.
    "graphs_pow2": {"extra_args": ["--cuda-graph-bs-decode", *[str(2 ** i) for i in range(9)],           # 1..256: 9 vs 36
                                   "--cuda-graph-bs-prefill", *[str(2 ** i) for i in range(2, 14)]],     # 4..8192: 12 vs 58
                    "prefetch": {"workers": 16, "chunk_mb": 16}, "restore_cache": True, "bytecode": True},
    # Keep decode graphs (where serving benefits most), skip the 58-size prefill capture.
    "graphs_decode_only": {"extra_args": ["--disable-prefill-cuda-graph"], "prefetch": {"workers": 16, "chunk_mb": 16},
                           "restore_cache": True, "bytecode": True},
    # Diagnostic: the current engine plus per-process start markers and Python's import timing, to
    # split "imports + spawn" into launcher imports, worker spawn and worker imports (dashboard.proc_breakdown).
    "trace_imports": {"extra_args": ["--cuda-graph-bs-decode", *[str(2 ** i) for i in range(9)],
                                     "--cuda-graph-bs-prefill", *[str(2 ** i) for i in range(2, 14)]],
                      "prefetch": {"workers": 16, "chunk_mb": 16}, "restore_cache": True, "bytecode": True,
                      "env": {"FES_TRACE_PROCS": "1", "PYTHONPROFILEIMPORTTIME": "1", "PYTHONPATH": "/root/proc_trace:/pkg/:/root/"}},
    # The current engine plus the big shared libraries the launcher maps (20 files over 16 MB, 3.6 GB,
    # which the import prefetch skips), read first at boot; the weight prefetch starts after them.
    "prefetch_torch_libs": {"extra_args": ["--cuda-graph-bs-decode", *[str(2 ** i) for i in range(9)],
                                           "--cuda-graph-bs-prefill", *[str(2 ** i) for i in range(2, 14)]],
                            "prefetch": {"workers": 16, "chunk_mb": 16}, "restore_cache": True, "bytecode": True,
                            "prefetch_libs": BIG_LIBS},
    "fastsafetensors": {"extra_args": ["--load-format", "fastsafetensors"]},
    "runai_streamer": {"extra_args": ["--load-format", "runai_streamer"], "env": {"RUNAI_STREAMER_CONCURRENCY": "32"}},
    "no_cuda_graph": {"extra_args": ["--disable-cuda-graph"]},
    "small_graphs": {"extra_args": ["--cuda-graph-max-bs-decode", "16"]},
}

SERVING = [
    {"n": 128, "concurrency": 32, "input_words": 800, "output_tokens": 256},  # throughput-bound
    {"n": 16, "concurrency": 1, "input_words": 800, "output_tokens": 256},    # latency-bound: where CUDA graphs matter most
]

# Where a trial's weights come from. "none": already on the model's volume (every result so far).
# "cpu" / "gpu": start from nothing and download from Hugging Face into a temporary volume first,
# on a CPU sandbox (the scheduler's way: no GPU billed) or on the GPU box (vanilla: GPUs wait).
DOWNLOAD_MODES = ("none", "cpu", "gpu")

# The benchmarked workloads (the Benchmark page offers these). trial_min: rough minutes per trial,
# for "fast" approaches (prefetch / compile cache) and vanilla ones, used for the cost estimate.
WORKLOADS = {
    "30b-2xh100": {"label": "Qwen3-30B · 2×H100", "model": MODEL_ID, "gpu": "H100!:2", "tp": 2, "groups": 1,
                   "gpus": 2, "gpu_price": 3.95, "trial_min": {"fast": 5, "vanilla": 7}, "download_min": 3},
    "30b-4xh100": {"label": "Qwen3-30B · 2 engines on 4×H100", "model": MODEL_ID, "gpu": "H100!:2", "tp": 2, "groups": 2,
                   "gpus": 4, "gpu_price": 3.95, "trial_min": {"fast": 6, "vanilla": 7}, "download_min": 3},
    "235b-8xb200": {"label": "Qwen3-235B · 8×B200", "model": "Qwen/Qwen3-235B-A22B-Instruct-2507", "gpu": "B200:8", "tp": 8,
                    "groups": 1, "gpus": 8, "gpu_price": 6.25, "trial_min": {"fast": 8, "vanilla": 22}, "download_min": 17},
}


def serving_profiles(levels) -> list[dict]:
    """Serving loads for a trial: the defaults, or one profile per concurrency level (4x as many requests)."""
    if not levels:
        return SERVING
    if isinstance(levels, str):
        levels = [x for x in levels.replace(" ", "").split(",") if x]
    return [{"n": 4 * int(c), "concurrency": int(c), "input_words": 800, "output_tokens": 256} for c in levels]


def approach_summary(name: str) -> str:
    """One line describing what an approach turns on, from its settings."""
    a, args = APPROACHES[name], " ".join(APPROACHES[name]["extra_args"])
    parts = []
    if a.get("restore_cache"):
        parts.append("compile cache")
    if a.get("prefetch"):
        parts.append("weight prefetch")
    if a.get("bytecode"):
        parts.append("bytecode")
    if a.get("prefetch_libs"):
        parts.append("big-library prefetch")
    if a.get("image") in ("lazy", "text_only"):
        parts.append("lazy imports" + (", no vision packages" if a["image"] == "text_only" else ""))
    if a.get("image") == "mx":
        parts.append("ModelExpress streamer")
    if "--cuda-graph-bs-decode" in args:
        parts.append("pow2 CUDA graphs")
    elif "--disable-cuda-graph" in args:
        parts.append("no CUDA graphs")
    elif "--cuda-graph-max-bs-decode" in args:
        parts.append("small CUDA graphs")
    elif "--disable-prefill-cuda-graph" in args:
        parts.append("decode graphs only")
    elif "--load-format" in args and a.get("image") != "mx":
        parts.append(args.split("--load-format ")[1].split()[0] + " loader")
    return ", ".join(parts) or "vanilla SGLang"

_base = modal.Image.from_registry(f"lmsysorg/sglang:{SGLANG_VERSION}").entrypoint([])
_layers = lambda img: (img.env({"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
                       .add_local_file(str(ENGINE / "sglang_engine.py"), "/root/sglang_engine.py", copy=True)
                       .add_local_file(str(ENGINE / "bench_runner.py"), "/root/bench_runner.py", copy=True)
                       .add_local_file(str(ENGINE / "proc_trace" / "sitecustomize.py"), "/root/proc_trace/sitecustomize.py",
                                       copy=True))
# Stock image for baseline and existing approaches, so the original comparison stays like-for-like.
image = _layers(_base)
image_bytecode = _layers(with_bytecode(_base))
IMAGES = {  # built on first use
    "lazy": lambda: _layers(with_bytecode(with_lazy_imports(_base))),
    "text_only": lambda: _layers(with_bytecode(without_vision_packages(with_lazy_imports(_base)))),
}


def _mx_image():  # built only when an mx approach runs (it compiles the ModelExpress server)
    sys.path.insert(0, str(EXPERIMENTS / "modelexpress"))
    import mx_scaleout
    return mx_scaleout.image




def _weights_bytes(model: str) -> float:
    """Checkpoint size on the model's weights volume (sizes the sandbox's host RAM)."""
    try:
        return sum(e.size for e in model_volume(model).listdir(model, recursive=True) if e.type.name == "FILE")
    except Exception:
        return 61e9


def _read_volume_json(path: str) -> dict | None:
    """A trial's summary from the results volume (fallback when the stdout line was lost)."""
    try:
        return json.loads(b"".join(results_vol.read_file(path)))
    except Exception:
        return None


def _print_emit(type_: str, **d) -> None:
    """CLI rendering of trial events."""
    if type_ == "log":
        print("  " + d["line"])
    elif type_ == "trial_started":
        print(f"\n=== {d['run_id']}: {d['approach']} rep {d['rep']} ({d['scenario']}) ===")
    elif type_ == "sandbox":
        print(f"sandbox {d['sandbox_id']} up in {d['container_start_s']}s (scheduling {d.get('scheduling_wait_s')}s)")
    elif type_ == "trial_done":
        print(f"--> {d['run_id']}: ready {d['ready_s']:.1f}s after spawn (+{d['container_start_s']}s container); "
              f"correctness {d['correctness_hash']}; saved to {d['local_dir']}/")
    elif type_ == "trial_failed":
        print(f"!! {d['run_id']} failed: {d['error'][:500]}")


def run_trial(app: modal.App, approach: str, rep: int, scenario: str, serving: bool,
              emit=_print_emit, cancelled: threading.Event | None = None,
              model: str = MODEL_ID, gpu: str = GPU, tp: int = TP, groups: int = 1, profiles: list | None = None,
              download: str = "none", cpus: int | None = None) -> dict:
    """One cold-start trial in a fresh GPU sandbox. Emits events; raises on failure.

    download "cpu" / "gpu": the weights start absent. They are downloaded into a temporary volume
    (deleted afterwards; the model's own volume is never touched), on a CPU sandbox before the GPU
    sandbox exists, or on the GPU sandbox before SGLang starts. The time is recorded as "download".

    groups > 1: that many SGLang servers on one host, `tp` GPUs each (A: GPU 0-1, B: 2-3, ...)."""
    spec = APPROACHES[approach]
    tag = "" if model == MODEL_ID else "-" + model.split("/")[-1].split("-")[1].lower()  # e.g. "-235b"
    if groups > 1:
        gpu = f"{gpu.split(':')[0]}:{tp * groups}"
        tag += f"-{groups}x"
    if download != "none":
        tag += f"-dl{download}"
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{approach}{tag}-r{rep}"
    out = f"{RESULTS_DIR}/bench/{run_id}"
    cfg = {
        "run_id": run_id, "scenario": scenario, "approach": approach, "rep": rep,
        "model": model, "model_path": model_path(model), "tp": tp, "sglang": SGLANG_VERSION,
        "extra_args": spec["extra_args"], "env": spec.get("env", {}),
        "restore_cache": spec.get("restore_cache", False), "save_cache": spec.get("save_cache", False),
        "discover": spec.get("discover", False), "serving": (profiles or SERVING) if serving else None, "out": out,
        "prefetch": spec.get("prefetch"), "mx_infra": spec.get("mx_infra", False), "probe_read": spec.get("probe_read", True),
        "prefetch_libs": spec.get("prefetch_libs"),
        "groups": [{"name": chr(ord("a") + i), "devices": ",".join(str(i * tp + j) for j in range(tp)), "port": 30000 + 10 * i}
                   for i in range(groups)] if groups > 1 else [],
    }
    emit("trial_started", run_id=run_id, approach=approach, rep=rep, scenario=scenario, download=download)

    from fes import workers  # CPU prep worker + its download script
    weights_vol, dl_name, dl = model_volume(model), None, None
    if download != "none":
        dl_name = f"fes-bench-dl-{run_id}".lower()[:60]
        weights_vol = modal.Volume.from_name(dl_name, create_if_missing=True, version=2)
        emit("log", line=f"weights start absent: downloading into temporary volume {dl_name} ({download})")
    try:
        if download == "cpu":
            t0 = time.time()
            r = workers.prepare_weights(model, on_sandbox=lambda sid: emit("sandbox", role="prep", sandbox_id=sid),
                                        volume=weights_vol)
            dl = {"mode": "cpu", "seconds": r["seconds"], "wall_s": round(time.time() - t0, 1), "bytes": r["bytes"]}
            emit("log", line=f"download done on CPU: {dl}")
        return _trial(app, approach, rep, scenario, spec, cfg, run_id, model, gpu, weights_vol, download, dl, emit, cancelled,
                      cpus or engine_cpus(gpu))
    finally:
        if dl_name:
            try:
                modal.Volume.objects.delete(dl_name)
                emit("log", line=f"deleted temporary volume {dl_name}")
            except Exception as e:
                emit("log", line=f"could not delete temporary volume {dl_name}: {e}")


def _exec_text(sb: modal.Sandbox, *cmd: str, emit=_print_emit, tries: int = 3) -> tuple[str, int]:
    """Run a command in the sandbox and return (stdout, returncode). Modal occasionally loses an
    exec's output stream ("Failed to read exec stdio stream"); the command is then run again.
    Only for commands that are safe to repeat (the uptime probe, a resumable download)."""
    for attempt in range(1, tries + 1):
        try:
            p = sb.exec(*cmd)
            out = p.stdout.read()
            p.wait()
            return out, p.returncode
        except Exception as e:
            if attempt == tries or "stdio stream" not in str(e):
                raise
            emit("log", line=f"exec output stream lost ({e}); retrying {attempt}/{tries - 1}")
            time.sleep(5)


def _trial(app, approach, rep, scenario, spec, cfg, run_id, model, gpu, weights_vol, download, dl, emit, cancelled,
           cpus: int) -> dict:
    """The GPU part of a trial: sandbox, (GPU-side download), bench_runner, results."""
    from fes import workers
    t_create = time.time()
    sb = modal.Sandbox.create(
        "sleep", "infinity",
        app=app, image=(_mx_image() if spec.get("image") == "mx" else IMAGES[spec["image"]]() if spec.get("image") in IMAGES
                        else image_bytecode if spec.get("bytecode") else image),
        # 235B vanilla (no prefetch) needs ~55 min for weights alone; 45 min killed it mid-load.
        gpu=gpu, timeout=(100 if "235B" in cfg["model"] else 45) * 60,
        memory=engine_memory(_weights_bytes(model), gpu)[0],  # (request, limit) MiB
        cpu=cpus,  # reserved CPUs, as for real engines (common.engine_cpus: 4 per GPU) unless overridden
        volumes={MODELS_DIR: weights_vol, RESULTS_DIR: results_vol, "/artifacts": artifacts_volume(model)},
        secrets=[hf_secret] if download == "gpu" else [],
        tags={"scenario": scenario, "approach": approach, "rep": str(rep)},
    )
    emit("log", line=f"sandbox {sb.object_id} created; waiting for {gpu} and the container")
    summary = None
    local_dir = str(RESULTS / run_id)
    os.makedirs(local_dir, exist_ok=True)
    try:
        up_out, _ = _exec_text(sb, "bash", "-c", "cut -d' ' -f1 /proc/uptime; date +%s.%N", emit=emit)
        uptime_s, container_now = (float(x) for x in up_out.split())
        container_start_s = round(time.time() - t_create, 1)
        # The sandbox's uptime starts when its container starts, so everything before that
        # since create() is Modal scheduling (waiting for a host with free GPUs): not counted.
        scheduling_wait_s = round(min(max(container_now - uptime_s - t_create, 0.0), container_start_s), 1)
        emit("sandbox", role="bench", sandbox_id=sb.object_id, container_start_s=container_start_s,
             scheduling_wait_s=scheduling_wait_s, run_id=run_id)

        if download == "gpu":  # vanilla order: the GPUs are claimed and wait while the weights download
            t0 = time.time()
            # The bench image runs offline (HF_HUB_OFFLINE=1) so a trial can never download by accident;
            # this one command is the intended download.
            out, rc = _exec_text(sb, "env", "HF_HUB_OFFLINE=0", "python", "-c", workers.PREP_WEIGHTS, model, model_path(model),
                                 emit=emit)  # snapshot_download resumes, so a retry continues where it stopped
            r = json.loads(out.split("__PREP__", 1)[1].split("\n", 1)[0]) if "__PREP__" in out else {}
            if not r.get("complete"):
                raise RuntimeError(f"GPU-side download failed (exit {rc}): {out[-1500:]}")
            dl = {"mode": "gpu", "seconds": r["seconds"], "wall_s": round(time.time() - t0, 1), "bytes": r["bytes"]}
            emit("log", line=f"download done on the GPU box (GPUs idle, billed): {dl}")

        p = sb.exec("python", "/root/bench_runner.py", json.dumps(cfg))
        with open(os.path.join(local_dir, "runner.log"), "w") as rlog:
            chunks = []
            try:
                for line in p.stdout:
                    rlog.write(line)
                    chunks.append(line)
                    if "__SUMMARY__" not in line:
                        emit("log", line=line.rstrip())
                    if cancelled is not None and cancelled.is_set():
                        break
            except Exception as e:
                # Modal occasionally drops the exec stream; the trial keeps running inside.
                emit("log", line=f"(output stream lost: {e}; waiting for the trial to finish)")
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError("cancelled")
        p.wait()
        # The summary line can arrive split across chunks, so parse the joined output.
        out_text = "".join(chunks)
        if "__SUMMARY__" in out_text:
            try:
                summary = json.loads(out_text.split("__SUMMARY__", 1)[1].split("\n", 1)[0])
            except ValueError:
                summary = None
        for _ in range(12):  # fall back to the copy on the volume; commits can lag a few seconds
            summary = summary or _read_volume_json(f"bench/{run_id}/summary.json")
            if summary:
                break
            time.sleep(5)
        if summary is None:
            raise RuntimeError(f"trial {run_id} failed (exit {p.returncode})")

        summary["container_start_s"] = container_start_s
        summary["scheduling_wait_s"] = scheduling_wait_s
        summary["download"] = dl  # None when the weights were already on the volume
        summary["cpus_reserved"] = cpus
        for name in ("run1.log", "gpu.csv"):
            try:
                data = b"".join(results_vol.read_file(f"bench/{run_id}/{name}"))
                with open(os.path.join(local_dir, name), "wb") as f:
                    f.write(data)
            except Exception as e:  # keep the trial even if one file is missing
                emit("log", line=f"could not fetch {name}: {e}")
        with open(os.path.join(local_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    finally:
        sb.terminate()
        sb.detach()

    r = max(summary["runs"], key=lambda x: x["marks"]["first_request_done"])  # slowest group, if several
    b = summary["bench"]
    emit("trial_done", run_id=run_id, approach=approach, rep=rep, ready_s=r["marks"]["first_request_done"],
         container_start_s=summary["container_start_s"], correctness_hash=b["correctness"]["hash"],
         serving=b.get("serving"), prep=b.get("prep"), local_dir=local_dir, download=dl)
    return summary


def run_matrix(approaches: list[str], reps: int, scenario: str = "fresh_start", serving: bool = True,
               emit=_print_emit, cancelled: threading.Event | None = None,
               model: str = MODEL_ID, gpu: str = GPU, tp: int = TP, groups: int = 1, profiles: list | None = None,
               download: str = "none", cpus: int | None = None) -> list[dict]:
    """Sequential on purpose: parallel trials would share volume bandwidth and skew load times."""
    app = modal.App.lookup("fes-bench", create_if_missing=True)
    rows = []
    for rep in range(1, reps + 1):
        for a in approaches:
            if cancelled is not None and cancelled.is_set():
                return rows
            try:
                s = run_trial(app, a, rep, scenario, serving, emit=emit, cancelled=cancelled, model=model, gpu=gpu, tp=tp,
                              groups=groups, profiles=profiles, download=download, cpus=cpus)
                rows.append({"approach": a, "rep": rep, "run_id": s["run_id"],
                             "ready_s": max(r["marks"]["first_request_done"] for r in s["runs"]),
                             "container_start_s": s["container_start_s"],
                             "download_s": (s.get("download") or {}).get("wall_s")})
            except Exception as e:
                emit("trial_failed", run_id=f"{a}-r{rep}", approach=a, rep=rep, error=str(e))
                rows.append({"approach": a, "rep": rep, "ready_s": None, "container_start_s": None, "error": str(e)[-500:]})
    return rows


def recover(container_starts: dict[str, float] | None = None) -> None:
    """Copy trials that exist on the fes-results volume but have no local summary."""
    for e in results_vol.listdir("bench"):
        run_id = os.path.basename(e.path.rstrip("/"))
        local_dir = str(RESULTS / run_id)
        if os.path.exists(os.path.join(local_dir, "summary.json")):
            continue
        summary = _read_volume_json(f"bench/{run_id}/summary.json")
        if not summary:
            print(f"skip {run_id}: no summary on volume (trial did not finish)")
            continue
        os.makedirs(local_dir, exist_ok=True)
        summary.setdefault("container_start_s", (container_starts or {}).get(run_id))
        for name in ("run1.log", "gpu.csv"):
            try:
                with open(os.path.join(local_dir, name), "wb") as f:
                    f.write(b"".join(results_vol.read_file(f"bench/{run_id}/{name}")))
            except Exception as ex:
                print(f"{run_id}: could not fetch {name}: {ex}")
        with open(os.path.join(local_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"recovered {run_id}")


def main():
    """CLI: run approaches x reps on a model/GPU/TP, or --list / --recover past trials."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--approaches", default="baseline")
    ap.add_argument("-n", "--reps", type=int, default=1)
    ap.add_argument("-s", "--scenario", default="fresh_start")
    ap.add_argument("--no-serving", action="store_true")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--gpu", default=GPU, help='e.g. "H100!:2" or "B200:8"')
    ap.add_argument("--tp", type=int, default=TP)
    ap.add_argument("--groups", type=int, default=1, help="SGLang servers per host, --tp GPUs each")
    ap.add_argument("--serving-concurrency", default="",
                    help="comma-separated load levels to run instead of the defaults, e.g. 20,32,100 "
                         "(4x that many requests each, same prompt and output length)")
    ap.add_argument("--download", choices=DOWNLOAD_MODES, default="none",
                    help="start without weights: download into a temporary volume on a CPU sandbox or on the GPU box")
    ap.add_argument("--cpus", type=int, default=None, help="CPUs to reserve (default: as for engines, 4 per GPU)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--recover", action="store_true", help="pull finished trials missing locally from the volume")
    args = ap.parse_args()
    if args.recover:
        return recover()
    if args.list:
        for k, v in APPROACHES.items():
            print(f"{k:16} {' '.join(v['extra_args']) or '(vanilla flags)'}")
        return
    approaches = args.approaches.split(",")
    unknown = [a for a in approaches if a not in APPROACHES]
    if unknown:
        raise SystemExit(f"unknown approach(es): {unknown}; see --list")

    with modal.enable_output():
        rows = run_matrix(approaches, args.reps, args.scenario, serving=not args.no_serving,
                          model=args.model, gpu=args.gpu, tp=args.tp, groups=args.groups,
                          profiles=serving_profiles(args.serving_concurrency), download=args.download, cpus=args.cpus)
    print("\napproach          rep   ready_s  container_s")
    for r in rows:
        ready = "failed" if r["ready_s"] is None else round(r["ready_s"], 1)
        print(f"{r['approach']:16} {r['rep']:4} {ready:>9} {r['container_start_s']!s:>12}")


if __name__ == "__main__":
    main()
