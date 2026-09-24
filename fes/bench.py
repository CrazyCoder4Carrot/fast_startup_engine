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
from fes.common import (MODEL_ID, MODELS_DIR, RESULTS_DIR, SGLANG_VERSION, artifacts_volume, engine_memory, model_path, model_volume,
                    results_vol, with_bytecode, with_lazy_imports, without_vision_packages)

GPU = "H100!:2"  # "!" pins H100 so Modal doesn't silently upgrade to H200
TP = 2

MX_ENV = {
    "MX_SERVER_ADDRESS": "127.0.0.1:8001", "MX_METADATA_BACKEND": "redis", "MX_REDIS_URL": "redis://127.0.0.1:6379",
    "MX_MODEL_URI": model_path(MODEL_ID), "MX_MS_DISTRIBUTED": "1", "MODEL_EXPRESS_LOG_LEVEL": "info",
}
MX_FLAGS = ["--load-format", "remote_instance", "--remote-instance-weight-loader-backend", "modelexpress",
            "--modelexpress-config", json.dumps({"transport": "nixl", "url": "127.0.0.1:8001"})]

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
    "fastsafetensors": {"extra_args": ["--load-format", "fastsafetensors"]},
    "runai_streamer": {"extra_args": ["--load-format", "runai_streamer"], "env": {"RUNAI_STREAMER_CONCURRENCY": "32"}},
    "no_cuda_graph": {"extra_args": ["--disable-cuda-graph"]},
    "small_graphs": {"extra_args": ["--cuda-graph-max-bs-decode", "16"]},
}

SERVING = [
    {"n": 128, "concurrency": 32, "input_words": 800, "output_tokens": 256},  # throughput-bound
    {"n": 16, "concurrency": 1, "input_words": 800, "output_tokens": 256},    # latency-bound: where CUDA graphs matter most
]

_base = modal.Image.from_registry(f"lmsysorg/sglang:{SGLANG_VERSION}").entrypoint([])
_layers = lambda img: (img.env({"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
                       .add_local_file(str(ENGINE / "sglang_engine.py"), "/root/sglang_engine.py", copy=True)
                       .add_local_file(str(ENGINE / "bench_runner.py"), "/root/bench_runner.py", copy=True))
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
              model: str = MODEL_ID, gpu: str = GPU, tp: int = TP, groups: int = 1) -> dict:
    """One cold-start trial in a fresh GPU sandbox. Emits events; raises on failure.

    groups > 1: that many SGLang servers on one host, `tp` GPUs each (A: GPU 0-1, B: 2-3, ...)."""
    spec = APPROACHES[approach]
    tag = "" if model == MODEL_ID else "-" + model.split("/")[-1].split("-")[1].lower()  # e.g. "-235b"
    if groups > 1:
        gpu = f"{gpu.split(':')[0]}:{tp * groups}"
        tag += f"-{groups}x"
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{approach}{tag}-r{rep}"
    out = f"{RESULTS_DIR}/bench/{run_id}"
    cfg = {
        "run_id": run_id, "scenario": scenario, "approach": approach, "rep": rep,
        "model": model, "model_path": model_path(model), "tp": tp, "sglang": SGLANG_VERSION,
        "extra_args": spec["extra_args"], "env": spec.get("env", {}),
        "restore_cache": spec.get("restore_cache", False), "save_cache": spec.get("save_cache", False),
        "discover": spec.get("discover", False), "serving": SERVING if serving else None, "out": out,
        "prefetch": spec.get("prefetch"), "mx_infra": spec.get("mx_infra", False), "probe_read": spec.get("probe_read", True),
        "groups": [{"name": chr(ord("a") + i), "devices": ",".join(str(i * tp + j) for j in range(tp)), "port": 30000 + 10 * i}
                   for i in range(groups)] if groups > 1 else [],
    }
    emit("trial_started", run_id=run_id, approach=approach, rep=rep, scenario=scenario)

    t_create = time.time()
    sb = modal.Sandbox.create(
        "sleep", "infinity",
        app=app, image=(_mx_image() if spec.get("image") == "mx" else IMAGES[spec["image"]]() if spec.get("image") in IMAGES
                        else image_bytecode if spec.get("bytecode") else image),
        # 235B vanilla (no prefetch) needs ~55 min for weights alone; 45 min killed it mid-load.
        gpu=gpu, timeout=(100 if "235B" in cfg["model"] else 45) * 60,
        memory=engine_memory(_weights_bytes(model), gpu)[0],  # (request, limit) MiB
        volumes={MODELS_DIR: model_volume(model), RESULTS_DIR: results_vol, "/artifacts": artifacts_volume(model)},
        tags={"scenario": scenario, "approach": approach, "rep": str(rep)},
    )
    summary = None
    local_dir = str(RESULTS / run_id)
    os.makedirs(local_dir, exist_ok=True)
    try:
        up = sb.exec("bash", "-c", "cut -d' ' -f1 /proc/uptime; date +%s.%N")
        uptime_s, container_now = (float(x) for x in up.stdout.read().split())
        up.wait()
        container_start_s = round(time.time() - t_create, 1)
        # The sandbox's uptime starts when its container starts, so everything before that
        # since create() is Modal scheduling (waiting for a host with free GPUs): not counted.
        scheduling_wait_s = round(min(max(container_now - uptime_s - t_create, 0.0), container_start_s), 1)
        emit("sandbox", role="bench", sandbox_id=sb.object_id, container_start_s=container_start_s,
             scheduling_wait_s=scheduling_wait_s, run_id=run_id)

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
         serving=b.get("serving"), prep=b.get("prep"), local_dir=local_dir)
    return summary


def run_matrix(approaches: list[str], reps: int, scenario: str = "fresh_start", serving: bool = True,
               emit=_print_emit, cancelled: threading.Event | None = None,
               model: str = MODEL_ID, gpu: str = GPU, tp: int = TP, groups: int = 1) -> list[dict]:
    """Sequential on purpose: parallel trials would share volume bandwidth and skew load times."""
    app = modal.App.lookup("fes-bench", create_if_missing=True)
    rows = []
    for rep in range(1, reps + 1):
        for a in approaches:
            if cancelled is not None and cancelled.is_set():
                return rows
            try:
                s = run_trial(app, a, rep, scenario, serving, emit=emit, cancelled=cancelled, model=model, gpu=gpu, tp=tp,
                              groups=groups)
                rows.append({"approach": a, "rep": rep, "run_id": s["run_id"],
                             "ready_s": max(r["marks"]["first_request_done"] for r in s["runs"]),
                             "container_start_s": s["container_start_s"]})
            except Exception as e:
                emit("trial_failed", run_id=f"{a}-r{rep}", approach=a, rep=rep, error=str(e))
                rows.append({"approach": a, "rep": rep, "ready_s": None, "container_start_s": None})
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
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--recover", action="store_true", help="pull finished trials missing locally from the volume")
    args = ap.parse_args()
    if args.recover:
        return recover()
    if args.list:
        for k, v in APPROACHES.items():
            print(f"{k:16} {' '.join(v['extra_args']) or '(vanilla flags)'}")
        return
    if args.serving_concurrency:  # e.g. batch sizes that fall between two power-of-two graphs
        SERVING[:] = [{"n": 4 * c, "concurrency": c, "input_words": 800, "output_tokens": 256}
                      for c in map(int, args.serving_concurrency.split(","))]
    approaches = args.approaches.split(",")
    unknown = [a for a in approaches if a not in APPROACHES]
    if unknown:
        raise SystemExit(f"unknown approach(es): {unknown}; see --list")

    with modal.enable_output():
        rows = run_matrix(approaches, args.reps, args.scenario, serving=not args.no_serving,
                          model=args.model, gpu=args.gpu, tp=args.tp, groups=args.groups)
    print("\napproach          rep   ready_s  container_s")
    for r in rows:
        ready = "failed" if r["ready_s"] is None else round(r["ready_s"], 1)
        print(f"{r['approach']:16} {r['rep']:4} {ready:>9} {r['container_start_s']!s:>12}")


if __name__ == "__main__":
    main()
