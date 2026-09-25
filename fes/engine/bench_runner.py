"""One benchmark trial, executed inside a GPU sandbox (stdlib only).

bench.py creates a fresh sandbox per trial and runs:
    python /root/bench_runner.py '<json config>'

Steps: optional compile-cache restore -> SGLang launch with per-phase timing (one server per
GPU group, all at once, if cfg["groups"]) -> correctness probe -> serving load -> optional
discovery of files written during startup -> optional compile-cache write-back. Launch,
cache and prefetch come from sglang_engine.py. Artifacts go to cfg["out"]; the last stdout
line is the trial summary as JSON.
"""

import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The engine library: launch, compile cache, prefetch. (experiments/modelexpress/mx_runner.py
# imports this module as `eng` and uses launch/log/post from it too.)
from sglang_engine import (BASE, PORT, cache_key, cache_roots, discover_writes, launch, log, post,
                           prefetch_weights, restore_cache, save_cache)

CORRECTNESS_PROMPTS = [
    "The capital of France is",
    "Write a haiku about GPUs:",
    "def fibonacci(n):",
]


def read_files(paths: list[str], workers: int = 8) -> dict:
    """Read whole files in parallel (to pull them out of Modal's lazily fetched image into memory)."""
    from concurrent.futures import ThreadPoolExecutor

    def read(path: str) -> int:
        n = 0
        try:
            with open(path, "rb", buffering=0) as f:
                while (b := f.read(8 << 20)):
                    n += len(b)
        except OSError:
            pass
        return n
    with ThreadPoolExecutor(workers) as ex:
        total = sum(ex.map(read, paths))
    return {"files": len(paths), "bytes": total}


def correctness(base: str = BASE) -> dict:
    """Greedy answers to fixed prompts, hashed: equal hashes = same outputs across configs."""
    texts = [post("/generate", {"text": p, "sampling_params": {"max_new_tokens": 32, "temperature": 0}},
                  base=base)["text"] for p in CORRECTNESS_PROMPTS]
    return {"hash": hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16], "texts": texts}


def serving_load(n: int, concurrency: int, input_words: int, output_tokens: int) -> dict:
    """Streaming load at fixed concurrency; TTFT, inter-token latency, throughput."""
    prompt = ("Summarize the history of computing in great detail. " * (input_words // 8 + 1))
    prompt = " ".join(prompt.split()[:input_words])

    def one(i: int) -> dict:
        body = json.dumps({"text": f"[{i}] {prompt}", "stream": True,
                           "sampling_params": {"max_new_tokens": output_tokens, "temperature": 0,
                                               "ignore_eos": True}}).encode()
        req = urllib.request.Request(BASE + "/generate", data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        ttft, last, tokens = None, None, 0
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                last = now
                meta = json.loads(line[5:]).get("meta_info", {})
                tokens = meta.get("completion_tokens", tokens)
        e2e = (last or time.time()) - t0
        itl = (e2e - ttft) / max(tokens - 1, 1) if ttft is not None else None
        return {"ttft": ttft, "e2e": e2e, "tokens": tokens, "itl": itl}

    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t0
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    ttfts = [r["ttft"] for r in res if r["ttft"] is not None]
    itls = [r["itl"] for r in res if r["itl"] is not None]
    return {
        "requests": n, "concurrency": concurrency, "input_words": input_words, "output_tokens": output_tokens,
        "wall_s": round(wall, 2),
        "output_tok_per_s": round(sum(r["tokens"] for r in res) / wall, 1),
        "ttft_p50_ms": round(1000 * statistics.median(ttfts), 1), "ttft_p99_ms": round(1000 * q(ttfts, 0.99), 1),
        "itl_p50_ms": round(1000 * statistics.median(itls), 2), "itl_p99_ms": round(1000 * q(itls, 0.99), 2),
    }


def probe_read_gbps(model_path: str, mb: int = 2048) -> float:
    """Raw storage throughput for this trial: parallel read of the first `mb` of the last shard.

    It warms those ~2 GB (~3% of the checkpoint) for the engine; every approach pays the
    same bias, so comparisons stay fair, and load time can be read against this number.
    """
    files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    path = os.path.join(model_path, files[-1])
    size = min(os.path.getsize(path), mb << 20)
    step = 64 << 20
    t0 = time.time()

    def read(off: int) -> int:
        with open(path, "rb", buffering=0) as f:
            f.seek(off)
            return len(f.read(min(step, size - off)))

    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(read, range(0, size, step)))
    return round(n / 1e9 / (time.time() - t0), 2)


def host_fingerprint() -> dict:
    """Which machine the trial landed on: explains variance that the code doesn't."""
    def sh(cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    return {
        "cpu_model": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(),
        "cpus": os.cpu_count(),
        "boot_id": sh("cat /proc/sys/kernel/random/boot_id"),  # same value = same host kernel
        "gpu_uuids": sh("nvidia-smi --query-gpu=uuid --format=csv,noheader").splitlines(),
        "loadavg": sh("cat /proc/loadavg"),
        "modal": {k: v for k, v in os.environ.items() if k.startswith(("MODAL_REGION", "MODAL_CLOUD", "MODAL_TASK"))},
    }


# ---------------------------------------------------------------- main

def main(cfg: dict) -> dict:
    """One benchmark trial inside a fresh sandbox: prep for the approach (cache restore,
    prefetch, ...), launch, check correctness, optionally run the serving load, write back the
    cache, and save summary.json. Prints __SUMMARY__ for bench.py to pick up."""
    fn_start = time.time()
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    log(gpu_info.replace("\n", " | "))
    host = host_fingerprint()
    log(f"host: {host['cpu_model']} x{host['cpus']} boot_id={host['boot_id'][:8]} {host['modal']}")

    marker = "/tmp/.trial_start"
    open(marker, "w").close()
    time.sleep(1.1)  # find -newer has 1 s resolution on some filesystems

    key = cache_key(cfg)
    cache_path = f"/artifacts/compile/{key}.tar"
    prep = {"cache_key": key}
    if cfg.get("restore_cache"):
        prep["cache"] = restore_cache(cache_path)

    if cfg.get("probe_read", True):
        prep["volume_read_gbps"] = probe_read_gbps(cfg["model_path"])
        log(f"raw volume read: {prep['volume_read_gbps']} GB/s")
    # Big shared libraries first (torch, cuBLAS, Triton, NCCL, ...): the launcher's imports need
    # them now, the weights only ~60 s later, so the weight prefetch waits until these are read.
    libs_done = threading.Event()
    libs_done.set()
    if cfg.get("prefetch_libs"):
        libs_done.clear()

        def _libs():
            t0 = time.time()
            prep["libs"] = {**read_files(cfg["prefetch_libs"]), "done_at_s": round(time.time() - fn_start, 1),
                            "seconds": round(time.time() - t0, 1)}
            log(f"library prefetch done: {prep['libs']}")
            libs_done.set()
        threading.Thread(target=_libs, daemon=True).start()

    prefetch_result: dict = {}
    if cfg.get("prefetch"):
        def _prefetch():
            libs_done.wait()
            prefetch_result.update(prefetch_weights(cfg["model_path"], **cfg["prefetch"]))
            log(f"prefetch done: {prefetch_result}")
        threading.Thread(target=_prefetch, daemon=True).start()

    gpu_csv = open(f"{out}/gpu.csv", "w")
    sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits", "-lms", "500"], stdout=gpu_csv)
    infra = []
    if cfg.get("mx_infra"):  # Redis + modelexpress-server, as in the scale-out experiment
        import mx_runner
        infra = mx_runner.start_mx(out)

    # One SGLang server per GPU group (cfg["groups"]), all launched at once; one server on all
    # the GPUs otherwise. Group i logs to run{i+1}.log. The trial is ready when every group is.
    groups = cfg.get("groups") or [{"name": None, "devices": None, "port": PORT}]
    t_launch = time.time()
    procs, runs, errors, lines = [], [None] * len(groups), {}, []

    def _launch(i, g):
        env = {**cfg.get("env", {}), **({"CUDA_VISIBLE_DEVICES": g["devices"]} if g["devices"] else {})}
        try:
            p_, r_, ls = launch({**cfg, "port": g["port"], "env": env, "label": g["name"]}, f"{out}/run{i + 1}.log")
            procs.append(p_)
            runs[i] = {**r_, "run": i + 1, "launch_offset_s": round(t_launch - fn_start, 1),
                       **({"group": g["name"], "devices": g["devices"], "port": g["port"]} if g["name"] else {})}
            if i == 0:
                lines.extend(ls)
        except Exception as e:  # noqa: BLE001  re-raised below, after every group has finished
            errors[g["name"]] = e

    try:
        threads = [threading.Thread(target=_launch, args=(i, g), daemon=True) for i, g in enumerate(groups)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise next(iter(errors.values()))
        run = max(runs, key=lambda r: r["marks"]["first_request_done"])  # the slowest group gates the trial
        if cfg.get("prefetch"):
            prep["prefetch"] = prefetch_result or {"finished_before_ready": False}
        log(f"ready in {run['marks']['first_request_done']:.1f}s after spawn"
            + (f" (slowest of {len(groups)} engines)" if len(groups) > 1 else ""))
        result = {"correctness": correctness()}  # on the first group (port 30000)
        if cfg.get("serving"):
            # One profile (dict) or several (list); the first stays under "serving" for older readers.
            profiles = cfg["serving"] if isinstance(cfg["serving"], list) else [cfg["serving"]]
            result["serving_profiles"] = []
            for prof in profiles:
                log(f"serving load (concurrency {prof['concurrency']}) ...")
                r = serving_load(**prof)
                result["serving_profiles"].append(r)
                log(f"serving: {r}")
            result["serving"] = result["serving_profiles"][0]
    finally:
        sampler.terminate()
        gpu_csv.close()
        for p_ in infra:
            p_.terminate()
        for proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    if cfg.get("mx_infra"):  # which ModelExpress strategy actually loaded the weights
        rx = re.compile(r"Eligible loaders|Trying strategy|Strategy .*(succeeded|failed)|Loaded .* via|model_streamer|fallback", re.I)
        result["mx_log"] = [f"{t:7.1f}s {l[:240]}" for t, l in lines if rx.search(l) and "TP1]" not in l][:40]
        log("modelexpress: " + " | ".join(x.split("] ", 1)[-1][:90] for x in result["mx_log"][:6]))

    if cfg.get("discover") or cfg.get("save_cache"):
        found = discover_writes(marker)
        result["startup_writes"] = found[:25]
        roots = cache_roots(found)
        result["cache_roots"] = roots
        log("files written during startup (top): " + ", ".join(
            f"{r['dir']}={r['bytes'] / 1e6:.0f}MB" for r in found[:8]))
        if cfg.get("save_cache") and roots and not prep.get("cache", {}).get("hit"):
            prep["cache_saved"] = save_cache(cache_path, roots)

    summary = {
        "run_id": cfg["run_id"], "model": cfg["model"], "tp": cfg["tp"], "sglang": cfg["sglang"],
        "gpu": gpu_info, "cmd": ["python", "-m", "sglang.launch_server", "--port", str(PORT), *cfg["extra_args"]],
        "fn_start": fn_start, "runs": runs, "groups": len(groups), "cache_dirs_after_run1": {
            r["dir"]: f"{r['bytes'] / 1e6:.0f}M" for r in result.get("startup_writes", [])[:15]},
        "bench": {"scenario": cfg["scenario"], "approach": cfg["approach"], "rep": cfg["rep"],
                  "prep": prep, "host": host, **result},
    }
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    subprocess.run(["sync", out], check=False)
    return summary


if __name__ == "__main__":
    summary = main(json.loads(sys.argv[1]))
    print("__SUMMARY__" + json.dumps(summary), flush=True)
