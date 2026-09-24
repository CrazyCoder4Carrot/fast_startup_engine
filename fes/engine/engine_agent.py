"""Engine worker: entrypoint of a GPU sandbox that brings up and keeps serving its engine(s)
(one SGLang server, or one per GPU group).

    python /root/engine_agent.py '<json config>'

Protocol (stdout lines read by the scheduler):
    __STARTED__{...}    container is running, with its uptime (so the scheduler can separate
                        Modal's scheduling wait from boot); then import prefetch, weight
                        prefetch, compile-cache restore and SGLang all start at once
    __READY__{...}      every engine served its first request; phase marks + prep timings
    __WRITEBACK__{...}  compile cache written back (or skipped), after READY
Afterwards the agent just waits on the SGLang processes, which keep serving on 0.0.0.0
(port 30000; 30010, ... for more groups), each exposed through an encrypted tunnel.
"""

import json
import os
import sys
import threading
import time

import sglang_engine as engine


IMPORT_MANIFEST = "/root/import_manifest.json"


def emit(tag: str, obj: dict) -> None:
    """One protocol line on stdout (see the module docstring); the scheduler parses these."""
    print(f"__{tag}__" + json.dumps(obj), flush=True)


def ensure_weights(model_id: str, path: str) -> dict:
    """`asap` policy only: fetch weights from inside the GPU container (GPU time is billed)."""
    if os.path.exists(os.path.join(path, "config.json")):
        return {"downloaded": False}
    from huggingface_hub import snapshot_download

    t0 = time.time()
    snapshot_download(model_id, local_dir=path, max_workers=16,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"])
    os.sync()
    return {"downloaded": True, "download_s": round(time.time() - t0, 1)}


def main(cfg: dict) -> None:
    """Bring up the engine(s) with as much as possible off the critical path.

    Order: STARTED -> start the prefetches (imports, weights) -> launch SGLang for every GPU
    group at once, with the compile-cache restore beside it -> READY once each group served
    a request -> write the compile cache back on a miss -> keep serving.
    """
    t_start = time.time()
    engine.CPUS_REQUESTED = cfg.get("cpus")  # caps prefetch threads at 2x the requested CPUs
    # The sandbox's uptime starts with its container: t - uptime is when the container
    # started, so the scheduler can tell Modal's scheduling wait from container boot.
    emit("STARTED", {"t": t_start, "uptime_s": float(open("/proc/uptime").read().split()[0])})
    os.makedirs(cfg["out"], exist_ok=True)
    prep = {}
    if cfg.get("download_if_missing"):
        prep["weights"] = ensure_weights(cfg["model"], cfg["model_path"])

    # Import prefetch: read the image files SGLang's imports need (recorded at build time),
    # up to 2x the requested CPUs in parallel, so its sequential imports stop waiting on
    # lazy per-file fetches.
    imports_result: dict = {}
    if cfg.get("import_prefetch") and os.path.exists(IMPORT_MANIFEST):
        def _prefetch_imports():
            imports_result.update(engine.prefetch_files(IMPORT_MANIFEST))
            imports_result["done_at_s"] = round(time.time() - t_start, 2)
            engine.log(f"import prefetch done: {imports_result}")
        threading.Thread(target=_prefetch_imports, daemon=True).start()

    # Weight prefetch: pull shards into the page cache while SGLang imports and inits NCCL
    # (~45-60 s for 30B, ~115 s for 235B), so "Load weight begin" finds them in RAM.
    prefetch_result: dict = {}
    if cfg.get("prefetch"):
        def _prefetch():
            prefetch_result.update(engine.prefetch_weights(cfg["model_path"], **cfg["prefetch"]))
            prefetch_result["done_at_s"] = round(time.time() - t_start, 1)
            engine.log(f"prefetch done: {prefetch_result}")
        threading.Thread(target=_prefetch, daemon=True).start()

    # Marker for write-back discovery (files newer than it). Backdated instead of sleeping
    # past the filesystem's timestamp resolution, so nothing waits on it.
    marker = "/tmp/.engine_start"
    open(marker, "w").close()
    os.utime(marker, (t_start - 2, t_start - 2))

    # Cache key + compile-cache restore run beside SGLang, not before it: the restore takes
    # ~1 s, and SGLang first needs the cached kernels ~50-60 s in (NCCL/FlashInfer init,
    # then graph capture). Previously this sat on the critical path (3.5-11 s measured).
    cache: dict = {}

    def _restore():
        cache["key"] = engine.cache_key(cfg)
        cache["path"] = f"/artifacts/compile/{cache['key']}.tar"
        if cfg.get("restore_cache"):
            cache["result"] = engine.restore_cache(cache["path"])
        cache["done_at_s"] = round(time.time() - t_start, 2)

    restorer = threading.Thread(target=_restore, daemon=True)
    restorer.start()

    # ModelExpress (optional weight loader): its Redis + server must be up before SGLang loads.
    mx_procs = engine.start_modelexpress(cfg["out"]) if cfg.get("modelexpress") else []

    # One SGLang server per GPU group. Groups share this host's page cache (one weight
    # prefetch serves all), compile cache (one restore) and import prefetch.
    t_launch = time.time()
    groups = cfg.get("groups") or [{"name": None, "devices": None, "port": 30000}]
    procs, runs, errors, mx_logs = {}, {}, {}, {}

    def _launch(g):  # one group: its own GPUs, port and log; blocks until it served a request
        i = groups.index(g)
        env = {**cfg.get("env", {}), **({"CUDA_VISIBLE_DEVICES": g["devices"]} if g["devices"] else {}),
               **(engine.modelexpress_env(cfg["model_path"], i) if mx_procs else {})}
        extra = cfg["extra_args"] + (engine.modelexpress_args(i) if mx_procs else [])
        log_path = f"{cfg['out']}/engine{'-' + g['name'] if g['name'] else ''}.log"
        try:
            procs[g["name"]], run, lines = engine.launch({**cfg, "host": "0.0.0.0", "port": g["port"], "env": env,
                                                       "extra_args": extra, "label": g["name"]}, log_path)
            if mx_procs:  # which ModelExpress strategy actually loaded the weights
                mx_logs[g["name"]] = [f"{t:7.1f}s {l[:220]}" for t, l in lines if engine.MX_LOG.search(l) and "TP1]" not in l][:12]
            runs[g["name"]] = {**run, "devices": g["devices"], "port": g["port"],
                               "ready_since_start_s": round(time.time() - t_start, 2)}
        except Exception as e:  # noqa: BLE001  reported below; the agent exits non-zero
            errors[g["name"]] = str(e)

    threads = [threading.Thread(target=_launch, args=(g,), daemon=True) for g in groups]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        for name, err in errors.items():
            print(f"group {name}: {err}", flush=True)
        for p in procs.values():
            p.kill()
        sys.exit(1)
    run = runs[groups[0]["name"]]
    restorer.join(timeout=60)
    key, cache_path = cache.get("key"), cache.get("path")
    prep["cache_key"] = key
    if "result" in cache:
        prep["cache"] = cache["result"]
    prep["restore_done_at_s"] = cache.get("done_at_s")
    if cfg.get("prefetch"):
        prep["prefetch"] = prefetch_result or {"finished_before_ready": False}
    if cfg.get("import_prefetch"):
        prep["import_prefetch"] = imports_result or {"finished_before_ready": False}
    if mx_procs:
        prep["modelexpress"] = mx_logs
    emit("READY", {"prep": prep, "run": run, "pre_launch_s": round(t_launch - t_start, 2),
                   **({"groups": runs} if cfg.get("groups") else {}),
                   "ready_since_start_s": round(time.time() - t_start, 2)})

    wb = {"saved": False}
    if cfg.get("write_back") and not prep.get("cache", {}).get("hit"):
        roots = engine.cache_roots(engine.discover_writes(marker))
        if roots:
            wb = {"saved": True, "key": key, "path": cache_path, **engine.save_cache(cache_path, roots)}
        else:
            wb = {"saved": False, "reason": "no cache-like dirs written during startup"}
    emit("WRITEBACK", wb)

    # Serve until every group's server exits (a single group can be stopped on its own).
    codes = [p.wait() for p in procs.values()]
    sys.exit(max(codes, key=abs))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
