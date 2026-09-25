"""Result parsing for the Analysis and Compare pages (served by the control plane's API).

    uv run fes-sync-results     # pull runs that exist on the fes-results volume but not locally

Reads ./results/<run_id>/{summary.json,run*.log,gpu.csv}. Phases are re-derived from the raw
logs on every request (the API reloads this module), so fixing a regex here fixes every past run.
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess

from fes.paths import RESULTS as _RESULTS, ROOT as _ROOT, WEB

ROOT = str(_ROOT)
RESULTS = str(_RESULTS)
STATIC = str(WEB)
RUN_ID = re.compile(r"^\d{8}-\d{6}(-[\w.-]+)?$")

# First log line matching each pattern marks that boundary.
MARKS = [
    ("server_args", r"server_args="),
    ("dist_begin", r"Init torch distributed begin"),
    ("dist_end", r"Init torch distributed end"),
    ("load_begin", r"Load weight begin"),
    ("load_end", r"Load weight end"),
    ("kv_alloc", r"KV Cache is allocated"),
    ("prefill_graph_begin", r"Capture .*prefill CUDA graph begin"),
    ("prefill_graph_end", r"Capture .*prefill CUDA graph end"),
    ("decode_graph_begin", r"Capture .*decode CUDA graph begin|Capture cuda graph begin"),
    ("decode_graph_end", r"Capture .*decode CUDA graph end|Capture cuda graph end"),
    ("uvicorn", r"Uvicorn running on"),
    ("fired_up", r"The server is fired up"),
]

# Consecutive boundaries -> segments, each assigned to one of five groups.
GROUPS = [
    "Imports & process spawn",
    "CUDA / NCCL init",
    "Weight load",
    "CUDA graph capture",
    "Other (init, KV, warmup)",
]
SEGMENTS = [
    ("Launcher imports", "spawn", "server_args", 0),
    ("TP worker spawn + imports", "server_args", "dist_begin", 0),
    ("NCCL / distributed init", "dist_begin", "dist_end", 1),
    ("Model construction", "dist_end", "load_begin", 4),
    ("Weight load", "load_begin", "load_end", 2),
    ("KV cache + misc", "load_end", "prefill_graph_begin", 4),
    ("Prefill CUDA graphs", "prefill_graph_begin", "prefill_graph_end", 3),
    ("Decode CUDA graphs", "decode_graph_begin", "decode_graph_end", 3),
    ("HTTP server + warmup", "decode_graph_end", "fired_up", 4),
    ("Health + first request", "fired_up", "first_request_done", 4),
]
LINE_TS = re.compile(r"^\s*([\d.]+) (.*)$")


def parse_log(path: str) -> tuple[dict, list[dict], dict]:
    """Read a timestamped engine log (`<seconds since spawn> <line>`).

    Returns (when each phase in MARKS happened: the first rank's line for a begin, the last
    rank's for an `_end`, since the slowest rank gates startup; the key lines worth showing;
    extras such as SGLang's own elapsed timings and the captured CUDA-graph size counts).
    """
    marks, extras = {}, {}
    compiled = [(n, re.compile(p)) for n, p in MARKS]
    key_lines = []
    key_rx = re.compile("|".join(p for _, p in MARKS) + r"|elapsed=|Engine startup timings")
    with open(path, errors="replace") as f:
        for raw in f:
            m = LINE_TS.match(raw)
            if not m:
                continue
            t, line = float(m.group(1)), m.group(2)
            for name, rx in compiled:
                if (name not in marks or name.endswith("_end")) and rx.search(line):
                    marks[name] = t
            if key_rx.search(line) and "TP1]" not in line:
                key_lines.append({"t": t, "line": line[:400]})
            if (w := re.search(r"Load weight end.*?mem usage=([\d.]+) GB", line)):
                extras["weight_gb_per_gpu"] = float(w.group(1))
            if (w := re.search(r"#tokens: (\d+)", line)):
                extras["kv_tokens"] = int(w.group(1))
            if (w := re.search(r"prefill CUDA graph begin.*num_tokens=\[([^\]]*)\]", line)):
                extras["prefill_graph_sizes"] = len(w.group(1).split(","))
            if (w := re.search(r"decode CUDA graph begin.*bs=\[([^\]]*)\]", line)):
                extras["decode_graph_sizes"] = len(w.group(1).split(","))
    return marks, key_lines, extras


def build_launch(run_dir: str, r: dict) -> dict:
    """One launch's phase marks -> timeline segments and the 5 grouped totals the charts use."""
    log = os.path.join(run_dir, f"run{r['run']}.log")
    marks, key_lines, extras = parse_log(log) if os.path.exists(log) else ({}, [], {})
    marks["spawn"] = 0.0
    marks["first_request_done"] = r["marks"].get("first_request_done")
    marks["health_ok"] = r["marks"].get("health_ok")
    segments = []
    for name, a, b, g in SEGMENTS:
        if marks.get(a) is not None and marks.get(b) is not None and marks[b] >= marks[a]:
            segments.append({"name": name, "start": marks[a], "end": marks[b], "group": g})
    groups = [0.0] * len(GROUPS)
    for s in segments:
        groups[s["group"]] += s["end"] - s["start"]
    total = marks["first_request_done"] or 0
    load_s = (marks.get("load_end") or 0) - (marks.get("load_begin") or 0)
    return {
        "run": r["run"],
        # A launch per GPU group (all cold, started together) is labelled by its group.
        "label": f"group {r['group']}" if r.get("group") else ("cold" if r["run"] == 1 else "warm"),
        "launch_offset_s": r.get("launch_offset_s", 0),
        "ready_s": total,
        "first_request_s": r.get("first_request_s"),
        "output": r.get("output"),
        "marks": {k: v for k, v in marks.items() if v is not None},
        "segments": segments,
        "groups": [round(x, 2) for x in groups],
        "load_s": round(load_s, 2),
        "key_lines": key_lines,
        **extras,
    }


def parse_gpu(run_dir: str, fn_start: float) -> dict:
    """gpu.csv (nvidia-smi every 0.5 s) -> per-GPU utilization/memory series, time relative to trial start."""
    path = os.path.join(run_dir, "gpu.csv")
    series: dict[str, dict] = {}
    if not os.path.exists(path):
        return series
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 4:
                continue
            try:
                ts = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                continue
            t = ts.replace(tzinfo=dt.timezone.utc).timestamp() - fn_start
            g = series.setdefault(row[1].strip(), {"t": [], "util": [], "mem_gb": []})
            g["t"].append(round(t, 2))
            g["util"].append(int(row[2]))
            g["mem_gb"].append(round(int(row[3]) / 1024, 2))
    return series


def gpu_busy(gpu: dict, launch: dict) -> dict:
    """Seconds GPU 0 was held vs. showed any kernel activity during one launch."""
    g = gpu.get("0")
    if not g:
        return {}
    a = launch["launch_offset_s"]
    b = a + launch["ready_s"]
    idx = [i for i, t in enumerate(g["t"]) if a <= t <= b]
    if len(idx) < 2:
        return {}
    dt_s = (g["t"][idx[-1]] - g["t"][idx[0]]) / (len(idx) - 1)
    return {
        "window_s": round(launch["ready_s"], 1),
        "mem_held_s": round(sum(dt_s for i in idx if g["mem_gb"][i] > 1), 1),
        "active_s": round(sum(dt_s for i in idx if g["util"][i] > 0), 1),
    }


def load_run(run_id: str) -> dict | None:
    """Everything the dashboard shows for one trial: summary, launches, GPU series (None if absent)."""
    run_dir = os.path.join(RESULTS, run_id)
    path = os.path.join(run_dir, "summary.json")
    if not RUN_ID.match(run_id) or not os.path.exists(path):
        return None
    with open(path) as f:
        s = json.load(f)
    gpu = parse_gpu(run_dir, s["fn_start"])
    launches = [build_launch(run_dir, r) for r in s["runs"]]
    for l in launches:
        l["gpu_busy"] = gpu_busy(gpu, l)
    cmd = s.get("cmd", [])
    extra = cmd[cmd.index("--port") + 2:] if "--port" in cmd else []
    return {
        "run_id": run_id,
        "model": s.get("model"),
        "tp": s.get("tp"),
        "sglang": s.get("sglang"),
        "gpu": (s.get("gpu") or "").splitlines()[0] if s.get("gpu") else "",
        "gpu_count": len((s.get("gpu") or "").splitlines()),
        "extra_args": " ".join(extra),
        "container_start_s": s.get("container_start_s"),
        "scheduling_wait_s": s.get("scheduling_wait_s"),
        "download": s.get("download"),  # {mode, seconds, wall_s, bytes} when the trial started without weights
        "engine_groups": s.get("groups", 1),
        "reconstructed": s.get("reconstructed"),
        "cache_dirs": s.get("cache_dirs_after_run1", {}),
        "launches": launches,
        "gpu_series": gpu,
        "groups": GROUPS,
        "bench": s.get("bench"),
    }


def list_runs() -> list[dict]:
    """Short rows for every trial under results/, newest first."""
    out = []
    if not os.path.isdir(RESULTS):
        return out
    for run_id in sorted(os.listdir(RESULTS), reverse=True):
        r = load_run(run_id)
        if not r:
            continue
        out.append({
            "run_id": run_id,
            "model": r["model"],
            "tp": r["tp"],
            "gpu": r["gpu"],
            "extra_args": r["extra_args"],
            "container_start_s": r["container_start_s"],
            "bench": _bench_brief(r),
            "launches": [
                {"run": l["run"], "label": l["label"], "ready_s": l["ready_s"],
                 "load_s": l["load_s"], "groups": l["groups"]}
                for l in r["launches"]
            ],
        })
    return out


def _bench_brief(r: dict) -> dict | None:
    """The fields the benchmark tables need from a trial (None for non-benchmark runs)."""
    b = r.get("bench")
    if not b:
        return None
    l = r["launches"][0]
    seg = {x["name"]: x["end"] - x["start"] for x in l["segments"]}
    return {
        "scenario": b.get("scenario"), "approach": b.get("approach"), "rep": b.get("rep"),
        "cache_hit": bool(b.get("prep", {}).get("cache", {}).get("hit")),
        "cache_saved": bool(b.get("prep", {}).get("cache_saved")),
        "correctness_hash": b.get("correctness", {}).get("hash"),
        "serving": b.get("serving"),
        "serving_profiles": b.get("serving_profiles") or ([b["serving"]] if b.get("serving") else []),
        "ready_s": l["ready_s"], "load_s": l["load_s"],
        "prefill_graph_s": seg.get("Prefill CUDA graphs"), "decode_graph_s": seg.get("Decode CUDA graphs"),
        "groups": l["groups"],
    }


def load_catalog() -> dict:
    """Read-only view of the control plane's catalog (in Postgres; {} if it's unreachable)."""
    from fes.catalog import Catalog
    try:
        cat = Catalog(readonly=True)
    except (SystemExit, Exception):
        return {}
    try:
        return cat.dump()
    finally:
        cat.db.close()


def load_scheduler_runs() -> list[dict]:
    """Every scheduler start record (results/scheduler/<engine>.json), oldest first."""
    d = os.path.join(RESULTS, "scheduler")
    if not os.path.isdir(d):
        return []
    runs = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name)) as f:
                runs.append(json.load(f))
    return runs


# ---------------------------------------------------------------- original vs current

CURRENT_CONFIG = "compile cache + prefetch + pow2 graphs + parallel restore (+ bytecode, lazy imports)"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Modal scheduling (waiting for a host with free GPUs) is not part of startup. Runs since
# 2026-09-23 20:00 measure it (container uptime at start); for older runs it is estimated:
# normal container boots took 3.6-14.7 s, so anything over 15 s is taken as scheduling
# beyond a typical 5 s boot.
BOOT_S, SCHED_THRESHOLD_S = 5.0, 15.0


def workload(model: str, gpu_type: str, gpu_count: int, groups: int = 1) -> dict:
    """What a launch ran: model + hardware + engines per host. The Compare page groups by this,
    so e.g. two 30B engines on 4xH100 are compared separately from one 30B on 2xH100."""
    short = (model or DEFAULT_MODEL).split("/")[-1].replace("-Instruct-2507", "")
    name = short.split("-A")[0] if "-A" in short else short  # Qwen3-30B-A3B -> Qwen3-30B
    # "NVIDIA B200, 580.95.05, 183359 MiB" (nvidia-smi) or "B200" / "H100!" (Modal spec) -> "B200"
    gpu_type = gpu_type.split(",")[0].replace("NVIDIA ", "").split()[0].rstrip("!") if gpu_type else "GPU"
    label = f"{name} · {gpu_count}×{gpu_type}" + (f" · {groups} engines" if groups > 1 else "")
    return {"workload": f"{short}|{gpu_type}x{gpu_count}|{groups}", "workload_label": label}


def scheduling_wait(measured: float | None, container_s: float | None) -> tuple[float, bool]:
    """(seconds of Modal scheduling to exclude, estimated?)"""
    if measured is not None:
        return measured, False
    if container_s and container_s > SCHED_THRESHOLD_S:
        return round(container_s - BOOT_S, 1), True
    return 0.0, False


def _cmp_item(id_, label, side, stack, model, source, groups, ready_s, container_s, extra_pre_s=0.0,
              gpu_billed_s=None, e2e_s=None, load_s=None, graph_s=None, serving=None, corr=None, when=None,
              sched_measured=None, prep_s=0.0, wl=None, download_s=0.0, download_mode=None):
    """One bar in the comparison: a launch's startup (excluding Modal scheduling) and its phases."""
    g = list(groups)
    g[4] += extra_pre_s  # cache restore etc. before spawn counts as "other"
    e2e = e2e_s if e2e_s is not None else (container_s or 0) + extra_pre_s + ready_s
    sched, est = scheduling_wait(sched_measured, container_s)
    e2e -= sched + prep_s  # startup excludes Modal scheduling and a one-time weight download
    if gpu_billed_s is not None and sched_measured is None:
        gpu_billed_s -= sched  # older records counted the wait as billed
    if download_mode == "gpu":  # downloaded on the GPU box: the GPUs were claimed and idle meanwhile
        gpu_billed_s = (gpu_billed_s if gpu_billed_s is not None else e2e) + download_s
    return {"id": id_, "label": label, "side": side, "stack": stack, "model": model, "source": source,
            **(wl or workload(model, "", 0)),
            "groups": [round(x, 2) for x in g], "ready_s": round(ready_s + extra_pre_s, 2),
            "container_start_s": container_s, "end_to_end_s": round(e2e, 1),
            "scheduling_wait_s": sched, "scheduling_estimated": est, "prep_excluded_s": round(prep_s, 1),
            "gpu_billed_s": round(gpu_billed_s if gpu_billed_s is not None else e2e, 1),
            "load_s": load_s, "graph_s": graph_s, "serving": serving, "correctness_hash": corr, "when": when,
            "download_s": round(download_s, 1), "download_mode": download_mode}


BENCH_STACK = {
    "jit_cache": "cache", "prefetch": "prefetch", "prefetch_jit": "cache + prefetch",
    "prefetch_jit_pyc": "cache + prefetch + bytecode", "graphs_on": "cache + prefetch + bytecode",
    "graphs_pow2": "cache + prefetch + bytecode + pow2", "lazy_imports": "cache + prefetch + bytecode + lazy",
    "mx_streamer": "cache + ModelExpress streamer", "bytecode": "bytecode",
}


def compare_data() -> dict:
    """Original = vanilla baseline trials. Current = scheduler starts with today's full config.
    Everything else is an earlier iteration, labelled with what it had enabled."""
    items = []
    for r in list_runs():
        full = load_run(r["run_id"])
        b = full.get("bench") or {}
        a = b.get("approach")
        if not a or a not in ({"baseline"} | set(BENCH_STACK)) or a in ("graphs_off", "graphs_small", "graphs_decode_only"):
            continue  # profiler run (different harness) and serving-degrading graph variants stay out
        # Several engines on one host (groups): the trial is ready when the slowest one is.
        l = max(full["launches"], key=lambda x: x["ready_s"]) if full.get("engine_groups", 1) > 1 else full["launches"][0]
        seg = {x["name"]: x["end"] - x["start"] for x in l["segments"]}
        graph = seg.get("Prefill CUDA graphs", 0) + seg.get("Decode CUDA graphs", 0)
        restore = ((b.get("prep") or {}).get("cache") or {}).get("restore_s", 0.0)
        dl = full.get("download") or {}  # started without weights: never an "original" baseline (warm page cache after it)
        stack = ("vanilla" if a == "baseline" else BENCH_STACK[a]) + (f" + download on {dl['mode'].upper()}" if dl else "")
        items.append(_cmp_item(
            r["run_id"], f"{a} r{b.get('rep')} · {r['run_id'][4:6]}/{r['run_id'][6:8]} {r['run_id'][9:11]}:{r['run_id'][11:13]}",
            "original" if a == "baseline" and not dl else "iteration",
            stack, full.get("model") or DEFAULT_MODEL, "benchmark trial",
            l["groups"], l["ready_s"], full.get("container_start_s"), extra_pre_s=restore if b.get("prep", {}).get("cache") else 0.0,
            load_s=l["load_s"], graph_s=round(graph, 2), serving=b.get("serving"),
            corr=(b.get("correctness") or {}).get("hash"), when=r["run_id"][:15],
            sched_measured=full.get("scheduling_wait_s"), download_s=dl.get("wall_s") or 0.0, download_mode=dl.get("mode"),
            wl=workload(full.get("model"), full.get("gpu", ""), full.get("gpu_count") or 1, full.get("engine_groups", 1))))
        if full.get("reconstructed"):
            items[-1]["note"] = "summary rebuilt from logs (" + full["reconstructed"] + ")"
    for rec in load_scheduler_runs():
        a_ = rec["actual"]; req = rec["request"]; m = a_["marks"]
        groups = [0.0] * len(GROUPS)
        for _, s0, s1, g in SEGMENTS:
            if s0 in m and s1 in m and m[s1] >= m[s0]:
                groups[g] += m[s1] - m[s0]
        hit = bool((a_.get("cache") or {}).get("hit"))
        wrote_cache = bool((a_.get("writeback") or {}).get("saved"))  # a model's first start: nothing to hit yet
        prefetch = bool(a_.get("prefetch"))
        # Required wherever today's RAM sizing would prefetch (it now does for 235B too), so a
        # start from before that change counts as an earlier iteration, not today's config.
        from fes.common import engine_memory
        prefetch_expected = engine_memory(rec["plan"].get("weights_bytes") or 0, req.get("gpu", "H100!:2"))[1]
        graphs = req.get("graph_sizes", "full")
        parallel = (a_.get("pre_launch_s") or 99) < 1
        groups_n = int(req.get("groups") or 1)
        parts = [p for p, on in (("cache", hit), ("cache written back", wrote_cache and not hit), ("prefetch", prefetch),
                                 ("pow2", graphs == "pow2"), ("parallel restore", parallel)) if on]
        if groups_n > 1:
            parts.append(f"{groups_n} engines / {req.get('gpu', '').split(':')[-1]} GPUs")
        mx = bool(req.get("modelexpress"))
        if mx:
            parts.append("ModelExpress loader")
        # Today's default configuration; an opt-in loader (ModelExpress) is an iteration to compare against.
        current = (hit or wrote_cache) and (prefetch or not prefetch_expected) and graphs == "pow2" and parallel and not mx
        items.append(_cmp_item(
            rec["engine_id"], f"scheduler · {rec['engine_id'][9:15]}", "current" if current else "iteration",
            " + ".join(parts) or "no optimizations", req.get("model") or DEFAULT_MODEL, "scheduler start", groups,
            m["first_request_done"], a_.get("container_start_s"), extra_pre_s=a_.get("pre_launch_s", 0.0),
            gpu_billed_s=a_.get("gpu_billed_to_ready_s"), e2e_s=a_.get("request_to_ready_s"),
            load_s=a_.get("load_s"), graph_s=a_.get("graph_s"), when=rec["engine_id"][:15],
            sched_measured=a_.get("scheduling_wait_s"),
            prep_s=(prep_dl := ((a_.get("prep") or {}).get("seconds") or 0.0) if not (a_.get("prep") or {}).get("already_present", True) else 0.0),
            download_s=prep_dl, download_mode="cpu" if prep_dl else None,
            wl=workload(req.get("model"), req.get("gpu", "H100!:2").split(":")[0], int(req.get("gpu", "H100!:2").split(":")[-1]), groups_n)))
    items.sort(key=lambda i: i["when"] or "")
    preds = [{"engine_id": r["engine_id"], "predicted_s": r["plan"]["predicted_ready_s"], "model": r["request"].get("model") or DEFAULT_MODEL,
              "actual_s": r["actual"]["request_to_ready_s"], "error_s": r["error_s"]} for r in load_scheduler_runs()]
    models = sorted({i["model"] for i in items}, key=lambda m: (m != DEFAULT_MODEL, m))
    # Workloads in the order the Engines page lists them: the default model first, fewest GPUs first.
    seen = {i["workload"]: i for i in items}
    workloads = [{"key": k, "label": seen[k]["workload_label"]} for k in
                 sorted(seen, key=lambda k: (seen[k]["model"] != DEFAULT_MODEL, seen[k]["model"], int(k.split("|")[1].split("x")[-1]), k))]
    return {"groups": GROUPS, "items": items, "models": models, "workloads": workloads, "current_config": CURRENT_CONFIG,
            "original": [i for i in items if i["side"] == "original"],
            "current": [i for i in items if i["side"] == "current"], "predictions": preds}


def sync_from_modal() -> None:
    """Copy runs that exist on the fes-results volume but not locally."""
    out = subprocess.run(
        ["uv", "run", "modal", "volume", "ls", "fes-results", "baseline", "--json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if out.returncode:
        print("sync failed:", out.stderr.strip()[-300:])
        return
    for entry in json.loads(out.stdout or "[]"):
        run_id = os.path.basename(entry.get("Filename", entry.get("path", "")).rstrip("/"))
        if RUN_ID.match(run_id) and not os.path.isdir(os.path.join(RESULTS, run_id)):
            print("pulling", run_id)
            subprocess.run(
                ["uv", "run", "modal", "volume", "get", "fes-results",
                 f"baseline/{run_id}", os.path.join(RESULTS, run_id)],
                cwd=ROOT,
            )


def main():
    """CLI: pull new runs from the fes-results volume into ./results."""
    argparse.ArgumentParser(description=main.__doc__).parse_args()
    sync_from_modal()


if __name__ == "__main__":
    main()


_IMPORT_ROW = re.compile(r"import time:\s+(\d+) \|\s+(\d+) \|( *)(\S+)")
_MARKS = [(n, re.compile(p)) for n, p in (("server_args", r"server_args="), ("dist_begin", r"Init torch distributed begin"))]


def proc_breakdown(log_path: str) -> dict:
    """Split "imports + process spawn" for a `trace_imports` trial (its run log).

    Uses the per-process start markers (__FESPROC__, from proc_trace/sitecustomize.py) and
    Python's import timing (PYTHONPROFILEIMPORTTIME). Import rows are attributed by time: before
    the first spawned worker starts they are the launcher's; after, the workers', which import in
    parallel (so their total is divided by the number of spawned processes). Times in seconds
    since spawn. Import timing adds a little overhead, so treat imports as a slight overestimate."""
    procs, rows, marks = [], [], {}
    for line in open(log_path, errors="replace"):
        t_str, _, msg = line.strip().partition(" ")
        try:
            t = float(t_str)
        except ValueError:
            continue
        if "__FESPROC__" in msg:
            procs.append({"t": t, "cmd": msg.split("cmd=", 1)[-1], "spawned": "spawn_main" in msg})  # not resource_tracker
        elif (m := _IMPORT_ROW.search(msg)):
            rows.append((t, int(m.group(2)), len(m.group(3))))
        for name, rx in _MARKS:
            if name not in marks and rx.search(msg):
                marks[name] = t
    if not rows:
        return {}
    top = min(r[2] for r in rows)  # least-indented rows are top-level imports
    spawned = [p for p in procs if p["spawned"]]
    first_worker = min((p["t"] for p in spawned), default=float("inf"))
    launcher_us = sum(c for t, c, ind in rows if ind == top and t < first_worker)
    worker_us = sum(c for t, c, ind in rows if ind == top and t >= first_worker)
    out = {"launcher_imports_s": round(launcher_us / 1e6, 1),
           "launcher_other_s": round(marks.get("server_args", 0) - launcher_us / 1e6, 1),
           "spawned_processes": len(spawned)}
    if spawned:
        per_worker = worker_us / 1e6 / len(spawned)
        out.update({
            "launcher_to_first_worker_s": round(first_worker - marks.get("server_args", 0), 1),
            "worker_start_spread_s": round(max(p["t"] for p in spawned) - first_worker, 1),
            "worker_imports_s": round(per_worker, 1),
            "worker_setup_s": round(marks.get("dist_begin", 0) - first_worker - per_worker, 1),
        })
    return out
