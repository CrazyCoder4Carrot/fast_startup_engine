"""Scale-out with ModelExpress, inside one 4xH100 sandbox (stdlib only).

    python /root/mx_runner.py '<json config>'

  1. Redis + modelexpress-server (metadata only; weights never pass through it)
  2. Engine A on GPUs 0,1 with the ModelExpress loader: nothing to pull from yet,
     so it falls back to the volume, then publishes its weights once healthy
  3. Probe loop measures A's latency; engine B on GPUs 2,3 starts with the same
     flags and should pull weights GPU-to-GPU from A over NIXL
  4. Control: B stopped, then a vanilla B on the same GPUs (default loader,
     page cache already warm from A) - the bar a same-node peer copy must beat
The last stdout line is the summary as JSON (prefixed __SUMMARY__).
"""

import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

import bench_runner as eng

MX_PORT = 8001
MX_FLAGS = ["--load-format", "remote_instance", "--remote-instance-weight-loader-backend", "modelexpress",
            "--modelexpress-config", json.dumps({"transport": "nixl", "url": f"127.0.0.1:{MX_PORT}"})]
STRATEGY_HINT = re.compile(r"modelexpress|strategy|p2p|nixl|rdma|source|fallback|transfer", re.I)


def wait_port(port: int, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise RuntimeError(f"port {port} not up after {timeout}s")


def start_mx(out: str) -> list[subprocess.Popen]:
    redis = subprocess.Popen(["redis-server", "--port", "6379", "--save", "", "--appendonly", "no"],
                             stdout=open(f"{out}/redis.log", "w"), stderr=subprocess.STDOUT)
    wait_port(6379)
    env = {**os.environ, "MX_METADATA_BACKEND": "redis", "REDIS_URL": "redis://127.0.0.1:6379",
           "MODEL_EXPRESS_SERVER_PORT": str(MX_PORT), "MODEL_EXPRESS_SERVER_HOST": "127.0.0.1",
           "MODEL_EXPRESS_SERVER_METRICS_PORT": "0"}
    mx = subprocess.Popen(["modelexpress-server"], env=env,
                          stdout=open(f"{out}/mx-server.log", "w"), stderr=subprocess.STDOUT)
    wait_port(MX_PORT)
    eng.log("redis + modelexpress-server up")
    return [redis, mx]


def engine_cfg(cfg: dict, gpus: str, port: int, idx: int, mx: bool) -> dict:
    env = {"CUDA_VISIBLE_DEVICES": gpus}
    if mx:
        env.update({
            "MX_SERVER_ADDRESS": f"127.0.0.1:{MX_PORT}",
            "MX_METADATA_BACKEND": "redis", "MX_REDIS_URL": "redis://127.0.0.1:6379",
            "MX_P2P_METADATA": "1",
            "MX_WORKER_GRPC_PORT": str(52000 + 100 * idx), "MX_METADATA_PORT": str(53000 + 100 * idx),
            "MX_ARTIFACT_READY_URL": f"http://127.0.0.1:{port}/health",
            "MODEL_EXPRESS_LOG_LEVEL": "info",
            **({"MX_ARTIFACT_TRANSFER": "1"} if cfg.get("artifact_transfer") else {}),
        })
    extra = (MX_FLAGS if mx else []) + ["--engine-info-bootstrap-port", str(6789 + idx)]
    return {**cfg, "port": port, "env": env, "extra_args": extra}


def phases(run: dict) -> dict:
    m = run["marks"]
    span = lambda a, b: round(m[b] - m[a], 2) if a in m and b in m else None
    return {"ready_s": round(m["first_request_done"], 1), "load_s": span("load_begin", "load_end"),
            "prefill_graph_s": span("prefill_graph_begin", "prefill_graph_end"),
            "decode_graph_s": span("decode_graph_begin", "decode_graph_end")}


def mx_lines(lines: list) -> list[str]:
    return [f"{t:7.1f}s {l[:300]}" for t, l in lines if STRATEGY_HINT.search(l) and "TP1]" not in l][:80]


class Probe(threading.Thread):
    """Steady light load on engine A: one short request after another."""

    def __init__(self, base: str):
        super().__init__(daemon=True)
        self.base, self.samples, self.stop = base, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            t0 = time.time()
            try:
                eng.post("/generate", {"text": "Count to five:", "sampling_params": {"max_new_tokens": 32,
                         "temperature": 0}}, timeout=60, base=self.base)
                self.samples.append((t0, time.time() - t0))
            except Exception:
                self.samples.append((t0, None))
            time.sleep(0.2)

    def window(self, a: float, b: float) -> dict:
        xs = sorted(d for t, d in self.samples if a <= t <= b and d is not None)
        if not xs:
            return {}
        return {"n": len(xs), "p50_ms": round(1000 * statistics.median(xs), 1),
                "p99_ms": round(1000 * xs[min(len(xs) - 1, int(0.99 * len(xs)))], 1),
                "errors": sum(1 for t, d in self.samples if a <= t <= b and d is None)}


def stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main(cfg: dict) -> dict:
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    fn_start = time.time()
    infra = start_mx(out)
    res = {"run_id": cfg["run_id"], "fn_start": fn_start}
    procs = []
    try:
        # A: first replica. ModelExpress finds no peer, falls back to storage, then publishes.
        a_proc, a_run, a_lines = eng.launch(engine_cfg(cfg, "0,1", 30000, 0, mx=True), f"{out}/engineA.log")
        procs.append(a_proc)
        res["A"] = {**phases(a_run), "correctness": eng.correctness("http://127.0.0.1:30000"),
                    "mx_log": mx_lines(a_lines)}
        eng.log(f"A ready: {res['A']['ready_s']}s (load {res['A']['load_s']}s)")
        time.sleep(cfg.get("publish_wait_s", 20))  # A publishes after /health; give it time to register

        probe = Probe("http://127.0.0.1:30000")
        probe.start()
        time.sleep(15)
        t_b = time.time()

        # B: scale-out replica with ModelExpress -> expected GPU-to-GPU from A.
        b_proc, b_run, b_lines = eng.launch(engine_cfg(cfg, "2,3", 30001, 1, mx=True), f"{out}/engineB_mx.log")
        procs.append(b_proc)
        t_b_ready = time.time()
        res["B_mx"] = {**phases(b_run), "correctness": eng.correctness("http://127.0.0.1:30001"),
                       "mx_log": mx_lines(b_lines)}
        eng.log(f"B (ModelExpress) ready: {res['B_mx']['ready_s']}s (load {res['B_mx']['load_s']}s)")
        time.sleep(10)
        res["A_latency"] = {"before_B": probe.window(t_b - 15, t_b), "during_B": probe.window(t_b, t_b_ready),
                            "after_B": probe.window(t_b_ready, time.time())}
        probe.stop.set()
        stop(b_proc)
        time.sleep(10)

        # Control: vanilla B on the same GPUs; page cache is warm because A read the files.
        c_proc, c_run, c_lines = eng.launch(engine_cfg(cfg, "2,3", 30002, 2, mx=False), f"{out}/engineB_vanilla.log")
        procs.append(c_proc)
        res["B_vanilla"] = {**phases(c_run), "correctness": eng.correctness("http://127.0.0.1:30002")}
        eng.log(f"B (vanilla, warm page cache) ready: {res['B_vanilla']['ready_s']}s (load {res['B_vanilla']['load_s']}s)")
        stop(c_proc)
    except Exception as e:  # keep partial results: a failed step is still a finding
        res["error"] = str(e)[-4000:]
        eng.log(f"step failed: {str(e)[:500]}")
    finally:
        for p in procs + infra:
            try:
                stop(p)
            except Exception:
                pass
        with open(f"{out}/summary.json", "w") as f:
            json.dump(res, f, indent=2)
        subprocess.run(["sync", out], check=False)
    return res


if __name__ == "__main__":
    summary = main(json.loads(sys.argv[1]))
    print("__SUMMARY__" + json.dumps(summary), flush=True)
