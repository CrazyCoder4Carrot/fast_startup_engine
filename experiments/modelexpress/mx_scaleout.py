"""Scale-out experiment with ModelExpress: engine B pulls weights GPU-to-GPU from serving engine A.

    uv run python experiments/modelexpress/mx_scaleout.py                   # one 4xH100 sandbox, ~20 min
    uv run python experiments/modelexpress/mx_scaleout.py --artifacts       # also transfer JIT caches (MX_ARTIFACT_TRANSFER=1)

Results land in ./results/mx-<timestamp>/ (summary.json + engine logs).
"""

import argparse
import json
import os
import time
from pathlib import Path

import modal

from fes.common import MODEL_ID, MODELS_DIR, RESULTS_DIR, SGLANG_VERSION, model_path, models_vol, results_vol
from fes.paths import ENGINE, RESULTS

HERE = Path(__file__).resolve().parent

MX_REPO = "https://github.com/ai-dynamo/modelexpress.git"

image = (
    modal.Image.from_registry(f"lmsysorg/sglang:{SGLANG_VERSION}")
    .entrypoint([])
    .apt_install("redis-server", "protobuf-compiler", "git")
    .run_commands(
        # --no-deps: the SGLang image already owns CUDA, NIXL, torch, gRPC and protobuf.
        f'python3 -m pip install --no-cache-dir --no-deps "modelexpress @ git+{MX_REPO}#subdirectory=modelexpress_client/python"',
        f"git clone --depth 1 {MX_REPO} /opt/modelexpress",
        "cd /opt/modelexpress && /root/.cargo/bin/cargo build --release --bin modelexpress-server",
        "cp /opt/modelexpress/target/release/modelexpress-server /usr/local/bin/",
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
    .add_local_file(str(ENGINE / "sglang_engine.py"), "/root/sglang_engine.py", copy=True)
    .add_local_file(str(ENGINE / "bench_runner.py"), "/root/bench_runner.py", copy=True)
    .add_local_file(str(HERE / "mx_runner.py"), "/root/mx_runner.py", copy=True)
    .workdir("/root")
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", action="store_true", help="also transfer JIT caches from A")
    args = ap.parse_args()
    run_id = f"mx-{time.strftime('%Y%m%d-%H%M%S')}" + ("-artifacts" if args.artifacts else "")
    out = f"{RESULTS_DIR}/mx/{run_id}"
    cfg = {"run_id": run_id, "out": out, "model": MODEL_ID, "model_path": model_path(MODEL_ID), "tp": 2,
           "sglang": SGLANG_VERSION, "artifact_transfer": args.artifacts}
    app = modal.App.lookup("fes-mx", create_if_missing=True)
    local_dir = str(RESULTS / run_id)
    os.makedirs(local_dir, exist_ok=True)
    with modal.enable_output():
        t0 = time.time()
        sb = modal.Sandbox.create("sleep", "infinity", app=app, image=image, gpu="H100!:4", timeout=50 * 60,
                                  volumes={MODELS_DIR: models_vol, RESULTS_DIR: results_vol},
                                  tags={"experiment": "mx_scaleout"})
    summary = None
    try:
        sb.exec("true").wait()
        print(f"sandbox {sb.object_id} up in {time.time() - t0:.1f}s")
        p = sb.exec("python", "/root/mx_runner.py", json.dumps(cfg))
        chunks = []
        try:
            for line in p.stdout:
                chunks.append(line)
                if "__SUMMARY__" not in line:
                    print("  " + line, end="")
        except Exception as e:
            print(f"  (output stream lost: {e}; waiting)")
        p.wait()
        text = "".join(chunks)
        if "__SUMMARY__" in text:
            summary = json.loads(text.split("__SUMMARY__", 1)[1].split("\n", 1)[0])
        if p.returncode:
            print(p.stderr.read()[-4000:])
    finally:
        # Pull whatever exists (logs are useful even if a step failed), then always terminate.
        for name in ("summary.json", "engineA.log", "engineB_mx.log", "engineB_vanilla.log", "mx-server.log"):
            try:
                with open(os.path.join(local_dir, name), "wb") as f:
                    f.write(b"".join(results_vol.read_file(f"mx/{run_id}/{name}")))
            except Exception:
                pass
        sb.terminate()
        sb.detach()
    if summary:
        with open(os.path.join(local_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if summary.get("error"):
            print("!! step failed:", summary["error"][-1500:])
        for k in ("A", "B_mx", "B_vanilla"):
            if k in summary:
                s = summary[k]
                print(f"{k:10} ready {s['ready_s']:6.1f}s  weight load {s['load_s']}s  "
                      f"graphs {s['prefill_graph_s']}/{s['decode_graph_s']}s  hash {s['correctness']['hash']}")
        print("A latency:", json.dumps(summary.get("A_latency")))
    print(f"saved to {local_dir}/")


if __name__ == "__main__":
    main()
