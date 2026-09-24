"""Workers, as Modal Sandboxes created by the scheduler.

- prep worker   (CPU sandbox): fetch weights into the model's weights volume; no GPU billed.
- engine worker (GPU sandbox): engine_agent.py prefetches weights and imports, restores the
  compile cache, starts one SGLang server per GPU group (each on an encrypted tunnel), writes
  back a new compile cache on a miss, and keeps serving until stopped.

Workers never write the catalog; they report on stdout and the scheduler records.
"""

import json
import re
import time

import modal

from fes.engine.sglang_engine import PHASES
from fes.paths import ENGINE

from fes.common import MODELS_DIR, SGLANG_VERSION, artifacts_volume, engine_cpus, hf_secret, with_modelexpress, model_path, model_volume, results_vol, with_bytecode, with_import_manifest, with_lazy_imports

APP_NAME = "fes-engines"
PHASE_LINE = re.compile(r"^\[\s*([\d.]+)s\]")  # engine progress lines: "[  129.7s] ..."
PHASE_RX = [(n, re.compile(p)) for n, p in PHASES]
PORT = 30000

def _engine_image(modelexpress: bool) -> modal.Image:
    # Lazy imports (deferred training-loss / JPEG imports, ~0.5 s per import) and baked bytecode
    # (~7 s per cold start); both verified on GPU with identical output. See common.py.
    # The import manifest lets the agent pre-read the image files imports need (see common.py).
    base = with_import_manifest(with_bytecode(with_lazy_imports(
        modal.Image.from_registry(f"lmsysorg/sglang:{SGLANG_VERSION}").entrypoint([]))))
    if modelexpress:
        base = with_modelexpress(base)
    return (base.env({"PYTHONUNBUFFERED": "1"})
            .add_local_file(str(ENGINE / "sglang_engine.py"), "/root/sglang_engine.py", copy=True)
            .add_local_file(str(ENGINE / "engine_agent.py"), "/root/engine_agent.py", copy=True)
            .workdir("/root"))


engine_image = _engine_image(modelexpress=False)
engine_image_mx = _engine_image(modelexpress=True)  # built on first use (a ModelExpress start)

prep_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]>=0.34")

PREP_WEIGHTS = r"""
import json, os, sys, time
from huggingface_hub import snapshot_download
model_id, path = sys.argv[1], sys.argv[2]
t0 = time.time()

def missing_files(path):
    # A checkpoint is complete only if every shard in its index exists, plus config + tokenizer.
    need = ["config.json"]
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        need += sorted(set(json.load(open(idx))["weight_map"].values()))
    elif "model.safetensors" not in os.listdir(path):  # sharded checkpoints always ship an index
        need.append("model.safetensors.index.json")
    has_tok = os.path.exists(os.path.join(path, "tokenizer.json")) or all(
        os.path.exists(os.path.join(path, f)) for f in ("vocab.json", "merges.txt"))
    return [f for f in need if not os.path.exists(os.path.join(path, f))] + ([] if has_tok else ["tokenizer files"])

before = missing_files(path) if os.path.isdir(path) else ["(nothing downloaded)"]
# Always run the download: it skips finished files and resumes partial ones, so a
# complete checkpoint costs seconds and an interrupted one is finished, not trusted.
snapshot_download(model_id, local_dir=path, max_workers=16,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"])
os.sync()
after = missing_files(path)
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs if ".cache" not in r)
print("__PREP__" + json.dumps({"bytes": size, "seconds": round(time.time() - t0, 1), "already_present": not before,
                               "complete": not after, "missing": after[:20], "missing_before": len(before)}))
"""


def app() -> modal.App:
    """The Modal app every control-plane sandbox (prep and engine) belongs to."""
    return modal.App.lookup(APP_NAME, create_if_missing=True)


def prepare_weights(model_id: str, on_sandbox=lambda sid: None) -> dict:
    """CPU prep worker: make sure the weights are on the volume. Blocks until done."""
    sb = modal.Sandbox.create(
        "python", "-c", PREP_WEIGHTS, model_id, model_path(model_id),
        app=app(), image=prep_image, cpu=8, memory=16384, timeout=2 * 60 * 60,
        volumes={MODELS_DIR: model_volume(model_id)}, secrets=[hf_secret], tags={"role": "prep", "model": model_id},
    )
    on_sandbox(sb.object_id)
    try:
        for line in sb.stdout:
            if line.startswith("__PREP__"):
                r = json.loads(line[len("__PREP__"):])
                if not r["complete"]:
                    raise RuntimeError(f"weights for {model_id} are incomplete after download; missing {r['missing']}")
                return r
        sb.wait()
        raise RuntimeError(f"prep worker exited {sb.returncode}: {sb.stderr.read()[-2000:]}")
    finally:
        sb.terminate()
        sb.detach()


class EngineTimeout(RuntimeError):
    """The sandbox hit its lifetime cap (Modal timeout, exit 124) before the engine was ready."""


class Engine:
    """Handle to a running engine sandbox."""

    def __init__(self, sb: modal.Sandbox, t_create: float):
        self.sb = sb
        self.t_create = t_create
        self.started = None
        self.ready = None
        self.writeback = None
        self.log: list[str] = []

    def _timeout_msg(self, seen: set) -> str:
        last = max(seen, key=lambda n: [n for n, _ in PHASE_RX].index(n)) if seen else "container start"
        return (f"engine lifetime cap reached before it became ready (last phase: {last}, "
                f"{time.time() - self.t_create:.0f}s after request); raise the lifetime above the predicted ready time")

    @property
    def url(self) -> str:
        return self.sb.tunnels()[PORT].url

    def url_for(self, port: int) -> str:
        return self.sb.tunnels()[port].url

    def wait(self, expect_writeback: bool, emit=lambda *a, **k: None, cancelled=None) -> "Engine":
        """Follow the agent's stdout until READY (and WRITEBACK, if expected).

        Emits log, phase (container_started, then SGLang's phases) and writeback events.
        Cancellation terminates the sandbox from outside, which ends this stream.
        """
        seen = set()
        for line in self.sb.stdout:
            now = time.time()
            if cancelled is not None and cancelled.is_set():
                break
            if line.startswith("__STARTED__"):
                self.started = {"container_start_s": round(now - self.t_create, 1),
                                **json.loads(line[len("__STARTED__"):])}
                emit("phase", name="container_started", t=self.started["container_start_s"])
            elif line.startswith("__READY__"):
                self.ready = json.loads(line[len("__READY__"):])
                self.ready["request_to_ready_s"] = round(now - self.t_create, 1)
                emit("phase", name="first_request_done", t=self.ready["run"]["marks"].get("first_request_done"))
                if not expect_writeback:
                    break
            elif line.startswith("__WRITEBACK__"):
                self.writeback = json.loads(line[len("__WRITEBACK__"):])
                emit("writeback", **self.writeback)
                break
            else:
                self.log.append(line)
                emit("log", line=line.rstrip())
                m = PHASE_LINE.match(line)
                if m:
                    for name, rx in PHASE_RX:
                        if name not in seen and rx.search(line):
                            seen.add(name)
                            emit("phase", name=name, t=float(m.group(1)))
                            break
        if cancelled is not None and cancelled.is_set():
            return self
        if self.ready is None:
            try:
                self.sb.wait()
            except modal.exception.SandboxTimeoutError:
                raise EngineTimeout(self._timeout_msg(seen)) from None
            if self.sb.returncode == 124:
                raise EngineTimeout(self._timeout_msg(seen))
            raise RuntimeError(f"engine exited {self.sb.returncode} before ready:\n" + "".join(self.log[-40:])
                               + self.sb.stderr.read()[-2000:])
        return self


def start_engine(cfg: dict, gpu: str, ttl_s: int, tags: dict, secrets: bool = False,
                 ports: list[int] | None = None, memory_mib: tuple[int, int] | None = None) -> Engine:
    """GPU engine worker. Returns once the sandbox is created; call .wait() for readiness."""
    t_create = time.time()
    sb = modal.Sandbox.create(
        "python", "/root/engine_agent.py", json.dumps(cfg),
        app=app(), image=engine_image_mx if cfg.get("modelexpress") else engine_image, gpu=gpu, timeout=ttl_s,
        encrypted_ports=ports or [PORT],
        memory=memory_mib, cpu=engine_cpus(gpu),
        volumes={MODELS_DIR: model_volume(cfg["model"]), "/results": results_vol, "/artifacts": artifacts_volume(cfg["model"])},
        secrets=[hf_secret] if secrets else [], tags={"role": "engine", **tags},
    )
    return Engine(sb, t_create)


def stop_engine(sandbox_id: str) -> None:
    """Terminate the whole sandbox (every engine in it). Its exit code becomes 137."""
    sb = modal.Sandbox.from_id(sandbox_id)
    sb.terminate()
    sb.detach()


def stop_group(sandbox_id: str, port: int) -> None:
    """Stop one engine of a sandbox that serves several: kill the server on that port only."""
    sb = modal.Sandbox.from_id(sandbox_id)
    sb.exec("pkill", "-f", f"sglang.launch_server.*--port {port}").wait()


def list_sandboxes() -> list[tuple[str, dict, int | None]]:
    """(sandbox id, tags, exit code or None if running) for every sandbox of the app."""
    a = app()
    return [(sb.object_id, sb.get_tags(), sb.poll()) for sb in modal.Sandbox.list(app_id=a.app_id)]
