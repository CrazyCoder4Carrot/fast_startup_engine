"""Shared Modal objects and constants: volumes (one set per model), images and their
build steps, and per-sandbox sizing (host RAM and CPUs) derived from the checkpoint and GPUs."""

import re

import modal

from fes.paths import PATCHES

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SGLANG_VERSION = "v0.5.20"
MODELS_DIR = "/models"
RESULTS_DIR = "/results"

# Modal v2 volumes have no fixed size, so there is no 3 TB to reserve up front.
# One weights volume per model, so loading one model never competes with another's reads.
MODEL_VOLUMES = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "fes-models-qwen3-30b",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "fes-models-qwen3-235b",
}


def model_volume_name(model_id: str) -> str:
    """Weights volume for a model: the fixed name above, else derived from the model name."""
    return MODEL_VOLUMES.get(model_id) or "fes-models-" + re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def model_volume(model_id: str, create: bool = True) -> modal.Volume:
    """The model's own weights volume, mounted at MODELS_DIR (layout inside: <org>/<name>/).

    Created on first use: a model the catalog has never seen (weights cache miss) gets a new
    volume when its prep worker mounts it. Lookups pass create=False, so checking a deleted
    volume raises NotFoundError instead of silently recreating it empty."""
    return modal.Volume.from_name(model_volume_name(model_id), create_if_missing=create, version=2)


# Host RAM for a GPU sandbox, sized from the checkpoint. With prefetch, the checkpoint is held
# twice at the peak: once in page cache, and again in the SGLang ranks' copies as they load.
# 235B (470 GB) was OOM-killed with 601 GiB (weights x1.3) at 25% of loading; the vanilla
# load (no prefetch) passed with 1 TiB; with Modal's default it died at 79%.
# Request = weights x2 + 64 GiB; hard limit = 180 GiB per GPU (an 8xB200 host showed 1,586 GB,
# ~198 GB per GPU), so a peak above the request has headroom instead of being killed there.
RAM_PER_WEIGHT_BYTE = 2.0
RAM_BASE_GIB = 64
RAM_MAX_GIB_PER_GPU = 180


def gpu_count(gpu_spec: str) -> int:
    return int(gpu_spec.split(":")[1]) if ":" in gpu_spec else 1


# CPUs for a GPU sandbox. Modal's default gave a 2xH100 engine 2 CPUs, shared by SGLang's
# launcher, one process per GPU, the detokenizer and the prefetch threads (capped at 2x CPUs).
CPUS_PER_GPU = 4


def engine_cpus(gpu_spec: str) -> int:
    return CPUS_PER_GPU * gpu_count(gpu_spec)


def engine_memory(weights_bytes: float, gpu_spec: str) -> tuple[tuple[int, int], bool]:
    """((request MiB, limit MiB) for Modal's `memory=`, whether the checkpoint fits for prefetch)."""
    need = weights_bytes * RAM_PER_WEIGHT_BYTE + RAM_BASE_GIB * 2**30
    cap = RAM_MAX_GIB_PER_GPU * 2**30 * gpu_count(gpu_spec)
    return (int(min(need, cap) // 2**20), int(cap // 2**20)), need <= cap


def artifacts_volume_name(model_id: str) -> str:
    """Compile-cache volume for a model: fes-models-X -> fes-artifacts-X."""
    return model_volume_name(model_id).replace("fes-models-", "fes-artifacts-", 1)


def artifacts_volume(model_id: str, create: bool = True) -> modal.Volume:
    """The model's compile caches (/artifacts/compile/<key>.tar); created on the first compile-cache
    miss, when the engine that missed writes its cache back. Lookups pass create=False."""
    return modal.Volume.from_name(artifacts_volume_name(model_id), create_if_missing=create, version=2)


models_vol = model_volume(MODEL_ID)  # the default model's volume (setup / legacy scripts)
results_vol = modal.Volume.from_name("fes-results", create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("huggingface-secret")  # provides HF_TOKEN

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]>=0.34")
    .add_local_python_source("fes")
)

sglang_image = (
    modal.Image.from_registry(f"lmsysorg/sglang:{SGLANG_VERSION}")
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("fes")
)


# Python source trees in the SGLang image (SGLang itself is an editable install).
PY_ROOTS = ["/opt/sglang/lib/python3.12/site-packages", "/sgl-workspace/sglang/python"]


def with_bytecode(image: modal.Image) -> modal.Image:
    """Compile .pyc files at image build time.

    A cold `import sglang.launch_server` otherwise compiles ~3,900 modules: 5.6-7.8 s
    (median 7.0 s) per import, measured paired on 3 hosts. Default timestamp
    invalidation: unchecked-hash measured no faster (+0.35 s) and can go stale.
    """
    return image.run_commands(f"python -m compileall -q -j 0 {' '.join(PY_ROOTS)} > /dev/null 2>&1 || true")


def with_lazy_imports(image: modal.Image) -> modal.Image:
    """Defer torchvision.io and transformers' training-loss imports (see patches/lazy_imports.py).

    Apply before with_bytecode(), so the patched sources are the ones compiled.
    """
    return (image.add_local_file(str(PATCHES / "lazy_imports.py"), "/tmp/lazy_imports.py", copy=True)
            .run_commands("python /tmp/lazy_imports.py"))


def with_import_manifest(image: modal.Image) -> modal.Image:
    """Record the files the engine's imports read (see patches/record_import_manifest.py).

    Apply last among the Python-changing steps, so the recorded .pyc paths are final.
    """
    return (image.add_local_file(str(PATCHES / "record_import_manifest.py"), "/tmp/record_import_manifest.py", copy=True)
            .run_commands("python /tmp/record_import_manifest.py"))


MX_REPO = "https://github.com/ai-dynamo/modelexpress.git"


def with_modelexpress(image: modal.Image) -> modal.Image:
    """Install ModelExpress (https://github.com/ai-dynamo/modelexpress): its Python client for
    SGLang's `modelexpress` weight-loader backend, plus Redis and modelexpress-server (built
    with the image's Rust toolchain). --no-deps: the SGLang image already owns CUDA, NIXL,
    torch, gRPC and protobuf."""
    # Earlier build steps copy files into /tmp, leaving it non-world-writable; apt needs 1777.
    return (image.run_commands("chmod 1777 /tmp")
            .apt_install("redis-server", "protobuf-compiler", "git")
            .run_commands(
                f'python3 -m pip install --no-cache-dir --no-deps "modelexpress @ git+{MX_REPO}#subdirectory=modelexpress_client/python"',
                f"git clone --depth 1 {MX_REPO} /opt/modelexpress",
                "cd /opt/modelexpress && /root/.cargo/bin/cargo build --release --bin modelexpress-server",
                "cp /opt/modelexpress/target/release/modelexpress-server /usr/local/bin/"))


def without_vision_packages(image: modal.Image) -> modal.Image:
    """DOES NOT WORK with SGLang v0.5.20 (kept for the record, not used by any engine).

    Import succeeds, but at server start sglang.srt.configs registers multimodal image
    processors through transformers' AutoImageProcessor, which hard-requires torchvision
    (ImportError, text_only trials r1/r2). The saving was ~0.54 s per import.

    Text-only image: drop torchvision and torchcodec (~0.54 s per import, measured paired).

    transformers checks is_torchvision_available() and SGLang's video decoder catches
    ImportError, so both skip cleanly; the one hard import (decode_jpeg) is deferred by
    with_lazy_imports(), which must be applied too. Not for multimodal models.
    """
    return image.run_commands("python -m pip uninstall -y torchvision torchcodec > /dev/null 2>&1 || "
                              "uv pip uninstall --python $(which python) torchvision torchcodec")


def model_path(model_id: str) -> str:
    """Where the model's weights appear inside a sandbox (its volume is mounted at MODELS_DIR)."""
    return f"{MODELS_DIR}/{model_id}"
