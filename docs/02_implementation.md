# Fast Engine Startup — Implementation and Results

**Scope:** cold start of SGLang v0.5.20 on Modal for the two models in the brief.
- Qwen3-30B-A3B-Instruct-2507 (BF16, 61 GB) on 2× H100, TP=2.
- Qwen3-235B-A22B-Instruct-2507 (BF16, 470 GB) on 8× B200, TP=8.

**Definition used throughout:** *startup* = time from a start request to the first successful `/generate`. It excludes Modal scheduling, meaning time spent waiting for a host with free GPUs, which we can't control and aren't billed for.

## Headline

| Qwen3-30B · 2× H100 | Vanilla SGLang | This system | Change |
|---|---:|---:|---:|
| Startup, median | **229.8 s** (n=3, range 226–241) | **84.5 s** (n=12, range 75–135) | **−63% (2.7× faster)** |
| Weight load | 52–63 s (median 61 s) | 9.5–16.8 s | read from page cache |
| CUDA graph capture | 68–74 s (median 72 s) | 7.6–10.0 s | −88% |
| Output (greedy, 3 prompts) | hash e04d4eae… | hash e04d4eae… | identical |
| Serving throughput, 32 concurrent | 4,630 tok/s | 4,547 tok/s | −2% (pow2 graph sizes) |

| Qwen3-235B · 8× B200 | Vanilla SGLang | This system | Change |
|---|---:|---:|---:|
| Startup | **951.5 s** (n=1) | **220.8 s** (n=1) | **−77% (4.3× faster)** |
| Weight load | 564 s | 75 s (470 GB prefetched at 4.17 GB/s before load began) | −87% |
| NCCL init | 24.9 s | 9.8 s (compile cache) | |
| CUDA graph capture | 79.6 s (94 sizes) | 13.9 s (pow2, cached kernels) | −83% |
| Host RAM reserved | 1 TiB (Modal's default was OOM-killed at 79% of loading) | 940 GiB request, 1,440 GiB limit (sized from the checkpoint) | |
| Weight download | would bill 8× B200 for ~17 min | done on a CPU worker before claiming GPUs | |
| Test prompt | — | "Paris" | |

## Architecture

Startup is treated as **assembling reusable artifacts** (weights, compiled kernels) that a scheduler plans from, rather than as one opaque `launch_server` call.

```
            Browser UI / JSON API
                     │  HTTP + SSE
        ┌────────────▼─────────────┐
        │ Control plane (local)    │  jobs · events · engine registry · reconciler
        │ Postgres: jobs · events  │
        │  Scheduler ──► Catalog   │  Postgres: artifacts, locations, engines, learned timings
        └──────┬──────────┬────────┘
               │ Modal SDK│
   CPU prep sandbox     GPU engine sandbox ─ engine_agent.py
   (download weights,   (restore compile cache, prefetch weights + imports,
    verify, no GPU)      launch SGLang per GPU group, write back cache)
               │          │
   Volumes, one set per model:  fes-models-<model>   (weights)
                                fes-artifacts-<model> (compile caches)
                                fes-results           (logs, summaries)
```

**Control plane** (`fes-control-plane`): two processes that share one Postgres database (in Docker), so restarting the API never interrupts a running job.
- **API** (`--role api`, Flask): validates requests, writes job rows, streams progress to the UI over SSE, and proxies prompts to engines.
- **Worker** (`--role worker`): claims the oldest queued job with `FOR UPDATE SKIP LOCKED` (many workers, even on other hosts, never get the same job), runs it (one GPU job at a time), and heartbeats every 5 s. Jobs move queued → running → serving → timeout / stopped / failed.
- **Cancellation:** a queued job is cancelled in the database; a running one is flagged for its worker, and its sandboxes are terminated right away.
- **Reconciler** (in the worker, every 30 s): compares the engine registry with Modal, enforces lifetimes, and marks jobs whose worker stopped heartbeating as interrupted (terminating their sandboxes). Serving engines are re-attached after a restart.

**Scheduler** (`scheduler.py`): builds a plan from the catalog, then executes it.
- Are the weights staged? Is there a compile cache for (SGLang version, GPU, TP, model config)? Should weights be prefetched?
- It predicts time to ready from learned timings, runs the steps, and feeds the measurements back as moving averages.
- **Late GPU claim:** anything that doesn't need a GPU, such as downloading weights, runs on a CPU sandbox before the GPU sandbox is created.

**Catalog** (`catalog.py`, Postgres): records what reusable state exists and where.
- `artifacts` and `artifact_locations`: one artifact can have several copies, each validated once an engine has served traffic with it.
- `engines`: the live engine registry.
- `stats`: learned timings, compared against priors taken from the vanilla baseline.

**Engine agent** (`engine_agent.py`): the entrypoint inside the GPU sandbox. It reports `STARTED` → `READY` → `WRITEBACK` on stdout and then keeps serving.

**Per-model volumes:** each model has its own weights volume and its own compile-cache volume, so loading one model never competes with reads of another. On a cache miss, the scheduler creates the model's volume on first use.

**UI** (served by the control plane):
- Engines: launch by workload with a live prediction; running engines.
- Jobs: history, live progress, phase chart.
- Playground: send prompts to a serving engine.
- Catalog: artifacts, engine registry, learned timings.
- Compare: vanilla vs current, per model; any two runs side by side.

## What happens on one start (30B, all optimizations)

1. The scheduler plans: weights are on `fes-models-qwen3-30b`, the compile cache hits, prefetch is on, pow2 graphs.
2. It creates the GPU sandbox with 2× H100. Modal scheduling time is recorded separately and excluded from startup.
3. As soon as the container starts, the engine agent does these in parallel:
   - **Launches SGLang immediately.**
   - Restores the compile cache (29 MB, about 1 s, in a background thread).
   - Prefetches the 61 GB of weights with 16 threads × 16 MB reads.
   - Prefetches the 262 MB of Python files that the imports read.
4. SGLang imports (bytecode pre-baked, two heavy imports deferred), spawns its TP workers and initializes NCCL. Its all-reduce kernels come from the cache.
5. Weights load from page cache (about 10–16 s), then pow2 CUDA graphs are captured with cached kernels (about 8–10 s).
6. The first request is served, and the engine is registered with its tunnel URL and lifetime.

## Optimizations and measured effect

Savings are per start on 30B, relative to vanilla. Each was measured as an A/B against the stack before it, using fresh cold sandboxes.

| # | Change | Saving | How it works | Evidence |
|---|---|---:|---|---|
| 1 | **Compile cache** | ~60 s | After a miss, the engine tars `~/.cache/sglang/{triton,jit}` (~29 MB) to the model's artifacts volume, keyed by sha256(SGLang version, GPU, TP, config.json). Later starts restore it. | Graphs 73 → 27 s. NCCL init 17 → 1 s (the custom all-reduce JIT). Warmup −9 s. |
| 2 | **Weight prefetch** | ~45 s | 16 threads read the shards into page cache while SGLang is still importing. 4.3 GB/s, against 0.5–2 GB/s for SGLang's own reads. | Weight load 54 → 10–16 s. |
| 3 | **pow2 CUDA graph sizes** | ~17 s | Capture 9 decode sizes (1…256) and 12 prefill sizes (4…8192) instead of SGLang's 36 + 58. Batches pad up to the next captured size. | Capture 27 → 9.6 s. Throughput −2%, same outputs. |
| 4 | **Parallel cache restore** | 4–11 s | The restore moved off the critical path: SGLang launches first, and restore finishes long before the kernels are needed. | The launch delay was 3.5–11 s; now ~0 s. |
| 5 | **Bytecode in the image** | ~7 s | `compileall` at image build, so 3,900 modules aren't compiled on every cold import. | Paired runs on 3 hosts: 5.6–7.8 s per import. |
| 6 | **Lazy imports** | ~0.5 s/import | Build-time patch that defers `torchvision.io` (JPEG decode) and transformers' training-loss module, with asserts on the exact source lines. | Import profile. |
| 7 | **Import prefetch** | ~2–4 s (unconfirmed) | A build-time manifest of the 13.8k Python files imports read; 64 threads read them at container start (files > 16 MB skipped). | n=3 per arm; effect within host-to-host noise. |
| 8 | **Late GPU claim** | 235B: ~17 min of GPU | The weight download runs on a CPU sandbox before the GPU sandbox exists. | 235B: 1,004 s download on CPU. |

### CUDA graphs in detail

| | Vanilla | Compile cache, all 94 sizes | Compile cache + pow2 (21 sizes) |
|---|---:|---:|---:|
| Prefill graphs | 57–68 s (58 sizes) | 18.1 s | 6.5 s (12 sizes) |
| Decode graphs | 10.5–13.3 s (36 sizes) | 8.7 s | 3.1 s (9 sizes) |
| Graph memory (prefill + decode) | 1.69 GB | 1.69 GB | 1.18 GB |
| Throughput, 32 concurrent | 4,630 tok/s | 4,645 tok/s | 4,547 tok/s |
| Inter-token latency, 1 request | — | 3.45 ms | 3.50 ms |

**Caveat on pow2:** the worst-case padding is about 2×, for example 129 requests run as 256. The serving test used 32 concurrent requests and 1 request, both powers of two, so it's the favourable case. Non-power-of-two concurrency hasn't been measured yet.

## What didn't work

| Attempt | Result | Why |
|---|---|---|
| CUDA graphs off | Serving 10–19× slower (458 tok/s; 64.5 ms/token) | Kernel-launch overhead dominates for a 128-expert MoE |
| Decode graphs only up to batch 16 | 11% of throughput | Larger batches get no graph |
| Prefill graphs off | 85% throughput, 430 ms TTFT at 32 concurrent | Worse than pow2 on both axes |
| Text-only image (drop torchvision) | Server fails to start | SGLang registers image processors that require torchvision |
| ModelExpress P2P weight transfer | Not possible on Modal | No RDMA, and NIXL/UCX has no usable transport between sandboxes |
| ModelExpress streamer | 44–48 s vs 49–56 s load | A small gain, superseded by prefetch |
| Whole-manifest import prefetch (5.2 GB) | No gain; read took 28–41 s on GPU hosts | 95% of the bytes are large CUDA libraries, which are mmapped anyway |

## Several engines on one host

The 30B workload can also run as **two TP=2 engines on one 4× H100 host**: group A on GPUs 0–1, group B on GPUs 2–3, each with its own endpoint. The groups share one weight prefetch, one compile-cache restore and one import prefetch, and each can be stopped on its own.

| | Group A | Group B | Single engine, 2× H100 |
|---|---:|---:|---:|
| Weight load (shared page cache) | 13.8 s | 13.8 s | 10–16 s |
| Graph capture | 9.6 s | 9.8 s | ~9.6 s |
| Ready after container start | 126 s | 126 s | ~80 s |

Two engines were ready in about 1.6× the time of one (n=1). The extra time is concurrent imports and process spawn (about 90 s against about 50 s): two launchers and four TP workers import at once while the prefetches run. Staggering the two launches is the next thing to try.

## Measurement method

- **Harness:** each trial is a fresh Modal sandbox (`bench.py`), so every run starts cold.
  - Phase times come from SGLang's own log lines, timestamped on arrival.
  - GPU utilization and memory are sampled every 0.5 s.
- **Correctness:** 3 fixed prompts with greedy decoding, hashed. Every configuration reported here produced the same hash.
- **Serving:** 128 requests at 32 concurrent and 16 requests at 1, with 800-word prompts and 256 output tokens. Reports tok/s, TTFT and ITL (p50/p99).
- **Scheduling excluded:**
  - At start, the sandbox reports its own uptime. Since uptime begins when the container starts, *create → container start* is Modal scheduling.
  - Runs before this was added are estimated: container starts over 15 s, minus a typical 5 s boot. Those runs are flagged "est." in the UI.
- **Sample sizes are small** (n=2–9 per configuration), and host-to-host variation is large: import time alone varies about 2× across hosts. Treat differences under about 5 s as noise.

## Reliability fixes found along the way

- **Incomplete downloads:** a partial 235B download was once treated as complete, because only `config.json` was checked. Weights are now verified against every shard in the index plus the tokenizer before the catalog lists them.
- **Lifetime vs startup:** a 5-minute lifetime killed a 235B engine mid-load, because the sandbox timeout counts from container start. Lifetimes now have a per-model minimum (235B: 15 min), and a start that hits its lifetime is reported as `timeout`.
- **Host memory:** a vanilla 235B trial was OOM-killed at 79% of weight loading, with 8 ranks memory-mapping 470 GB. Host RAM is now sized from the checkpoint (weights ×2 + 64 GiB, limit 180 GiB per GPU; 940 GiB for 235B), which also leaves room for the weight prefetch.
- **Delayed log stream:** Modal sometimes delivers sandbox stdout minutes late. One engine that was ready at 87 s was reported only after its lifetime expired. Readiness should also be confirmed by polling `/health` through the tunnel (not yet done).

## Open items

1. **235B:** n=1 on each side; repeat both. Weight load is now 75 s; the rest of startup
   (imports, NCCL, KV cache setup: ~128 s) is the largest remaining piece.
3. Confirm readiness via `/health` polling, not only the log stream.
4. Serving cost of pow2 at non-power-of-two concurrency (for example 40, 96, 160) and longer prompts.
5. Larger samples (n ≥ 5) for the small effects: import prefetch, lazy imports.
