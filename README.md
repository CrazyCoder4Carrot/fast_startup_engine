# Fast engine startup

Measuring and shortening SGLang engine cold start on Modal, with a control plane that plans each
start from what is already cached, claims GPUs late, and learns its estimates from every launch.

- Models: `Qwen/Qwen3-30B-A3B-Instruct-2507` (2×H100, TP=2) and `Qwen/Qwen3-235B-A22B-Instruct-2507` (8×B200, TP=8), BF16
- Engine: SGLang `v0.5.20` (`lmsysorg/sglang:v0.5.20`)
- Results so far: 30B startup 229.8 s → 84.5 s (median, 2.7×), 235B 951.5 s → 220.8 s (4.3×); see `docs/02_implementation.md`

## Setup

Needs [uv](https://docs.astral.sh/uv/), Docker (Docker Desktop or OrbStack), and a Modal account
(`uv run modal setup`) with a `huggingface-secret` holding `HF_TOKEN`.

```bash
docker compose up -d --build   # Postgres + API + worker, each in its own container: http://127.0.0.1:8060
```

The API and the worker run from one image (`Dockerfile`) as separate containers, so rebuilding the
API after a code change (`docker compose up -d --build api`) never interrupts a running job. Both
mount `~/.modal.toml` read-only for Modal access (never copied into the image) and `./results`
for trial results. For local development without the app containers:

```bash
uv sync                        # installs the fes package and its commands
docker compose up -d postgres  # only the database
uv run fes-control-plane       # UI + API + worker in one process
```

Weights need no separate download step: on a model's first start the scheduler downloads them on
a CPU sandbox into the model's own volume (`fes-models-<model>`) before claiming any GPU.

## Database

Postgres 16 in Docker (`docker-compose.yml`), listening on `127.0.0.1:5432`. It holds the jobs table
(which is also the job queue), job events, worker heartbeats, and the artifact catalog with the
learned timings.

```bash
docker compose down                    # stop everything, keep the data (add -v to delete it)
docker compose logs -f worker          # follow the worker (or api, postgres)
docker exec -it fes-postgres psql -U fes -d fes     # look inside the database
```

The control plane connects with `DATABASE_URL`: `postgresql://fes:fes@127.0.0.1:5432/fes` by default
(`fes/db.py`), and `postgres:5432` inside compose. Set `FES_PG_PASSWORD` before the first `docker compose up` to use another password,
and put it in `DATABASE_URL` too. The earlier SQLite files are kept in `db/` as a backup;
`tmp/migrate_sqlite_to_postgres.py` copied them over.

## Control plane

In Docker (above) the API and the worker are already separate containers. Without Docker:

```bash
uv run fes-control-plane                  # UI + API + worker in one process
# or as two processes (restarting the API never interrupts a running job):
uv run fes-control-plane --role worker    # claims queued jobs from the database and runs them
uv run fes-control-plane --role api       # UI + API: writes jobs to the database, runs nothing
```

Pages at http://127.0.0.1:8060: **Engines** (start one: workload, CUDA graphs, weight loader,
lifetime; live prediction), **Jobs** (history and live progress), **Playground** (prompt a serving
engine), **Catalog** (artifacts, volumes, learned timings), **Compare** (vanilla vs current, per
workload), **Analysis** (experiments, benchmark report, architecture).

Scripting uses the same JSON API as the pages (route list in `fes/api.py`), e.g.:

```bash
curl -X POST localhost:8060/api/jobs -H 'Content-Type: application/json' \
     -d '{"kind": "start_engine", "params": {"model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "tp": 2, "gpu": "H100!:2", "ttl_s": 300}}'
```

Start options: `model`, `tp`, `gpu`, `groups` (engines per host, e.g. 2 on 4×H100), `graph_sizes`
(`pow2` | `full`), `modelexpress` (weight loader), `policy` (`late_claim` | `asap`), `ttl_s` (lifetime,
with a per-model minimum), `extra_args` (more SGLang flags).

## Benchmarks

```bash
uv run fes-bench -a baseline                        # vanilla cold start in a fresh sandbox (the "original")
uv run fes-bench -a baseline,graphs_pow2 -n 3       # compare approaches, 3 cold trials each
uv run fes-bench --gpu 'H100!:2' --tp 2 --groups 2  # two engines on one 4×H100 host
uv run fes-bench --model Qwen/Qwen3-235B-A22B-Instruct-2507 --gpu B200:8 --tp 8
uv run fes-bench --list                             # all approaches
uv run fes-sync-results                             # pull trials that exist on the fes-results volume only
```

Each trial lands in `results/<run_id>/`; the Analysis and Compare pages read it from there.

## How a start works

1. **Plan** (`scheduler.plan`): are the weights on the model's volume? Is there a compile cache
   for this SGLang version, GPU, TP and model config? Host RAM is sized from the checkpoint, and the
   weights are prefetched if they fit. Each step is estimated from the model's learned timings.
2. **CPU prep** (only if weights are missing): download on a CPU sandbox; no GPU is billed.
3. **GPU sandbox** (`fes/engine/engine_agent.py`): at container start it prefetches weights and import
   files into RAM, restores the compile cache, and launches SGLang (one server per GPU group) with
   power-of-two CUDA graph sizes, all at once.
4. **Ready**: the first request is served; the engine is registered with its tunnel URL, and the
   measured phases update the learned timings. The reconciler stops it at its lifetime.

## Layout

```
fes/                        the system (installed by `uv sync`; the commands above are its entry points)
  control_plane.py          entrypoint: --role api | worker | all
  api.py                    HTTP API (Flask), pages, SSE streams, engine proxy for the Playground
  db.py                     Postgres connection (DATABASE_URL) and schema setup
  store.py                  jobs / events / workers; the jobs table is the queue (FOR UPDATE SKIP LOCKED)
  jobs.py                   JobAPI (writes job rows) and JobWorker (claims and runs them)
  reconciler.py             engine exit codes and lifetimes; reaps jobs of dead workers
  scheduler.py              plan() and execute(): artifacts -> steps -> predicted time to ready
  catalog.py                artifact catalog + per-model learned timings
  workers.py                CPU prep and GPU engine sandboxes on Modal
  common.py                 Modal volumes (one set per model), images, RAM and CPU sizing
  bench.py                  cold-start benchmark harness (one fresh sandbox per trial)
  dashboard.py              result parsing for the Analysis and Compare pages
  paths.py                  repository paths
  engine/                   uploaded into GPU sandboxes (stdlib only, imported by plain name in /root)
    engine_agent.py         entrypoint of an engine sandbox: prefetch, launch SGLang, report
    sglang_engine.py        engine library: launch SGLang, phase parsing, compile cache, prefetch, ModelExpress
    bench_runner.py         benchmark trial on top of it: correctness, serving load, probes
  patches/                  run at image build: lazy imports, import manifest
  web/                      pages
experiments/modelexpress/   ModelExpress scale-out experiment (4×H100); also the fes-bench mx_streamer image
docs/                       reports (Markdown + .docx), figure and architecture-diagram scripts
Dockerfile                  control-plane image (API and worker)
docker-compose.yml          Postgres + API + worker
db/                         the earlier SQLite files, kept as a backup (gitignored)
results/                    trial results and scheduler start records (gitignored)
tmp/                        retired scripts (one-off probes, first setup/profiler, the SQLite migration) (gitignored)
```
