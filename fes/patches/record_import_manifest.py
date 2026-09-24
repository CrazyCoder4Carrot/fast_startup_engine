"""Record the files an engine's imports read, at image build time -> /root/import_manifest.json.

Modal fetches image files lazily, on first read, so a cold import pays one fetch per file
(~14k files). The engine agent reads this list at container start (files over 16 MB skipped,
at most 2x the requested CPUs in parallel), while SGLang imports, so the imports find the
files local. Measured on CPU sandboxes: the cold-read penalty drops from 6-10 s to 1-3 s per
import. On real starts the gain was ~2-4 s, within host-to-host noise (n=3 per arm).

Order: Python files in import order (the order sys.modules fills), then shared libraries.
Recorded after bytecode compilation, so .pyc paths are the ones imports will read.
"""

import json
import sys
import time

t0 = time.time()
import sglang.launch_server  # noqa: E402,F401  launcher process
import sglang.srt.managers.scheduler  # noqa: E402,F401  TP worker process
import sglang.srt.model_executor.model_runner  # noqa: E402,F401
import sglang.srt.models.qwen3_moe  # noqa: E402,F401  both served models are Qwen3-MoE

py, libs = [], []
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if not f:
        continue
    cached = getattr(getattr(m, "__spec__", None), "cached", None)
    py += [cached, f] if cached else [f]
for line in open("/proc/self/maps"):
    p = line.split()[-1]
    if p.startswith("/") and ".so" in p:
        libs.append(p)
seen: set = set()
files = [f for f in py + libs if not (f in seen or seen.add(f))]
json.dump(files, open("/root/import_manifest.json", "w"))
print(f"import manifest: {len(files)} files ({len(libs)} shared-library mappings), imports took {time.time() - t0:.1f} s")
