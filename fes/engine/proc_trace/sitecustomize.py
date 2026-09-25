"""Process-start marker for the `trace_imports` benchmark approach (on PYTHONPATH only there).

Python runs sitecustomize at interpreter start in every process, including the TP workers
SGLang spawns, so one line per process marks when it started, before any of its imports. The
runner timestamps the line. The image's own sitecustomize (Debian's) is then loaded as usual.
"""

import os
import sys

if os.environ.get("FES_TRACE_PROCS"):
    try:
        cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:160]
    except OSError:
        cmd = "?"
    sys.stderr.write(f"__FESPROC__ pid={os.getpid()} ppid={os.getppid()} cmd={cmd}\n")
    sys.stderr.flush()

_system = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_system):
    import importlib.util as _u
    _spec = _u.spec_from_file_location("_system_sitecustomize", _system)
    try:
        _spec.loader.exec_module(_u.module_from_spec(_spec))
    except Exception:
        pass
