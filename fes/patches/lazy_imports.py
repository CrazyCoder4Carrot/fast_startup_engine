"""Defer imports a text-only server never uses, at image build time.

Measured (warm import, bytecode baked): ~1.8 s of ~7.8 s per process goes to
    torchvision.io   1.26 s  <- sglang/srt/utils/common.py, only for JPEG decoding
    loss_utils       0.55 s  <- transformers/modeling_utils.py, only for training losses
                               (pulls in scipy.optimize, torchaudio, detection losses)
Both are `from X import Y` imports, so importlib's LazyLoader can't help; each line is
replaced by a stand-in that performs the real import on first use.

Every patch asserts the exact original line, so a version bump fails the build
instead of patching the wrong code. Originals are kept as <file>.orig.
"""

import shutil
import sys

SG = "/sgl-workspace/sglang/python/sglang/srt/utils/common.py"
TF = "/opt/sglang/lib/python3.12/site-packages/transformers/modeling_utils.py"

PATCHES = [
    (SG, "from torchvision.io import decode_jpeg\n", '''def decode_jpeg(*args, **kwargs):  # lazy: torchvision.io costs ~1.3 s to import
    from torchvision.io import decode_jpeg as _decode_jpeg

    return _decode_jpeg(*args, **kwargs)
'''),
    (TF, "from .loss.loss_utils import LOSS_MAPPING\n", '''class _LazyLossMapping:  # lazy: loss_utils imports scipy/torchaudio (~0.55 s); inference never needs it
    def _m(self):
        from .loss.loss_utils import LOSS_MAPPING as _mapping

        return _mapping

    def __getitem__(self, k):
        return self._m()[k]

    def __contains__(self, k):
        return k in self._m()

    def __iter__(self):
        return iter(self._m())

    def __len__(self):
        return len(self._m())

    def get(self, k, default=None):
        return self._m().get(k, default)

    def keys(self):
        return self._m().keys()


LOSS_MAPPING = _LazyLossMapping()
'''),
]


def main() -> int:
    """Apply the patches in order; exit 1 at the first whose original line isn't found exactly
    once, which fails the image build instead of shipping a half-understood patch."""
    for path, old, new in PATCHES:
        src = open(path).read()
        if src.count(old) != 1:
            print(f"lazy_imports: expected exactly one {old.strip()!r} in {path}; refusing to patch", file=sys.stderr)
            return 1
        shutil.copy(path, path + ".orig")
        open(path, "w").write(src.replace(old, new))
        print(f"lazy_imports: patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
