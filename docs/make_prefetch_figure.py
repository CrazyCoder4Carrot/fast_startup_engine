"""Weight prefetch, simple -> docs/figures/prefetch_threads.png (1280x448).

Each thread copies one whole shard file from the network volume into RAM (the page cache), all at
once (fes/engine/sglang_engine.py, prefetch_weights()).

    uv run --with matplotlib python docs/make_prefetch_figure.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures", "prefetch_threads.png")
BLUE, INK, INK2, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb"

W, H = 100, 35.0  # 1280 x 448
fig = plt.figure(figsize=(8, 2.8), dpi=160)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def rbox(x, y, w, h, fc, ec, text="", color=INK, weight="normal", mono=False, r=0.8, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.3, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, fontsize=size, color=color, weight=weight, ha="center", va="center",
                zorder=3, family="monospace" if mono else None)


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.6, color=MUTED, zorder=4))



FX, TX, RX = 4, 39, 70           # columns: files, threads, RAM
FW, TW, RW = 20, 15, 26
for x, w, t in ((FX, FW, "Network volume"), (TX, TW, "Threads"), (RX, RW, "RAM (page cache)")):
    ax.text(x + w / 2, 31.5, t, fontsize=11, weight="bold", color=INK2, ha="center")
rbox(RX, 4.5, RW, 25.0, "#e6f6f0", "#9fdcc0", r=1.2)

rows = [("model-00001", "thread 1"), ("model-00002", "thread 2"), ("⋮", "⋮"), ("model-00016", "thread 16")]
for i, (f, t) in enumerate(rows):
    y = 24.0 - i * 6.2
    if f == "⋮":
        for x, w in ((FX, FW), (TX, TW), (RX, RW)):
            ax.text(x + w / 2, y + 1.9, "⋮", fontsize=15, color=MUTED, ha="center", va="center")
        continue
    rbox(FX, y, FW, 3.8, "#fde2e2", "#f5b4b4", f, mono=True)
    rbox(TX, y, TW, 3.8, "#e8f0fd", "#93b4ea", t, color=BLUE, weight="bold", r=1.8)
    rbox(RX + 1.5, y, RW - 3.0, 3.8, "#a7e3c9", "none", f, color="#065f46", mono=True)
    arrow((FX + FW + 0.5, y + 1.9), (TX - 0.5, y + 1.9))
    arrow((TX + TW + 0.5, y + 1.9), (RX + 1.0, y + 1.9))

ax.text(4, 1.2, "All at once, while SGLang imports. SGLang then loads from RAM: weight load 62 → 13 s (Qwen3-30B).",
        fontsize=9.5, color=INK2, va="bottom")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160, facecolor=SURFACE)
print("wrote", OUT)
