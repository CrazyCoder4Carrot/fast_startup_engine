"""Architecture diagram for presentations -> docs/figures/architecture.png (1920x1080).

    uv run --with matplotlib python docs/make_architecture.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures", "architecture.png")

# Reference palette, first three categorical slots (validated all-pairs): one hue per zone.
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, MUTED, SURFACE, LINE = "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb", "#d6d5cf"
TINT = {BLUE: "#eaf2fc", AQUA: "#e6f6f0", None: "#f3f3f1"}

W, H = 100, 56.25  # 16:9 canvas in layout units
fig = plt.figure(figsize=(12, 6.75), dpi=160)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def zone(x, y, w, h, color, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=TINT[color], ec="none", zorder=0))
    ax.text(x + 1.4, y + h - 1.6, label, fontsize=12, fontweight="bold", color=INK2, va="top", zorder=1)


def box(x, y, w, h, title, lines=(), color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                fc="white", ec=color or LINE, lw=1.8 if color else 1.2, zorder=2))
    ax.text(x + 1.3, y + h - 1.4, title, fontsize=13, fontweight="bold", color=INK, va="top", zorder=3)
    for i, t in enumerate(lines):
        ax.text(x + 1.3, y + h - 4.1 - i * 2.2, t, fontsize=9.5, color=INK2, va="top", zorder=3)


def arrow(p, q, label=None, lx=0.0, ly=0.0, both=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=14,
                                 lw=1.6, color=MUTED, zorder=4, shrinkA=2, shrinkB=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=9, color=INK2,
                ha="center", va="center", zorder=5, bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="none"))


# ---- title
ax.text(2.5, H - 2.5, "Fast engine startup · architecture", fontsize=20, fontweight="bold", color=INK, va="top")
ax.text(2.5, H - 6.3, "SGLang cold start on Modal · Qwen3-30B on 2×H100 · Qwen3-235B on 8×B200",
        fontsize=11, color=INK2, va="top")

# ---- zones
BOT, TOP = 4.0, 45.5
zone(1.5, BOT, 14.0, TOP - BOT, None, "Clients")
zone(17.5, BOT, 39.0, TOP - BOT, BLUE, "Control plane")
zone(58.5, BOT, 40.0, TOP - BOT, AQUA, "Modal")

# ---- boxes
box(3.0, 22.5, 11.0, 8.0, "Web UI", ["and JSON API"])
box(19.5, 27.5, 16.0, 11.0, "API", ["writes jobs", "streams progress"], BLUE)
box(38.5, 27.5, 16.0, 11.0, "Job queue", ["Postgres table", "(Docker)"], ORANGE)
box(19.5, 7.5, 16.0, 11.0, "Catalog", ["what is cached", "learned timings"], ORANGE)
box(38.5, 7.5, 16.0, 11.0, "Worker", ["claims a job", "plans and runs it"], BLUE)

box(60.5, 27.5, 36.0, 11.0, "GPU engine",
    ["1  prefetch weights into RAM", "2  restore compiled kernels", "3  start SGLang (fewer CUDA graphs)"], AQUA)
ax.text(95.2, 37.1, "all three at once", fontsize=9, color=MUTED, ha="right", va="top", zorder=3)
box(60.5, 7.5, 16.5, 11.0, "CPU prep", ["downloads weights", "no GPU billed"], AQUA)
box(80.0, 7.5, 16.5, 11.0, "Volumes", ["one set per model"], ORANGE)

# ---- flows
arrow((14.0, 27.0), (19.5, 33.0))
arrow((35.5, 33.0), (38.5, 33.0))
arrow((46.5, 27.5), (46.5, 18.5), "claim", lx=2.4)
arrow((38.5, 13.0), (35.5, 13.0), both=True)
arrow((54.5, 15.5), (60.5, 30.5), "start /\nprogress", lx=-0.4, ly=0.4, both=True)
arrow((54.5, 11.0), (60.5, 11.0))
arrow((77.0, 13.0), (80.0, 13.0))
arrow((88.25, 18.5), (88.25, 27.5), both=True)
ax.text(89.2, 23.0, "mount", fontsize=9, color=INK2, va="center", zorder=5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160, facecolor=SURFACE)
print("wrote", OUT)
