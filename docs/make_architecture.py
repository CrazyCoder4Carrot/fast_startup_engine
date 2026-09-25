"""Architecture diagram for presentations -> docs/figures/architecture.png (1920x1080).

Two rows, straight arrows only, short labels on the arrows.

    uv run --with matplotlib python docs/make_architecture.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures", "architecture.png")

# Reference palette, first three categorical slots (validated all-pairs). Outline colour = kind:
# blue = a service we run, green = a Modal sandbox, orange = storage.
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, MUTED, SURFACE, LINE = "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb", "#d6d5cf"
TINT = {"blue": "#eaf2fc", "aqua": "#e6f6f0", "grey": "#f3f3f1"}

W, H = 100, 56.25  # 16:9 canvas in layout units
fig = plt.figure(figsize=(12, 6.75), dpi=160)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def zone(x, y, w, h, tint, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=TINT[tint], ec="none", zorder=0))
    ax.text(x + 1.4, y + h - 1.6, label, fontsize=12, fontweight="bold", color=INK2, va="top", zorder=1)


def box(x, y, w, h, title, lines=(), color=None, note=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                fc="white", ec=color or LINE, lw=2.0 if color else 1.2, zorder=2))
    if not lines:  # title only: centre it
        ax.text(x + w / 2, y + h / 2, title, fontsize=13, fontweight="bold", color=INK, ha="center", va="center", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2 + 1.1, title, fontsize=13, fontweight="bold", color=INK, ha="center", va="center", zorder=3)
    if note:
        ax.text(x + w - 1.2, y + h - 1.55, note, fontsize=9, color=MUTED, ha="right", va="top", zorder=3)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 - 1.5 - i * 2.2, t, fontsize=9.5, color=INK2, ha="center", va="center", zorder=3)


def arrow(points, both=False, dashed=False):
    """Straight or right-angled arrow through `points`; the head is on the last segment."""
    kw = dict(color=MUTED, lw=1.7, zorder=4, solid_capstyle="butt")
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], linestyle=(0, (4, 3)) if dashed else "-", **kw)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="<|-|>" if both else "-|>", mutation_scale=14,
                                 lw=1.7, color=MUTED, zorder=4, shrinkA=0, shrinkB=2,
                                 linestyle=(0, (4, 3)) if dashed else "-"))


# ---- title
ax.text(2.5, H - 2.5, "Fast engine startup · architecture", fontsize=20, fontweight="bold", color=INK, va="top")


def label(x, y, t, ha="center"):
    ax.text(x, y, t, fontsize=10, color=INK2, ha=ha, va="center", zorder=7, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))


# ---- zones
BOT, TOP = 3.0, 46.5
zone(1.5, BOT, 12.5, TOP - BOT, "grey", "You")
zone(15.5, BOT, 40.5, TOP - BOT, "blue", "Control plane")
zone(58.5, BOT, 40.0, TOP - BOT, "aqua", "Modal")

RA, RB, BH = 30.0, 10.0, 7.0  # row A / row B bottoms, box height

# ---- boxes
box(3.0, RA, 9.0, BH, "Web UI")
box(18.5, RA, 13.0, BH, "API", color=BLUE)
box(40.5, RA, 13.5, BH, "Job queue", color=ORANGE)
box(18.5, RB, 13.0, BH, "Catalog", color=ORANGE)
box(40.5, RB, 13.5, BH, "Worker", color=BLUE)
box(61.0, RA, 14.0, BH, "CPU prep", ["first run only"], AQUA)
box(83.0, RA, 14.0, BH, "Volumes", color=ORANGE)
box(61.0, RB, 36.0, BH, "GPU engine", ["runs SGLang"], AQUA)

# ---- flows
MA, MB = RA + BH / 2, RB + BH / 2          # row mid-lines
arrow([(12.0, MA), (18.5, MA)]);             label(15.25, MA + 2.2, "submit")
arrow([(31.5, MA), (40.5, MA)]);             label(36.0, MA + 2.2, "enqueue")
arrow([(47.25, RA), (47.25, RB + BH)]);      label(48.3, (RA + RB + BH) / 2, "claim", ha="left")
arrow([(40.5, MB), (31.5, MB)], both=True);  label(36.0, MB + 2.2, "look up")
arrow([(54.0, MB + 1.5), (57.5, MB + 1.5), (57.5, MA), (61.0, MA)], dashed=True)
label(57.5, (MA + MB) / 2, "download\n(first run only)")
arrow([(75.0, MA), (83.0, MA)], dashed=True); label(79.0, MA + 2.2, "writes")
arrow([(90.0, RA), (90.0, RB + BH)]);        label(91.0, (RA + RB + BH) / 2, "mount", ha="left")
arrow([(54.0, MB - 1.5), (61.0, MB - 1.5)]); label(57.5, MB - 3.8, "start")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160, facecolor=SURFACE)
print("wrote", OUT)
