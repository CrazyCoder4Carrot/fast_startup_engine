"""Figures for docs/01_baseline_startup.md from results/20260923-184715."""
import csv, datetime as dt, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(__file__)
RUN = os.path.join(HERE, "..", "results", "20260923-184715")
OUT = os.path.join(HERE, "figures")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

# ---- Figure 1: cold vs warm stacked phase breakdown ----
groups = ["Imports & process spawn", "CUDA / NCCL init", "Weight load",
          "CUDA graph capture (incl. JIT)", "Other (model init, KV, warmup, HTTP)"]
cold = [68.0, 18.7, 53.8, 75.9, 20.6]
warm = [38.9, 1.6, 12.7, 29.3, 7.2]
rows = [("Cold (fresh container)", cold), ("Warm (caches populated)", warm)]

fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
gap, h = 0.6, 0.5
for yi, (label, vals) in enumerate(rows):
    y = len(rows) - 1 - yi
    x = 0
    for i, v in enumerate(vals):
        w = max(v - gap, 0.2)
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                     boxstyle="round,pad=0,rounding_size=0.06",
                     mutation_aspect=1 / 30, linewidth=0, facecolor=SERIES[i]))
        if v >= 9:
            ax.text(x + w / 2, y, f"{v:.0f}s", ha="center", va="center",
                    fontsize=10, color=INK if i in (2, 3, 4) else "white")
        x += v
    ax.text(x + 3, y, f"{sum(vals):.0f} s", va="center", ha="left",
            fontsize=12, fontweight="bold", color=INK)
ax.set_yticks([1, 0], [r[0] for r in rows], fontsize=11, color=INK)
ax.set_xlim(0, 262); ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("seconds from process spawn to first request served")
ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES]
ax.legend(handles, groups, loc="upper center", bbox_to_anchor=(0.45, -0.32),
          ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.2)
fig.suptitle("SGLang startup, Qwen3-30B-A3B TP=2 on 2×H100: cold vs warm",
             x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.01, 0.885, "Warm run reuses the same container: page cache, JIT caches and image files already local. "
         "Container start (13.9 s) not included.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT, "fig1_cold_vs_warm.png"))

# ---- Figure 2: GPU0 activity during the cold launch ----
p = lambda s: dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f")
spawn = dt.datetime(2026, 9, 23, 18, 47, 15, 240000)  # from log: server_args at 45.756 s = 18:48:01
t, util, mem = [], [], []
for r in csv.reader(open(os.path.join(RUN, "gpu.csv"))):
    if len(r) != 4 or r[1].strip() != "0":
        continue
    s = (p(r[0]) - spawn).total_seconds()
    if 0 <= s <= 237:
        t.append(s); util.append(int(r[2])); mem.append(int(r[3]) / 1024)

phases = [(0, 68.0, "imports"), (68.0, 86.7, "NCCL"), (86.7, 92.8, ""),
          (92.8, 146.6, "weight load"), (146.6, 150.0, ""),
          (150.0, 225.8, "CUDA graph capture"), (225.8, 237.0, "warmup")]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), dpi=200, sharex=True,
                             gridspec_kw={"hspace": 0.18})
for ax_ in (a1, a2):
    for i, (s, e, name) in enumerate(phases):
        if name in ("weight load", "CUDA graph capture"):
            ax_.axvspan(s, e, color="#f0efec", linewidth=0, zorder=0)
    ax_.yaxis.grid(True, color=GRID, linewidth=0.8); ax_.set_axisbelow(True)
    for sp in ("top", "right"): ax_.spines[sp].set_visible(False)
for s, e, name in phases:
    if name:
        a1.text((s + e) / 2, 108, name, ha="center", va="bottom", fontsize=9, color=INK2)
a1.plot(t, util, color=SERIES[0], linewidth=2)
a1.set_ylim(0, 105); a1.set_ylabel("GPU util (%)")
a2.plot(t, mem, color=SERIES[0], linewidth=2)
a2.set_ylim(0, 82); a2.set_yticks([0, 20, 40, 60, 80]); a2.set_ylabel("GPU memory (GB)")
a2.set_xlabel("seconds from process spawn (cold run, GPU 0)"); a2.set_xlim(0, 240)
fig.suptitle("The GPU is reserved for all 237 s but busy for a fraction of it",
             x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.995)
fig.text(0.01, 0.935, "nvidia-smi samples every 0.5 s. Shaded: weight load and CUDA graph capture. "
         "Samples with any kernel activity: ~50 s of 237 s.", fontsize=9, color=MUTED)
fig.subplots_adjust(top=0.84, left=0.07, right=0.98, bottom=0.1)
fig.savefig(os.path.join(OUT, "fig2_gpu_idle_cold.png"))
print("ok")
