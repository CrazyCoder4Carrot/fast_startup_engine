"""Build docs/slides.html: a self-contained HTML deck (no external files). Arrow keys / space / click to move; print gives one slide per page.

Structure: define the metrics -> find the critical path -> a different solution for each
component (and what was rejected) -> make it work, then make it reliable (the control plane's storage:
JSON -> SQLite -> Postgres in Docker) -> architecture, results, future work.
Style follows the "Fast Engine Startup" Google Slides deck: Arial, navy text, pale panels,
pros/cons lists, struck-through REJECTED slides.

    uv run python docs/make_slides.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Phase medians per workload (Compare page; startup excludes Modal scheduling).
PHASES = ["Imports & process spawn", "CUDA / NCCL init", "Weight load", "CUDA graph capture", "Other (init, KV, warmup)"]
BARS = {
    "Qwen3-30B · 2×H100": [("Vanilla", [57.8, 19.2, 61.5, 71.7, 18.5], 229.8),
                           ("Now", [47.9, 1.0, 13.1, 9.2, 8.1], 84.5)],
    "Qwen3-30B · 4×H100 · 2 engines": [("Vanilla", [71.8, 17.8, 48.9, 67.9, 17.4], 223.9),
                                       ("Now", [83.9, 1.6, 14.3, 10.0, 10.3], 155.5)],
    "Qwen3-235B · 8×B200": [("Vanilla", [152.2, 24.9, 564.4, 79.7, 129.8], 951.5),
                            ("Now", [102.0, 9.8, 75.0, 13.9, 16.7], 220.8)],
}
THEME = "midnight"  # default; also minimal | dark | editorial (press T in the deck to cycle)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # reference palette, fixed order


def bars(title: str, only_first: bool = False, rename: dict | None = None) -> str:
    rows = BARS[title][:1] if only_first else BARS[title]
    scale = max(t for *_, t in BARS[title])
    out = [f'<div class="chart"><div class="chart-title">{title}</div>']
    for label, segs, total in rows:
        label = (rename or {}).get(label, label)
        cells = "".join(
            f'<div class="seg" style="width:{100 * v / scale:.2f}%;background:{COLORS[i]}" '
            f'title="{PHASES[i]}: {v:.1f} s">{f"{v:.0f}" if v / scale > 0.06 else ""}</div>' for i, v in enumerate(segs))
        out.append(f'<div class="bar-row"><div class="bar-label">{label}</div><div class="bar">{cells}</div>'
                   f'<div class="bar-total">{total:.1f} s</div></div>')
    return "\n".join(out) + "</div>"


RESULTS_LABELS = {"Vanilla": "Baseline"}

STAGE_FIX = ["bytecode, import prefetch", "compile cache", "prefetch into RAM", "compile cache + pow2", "compile cache"]
STAGE_NAME = ["Imports + spawn", "CUDA / NCCL init", "Weight load", "CUDA graphs", "Other (KV, warmup)"]


def stage_table(title: str, base_total: float, now_total: float, cost: str, pct: bool = False) -> str:
    """Stage-by-stage baseline vs now for one workload, plus the time outside SGLang's phases."""
    (_, base, _), (_, now, _) = BARS[title]
    rows = [(STAGE_NAME[i], b, n, STAGE_FIX[i]) for i, (b, n) in enumerate(zip(base, now))]
    rows.append(("Outside the phases", base_total - sum(base), now_total - sum(now), "container start, first request"))
    def change(b, n):  # now - baseline: negative = faster; pct: relative to the baseline stage
        d = n - b
        cls = "good" if d < -0.5 else ("bad" if d > 0.5 else "")
        sign = "−" if d < 0 else "+"
        if pct and b >= 2:  # a percentage of a ~1 s baseline would mean nothing
            return f'<td class="{cls}">{sign}{abs(100 * d / b):.0f}%</td>'
        return f'<td class="{cls}">{sign}{abs(d):.1f} s</td>'
    tr = "".join(f"<tr><td>{name}</td><td>{b:.1f} s</td><td>{n:.1f} s</td>{change(b, n)}<td>{fix}</td></tr>"
                 for name, b, n, fix in rows)
    total = (f"−{100 * (1 - now_total / base_total):.0f}% ({base_total / now_total:.1f}×)" if pct
             else f"{base_total / now_total:.1f}× faster")
    return (f'<table class="plain compact stage"><tr><th>Stage</th><th>Baseline</th><th>Now</th><th>Change</th><th>What changed</th></tr>{tr}'
            f'<tr class="tot"><td>Startup</td><td>{base_total:.1f} s</td><td>{now_total:.1f} s</td><td>{total}</td>'
            f'<td>GPU cost per start {cost}</td></tr></table>')



def phase_savings(title: str, base_total: float, now_total: float) -> str:
    """Savings by phase, as on the Compare page: baseline (grey) and now (blue) bars per phase,
    the change on the right, and startup as the last row."""
    (_, base, _), (_, now, _) = BARS[title]
    rows = [(STAGE_NAME[i], STAGE_FIX[i], b, n) for i, (b, n) in enumerate(zip(base, now))]
    rows.append(("Startup", "", base_total, now_total))
    W, left, right, rh, gi, go, top = 1000, 190, 235, 13, 3, 15, 4
    H = top + len(rows) * (2 * rh + gi + go) + 20
    xmax = max(base_total, now_total) * 1.02
    step = next(t for t in (10, 20, 25, 50, 100, 200, 250) if xmax / t <= 6)
    x = lambda v: left + v / xmax * (W - left - right)
    out = [f'<svg class="savings" viewBox="0 0 {W} {H}" role="img">']
    for t in range(0, int(xmax) + 1, step):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{H - 18}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{H - 4}" class="tick">{t}s</text>')
    y = top
    for name, fix, b, n in rows:
        total = name == "Startup"
        out.append(f'<text x="{left - 12}" y="{y + rh + (0 if fix else 4)}" class="lab{" tot" if total else ""}">{name}</text>')
        if fix:
            out.append(f'<text x="{left - 12}" y="{y + rh + 15}" class="fix">{fix}</text>')
        for v, cls in ((b, "base"), (n, "now")):
            out.append(f'<rect x="{left}" y="{y}" width="{max(x(v) - left, 1.5):.1f}" height="{rh}" rx="3" class="{cls}"/>'
                       f'<text x="{x(v) + 6:.1f}" y="{y + rh - 2.5}" class="val">{v:.1f} s</text>')
            y += rh + gi
        d = n - b
        cls = "good" if d < -0.5 else ("bad" if d > 0.5 else "flat")
        out.append(f'<text x="{W - 2}" y="{y - rh - gi / 2 - 1}" class="delta {cls}" text-anchor="end">'
                   f'{"−" if d < 0 else "+"}{abs(d):.1f} s ({"−" if d < 0 else "+"}{abs(100 * d / b):.0f}%)</text>')
        y += go - gi
    out.append("</svg>")
    return "".join(out)


SIDES_LEGEND = '<div class="legend sides"><span><i class="lb"></i>Baseline</span><span><i class="ln"></i>Now</span></div>'


LEGEND = '<div class="legend">' + "".join(
    f'<span><i style="background:{c}"></i>{p}</span>' for p, c in zip(PHASES, COLORS)) + "</div>"


def section(title: str, sub: str = "") -> str:
    num, name = title.split(" · ", 1)
    major, _, minor = num.partition(".")
    label = f"{int(major):02d}" + (f".{minor}" if minor else "")
    return (f'<section class="divider{" sub" if minor else ""}" data-kicker="{name}"><div class="num">{label}</div><h1>{name}</h1>'
            f'{f"<p>{sub}</p>" if sub else ""}</section>')


def proscons(title: str, what: str, pros: list[str], cons: list[str], side: str = "", tag: str = "", label: str = "") -> str:
    li = lambda xs, s: "".join(f"<li><span class=\"pm\">{s}</span>{x}</li>" for x in xs)
    return f"""<section{f' data-label="{label}"' if label else ''}>
  <h2>{title}{f' <span class="tag">{tag}</span>' if tag else ''}</h2>
  <p class="what">{what}</p>
  <div class="pc">
    <div><h4 class="pro">Pros</h4><ul class="pmlist pros">{li(pros, "+")}</ul>
         {f'<h4 class="con">Cons</h4><ul class="pmlist cons">{li(cons, "−")}</ul>' if cons else ''}</div>
    <div>{side}</div>
  </div>
</section>"""


def rejected(title: str, pros: list[str], cons: list[str], why: str, label: str = "") -> str:
    li = lambda xs, m: "".join(f"<li><span class=\"pm\">{m}</span>{x}</li>" for x in xs)
    return f"""<section class="rejected"{f' data-label="{label}"' if label else ''}>
  <h2>{title} <span class="badge">Rejected</span></h2>
  <div class="panels">
    <div class="panel green"><h4>Upside</h4><ul class="pmlist pros">{li(pros, "+")}</ul></div>
    <div class="panel red"><h4>Downside</h4><ul class="pmlist cons">{li(cons, "−")}</ul></div>
  </div>
  {f'<p class="why"><b>Decision</b>{why}</p>' if why else ''}
</section>"""




def timeline(rows: list[tuple[str, list[tuple[float, float, str, str]]]], scale: float, ticks: list[int]) -> str:
    """Horizontal timeline: rows of (label, [(start_s, end_s, text, css class)]) on a 0..scale s axis."""
    out = ['<div class="tl">']
    for label, segs in rows:
        cells = "".join(f'<div class="tl-seg {c}" style="left:{100 * a / scale:.2f}%;width:{100 * (b - a) / scale:.2f}%">{t}</div>'
                        for a, b, t, c in segs)
        out.append(f'<div class="tl-row"><div class="tl-label">{label}</div><div class="tl-track">{cells}</div></div>')
    marks = "".join(f'<span style="left:{100 * t / scale:.2f}%">{t} s</span>' for t in ticks)
    out.append(f'<div class="tl-row"><div class="tl-label"></div><div class="tl-axis">{marks}</div></div></div>')
    return "\n".join(out)



def sequence(lanes: list[tuple[str, str, str]], events: list[tuple], height: int, blocked: tuple | None = None) -> str:
    """SVG sequence diagram. lanes: (title, subtitle, css class). events: ("msg", from, to, y, text, dashed)
    or ("note", lane, y, text). blocked: (lane, y0, y1, label), a bar on that lane's lifeline."""
    W = 1000
    xs = [W * (i + 0.5) / len(lanes) for i in range(len(lanes))]
    sequence.n = getattr(sequence, "n", 0) + 1
    mid = f"ah{sequence.n}"  # marker ids are document-wide: one per diagram
    out = [f'<svg class="seqd" viewBox="0 0 {W} {height}" role="img">',
           f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           '<path d="M0,0 L10,5 L0,10 z" class="ahead"/></marker></defs>']
    for x, (t, sub_, cls) in zip(xs, lanes):
        out.append(f'<rect x="{x - 105}" y="4" width="210" height="50" rx="9" class="lane {cls}"/>'
                   f'<text x="{x}" y="26" class="lt">{t}</text><text x="{x}" y="44" class="ls">{sub_}</text>'
                   f'<line x1="{x}" y1="56" x2="{x}" y2="{height - 4}" class="life"/>')
    if blocked:
        lane, y0, y1, label = blocked
        x = xs[lane]
        out.append(f'<rect x="{x - 7}" y="{y0}" width="14" height="{y1 - y0}" rx="4" class="blk"/>')
        if len(label) <= 4:  # short labels read better horizontally
            out.append(f'<text x="{x - 14}" y="{(y0 + y1) / 2 + 4}" class="blkt" text-anchor="end" style="text-anchor:end">{label}</text>')
        else:
            out.append(f'<text x="{x - 16}" y="{(y0 + y1) / 2}" class="blkt" transform="rotate(-90 {x - 16} {(y0 + y1) / 2})">{label}</text>')
    out.append(f'<marker id="{mid}h" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahead hl"/></marker>')
    for e in events:  # bands first, so they sit behind everything
        if e[0] == "band":
            _, a, b, y0, y1 = e
            out.append(f'<rect x="{xs[a] - 20}" y="{y0}" width="{xs[b] - xs[a] + 40}" height="{y1 - y0}" rx="10" class="band"/>')
    for e in events:
        if e[0] == "band":
            continue
        if e[0] == "msg":
            _, a, b, y, text, dashed, *hl = e
            hl = bool(hl and hl[0])
            x1, x2 = xs[a], xs[b]
            pad = 9 if x2 > x1 else -9
            cls = "msg" + (" dash" if dashed else "") + (" hl" if hl else "")
            out.append(f'<line x1="{x1 + pad}" y1="{y}" x2="{x2 - pad}" y2="{y}" class="{cls}" marker-end="url(#{mid}{"h" if hl else ""})"/>'
                       f'<text x="{(x1 + x2) / 2}" y="{y - 8}" class="mt{" hl" if hl else ""}">{text}</text>')
        else:
            _, lane, y, text = e
            x = xs[lane]
            out.append(f'<rect x="{x + 14}" y="{y - 17}" width="{14 + 7.1 * len(text)}" height="26" rx="6" class="note"/>'
                       f'<text x="{x + 20}" y="{y}" class="nt">{text}</text>')
    out.append("</svg>")
    return "".join(out)



def prefetch_diagram() -> str:
    """Weight prefetch: each thread reads one whole shard file from the volume into the page cache."""
    rows = [("model-00001", "thread 1"), ("model-00002", "thread 2"), ("model-00003", "thread 3"),
            ("…", "…"), ("model-00016", "thread 16")]
    W, top, rh, gap = 620, 76, 40, 16
    H = top + len(rows) * (rh + gap) + 6
    out = [f'<svg class="pfd" viewBox="0 0 {W} {H}" role="img">',
           '<defs><marker id="pfa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           '<path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker></defs>',
           '<text x="80" y="22" class="hd">Network volume</text><text x="80" y="40" class="sub">16 shard files</text>',
           '<text x="300" y="22" class="hd">16 threads</text><text x="300" y="40" class="sub">one file each, in parallel</text>',
           f'<text x="530" y="22" class="hd">Page cache (RAM)</text>',
           f'<rect x="450" y="{top}" width="162" height="{H - top - 6}" rx="10" class="ram"/>']
    for i, (f, t) in enumerate(rows):
        y = top + i * (rh + gap)
        dots = f == "…"
        if dots:
            out.append(f'<text x="80" y="{y + rh / 2 + 5}" class="dots">⋮</text><text x="300" y="{y + rh / 2 + 5}" class="dots">⋮</text>'
                       f'<text x="531" y="{y + rh / 2 + 5}" class="dots">⋮</text>')
            continue
        out.append(f'<rect x="6" y="{y}" width="148" height="{rh}" rx="7" class="file"/><text x="80" y="{y + rh / 2 + 5}" class="ft">{f}</text>'
                   f'<rect x="240" y="{y}" width="120" height="{rh}" rx="20" class="thr"/><text x="300" y="{y + rh / 2 + 5}" class="tt">{t}</text>'
                   f'<rect x="462" y="{y + 4}" width="138" height="{rh - 8}" rx="5" class="pg"/><text x="531" y="{y + rh / 2 + 5}" class="pt">{f}</text>'
                   f'<line x1="158" y1="{y + rh / 2}" x2="234" y2="{y + rh / 2}" class="ar" marker-end="url(#pfa)"/>'
                   f'<line x1="364" y1="{y + rh / 2}" x2="456" y2="{y + rh / 2}" class="ar" marker-end="url(#pfa)"/>')
    out.append(f'<text x="197" y="{top - 8}" class="lb">read</text><text x="410" y="{top - 8}" class="lb">16 MB chunks</text></svg>')
    return "".join(out)


def component(title: str, what: str, how: list[str], side: str) -> str:
    """Architecture detail slide: what the part does, how it works, one key fact on the side."""
    li = "".join(f"<li>{x}</li>" for x in how)
    return f"""<section data-label="Architecture">
  <h2>{title}</h2>
  <p class="what">{what}</p>
  <div class="pc">
    <div><h4>How it works</h4><ul class="dots">{li}</ul></div>
    <div>{side}</div>
  </div>
</section>"""


SLIDES = [
    # ---------------------------------------------------------------- title
    """<section class="cover">
  <h1>Fast Engine Startup</h1>
  <p class="author">Zhenbang Liu</p>
</section>""",

    # ---------------------------------------------------------------- 1. metrics
    section("1 · Requirements"),
    """<section>
  <h2>Requirements</h2>
  <div class="panels">
    <div class="panel blue"><h4>Requirement 1</h4><p class="big"><b>Preserve GPU capacity</b></p></div>
    <div class="panel green"><h4>Requirement 2</h4><p class="big"><b>Faster new-engine startup</b></p></div>
  </div>
  <div class="panel red"><h4>Constraints</h4><p>No loss in throughput or latency</p></div>
</section>""",

    section("2 · Metrics"),
    """<section>
  <h2>Metrics</h2>
  <p class="statement">Time from request to first reply, and the GPU&#8209;seconds billed before it.</p>
</section>""",
    """<section>
  <h2>Baseline</h2>
  <table class="plain">
    <tr><th>Workload</th><th>Weights</th><th>Startup</th><th>GPU cost per start</th></tr>
    <tr><td>Qwen3-30B · 2×H100</td><td>61 GB</td><td>229.8 s</td><td>$0.50</td></tr>
    <tr><td>Qwen3-30B · 2 engines on 4×H100</td><td>61 GB</td><td>223.9 s</td><td>$0.98</td></tr>
    <tr><td>Qwen3-235B · 8×B200</td><td>470 GB</td><td>951.5 s</td><td>$13.22</td></tr>
  </table>
  <p class="small">Cold start: a fresh sandbox per trial, stock SGLang v0.5.20, no caches. Weights pre-downloaded to a Modal volume.</p>
</section>""",

    # ---------------------------------------------------------------- 2. critical path
    section("3 · Find the critical path"),
    f"""<section>
  <h2>Where the time goes</h2>
  {bars("Qwen3-30B · 2×H100", only_first=True)}
  {bars("Qwen3-235B · 8×B200", only_first=True)}
  {LEGEND}
  <div class="panels">
    <div class="panel blue"><h4>GPUs mostly wait</h4><p>On imports, file reads and kernel compilation.</p></div>
    <div class="panel green"><h4>Each phase needs its own fix</h4><p>Cache, prefetch, fewer graphs, less Python.</p></div>
  </div>
  <p class="guard"><b>First cold start of a model is worse:</b> GPUs are claimed first, so they also sit idle through the weight download: 30B&nbsp;2–3&nbsp;min on 2×H100, 235B&nbsp;~17&nbsp;min on 8×B200.</p>
</section>""",
    """<section>
  <h2>Phases and fixes</h2>
  <table class="plain">
    <tr><th>Phase</th><th>30B</th><th>Bottleneck</th><th>Fix</th></tr>
    <tr><td>Imports + spawn</td><td>58 s</td><td>Python imports, repeated per process</td><td>Bytecode, lazy imports</td></tr>
    <tr><td>NCCL init</td><td>19 s</td><td>Kernel compilation</td><td>Compile cache</td></tr>
    <tr><td>Weight load</td><td>62 s</td><td>Reading from the volume</td><td>Prefetch into RAM</td></tr>
    <tr><td>CUDA graphs</td><td>72 s</td><td>Compilation + 94 graphs</td><td>Compile cache + pow2</td></tr>
    <tr><td>Other</td><td>19 s</td><td>Warmup kernels</td><td>Compile cache</td></tr>
  </table>
  <p class="small">For 235B, weight load alone is 564 of 951 s.</p>
</section>""",

    # ---------------------------------------------------------------- 3. solutions per component
    section("4 · A solution for each component"),
    proscons("Weights: cold start",
             "A model's first start downloads its weights, on a CPU sandbox, before any GPU is claimed.",
             ["No GPU billed during the download",
              "Saved to the model's own volume: later starts skip the download"],
             ["Storage cost: every model's weights stay on a volume (235B: 470 GB)",
              "Volumes must be managed: remove unused models, keep the catalog in sync"],
             '<table class="plain nowrap"><tr><th>Model</th><th>Download</th><th>GPU cost if claimed first</th></tr>'
             '<tr><td>30B · 61 GB</td><td>135–195&nbsp;s</td><td>$0.30–0.43</td></tr>'
             '<tr><td>235B · 470 GB</td><td>1,004 s</td><td>~$14</td></tr></table>'
             '<p class="small">Now: $0 on GPUs. The cost is the download time × the GPU price.</p>', label="Weights"),
    proscons("Weights: prefetch into RAM",
             "16 threads read all shards into RAM while SGLang is still importing.",
             ["Faster SGLang init: weight load 62&nbsp;→&nbsp;13&nbsp;s (30B), 564&nbsp;→&nbsp;75&nbsp;s (235B)"],
             ["Needs a larger-memory host: about 2× the weights (235B: 940 GiB)"],
             prefetch_diagram()
             + '<p class="small">Before: SGLang reads the shards on demand, ~1&nbsp;GB/s. Now: ~2&nbsp;GB/s median (30B), 4.2&nbsp;GB/s (235B). '
             'Models with more shards: a thread takes the next file when it finishes.</p>', label="Weights"),
    f"""<section data-label="Weights">
  <h2>Why prefetch helps</h2>
  {timeline([
      ("Before", [(0, 77, "Imports + NCCL init · 77 s", "t-init"), (77, 139, "Read weights from volume · 62 s", "t-load")]),
      ("Now", [(0, 49, "Imports + NCCL init · 49 s", "t-init"), (49, 62, "13 s", "t-load")]),
      ("", [(0, 30, "Prefetch: 16 threads · ~30 s", "t-pre")]),
  ], scale=140, ticks=[0, 30, 60, 90, 120])}
  <div class="panels">
    <div class="panel red"><h4>Why it is slow</h4><p>SGLang starts reading weights only after imports and NCCL init, then reads them on demand from the network volume: ~1&nbsp;GB/s.</p></div>
    <div class="panel green"><h4>How prefetch fixes it</h4><p>At container start, 16 threads read whole shard files into the OS page cache, during the imports. SGLang's reads then hit RAM; only the copy to the GPU is left.</p></div>
  </div>
  <p class="small">Qwen3-30B, 2×H100: baseline vs the current engine (median). "Now" also includes the other startup fixes.</p>
</section>""",
    """<section data-label="Background">
  <h2>Memory layout: user space and kernel space</h2>
  <div class="osm">
    <div class="vas">
      <div class="vas-title">Virtual address space of one process (Linux, x86-64)</div>
      <div class="vas-col">
        <div class="vseg k"><b>Kernel space</b><span>kernel code + data · direct map of all RAM (incl. page cache)</span><em>0xffff_8000_0000_0000</em></div>
        <div class="vseg hole">non-canonical gap</div>
        <div class="vseg st"><b>Stack</b> ↓<em>0x0000_7fff_ffff_ffff</em></div>
        <div class="vseg gap"></div>
        <div class="vseg mm"><b>mmap region</b><span>shared libraries · <code>mmap</code> of model-00001.safetensors</span></div>
        <div class="vseg gap"></div>
        <div class="vseg hp"><b>Heap</b> ↑ <span>Python objects, tensors</span></div>
        <div class="vseg tx"><b>Code + data</b><span>python, libtorch</span><em>0x0000_0000_0040_0000</em></div>
      </div>
      <div class="vas-legend"><span class="lk">kernel space: shared by every process, kernel mode only</span><span class="lu">user space: private to this process</span></div>
    </div>
    <div class="osr">
      <div class="pc-box pt"><h5>Page table (per process) + MMU</h5><p>Translates each 4 KB virtual page to a physical frame. A missing page traps into the kernel: a <b>page fault</b>.</p></div>
      <div class="osr-arr">↓</div>
      <div class="pc-box ram"><h5>Physical RAM (4 KB frames)</h5>
        <div class="frames"><i class="fk">kernel</i><i class="fa">process heap / stack</i><i class="fc">page cache: file pages</i><i class="ff">free</i></div></div>
      <ul class="dots">
        <li><b>User space:</b> the program's own memory; it can't touch the kernel's</li>
        <li><b>Kernel space:</b> entered through system calls and page faults</li>
        <li><b><code>mmap</code> of a file:</b> the user-space range points straight at page-cache frames, with no copy</li>
      </ul>
    </div>
  </div>
</section>""",
    f"""<section data-label="Background">
  <h2>mmap read, page not cached: major fault</h2>
  {sequence([("SGLang", "user space", "l-user"), ("MMU", "page table", "l-hw"),
             ("Kernel", "page cache", "l-kern"), ("Network volume", "shard files", "l-vol")], [
      ("msg", 0, 2, 84, "① mmap(): record the file range, read nothing", False),
      ("msg", 0, 1, 116, "② read 0x7f00_1000", False),
      ("msg", 1, 2, 148, "③ entry empty: page fault", False),
      ("note", 2, 182, "④ cache miss: allocate a frame"),
      ("band", 2, 3, 200, 266),
      ("msg", 2, 3, 222, "⑤ read the page (+ read-ahead)", False, True),
      ("msg", 3, 2, 254, "data over the network, ~1 GB/s", True, True),
      ("note", 2, 286, "⑥ add to page cache, fill the entry"),
      ("msg", 2, 0, 318, "wake SGLang", True),
      ("msg", 0, 1, 350, "⑦ retry the read: hits RAM", False),
  ], height=362, blocked=(0, 124, 324, "blocked"))}
  <p class="seq-foot"><b>Major fault:</b> every miss blocks SGLang for a network read; the 30B weights take 62&nbsp;s this way.</p>
</section>""",
    f"""<section data-label="Background">
  <h2>mmap read, page cached: minor fault</h2>
  {sequence([("SGLang", "user space", "l-user"), ("MMU", "page table", "l-hw"),
             ("Kernel", "page cache", "l-kern"), ("Network volume", "shard files", "l-vol")], [
      ("msg", 0, 2, 84, "① mmap(): record the file range, read nothing", False),
      ("msg", 0, 1, 116, "② read 0x7f00_1000", False),
      ("msg", 1, 2, 148, "③ entry empty: page fault", False),
      ("note", 2, 182, "④ cache hit: prefetch loaded it"),
      ("note", 2, 216, "⑤ fill the entry (+ nearby pages)"),
      ("msg", 2, 0, 250, "return to SGLang, no I/O", True),
      ("msg", 0, 1, 284, "⑥ retry the read: hits RAM", False),
      ("msg", 0, 1, 318, "⑦ next reads: plain memory loads", False),
  ], height=332, blocked=(0, 124, 256, "~µs"))}
  <p class="seq-foot"><b>Minor fault:</b> no I/O and the volume is never touched. The 30B weights load in 13&nbsp;s, mostly the copy to the GPU. With prefetch, every fault is a minor one.</p>
</section>""",
    rejected("ModelExpress GPU-to-GPU transfer", ["Load weights from a running engine"],
             ["Transfer backend does not start on Modal", "Streaming loader: smaller gain than prefetch"],
             "", label="Weights"),
    proscons("CUDA graphs: power-of-two sizes",
             "Capture only power-of-two batch sizes. Other batches pad up to the next size.",
             ["Faster SGLang init: graph capture 27&nbsp;→&nbsp;9.6&nbsp;s", "Less graph memory: 1.69 → 1.18 GB"],
             ["Lower throughput: ~2%", "Higher latency: +0.05&nbsp;ms per token",
              "Wastes GPU resources on padding: up to 2× (129 requests run as 256)"],
             '<table class="plain"><tr><th></th><th>Original</th><th>pow2</th></tr>'
             '<tr><td>Decode</td><td>36</td><td>9</td></tr><tr><td>Prefill</td><td>58</td><td>12</td></tr>'
             '<tr><td>Total</td><td>94</td><td>21</td></tr></table>', label="CUDA graphs"),
    """<section data-label="CUDA graphs">
  <h2>Why power-of-two graphs work here</h2>
  <div class="bs">
    <div class="bs-label">Original<small>36 decode sizes</small></div><div class="bs-track"><i style="left:0.00%"></i><i style="left:12.50%"></i><i style="left:25.00%"></i><i style="left:37.50%"></i><i style="left:44.81%"></i><i style="left:50.00%"></i><i style="left:57.31%"></i><i style="left:62.50%"></i><i style="left:66.52%"></i><i style="left:69.81%"></i><i style="left:72.59%"></i><i style="left:75.00%"></i><i style="left:77.12%"></i><i style="left:79.02%"></i><i style="left:80.74%"></i><i style="left:82.31%"></i><i style="left:83.76%"></i><i style="left:85.09%"></i><i style="left:86.34%"></i><i style="left:87.50%"></i><i style="left:88.59%"></i><i style="left:89.62%"></i><i style="left:90.60%"></i><i style="left:91.52%"></i><i style="left:92.40%"></i><i style="left:93.24%"></i><i style="left:94.04%"></i><i style="left:94.81%"></i><i style="left:95.55%"></i><i style="left:96.26%"></i><i style="left:96.94%"></i><i style="left:97.59%"></i><i style="left:98.22%"></i><i style="left:98.84%"></i><i style="left:99.43%"></i><i style="left:100.00%"></i></div>
    <div class="bs-label">pow2<small>9 decode sizes</small></div><div class="bs-track pow"><i class="big" style="left:0.00%"><b>1</b></i><i class="big" style="left:12.50%"><b>2</b></i><i class="big" style="left:25.00%"><b>4</b></i><i class="big" style="left:37.50%"><b>8</b></i><i class="big" style="left:50.00%"><b>16</b></i><i class="big" style="left:62.50%"><b>32</b></i><i class="big" style="left:75.00%"><b>64</b></i><i class="big" style="left:87.50%"><b>128</b></i><i class="big" style="left:100.00%"><b>256</b></i>
      <span class="bs-ex" style="left:83.05%">batch of 100 → runs in the 128 graph</span></div>
    <div></div><div class="bs-axis">decode batch size (log scale)</div>
  </div>
  <h4 style="margin-top:.4em">Padding is cheap here: how we measured</h4>
  <table class="plain compact nowrap"><tr><th>Metric</th><th>Load on the fresh engine</th><th>Full</th><th>pow2</th></tr>
    <tr><td>Throughput</td><td>128 requests, 32 at a time</td><td>4,645 tok/s</td><td>4,547 tok/s (−2.1%)</td></tr>
    <tr><td>Latency per token</td><td>16 requests, one at a time</td><td>3.45 ms</td><td>3.50 ms (+0.05)</td></tr></table>
  <p class="small">800-word prompt, 256 output tokens. Latency = time between output tokens (median). Mean of 2 cold-start trials each.</p>
</section>""",
    rejected("Disable CUDA graphs", ["~27 s faster startup"],
             ["10–19× lower throughput", "Latency 3.5 → 64.5 ms per token"],
             "Serving cost is far higher than the startup gain.", label="CUDA graphs"),
    proscons("Python imports",
             "Imports are single-threaded and repeated by every worker process.",
             ["Precompiled bytecode: ~7 s", "Import file prefetch"],
             ["Still ~48 s: the largest remaining phase"], label="Python imports"),
    """<section data-label="Python imports">
  <h2>How precompiled bytecode shortens init</h2>
  <div class="bcw">
    <div class="flows">
      <div class="flow"><div class="flow-tag red">Before</div>
        <div class="fbox">import a module</div><div class="farr">→</div>
        <div class="fbox slow">no .pyc: read, parse and compile the source<small>CPU, single-threaded</small></div><div class="farr">→</div>
        <div class="fbox">run it</div></div>
      <div class="flow"><div class="flow-tag green">Now</div>
        <div class="fbox">import a module</div><div class="farr">→</div>
        <div class="fbox ram">.pyc already in the image: load the bytecode</div><div class="farr">→</div>
        <div class="fbox">run it</div></div>
      <p class="small" style="margin:.4em 0 0 5.4em">SGLang imports ~3,900 modules. A fresh container has no .pyc files, so every cold start compiled all of them.</p>
    </div>
    <div class="layers"><h4>Engine image layers</h4>
      <div class="ly">SGLang base image</div>
      <div class="ly">lazy-import patch</div>
      <div class="ly hot">compileall: every .pyc<small>built once, cached by Modal</small></div>
      <div class="ly">import manifest</div>
      <div class="ly">our engine files</div>
    </div>
  </div>
  <pre class="code"><span class="c"># fes/common.py · with_bytecode(): runs once, at image build</span>
python -m compileall -q -j 0 /opt/sglang/lib/python3.12/site-packages /sgl-workspace/sglang/python</pre>
</section>""",
    """<section data-label="Python imports">
  <h2>How import-file prefetch shortens init</h2>
  <p class="what">Modal fetches image files on first read; SGLang's imports read ~14k of them one by one.</p>
  <div class="panels">
    <div class="panel blue"><h4>At image build</h4><p>Record every file an engine's imports read, and save the list in the image.</p></div>
    <div class="panel green"><h4>At container start</h4><p>Load those files into memory (the OS page cache) in parallel, while SGLang imports. Its imports then read them from RAM.</p></div>
  </div>
  <p class="guard"><b>Result:</b> ~2–4&nbsp;s faster on real starts, within host-to-host noise.</p>
</section>""",
    f"""<section data-label="Python imports">
  <h2>Why spawn is slower than fork</h2>
  {timeline([
      ("spawn", [(0, 15, "Launcher: import torch + SGLang · ~15 s", "t-init"), (15, 30, "Worker: new Python, import again · ~15 s", "t-bad"), (30, 33, "CUDA", "t-load")]),
      ("fork", [(0, 15, "Launcher: import torch + SGLang · ~15 s", "t-init"), (15, 18, "CUDA", "t-load")]),
  ], scale=35, ticks=[0, 10, 20, 30])}
  <div class="panels">
    <div class="panel red"><h4>spawn: a fresh process</h4><p>Starts a new Python interpreter, which imports torch and SGLang again. Every TP worker pays it.</p></div>
    <div class="panel green"><h4>fork: a copy of the parent</h4><p>Modules are already in memory. Pages are shared copy-on-write, so the worker starts in milliseconds.</p></div>
  </div>
  <p class="small" style="margin-top:-.3em">One TP worker. ~15 s is the measured warm import per process; the CUDA block is only illustrative.</p>
  <p class="guard"><b>Why SGLang uses spawn:</b> CUDA can't be used in a child forked after CUDA started.</p>
</section>""",
    rejected("Fork instead of spawn", ["Faster process startup"],
             ["Unsafe once CUDA is initialized"],
             "A forkserver started before CUDA is future work.", label="Python imports"),

    proscons("Compiled kernels: compile cache",
             "Save compiled kernels after the first start. Restore them on later starts.",
             ["~60 s saved on 30B", "Shared across starts with the same SGLang, GPU, TP and model"],
             ["First start of a new configuration still compiles",
              "A dedicated volume per model (fes-artifacts-&lt;model&gt;) to store and manage"],
             '<table class="plain"><tr><th></th><th>Cold</th><th>Cached</th></tr>'
             '<tr><td>NCCL init</td><td>19 s</td><td>1 s</td></tr><tr><td>CUDA graphs</td><td>73 s</td><td>27 s</td></tr></table>', label="Compiled kernels"),
    f"""<section data-label="Compiled kernels">
  <h2>How the compile cache helps</h2>
  {timeline([
      ("Cold", [(0, 19, "NCCL init · 19 s", "t-bad"), (19, 92, "CUDA graphs: compile kernels + record · 73 s", "t-bad")]),
      ("Cached", [(0, 28, "NCCL 1 s · CUDA graphs: record 27 s", "t-load")]),
      ("", [(0, 1.2, "", "t-pre"), (1.6, 40, "restore the tar in the background · ~1 s", "t-note")]),
  ], scale=95, ticks=[0, 20, 40, 60, 80])}
  <div class="panels">
    <div class="panel red"><h4>First start: compile, then save</h4><p>SGLang compiles its GPU kernels (Triton, FlashInfer, all-reduce). After READY, they are packed into one ~29&nbsp;MB tar on <code>fes-artifacts-&lt;model&gt;</code>.</p></div>
    <div class="panel green"><h4>Later starts: restore, then reuse</h4><p>At boot the tar is unpacked beside the launch. SGLang finds every kernel already compiled and skips the compile.</p></div>
  </div>
  <p class="small">Qwen3-30B, full CUDA graphs. Also saves ~9 s of warmup. Reused by any start with the same SGLang version, GPU, TP and model config.</p>
</section>""",
    # ---------------------------------------------------------------- 5. make it work, then make it reliable
    section("5 · Make it work, then make it reliable"),
    """<section class="arch-slide" data-label="Architecture">
  <h2>Architecture</h2>
  <div class="lane local"><h4>Local control plane</h4>
    <div class="arch">
      <div class="box"><b>Web UI</b><span>pages + JSON API</span></div><div class="arr">→</div>
      <div class="box"><b>API</b><span>writes jobs, streams progress</span></div><div class="arr">→</div>
      <div class="box"><b>Job queue</b><span>Postgres table (Docker)</span></div><div class="arr">→</div>
      <div class="box key"><b>Worker</b><span>claims a job, plans and runs it</span></div><div class="arr">⇄</div>
      <div class="box"><b>Catalog</b><span>what is cached</span></div>
    </div>
  </div>
  <div class="arch link"><div class="down">↓ starts sandboxes</div></div>
  <div class="lane modal"><h4>Modal</h4>
    <div class="arch">
      <div class="box key wide"><b>GPU engine</b><span>at once: prefetch weights into RAM · restore compiled kernels · start SGLang</span></div>
      <div class="arr">←<small>mount</small></div>
      <div class="box"><b>Volumes</b><span>weights + compile cache, one set per model</span></div><div class="arr">←</div>
      <div class="box"><b>CPU prep</b><span>only on a weights cache miss: downloads them, no GPU billed</span></div>
    </div>
  </div>
</section>""",
    component("API + job queue",
              "The front door. It writes jobs; it never runs them.",
              ["Flask app: Engines, Jobs, Playground, Catalog, Compare pages + JSON API",
               "<code>POST /api/jobs</code> inserts a row with status <b>queued</b>",
               "Progress streams to the page from the job's event log (SSE)",
               "Cancel sets a flag the worker checks every second",
               "Restarting the API never interrupts a running job"],
              '<div class="panel blue"><h4>Queue = a Postgres table</h4><p>Workers claim the oldest queued row with '
              '<code>FOR UPDATE SKIP LOCKED</code>, so two workers never get the same job.</p></div>'
              '<div class="panel green"><h4>Job states</h4><p>queued → running → serving → stopped · timeout · failed</p></div>'),

    # ---------------------------------------------------------------- results
    f"""<section class="results" data-label="Results">
  <h2>Results: Qwen3-30B · 2×H100</h2>
  {phase_savings("Qwen3-30B · 2×H100", 229.8, 84.5)}
  {SIDES_LEGEND}
  <p class="small">GPU cost per start $0.50 → $0.18. Medians: baseline n=3, now n=12.</p>
</section>""",
    f"""<section class="results" data-label="Results">
  <h2>Results: Qwen3-235B · 8×B200</h2>
  {phase_savings("Qwen3-235B · 8×B200", 951.5, 220.8)}
  {SIDES_LEGEND}
  <p class="small">GPU cost per start $13.22 → $3.03. n=1 on each side.</p>
</section>""",
    f"""<section class="results" data-label="Results">
  <h2>Results: Qwen3-30B · 2 engines on 4×H100</h2>
  {phase_savings("Qwen3-30B · 4×H100 · 2 engines", 223.9, 155.5)}
  {SIDES_LEGEND}
  <p class="small">GPU cost per start $0.98 → $0.53. Baseline n=1, now n=2. Startup also includes ~35 s outside the phases: slow container starts on this 4×H100 host. Both engines are ready within 0.2 s of each other.</p>
</section>""",

    # ---------------------------------------------------------------- future work
    """<section data-label="Outlook">
  <h2>Future Work</h2>
  <div class="panels three">
    <div class="panel blue"><h4>Engine</h4><ol>
      <li>Validate pow2 graphs on production traffic</li>
      <li>Fork TP workers from a pre-imported process (forkserver)</li>
      <li>Pre-sharded checkpoints per GPU</li></ol></div>
    <div class="panel green"><h4>Scheduler</h4><ol>
      <li>Warm volume pool</li>
      <li>Quotas and rate limits</li>
      <li>Multi-cloud GPU scheduling</li></ol></div>
    <div class="panel amber"><h4>Engineering</h4><ol>
      <li>CI/CD: build and deploy the images on every merge</li>
      <li>Unit tests: plan, catalog, queue</li>
      <li>E2E tests: a real start on the smallest GPU, nightly</li>
      <li>Monitoring and alerting: startup time, failures, queue wait</li>
      <li>API authentication</li></ol></div>
  </div>
</section>""",
]

HTML = """<!doctype html>
<html lang="en" data-theme="__THEME__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fast Engine Startup · slides</title>
<style>

:root { --ink: #0f172a; --ink2: #334155; --muted: #64748b; --faint: #94a3b8; --bg: #ffffff; --bg2: #f8fafc;
  --line: #e2e8f0; --accent: #2563eb; --accent-soft: #eff4ff; --good: #059669; --good-soft: #ecfdf5;
  --bad: #dc2626; --bad-soft: #fef2f2; --dark: #0b1220; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: #05080f; color: var(--ink);
  font-family: Inter, -apple-system, "SF Pro Text", "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased; font-feature-settings: "tnum" 1; }
.deck { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; }
section { display: none; width: min(100vw, calc(100vh * 16 / 9)); height: min(100vh, calc(100vw * 9 / 16));
  background: var(--bg); padding: 5.2% 6% 4%; overflow: hidden; font-size: min(1.6vw, 2.85vh); line-height: 1.45; position: relative; }
section.active { display: block; }
h1 { font-size: 3.2em; margin: 0; letter-spacing: -.025em; line-height: 1.05; font-weight: 700; }
h2 { font-size: 2em; margin: 0 0 .8em; letter-spacing: -.02em; font-weight: 700; line-height: 1.15; }
h4 { margin: 0 0 .4em; font-size: .78em; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
p, li, td { color: var(--ink2); } b { color: var(--ink); font-weight: 650; }
small, .small { color: var(--muted); font-size: .82em; }
code { font-family: "SF Mono", Menlo, monospace; font-size: .85em; background: var(--bg2); padding: .05em .3em; border-radius: 4px; }
.kicker { font-size: .72em; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin-bottom: .7em; }
/* cover + section dividers: dark */
.cover, .divider { background: radial-gradient(120% 90% at 85% 10%, #1d3a8a 0%, rgba(11,18,32,0) 55%), var(--dark); color: #fff; }
.cover.active, .divider.active { display: flex; flex-direction: column; justify-content: center; }
.cover h1, .divider h1 { color: #fff; }
.cover h1 { font-size: 4em; }
.cover .sub { font-size: 1.4em; color: #cbd5e1; margin: .5em 0 .3em; }
.cover .author { font-size: 1.4em; color: #cbd5e1; margin: 1em 0 0; }
.divider .num { font-size: 1.1em; font-weight: 700; letter-spacing: .2em; color: #60a5fa; margin-bottom: .6em; }
.divider h1 { font-size: 3.4em; } .divider p { color: #cbd5e1; font-size: 1.2em; margin-top: .6em; }
.hero { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2em; width: 100%; }
.hero > div { border-radius: 14px; padding: 1.1em 1.3em; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); }
.hero b { display: block; font-size: 2.8em; line-height: 1.05; font-weight: 700; letter-spacing: -.03em; margin-bottom: .25em;
  background: linear-gradient(90deg, #93c5fd, #60a5fa); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero .green b { background-image: linear-gradient(90deg, #6ee7b7, #34d399); }
.hero span { color: #cbd5e1; font-size: .92em; }
/* cards */
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2em; margin: 1em 0; }
.panel { background: var(--bg); border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 12px;
  padding: .9em 1.1em; box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.04); }
.panel.green { border-left-color: var(--good); } .panel.amber { border-left-color: #d97706; } .panel.amber h4 { color: #d97706; }
.panels.three { grid-template-columns: repeat(3, 1fr); } .panel.red { border-left-color: var(--bad); }
.panel.blue h4 { color: var(--accent); } .panel.green h4 { color: var(--good); } .panel.red h4 { color: var(--bad); }
.panel p { margin: .2em 0; } .panel ol { margin: .3em 0 0; padding-left: 1.2em; } .panel li { margin: .35em 0; }
.big { font-size: 1.35em; margin: .1em 0 .3em !important; }
.statement { font-size: 1.9em; font-weight: 400; line-height: 1.3; color: var(--ink); max-width: 22em; margin-top: 1.2em; letter-spacing: -.01em; }
.guard { background: var(--bg2); border: 1px solid var(--line); border-radius: 12px; padding: .7em 1em; margin-top: 1.2em; }
/* pros / cons */
.what { font-size: 1.1em; margin: -.3em 0 1.2em; max-width: 50em; color: var(--ink2); }
.pc { display: grid; grid-template-columns: 1.1fr 1fr; gap: 2.4em; align-items: start; }
.pc .panel { margin-bottom: .9em; }
h4.pro { color: var(--good); } h4.con { color: var(--bad); margin-top: 1em; }
.pmlist { list-style: none; padding: 0; margin: 0; } .pmlist li { margin: .45em 0; padding-left: 1.9em; position: relative; }
.pm { position: absolute; left: 0; top: .12em; width: 1.25em; height: 1.25em; border-radius: 50%; font-size: .9em; font-weight: 700;
  display: flex; align-items: center; justify-content: center; }
.pros .pm { background: var(--good-soft); color: var(--good); } .cons .pm { background: var(--bad-soft); color: var(--bad); }
/* rejected */
.badge { display: inline-block; vertical-align: .35em; margin-left: .5em; background: var(--bad-soft); color: var(--bad);
  font-size: .36em; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; padding: .35em .9em; border-radius: 999px; }
.rejected .pmlist { font-size: 1.1em; }
.why { margin-top: 1.4em; font-size: 1.05em; color: var(--ink2); }
.why b { font-size: .7em; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-right: 1em; }
/* tables */
table.plain { border-collapse: collapse; width: 100%; font-size: .92em; }
table.plain th, table.plain td { border-bottom: 1px solid var(--line); padding: .6em .8em; text-align: left; vertical-align: top; }
table.plain th { font-size: .74em; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
  border-bottom: 2px solid var(--ink); padding-bottom: .5em; }
table.plain td:first-child { color: var(--ink); font-weight: 600; }
table.plain tr:last-child td { border-bottom: none; }
table.plain.nowrap td { white-space: nowrap; }
/* evolution steps */
.steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1em; margin: .6em 0 1.2em; }
.step { border: 1px solid var(--line); border-radius: 12px; padding: 1em 1.1em; background: var(--bg);
  box-shadow: 0 4px 16px rgba(15,23,42,.04); }
.step:last-child { border-color: var(--accent); background: var(--accent-soft); }
.step .n { font-size: .75em; font-weight: 700; letter-spacing: .12em; color: var(--accent); margin-bottom: .6em; }
.step .n::before { content: "STEP "; }
.step h4 { font-size: 1.05em; text-transform: none; letter-spacing: -.01em; color: var(--ink); }
.step p { font-size: .88em; margin: .3em 0; }
.step .lim { color: var(--muted); font-size: .8em; border-top: 1px solid var(--line); padding-top: .5em; margin-top: .8em; }
/* charts */
.chart { margin: 0 0 1em; } .chart-title { font-weight: 650; margin-bottom: .35em; color: var(--ink); }
.bar-row { display: grid; grid-template-columns: 8.5em 1fr 5.5em; align-items: center; gap: .8em; margin: .3em 0; }
.bar-label { color: var(--muted); font-size: .88em; text-align: right; }
.bar { display: flex; height: 1.7em; gap: 2px; }
.seg { color: #fff; font-size: .76em; font-weight: 700; display: flex; align-items: center; justify-content: center; min-width: 2px; }
.seg:first-child { border-radius: 4px 0 0 4px; } .seg:last-child { border-radius: 0 4px 4px 0; }
.bar-total { font-weight: 700; color: var(--ink); }
.legend { display: flex; flex-wrap: wrap; gap: .4em 1.3em; font-size: .78em; color: var(--muted); margin: .1em 0 .4em 9.3em; }
.legend i { display: inline-block; width: .75em; height: .75em; border-radius: 3px; margin-right: .45em; vertical-align: -.05em; }
.results h2 { margin-bottom: .4em; } .results .chart { margin-bottom: .3em; } .results .chart-title { font-size: .9em; margin-bottom: .1em; }
.results .bar { height: 1.25em; } .results .bar-row { margin: .12em 0; } .results .seg { font-size: .68em; }
.results .chart-title { display: none; }
svg.savings { width: 100%; height: auto; display: block; margin: -.2em 0 .2em; font-family: inherit; }
svg.savings .grid { stroke: var(--line); } svg.savings .tick { font-size: 11px; fill: var(--faint); text-anchor: middle; }
svg.savings .lab { font-size: 13.5px; fill: var(--ink); text-anchor: end; font-weight: 600; } svg.savings .lab.tot { font-weight: 800; font-size: 14.5px; }
svg.savings .fix { font-size: 11px; fill: var(--muted); text-anchor: end; }
svg.savings .base { fill: #94a3b8; } svg.savings .now { fill: #2a78d6; }
svg.savings .val { font-size: 11.5px; fill: var(--ink2); }
svg.savings .delta { font-size: 14px; font-weight: 700; text-anchor: end; } svg.savings .delta.good { fill: var(--good); } svg.savings .delta.bad { fill: var(--bad); } svg.savings .delta.flat { fill: var(--muted); }
.legend.sides { margin-left: 0; } .legend.sides i.lb { background: #94a3b8; } .legend.sides i.ln { background: #2a78d6; }
.stage td.good { color: var(--good); font-weight: 600; } .stage td.bad { color: var(--bad); font-weight: 600; }
.stage tr.tot td { border-top: 2px solid var(--ink); font-weight: 700; }
.results table.compact { font-size: .78em; margin-top: .3em; } .results table.compact td, .results table.compact th { padding: .3em .6em; }
/* code block */
.plain-seq li { font-size: .95em; margin-bottom: .8em; }
pre.code { background: #0f172a; color: #e2e8f0; border-radius: 10px; padding: .8em 1em; font-size: .72em; line-height: 1.5;
  font-family: "SF Mono", Menlo, monospace; margin: .3em 0 .5em; overflow: hidden; white-space: pre; }
pre.code .c { color: #94a3b8; } pre.code .k { color: #93c5fd; }
html[data-theme="dark"] pre.code { background: #020617; }
/* batch-size ticks */
.bs { display: grid; grid-template-columns: 7em 1fr; gap: .5em 1em; align-items: center; margin: 1.4em 0 1.2em; }
.bs-label { text-align: right; font-weight: 600; font-size: .9em; line-height: 1.2; }
.bs-label small { display: block; font-weight: 400; color: var(--muted); font-size: .82em; }
.bs-track { position: relative; height: 2.4em; border-bottom: 1px solid var(--line); margin-right: 1.2em; }
.bs-track i { position: absolute; bottom: 0; width: 2px; height: 1.2em; background: #94a3b8; transform: translateX(-1px); }
.bs-track i.big { width: 4px; height: 1.8em; background: var(--accent); transform: translateX(-2px); border-radius: 2px; }
.bs-track i.big b { position: absolute; top: calc(100% + .25em); left: 50%; transform: translateX(-50%); font-size: .7em; color: var(--ink); }
.bs-ex { position: absolute; top: calc(100% + 1.55em); transform: translateX(-50%); font-size: .72em; color: var(--good); font-weight: 600; white-space: nowrap; }
.bs-ex::before { content: "▲ "; }
.bs-axis { font-size: .72em; color: var(--faint); text-align: right; margin-right: 1.2em; padding-top: 2.6em; }
/* prefetch diagram */
svg.pfd { width: 100%; height: auto; display: block; font-family: inherit; }
svg.pfd .hd { font-size: 15px; font-weight: 700; fill: #0f172a; text-anchor: middle; } svg.pfd .sub { font-size: 12px; fill: #64748b; text-anchor: middle; }
svg.pfd .file { fill: #fde2e2; stroke: #f5b4b4; } svg.pfd .ft { font-size: 13px; fill: #0f172a; text-anchor: middle; font-family: "SF Mono", Menlo, monospace; }
svg.pfd .thr { fill: #e8f0fd; stroke: #93b4ea; } svg.pfd .tt { font-size: 13px; fill: #1d4ed8; font-weight: 600; text-anchor: middle; }
svg.pfd .ram { fill: #ecfdf5; stroke: #9fdcc0; } svg.pfd .pg { fill: #a7e3c9; } svg.pfd .pt { font-size: 12px; fill: #065f46; text-anchor: middle; font-family: "SF Mono", Menlo, monospace; }
svg.pfd .ar { stroke: #64748b; stroke-width: 1.8; } svg.pfd .dots { font-size: 22px; fill: #94a3b8; text-anchor: middle; }
svg.pfd .lb { font-size: 11.5px; fill: #64748b; text-anchor: middle; }
/* sequence diagram */
svg.seqd { width: 100%; height: auto; display: block; margin: -.3em 0 .4em; font-family: inherit; }
svg.seqd .lane { stroke-width: 1; } .l-user { fill: #e8f0fd; stroke: #c9dcf8; } .l-hw { fill: #f1f5f9; stroke: #cbd5e1; }
.l-kern { fill: #fff1e6; stroke: #fcd9b6; } .l-vol { fill: #fde2e2; stroke: #f5b4b4; }
svg.seqd .lt { font-size: 17px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
svg.seqd .ls { font-size: 12px; fill: #64748b; text-anchor: middle; }
svg.seqd .life { stroke: #cbd5e1; stroke-width: 1.5; stroke-dasharray: 5 5; }
svg.seqd .msg { stroke: #334155; stroke-width: 1.8; } svg.seqd .msg.dash { stroke-dasharray: 6 5; stroke: #64748b; }
svg.seqd .ahead { fill: #334155; } svg.seqd .ahead.hl { fill: #dc2626; }
svg.seqd .msg.hl { stroke: #dc2626; stroke-width: 2.6; } svg.seqd .mt.hl { fill: #b91c1c; font-weight: 700; }
svg.seqd .band { fill: #fef2f2; stroke: #fca5a5; stroke-width: 1.2; stroke-dasharray: 4 4; }
svg.seqd .mt { font-size: 14px; fill: #0f172a; text-anchor: middle; }
svg.seqd .note { fill: #fff7ed; stroke: #fcd9b6; } svg.seqd .nt { font-size: 13.5px; fill: #9a3412; }
svg.seqd .blk { fill: #fca5a5; } svg.seqd .blkt { font-size: 12px; font-weight: 700; fill: #b91c1c; text-anchor: middle; letter-spacing: .08em; }
.seq-foot { font-size: .9em; color: var(--ink2); margin: 0; }
/* mmap read sequence */
.seqwrap { display: grid; grid-template-columns: 1.35fr 1fr; gap: 2em; align-items: start; }
ol.seq { list-style: none; counter-reset: seq; padding: 0; margin: 0; }
ol.seq li { counter-increment: seq; display: grid; grid-template-columns: 1.6em 4.6em 1fr; gap: .6em; align-items: start; margin: 0 0 .55em; font-size: .88em; }
ol.seq li::before { content: counter(seq); width: 1.6em; height: 1.6em; border-radius: 50%; background: var(--ink); color: #fff;
  font-size: .8em; font-weight: 700; display: flex; align-items: center; justify-content: center; margin-top: .1em; }
.where { font-size: .72em; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; border-radius: 999px; padding: .25em .6em; text-align: center; margin-top: .1em; }
.wu { background: #e8f0fd; color: #1d4ed8; } .wi { background: #fde2e2; color: #b91c1c; } .wk { background: #fff1e6; color: #c2410c; } .wh { background: #f1f5f9; color: #475569; }
/* OS memory layout */
.osm { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; font-size: .8em; }
.vas-title { font-weight: 600; font-size: .9em; color: var(--ink2); margin-bottom: .4em; }
.vas-col { display: flex; flex-direction: column; gap: 2px; margin-left: 11.5em; }
.vseg { position: relative; border-radius: 5px; padding: .35em .7em; font-size: .88em; color: var(--ink); }
.vseg b { margin-right: .3em; } .vseg span { color: var(--muted); font-size: .9em; display: block; }
.vseg em { position: absolute; right: calc(100% + .7em); top: .35em; font-style: normal; font-family: "SF Mono", Menlo, monospace;
  font-size: .78em; color: var(--muted); white-space: nowrap; }
.vseg.k { background: #fff1e6; border: 1px solid #fcd9b6; padding: .6em .7em; }
.vseg.hole { background: repeating-linear-gradient(45deg, #f1f5f9 0 6px, #fff 6px 12px); color: var(--faint); font-size: .75em; text-align: center; padding: .25em; }
.vseg.st, .vseg.hp, .vseg.tx { background: #eff4ff; border: 1px solid #c9dcf8; }
.vseg.mm { background: #fde2e2; border: 1px solid #f5b4b4; }
.vseg.gap { height: .9em; background: none; border: 1px dashed var(--line); }
.vas-legend { display: flex; gap: 1em; margin: .5em 0 0 11.5em; font-size: .78em; color: var(--muted); flex-wrap: wrap; }
.vas-legend span::before { content: ""; display: inline-block; width: .8em; height: .8em; border-radius: 2px; margin-right: .35em; vertical-align: -.1em; }
.lk::before { background: #fcd9b6; } .lu::before { background: #c9dcf8; }
.osr .pc-box p { margin: .1em 0; font-size: .92em; }
.osr-arr { color: var(--muted); padding: .1em .5em; }
.frames { display: flex; gap: 2px; margin-top: .3em; }
.frames i { font-style: normal; font-size: .8em; padding: .35em .5em; border-radius: 4px; }
.fk { background: #fcd9b6; flex: 1; } .fa { background: #c9dcf8; flex: 1.4; } .fc { background: #a7e3c9; flex: 2; } .ff { background: #e2e8f0; flex: .8; color: var(--muted); }
.osr ul.dots { margin-top: .8em; } .osr ul.dots li { margin: .4em 0; }
/* page-cache diagram */
.pcd { display: grid; grid-template-columns: 1fr 1fr .9fr; gap: .15em 1em; font-size: .78em; margin: -.2em 0 .5em; }
.pc-box { border: 1px solid var(--line); border-radius: 10px; padding: .45em .7em; background: var(--bg); }
.pc-box h5 { margin: 0 0 .3em; font-size: .95em; color: var(--ink); }
.pc-box.blue { background: #eff4ff; border-color: #c9dcf8; } .pc-box.green { background: #ecfdf5; border-color: #b7e6cf; }
.pc-box.kernel { background: #fff7ed; border-color: #fcd9b6; } .pc-box.ram, .pc-box.pt { background: #f8fafc; }
.pc-box.store { background: #f8fafc; } .pc-box.pc-gpu { background: #f5f3ff; border-color: #ddd6fe; }
.span2 { grid-column: span 2; }
.pc-in { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: .2em .5em; margin-bottom: .3em; }
.pc-vm { font-family: "SF Mono", Menlo, monospace; font-size: .85em; background: #fde2e2; border-radius: 4px; padding: .15em .4em; }
.pc-vm span, .pc-map span { color: var(--muted); margin-right: .3em; }
.pc-map { font-family: "SF Mono", Menlo, monospace; font-size: .85em; margin: .15em 0; }
.pc-note { font-size: .85em; color: var(--good); font-weight: 600; margin-top: .25em; }
.pc-file { font-family: "SF Mono", Menlo, monospace; font-size: .85em; color: var(--ink2); margin-bottom: .3em; }
.pc-pages { display: flex; gap: .35em; align-items: center; }
.pc-pages i { font-style: normal; background: #fde2e2; border: 1px solid #f5b4b4; border-radius: 4px; padding: .2em .55em; font-size: .85em; }
.pc-pages i.dots { background: none; border: none; } .pc-pages small { color: var(--muted); margin-left: .3em; }
.pc-row { background: #fff; border: 1px solid var(--line); border-radius: 5px; padding: .12em .5em; margin: .18em 0; font-size: .85em; }
.pc-row.w { background: #ddd6fe; border-color: #c4b5fd; font-weight: 600; } .pc-row.dim { color: var(--muted); border-style: dashed; }
.pc-box.store .pc-row { font-family: "SF Mono", Menlo, monospace; }
.pc-arr { color: var(--muted); font-size: .88em; padding: .1em .4em; } .pc-arr.up { color: var(--good); font-weight: 600; }
/* memory layout */
.mem { display: grid; grid-template-columns: 7em 1fr; gap: .25em .9em; align-items: center; margin: .4em 0 1em; }
.mem-label { text-align: right; font-size: .85em; font-weight: 600; color: var(--ink2); line-height: 1.2; }
.mem-label small { display: block; font-weight: 400; color: var(--muted); font-size: .85em; }
.mem-track { display: flex; gap: 2px; height: 2.3em; }
.mem-track.gpus { gap: 1em; } .gpu { flex: 1; display: flex; gap: 2px; }
.mseg { border-radius: 5px; display: flex; align-items: center; padding: 0 .65em; font-size: .72em; font-weight: 600;
  white-space: nowrap; overflow: hidden; color: #fff; }
.m-vol { background: #e8837b; } .m-cache { background: #1baf7a; } .m-proc { background: #2a78d6; }
.m-free { background: #e2e8f0; color: var(--muted); } .m-w { background: #1baf7a; } .m-kv { background: #e2e8f0; color: var(--muted); }
.mem-arrow { font-size: .72em; color: var(--muted); padding-left: .5em; white-space: nowrap; }
/* page-cache flows */
.flows { margin: .4em 0 1.1em; display: flex; flex-direction: column; gap: .55em; }
.flow { display: grid; grid-template-columns: 5em 1fr 2em 1fr 2em 1fr; align-items: stretch; gap: 0 .2em; }
.flow-tag { font-size: .72em; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; align-self: center; color: var(--muted); }
.flow-tag.red { color: var(--bad); } .flow-tag.green { color: var(--good); }
.fbox { border: 1px solid var(--line); border-radius: 10px; padding: .5em .8em; background: var(--bg); font-size: .9em; font-weight: 600;
  display: flex; flex-direction: column; justify-content: center; color: var(--ink); }
.fbox small { font-weight: 400; color: var(--muted); font-size: .82em; }
.fbox.slow { border-color: #f5b4b4; background: var(--bad-soft); } .fbox.ram { border-color: #9fdcc0; background: var(--good-soft); }
.farr { display: flex; align-items: center; justify-content: center; color: var(--faint); font-size: 1.2em; }
/* bytecode page */
.bcw { display: grid; grid-template-columns: 1fr 13em; gap: 2em; align-items: start; margin-bottom: 1em; }
.layers h4 { margin-bottom: .45em; }
.ly { border: 1px solid var(--line); border-radius: 7px; padding: .35em .7em; margin-bottom: 3px; font-size: .8em; background: var(--bg2); color: var(--ink2); }
.ly.hot { background: var(--good-soft); border-color: #9fdcc0; color: var(--ink); font-weight: 600; }
.ly small { display: block; font-weight: 400; color: var(--good); }
/* timeline */
.tl { margin: .6em 0 1.2em; }
.tl-row { display: grid; grid-template-columns: 5em 1fr; gap: .8em; align-items: center; margin: .35em 0; }
.tl-label { text-align: right; color: var(--muted); font-size: .88em; }
.tl-track { position: relative; height: 1.9em; }
.tl-seg { position: absolute; top: 0; bottom: 0; border-radius: 4px; color: #fff; font-size: .72em; font-weight: 600;
  display: flex; align-items: center; padding: 0 .6em; white-space: nowrap; overflow: hidden; }
.t-init { background: #94a3b8; } .t-note { background: none; color: var(--good); padding-left: 0; } .t-bad { background: #e8837b; } .t-load { background: #1baf7a; } .t-pre { background: #a7e3c9; color: #065f46; }
.tl-axis { position: relative; height: 1.2em; border-top: 1px solid var(--line); }
.tl-axis span { position: absolute; top: .2em; transform: translateX(-50%); font-size: .7em; color: var(--faint); }
.tl-axis span:first-child { transform: none; }
/* architecture detail */
ul.dots { list-style: none; padding: 0; margin: 0; } ul.dots li { position: relative; padding-left: 1.2em; margin: .5em 0; }
ul.dots li::before { content: ""; position: absolute; left: 0; top: .55em; width: .45em; height: .45em; border-radius: 50%; background: var(--accent); }
.pc > div > table.plain + .small { margin-top: .8em; }
/* architecture */
.lane { border-radius: 14px; padding: .8em 1em 1em; }
.lane.local { background: var(--accent-soft); } .lane.modal { background: var(--good-soft); }
.lane.local h4 { color: var(--accent); } .lane.modal h4 { color: var(--good); }
.arch { display: grid; grid-template-columns: 1fr 2.2em 1fr 2.2em 1fr 2.2em 1fr 2.2em 1fr; align-items: stretch; }
.arch .box { background: var(--bg); border: 1px solid var(--line); border-radius: 10px; padding: .6em .75em; display: flex; flex-direction: column; gap: .2em; }
.arch .box b { font-size: 1em; } .arch .box span { font-size: .74em; color: var(--muted); line-height: 1.35; }
.arch .box.key { border: 1.5px solid var(--ink); }
.arch .box.wide { grid-column: span 3; }
.arch .arr { display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--faint); font-size: 1.2em; }
.arch .arr small { font-size: .5em; letter-spacing: .05em; }
.arch.link { height: 2.6em; } .arch.link .down { grid-column: 7; display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: .8em; font-weight: 600; }
.image img { display: block; max-width: 100%; max-height: 78%; margin: 0 auto; border: 1px solid var(--line); border-radius: 12px;
  box-shadow: 0 4px 24px rgba(15,23,42,.06); }
.footer { position: absolute; left: 6%; right: 6%; bottom: 3.2%; display: flex; justify-content: space-between; font-size: .66em;
  color: var(--faint); letter-spacing: .04em; }
.divider .footer, .cover .footer { color: #475569; }
.progress { position: fixed; left: 0; bottom: 0; height: 3px; background: linear-gradient(90deg, #2563eb, #34d399); transition: width .2s; }
@media print {
  @page { size: 16in 9in; margin: 0; }
  html, body { background: #fff; } .deck { position: static; display: block; }
  section { display: block !important; width: 16in; height: 9in; font-size: 18pt; page-break-after: always;
    -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .cover, .divider { display: flex !important; } .progress { display: none; }
}
/* ---- themes (press T in the deck to cycle; ?theme=name picks one) */
html[data-theme="minimal"] { --accent: #111827; --accent-soft: #f3f4f6; }
html[data-theme="minimal"] .cover, html[data-theme="minimal"] .divider { background: var(--bg); }
html[data-theme="minimal"] .cover h1, html[data-theme="minimal"] .divider h1 { color: var(--ink); }
html[data-theme="minimal"] .divider h1 { font-size: 4.2em; }
html[data-theme="minimal"] .divider .num { color: var(--faint); font-size: 3em; letter-spacing: 0; font-weight: 300; }
html[data-theme="minimal"] .cover .author, html[data-theme="minimal"] .divider p { color: var(--muted); }
html[data-theme="minimal"] .panel { box-shadow: none; }
html[data-theme="minimal"] .kicker { color: var(--faint); }

html[data-theme="dark"] { --ink: #f1f5f9; --ink2: #cbd5e1; --muted: #94a3b8; --faint: #64748b; --bg: #0b1220; --bg2: #111a2e;
  --line: #1e293b; --accent: #60a5fa; --accent-soft: #0f1d3a; --good: #34d399; --good-soft: #0b2a22; --bad: #f87171; --bad-soft: #2a1215; }
html[data-theme="dark"] .panel, html[data-theme="dark"] .step, html[data-theme="dark"] .arch .box { background: #0f172a; box-shadow: none; }
html[data-theme="dark"] .step:last-child { background: var(--accent-soft); }
html[data-theme="dark"] table.plain th { border-bottom-color: var(--muted); }

html[data-theme="editorial"] { --ink: #1c1917; --ink2: #44403c; --muted: #78716c; --faint: #a8a29e; --bg: #faf7f2; --bg2: #f3eee6;
  --line: #e7e0d6; --accent: #c2410c; --accent-soft: #fbeee4; --good: #3f6212; --good-soft: #f1f5e6; --bad: #b91c1c; --bad-soft: #fbeaea; --dark: #1c1917; }
html[data-theme="editorial"] h1, html[data-theme="editorial"] h2, html[data-theme="editorial"] .statement,
html[data-theme="editorial"] .arch .box b, html[data-theme="editorial"] .step h4 { font-family: Georgia, "Times New Roman", serif; font-weight: 700; letter-spacing: -.01em; }
html[data-theme="editorial"] .cover, html[data-theme="editorial"] .divider { background: var(--dark); }
html[data-theme="editorial"] .divider .num { color: #fb923c; }
html[data-theme="editorial"] .panel { box-shadow: none; }
.theme-toast { position: fixed; top: 14px; right: 18px; background: rgba(0,0,0,.75); color: #fff; font-size: 13px; padding: 6px 12px;
  border-radius: 6px; opacity: 0; transition: opacity .3s; pointer-events: none; }
.theme-toast.on { opacity: 1; }
</style>
</head>
<body>
<div class="deck">
__SLIDES__
</div>
<div class="progress" id="progress"></div>
<script>
const slides = [...document.querySelectorAll("section")];
let kicker = "";
slides.forEach((s, i) => {
  if (s.dataset.kicker) kicker = s.dataset.kicker;
  else if (kicker && !s.classList.contains("cover")) s.insertAdjacentHTML("afterbegin", `<div class="kicker">${s.dataset.label || kicker}</div>`);
  if (i) s.insertAdjacentHTML("beforeend",
    `<div class="footer"><span>Fast Engine Startup</span><span>${String(i + 1).padStart(2, "0")} / ${slides.length}</span></div>`);
});
let cur = Math.min(Math.max(parseInt(location.hash.slice(1)) - 1 || 0, 0), slides.length - 1);
function show(i) {
  cur = Math.min(Math.max(i, 0), slides.length - 1);
  slides.forEach((s, k) => s.classList.toggle("active", k === cur));
  document.getElementById("progress").style.width = `${100 * (cur + 1) / slides.length}%`;
  history.replaceState(null, "", `#${cur + 1}`);
}
document.addEventListener("keydown", (e) => {
  if (["ArrowRight", "ArrowDown", "PageDown", " ", "Enter"].includes(e.key)) { e.preventDefault(); show(cur + 1); }
  if (["ArrowLeft", "ArrowUp", "PageUp", "Backspace"].includes(e.key)) { e.preventDefault(); show(cur - 1); }
  if (e.key === "Home") show(0);
  if (e.key === "End") show(slides.length - 1);
});
document.addEventListener("click", (e) => { if (!e.target.closest("a")) show(cur + (e.clientX < innerWidth / 3 ? -1 : 1)); });
show(cur);
const THEMES = ["midnight", "minimal", "dark", "editorial"];
const q = new URLSearchParams(location.search).get("theme");
if (THEMES.includes(q)) document.documentElement.dataset.theme = q;
const toast = document.body.appendChild(Object.assign(document.createElement("div"), {className: "theme-toast"}));
document.addEventListener("keydown", (e) => {
  if (e.key !== "t" && e.key !== "T") return;
  const t = THEMES[(THEMES.indexOf(document.documentElement.dataset.theme) + 1) % THEMES.length];
  document.documentElement.dataset.theme = t;
  toast.textContent = "theme: " + t; toast.classList.add("on"); setTimeout(() => toast.classList.remove("on"), 1200);
});
</script>
</body>
</html>
"""

out = os.path.join(HERE, "slides.html")
open(out, "w").write(HTML.replace("__THEME__", THEME).replace("__SLIDES__", "\n".join(SLIDES)))
print("wrote", out, f"({len(SLIDES)} slides, {os.path.getsize(out) / 1e3:.0f} kB)")
