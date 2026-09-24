"use strict";
// Shared chart helpers for the dashboard pages (extracted from index.html).
const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle($("app")).getPropertyValue(v).trim();
const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5"];
const color = (i) => css(SERIES[i % SERIES.length]);
const fmt = (s, d = 1) => (s == null || isNaN(s) ? "—" : Number(s).toFixed(d));
const state = { tables: {} };

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
// h("div", {class: "x"}, "text") or h("div", {}, childElement, "more text", ...): strings and
// numbers become text, elements are appended (passing an element used to render "[object …]").
function h(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const c of children) if (c != null) e.append(c instanceof Node ? c : String(c));
  return e;
}
function text(parent, x, y, str, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}
function niceTicks(max, n = 5) {
  if (max <= 0) return [0];
  const raw = max / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  if (out[out.length - 1] < max) out.push(+(out[out.length - 1] + step).toFixed(6));  // axis always covers the data
  return out;
}
function roundedBar(parent, x, y, w, hgt, fill, { left = true, right = true, r = 4 } = {}) {
  // 4px rounded data-ends; square where a segment touches its neighbor.
  r = Math.min(r, w / 2, hgt / 2);
  const rl = left ? r : 0, rr = right ? r : 0;
  const d = `M${x + rl},${y} H${x + w - rr} ${rr ? `A${rr},${rr} 0 0 1 ${x + w},${y + rr}` : ""} V${y + hgt - rr}
    ${rr ? `A${rr},${rr} 0 0 1 ${x + w - rr},${y + hgt}` : ""} H${x + rl} ${rl ? `A${rl},${rl} 0 0 1 ${x},${y + hgt - rl}` : ""}
    V${y + rl} ${rl ? `A${rl},${rl} 0 0 1 ${x + rl},${y}` : ""} Z`;
  return el("path", { d, fill }, parent);
}

// ---------- tooltip ----------
const tip = $("tip");
function showTip(evt, title, rows) {
  tip.replaceChildren();
  tip.appendChild(h("div", { class: "t" }, title));
  for (const r of rows) {
    const row = h("div", { class: "r" });
    if (r.color) { const k = h("span", { class: "k" }); k.style.background = r.color; row.appendChild(k); }
    row.appendChild(h("span", { class: "v" }, r.value));
    if (r.name) row.appendChild(h("span", { class: "n" }, r.name));
    tip.appendChild(row);
  }
  tip.style.display = "block";
  const pad = 14, bw = tip.offsetWidth, bh = tip.offsetHeight;
  let x = (evt.clientX ?? 0) + pad, y = (evt.clientY ?? 0) + pad;
  if (evt.type === "focus") { const b = evt.target.getBoundingClientRect(); x = b.right + pad; y = b.top; }
  if (x + bw > innerWidth - 8) x = (evt.clientX ?? innerWidth) - bw - pad;
  if (y + bh > innerHeight - 8) y = innerHeight - bh - 8;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => (tip.style.display = "none");
function hover(node, fn) {
  node.classList.add("hit");
  node.setAttribute("tabindex", "0");
  const on = (e) => { const [t, rows] = fn(); showTip(e, t, rows); };
  node.addEventListener("pointermove", on);
  node.addEventListener("focus", on);
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("blur", hideTip);
}

// ---------- table-view toggle ----------
function mount(id, drawChart, drawTable) {
  const host = $(id);
  host.replaceChildren();
  if (state.tables[id]) host.appendChild(drawTable()); else drawChart(host);
}
function table(headers, rows, numCols = []) {
  const t = h("table"), thead = h("thead"), tr = h("tr");
  headers.forEach((c, i) => tr.appendChild(h("th", numCols.includes(i) ? { class: "num" } : {}, c)));
  thead.appendChild(tr); t.appendChild(thead);
  const tb = h("tbody");
  for (const r of rows) {
    const row = h("tr");
    r.forEach((c, i) => row.appendChild(h("td", numCols.includes(i) ? { class: "num" } : {}, c)));
    tb.appendChild(row);
  }
  t.appendChild(tb);
  return t;
}
function kpi(value, unit, label, detail, good) {
  const d = h("div", { class: "kpi" });
  const v = h("div", { class: "v" }, value);
  if (unit) v.appendChild(h("small", {}, unit));
  d.append(v, h("div", { class: "l" }, label));
  if (detail) d.appendChild(h("div", { class: "d" + (good ? " good" : "") }, detail));
  return d;
}
const median = (xs) => { const a = [...xs].sort((x, y) => x - y); const m = a.length >> 1; return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2; };
