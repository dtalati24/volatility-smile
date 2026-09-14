// Smile Board front end: draws what app/server.py computes; all maths is in Python.
"use strict";

const BUTTONS = ["put_1", "put_2", "put_5", "put_10", "put_25", "atm", "call_25", "call_10", "call_5", "call_2", "call_1"];
const LABELS = { atm: "ATM (50Δ)" };
for (const b of BUTTONS) if (!LABELS[b]) LABELS[b] = b.replace(/(put|call)_(\d+)/, (_, s, d) => `${d}Δ ${s === "put" ? "put" : "call"}`);
const ZERO = Object.fromEntries(BUTTONS.map((b) => [b, 0]));

const $ = (id) => document.getElementById(id);
let state = null;      // last good response from the server
let request = 0;       // newest request number; older responses are ignored
let fitsRunning = 0;   // fits in flight; while any runs, button and setting changes are ignored
const refitting = () => fitsRunning > 0;

// The smile drawn uses fitted + offsets for every button.
let fitted = { ...ZERO };    // bells fitted to the mids, by Refit or the automatic fitter
let offsets = { ...ZERO };   // your clicks, kept on top of every automatic fit; Refit clears them
let accepted = { fitted: { ...ZERO }, offsets: { ...ZERO } };  // what the last good response drew
let rmseAtFit = null;        // the bells' fit error just after the last Refit (for the stale check)
let lastFit = null;          // {method, handover, converged, ms, spline_ms?, rmse} of the last fit
let stale = [];              // reasons the spline looks stale (from the automatic fitter)

const total = (f, o) => Object.fromEntries(BUTTONS.map((b) => [b, f[b] + o[b]]));

// ---------- server ----------

async function api(path, options) {
  const r = await fetch(path, options);
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

// Runs a request; if it is still the newest when it returns, keeps it and calls onAccept.
async function load(promise, onAccept) {
  const n = ++request;
  try {
    const result = await promise;
    if (n !== request) return true;  // superseded by a newer request, not rejected
    state = result;
    if (onAccept) onAccept(result);
    showError("");
    render();
    return true;
  } catch (e) {
    if (n === request) showError(e.message);
    return false;
  }
}

// Counted, so a quick fit finishing cannot unlock the buttons while a Refit is still running.
function busy(on) {
  fitsRunning += on ? 1 : -1;
  document.body.classList.toggle("busy", refitting());
}

// Refit: a fresh spline and fresh bells fitted to the mids. Clears the offsets.
async function refit() {
  busy(true);
  const q = new URLSearchParams({ slice: $("slice").value, knots: $("knots").value, k: $("k").value, delta: $("delta").value });
  const ok = await load(api(`/api/fit?${q}`), (r) => {
    fitted = r.fitted;
    offsets = { ...ZERO };
    accepted = { fitted, offsets };
    rmseAtFit = r.fit.rmse;
    lastFit = r.fit;
    stale = [];
  });
  busy(false);
  if (!ok && state) keepInputsOnWhatIsShown();
  return ok;
}

// Automatic fitter: the spline stays, the bells are refitted to the mids from
// the current fitted values, your offsets stay on top.
async function autofit() {
  if (!state || refitting()) return false;
  busy(true);
  const sentOffsets = offsets;
  const ok = await load(post("/api/autofit", {
    slice: state.slice.name, base_slice: state.base_slice, knots: state.knots, fitted, offsets: sentOffsets,
    k: $("k").value, delta: $("delta").value, stiffness: Number($("stiffness").value) || 0, rmse_at_fit: rmseAtFit,
  }), (r) => {
    fitted = r.fitted;
    accepted = { fitted, offsets: sentOffsets };
    lastFit = r.fit;
    stale = r.stale;
  });
  busy(false);
  if (!ok && state) keepInputsOnWhatIsShown();
  return ok;
}

// Draw the smile for the current fitted values plus these offsets (no fitting).
function draw(sentOffsets) {
  const sentFitted = fitted;
  return load(post("/api/smile", {
    slice: state.slice.name, base_slice: state.base_slice, knots: state.knots,
    buttons: total(sentFitted, sentOffsets), k: $("k").value, delta: $("delta").value,
  }), () => { accepted = { fitted: sentFitted, offsets: sentOffsets }; });
}

function keepInputsOnWhatIsShown() {
  $("slice").value = state.slice.name;
  $("knots").value = state.knots;
  $("k").value = state.k;
  $("delta").value = state.delta;
}

// k or delta changed: with Auto on, refit the bells; otherwise just redraw.
async function settingChanged() {
  if (!state || refitting()) { if (state) keepInputsOnWhatIsShown(); return; }
  const ok = $("auto").checked ? await autofit() : await draw(offsets);
  if (!ok && state) keepInputsOnWhatIsShown();
}

// ---------- buttons ----------

function buildButtons() {
  const box = $("buttons");
  for (const name of BUTTONS) {
    const el = document.createElement("div");
    el.className = "knob" + (name === "atm" ? " atm" : "");
    el.dataset.name = name;
    el.innerHTML = `<div class="name">${LABELS[name]}</div><div class="value">0.00</div><div class="offset"></div>`;
    // Any press of the left or right mouse button raises your offset; with Ctrl
    // held it lowers. One mousedown per press, so a browser that also sends
    // click/contextmenu (Mac Ctrl + click) cannot count it twice.
    el.addEventListener("mousedown", (e) => { if (e.button === 0 || e.button === 2) press(name, e.ctrlKey ? -1 : 1); });
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    box.appendChild(el);
  }
}

// A click moves your offset for that button. Quick clicks build on each other
// (offsets change here at once, ahead of the server).
async function press(name, direction) {
  if (!state || refitting()) return;
  const step = Number($("step").value) / 100;
  offsets = { ...offsets, [name]: Math.round((offsets[name] + direction * step) * 1e6) / 1e6 };
  showButtons();
  const sent = offsets;
  const ok = await draw(sent);
  if (!ok && offsets === sent) {  // rejected (e.g. ambiguous vols): back to what is drawn
    ({ fitted, offsets } = accepted);
    showButtons();
  }
}

async function setAll(newFitted, newOffsets) {
  if (!state || refitting()) return;
  fitted = newFitted;
  offsets = newOffsets;
  showButtons();
  const sent = offsets;
  const ok = await draw(sent);
  if (!ok && offsets === sent) {  // rejected: back to what is drawn
    ({ fitted, offsets } = accepted);
    showButtons();
  }
}

// ---------- chart view: zoom buttons and drag ----------

let view = null;   // strike range picked by zoom or drag; null = automatic
let shown = null;  // strike range and plot width last drawn
let drag = null;   // pointer drag in progress

function fullRange() {
  return [Math.min(...state.board.strike), Math.max(...state.board.strike)];
}

// Default: the 1Δ put to 1Δ call nodes plus a margin, or every quote.
function autoRange() {
  let [xlo, xhi] = fullRange();
  if (!$("full").checked && state.nodes.length >= 2) {
    const ns = state.nodes.map((n) => n.strike);
    const pad = 0.15 * (Math.max(...ns) - Math.min(...ns));
    const lo = Math.max(xlo, Math.min(...ns) - pad), hi = Math.min(xhi, Math.max(...ns) + pad);
    if (hi > lo) { xlo = lo; xhi = hi; }
  }
  return [xlo, xhi];
}

// Keep a view inside the quoted strikes, between 1/200 of them and all of them.
function clampView(lo, hi) {
  const [flo, fhi] = fullRange();
  const w = Math.max(Math.min(hi - lo, fhi - flo), (fhi - flo) / 200);
  const start = Math.min(Math.max(lo, flo), fhi - w);
  return [start, start + w];
}

function setView(lo, hi) {
  const [a, z] = clampView(lo, hi);
  view = { lo: a, hi: z };
  drawChart();
}

function zoom(factor) {
  if (!state || !shown) return;
  const mid = (shown.lo + shown.hi) / 2, half = ((shown.hi - shown.lo) * factor) / 2;
  setView(mid - half, mid + half);
}

function resetView() {
  view = null;
  if (state) drawChart();
}

function setupDrag() {
  const svg = $("chart");
  svg.addEventListener("pointerdown", (e) => {
    if (!state || !shown || e.button !== 0) return;
    drag = { x: e.clientX, lo: shown.lo, hi: shown.hi };
    svg.setPointerCapture(e.pointerId);
    svg.classList.add("dragging");
    $("tooltip").hidden = true;
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const shift = ((e.clientX - drag.x) / shown.plotWidth) * (drag.hi - drag.lo);
    setView(drag.lo - shift, drag.hi - shift);
  });
  const end = () => { drag = null; svg.classList.remove("dragging"); };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
}

// ---------- chart ----------

const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function niceTicks(lo, hi, count) {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(t);
  return out;
}

function drawChart() {
  const svg = $("chart");
  const { width, height } = svg.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const m = { l: 48, r: 12, t: 22, b: 34 };
  const b = state.board, c = state.curve;

  let [xlo, xhi] = view ? clampView(view.lo, view.hi) : autoRange();
  if (!(xhi > xlo)) { xlo -= 1; xhi += 1; }
  shown = { lo: xlo, hi: xhi, plotWidth: width - m.l - m.r };
  $("tooltip").hidden = true;
  const inX = (k) => k >= xlo && k <= xhi;
  const vols = [];
  b.strike.forEach((k, i) => { if (inX(k)) vols.push(b.iv_mid[i], b.model_vol[i]); });
  const showSvi = $("show-svi").checked;
  c.strike.forEach((k, i) => { if (inX(k)) vols.push(c.model[i], c.base[i], showSvi ? c.svi[i] : null); });
  const finite = vols.filter((v) => v != null);
  let ylo = finite.length ? Math.min(...finite) * 100 : 0, yhi = finite.length ? Math.max(...finite) * 100 : 1;
  const ypad = 0.08 * (yhi - ylo || 1);
  ylo = Math.max(0, ylo - ypad); yhi += ypad;

  const X = (k) => m.l + ((k - xlo) / (xhi - xlo)) * (width - m.l - m.r);
  const Y = (v) => height - m.b - ((v * 100 - ylo) / (yhi - ylo)) * (height - m.t - m.b);

  const grid = el("g", { stroke: "#eceef1" }, svg);
  const labels = el("g", { fill: "#6b7280", "font-size": 11 }, svg);
  for (const t of niceTicks(ylo, yhi, 6)) {
    const y = Y(t / 100);
    el("line", { x1: m.l, x2: width - m.r, y1: y, y2: y }, grid);
    el("text", { x: m.l - 6, y: y + 4, "text-anchor": "end" }, labels).textContent = t.toFixed(0);
  }
  for (const t of niceTicks(xlo, xhi, 8)) {
    const x = X(t);
    el("line", { x1: x, x2: x, y1: m.t, y2: height - m.b }, grid);
    el("text", { x, y: height - m.b + 14, "text-anchor": "middle" }, labels).textContent = t.toFixed(0);
  }
  el("text", { x: 12, y: m.t - 8 }, labels).textContent = "vol %";

  const clip = el("clipPath", { id: "plot" }, el("defs", {}, svg));
  el("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b }, clip);
  const plot = el("g", { "clip-path": "url(#plot)" }, svg);

  const fx = X(state.slice.forward);
  el("line", { x1: fx, x2: fx, y1: m.t, y2: height - m.b, stroke: "#6b7280", "stroke-width": 1 }, plot);

  for (const n of state.nodes) {
    if (!inX(n.strike)) continue;
    const x = X(n.strike);
    el("line", { x1: x, x2: x, y1: height - m.b - 6, y2: height - m.b, stroke: "#d4763b", "stroke-width": 1.5 }, plot);
  }

  b.strike.forEach((k, i) => {
    if (!inX(k) || b.iv_bid[i] == null || b.iv_ask[i] == null) return;
    el("line", { x1: X(k), x2: X(k), y1: Y(b.iv_bid[i]), y2: Y(b.iv_ask[i]), stroke: "#b8bcc4", "stroke-width": 2 }, plot);
  });
  // A point after a gap starts a new line segment.
  const path = (ys) => c.strike.map((k, i) => (ys[i] == null ? "" : `${i && ys[i - 1] != null ? "L" : "M"}${X(k).toFixed(1)},${Y(ys[i]).toFixed(1)}`)).join("");
  if (showSvi) el("path", { d: path(c.svi), fill: "none", stroke: "#1e9e5a", "stroke-width": 2 }, plot);
  el("path", { d: path(c.base), fill: "none", stroke: "#3b74d4", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, plot);
  el("path", { d: path(c.model), fill: "none", stroke: "#d4763b", "stroke-width": 2 }, plot);
  b.strike.forEach((k, i) => {
    if (!inX(k) || b.iv_mid[i] == null) return;
    el("circle", { cx: X(k), cy: Y(b.iv_mid[i]), r: 2.5, fill: "#1f2328" }, plot);
  });

  // Hover: nearest quote.
  const hit = el("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent" }, svg);
  const tip = $("tooltip");
  const cross = el("line", { y1: m.t, y2: height - m.b, stroke: "#1f2328", "stroke-width": 1, opacity: 0, "pointer-events": "none" }, svg);
  hit.addEventListener("mousemove", (e) => {
    if (drag) return;
    const r = svg.getBoundingClientRect();
    const px = e.clientX - r.left;
    const k = xlo + ((px - m.l) / (width - m.l - m.r)) * (xhi - xlo);
    let best = -1;
    b.strike.forEach((s, i) => { if (inX(s) && (best < 0 || Math.abs(s - k) < Math.abs(b.strike[best] - k))) best = i; });
    if (best < 0) return;
    const x = X(b.strike[best]);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("opacity", 0.25);
    const pct = (v) => (v == null ? "–" : (100 * v).toFixed(2));
    tip.textContent = `${b.is_call[best] ? "Call" : "Put"} ${b.strike[best]}\n` +
      `mid IV   ${pct(b.iv_mid[best])}  (${pct(b.iv_bid[best])} / ${pct(b.iv_ask[best])})\n` +
      `model IV ${pct(b.model_vol[best])}\ndelta    ${b.model_delta[best] == null ? "–" : b.model_delta[best].toFixed(3)}`;
    tip.hidden = false;
    const panel = $("chart-panel").getBoundingClientRect();
    tip.style.left = `${Math.min(e.clientX - panel.left + 12, panel.width - 180)}px`;
    tip.style.top = `${Math.min(e.clientY - panel.top + 12, panel.height - 90)}px`;
  });
  hit.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("opacity", 0); });
}

// ---------- board ----------

function drawBoard() {
  const b = state.board;
  const fmt = (v, d) => (v == null ? "–" : v.toFixed(d));
  const pts = (v) => (v == null ? null : 100 * v);
  let atm = 0;
  b.strike.forEach((k, i) => { if (Math.abs(k - state.slice.forward) < Math.abs(b.strike[atm] - state.slice.forward)) atm = i; });
  const rows = b.strike.map((k, i) => {
    const diff = b.model_vol[i] == null || b.iv_mid[i] == null ? null : b.model_vol[i] - b.iv_mid[i];
    const outside = b.iv_bid[i] != null && b.iv_ask[i] != null && (b.model_vol[i] < b.iv_bid[i] || b.model_vol[i] > b.iv_ask[i]);
    return `<tr${i === atm ? ' class="atm"' : ""}>` +
      `<td>${b.is_call[i] ? "Call" : "Put"}</td><td>${fmt(k, 0)}</td><td>${fmt(b.model_delta[i], 3)}</td>` +
      `<td>${fmt(b.bid[i], 2)}</td><td>${fmt(b.ask[i], 2)}</td><td>${fmt(b.mid[i], 2)}</td><td>${fmt(b.model_price[i], 2)}</td>` +
      `<td>${fmt(pts(b.iv_bid[i]), 2)}</td><td>${fmt(pts(b.iv_mid[i]), 2)}</td><td>${fmt(pts(b.iv_ask[i]), 2)}</td>` +
      `<td${outside ? ' class="outside" title="Model vol outside the bid/ask"' : ""}>${fmt(pts(b.model_vol[i]), 2)}</td>` +
      `<td>${(diff > 0 ? "+" : "") + fmt(pts(diff), 2)}</td>` +
      `<td>${fmt(b.volume[i], 0)}</td><td>${fmt(b.open_interest[i], 0)}</td></tr>`;
  });
  document.querySelector("#board tbody").innerHTML = rows.join("");
}

// ---------- render ----------

function render() {
  const s = state.slice;
  const pts = (v) => (v == null ? "n/a" : (100 * v).toFixed(3));
  let fitText = "";
  if (lastFit) {
    const how = lastFit.method === "fast" ? "fast" : `scipy${lastFit.handover ? ` after fast: ${lastFit.handover}` : ""}`;
    fitText = ` · bells fit ${lastFit.ms.toFixed(1)} ms (${how})` + (lastFit.converged ? "" : " · did not converge");
  }
  $("stats").textContent = `F ${s.forward.toFixed(2)} · τ ${s.tau.toFixed(4)} · quotes ${s.as_of.slice(0, 16).replace("T", " ")} UTC · ` +
    `RMSE spline ${pts(state.rmse_base)} → +bells ${pts(state.rmse_model)} · SVI ${pts(state.rmse_svi)} vol pts` + fitText;
  $("knots").value = state.knots;  // the box always shows the base that is drawn
  $("stale").hidden = stale.length === 0;
  $("stale").textContent = stale.length ? `Spline looks stale, press Refit: ${stale.join("; ")}` : "";
  showButtons();
  drawChart();
  drawBoard();
}

// Each button shows fitted + offset, and the offset underneath when there is one.
function showButtons() {
  for (const knob of document.querySelectorAll(".knob")) {
    const name = knob.dataset.name;
    const v = 100 * (fitted[name] + offsets[name]), o = 100 * offsets[name];
    knob.querySelector(".value").textContent = (v > 0 ? "+" : "") + v.toFixed(2);
    knob.querySelector(".offset").textContent = Math.abs(o) > 1e-9 ? `offset ${o > 0 ? "+" : ""}${o.toFixed(2)}` : "";
    knob.classList.toggle("up", v > 1e-9);
    knob.classList.toggle("down", v < -1e-9);
  }
}

// ---------- start ----------

async function main() {
  buildButtons();
  const names = await api("/api/slices");
  $("slice").innerHTML = names.map((n) => `<option>${n}</option>`).join("");
  const month = names.find((n) => n.startsWith("spx_") && n.includes("2026-10-16"));
  if (month) $("slice").value = month;
  $("slice").addEventListener("change", () => { view = null; refit(); });
  $("refit").addEventListener("click", refit);
  $("autofit").addEventListener("click", autofit);
  $("zoom-in").addEventListener("click", () => zoom(1 / 1.5));
  $("zoom-out").addEventListener("click", () => zoom(1.5));
  $("zoom-reset").addEventListener("click", resetView);
  setupDrag();
  $("reset").addEventListener("click", () => setAll({ ...ZERO }, { ...ZERO }));
  $("clear-offsets").addEventListener("click", () => setAll(fitted, { ...ZERO }));
  $("k").addEventListener("change", settingChanged);
  $("delta").addEventListener("change", settingChanged);
  $("stiffness").addEventListener("change", () => { if ($("auto").checked) settingChanged(); });
  $("full").addEventListener("change", resetView);
  $("show-svi").addEventListener("change", () => { $("svi-legend").hidden = !$("show-svi").checked; if (state) drawChart(); });
  $("knots").addEventListener("change", () => {
    if (refitting()) { if (state) $("knots").value = state.knots; return; }
    refit();
  });
  window.addEventListener("resize", () => state && drawChart());
  await refit();
}

main().catch((e) => showError(e.message));
