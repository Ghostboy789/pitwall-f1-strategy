/* Pit Wall charts.
 *
 * Hand-rolled SVG rather than a charting library, so every mark follows the
 * same spec: thin bars with 4px rounded data-ends anchored to the baseline,
 * a 2px surface gap between adjacent fills, recessive grid, selective direct
 * labels, and a hover layer on every plot.
 */

const TIP = document.getElementById("tip");

function showTip(html, evt) {
  TIP.innerHTML = html;
  TIP.classList.add("on");
  const pad = 14;
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  const r = TIP.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  TIP.style.left = x + "px";
  TIP.style.top = y + "px";
}
function hideTip() { TIP.classList.remove("on"); }

const svgNS = "http://www.w3.org/2000/svg";
function el(name, attrs = {}, parent = null) {
  const n = document.createElementNS(svgNS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const fmt = (v, d = 1) => Number(v).toFixed(d);

/* Rounded only on the data end, square on the baseline. */
function barPath(x, y, w, h, r) {
  const rr = Math.min(r, h, w / 2);
  return `M${x},${y + h} L${x},${y + rr} Q${x},${y} ${x + rr},${y}
          L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr}
          L${x + w},${y + h} Z`;
}
function barPathH(x, y, w, h, r) {
  const rr = Math.min(r, w, h / 2);
  return `M${x},${y} L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr}
          L${x + w},${y + h - rr} Q${x + w},${y + h} ${x + w - rr},${y + h}
          L${x},${y + h} Z`;
}

/* ------------------------------------------------------------------ */
/* Headline: value of track position by circuit, horizontal bars + CI  */
/* ------------------------------------------------------------------ */

function drawTrackPosition(mount, rows) {
  mount.innerHTML = "";
  const labelW = 168, padR = 78, rowH = 27, top = 34, bot = 34;
  const W = mount.clientWidth || 860;
  const H = top + rows.length * rowH + bot;
  const plotW = W - labelW - padR;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, mount);

  const maxV = Math.max(...rows.map(r => r.value_hi ?? r.value_s)) * 1.02;
  const x = v => labelW + (v / maxV) * plotW;

  // Recessive grid + ticks
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = (maxV / ticks) * i;
    el("line", { class: "gridline", x1: x(v), x2: x(v), y1: top - 8, y2: H - bot }, svg);
    const t = el("text", { class: "tick-label", x: x(v), y: H - bot + 16, "text-anchor": "middle" }, svg);
    t.textContent = fmt(v, 0);
  }
  const ax = el("text", { class: "axis-label", x: labelW + plotW / 2, y: H - 4, "text-anchor": "middle" }, svg);
  ax.textContent = "seconds of race time lost while stuck behind a slower car";

  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const g = el("g", { class: "row" }, svg);
    const barH = 13;
    const by = y + (rowH - barH) / 2;

    const lab = el("text", {
      class: "cat-label" + (r.is_focus ? " focus" : ""),
      x: labelW - 12, y: by + barH - 2, "text-anchor": "end",
    }, g);
    lab.textContent = r.name;

    el("path", {
      class: "bar",
      d: barPathH(labelW, by, Math.max(x(r.value_s) - labelW, 2), barH, 4),
      fill: r.is_focus ? "var(--s1)" : "var(--s3)",
      opacity: r.is_focus ? 1 : 0.55,
    }, g);

    // 95% interval
    if (r.value_lo != null && r.value_hi != null) {
      const cy = by + barH / 2;
      el("line", { class: "whisker", x1: x(r.value_lo), x2: x(r.value_hi), y1: cy, y2: cy }, g);
      for (const v of [r.value_lo, r.value_hi]) {
        el("line", { class: "whisker", x1: x(v), x2: x(v), y1: cy - 4, y2: cy + 4 }, g);
      }
    }

    const val = el("text", { class: "mark-label", x: x(r.value_hi ?? r.value_s) + 9, y: by + barH - 2 }, g);
    val.textContent = fmt(r.value_s, 1) + "s";

    const hit = el("rect", { class: "hit", x: 0, y, width: W, height: rowH }, g);
    hit.addEventListener("mousemove", e => showTip(
      `<b>${r.name}</b><br>
       <span class="k">Cost of being stuck</span> <b>${fmt(r.value_s, 1)}s</b><br>
       <span class="k">95% interval</span> <b>${fmt(r.value_lo, 1)}–${fmt(r.value_hi, 1)}s</b><br>
       <span class="k">Pass chance per lap</span> <b>${(r.p_pass_per_lap * 100).toFixed(1)}%</b><br>
       <span class="k">Races in sample</span> <b>${r.n_races}</b>`, e));
    hit.addEventListener("mouseleave", hideTip);
  });
}

/* ------------------------------------------------------------------ */
/* Degradation: naive vs two-way vs censoring-corrected, grouped bars  */
/* ------------------------------------------------------------------ */

function drawDegradation(mount, rows) {
  mount.innerHTML = "";
  const labelW = 150, padR = 26, rowH = 46, top = 20, bot = 40;
  const W = mount.clientWidth || 520;
  const H = top + rows.length * rowH + bot;
  const plotW = W - labelW - padR;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, mount);

  const vals = rows.flatMap(r => [r.slope_naive, r.slope_twoway, r.slope_ipcw]);
  const lo = Math.min(0, ...vals), hi = Math.max(...vals);
  const span = (hi - lo) || 1;
  const x = v => labelW + ((v - lo) / span) * plotW;

  el("line", { class: "baseline", x1: x(0), x2: x(0), y1: top - 6, y2: H - bot }, svg);
  const zt = el("text", { class: "tick-label", x: x(0), y: H - bot + 16, "text-anchor": "middle" }, svg);
  zt.textContent = "0";
  const ax = el("text", { class: "axis-label", x: labelW + plotW / 2, y: H - 4, "text-anchor": "middle" }, svg);
  ax.textContent = "seconds lost per lap of tyre age";

  const series = [
    ["slope_naive", "var(--s2)", "Naive"],
    ["slope_twoway", "var(--s3)", "Two-way FE"],
    ["slope_ipcw", "var(--s1)", "+ censoring"],
  ];

  rows.forEach((r, i) => {
    const y0 = top + i * rowH;
    const g = el("g", { class: "row" }, svg);
    const lab = el("text", { class: "cat-label", x: labelW - 12, y: y0 + 24, "text-anchor": "end" }, g);
    lab.textContent = r.label;

    const barH = 9, gap = 2;   // 2px surface gap between adjacent fills
    series.forEach(([key, colour, name], s) => {
      const v = r[key];
      if (v == null || Number.isNaN(v)) return;
      const by = y0 + 6 + s * (barH + gap);
      const xa = Math.min(x(0), x(v)), xb = Math.max(x(0), x(v));
      el("path", {
        class: "bar", d: barPathH(xa, by, Math.max(xb - xa, 1.5), barH, 3), fill: colour,
      }, g);
      const hit = el("rect", { class: "hit", x: labelW, y: by - 1, width: plotW, height: barH + 2 }, g);
      hit.addEventListener("mousemove", e => showTip(
        `<b>${r.label}</b><br><span class="k">${name}</span> <b>${v.toFixed(4)} s/lap</b>` +
        (v < 0 ? '<br><span class="k">negative — physically impossible</span>' : ""), e));
      hit.addEventListener("mouseleave", hideTip);
    });
  });
}

/* ------------------------------------------------------------------ */
/* Reliability curve                                                   */
/* ------------------------------------------------------------------ */

function drawReliability(mount, rows) {
  mount.innerHTML = "";
  const W = mount.clientWidth || 420, H = 300;
  const m = { t: 16, r: 16, b: 44, l: 50 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, mount);

  const hi = Math.max(...rows.flatMap(d => [d.mean_predicted, d.observed_rate])) * 1.08 || 1;
  const x = v => m.l + (v / hi) * pw;
  const y = v => m.t + ph - (v / hi) * ph;

  for (let i = 0; i <= 4; i++) {
    const v = (hi / 4) * i;
    el("line", { class: "gridline", x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v) }, svg);
    const t = el("text", { class: "tick-label", x: m.l - 9, y: y(v) + 4, "text-anchor": "end" }, svg);
    t.textContent = (v * 100).toFixed(0) + "%";
    const t2 = el("text", { class: "tick-label", x: x(v), y: H - m.b + 17, "text-anchor": "middle" }, svg);
    t2.textContent = (v * 100).toFixed(0) + "%";
  }

  // Perfect calibration
  el("line", {
    x1: x(0), y1: y(0), x2: x(hi), y2: y(hi),
    stroke: "var(--axis)", "stroke-width": 2, "stroke-dasharray": "5 5",
  }, svg);

  const pts = rows.map(d => `${x(d.mean_predicted)},${y(d.observed_rate)}`).join(" ");
  el("polyline", { points: pts, fill: "none", stroke: "var(--s1)", "stroke-width": 2 }, svg);

  rows.forEach(d => {
    // 2px surface ring so overlapping markers stay separable
    el("circle", { cx: x(d.mean_predicted), cy: y(d.observed_rate), r: 6.5, fill: "var(--surface)" }, svg);
    const c = el("circle", { cx: x(d.mean_predicted), cy: y(d.observed_rate), r: 4.5, fill: "var(--s1)" }, svg);
    const hit = el("circle", { class: "hit", cx: x(d.mean_predicted), cy: y(d.observed_rate), r: 14 }, svg);
    hit.addEventListener("mousemove", e => showTip(
      `<span class="k">Model said</span> <b>${(d.mean_predicted * 100).toFixed(1)}%</b><br>
       <span class="k">Actually happened</span> <b>${(d.observed_rate * 100).toFixed(1)}%</b><br>
       <span class="k">Attempts in bin</span> <b>${d.n}</b>`, e));
    hit.addEventListener("mouseleave", hideTip);
  });

  const xl = el("text", { class: "axis-label", x: m.l + pw / 2, y: H - 6, "text-anchor": "middle" }, svg);
  xl.textContent = "predicted chance of completing the pass";
  const yl = el("text", {
    class: "axis-label", transform: `translate(13,${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle",
  }, svg);
  yl.textContent = "observed rate";
}

/* ------------------------------------------------------------------ */
/* Finishing-position histogram (live simulator)                       */
/* ------------------------------------------------------------------ */

function drawHistogram(mount, hist, nSims) {
  mount.innerHTML = "";
  const W = mount.clientWidth || 520, H = 220;
  const m = { t: 14, r: 12, b: 40, l: 40 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, height: H }, mount);

  const maxN = Math.max(...hist) || 1;
  const bw = pw / hist.length;
  const y = v => m.t + ph - (v / maxN) * ph;

  el("line", { class: "baseline", x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }, svg);

  hist.forEach((n, i) => {
    const share = n / nSims;
    const bx = m.l + i * bw + 1;         // 2px total gap between bars
    const bwid = Math.max(bw - 2, 1);
    const h = m.t + ph - y(n);
    if (h > 0.5) {
      el("path", { class: "bar", d: barPath(bx, y(n), bwid, h, 4), fill: "var(--s1)" }, svg);
    }
    const t = el("text", { class: "tick-label", x: bx + bwid / 2, y: H - m.b + 17, "text-anchor": "middle" }, svg);
    t.textContent = i + 1;
    const hit = el("rect", { class: "hit", x: bx - 1, y: m.t, width: bw, height: ph }, svg);
    hit.addEventListener("mousemove", e => showTip(
      `<span class="k">Finish</span> <b>P${i + 1}</b><br>
       <span class="k">Chance</span> <b>${(share * 100).toFixed(1)}%</b><br>
       <span class="k">Of</span> <b>${nSims}</b> simulated races`, e));
    hit.addEventListener("mouseleave", hideTip);
  });

  const xl = el("text", { class: "axis-label", x: m.l + pw / 2, y: H - 4, "text-anchor": "middle" }, svg);
  xl.textContent = "finishing position";
}

window.PitWall = { drawTrackPosition, drawDegradation, drawReliability, drawHistogram };
