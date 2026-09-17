/* Pit Wall - charts and interaction.
 *
 * One grammar everywhere: channels on shared axes, a cursor that snaps to the
 * nearest datum and writes its value into a readout, amber for limits, red
 * only for a breach. Hand-built SVG so every mark follows that grammar.
 */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const DATA = JSON.parse(document.getElementById("data").textContent || "{}");
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (id) => document.getElementById(id);

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function text(parent, x, y, str, cls, anchor) {
    const t = el("text", { x, y, class: cls || "tick", "text-anchor": anchor || "start" }, parent);
    t.textContent = str;
    return t;
  }
  function svg(mount, w, h) {
    mount.innerHTML = "";
    return el("svg", { class: "chart", viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img" }, mount);
  }
  const f = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const pct = (v, d = 1) => (v == null ? "–" : (100 * v).toFixed(d) + "%");
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  function niceStep(span, target) {
    const raw = span / Math.max(target, 1);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }
  function ticks(lo, hi, target) {
    const s = niceStep(hi - lo, target);
    const out = [];
    for (let v = Math.ceil(lo / s) * s; v <= hi + s * 1e-6; v += s) out.push(+v.toFixed(10));
    return out;
  }

  /* tooltip */
  const TIP = $("tip");
  function tip(html, evt) {
    if (!TIP) return;
    TIP.innerHTML = html;
    TIP.classList.add("on");
    const r = TIP.getBoundingClientRect();
    let x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - 14;
    TIP.style.left = x + "px";
    TIP.style.top = y + "px";
  }
  const untip = () => TIP && TIP.classList.remove("on");

  /* redraw on width change only */
  function responsive(mount, draw) {
    if (!mount) return;
    let w = 0;
    const run = () => {
      const nw = Math.round(mount.clientWidth);
      if (nw && nw !== w) { w = nw; draw(nw); }
    };
    new ResizeObserver(run).observe(mount);
    run();
  }

  /* expo ease-out, the one motion curve on the page */
  const ease = (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
  function sweep(ms, onFrame, done) {
    if (REDUCED) { onFrame(1); if (done) done(); return; }
    const t0 = performance.now();
    const step = (now) => {
      const t = clamp((now - t0) / ms, 0, 1);
      onFrame(ease(t));
      if (t < 1) requestAnimationFrame(step); else if (done) done();
    };
    requestAnimationFrame(step);
  }

  /* colour theme: follows the system unless the visitor picks one */
  const root = document.documentElement;
  const themeButtons = document.querySelectorAll("[data-theme-choice]");
  function applyTheme(choice) {
    if (choice === "light" || choice === "dark") root.dataset.theme = choice;
    else delete root.dataset.theme;
    themeButtons.forEach((b) => b.setAttribute("aria-checked", String(b.dataset.themeChoice === (choice || "system"))));
    try {
      if (choice === "light" || choice === "dark") localStorage.setItem("pitwall-theme", choice);
      else localStorage.removeItem("pitwall-theme");
    } catch (e) { /* storage unavailable: the choice lasts for this page only */ }
  }
  themeButtons.forEach((b) => b.addEventListener("click", () => applyTheme(b.dataset.themeChoice)));
  applyTheme(root.dataset.theme || "system");

  /* lap progress under the header */
  const lap = document.querySelector(".lap");
  if (lap) {
    const upd = () => {
      const h = document.documentElement.scrollHeight - innerHeight;
      lap.style.setProperty("--p", (h > 0 ? (100 * scrollY) / h : 0).toFixed(2) + "%");
    };
    addEventListener("scroll", upd, { passive: true });
    upd();
  }

  /* ================================================================ charts */

  /* Track position: circuits ranked, 95% band, snapping cursor. */
  function trackPosition(mount, rows, ids, withSweep) {
    const data = rows.slice().sort((a, b) => b.value_s - a.value_s);
    const set = (r) => {
      if (!r || !ids) return;
      $(ids.name).textContent = r.name;
      $(ids.val).textContent = f(r.value_s) + " s";
      $(ids.ci).textContent = f(r.value_lo) + "–" + f(r.value_hi) + " s";
      $(ids.pass).textContent = pct(r.p_pass_per_lap) + " · " + r.n_races;
    };
    let swept = !withSweep;

    responsive(mount, (W) => {
      if (W < 560) return trackRows(mount, data, W, set);
      const H = Math.round(clamp(W * 0.52, 300, 420));
      const m = { t: 18, r: 14, b: 44, l: 40 };
      const s = svg(mount, W, H);
      const hi = Math.max(...data.map((d) => d.value_hi)) * 1.05;
      const x = scale(0, data.length - 1, m.l + 8, W - m.r - 8);
      const y = scale(0, hi, H - m.b, m.t);

      for (const v of ticks(0, hi, 5)) {
        el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, s);
        text(s, m.l - 8, y(v) + 4, f(v, 0), "tick", "end");
      }
      el("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, s);

      const g = el("g", { class: "reveal" }, s);
      const band = data.map((d, i) => `${x(i)},${y(d.value_hi)}`).concat(
        data.map((d, i) => `${x(data.length - 1 - i)},${y(data[data.length - 1 - i].value_lo)}`));
      el("polygon", { points: band.join(" "), fill: "var(--finding-band)" }, g);
      el("polyline", { points: data.map((d, i) => `${x(i)},${y(d.value_s)}`).join(" "), fill: "none", stroke: "var(--finding)", "stroke-width": 2.25, "stroke-linejoin": "round" }, g);
      data.forEach((d, i) => el("circle", { cx: x(i), cy: y(d.value_s), r: d.is_focus ? 3.6 : 2.4, fill: d.is_focus ? "var(--finding)" : "var(--pane)", stroke: "var(--finding)", "stroke-width": 1.5 }, g));

      // label the first and last circuits and the focus circuits along the axis
      const labelled = new Set([0, data.length - 1]);
      data.forEach((d, i) => d.is_focus && labelled.add(i));
      const short = (n) => n.replace("Circuit of the Americas", "COTA");
      const room = (i) => 7 * short(data[i].name).length;
      const last = data.length - 1;
      let lastX = x(0) + room(0);
      [...labelled].sort((a, b) => a - b).forEach((i) => {
        const mid = i !== 0 && i !== last;
        if (mid && (x(i) - room(i) / 2 < lastX + 12 || x(i) + room(i) / 2 > x(last) - room(last) - 12)) return;
        if (mid) lastX = x(i) + room(i) / 2;
        const anchor = i === 0 ? "start" : i === data.length - 1 ? "end" : "middle";
        text(s, x(i), H - m.b + 18, data[i].name.replace("Circuit of the Americas", "COTA"), "lab focus", anchor);
      });
      text(s, (m.l + W - m.r) / 2, H - 6, "circuits, ranked by cost", "axis-t", "middle");

      const cur = el("g", { style: "pointer-events:none" }, s);
      const cl = el("line", { class: "cursor-line", y1: m.t, y2: H - m.b }, cur);
      const cd = el("circle", { class: "cursor-dot", r: 5 }, cur);
      const place = (i) => {
        i = clamp(Math.round(i), 0, data.length - 1);
        cl.setAttribute("x1", x(i)); cl.setAttribute("x2", x(i));
        cd.setAttribute("cx", x(i)); cd.setAttribute("cy", y(data[i].value_s));
        set(data[i]);
      };

      const hit = el("rect", { class: "hit", x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, s);
      const toIndex = (evt) => {
        const b = s.getBoundingClientRect();
        const px = ((evt.clientX - b.left) / b.width) * W;
        return ((px - x(0)) / (x(1) - x(0)));
      };
      hit.addEventListener("pointermove", (e) => swept && place(toIndex(e)));
      hit.addEventListener("pointerdown", (e) => swept && place(toIndex(e)));

      if (!swept) {
        requestAnimationFrame(() => g.classList.add("on"));
        sweep(1500, (p) => place(p * (data.length - 1)), () => { swept = true; place(0); });
      } else {
        g.classList.add("on");
        place(0);
      }
    });
  }

  function trackRows(mount, data, W, set) {
    const rowH = 26, lw = 128, H = data.length * rowH + 8;
    const s = svg(mount, W, H);
    const hi = Math.max(...data.map((d) => d.value_hi));
    const x = scale(0, hi, lw, W - 44);
    data.forEach((d, i) => {
      const cy = i * rowH + rowH / 2 + 4;
      text(s, lw - 10, cy + 4, d.name.replace("Circuit of the Americas", "COTA"), d.is_focus ? "lab focus" : "lab", "end");
      el("line", { x1: x(d.value_lo), x2: x(d.value_hi), y1: cy, y2: cy, stroke: "var(--finding)", "stroke-opacity": 0.35, "stroke-width": 6, "stroke-linecap": "round" }, s);
      el("circle", { cx: x(d.value_s), cy, r: 4, fill: "var(--finding)" }, s);
      text(s, x(d.value_hi) + 8, cy + 4, f(d.value_s), "tick");
      const hit = el("rect", { class: "hit", x: 0, y: cy - rowH / 2, width: W, height: rowH }, s);
      hit.addEventListener("pointerdown", () => set(d));
    });
    set(data[0]);
  }

  /* Median wear by compound under each estimator. */
  function degMedians(mount, deg) {
    const med = (a) => {
      const s = a.filter((v) => v != null && !Number.isNaN(v)).sort((p, q) => p - q);
      return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null;
    };
    const ranks = [["SOFTEST", "Softest"], ["MIDDLE", "Middle"], ["HARDEST", "Hardest"]];
    const series = [["slope_naive", "var(--muted)", "Naive"], ["slope_twoway", "var(--sim)", "Two-way FE"], ["slope_ipcw", "var(--finding)", "+ censoring"]];
    const rows = ranks.map(([k, label]) => {
      const sub = deg.filter((d) => d.compound_rank_label === k);
      return { label, vals: series.map(([key]) => med(sub.map((d) => d[key]))) };
    });
    responsive(mount, (W) => {
      const rowH = 52, lw = 84, H = rows.length * rowH + 40;
      const s = svg(mount, W, H);
      const all = rows.flatMap((r) => r.vals);
      const lo = Math.min(...all) * 0.9, hi = Math.max(...all) * 1.06;
      const x = scale(lo, hi, lw, W - 16);
      for (const v of ticks(lo, hi, 4)) {
        el("line", { class: "gridline", x1: x(v), x2: x(v), y1: 6, y2: H - 34 }, s);
        text(s, x(v), H - 18, f(v, 3), "tick", "middle");
      }
      text(s, (lw + W) / 2, H - 2, "seconds lost per lap of tyre age", "axis-t", "middle");
      rows.forEach((r, i) => {
        const cy = 10 + i * rowH + rowH / 2;
        text(s, lw - 12, cy + 4, r.label, "lab focus", "end");
        el("line", { x1: x(Math.min(...r.vals)), x2: x(Math.max(...r.vals)), y1: cy, y2: cy, stroke: "var(--rule)", "stroke-width": 2 }, s);
        r.vals.forEach((v, j) => {
          el("circle", { cx: x(v), cy, r: j === 2 ? 6.5 : 5, fill: series[j][1], stroke: "var(--pane)", "stroke-width": 2 }, s);
          const h = el("circle", { class: "hit", cx: x(v), cy, r: 13 }, s);
          h.addEventListener("pointermove", (e) => tip(`<span class="k">${r.label} · ${series[j][2]}</span><br><b>${f(v, 4)} s/lap</b>`, e));
          h.addEventListener("pointerleave", untip);
        });
      });
    });
  }

  /* Calibration: predicted vs observed, cursor snaps to a bin. */
  function reliability(mount, rows) {
    if (!mount || !rows || !rows.length) return;
    responsive(mount, (W) => {
      const H = Math.round(clamp(W * 0.62, 260, 380));
      const m = { t: 14, r: 16, b: 42, l: 48 };
      const s = svg(mount, W, H);
      const hi = Math.max(...rows.flatMap((d) => [d.mean_predicted, d.observed_rate])) * 1.08;
      const x = scale(0, hi, m.l, W - m.r), y = scale(0, hi, H - m.b, m.t);
      for (const v of ticks(0, hi, 5)) {
        el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, s);
        text(s, m.l - 8, y(v) + 4, pct(v, 0), "tick", "end");
        text(s, x(v), H - m.b + 17, pct(v, 0), "tick", "middle");
      }
      el("line", { x1: x(0), y1: y(0), x2: x(hi), y2: y(hi), stroke: "var(--ink)", "stroke-opacity": 0.35, "stroke-width": 1.5, "stroke-dasharray": "5 5" }, s);
      el("polyline", { points: rows.map((d) => `${x(d.mean_predicted)},${y(d.observed_rate)}`).join(" "), fill: "none", stroke: "var(--finding)", "stroke-width": 2 }, s);
      rows.forEach((d) => el("circle", { cx: x(d.mean_predicted), cy: y(d.observed_rate), r: 4.5, fill: "var(--finding)", stroke: "var(--pane)", "stroke-width": 2 }, s));
      text(s, (m.l + W - m.r) / 2, H - 4, "predicted chance of completing the pass", "axis-t", "middle");
      const yl = text(s, 0, 0, "observed rate", "axis-t", "middle");
      yl.setAttribute("transform", `translate(13,${(m.t + H - m.b) / 2}) rotate(-90)`);

      const cur = el("g", { style: "pointer-events:none;opacity:0" }, s);
      const vx = el("line", { class: "cursor-line" }, cur), hy = el("line", { class: "cursor-line" }, cur);
      const dot = el("circle", { class: "cursor-dot", r: 6 }, cur);
      const hit = el("rect", { class: "hit", x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, s);
      hit.addEventListener("pointermove", (e) => {
        const b = s.getBoundingClientRect();
        const px = ((e.clientX - b.left) / b.width) * W;
        let best = rows[0], bd = 1e9;
        for (const d of rows) { const dd = Math.abs(x(d.mean_predicted) - px); if (dd < bd) { bd = dd; best = d; } }
        const cx = x(best.mean_predicted), cy = y(best.observed_rate);
        cur.style.opacity = 1;
        vx.setAttribute("x1", cx); vx.setAttribute("x2", cx); vx.setAttribute("y1", cy); vx.setAttribute("y2", H - m.b);
        hy.setAttribute("x1", m.l); hy.setAttribute("x2", cx); hy.setAttribute("y1", cy); hy.setAttribute("y2", cy);
        dot.setAttribute("cx", cx); dot.setAttribute("cy", cy);
        tip(`<span class="k">Model said</span> <b>${pct(best.mean_predicted)}</b><br><span class="k">Happened</span> <b>${pct(best.observed_rate)}</b><br><span class="k">Attempts</span> <b>${best.n.toLocaleString()}</b>`, e);
      });
      hit.addEventListener("pointerleave", () => { cur.style.opacity = 0; untip(); });
    });
  }

  /* Gate: claimed-gain distribution against the 2 s limit. */
  function gateHist(mount, g, mean, limit) {
    if (!mount || !g) return;
    responsive(mount, (W) => {
      const H = Math.round(clamp(W * 0.42, 210, 300));
      const m = { t: 22, r: 12, b: 40, l: 40 };
      const s = svg(mount, W, H);
      const ed = g.bin_edges, c = g.counts;
      const x = scale(0, ed[ed.length - 1], m.l, W - m.r);
      const y = scale(0, Math.max(...c) * 1.1, H - m.b, m.t);
      for (const v of ticks(0, Math.max(...c) * 1.1, 4)) {
        el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, s);
        text(s, m.l - 8, y(v) + 4, f(v, 0), "tick", "end");
      }
      c.forEach((n, i) => {
        const x0 = x(ed[i]) + 1, x1 = x(ed[i + 1]) - 1, top = y(n);
        el("rect", { x: x0, y: top, width: Math.max(x1 - x0, 1), height: H - m.b - top, fill: "var(--ink)", "fill-opacity": 0.82, rx: 1.5 }, s);
        const h = el("rect", { class: "hit", x: x0 - 1, y: m.t, width: x1 - x0 + 2, height: H - m.t - m.b }, s);
        h.addEventListener("pointermove", (e) => tip(`<span class="k">Claimed gain</span> <b>${ed[i]}–${ed[i + 1]} s</b><br><span class="k">Car-races</span> <b>${n}</b>`, e));
        h.addEventListener("pointerleave", untip);
      });
      el("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, s);
      for (const v of ticks(0, ed[ed.length - 1], 6)) text(s, x(v), H - m.b + 17, f(v, 0), "tick", "middle");
      text(s, (m.l + W - m.r) / 2, H - 4, "seconds the optimiser claims to gain per car-race", "axis-t", "middle");
      el("line", { class: "limit-line", x1: x(limit), x2: x(limit), y1: m.t - 8, y2: H - m.b }, s);
      text(s, x(limit) + 5, m.t - 10, `limit ${f(limit)} s`, "limit-t");
      el("line", { x1: x(mean), x2: x(mean), y1: m.t - 8, y2: H - m.b, stroke: "var(--breach)", "stroke-width": 2 }, s);
      text(s, x(mean) + 5, m.t + 4, `mean ${f(mean)} s`, "limit-t").setAttribute("style", "fill:var(--breach)");
    });
  }

  /* Forest: real races vs simulator for one strategy moment. */
  function forest(mount, real, sim, title, unit, digits) {
    if (!mount || !real || !sim) return;
    responsive(mount, (W) => {
      const H = 128, lw = Math.min(150, W * 0.34), m = { t: 30, r: 18, b: 26 };
      const s = svg(mount, W, H);
      const lo = Math.min(0, real.ci_lo, sim.ci_lo), hi = Math.max(0, real.ci_hi, sim.ci_hi);
      const pad = (hi - lo) * 0.08;
      const x = scale(lo - pad, hi + pad, lw, W - m.r);
      text(s, 0, 16, title, "lab focus");
      for (const v of ticks(lo - pad, hi + pad, 5)) {
        el("line", { class: v === 0 ? "axis" : "gridline", x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, s);
        text(s, x(v), H - 8, (v > 0 ? "+" : "") + f(v, 0), "tick", "middle");
      }
      [["Real races", real, "var(--real)"], ["Simulator", sim, "var(--sim)"]].forEach(([label, d, col], i) => {
        const cy = m.t + 18 + i * 34;
        text(s, lw - 12, cy + 4, label, "lab", "end");
        el("line", { x1: x(d.ci_lo), x2: x(d.ci_hi), y1: cy, y2: cy, stroke: col, "stroke-width": 3, "stroke-linecap": "round", "stroke-opacity": 0.45 }, s);
        el("circle", { cx: x(d.estimate), cy, r: 6, fill: col, stroke: "var(--pane)", "stroke-width": 2 }, s);
        const h = el("rect", { class: "hit", x: lw, y: cy - 14, width: W - lw, height: 28 }, s);
        h.addEventListener("pointermove", (e) => tip(`<span class="k">${label}</span><br><b>${d.estimate > 0 ? "+" : ""}${f(d.estimate, digits)} ${unit}</b><br><span class="k">95% CI</span> <b>${f(d.ci_lo, digits)} to ${f(d.ci_hi, digits)}</b><br><span class="k">Races</span> <b>${d.n_races}</b>`, e));
        h.addEventListener("pointerleave", untip);
      });
    });
  }

  /* Horizontal ranked bars (pass detector). */
  function rankedBars(mount, rows, key, labelKey, fmtv) {
    if (!mount || !rows || !rows.length) return;
    const data = rows.slice().sort((a, b) => b[key] - a[key]);
    responsive(mount, (W) => {
      const rowH = 22, lw = Math.min(170, W * 0.4), H = data.length * rowH + 8;
      const s = svg(mount, W, H);
      const x = scale(0, Math.max(...data.map((d) => d[key])), lw, W - 48);
      data.forEach((d, i) => {
        const cy = 4 + i * rowH + rowH / 2;
        text(s, lw - 10, cy + 4, pretty(d[labelKey]), "lab", "end");
        el("rect", { x: lw, y: cy - 5, width: Math.max(x(d[key]) - lw, 1), height: 10, rx: 2, fill: "var(--ink)", "fill-opacity": 0.78 }, s);
        text(s, x(d[key]) + 8, cy + 4, fmtv(d[key]), "tick");
      });
    });
  }
  const NAMES = {};
  (DATA.profiles || []).forEach((p) => { NAMES[p.key] = p.name; });
  function pretty(k) { return NAMES[k] || String(k).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()); }

  /* ================================================================ pages */

  const page = document.body.dataset.page;

  if (page === "overview") {
    trackPosition($("tp-chart"), DATA.tp, { name: "tp-name", val: "tp-val", ci: "tp-ci", pass: "tp-pass" }, true);
    degMedians($("deg-chart"), DATA.deg);
  }
  if (page === "overview" || page === "validation") {
    reliability($("rel-chart"), DATA.rel);
    const gm = document.querySelector("#gate .readout dd");
    if (DATA.gate) gateHist($("gate-chart"), DATA.gate, parseFloat(gm ? gm.textContent : "0"), 2.0);
    if (DATA.sv) {
      forest($("why-stop"), DATA.sv.real.extra_stop_s, DATA.sv.simulated.extra_stop_s, "One extra pit stop", "s", 1);
      forest($("why-balance"), DATA.sv.real.stint_balance_s, DATA.sv.simulated.stint_balance_s, "Uneven stints (per 1.0 share)", "s", 0);
    }
  }
  if (page === "validation") {
    rankedBars($("det-chart"), DATA.detector, "mean_passes_per_race", "circuit", (v) => f(v, 1));
  }
  if (page === "circuits") circuitsPage();
  if (page === "tyres") tyresPage();
  if (page === "simulator") simulatorPage();

  /* ---------------------------------------------------------------- circuits */

  function circuitsPage() {
    const P = DATA.profiles;
    const byKey = Object.fromEntries(P.map((p) => [p.key, p]));
    const selA = $("c-a"), selB = $("c-b");
    const CH = [
      { key: "value_s", lo: "value_lo", hi: "value_hi", label: "Track position value", unit: "s", d: 1, note: "cost of being stuck behind a slower car" },
      { key: "pass_per_lap", label: "Pass chance per lap", unit: "%", d: 1, pct: true, note: "same scenario at every circuit" },
      { key: "passes_per_race", label: "Passes per race", unit: "", d: 1, note: "detected from timing data" },
      { key: "pit_loss_s", se: "pit_loss_se", label: "Pit loss", unit: "s", d: 1, note: "time lost to a stop, pooled" },
      { key: "cautions_per_race", label: "Cautions per race", unit: "", d: 2, note: "safety car and virtual safety car" },
      { key: "median_lap_s", label: "Median lap", unit: "s", d: 1, note: "green-flag race pace" },
    ];
    const box = $("channels");

    function draw() {
      const a = byKey[selA.value], b = selB.value ? byKey[selB.value] : null;
      $("k-a").textContent = a.name;
      $("k-b").textContent = b ? b.name : "";
      $("k-b-wrap").style.display = b ? "" : "none";
      box.innerHTML = "";
      CH.forEach((ch) => {
        const row = document.createElement("div");
        row.className = "channel";
        const val = (p) => (p[ch.key] == null ? "–" : ch.pct ? pct(p[ch.key]) : f(p[ch.key], ch.d) + (ch.unit ? " " + ch.unit : ""));
        row.innerHTML = `<div class="ch-l"><b>${ch.label}</b><span>${ch.note}</span></div><div class="ch-g"></div><div class="ch-v"><span class="a">${val(a)}</span>${b ? `<span class="b">${val(b)}</span>` : ""}</div>`;
        box.appendChild(row);
        const mount = row.querySelector(".ch-g");
        responsive(mount, (W) => channelSvg(mount, W, ch, a, b));
      });
      compoundChart($("c-tyres"), a, b);
      markRows();
    }

    function channelSvg(mount, W, ch, a, b) {
      const H = 46, s = svg(mount, W, H);
      const vals = P.map((p) => p[ch.key]).filter((v) => v != null);
      const lows = P.map((p) => (ch.lo ? p[ch.lo] : ch.se && p[ch.se] != null ? p[ch.key] - 1.96 * p[ch.se] : p[ch.key])).filter((v) => v != null);
      const highs = P.map((p) => (ch.hi ? p[ch.hi] : ch.se && p[ch.se] != null ? p[ch.key] + 1.96 * p[ch.se] : p[ch.key])).filter((v) => v != null);
      const lo = Math.min(...vals, ...lows), hi = Math.max(...vals, ...highs);
      const pad = (hi - lo) * 0.04 || 1;
      const x = scale(lo - pad, hi + pad, 8, W - 8);
      const cy = 20;
      el("line", { x1: 8, x2: W - 8, y1: cy, y2: cy, class: "axis" }, s);
      [lo, hi].forEach((v, i) => text(s, x(v), H - 4, ch.pct ? pct(v, 0) : f(v, ch.d), "tick", i ? "end" : "start"));
      P.forEach((p) => p[ch.key] != null && el("line", { x1: x(p[ch.key]), x2: x(p[ch.key]), y1: cy - 6, y2: cy + 6, stroke: "var(--other)", "stroke-width": 1.5 }, s));
      [[b, "var(--sim)"], [a, "var(--finding)"]].forEach(([p, col]) => {
        if (!p || p[ch.key] == null) return;
        const l = ch.lo ? p[ch.lo] : ch.se && p[ch.se] != null ? p[ch.key] - 1.96 * p[ch.se] : null;
        const h = ch.hi ? p[ch.hi] : ch.se && p[ch.se] != null ? p[ch.key] + 1.96 * p[ch.se] : null;
        if (l != null) el("line", { x1: x(l), x2: x(h), y1: cy, y2: cy, stroke: col, "stroke-width": 7, "stroke-opacity": 0.28, "stroke-linecap": "round" }, s);
        el("circle", { cx: x(p[ch.key]), cy, r: 6.5, fill: col, stroke: "var(--pane)", "stroke-width": 2 }, s);
      });
      const hit = el("rect", { class: "hit", x: 0, y: 0, width: W, height: H }, s);
      hit.addEventListener("pointermove", (e) => {
        const bb = s.getBoundingClientRect();
        const px = ((e.clientX - bb.left) / bb.width) * W;
        let best = null, bd = 1e9;
        P.forEach((p) => { if (p[ch.key] == null) return; const dd = Math.abs(x(p[ch.key]) - px); if (dd < bd) { bd = dd; best = p; } });
        if (best) tip(`<b>${best.name}</b><br><span class="k">${ch.label}</span> <b>${ch.pct ? pct(best[ch.key]) : f(best[ch.key], ch.d) + " " + ch.unit}</b><br><span class="k">Races</span> <b>${best.races}</b>`, e);
      });
      hit.addEventListener("pointerleave", untip);
      hit.addEventListener("click", (e) => {
        const bb = s.getBoundingClientRect();
        const px = ((e.clientX - bb.left) / bb.width) * W;
        let best = null, bd = 1e9;
        P.forEach((p) => { if (p[ch.key] == null) return; const dd = Math.abs(x(p[ch.key]) - px); if (dd < bd) { bd = dd; best = p; } });
        if (best && best.key !== selA.value) { selA.value = best.key; draw(); }
      });
    }

    function compoundChart(mount, a, b) {
      responsive(mount, (W) => {
        const sides = b ? [[a, "var(--finding)"], [b, "var(--sim)"]] : [[a, "var(--finding)"]];
        const rowH = b ? 50 : 36, lw = 92, H = 3 * rowH + 36;
        const s = svg(mount, W, H);
        const vals = [];
        sides.forEach(([p]) => p.compounds.forEach((c) => c.estimated && vals.push(c.sim - 1.96 * (c.se || 0), c.sim + 1.96 * (c.se || 0))));
        const lo = Math.min(0, ...vals), hi = Math.max(...vals, 0.02) * 1.05;
        const x = scale(lo, hi, lw, W - 110);
        for (const v of ticks(lo, hi, 4)) {
          el("line", { class: v === 0 ? "axis" : "gridline", x1: x(v), x2: x(v), y1: 6, y2: H - 30 }, s);
          text(s, x(v), H - 14, f(v, 2), "tick", "middle");
        }
        ["Softest", "Middle", "Hardest"].forEach((label, i) => {
          const top = 8 + i * rowH;
          text(s, lw - 12, top + rowH / 2 + 4, label, "lab focus", "end");
          sides.forEach(([p, col], j) => {
            const c = p.compounds[i];
            const cy = top + (b ? 14 + j * 22 : rowH / 2);
            if (!c.estimated) { text(s, lw, cy + 4, "not estimable", "tick"); return; }
            if (c.se != null) el("line", { x1: x(c.sim - 1.96 * c.se), x2: x(c.sim + 1.96 * c.se), y1: cy, y2: cy, stroke: col, "stroke-width": 6, "stroke-opacity": 0.28, "stroke-linecap": "round" }, s);
            el("circle", { cx: x(c.sim), cy, r: 5.5, fill: c.sim < 0 ? "var(--breach)" : col, stroke: "var(--pane)", "stroke-width": 2 }, s);
            text(s, W - 104, cy + 4, `${f(c.sim, 3)}${c.max_stint ? " · ≤" + c.max_stint + " laps" : ""}`, "tick");
          });
        });
        text(s, (lw + W - 110) / 2, H - 1, "seconds lost per lap of tyre age", "axis-t", "middle");
      });
    }

    /* sortable table */
    const tb = document.querySelector("#c-table tbody");
    let sortKey = "value_s", dir = -1;
    function renderTable() {
      const rows = P.slice().sort((p, q) => {
        const u = p[sortKey], v = q[sortKey];
        if (typeof u === "string") return dir * u.localeCompare(v);
        return dir * ((u == null ? -1e9 : u) - (v == null ? -1e9 : v));
      });
      tb.innerHTML = rows.map((p) => `<tr data-key="${p.key}" tabindex="0"><td>${p.name}</td><td class="n">${p.races}</td><td class="n">${f(p.value_s)} <span class="dim">${f(p.value_lo)}–${f(p.value_hi)}</span></td><td class="n">${pct(p.pass_per_lap)}</td><td class="n">${f(p.passes_per_race)}</td><td class="n">${f(p.pit_loss_s)}</td><td class="n">${f(p.cautions_per_race, 2)}</td></tr>`).join("");
      markRows();
    }
    function markRows() {
      tb.querySelectorAll("tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.key === selA.value || tr.dataset.key === selB.value));
    }
    tb.addEventListener("click", (e) => {
      const tr = e.target.closest("tr");
      if (tr) { selA.value = tr.dataset.key; draw(); window.scrollTo({ top: 0, behavior: REDUCED ? "auto" : "smooth" }); }
    });
    tb.addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.closest("tr").click(); });
    document.querySelectorAll("#c-table th").forEach((th) => {
      th.querySelector("button").addEventListener("click", () => {
        const k = th.dataset.k;
        dir = sortKey === k ? -dir : k === "name" ? 1 : -1;
        sortKey = k;
        document.querySelectorAll("#c-table th").forEach((o) => o.setAttribute("aria-sort", "none"));
        th.setAttribute("aria-sort", dir > 0 ? "ascending" : "descending");
        renderTable();
      });
    });

    selA.addEventListener("change", draw);
    selB.addEventListener("change", draw);
    renderTable();
    draw();
  }

  /* ---------------------------------------------------------------- tyres */

  function tyresPage() {
    const D = DATA.deg;
    const KEY = { naive: "slope_naive", twoway: "slope_twoway", ipcw: "slope_ipcw", sim: "slope_s_per_lap_sim" };
    const ranks = ["SOFTEST", "MIDDLE", "HARDEST"];
    const colour = { SOFTEST: "var(--soft)", MIDDLE: "var(--middle)", HARDEST: "var(--hard)" };
    const circuits = [...new Set(D.map((d) => d.circuit))].map((c) => {
      const cells = {};
      D.filter((d) => d.circuit === c).forEach((d) => { cells[d.compound_rank_label] = d; });
      return { key: c, name: D.find((d) => d.circuit === c).name, cells };
    });
    let est = "sim";
    const mount = $("ty-chart");

    function summary() {
      const k = KEY[est];
      const med = (a) => { const s = a.filter((v) => v != null).sort((p, q) => p - q); return s.length ? s[Math.floor((s.length - 1) / 2)] : null; };
      const full = circuits.filter((c) => ranks.every((r) => c.cells[r]));
      const ordered = full.filter((c) => c.cells.SOFTEST[k] > c.cells.HARDEST[k]).length;
      $("o-order").textContent = `${ordered} / ${full.length}`;
      const neg = D.filter((d) => d[k] != null && d[k] < 0).length;
      const on = $("o-neg"); on.textContent = `${neg} / ${D.length}`; on.className = neg ? "bad" : "good";
      $("o-soft").textContent = f(med(D.filter((d) => d.compound_rank_label === "SOFTEST").map((d) => d[k])), 4);
      $("o-hard").textContent = f(med(D.filter((d) => d.compound_rank_label === "HARDEST").map((d) => d[k])), 4);
    }

    function order() {
      const k = KEY[est], mode = $("est-sort").value;
      const v = (c, r) => (c.cells[r] ? c.cells[r][k] : null);
      return circuits.slice().sort((a, b) => {
        if (mode === "name") return a.name.localeCompare(b.name);
        if (mode === "spread") return ((v(b, "SOFTEST") ?? -1) - (v(b, "HARDEST") ?? 0)) - ((v(a, "SOFTEST") ?? -1) - (v(a, "HARDEST") ?? 0));
        return (v(b, "SOFTEST") ?? -1) - (v(a, "SOFTEST") ?? -1);
      });
    }

    function draw() {
      summary();
      responsive(mount, (W) => {
        const k = KEY[est], withCI = est === "ipcw" || est === "sim";
        const data = order();
        const rowH = 24, lw = Math.min(170, W * 0.36), top = 8, H = data.length * rowH + top + 34;
        const s = svg(mount, W, H);
        const vals = D.map((d) => d[k]).filter((v) => v != null);
        const ext = withCI ? D.flatMap((d) => [d[k] - 1.96 * (d.se_ipcw_race_clustered || 0), d[k] + 1.96 * (d.se_ipcw_race_clustered || 0)]) : vals;
        const lo = Math.min(0, ...ext), hi = Math.max(...ext) * 1.03;
        const x = scale(lo, hi, lw, W - 12);
        for (const v of ticks(lo, hi, 6)) {
          el("line", { class: v === 0 ? "axis" : "gridline", x1: x(v), x2: x(v), y1: top - 4, y2: H - 30 }, s);
          text(s, x(v), H - 14, f(v, 2), "tick", "middle");
        }
        text(s, (lw + W) / 2, H - 1, "seconds lost per lap of tyre age", "axis-t", "middle");
        data.forEach((c, i) => {
          const cy = top + i * rowH + rowH / 2;
          if (i % 2) el("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "var(--stripe)" }, s);
          text(s, lw - 10, cy + 4, c.name, "lab", "end");
          ranks.forEach((r, j) => {
            const d = c.cells[r];
            if (!d || d[k] == null) return;
            const oy = cy + (j - 1) * 5;
            if (withCI && d.se_ipcw_race_clustered) {
              el("line", { x1: x(d[k] - 1.96 * d.se_ipcw_race_clustered), x2: x(d[k] + 1.96 * d.se_ipcw_race_clustered), y1: oy, y2: oy, stroke: colour[r], "stroke-opacity": 0.45, "stroke-width": 1.5 }, s);
            }
            el("circle", { cx: x(d[k]), cy: oy, r: 4.2, fill: d[k] < 0 ? "var(--breach)" : colour[r], stroke: r === "HARDEST" ? "var(--hard-stroke)" : "var(--pane)", "stroke-width": r === "HARDEST" ? 1 : 1.5 }, s);
          });
          const hit = el("rect", { class: "hit", x: 0, y: cy - rowH / 2, width: W, height: rowH }, s);
          hit.addEventListener("pointermove", (e) => tip(`<b>${c.name}</b><br>` + ranks.map((r) => {
            const d = c.cells[r];
            if (!d) return `<span class="k">${r.toLowerCase()}</span> <b>–</b>`;
            const ci = withCI && d.se_ipcw_race_clustered ? ` ± ${f(1.96 * d.se_ipcw_race_clustered, 3)}` : "";
            return `<span class="k">${r.charAt(0) + r.slice(1).toLowerCase()}</span> <b>${f(d[k], 4)}${ci}</b>`;
          }).join("<br>") + `<br><span class="k">Races</span> <b>${Math.max(...ranks.map((r) => (c.cells[r] ? c.cells[r].n_races || 0 : 0)))}</b>`, e));
          hit.addEventListener("pointerleave", untip);
        });
      });
    }

    const seg = $("est-seg");
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      est = b.dataset.est;
      seg.querySelectorAll("button").forEach((o) => o.setAttribute("aria-checked", o === b ? "true" : "false"));
      mount.innerHTML = "";
      draw();
    });
    $("est-sort").addEventListener("change", () => { mount.innerHTML = ""; draw(); });
    draw();
  }

  /* ---------------------------------------------------------------- simulator */

  function simulatorPage() {
    const sel = $("sim-circuit"), startSel = $("sim-start"), grid = $("sim-grid");
    const strip = $("stint"), list = $("stints-list");
    const RANK = ["Softest", "Middle", "Hardest"];
    const FILL = ["var(--soft)", "var(--middle)", "var(--hard)"];
    const INK = ["var(--on-soft)", "var(--on-middle)", "var(--on-hard)"];
    let laps = 53, stops = [[26, 2]], circuit = null;

    async function loadCircuit() {
      clearErr();
      try {
        const r = await fetch(`/api/circuit/${encodeURIComponent(sel.value)}`);
        if (!r.ok) throw new Error((await r.json()).detail || "could not load circuit");
        circuit = await r.json();
        laps = circuit.race_laps;
        $("f-laps").textContent = laps;
        $("f-pit").textContent = f(circuit.pit_loss_s) + " s";
        $("f-wear").textContent = [0, 1, 2].map((k) => f(circuit.deg_by_rank[String(k)], 3)).join(" · ");
        $("f-sc").textContent = f(circuit.caution_hazard_per_lap * laps, 2);
        stops = [[Math.round(laps / 2), 2]];
        render();
      } catch (e) { showErr(e.message); }
    }

    function stintsFromStops() {
      const edges = [0, ...stops.map((s) => s[0]), laps];
      const ranks = [+startSel.value, ...stops.map((s) => s[1])];
      return ranks.map((r, i) => ({ from: edges[i], to: edges[i + 1], rank: r }));
    }

    function render() {
      stops.sort((a, b) => a[0] - b[0]);
      drawStrip();
      list.innerHTML = "";
      stintsFromStops().forEach((st, i) => {
        const pill = document.createElement("span");
        pill.className = "stint-pill";
        pill.innerHTML = `<span class="swatch" style="background:${FILL[st.rank]}"></span><span class="num">L${st.from + 1}–${st.to}</span>`;
        const s = document.createElement("select");
        s.setAttribute("aria-label", `Tyre for stint ${i + 1}`);
        RANK.forEach((lab, k) => { const o = document.createElement("option"); o.value = k; o.textContent = lab; if (k === st.rank) o.selected = true; s.appendChild(o); });
        s.addEventListener("change", () => {
          if (i === 0) startSel.value = s.value; else stops[i - 1][1] = +s.value;
          render();
        });
        pill.appendChild(s);
        if (i > 0) {
          const rm = document.createElement("button");
          rm.type = "button"; rm.textContent = "Remove stop"; rm.setAttribute("aria-label", `Remove the stop on lap ${stops[i - 1][0]}`);
          rm.addEventListener("click", () => { stops.splice(i - 1, 1); render(); });
          pill.appendChild(rm);
        }
        list.appendChild(pill);
      });
    }

    function drawStrip() {
      const W = Math.max(strip.clientWidth, 280), H = 74, m = { l: 8, r: 8, t: 10 };
      const s = svg(strip, W, H);
      const x = scale(0, laps, m.l, W - m.r);
      const barY = m.t, barH = 34;
      stintsFromStops().forEach((st) => {
        el("rect", { x: x(st.from) + 1, y: barY, width: Math.max(x(st.to) - x(st.from) - 2, 1), height: barH, rx: 6, fill: FILL[st.rank], stroke: st.rank === 2 ? "var(--hard-stroke)" : "none" }, s);
        if (x(st.to) - x(st.from) > 58) {
          const t = text(s, (x(st.from) + x(st.to)) / 2, barY + barH / 2 + 4, `${RANK[st.rank].toUpperCase()} · ${st.to - st.from}`, "seg-lab", "middle");
          t.setAttribute("fill", INK[st.rank]);
        }
      });
      for (const v of ticks(0, laps, W < 480 ? 5 : 10)) text(s, x(v), H - 8, v === 0 ? "lap 1" : String(v), "tick", v === 0 ? "start" : "middle");

      const bg = el("rect", { x: m.l, y: barY, width: W - m.l - m.r, height: barH, fill: "transparent", style: "cursor:copy" }, s);
      bg.addEventListener("click", (e) => {
        const lapAt = Math.round(toLap(e, s, W, x));
        if (lapAt < 2 || lapAt > laps - 2 || stops.some((st) => Math.abs(st[0] - lapAt) < 2)) return;
        stops.push([lapAt, 1]);
        render();
      });

      stops.forEach((st, i) => {
        const g = el("g", { class: "handle", tabindex: 0, role: "slider", "aria-label": `Stop ${i + 1}`, "aria-valuemin": 1, "aria-valuemax": laps - 1, "aria-valuenow": st[0], "aria-valuetext": `lap ${st[0]}` }, s);
        el("line", { x1: x(st[0]), x2: x(st[0]), y1: barY - 6, y2: barY + barH + 6, stroke: "var(--ink)", "stroke-width": 2 }, g);
        el("rect", { class: "grip", x: x(st[0]) - 7, y: barY - 8, width: 14, height: 14, rx: 4 }, g);
        el("rect", { x: x(st[0]) - 16, y: barY - 10, width: 32, height: barH + 20, fill: "transparent" }, g);
        g.addEventListener("pointerdown", (e) => {
          e.preventDefault();
          g.setPointerCapture(e.pointerId);
          const move = (ev) => { st[0] = clamp(Math.round(toLap(ev, s, W, x)), 1, laps - 1); drawStrip(); };
          const up = () => { g.removeEventListener("pointermove", move); render(); };
          g.addEventListener("pointermove", move);
          g.addEventListener("pointerup", up, { once: true });
          g.addEventListener("pointercancel", up, { once: true });
        });
        g.addEventListener("keydown", (e) => {
          const step = e.shiftKey ? 5 : 1;
          if (e.key === "ArrowLeft" || e.key === "ArrowDown") st[0] = clamp(st[0] - step, 1, laps - 1);
          else if (e.key === "ArrowRight" || e.key === "ArrowUp") st[0] = clamp(st[0] + step, 1, laps - 1);
          else if (e.key === "Delete" || e.key === "Backspace") stops.splice(i, 1);
          else return;
          e.preventDefault();
          render();
          const again = strip.querySelectorAll(".handle")[Math.min(i, stops.length - 1)];
          if (again) again.focus();
        });
      });
    }
    function toLap(evt, s, W, x) {
      const b = s.getBoundingClientRect();
      const px = ((evt.clientX - b.left) / b.width) * W;
      return ((px - x(0)) / (x(laps) - x(0))) * laps;
    }

    async function run() {
      const btn = $("sim-run");
      clearErr();
      const g = +grid.value;
      if (!(g >= 1 && g <= 12)) { showErr("Start position must be between 1 and 12."); grid.focus(); return; }
      btn.disabled = true; btn.textContent = "Simulating…";
      try {
        const r = await fetch("/api/simulate", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ circuit: sel.value, stops, start_rank: +startSel.value, grid_position: g, n_sims: 600, field_size: 12 }),
        });
        const d = await r.json();
        if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : "The simulation request was rejected.");
        $("sim-results").hidden = false;
        $("sim-empty").hidden = true;
        $("r-mean").textContent = "P" + f(d.mean_position);
        $("r-range").textContent = `P${f(d.p05_position, 0)}–P${f(d.p95_position, 0)}`;
        $("r-win").textContent = `${pct(d.win_rate)} · ${pct(d.podium_rate, 0)}`;
        $("r-gap").textContent = f(d.mean_gap_s) + " s";
        histogram($("sim-chart"), d.position_histogram, d.n_sims, d.mean_position);
      } catch (e) {
        showErr(e.message + " Try again, or pick another circuit.");
      } finally {
        btn.disabled = false; btn.textContent = "Simulate 600 races";
      }
    }

    function histogram(mount, hist, n, mean) {
      responsive(mount, (W) => {
        const H = 220, m = { t: 20, r: 10, b: 38, l: 36 };
        const s = svg(mount, W, H);
        const x = scale(0, hist.length, m.l, W - m.r), y = scale(0, Math.max(...hist) * 1.12 / n, H - m.b, m.t);
        for (const v of ticks(0, Math.max(...hist) * 1.12 / n, 4)) {
          el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, s);
          text(s, m.l - 6, y(v) + 4, pct(v, 0), "tick", "end");
        }
        hist.forEach((c, i) => {
          const h0 = y(c / n);
          const bar = el("rect", { x: x(i) + 2, y: H - m.b, width: Math.max(x(1) - x(0) - 4, 1), height: 0, rx: 2, fill: "var(--ink)", "fill-opacity": 0.85 }, s);
          sweep(700, (p) => { bar.setAttribute("y", H - m.b - p * (H - m.b - h0)); bar.setAttribute("height", p * (H - m.b - h0)); });
          text(s, x(i + 0.5), H - m.b + 16, "P" + (i + 1), "tick", "middle");
          const hit = el("rect", { class: "hit", x: x(i), y: m.t, width: x(1) - x(0), height: H - m.t - m.b }, s);
          hit.addEventListener("pointermove", (e) => tip(`<span class="k">Finish</span> <b>P${i + 1}</b><br><span class="k">Chance</span> <b>${pct(c / n)}</b>`, e));
          hit.addEventListener("pointerleave", untip);
        });
        const mx = x(mean - 0.5);
        el("line", { x1: mx, x2: mx, y1: m.t - 6, y2: H - m.b, stroke: "var(--finding)", "stroke-width": 2 }, s);
        text(s, mx + 5, m.t, `mean P${f(mean)}`, "limit-t").setAttribute("style", "fill:var(--finding-ink)");
        text(s, (m.l + W - m.r) / 2, H - 2, `finishing position across ${n} simulated races`, "axis-t", "middle");
      });
    }

    async function best() {
      const btn = $("sim-best");
      clearErr();
      btn.disabled = true; btn.textContent = "Searching…";
      try {
        const r = await fetch("/api/optimise", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ circuit: sel.value, grid_position: +grid.value || 5, field_size: 12, n_sims: 400, stops: [] }),
        });
        const d = await r.json();
        if (!r.ok || !d.candidates || !d.candidates.length) throw new Error("The optimiser found no legal plan for this circuit.");
        stops = d.candidates[0].stops.map((s) => [s[0], s[1]]);
        startSel.value = String(d.candidates[0].start_rank);
        render();
        run();
      } catch (e) {
        showErr(e.message);
      } finally {
        btn.disabled = false; btn.textContent = "Load the optimiser's plan";
      }
    }

    function showErr(msg) { const e = $("sim-err"); e.textContent = msg; e.hidden = false; }
    function clearErr() { $("sim-err").hidden = true; }

    sel.addEventListener("change", loadCircuit);
    startSel.addEventListener("change", render);
    $("sim-run").addEventListener("click", run);
    $("sim-best").addEventListener("click", best);
    new ResizeObserver(() => drawStrip()).observe(strip);
    loadCircuit();
  }
})();
