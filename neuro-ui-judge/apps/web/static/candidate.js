/* Per-candidate dashboard charts and overlays. */

(function () {
  const report = window.__REPORT__;
  const candidateId = window.__CANDIDATE_ID__;
  const E = NUI.el;

  // ── Radar chart of sub-scores ────────────────────────────────────────────
  function renderRadar(host, subscores) {
    host.innerHTML = "";
    const W = host.clientWidth || 320;
    const H = 320;
    const cx = W / 2;
    const cy = H / 2;
    const R = Math.min(W, H) * 0.36;
    const svg = E("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    host.appendChild(svg);

    const keys = Object.keys(subscores);
    const angle = (i) => (-Math.PI / 2) + (2 * Math.PI * i) / keys.length;

    // Concentric grid
    for (let g = 1; g <= 4; g++) {
      const r = (R * g) / 4;
      const pts = keys.map((_, i) => {
        const a = angle(i);
        return `${cx + Math.cos(a) * r},${cy + Math.sin(a) * r}`;
      }).join(" ");
      E("polygon", { points: pts, fill: "none", stroke: "#1e293b", "stroke-width": 1 }, svg);
    }
    // Axes + labels
    keys.forEach((k, i) => {
      const a = angle(i);
      const x = cx + Math.cos(a) * R;
      const y = cy + Math.sin(a) * R;
      E("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: "#1e293b" }, svg);
      const lx = cx + Math.cos(a) * (R + 18);
      const ly = cy + Math.sin(a) * (R + 18);
      const text = E("text", {
        x: lx, y: ly, "font-size": 10, fill: "#94a3b8",
        "text-anchor": Math.cos(a) > 0.2 ? "start" : Math.cos(a) < -0.2 ? "end" : "middle",
        "dominant-baseline": "middle",
      }, svg);
      text.textContent = k.replace(/_/g, " ");
    });
    // Polygon
    const pts = keys.map((k, i) => {
      const v = NUI.clamp(subscores[k]);
      const a = angle(i);
      return `${cx + Math.cos(a) * R * v},${cy + Math.sin(a) * R * v}`;
    }).join(" ");
    E("polygon", { points: pts, fill: "rgba(124,92,255,0.30)", stroke: "#7c5cff", "stroke-width": 2 }, svg);
    // Dots
    keys.forEach((k, i) => {
      const v = NUI.clamp(subscores[k]);
      const a = angle(i);
      const x = cx + Math.cos(a) * R * v;
      const y = cy + Math.sin(a) * R * v;
      E("circle", { cx: x, cy: y, r: 3, fill: "#a48dff" }, svg);
    });
  }

  // ── Sub-score bars ───────────────────────────────────────────────────────
  function renderBars(host, subscores) {
    host.innerHTML = "";
    const W = host.clientWidth || 480;
    const rowH = 30;
    const padL = 160;
    const padR = 50;
    const keys = Object.keys(subscores);
    const H = keys.length * rowH + 12;
    const svg = E("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    host.appendChild(svg);

    const defs = E("defs", {}, svg);
    const grad = E("linearGradient", { id: "barGradC", x1: 0, x2: 1, y1: 0, y2: 0 }, defs);
    E("stop", { offset: "0%", "stop-color": "#7c5cff" }, grad);
    E("stop", { offset: "100%", "stop-color": "#d946ef" }, grad);

    const tip = NUI.tooltipHandler(host);

    keys.forEach((k, i) => {
      const y = 4 + i * rowH;
      E("text", { x: 0, y: y + rowH / 2, "font-size": 11, fill: "#cbd5e1", "dominant-baseline": "middle" }, svg).textContent = k.replace(/_/g, " ");
      E("rect", { x: padL, y: y + 8, width: W - padL - padR, height: 8, rx: 4, fill: "#1e293b" }, svg);
      const v = NUI.clamp(subscores[k]);
      const r = E("rect", {
        x: padL, y: y + 8, width: (W - padL - padR) * v, height: 8, rx: 4,
        fill: "url(#barGradC)",
      }, svg);
      r.addEventListener("mousemove", (ev) => {
        const r2 = host.getBoundingClientRect();
        tip.show(`${k}: ${v.toFixed(3)}`, ev.clientX - r2.left, ev.clientY - r2.top);
      });
      r.addEventListener("mouseleave", () => tip.hide());
      E("text", { x: W - padR + 4, y: y + rowH / 2, "font-size": 11, fill: "#e2e8f0", "dominant-baseline": "middle" }, svg).textContent = v.toFixed(2);
    });
  }

  // ── ROI time-series (BOLD-like) ──────────────────────────────────────────
  async function renderTimeSeries(host) {
    host.innerHTML = "Loading…";
    const data = await fetch(`/api/timeseries/${candidateId}`).then((r) => r.json());
    host.innerHTML = "";
    const series = data.series;
    const W = host.clientWidth || 600;
    const H = 280;
    const padL = 50, padR = 12, padT = 18, padB = 28;
    const svg = E("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    host.appendChild(svg);

    const keys = Object.keys(series);
    const colors = {
      visual: "#3b82f6",
      dorsal_attention: "#10b981",
      salience: "#f59e0b",
      multiple_demand: "#ef4444",
      language_vwfa: "#7c5cff",
      dmn: "#94a3b8",
      valuation_proxy: "#d946ef",
    };

    // Compute global max for shared y-axis.
    let maxY = 0;
    keys.forEach((k) => series[k].forEach((v) => { if (v > maxY) maxY = v; }));
    if (maxY === 0) maxY = 1;
    const n = series[keys[0]].length;

    const x = (i) => padL + (i / Math.max(1, n - 1)) * (W - padL - padR);
    const y = (v) => padT + (1 - v / maxY) * (H - padT - padB);

    // Gridlines + y-ticks
    for (let g = 0; g <= 4; g++) {
      const yy = padT + ((H - padT - padB) * g) / 4;
      E("line", { x1: padL, y1: yy, x2: W - padR, y2: yy, stroke: "#1e293b", "stroke-dasharray": "2 3" }, svg);
      const t = E("text", { x: padL - 8, y: yy + 3, "font-size": 9, fill: "#64748b", "text-anchor": "end" }, svg);
      t.textContent = ((maxY * (4 - g)) / 4).toFixed(2);
    }
    E("text", { x: padL - 36, y: padT - 4, "font-size": 10, fill: "#94a3b8" }, svg).textContent = "predicted activation (a.u.)";
    E("text", { x: W - padR, y: H - 6, "font-size": 10, fill: "#94a3b8", "text-anchor": "end" }, svg).textContent = "time →";

    keys.forEach((k) => {
      const path = series[k].map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(v)}`).join(" ");
      E("path", { d: path, fill: "none", stroke: colors[k] || "#cbd5e1", "stroke-width": 2, "stroke-linecap": "round" }, svg);
    });

    // Legend
    const legend = E("g", { transform: `translate(${padL + 6}, ${padT + 4})` }, svg);
    keys.forEach((k, i) => {
      const lx = (i % 4) * 150;
      const ly = Math.floor(i / 4) * 16;
      E("rect", { x: lx, y: ly, width: 10, height: 3, fill: colors[k] || "#cbd5e1" }, legend);
      const t = E("text", { x: lx + 14, y: ly + 4, "font-size": 10, fill: "#cbd5e1" }, legend);
      t.textContent = k;
    });
  }

  // ── Cortical-style heatmap ───────────────────────────────────────────────
  // Stylised flat-cortex representation: two hemispheres, each a smooth
  // outline with anatomically-suggestive ROI patches lit by the predicted
  // AUC values. Not anatomically accurate; clearly framed as illustrative.
  function renderCortical(host, neural) {
    host.innerHTML = "";
    const W = host.clientWidth || 380;
    const H = 280;
    const svg = E("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    host.appendChild(svg);

    const defs = E("defs", {}, svg);
    const blur = E("filter", { id: "blur", x: "-20%", y: "-20%", width: "140%", height: "140%" }, defs);
    E("feGaussianBlur", { stdDeviation: 8 }, blur);

    function hemi(cxOffset) {
      const g = E("g", { transform: `translate(${cxOffset},0)` }, svg);
      // Hemisphere outline (rounded peanut shape)
      E("path", {
        d: `M 30 140
            C 20 60, 100 30, 150 40
            C 200 50, 200 100, 195 130
            C 210 170, 170 230, 110 230
            C 50 230, 30 200, 30 140 Z`,
        fill: "#0f172a", stroke: "#1f2937", "stroke-width": 1.5,
      }, g);
      return g;
    }

    const left = hemi(0);
    const right = hemi(W / 2 - 20);

    // ROI patches (cx, cy, r, hemi) per ROI; mirrored.
    const layout = [
      { roi: "visual",          cx: 165, cy: 200, r: 26 }, // occipital
      { roi: "dorsal_attention",cx: 130, cy: 95,  r: 22 }, // dorsal frontal/parietal
      { roi: "salience",        cx: 95,  cy: 145, r: 20 }, // anterior insula-ish
      { roi: "multiple_demand", cx: 110, cy: 75,  r: 18 }, // dlPFC
      { roi: "language_vwfa",   cx: 180, cy: 165, r: 18 }, // ventral temporal
      { roi: "dmn",             cx: 150, cy: 130, r: 22 }, // medial-ish
      { roi: "valuation_proxy", cx: 75,  cy: 175, r: 18 }, // OFC-ish
    ];

    function paintHemi(g, mirror = false) {
      layout.forEach((p) => {
        const feats = neural.roi_features[p.roi] || {};
        const v = NUI.clamp(feats.auc ?? 0.5);
        const cx = mirror ? 230 - p.cx : p.cx;
        E("circle", {
          cx, cy: p.cy, r: p.r * 1.6,
          fill: NUI.colorRamp(v),
          opacity: 0.35,
          filter: "url(#blur)",
        }, g);
        E("circle", {
          cx, cy: p.cy, r: p.r,
          fill: NUI.colorRamp(v),
          opacity: 0.85,
        }, g);
        E("text", {
          x: cx, y: p.cy + 3,
          "font-size": 9, "text-anchor": "middle",
          fill: v > 0.55 ? "#0b1220" : "#e2e8f0",
        }, g).textContent = p.roi.split("_")[0];
      });
    }

    paintHemi(left);
    paintHemi(right, true);

    // Color legend
    const legend = E("g", { transform: `translate(${W - 110}, ${H - 28})` }, svg);
    for (let i = 0; i < 100; i++) {
      E("rect", {
        x: i, y: 0, width: 1, height: 8,
        fill: NUI.colorRamp(i / 99),
      }, legend);
    }
    E("text", { x: 0, y: 22, "font-size": 9, fill: "#94a3b8" }, legend).textContent = "low";
    E("text", { x: 100, y: 22, "font-size": 9, fill: "#94a3b8", "text-anchor": "end" }, legend).textContent = "high";
    const lbl = E("text", { x: 0, y: -4, "font-size": 9, fill: "#94a3b8" }, legend);
    lbl.textContent = "predicted ROI AUC";
  }

  // ── Element overlay (bboxes + attention/load heatmap) ────────────────────
  function renderOverlay(host, canvas, img) {
    const ctx = canvas.getContext("2d");
    const audit = report.deterministic_audit;
    const elements = (audit && audit.raw_features) ? null : null;
    const artElements = (window.__OVERLAY_ELEMENTS__ || []);

    function draw() {
      const W = host.clientWidth;
      const H = host.clientHeight;
      canvas.width = W;
      canvas.height = H;
      ctx.clearRect(0, 0, W, H);

      const naturalW = parseFloat(audit.raw_features.viewport_width) || 1440;
      const naturalH = parseFloat(audit.raw_features.viewport_height) || 900;
      const sx = W / naturalW;
      const sy = H / naturalH;
      const showB = document.getElementById("show-bboxes").checked;
      const showA = document.getElementById("show-attention").checked;
      const showL = document.getElementById("show-load").checked;

      for (const el of artElements) {
        const b = el.bbox || {};
        const x = b.x * sx, y = b.y * sy, w = b.width * sx, h = b.height * sy;
        if (w < 2 || h < 2) continue;

        if (showA) {
          // Attention proxy: large fonts + interactive elements + CTAs.
          const fs = el.font_size_px || 14;
          let intensity = NUI.clamp((fs - 12) / 40);
          if (el.is_interactive) intensity += 0.2;
          if (el.is_cta) intensity += 0.4;
          intensity = NUI.clamp(intensity);
          ctx.fillStyle = `rgba(245, 158, 11, ${0.10 + intensity * 0.45})`;
          ctx.fillRect(x, y, w, h);
        }
        if (showL) {
          // Load proxy: many small text elements pile up red.
          const fs = el.font_size_px || 14;
          const small = NUI.clamp((16 - fs) / 8);
          if ((el.text || "").length > 0) {
            ctx.fillStyle = `rgba(239, 68, 68, ${0.10 + small * 0.35})`;
            ctx.fillRect(x, y, w, h);
          }
        }
        if (showB) {
          ctx.strokeStyle = el.is_cta ? "#a48dff" : el.is_interactive ? "#10b981" : "rgba(148,163,184,0.45)";
          ctx.lineWidth = el.is_cta ? 2 : 1;
          ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        }
      }
    }

    draw();
    document.querySelectorAll("#show-bboxes,#show-attention,#show-load").forEach((cb) =>
      cb.addEventListener("change", draw),
    );
    new ResizeObserver(draw).observe(host);
    img.addEventListener("load", draw);
  }

  // ── Bootstrap ─────────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", async () => {
    const radarHost = document.getElementById("radar-host");
    const barsHost = document.getElementById("bars-host");
    const tsHost = document.getElementById("ts-host");
    const corticalHost = document.getElementById("cortical-host");
    if (radarHost) renderRadar(radarHost, report.subscores);
    if (barsHost) renderBars(barsHost, report.subscores);
    if (tsHost) renderTimeSeries(tsHost);
    if (corticalHost) renderCortical(corticalHost, report.neural_proxy);

    // Pull the artifact once for bbox overlay. Use the candidate's stored DOM tree.
    try {
      const r = await fetch(`/api/candidates/${candidateId}/report`).then((r) => r.json());
      // The artifact is already inside report.deterministic_audit's raw_features
      // for sizing; for elements we hit a dedicated endpoint.
      const art = await fetch(`/api/artifact/${candidateId}`).then((r) => r.ok ? r.json() : null).catch(() => null);
      if (art && art.elements) window.__OVERLAY_ELEMENTS__ = art.elements;
    } catch (_) {}

    const overlayHost = document.getElementById("overlay-host");
    const overlayCanvas = document.getElementById("overlay-canvas");
    const overlayImg = document.getElementById("overlay-img");
    if (overlayHost && overlayCanvas && overlayImg) {
      renderOverlay(overlayHost, overlayCanvas, overlayImg);
    }
  });
})();
