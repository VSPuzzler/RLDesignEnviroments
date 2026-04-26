/* Shared utilities for NeuroUI Judge dashboards. */

window.NUI = (function () {
  const ns = "http://www.w3.org/2000/svg";

  function el(tag, attrs = {}, parent = null) {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      e.setAttribute(k, v);
    }
    if (parent) parent.appendChild(e);
    return e;
  }

  function clamp(x, lo = 0, hi = 1) {
    return Math.max(lo, Math.min(hi, x));
  }

  function colorRamp(t) {
    // Magma-ish ramp from deep blue → fuchsia → amber.
    const stops = [
      [0.00, [15, 23, 42]],
      [0.20, [59, 130, 246]],
      [0.40, [124, 92, 255]],
      [0.60, [217, 70, 239]],
      [0.80, [245, 158, 11]],
      [1.00, [254, 240, 138]],
    ];
    t = clamp(t);
    for (let i = 1; i < stops.length; i++) {
      const [a, ca] = stops[i - 1];
      const [b, cb] = stops[i];
      if (t <= b) {
        const u = (t - a) / (b - a);
        const rgb = ca.map((c, j) => Math.round(c + (cb[j] - c) * u));
        return `rgb(${rgb.join(",")})`;
      }
    }
    return `rgb(${stops[stops.length - 1][1].join(",")})`;
  }

  function tooltipHandler(host) {
    const tip = document.createElement("div");
    tip.className = "tooltip";
    host.appendChild(tip);
    return {
      show(text, x, y) {
        tip.textContent = text;
        tip.style.left = `${x}px`;
        tip.style.top = `${y}px`;
        tip.classList.add("show");
      },
      hide() {
        tip.classList.remove("show");
      },
    };
  }

  return { el, clamp, colorRamp, tooltipHandler };
})();
