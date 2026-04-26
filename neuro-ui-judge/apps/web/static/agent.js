/* Agent-run reward curve. */
(function () {
  const run = window.__RUN__;
  if (!run) return;
  const E = NUI.el;

  const host = document.getElementById("agent-curve");
  if (!host) return;

  const W = host.clientWidth || 600;
  const H = 260;
  const padL = 50, padR = 20, padT = 20, padB = 36;
  const svg = E("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  host.appendChild(svg);

  const its = run.iterations || [];
  if (its.length === 0) return;

  const reports = run.reports || {};
  const xs = its.map((it) => it.iteration);
  // Aggregate per-iteration: best and mean of children rewards.
  const series = its.map((it) => {
    const childRewards = (it.candidate_ids || [])
      .map((cid) => reports[cid] && reports[cid].overall_reward)
      .filter((v) => typeof v === "number");
    const best = childRewards.length ? Math.max(...childRewards) : it.best_reward;
    const mean = childRewards.length ? childRewards.reduce((a, b) => a + b, 0) / childRewards.length : null;
    return { iter: it.iteration, best, mean, runningBest: it.best_reward };
  });

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMax = 1.0;
  const yMin = 0.0;

  function x(v) { return padL + ((v - xMin) / Math.max(1, xMax - xMin)) * (W - padL - padR); }
  function y(v) { return padT + (1 - (v - yMin) / (yMax - yMin)) * (H - padT - padB); }

  // Gridlines
  for (let g = 0; g <= 5; g++) {
    const v = yMax - (g * (yMax - yMin)) / 5;
    const yy = y(v);
    E("line", { x1: padL, y1: yy, x2: W - padR, y2: yy, stroke: "#1e293b", "stroke-dasharray": "2 3" }, svg);
    const t = E("text", { x: padL - 8, y: yy + 3, "font-size": 10, fill: "#64748b", "text-anchor": "end" }, svg);
    t.textContent = v.toFixed(2);
  }
  // X labels
  series.forEach((s) => {
    const xx = x(s.iter);
    E("text", { x: xx, y: H - 14, "font-size": 10, fill: "#94a3b8", "text-anchor": "middle" }, svg).textContent = `it ${s.iter}`;
  });
  E("text", { x: padL, y: padT - 4, "font-size": 10, fill: "#94a3b8" }, svg).textContent = "reward";

  function path(points, color, width = 2, dashed = false) {
    const d = points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p[0])} ${y(p[1])}`).join(" ");
    const attrs = { d, fill: "none", stroke: color, "stroke-width": width, "stroke-linecap": "round" };
    if (dashed) attrs["stroke-dasharray"] = "4 4";
    E("path", attrs, svg);
  }

  const bestLine = series.map((s) => [s.iter, s.best]);
  const runningLine = series.map((s) => [s.iter, s.runningBest]);
  const meanLine = series.filter((s) => s.mean != null).map((s) => [s.iter, s.mean]);

  if (meanLine.length) path(meanLine, "#94a3b8", 1.5, true);
  path(bestLine, "#7dd3fc", 2);
  path(runningLine, "#10b981", 3);

  // Dots on running best
  series.forEach((s) => {
    E("circle", { cx: x(s.iter), cy: y(s.runningBest), r: 4, fill: "#10b981" }, svg);
  });

  // Legend
  const lg = E("g", { transform: `translate(${padL + 10}, ${padT + 4})` }, svg);
  const items = [
    ["#10b981", "running best"],
    ["#7dd3fc", "iteration best child"],
    ["#94a3b8", "iteration mean"],
  ];
  items.forEach(([c, label], i) => {
    E("rect", { x: i * 140, y: 0, width: 10, height: 3, fill: c }, lg);
    E("text", { x: i * 140 + 14, y: 4, "font-size": 10, fill: "#cbd5e1" }, lg).textContent = label;
  });
})();
