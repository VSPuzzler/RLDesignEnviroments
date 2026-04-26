"""
Mocked TRIBE-like neural proxy.

We deliberately do *not* generate noise. Instead each ROI's predicted
"activation" is a transparent function of measurable visual / layout / text
features extracted by the renderer. This keeps the demo coherent: a busier
UI raises predicted attention/load, a clearer hierarchy raises dorsal-attention
"AUC", etc.

Every feature dict carries:
  - mode = "mock" (so consumers know not to overclaim)
  - confidence per channel (aesthetic confidence is intentionally low)

When real TRIBE v2 inference is wired in via `tribe_adapter.py`, this module
is bypassed and only used as a fallback / sanity baseline.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _visual_entropy(elements: list[dict[str, Any]]) -> float:
    """Shannon-style entropy over element area distribution; high = cluttered."""
    areas = []
    for e in elements:
        b = e.get("bbox", {})
        a = float(b.get("width", 0.0)) * float(b.get("height", 0.0))
        if a > 0:
            areas.append(a)
    if not areas:
        return 0.0
    total = sum(areas)
    p = [a / total for a in areas]
    return -sum(pi * math.log(pi + 1e-9) for pi in p) / math.log(len(p) + 1e-9)


def _text_density(visible_text: str, viewport_area: float) -> float:
    """Words per 1000px²; clipped."""
    n_words = len([w for w in visible_text.split() if w.strip()])
    return min(20.0, 1000.0 * n_words / max(1.0, viewport_area))


def _hierarchy_strength(elements: list[dict[str, Any]]) -> float:
    fonts = [e.get("font_size_px") for e in elements if e.get("font_size_px")]
    if len(fonts) < 2:
        return 0.0
    return statistics.pstdev(fonts) / (statistics.mean(fonts) + 1e-6)


def _cta_prominence(elements: list[dict[str, Any]]) -> float:
    ctas = [e for e in elements if e.get("is_cta")]
    if not ctas:
        return 0.0
    primary = max(ctas, key=lambda e: e["bbox"]["width"] * e["bbox"]["height"])
    a = primary["bbox"]["width"] * primary["bbox"]["height"]
    return min(1.0, a / (200.0 * 50.0))


def predict_neural_proxy(
    rendered: dict[str, Any],
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Produce a `NeuralProxyFeatures`-shaped dict from a RenderedArtifact.

    Each ROI is a small 1-D summary (auc, peak, variance) computed from
    interpretable visual features. We treat:
      - visual cortex: driven by element coverage + visual entropy
      - dorsal attention: driven by hierarchy strength + CTA prominence
      - salience: driven by color contrast variance + CTA density
      - multiple-demand (load): driven by element count + text density
      - language/VWFA: driven by readable-text proportion
      - DMN suppression: rises when the page strongly directs attention
      - valuation_proxy: weakly correlates with balance + harmony
        (LOW CONFIDENCE — flagged accordingly)
    """
    elements = rendered.get("elements", []) or []
    vw = float(rendered.get("viewport_width") or 1440)
    vh = float(rendered.get("viewport_height") or 900)
    viewport_area = vw * vh
    visible_text = rendered.get("visible_text", "") or ""

    n_elements = len(elements)
    interactive = sum(1 for e in elements if e.get("is_interactive"))

    coverage = (
        sum(
            float(e.get("bbox", {}).get("width", 0.0))
            * float(e.get("bbox", {}).get("height", 0.0))
            for e in elements
        )
        / max(1.0, viewport_area)
    )
    entropy = _visual_entropy(elements)
    hierarchy = _hierarchy_strength(elements)
    cta_prom = _cta_prominence(elements)
    text_density = _text_density(visible_text, viewport_area)
    font_sizes = [e["font_size_px"] for e in elements if e.get("font_size_px")]
    readable_share = (
        sum(1 for f in font_sizes if f >= 14) / len(font_sizes)
        if font_sizes else 0.5
    )

    audit = audit or {}
    layout_balance = float(audit.get("layout_balance", 0.5))
    color_harmony = float(audit.get("color_harmony", 0.5))

    # ── ROI summaries (auc/peak/variance scaled to roughly [0, 1]) ──────────
    visual_auc = _sigmoid(2.0 * (coverage - 0.4) + 0.5 * entropy)
    visual_peak = _sigmoid(0.8 * entropy + 0.6 * coverage)
    visual_var = min(1.0, 0.3 + 0.4 * entropy)

    da_auc = _sigmoid(2.5 * hierarchy + 1.5 * cta_prom - 0.5)
    da_peak = _sigmoid(2.0 * cta_prom + hierarchy)
    da_var = min(1.0, 0.2 + 0.5 * hierarchy)

    sal_auc = _sigmoid(
        1.5 * cta_prom
        + 1.0 * (0.6 - color_harmony)  # disharmonious palettes spike salience
        + 0.5 * entropy
        - 0.5
    )
    sal_peak = _sigmoid(2.0 * cta_prom + entropy)
    sal_var = min(1.0, 0.3 + 0.4 * (1 - color_harmony))

    md_auc = _sigmoid(0.06 * n_elements + 0.15 * text_density - 1.0)
    md_peak = _sigmoid(0.05 * n_elements + 0.2 * interactive - 1.0)
    md_var = min(1.0, 0.2 + 0.05 * n_elements)

    lang_auc = _sigmoid(2.0 * readable_share - 1.0 + 0.05 * text_density)
    lang_peak = _sigmoid(2.0 * readable_share)
    lang_var = min(1.0, 0.2 + 0.3 * (1 - readable_share))

    dmn_auc = _sigmoid(0.5 - 1.5 * cta_prom - hierarchy)
    dmn_suppression = max(0.0, 1.0 - dmn_auc)

    val_auc = _sigmoid(1.0 * layout_balance + 0.8 * color_harmony - 1.0)

    rois = {
        "visual": {"auc": visual_auc, "peak": visual_peak, "variance": visual_var},
        "dorsal_attention": {"auc": da_auc, "peak": da_peak, "variance": da_var},
        "salience": {"auc": sal_auc, "peak": sal_peak, "variance": sal_var},
        "multiple_demand": {"auc": md_auc, "peak": md_peak, "variance": md_var},
        "language_vwfa": {
            "auc": lang_auc, "peak": lang_peak, "variance": lang_var,
        },
        "dmn": {"auc": dmn_auc, "suppression": dmn_suppression, "variance": 0.3},
        "valuation_proxy": {"auc": val_auc, "confidence": "low"},
    }

    # ── Confidences ────────────────────────────────────────────────────────
    n_signal = min(1.0, n_elements / 25.0)
    confidence = {
        "attention": float(0.55 + 0.35 * n_signal),
        "load": float(0.55 + 0.35 * n_signal),
        "aesthetic": 0.25,  # explicitly low — this is the honest move
        "accessibility": 0.10,  # neural proxy is a poor a11y signal
    }

    notes = (
        "MOCK MODE. ROI features are deterministic functions of visual/layout "
        "features, not real fMRI predictions. Aesthetic / valuation channel "
        "is intentionally low-confidence."
    )

    proxy: dict[str, Any] = {
        "mode": "mock",
        "roi_features": rois,
        "confidence": confidence,
        "notes": notes,
    }
    # Synthesize a length-20484 vertex map so the dashboard's 3D brain has
    # something coherent to render even without real TRIBE running.
    proxy["vertex_activation"] = synthesize_vertex_activation(
        proxy,
        seed=int(abs(hash(rendered.get("screenshot_path") or rendered.get("visible_text") or "neuroui")) & 0xFFFF),
    )
    proxy["n_segments"] = None  # mock has no temporal segmentation
    return proxy


# ── Time-series synthesis for the dashboard ─────────────────────────────────


def synthesize_roi_timeseries(
    neural_proxy: dict[str, Any],
    n_steps: int = 20,
    seed: int | None = None,
) -> dict[str, list[float]]:
    """
    Build a plausible-looking time series per ROI from the AUC/peak summaries.

    This is purely for visualisation — it interpolates a smooth gamma-shaped
    response anchored at each ROI's peak / auc so the dashboard chart looks
    like a real BOLD trace. We make it a deterministic function of the
    summaries so identical UIs produce identical charts.
    """
    import random

    rng = random.Random(seed if seed is not None else 0)
    out: dict[str, list[float]] = {}
    rois = neural_proxy.get("roi_features", {})
    for name, feats in rois.items():
        auc = float(feats.get("auc", 0.5))
        peak_val = float(feats.get("peak", auc))
        variance = float(feats.get("variance", 0.3))
        peak_t = 0.35 + 0.3 * (1.0 - auc)  # earlier peak when AUC is high
        series = []
        for i in range(n_steps):
            t = i / max(1, n_steps - 1)
            shape = (t / max(peak_t, 1e-3)) ** 2 * math.exp(
                -((t - peak_t) ** 2) / (2 * (0.15 + variance * 0.15) ** 2)
            )
            jitter = rng.uniform(-0.04, 0.04) * variance
            series.append(max(0.0, peak_val * shape + jitter))
        out[name] = series
    return out
