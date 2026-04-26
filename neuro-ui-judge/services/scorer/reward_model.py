"""
Hybrid reward model.

Reward = function of:
  - d(u)  deterministic UI audit
  - n(u)  neural proxy features (TRIBE-mock or real)
  - h(u)  human preference calibration weights (optional)
  - A(u)  hard accessibility gate (binary)
  - D(u)  deterministic defect penalty (continuous)

Sub-scores returned (each on [0, 1], 1 = best):
  usability, attention_guidance, visual_hierarchy, cognitive_load,
  readability, aesthetic_quality, accessibility, engagement_proxy, trust

Cognitive load is *inverted at the sub-score level*: we report
`1 - load` so all sub-scores are "higher is better".  The internal
intermediate `cognitive_load_raw` is exposed in the report explanation.

This module also produces the natural-language explanation, recommendations,
and a letter grade.
"""

from __future__ import annotations

import math
from typing import Any

from .schemas import (
    CandidateReport,
    DeterministicAudit,
    NeuralProxyConfidence,
    NeuralProxyFeatures,
    Subscores,
)


# ── Default weights ─────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "usability": 0.13,
    "attention_guidance": 0.12,
    "visual_hierarchy": 0.12,
    "cognitive_load": 0.13,
    "readability": 0.10,
    "aesthetic_quality": 0.07,  # lower because confidence is low
    "accessibility": 0.18,      # weighted high
    "engagement_proxy": 0.07,
    "trust": 0.08,
}

# Multiplicative knob applied to confidence: 1.0 = honor confidence fully,
# 0.0 = ignore confidence and treat all metrics equally.
CONFIDENCE_GAIN = 1.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ── Sub-score formulas ──────────────────────────────────────────────────────


def _usability(audit: dict[str, Any]) -> float:
    return _clip01(
        0.45 * audit["readability"]
        + 0.35 * audit["cta_clarity"]
        + 0.20 * audit["spacing_consistency"]
    )


def _attention_guidance(audit: dict[str, Any], neural: dict[str, Any]) -> float:
    da = neural["roi_features"]["dorsal_attention"]["auc"]
    visual_var = neural["roi_features"]["visual"]["variance"]
    cta = audit["cta_clarity"]
    hierarchy = audit["visual_hierarchy"]
    return _sigmoid(
        2.0 * da
        + 1.5 * cta
        + 1.0 * hierarchy
        - 1.5 * visual_var
        - 1.0
    )


def _visual_hierarchy(audit: dict[str, Any], neural: dict[str, Any]) -> float:
    da_peak = neural["roi_features"]["dorsal_attention"]["peak"] or 0.5
    return _clip01(0.7 * audit["visual_hierarchy"] + 0.3 * da_peak)


def _cognitive_load_inverted(
    audit: dict[str, Any], neural: dict[str, Any]
) -> tuple[float, float]:
    """Return (1 - load, load_raw)."""
    md_auc = neural["roi_features"]["multiple_demand"]["auc"]
    sal_var = neural["roi_features"]["salience"]["variance"] or 0.3
    density_score = audit["density_penalty"]  # 1 = sparse
    hierarchy = audit["visual_hierarchy"]
    readability = audit["readability"]
    load_raw = _sigmoid(
        2.5 * md_auc
        + 1.5 * sal_var
        + 1.5 * (1.0 - density_score)
        - 1.0 * hierarchy
        - 1.0 * readability
    )
    return 1.0 - load_raw, load_raw


def _readability(audit: dict[str, Any], neural: dict[str, Any]) -> float:
    lang_auc = neural["roi_features"]["language_vwfa"]["auc"]
    return _clip01(0.7 * audit["readability"] + 0.3 * lang_auc)


def _aesthetic_quality(audit: dict[str, Any], neural: dict[str, Any]) -> float:
    val_auc = neural["roi_features"]["valuation_proxy"]["auc"]
    return _sigmoid(
        1.5 * audit["layout_balance"]
        + 1.2 * audit["spacing_consistency"]
        + 1.0 * audit["color_harmony"]
        + 0.6 * val_auc
        - 1.5 * (1.0 - audit["density_penalty"])
        - 1.0
    )


def _accessibility(audit: dict[str, Any]) -> float:
    return _clip01(
        (1.0 if audit["wcag_pass"] else 0.6) * audit["accessibility"]
    )


def _engagement_proxy(audit: dict[str, Any], neural: dict[str, Any]) -> float:
    sal_auc = neural["roi_features"]["salience"]["auc"]
    val_auc = neural["roi_features"]["valuation_proxy"]["auc"]
    dmn_supp = neural["roi_features"]["dmn"]["suppression"] or 0.5
    return _sigmoid(
        1.2 * sal_auc
        + 1.0 * val_auc
        + 1.0 * dmn_supp
        + 0.8 * audit["cta_clarity"]
        - 1.0
    )


def _trust(audit: dict[str, Any]) -> float:
    """Trust proxy: balance, consistent spacing, conservative palette,
    accessibility pass, no critical violations."""
    crit = sum(
        1 for v in audit.get("violations", []) if v.get("severity") == "critical"
    )
    crit_penalty = min(0.4, 0.2 * crit)
    return _clip01(
        0.30 * audit["layout_balance"]
        + 0.25 * audit["spacing_consistency"]
        + 0.15 * audit["color_harmony"]
        + 0.20 * (1.0 if audit["wcag_pass"] else 0.5)
        + 0.10 * audit["readability"]
        - crit_penalty
    )


# ── Confidence routing ─────────────────────────────────────────────────────


def _metric_confidence(
    metric: str, conf: dict[str, float], wcag_pass: bool
) -> float:
    """
    Map sub-scores to a [0, 1] confidence. Deterministic-only metrics get
    confidence 1.0; neural-influenced metrics inherit from the proxy.

    A failed WCAG pass *raises* accessibility's confidence (we trust the
    deterministic check) and lowers everything else slightly because the
    page may not be representative.
    """
    base = {
        "usability": 0.95,
        "attention_guidance": conf.get("attention", 0.6),
        "visual_hierarchy": 0.85,
        "cognitive_load": conf.get("load", 0.6),
        "readability": 0.9,
        "aesthetic_quality": conf.get("aesthetic", 0.25),
        "accessibility": 1.0,
        "engagement_proxy": (conf.get("aesthetic", 0.25) + conf.get("attention", 0.6)) / 2.0,
        "trust": 0.85,
    }[metric]
    if not wcag_pass and metric != "accessibility":
        base *= 0.9
    return _clip01(base)


# ── Defect / uncertainty penalties ─────────────────────────────────────────


def _defect_penalty(audit: dict[str, Any]) -> float:
    sev_weight = {"critical": 0.25, "major": 0.10, "minor": 0.04, "info": 0.01}
    p = sum(sev_weight.get(v.get("severity", "info"), 0.01)
            for v in audit.get("violations", []))
    return min(1.0, p)


def _uncertainty_penalty(conf: dict[str, float]) -> float:
    """Mean confidence across the four channels, mapped to a small penalty."""
    avg_conf = sum(conf.values()) / max(1, len(conf))
    return _clip01(0.5 * (1.0 - avg_conf))


# ── Public API ─────────────────────────────────────────────────────────────


def _grade(reward: float) -> str:
    if reward >= 0.85:
        return "A"
    if reward >= 0.70:
        return "B"
    if reward >= 0.55:
        return "C"
    if reward >= 0.40:
        return "D"
    return "F"


def _make_recommendations(
    audit: dict[str, Any],
    neural: dict[str, Any],
    subs: dict[str, float],
) -> list[str]:
    recs: list[str] = []
    if not audit["wcag_pass"]:
        recs.append(
            "Fix WCAG contrast / labeling violations — accessibility gate currently failing."
        )
    if subs["attention_guidance"] < 0.55:
        recs.append(
            "Strengthen the visual hierarchy: enlarge the primary heading and "
            "give the main CTA more contrast or whitespace."
        )
    if subs["cognitive_load"] < 0.55:
        recs.append(
            "Reduce density: fewer simultaneous focal points, more whitespace, "
            "shorter copy blocks."
        )
    if subs["readability"] < 0.55:
        recs.append(
            "Increase body font size to ≥ 16px and break long paragraphs with subheadings."
        )
    if subs["aesthetic_quality"] < 0.55:
        recs.append(
            "Tighten the palette to 3–6 distinct colors and even out vertical spacing."
        )
    if not recs:
        recs.append("Strong baseline; consider A/B testing against a simplified variant.")
    return recs


def _make_explanation(
    subs: dict[str, float],
    audit: dict[str, Any],
    neural: dict[str, Any],
    load_raw: float,
    overall: float,
    a_gate: bool,
    weights: dict[str, float],
) -> str:
    top = sorted(subs.items(), key=lambda kv: kv[1], reverse=True)[:2]
    bot = sorted(subs.items(), key=lambda kv: kv[1])[:2]
    parts = [
        f"Overall reward {overall:.2f}.",
        "Strengths: " + ", ".join(f"{k} ({v:.2f})" for k, v in top) + ".",
        "Weaknesses: " + ", ".join(f"{k} ({v:.2f})" for k, v in bot) + ".",
        f"Predicted cognitive load (raw) {load_raw:.2f}.",
        f"Neural proxy mode: {neural.get('mode', 'mock')} — aesthetic / valuation "
        "treated as low-confidence by design.",
    ]
    if not a_gate:
        parts.append(
            "Accessibility gate FAILED: overall reward is hard-capped until "
            "WCAG violations are resolved."
        )
    return " ".join(parts)


def score_candidate(
    candidate_id: str,
    audit: dict[str, Any],
    neural: dict[str, Any],
    weights: dict[str, float] | None = None,
    weights_version: str = "default-v1",
) -> dict[str, Any]:
    """
    Run the full hybrid reward model and return a `CandidateReport`-shaped dict.

    Args:
        candidate_id: stable id.
        audit: dict from `deterministic_audit.run_audit`.
        neural: dict from `tribe_adapter.predict`.
        weights: per-metric weights; falls back to DEFAULT_WEIGHTS.
        weights_version: tag for the weight vector (used by the calibrator).
    """
    weights = weights or DEFAULT_WEIGHTS
    conf = neural.get("confidence", {})

    cog_inv, load_raw = _cognitive_load_inverted(audit, neural)

    subs: dict[str, float] = {
        "usability": _usability(audit),
        "attention_guidance": _attention_guidance(audit, neural),
        "visual_hierarchy": _visual_hierarchy(audit, neural),
        "cognitive_load": cog_inv,
        "readability": _readability(audit, neural),
        "aesthetic_quality": _aesthetic_quality(audit, neural),
        "accessibility": _accessibility(audit),
        "engagement_proxy": _engagement_proxy(audit, neural),
        "trust": _trust(audit),
    }

    # Confidence-weighted sum.
    a_gate = bool(audit["wcag_pass"])
    weighted_num = 0.0
    weighted_den = 0.0
    for m, s in subs.items():
        w = weights.get(m, 0.0)
        c = _metric_confidence(m, conf, a_gate) ** CONFIDENCE_GAIN
        weighted_num += w * c * s
        weighted_den += w * c
    base_reward = weighted_num / weighted_den if weighted_den > 0 else 0.0

    defect_pen = _defect_penalty(audit)
    uncertainty_pen = _uncertainty_penalty(conf)

    # Soft penalties before the hard gate.
    pre_gate = base_reward - 0.25 * defect_pen - 0.10 * uncertainty_pen
    pre_gate = _clip01(pre_gate)

    # Hard accessibility gate: critical a11y failure caps the score.
    if a_gate:
        overall = pre_gate
    else:
        overall = min(pre_gate, 0.55)  # explicit cap; communicated in explanation.

    recommendations = _make_recommendations(audit, neural, subs)
    explanation = _make_explanation(
        subs, audit, neural, load_raw, overall, a_gate, weights
    )

    report = {
        "candidate_id": candidate_id,
        "overall_reward": float(overall),
        "grade": _grade(overall),
        "subscores": subs,
        "deterministic_audit": audit,
        "neural_proxy": neural,
        "confidence": conf,
        "violations": audit.get("violations", []),
        "recommendations": recommendations,
        "explanation": explanation,
        "weights_version": weights_version,
        "accessibility_gate_passed": a_gate,
        "defect_penalty": float(defect_pen),
        "uncertainty_penalty": float(uncertainty_pen),
    }
    # Validate via pydantic so downstream consumers can rely on shape.
    CandidateReport.model_validate(
        {
            **report,
            "subscores": Subscores(**subs).model_dump(),
            "deterministic_audit": DeterministicAudit(**audit).model_dump(),
            "neural_proxy": NeuralProxyFeatures(**neural).model_dump(),
            "confidence": NeuralProxyConfidence(**conf).model_dump(),
        }
    )
    return report
