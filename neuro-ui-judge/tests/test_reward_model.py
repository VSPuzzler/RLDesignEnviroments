"""Unit tests for the hybrid reward model."""

from __future__ import annotations

import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

from services.scorer.reward_model import DEFAULT_WEIGHTS, score_candidate  # noqa: E402


def _audit(**overrides):
    base = {
        "accessibility": 0.95,
        "readability": 0.85,
        "visual_hierarchy": 0.7,
        "layout_balance": 0.75,
        "cta_clarity": 0.8,
        "density_penalty": 0.9,
        "spacing_consistency": 0.7,
        "color_harmony": 0.8,
        "wcag_pass": True,
        "violations": [],
        "raw_features": {"viewport_width": 1440.0, "viewport_height": 900.0},
    }
    base.update(overrides)
    return base


def _neural(**overrides):
    base = {
        "mode": "mock",
        "roi_features": {
            "visual": {"auc": 0.6, "peak": 0.7, "variance": 0.3},
            "dorsal_attention": {"auc": 0.7, "peak": 0.8, "variance": 0.2},
            "salience": {"auc": 0.5, "peak": 0.6, "variance": 0.3},
            "multiple_demand": {"auc": 0.3, "peak": 0.4, "variance": 0.2},
            "language_vwfa": {"auc": 0.7, "peak": 0.7, "variance": 0.2},
            "dmn": {"auc": 0.3, "suppression": 0.7, "variance": 0.3},
            "valuation_proxy": {"auc": 0.6, "confidence": "low"},
        },
        "confidence": {
            "attention": 0.8, "load": 0.8, "aesthetic": 0.25, "accessibility": 0.1,
        },
        "notes": "test",
    }
    base.update(overrides)
    return base


def test_score_returns_full_shape():
    rep = score_candidate("c1", _audit(), _neural())
    for k in (
        "candidate_id", "overall_reward", "grade", "subscores", "deterministic_audit",
        "neural_proxy", "confidence", "violations", "recommendations", "explanation",
        "weights_version", "accessibility_gate_passed", "defect_penalty",
        "uncertainty_penalty",
    ):
        assert k in rep
    for sk in (
        "usability", "attention_guidance", "visual_hierarchy", "cognitive_load",
        "readability", "aesthetic_quality", "accessibility", "engagement_proxy", "trust",
    ):
        assert 0.0 <= rep["subscores"][sk] <= 1.0


def test_accessibility_gate_caps_overall_score():
    failing = _audit(
        wcag_pass=False, accessibility=0.4,
        violations=[{"rule": "wcag.contrast", "severity": "major", "message": "x"}] * 3,
    )
    rep = score_candidate("c", failing, _neural())
    assert rep["accessibility_gate_passed"] is False
    assert rep["overall_reward"] <= 0.55


def test_higher_load_lowers_reward():
    light = score_candidate("a", _audit(), _neural())["overall_reward"]
    heavy_neural = _neural()
    heavy_neural["roi_features"]["multiple_demand"]["auc"] = 0.95
    heavy_neural["roi_features"]["salience"]["variance"] = 0.9
    heavy_audit = _audit(density_penalty=0.3)
    heavy = score_candidate("b", heavy_audit, heavy_neural)["overall_reward"]
    assert heavy < light


def test_better_audit_yields_higher_reward():
    weak = score_candidate(
        "weak",
        _audit(
            readability=0.4, visual_hierarchy=0.3, cta_clarity=0.3,
            color_harmony=0.4, spacing_consistency=0.4, layout_balance=0.4,
        ),
        _neural(),
    )
    strong = score_candidate("strong", _audit(), _neural())
    assert strong["overall_reward"] > weak["overall_reward"]


def test_default_weights_sum_close_to_one():
    s = sum(DEFAULT_WEIGHTS.values())
    assert 0.99 < s < 1.01
