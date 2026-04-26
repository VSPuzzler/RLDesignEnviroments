"""Unit tests for Bradley-Terry preference calibration."""

from __future__ import annotations

import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

from services.scorer.preference_model import (  # noqa: E402
    METRIC_ORDER,
    fit_preference_weights,
    predict_pairwise_probability,
)


def _report(cid, **overrides):
    base = {
        "candidate_id": cid,
        "subscores": {m: 0.5 for m in METRIC_ORDER},
        "overall_reward": 0.5,
        "deterministic_audit": {},
        "neural_proxy": {},
        "confidence": {},
    }
    base["subscores"].update(overrides)
    return base


def test_fit_returns_default_weights_on_too_few_pairs():
    out = fit_preference_weights([], {}, n_steps=10)
    assert "weights" in out
    assert "metrics" in out
    assert out["metrics"]["n_train"] == 0


def test_fit_recovers_a_clear_signal():
    """If A has higher 'accessibility' on every pair where A wins, the
    learned weights must place positive mass on accessibility."""
    reports = {
        "A1": _report("A1", accessibility=0.9), "B1": _report("B1", accessibility=0.2),
        "A2": _report("A2", accessibility=0.85), "B2": _report("B2", accessibility=0.25),
        "A3": _report("A3", accessibility=0.95), "B3": _report("B3", accessibility=0.15),
        "A4": _report("A4", accessibility=0.88), "B4": _report("B4", accessibility=0.30),
        "A5": _report("A5", accessibility=0.92), "B5": _report("B5", accessibility=0.20),
        "A6": _report("A6", accessibility=0.80), "B6": _report("B6", accessibility=0.35),
    }
    prefs = [
        {"ui_a_id": f"A{i}", "ui_b_id": f"B{i}", "winner": "a"}
        for i in range(1, 7)
    ]
    out = fit_preference_weights(prefs, reports, n_steps=600, val_fraction=0.2)
    w = out["weights"]
    assert w["accessibility"] > 0.05
    # Pairwise accuracy should be much better than chance.
    assert out["metrics"]["pairwise_accuracy"] >= 0.8


def test_predict_pairwise_probability_is_well_defined():
    a = _report("a", readability=0.9)
    b = _report("b", readability=0.1)
    weights = {m: 1.0 / len(METRIC_ORDER) for m in METRIC_ORDER}
    p = predict_pairwise_probability(a, b, weights)
    assert 0.5 < p <= 1.0
