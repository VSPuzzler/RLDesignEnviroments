"""Unit tests for the deterministic UI audit."""

from __future__ import annotations

import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

from services.scorer.deterministic_audit import (  # noqa: E402
    contrast_ratio,
    run_audit,
    score_accessibility,
    score_color_harmony,
    score_density_penalty,
    score_layout_balance,
    score_readability,
    score_visual_hierarchy,
)


def _el(
    tag="p", text="Hello world this is a sample paragraph",
    color="rgb(20, 20, 20)", bg="rgb(255, 255, 255)",
    fs=16.0, x=100.0, y=100.0, w=400.0, h=24.0,
    interactive=False, cta=False, has_alt=True, role=None,
):
    return {
        "tag": tag, "role": role, "text": text, "name": None,
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "font_size_px": fs, "color": color, "background_color": bg,
        "is_interactive": interactive, "is_cta": cta,
        "has_alt_or_label": has_alt, "tab_index": 0,
    }


def test_contrast_ratio_black_on_white_is_21():
    assert contrast_ratio("rgb(0,0,0)", "rgb(255,255,255)") == 21.0


def test_contrast_ratio_returns_none_for_transparent_bg():
    assert contrast_ratio("rgb(0,0,0)", "rgba(255,255,255,0.0)") is None


def test_accessibility_passes_for_high_contrast_labelled_ui():
    elements = [
        _el(tag="h1", text="Welcome", fs=32),
        _el(tag="p", text="Body copy that is readable.", fs=16),
        _el(tag="button", text="Sign up", fs=16, interactive=True),
    ]
    score, wcag, viols = score_accessibility(elements)
    assert wcag is True
    assert score >= 0.9
    assert viols == []


def test_accessibility_flags_low_contrast_text():
    elements = [
        _el(tag="p", text="bad", color="rgb(180,180,180)", bg="rgb(200,200,200)"),
    ]
    _, wcag, viols = score_accessibility(elements)
    assert wcag is False
    assert any(v["rule"] == "wcag.contrast" for v in viols)


def test_accessibility_flags_unlabelled_interactive():
    elements = [
        _el(tag="button", text="", interactive=True),
    ]
    _, _, viols = score_accessibility(elements)
    assert any(v["rule"] == "a11y.interactive_label" for v in viols)


def test_readability_rewards_large_fonts():
    elements_big = [_el(fs=18.0, text="Some readable copy") for _ in range(5)]
    elements_small = [_el(fs=10.0, text="Tiny copy") for _ in range(5)]
    big = score_readability(elements_big, "Some readable copy " * 5)
    small = score_readability(elements_small, "Tiny copy " * 5)
    assert big > small


def test_visual_hierarchy_higher_with_dispersion():
    flat = [_el(fs=16.0, text=f"x {i}") for i in range(6)]
    layered = [
        _el(tag="h1", fs=48.0, text="Big"),
        _el(tag="h2", fs=28.0, text="Medium"),
        _el(tag="p", fs=16.0, text="Body"),
        _el(tag="p", fs=16.0, text="Body 2"),
    ]
    assert score_visual_hierarchy(layered) > score_visual_hierarchy(flat)


def test_layout_balance_centered_is_higher():
    centered = [_el(x=620, y=440, w=200, h=20)]  # 1440x900 viewport
    skewed = [_el(x=0, y=0, w=200, h=20)]
    assert score_layout_balance(centered, 1440, 900) > score_layout_balance(skewed, 1440, 900)


def test_density_penalty_caps_at_one_for_sparse_pages():
    sparse = [_el(w=100, h=20)]
    s = score_density_penalty(sparse, 1440, 900)
    assert s == 1.0


def test_color_harmony_optimal_palette_size():
    # Build elements with 4 distinct colours (the sweet spot).
    palette = ["rgb(20,20,20)", "rgb(120,120,255)", "rgb(255,80,80)", "rgb(20,120,80)"]
    els = [_el(color=c) for c in palette]
    assert score_color_harmony(els) == 1.0


def test_run_audit_returns_full_shape():
    elements = [
        _el(tag="h1", text="Welcome", fs=44),
        _el(tag="p", text="Some body copy."),
        _el(tag="button", text="Sign up free", fs=18, interactive=True, cta=True),
    ]
    rendered = {
        "elements": elements, "visible_text": "Welcome Some body copy. Sign up free",
        "viewport_width": 1440, "viewport_height": 900,
    }
    audit = run_audit(rendered)
    for k in (
        "accessibility", "readability", "visual_hierarchy", "layout_balance",
        "cta_clarity", "density_penalty", "spacing_consistency", "color_harmony",
        "wcag_pass", "violations", "raw_features",
    ):
        assert k in audit, f"missing key: {k}"
    assert 0.0 <= audit["accessibility"] <= 1.0
    assert isinstance(audit["wcag_pass"], bool)
