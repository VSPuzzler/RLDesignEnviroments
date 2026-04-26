"""
Deterministic, standards-based UI audit.

Every public scoring function returns a value in [0, 1] (1 = best) and is
unit-testable in isolation. Where the spec asks for normative thresholds
(e.g. WCAG contrast), we use the published numeric standards; where it asks
for design heuristics (hierarchy, balance), we use transparent formulas that
operate on the structured element list returned by the renderer.

Important: this module never calls any LLM and never depends on neural
features. It is the *hard floor* of the reward model.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from typing import Any

logger = logging.getLogger(__name__)


# ── Color utilities ─────────────────────────────────────────────────────────


_RGB_RE = re.compile(
    r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)"
)


def _parse_color(value: str | None) -> tuple[int, int, int, float] | None:
    """Parse a CSS color string. Returns (r, g, b, a) or None."""
    if not value:
        return None
    m = _RGB_RE.match(value.strip())
    if not m:
        return None
    r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
    a = float(m.group(4)) if m.group(4) is not None else 1.0
    return r, g, b, a


def _relative_luminance(r: int, g: int, b: int) -> float:
    """WCAG 2.x relative luminance."""

    def _ch(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * _ch(r) + 0.7152 * _ch(g) + 0.0722 * _ch(b)


def contrast_ratio(fg: str | None, bg: str | None) -> float | None:
    """WCAG contrast ratio between fg and bg colors (range 1..21).

    Returns None when either color cannot be parsed (e.g. transparent
    backgrounds — we explicitly skip those rather than fabricating a number).
    """
    pfg, pbg = _parse_color(fg), _parse_color(bg)
    if not pfg or not pbg:
        return None
    if pbg[3] < 0.95:  # transparent background → unknown
        return None
    L1 = _relative_luminance(*pfg[:3])
    L2 = _relative_luminance(*pbg[:3])
    hi, lo = max(L1, L2), min(L1, L2)
    return (hi + 0.05) / (lo + 0.05)


# ── Sub-scores (each on [0, 1], 1 = best) ───────────────────────────────────


def score_accessibility(
    elements: list[dict[str, Any]],
) -> tuple[float, bool, list[dict[str, Any]]]:
    """
    Returns (score, wcag_pass, violations).

    - For every text-bearing element we attempt a WCAG contrast check; we
      require >= 4.5:1 for body text and >= 3:1 for "large" text (>= 18px or
      bold >= 14px — we approximate "bold" as missing here and use 18px).
    - We penalise images without alt and interactive elements without
      accessible names.
    - WCAG pass = no critical contrast / labeling failures.
    """
    violations: list[dict[str, Any]] = []
    contrast_passes = 0
    contrast_total = 0

    for i, el in enumerate(elements):
        text = (el.get("text") or "").strip()
        font_size = el.get("font_size_px") or 0.0
        if text and font_size > 0:
            ratio = contrast_ratio(el.get("color"), el.get("background_color"))
            if ratio is not None:
                contrast_total += 1
                req = 3.0 if font_size >= 18 else 4.5
                if ratio >= req:
                    contrast_passes += 1
                else:
                    violations.append(
                        {
                            "rule": "wcag.contrast",
                            "severity": "major" if ratio < 3.0 else "minor",
                            "message": (
                                f"Contrast {ratio:.2f}:1 below WCAG minimum "
                                f"{req:.1f}:1 for '{text[:40]}'"
                            ),
                            "element_index": i,
                        }
                    )

        if el.get("tag") == "img" and not el.get("has_alt_or_label", True):
            violations.append(
                {
                    "rule": "a11y.alt_text",
                    "severity": "major",
                    "message": "Image missing alt text",
                    "element_index": i,
                }
            )

        if el.get("is_interactive") and not (text or el.get("name")):
            violations.append(
                {
                    "rule": "a11y.interactive_label",
                    "severity": "major",
                    "message": (
                        f"Interactive <{el.get('tag')}> has no accessible name"
                    ),
                    "element_index": i,
                }
            )

    contrast_score = (
        contrast_passes / contrast_total if contrast_total > 0 else 0.6
    )
    label_penalty = 0.05 * sum(
        1 for v in violations if v["rule"] != "wcag.contrast"
    )
    score = max(0.0, contrast_score - label_penalty)

    critical_failures = sum(
        1 for v in violations if v["severity"] in ("major", "critical")
    )
    wcag_pass = critical_failures == 0 and contrast_score >= 0.9
    return score, wcag_pass, violations


def score_readability(elements: list[dict[str, Any]], visible_text: str) -> float:
    """
    Readability heuristic combining:
      - share of body text rendered at >= 14px
      - average word length sanity (not too long)
      - presence of headings vs wall-of-text
    """
    if not elements:
        return 0.3

    text_elements = [
        e for e in elements if (e.get("text") or "").strip() and e.get("font_size_px")
    ]
    if not text_elements:
        return 0.4

    big_enough = sum(1 for e in text_elements if e["font_size_px"] >= 14)
    size_score = big_enough / len(text_elements)

    words = re.findall(r"\w+", visible_text or "")
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        len_score = max(0.0, 1.0 - abs(avg_len - 5.0) / 6.0)
    else:
        len_score = 0.5

    headings = sum(1 for e in elements if e.get("tag") in ("h1", "h2", "h3"))
    heading_score = min(1.0, headings / 3.0)

    return 0.55 * size_score + 0.25 * len_score + 0.20 * heading_score


def score_visual_hierarchy(elements: list[dict[str, Any]]) -> float:
    """
    Hierarchy = how clearly the page differentiates important from secondary
    content.  We measure font-size dispersion (CV of font sizes), the
    presence of >=1 dominant element (largest font ≥ 1.5× median), and a
    monotone vertical structure (top elements tend to be larger).
    """
    fonts = [e.get("font_size_px") for e in elements if e.get("font_size_px")]
    if len(fonts) < 3:
        return 0.4

    median = statistics.median(fonts)
    largest = max(fonts)
    cv = (statistics.pstdev(fonts) / (statistics.mean(fonts) + 1e-6))
    cv_score = min(1.0, cv * 3.0)  # 0.33 CV ≈ 1.0

    dominance = 1.0 if largest >= 1.5 * median else largest / (1.5 * median)

    text_els = [
        e for e in elements
        if e.get("font_size_px") and (e.get("text") or "").strip()
    ]
    if len(text_els) >= 4:
        sorted_by_y = sorted(text_els, key=lambda e: e["bbox"]["y"])
        sizes = [e["font_size_px"] for e in sorted_by_y]
        top_third = sizes[: max(1, len(sizes) // 3)]
        bottom_third = sizes[-max(1, len(sizes) // 3):]
        monotone = (
            1.0 if statistics.mean(top_third) >= statistics.mean(bottom_third)
            else 0.5
        )
    else:
        monotone = 0.6

    return 0.45 * cv_score + 0.35 * dominance + 0.20 * monotone


def score_layout_balance(
    elements: list[dict[str, Any]], viewport_width: int, viewport_height: int
) -> float:
    """
    Balance = how evenly content is distributed.  We compute the centroid of
    element area-weighted bbox centers and compare to the viewport center.
    Closer centroid → higher balance.
    """
    if not elements or viewport_width <= 0:
        return 0.5

    cx_acc = cy_acc = w_acc = 0.0
    for e in elements:
        b = e.get("bbox", {})
        w = float(b.get("width", 0.0))
        h = float(b.get("height", 0.0))
        if w <= 0 or h <= 0:
            continue
        area = w * h
        cx_acc += area * (b.get("x", 0.0) + w / 2.0)
        cy_acc += area * (b.get("y", 0.0) + h / 2.0)
        w_acc += area
    if w_acc == 0:
        return 0.5
    cx = cx_acc / w_acc
    cy = cy_acc / w_acc
    # Distance to viewport center, normalised by half-diagonal.
    half_diag = math.hypot(viewport_width / 2.0, viewport_height / 2.0)
    d = math.hypot(cx - viewport_width / 2.0, cy - viewport_height / 2.0)
    return max(0.0, 1.0 - d / half_diag)


def score_cta_clarity(elements: list[dict[str, Any]]) -> float:
    """
    CTA clarity rewards: a single dominant CTA above the fold, with
    sufficient size and contrast.  Multiple equally-prominent CTAs reduce
    the score (decision paralysis).
    """
    ctas = [e for e in elements if e.get("is_cta")]
    if not ctas:
        return 0.2
    # Prefer CTAs in the top half of the viewport.
    above_fold = [
        e for e in ctas
        if e["bbox"]["y"] + e["bbox"]["height"] / 2 < 600
    ]
    pos_score = 1.0 if above_fold else 0.5
    # Contrast on the primary CTA.
    primary = max(ctas, key=lambda e: e["bbox"]["width"] * e["bbox"]["height"])
    ratio = contrast_ratio(primary.get("color"), primary.get("background_color"))
    contrast_part = (
        min(1.0, (ratio - 1.0) / 6.0) if ratio is not None else 0.5
    )
    # Penalise if there are >=3 CTAs of similar area.
    areas = sorted(
        (e["bbox"]["width"] * e["bbox"]["height"] for e in ctas), reverse=True
    )
    if len(areas) >= 3 and areas[2] > 0.7 * areas[0]:
        clarity_part = 0.5
    elif len(areas) == 1:
        clarity_part = 1.0
    else:
        clarity_part = 0.85
    return 0.4 * pos_score + 0.3 * contrast_part + 0.3 * clarity_part


def score_density_penalty(
    elements: list[dict[str, Any]], viewport_width: int, viewport_height: int
) -> float:
    """
    Returns a *score* in [0, 1] where 1 = uncrowded (no penalty), 0 = very dense.
    We measure visible-element area coverage and a rough overlap proxy.
    """
    if not elements or viewport_width <= 0 or viewport_height <= 0:
        return 0.5
    viewport_area = viewport_width * viewport_height
    total_area = 0.0
    for e in elements:
        b = e.get("bbox", {})
        w = float(b.get("width", 0.0))
        h = float(b.get("height", 0.0))
        if 0 < w <= viewport_width and 0 < h <= viewport_height:
            total_area += w * h
    coverage = min(2.5, total_area / max(1.0, viewport_area))
    # 1.0 coverage ≈ everything once; >2 implies heavy overlap → penalise.
    if coverage <= 1.0:
        return 1.0
    return max(0.0, 1.0 - (coverage - 1.0) / 1.5)


def score_spacing_consistency(elements: list[dict[str, Any]]) -> float:
    """
    Lower variance in vertical gaps between text blocks ≈ more consistent
    spacing and a more "designed" look.
    """
    text_els = sorted(
        [e for e in elements if (e.get("text") or "").strip()],
        key=lambda e: e["bbox"]["y"],
    )
    if len(text_els) < 4:
        return 0.6
    gaps = []
    for a, b in zip(text_els, text_els[1:]):
        ay2 = a["bbox"]["y"] + a["bbox"]["height"]
        by1 = b["bbox"]["y"]
        if by1 > ay2:
            gaps.append(by1 - ay2)
    if len(gaps) < 3:
        return 0.6
    mean_gap = statistics.mean(gaps)
    if mean_gap <= 0:
        return 0.5
    cv = statistics.pstdev(gaps) / mean_gap
    return max(0.0, 1.0 - min(1.0, cv / 1.2))


def score_color_harmony(elements: list[dict[str, Any]]) -> float:
    """
    Color harmony heuristic: count distinct foreground colors; reward 3–6
    distinct colors (a typical palette), penalise 1 (monotonous) or >10
    (chaotic).
    """
    colors = set()
    for e in elements:
        c = _parse_color(e.get("color"))
        if c:
            colors.add(c[:3])
    n = len(colors)
    if n == 0:
        return 0.5
    if 3 <= n <= 6:
        return 1.0
    if n in (1, 2):
        return 0.6
    return max(0.3, 1.0 - (n - 6) * 0.05)


# ── Orchestration ───────────────────────────────────────────────────────────


def run_audit(rendered: dict[str, Any]) -> dict[str, Any]:
    """
    Run all sub-scores and return a `DeterministicAudit`-shaped dict.

    Args:
        rendered: a `RenderedArtifact` dict from the renderer.
    """
    elements = rendered.get("elements", []) or []
    vw = int(rendered.get("viewport_width") or 1440)
    vh = int(rendered.get("viewport_height") or 900)
    visible_text = rendered.get("visible_text", "") or ""

    accessibility, wcag_pass, violations = score_accessibility(elements)
    readability = score_readability(elements, visible_text)
    hierarchy = score_visual_hierarchy(elements)
    balance = score_layout_balance(elements, vw, vh)
    cta = score_cta_clarity(elements)
    density = score_density_penalty(elements, vw, vh)
    spacing = score_spacing_consistency(elements)
    harmony = score_color_harmony(elements)

    raw = {
        "n_elements": float(len(elements)),
        "n_interactive": float(sum(1 for e in elements if e.get("is_interactive"))),
        "n_ctas": float(sum(1 for e in elements if e.get("is_cta"))),
        "n_headings": float(
            sum(1 for e in elements if e.get("tag") in ("h1", "h2", "h3"))
        ),
        "viewport_width": float(vw),
        "viewport_height": float(vh),
    }

    return {
        "accessibility": float(accessibility),
        "readability": float(readability),
        "visual_hierarchy": float(hierarchy),
        "layout_balance": float(balance),
        "cta_clarity": float(cta),
        "density_penalty": float(density),
        "spacing_consistency": float(spacing),
        "color_harmony": float(harmony),
        "wcag_pass": bool(wcag_pass),
        "violations": violations,
        "raw_features": raw,
    }
