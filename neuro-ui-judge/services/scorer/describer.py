"""
Build the text we feed into TRIBE v2.

TRIBE v2 is video / audio / text only — there is no image input pathway. UIs
are heavily image-driven, so feeding TRIBE just the page's literal
``visible_text`` underrepresents layout, colour, and visual hierarchy. To
give TRIBE something its language pathway can usefully predict on, we build
a *composite description* by combining:

  1. A vision-LLM description of the screenshot (via ``llm_client.describe_image``)
  2. A deterministic layout summary derived from the rendered DOM elements
  3. The raw ``visible_text`` of the page

The vision step is optional — if ``OPENROUTER_API_KEY`` is unset or the
request fails, we fall back to (2) + (3), which still produces a coherent
description that TRIBE can encode.

The output is a single paragraph that flows naturally so TRIBE's word-level
context window has continuous text. This module is pure (no I/O beyond the
optional vision call) and side-effect free.
"""

from __future__ import annotations

import logging
import os
import statistics
from typing import Any

from . import llm_client

logger = logging.getLogger(__name__)


def _layout_summary(rendered: dict[str, Any]) -> str:
    """
    Build a short deterministic prose summary of the page's layout.

    Designed to read like real English so it concatenates cleanly with the
    vision description and the page's body text.
    """
    elements = rendered.get("elements") or []
    vw = rendered.get("viewport_width") or 1440
    vh = rendered.get("viewport_height") or 900

    n = len(elements)
    if n == 0:
        return (
            f"The page renders into a {vw} by {vh} viewport but no visible "
            "elements were captured."
        )

    n_interactive = sum(1 for e in elements if e.get("is_interactive"))
    n_ctas = sum(1 for e in elements if e.get("is_cta"))
    n_headings = sum(
        1 for e in elements if str(e.get("tag", "")).lower() in {"h1", "h2", "h3"}
    )
    n_images = sum(
        1 for e in elements if str(e.get("tag", "")).lower() in {"img", "svg", "video"}
    )

    fonts = [e.get("font_size_px") for e in elements if e.get("font_size_px")]
    if fonts:
        max_font = max(fonts)
        median_font = statistics.median(fonts)
        font_phrase = (
            f"The largest text is around {max_font:.0f} pixels and the "
            f"median typographic size is {median_font:.0f} pixels."
        )
    else:
        font_phrase = "Typographic sizing could not be measured."

    # Crude regional summary: split the viewport into top/middle/bottom thirds
    # and count which third holds the heaviest content area.
    thirds = [0.0, 0.0, 0.0]
    for e in elements:
        b = e.get("bbox") or {}
        y = float(b.get("y", 0.0))
        h = float(b.get("height", 0.0))
        cy = y + h / 2.0
        idx = 0 if cy < vh / 3 else (1 if cy < 2 * vh / 3 else 2)
        thirds[idx] += float(b.get("width", 0.0)) * h
    region_names = ["upper third", "middle band", "lower third"]
    heaviest = region_names[max(range(3), key=lambda i: thirds[i])]

    parts = [
        f"The screenshot captures {n} visible elements across a "
        f"{vw} by {vh} pixel viewport.",
        f"Roughly {n_interactive} elements are interactive and {n_ctas} of "
        f"them are prominent calls to action.",
        f"There are {n_headings} headings and {n_images} embedded images or "
        "icon-style graphics.",
        font_phrase,
        f"The bulk of the visible content sits in the {heaviest} of the page.",
    ]
    return " ".join(parts)


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"


def build_tribe_text(
    rendered: dict[str, Any],
    *,
    use_vision: bool = True,
    max_total_chars: int = 4000,
) -> tuple[str, dict[str, Any]]:
    """
    Build the text payload for TRIBE v2 from a RenderedArtifact.

    Args:
        rendered: a RenderedArtifact dict (must contain ``visible_text`` and
            ``elements``; if a ``screenshot_path`` is present and
            ``use_vision`` is True, a vision LLM is also asked to describe
            the screenshot).
        use_vision: when False (or no API key), skip the vision call. Useful
            for offline tests and to keep the demo cheap.
        max_total_chars: hard upper bound on the produced text. TRIBE's word
            window is large but not infinite; we keep the whole thing under
            ~4k chars by default so inference stays fast.

    Returns:
        ``(text, debug)`` where ``text`` is the concatenated description
        ready to be written to a ``.txt`` file, and ``debug`` exposes which
        components were used (vision_used, vision_chars, layout_chars,
        body_chars, total_chars, model). Useful for the dashboard panel.
    """
    visible_text = (rendered.get("visible_text") or "").strip()
    screenshot = rendered.get("screenshot_path") or ""

    layout = _layout_summary(rendered)
    description: str | None = None
    vision_used = False
    vision_model = None

    can_vision = (
        use_vision
        and bool(screenshot)
        and os.path.isfile(screenshot)
        and llm_client.is_configured()
    )
    if can_vision:
        vision_model = llm_client.default_model()
        description = llm_client.describe_image(screenshot)
        if description:
            vision_used = True

    pieces: list[str] = []
    if vision_used and description:
        pieces.append("Visual description. " + description)
    pieces.append("Layout summary. " + layout)
    if visible_text:
        pieces.append("On-page text. " + visible_text)

    text = "\n\n".join(pieces).strip()
    text = _truncate_text(text, max_total_chars)

    debug = {
        "vision_used": vision_used,
        "vision_model": vision_model,
        "vision_chars": len(description or ""),
        "layout_chars": len(layout),
        "body_chars": len(visible_text),
        "total_chars": len(text),
    }
    return text, debug
