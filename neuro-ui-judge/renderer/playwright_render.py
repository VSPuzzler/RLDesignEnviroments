"""
Playwright-based renderer for NeuroUI Judge.

Captures a rich `RenderedArtifact` per candidate:
  - PNG screenshot at a fixed viewport
  - Optional frame sequence (for short interaction episodes)
  - Visible-text dump
  - Per-element record: tag, role, accessible name, bbox, font size, colors,
    interactive flag, CTA flag, alt/label presence, tab index
  - Lightweight page metrics (DOM size, paint timings if available)

The element extraction runs *inside the page* via a single ``page.evaluate``
call so we get computed styles and layout boxes consistently.

The renderer is robust to Playwright not being installed: it falls back to a
text-only stub artifact so downstream scoring still produces a (clearly
flagged) report.  This keeps the demo runnable on machines that haven't yet
installed browsers.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import so the package can be inspected without playwright installed.
try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _PLAYWRIGHT_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _PLAYWRIGHT_AVAILABLE = False


# JS that runs inside the page to extract a flattened element list.
_EXTRACT_JS = r"""
() => {
  const INTERACTIVE_TAGS = new Set(['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA']);
  const CTA_KEYWORDS = ['sign up', 'get started', 'try', 'buy', 'subscribe',
                        'start', 'download', 'join', 'continue', 'next',
                        'submit', 'create', 'go'];

  function visible(el) {
    const rect = el.getBoundingClientRect();
    if (rect.width <= 1 || rect.height <= 1) return false;
    const style = getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') return false;
    if (parseFloat(style.opacity) < 0.05) return false;
    return true;
  }

  function ctaScore(el, text, fontSize) {
    const tag = el.tagName;
    if (tag !== 'A' && tag !== 'BUTTON' && el.getAttribute('role') !== 'button')
      return 0;
    const t = (text || '').trim().toLowerCase();
    let s = 0;
    if (CTA_KEYWORDS.some(k => t.includes(k))) s += 0.6;
    if (fontSize >= 16) s += 0.2;
    const rect = el.getBoundingClientRect();
    if (rect.width >= 100 && rect.height >= 32) s += 0.2;
    return s;
  }

  const out = [];
  const all = document.querySelectorAll('body *');
  let dom_size = all.length;
  for (const el of all) {
    if (!visible(el)) continue;
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    const tag = el.tagName.toLowerCase();
    const text = (el.innerText || el.textContent || '').slice(0, 200).trim();
    // Only record elements that meaningfully contribute (have text, are
    // interactive, or are media/containers we'll measure).
    const interactive = INTERACTIVE_TAGS.has(el.tagName)
                        || el.getAttribute('role') === 'button'
                        || el.getAttribute('role') === 'link';
    const isMedia = ['IMG', 'SVG', 'CANVAS', 'VIDEO'].includes(el.tagName);
    if (!text && !interactive && !isMedia) continue;

    const fontSize = parseFloat(style.fontSize) || 0;
    const cta = ctaScore(el, text, fontSize) >= 0.6;
    const altOrLabel =
      el.tagName !== 'IMG' ||
      !!(el.getAttribute('alt') && el.getAttribute('alt').trim().length);

    out.push({
      tag,
      role: el.getAttribute('role'),
      text: text || null,
      name: el.getAttribute('aria-label') || el.getAttribute('alt') || null,
      bbox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      font_size_px: fontSize || null,
      color: style.color,
      background_color: style.backgroundColor,
      is_interactive: interactive,
      is_cta: cta,
      has_alt_or_label: altOrLabel,
      tab_index: el.tabIndex,
    });
  }
  return {
    elements: out,
    visible_text: document.body.innerText || '',
    page_metrics: {
      dom_size,
      scroll_height: document.documentElement.scrollHeight,
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
    },
  };
}
"""


def _stub_artifact(
    candidate_id: str,
    html: str,
    output_dir: str,
    viewport_width: int,
    viewport_height: int,
) -> dict[str, Any]:
    """
    Produce a minimal artifact when Playwright is unavailable.

    We do NOT pretend to have rendered anything: screenshot_path is empty
    and elements is empty.  Downstream scoring marks confidence accordingly.
    """
    logger.warning(
        "Playwright not available; producing stub artifact for candidate=%s",
        candidate_id,
    )
    return {
        "candidate_id": candidate_id,
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
        "screenshot_path": "",
        "frames_dir": None,
        "dom_tree_path": None,
        "elements": [],
        "visible_text": "",
        "accessibility_tree": None,
        "page_metrics": {"playwright_available": 0.0, "html_chars": float(len(html))},
    }


def render_html(
    candidate_id: str,
    html: str,
    output_dir: str,
    viewport_width: int = 1440,
    viewport_height: int = 900,
    capture_frames: int = 0,
    frame_interval_ms: int = 500,
) -> dict[str, Any]:
    """
    Render an HTML candidate and return a `RenderedArtifact`-shaped dict.

    Args:
        candidate_id: stable id; used to name files.
        html: full HTML document (with inline CSS).
        output_dir: directory where screenshots/frames/DOM are written.
        viewport_width / viewport_height: fixed viewport for fairness.
        capture_frames: if > 0, also save N frames spaced by
            `frame_interval_ms` (used by the neural proxy / future TRIBE).
        frame_interval_ms: spacing between frames in ms.

    Returns:
        Plain dict matching the `RenderedArtifact` schema.
    """
    os.makedirs(output_dir, exist_ok=True)
    screenshot_path = os.path.join(output_dir, f"{candidate_id}.png")
    dom_tree_path = os.path.join(output_dir, f"{candidate_id}.dom.json")

    if not _PLAYWRIGHT_AVAILABLE:
        return _stub_artifact(
            candidate_id, html, output_dir, viewport_width, viewport_height
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(html)
        tmp_path = tmp.name

    frames_dir: str | None = None
    elements: list[dict[str, Any]] = []
    visible_text = ""
    page_metrics: dict[str, float] = {"playwright_available": 1.0}
    accessibility_tree: dict[str, Any] | None = None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": viewport_width, "height": viewport_height},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.goto(f"file://{tmp_path}", wait_until="networkidle")

            page.screenshot(path=screenshot_path, full_page=False)

            if capture_frames > 0:
                frames_dir = os.path.join(output_dir, f"{candidate_id}_frames")
                os.makedirs(frames_dir, exist_ok=True)
                for i in range(capture_frames):
                    page.wait_for_timeout(frame_interval_ms)
                    page.screenshot(
                        path=os.path.join(frames_dir, f"{i:03d}.png"),
                        full_page=False,
                    )

            extract = page.evaluate(_EXTRACT_JS)
            elements = extract.get("elements", [])
            visible_text = extract.get("visible_text", "")
            page_metrics.update(extract.get("page_metrics", {}))

            try:
                snapshot = page.accessibility.snapshot()
                accessibility_tree = snapshot
            except Exception as exc:  # pragma: no cover
                logger.debug("accessibility snapshot failed: %s", exc)

            browser.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    with open(dom_tree_path, "w", encoding="utf-8") as f:
        json.dump({"elements": elements, "visible_text": visible_text}, f)

    return {
        "candidate_id": candidate_id,
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
        "screenshot_path": screenshot_path,
        "frames_dir": frames_dir,
        "dom_tree_path": dom_tree_path,
        "elements": elements,
        "visible_text": visible_text,
        "accessibility_tree": accessibility_tree,
        "page_metrics": page_metrics,
    }


def new_candidate_id() -> str:
    return uuid.uuid4().hex[:10]
