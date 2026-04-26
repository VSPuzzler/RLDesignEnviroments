"""Render HTML strings to PNG screenshots using Playwright."""

import logging
import os
import tempfile
from typing import TypedDict

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)


class Viewport(TypedDict):
    width: int
    height: int


_DEFAULT_VIEWPORT: Viewport = {"width": 1920, "height": 1080}


def screenshot_html(
    html_code: str,
    output_path: str,
    viewport: Viewport = _DEFAULT_VIEWPORT,
) -> str:
    """
    Render an HTML string to a PNG screenshot.

    A temporary file is created to serve the HTML locally so that
    Playwright's ``file://`` loader handles relative resources correctly.

    Args:
        html_code:   Complete HTML document to render.
        output_path: Destination path for the PNG file (created if absent).
        viewport:    Browser viewport dimensions.  Defaults to 1920×1080.

    Returns:
        Absolute path to the saved PNG file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write HTML to a temp file so Playwright loads it via file:// URL.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(html_code)
        tmp_path = tmp.name

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": viewport["width"], "height": viewport["height"]}
            )
            page.goto(f"file://{tmp_path}", wait_until="networkidle")
            page.screenshot(path=output_path, full_page=False)
            browser.close()
    finally:
        os.unlink(tmp_path)

    abs_path = os.path.abspath(output_path)
    logger.info("Screenshot saved → %s", abs_path)
    return abs_path
