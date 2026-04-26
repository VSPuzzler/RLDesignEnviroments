"""Generate HTML/CSS UI variants via OpenRouter API."""

import logging
import re
from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, GENERATION_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert UI/UX designer and front-end developer.
Your output must be a single, self-contained HTML file with inline CSS and no
external dependencies.  The page should look polished and professional at
1920×1080.  Respond with ONLY the HTML – no markdown fences, no commentary.\
"""

_USER_TEMPLATE = """\
Design spec: {spec}

Variant style hint: {style_hint}

Produce a complete, renderable HTML page that fulfils the spec.
Use only inline <style> – no external stylesheets or scripts.\
"""

_STYLE_HINTS: dict[int, str] = {
    1: (
        "Clean, minimal, lots of white space, subtle shadows, "
        "cool blue–grey colour palette, sans-serif typography."
    ),
    2: (
        "Bold, vibrant, full-bleed hero images (use CSS gradients), "
        "warm accent colours, strong typographic hierarchy, "
        "slightly playful but still professional."
    ),
}


def _extract_html(raw: str) -> str:
    """Strip accidental markdown fences from the model response."""
    # Remove ```html ... ``` wrappers if present
    match = re.search(r"```(?:html)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()


def generate_ui_variant(spec: str, variant_seed: int) -> str:
    """
    Call OpenRouter to generate a self-contained HTML/CSS UI variant.

    Args:
        spec:         Natural-language description of the desired UI.
        variant_seed: 1 or 2 – controls temperature and style hint so the
                      two variants are meaningfully different.

    Returns:
        HTML string ready to be written to a file and rendered by Playwright.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Export it before running."
        )

    # Variant 1 → lower temperature (precise / conservative)
    # Variant 2 → higher temperature (creative / expressive)
    temperature = 0.3 if variant_seed == 1 else 0.9
    style_hint = _STYLE_HINTS.get(variant_seed, "Default clean style.")

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    logger.info(
        "Generating UI variant %d (temp=%.1f, model=%s) …",
        variant_seed,
        temperature,
        GENERATION_MODEL,
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    spec=spec, style_hint=style_hint
                ),
            },
        ],
        extra_headers={
            "HTTP-Referer": "https://github.com/vspuzzler/rldesignenviroments",
            "X-Title": "UI Brain Preference",
        },
    )

    raw = response.choices[0].message.content or ""
    html = _extract_html(raw)

    logger.info(
        "Variant %d generated (%d chars).", variant_seed, len(html)
    )
    return html
