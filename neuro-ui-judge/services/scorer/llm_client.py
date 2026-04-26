"""
Single LLM client for NeuroUI Judge.

All LLM calls in the project go through OpenRouter. This module centralises
the configuration so we never accidentally call the OpenAI / Anthropic APIs
directly. Both pure-text completions and vision (image+text) completions are
supported through ``chat()`` and ``describe_image()``.

Environment (loaded by ``env_loader.py`` from the project ``.env`` on import):
  OPENROUTER_API_KEY       — required for any real call. If unset, helpers
                             return None and callers should degrade gracefully.
  OPENROUTER_MODEL         — default model slug used by every helper. Pick a
                             vision-capable model so describe_image() works.
                             Defaults to "openai/gpt-4.1-mini".
  OPENROUTER_APP_NAME      — X-Title header for OpenRouter attribution.
  NEXT_PUBLIC_APP_URL      — HTTP-Referer header (used by OpenRouter rate
                             limiting / analytics).

Legacy env names are still honoured for backwards compatibility:
  NEUROUI_LLM_MODEL        — overridden by OPENROUTER_MODEL.
  NEUROUI_LLM_TITLE        — overridden by OPENROUTER_APP_NAME.
  NEUROUI_LLM_REFERER      — overridden by NEXT_PUBLIC_APP_URL.

Public API:
    is_configured() -> bool
    default_model() -> str
    chat(messages, *, model=None, temperature=0.5, max_tokens=None) -> str | None
    describe_image(image_path, *, prompt=None, model=None, max_tokens=600) -> str | None
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _env_first(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable in `names`."""
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return default


def default_model() -> str:
    """Resolve the active OpenRouter model slug at call time (not import time)."""
    return _env_first(
        "OPENROUTER_MODEL", "NEUROUI_LLM_MODEL", default="openai/gpt-4.1-mini"
    )


def _referer() -> str:
    return _env_first(
        "NEXT_PUBLIC_APP_URL",
        "NEUROUI_LLM_REFERER",
        default="http://localhost:8000",
    )


def _title() -> str:
    return _env_first(
        "OPENROUTER_APP_NAME", "NEUROUI_LLM_TITLE", default="NeuroUI Judge"
    )


def is_configured() -> bool:
    """True iff OPENROUTER_API_KEY is set."""
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def _client():
    """Lazily build an OpenAI-compatible client pointed at OpenRouter."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        logger.warning("openai SDK not installed: %s", exc)
        return None
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def chat(
    messages: Iterable[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int | None = None,
) -> str | None:
    """
    Run a chat completion through OpenRouter.

    Returns the assistant message content, or None on any failure (no API
    key, no network, model error). Callers must handle None.
    """
    client = _client()
    if client is None:
        return None
    model = model or default_model()
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": list(messages),
            "extra_headers": {
                "HTTP-Referer": _referer(),
                "X-Title": _title(),
            },
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("OpenRouter chat call failed (model=%s): %s", model, exc)
        return None


# ── Vision helper ──────────────────────────────────────────────────────────


def _encode_image_data_url(image_path: str) -> str | None:
    """Read an image and return a `data:` URL suitable for OpenRouter vision."""
    p = Path(image_path)
    if not p.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "image/png"
    try:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except OSError as exc:
        logger.warning("Could not read image %s: %s", p, exc)
        return None
    return f"data:{mime};base64,{b64}"


def describe_image(
    image_path: str,
    *,
    prompt: str | None = None,
    model: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> str | None:
    """
    Ask an OpenRouter vision model to produce a rich textual description of
    an image (typically a UI screenshot). Used by NeuroUI Judge to feed
    TRIBE v2 — TRIBE is text/audio/video-only, so converting the screenshot
    to a careful description gives TRIBE's language and reading networks
    something to predict on.

    Returns the description text, or None on failure (callers should fall
    back to the page's raw visible_text).
    """
    data_url = _encode_image_data_url(image_path)
    if data_url is None:
        return None
    client = _client()
    if client is None:
        return None
    model = model or default_model()
    user_prompt = prompt or (
        "You are describing a screenshot of a software UI for a "
        "neuroscience encoding model that maps text to predicted brain "
        "responses. Produce a concrete, dense paragraph (200-350 words) "
        "covering — in this order — the visual layout, dominant typographic "
        "hierarchy, color palette and contrast, primary calls to action, "
        "any prominent imagery or icons, the apparent task the user would "
        "perform, and the textual content visible. Avoid markdown, lists, "
        "or speculation about brand identity. Write in the third person."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            extra_headers={
                "HTTP-Referer": _referer(),
                "X-Title": _title(),
            },
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning(
            "OpenRouter describe_image failed (model=%s): %s", model, exc
        )
        return None
