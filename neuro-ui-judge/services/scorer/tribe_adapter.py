"""
Pluggable adapter for real TRIBE v2 inference.

This file defines the contract a real TRIBE v2 backend must satisfy and
ships an importable default that delegates to the mock proxy. To wire in
the actual model:

    1. Implement the `TribeBackend` Protocol in a new module (e.g.
       `tribe_v2_backend.py`) with a real `predict(rendered, audit)` that
       runs TRIBE v2 on the rendered video / text / (optional audio) and
       returns the same `NeuralProxyFeatures`-shaped dict as the mock.
    2. Set the env var `NEUROUI_TRIBE_BACKEND="my_module:MyBackend"` (or
       call `set_backend(MyBackend())` programmatically).
    3. The reward model and dashboard automatically use the real outputs.

Important: the schema this returns *must remain identical* between mock and
real modes; only the `mode` field flips from "mock" to "tribe_v2", and the
confidence values can be raised by a backend that has been validated against
human data. The reward model intentionally treats `aesthetic`/`valuation`
as low-confidence by default — a real TRIBE backend should justify any
upward adjustment with a citation.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Protocol, runtime_checkable

from . import neural_proxy_mock

logger = logging.getLogger(__name__)


@runtime_checkable
class TribeBackend(Protocol):
    """
    Contract any TRIBE backend must satisfy.

    Args:
        rendered: a `RenderedArtifact`-shaped dict. Will include
            ``frames_dir`` (directory of PNG frames) when video input is
            available; otherwise the backend should fall back to single-image
            inference using ``screenshot_path``.
        audit: deterministic audit dict; backends may use it as a side
            input or ignore it.

    Returns:
        a `NeuralProxyFeatures`-shaped dict with ``mode == "tribe_v2"``.
    """

    def predict(
        self,
        rendered: dict[str, Any],
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class _MockBackend:
    """Default backend: delegates to neural_proxy_mock."""

    def predict(
        self,
        rendered: dict[str, Any],
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return neural_proxy_mock.predict_neural_proxy(rendered, audit)


_backend: TribeBackend = _MockBackend()


def set_backend(backend: TribeBackend) -> None:
    """Override the active backend at runtime."""
    global _backend
    if not isinstance(backend, TribeBackend):
        raise TypeError("backend must implement the TribeBackend protocol")
    _backend = backend
    logger.info("TRIBE backend set to %s", backend.__class__.__name__)


def _maybe_load_env_backend() -> None:
    """Optionally load a backend from `NEUROUI_TRIBE_BACKEND=module:Class`."""
    spec = os.getenv("NEUROUI_TRIBE_BACKEND", "").strip()
    if not spec:
        return
    if ":" not in spec:
        logger.warning("NEUROUI_TRIBE_BACKEND must be 'module:Class', got %r", spec)
        return
    try:
        mod_name, cls_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        set_backend(cls())
    except Exception as exc:  # pragma: no cover - configuration dependent
        logger.warning("Failed to load TRIBE backend %r: %s", spec, exc)


_maybe_load_env_backend()


_FALLBACK_BACKEND = _MockBackend()


def predict(
    rendered: dict[str, Any],
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the active TRIBE backend (real or mock) on a rendered artifact.

    If the active backend raises (e.g. sidecar unreachable mid-run) we log
    a warning and fall back to the deterministic mock so the demo keeps
    rendering.
    """
    backend = _backend
    if isinstance(backend, _MockBackend):
        return backend.predict(rendered, audit)
    try:
        return backend.predict(rendered, audit)
    except Exception as exc:
        logger.warning(
            "Active TRIBE backend %s failed (%s); falling back to mock.",
            backend.__class__.__name__,
            exc,
        )
        result = _FALLBACK_BACKEND.predict(rendered, audit)
        result["mode"] = "mock"  # explicit downgrade for the dashboard badge
        result["notes"] = (
            "Real backend failed mid-run; this report uses the deterministic "
            f"mock proxy. Original error: {exc}"
        )
        return result


def active_mode() -> str:
    """Return ``"mock"`` or ``"tribe_v2"`` depending on the active backend."""
    return "mock" if isinstance(_backend, _MockBackend) else "tribe_v2"


def reset_to_mock() -> None:
    """Drop the active backend and revert to the deterministic mock."""
    global _backend
    _backend = _MockBackend()
    logger.info("TRIBE backend reset to mock.")
