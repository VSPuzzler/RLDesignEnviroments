"""
HTTP backend that delegates to the TRIBE v2 sidecar service.

This is the production path for real cortical predictions. The sidecar runs
TRIBE in its own Python 3.11 conda env (see ``services/tribe_v2_sidecar/``);
the main app talks to it over HTTP so the heavy ML stack never has to be
importable in the main venv.

The backend implements the ``TribeBackend`` Protocol from ``tribe_adapter``
and returns the same ``NeuralProxyFeatures``-shaped dict the mock returns,
with two additions:

  - ``mode = "tribe_v2"``
  - ``vertex_activation``: length-20484 list of [0, 1] floats (the cortical
    heatmap consumed by the dashboard's 3D brain renderer)

On any error (sidecar down, timeout, malformed response) we log a warning
and raise so ``tribe_adapter.predict()`` can fall back to the mock — the
demo never breaks, it just shows synthesised activations on the same mesh.
"""

from __future__ import annotations

import logging
import os
import statistics
from typing import Any

from . import describer

logger = logging.getLogger(__name__)


DEFAULT_SIDECAR_URL = os.getenv(
    "TRIBE_V2_SERVICE_URL", "http://localhost:7860"
)
DEFAULT_TIMEOUT = float(os.getenv("TRIBE_V2_TIMEOUT", "120"))


def _ping(sidecar_url: str, timeout: float = 1.5) -> dict[str, Any] | None:
    """Lightweight liveness check. Returns the parsed /health body or None."""
    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("httpx not installed; cannot reach TRIBE sidecar.")
        return None
    try:
        resp = httpx.get(f"{sidecar_url.rstrip('/')}/health", timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as exc:  # pragma: no cover - network dependent
        logger.debug("TRIBE sidecar /health failed: %s", exc)
        return None


class TribeV2HttpBackend:
    """
    ``TribeBackend`` implementation that calls the sidecar over HTTP.

    Construction is cheap; the actual TRIBE model is loaded lazily inside
    the sidecar on its first inference call.
    """

    def __init__(
        self,
        sidecar_url: str = DEFAULT_SIDECAR_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        use_vision_describer: bool = True,
        max_segments: int = 48,
    ) -> None:
        self.sidecar_url = sidecar_url.rstrip("/")
        self.timeout = float(timeout)
        self.use_vision_describer = bool(use_vision_describer)
        self.max_segments = int(max_segments)

    # ── Public API ─────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        return _ping(self.sidecar_url) is not None

    def predict(
        self,
        rendered: dict[str, Any],
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run TRIBE v2 on the rendered candidate and return a
        NeuralProxyFeatures-shaped dict.

        Falls back to a vanilla ``visible_text`` payload if the vision
        describer is disabled or unavailable. Raises ``RuntimeError`` on
        sidecar failure so the adapter can use the mock as a safety net.
        """
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("httpx is required for TRIBE HTTP backend") from exc

        text, debug = describer.build_tribe_text(
            rendered, use_vision=self.use_vision_describer
        )
        if not text.strip():
            raise RuntimeError("TRIBE input text is empty after describer")

        payload = {
            "text": text,
            "candidate_id": rendered.get("candidate_id"),
            "max_segments": self.max_segments,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.sidecar_url}/predict-text-with-rois", json=payload
                )
        except Exception as exc:
            raise RuntimeError(
                f"TRIBE sidecar unreachable at {self.sidecar_url}: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"TRIBE sidecar returned {resp.status_code}: {resp.text[:300]}"
            )
        body = resp.json()
        return self._to_neural_proxy_features(body, audit, debug)

    # ── Internal mapping ──────────────────────────────────────────────────

    @staticmethod
    def _confidence(roi_features: dict[str, Any]) -> dict[str, float]:
        """
        Confidence vector for the consumer (reward_model) given a real
        TRIBE prediction. We deliberately keep aesthetic / accessibility
        below 1.0 because:

          - aesthetic / valuation_proxy uses a cortical readout near vmPFC;
            cortical-only valuation prediction is well-known to be weak.
          - accessibility is a deterministic standards check; neural inference
            should never dominate it.

        Attention/load are raised vs. the mock because the encoder really is
        running, not synthesising.
        """
        # Sanity penalty: if every ROI is the same the prediction is degenerate.
        aucs = [float(v.get("auc", 0.5)) for v in roi_features.values()]
        spread = statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
        signal = min(1.0, spread * 4.0)  # ~0.25 spread → confidence 1.0
        return {
            "attention": float(0.65 + 0.25 * signal),
            "load": float(0.65 + 0.25 * signal),
            "aesthetic": 0.30,
            "accessibility": 0.10,
        }

    def _to_neural_proxy_features(
        self,
        body: dict[str, Any],
        audit: dict[str, Any] | None,
        describer_debug: dict[str, Any],
    ) -> dict[str, Any]:
        roi_in: dict[str, dict[str, Any]] = body.get("roi_features") or {}
        # Normalise into our schema (dropping None fields the schema treats as optional).
        roi_features: dict[str, dict[str, Any]] = {}
        for name, feats in roi_in.items():
            entry = {
                "auc": float(feats.get("auc", 0.5)),
                "peak": float(feats.get("peak", feats.get("auc", 0.5))),
                "variance": float(feats.get("variance", 0.0)),
            }
            if "suppression" in feats and feats["suppression"] is not None:
                entry["suppression"] = float(feats["suppression"])
            if name == "valuation_proxy":
                entry["confidence"] = "low"
            roi_features[name] = entry

        return {
            "mode": "tribe_v2",
            "roi_features": roi_features,
            "confidence": self._confidence(roi_features),
            "vertex_activation": list(body.get("vertex_activation") or []),
            "n_segments": int(body.get("n_segments") or 0),
            "describer": describer_debug,
            "notes": (
                "Real facebook/tribev2 inference. Predictions are on the "
                "fsaverage5 cortical surface (~20484 vertices). "
                "Aesthetic / valuation_proxy is treated as low-confidence "
                "even with the model running; subcortical structures are not "
                "exposed by the public predict() API in this release."
            ),
        }


# ── Auto-registration helper ────────────────────────────────────────────────


def auto_register(adapter_module) -> bool:
    """
    Try to wire ``TribeV2HttpBackend`` into the given ``tribe_adapter``
    module if the sidecar is alive. Returns True on success.

    Designed to be called from FastAPI's startup hook so the dashboard
    badge flips from ``mock`` → ``tribe_v2`` automatically.
    """
    if os.getenv("NEUROUI_DISABLE_TRIBE", "").strip() == "1":
        logger.info("NEUROUI_DISABLE_TRIBE=1 — skipping real TRIBE registration.")
        return False
    sidecar_url = os.getenv("TRIBE_V2_SERVICE_URL", "").strip() or DEFAULT_SIDECAR_URL
    if not sidecar_url:
        return False
    health = _ping(sidecar_url)
    if health is None:
        if os.getenv("TRIBE_V2_REQUIRED", "").strip() == "1":
            raise RuntimeError(
                f"TRIBE_V2_REQUIRED=1 but sidecar at {sidecar_url} is unreachable."
            )
        logger.info(
            "TRIBE sidecar at %s not reachable — staying in mock mode.",
            sidecar_url,
        )
        return False
    backend = TribeV2HttpBackend(sidecar_url)
    adapter_module.set_backend(backend)
    logger.info(
        "Registered TribeV2HttpBackend (sidecar=%s, model_loaded=%s, n_vertices=%s).",
        sidecar_url,
        health.get("loaded"),
        health.get("n_vertices"),
    )
    return True
