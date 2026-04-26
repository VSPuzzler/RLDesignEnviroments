"""
Brain response prediction using Meta's Tribe V2 model.

Tribe V2 predicts cortical fMRI activations from images.
Output shape: (T, 20484) — time steps × fsaverage5 vertices.

This module handles model loading, inference, and ROI extraction.
If `tribev2` is not installed, a deterministic simulation mode is used so the
rest of the pipeline can be exercised without GPU/model access.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any

import numpy as np

from config import (
    ATN_VERTICES,
    DMN_VERTICES,
    FFA_VERTICES,
    PFC_VERTICES,
    REWARD_VERTICES,
    TRIBE_MODEL_ID,
    TRIBE_N_VERTICES,
    V1_VERTICES,
    V4_VERTICES,
)

logger = logging.getLogger(__name__)

# ── Model singleton ───────────────────────────────────────────────────────────

_tribe_model: Any = None
_simulation_mode: bool = False


def _load_model() -> None:
    """Load Tribe V2 once; fall back to simulation if unavailable."""
    global _tribe_model, _simulation_mode

    tribev2_spec = importlib.util.find_spec("tribev2")
    if tribev2_spec is None:
        logger.warning(
            "tribev2 package not found. Running in SIMULATION mode – "
            "brain activations will be deterministically synthesised."
        )
        _simulation_mode = True
        return

    import tribev2  # type: ignore[import]

    logger.info("Loading Tribe V2 model: %s …", TRIBE_MODEL_ID)
    _tribe_model = tribev2.load_model(TRIBE_MODEL_ID)
    logger.info("Tribe V2 ready.")


def _ensure_loaded() -> None:
    if _tribe_model is None and not _simulation_mode:
        _load_model()


# ── Simulation ────────────────────────────────────────────────────────────────

def _simulate_brain_response(image_path: str) -> np.ndarray:
    """
    Return a synthetic (T=8, 20484) activation matrix derived from the image.

    Uses a hash of the file path as the RNG seed so repeated calls for the
    same file are deterministic, while different files produce different values.
    """
    seed = int(hash(image_path) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)

    T = 8
    base = rng.standard_normal((T, TRIBE_N_VERTICES)).astype(np.float32)

    # Inject a region-specific signal to make ROI values meaningful
    visual_boost = rng.uniform(0.5, 1.5)
    base[:, V1_VERTICES] += visual_boost
    base[:, V4_VERTICES] += visual_boost * 0.8
    base[:, FFA_VERTICES] += rng.uniform(0.2, 1.0)
    base[:, REWARD_VERTICES] += rng.uniform(-0.5, 1.0)
    base[:, PFC_VERTICES] += rng.uniform(-0.3, 0.8)
    base[:, DMN_VERTICES] += rng.uniform(-0.4, 0.6)
    base[:, ATN_VERTICES] += rng.uniform(0.0, 0.9)

    return base


# ── ROI extraction ────────────────────────────────────────────────────────────

def extract_roi_activations(brain_pred: np.ndarray) -> dict[str, float]:
    """
    Summarise a (T, 20484) activation matrix into per-ROI mean activations.

    Time is averaged first, then vertices within each ROI are averaged to
    produce a single scalar per region.

    Args:
        brain_pred: Float array of shape (T, 20484).

    Returns:
        Dict mapping ROI name → mean activation (float).
    """
    avg: np.ndarray = brain_pred.mean(axis=0)  # (20484,)

    return {
        "visual_cortex_v1": float(avg[V1_VERTICES].mean()),
        "visual_cortex_v4": float(avg[V4_VERTICES].mean()),
        "fusiform_face_area": float(avg[FFA_VERTICES].mean()),
        "reward_pathway": float(avg[REWARD_VERTICES].mean()),
        "prefrontal_cortex": float(avg[PFC_VERTICES].mean()),
        "default_mode_network": float(avg[DMN_VERTICES].mean()),
        "attention_network": float(avg[ATN_VERTICES].mean()),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def predict_brain_response(image_path: str) -> dict[str, float]:
    """
    Run Tribe V2 on a screenshot and return ROI activations.

    Args:
        image_path: Path to the PNG screenshot.

    Returns:
        Dict of ROI name → mean activation scalar.
    """
    _ensure_loaded()

    if _simulation_mode:
        logger.debug("Simulation mode: synthesising brain response for %s", image_path)
        raw = _simulate_brain_response(image_path)
    else:
        logger.info("Running Tribe V2 inference on %s …", image_path)
        # tribev2 API: model.predict returns np.ndarray (T, 20484)
        raw = _tribe_model.predict(image_path=image_path)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]  # treat as single time step

    roi = extract_roi_activations(raw)
    logger.info("ROI activations: %s", roi)
    return roi
