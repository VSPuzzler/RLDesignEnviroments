"""
FastAPI sidecar that runs Meta's facebook/tribev2 model.

The sidecar is intentionally minimal and runs in its own Python 3.11 conda
environment (named ``neuroui-tribe`` in the project setup) because TRIBE v2's
PyTorch and transformers pins are incompatible with the main application's
Python 3.14 venv. The main NeuroUI Judge app talks to this service over
HTTP instead of trying to import TRIBE in-process.

Run with::

    conda activate neuroui-tribe
    cd neuro-ui-judge
    uvicorn services.tribe_v2_sidecar.main:app --host 0.0.0.0 --port 7860

The first ``POST /predict-text-with-rois`` will trigger TRIBE's lazy weight
download (~12 GB across TRIBE itself, LLaMA-3.2-3B, V-JEPA2-Giant, and
W2v-BERT). Subsequent calls reuse the singleton model in memory.

Endpoints:
    GET  /health                         — liveness + load state
    POST /predict-text                   — raw (T, 20484) vertex predictions
    POST /predict-text-with-rois         — predictions + 7-channel ROI summary
                                           in NeuroUI Judge's schema

Reference: https://huggingface.co/facebook/tribev2
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("tribev2-sidecar")
logging.basicConfig(level=logging.INFO)


# ── ROI vertex ranges (approximate fsaverage5) ──────────────────────────────
# These mirror NeuroUI Judge's 7 ROI channels. Indices are *approximate*
# functional ranges on the fsaverage5 surface (10242 LH + 10242 RH = 20484
# total). Replace with HCP MMP 1.0 / Glasser parcels for production use.

N_VERTICES = 20484
VERTICES_PER_HEMI = 10242


def _bilateral(lh_start: int, lh_end: int) -> list[int]:
    rh_offset = VERTICES_PER_HEMI
    return list(range(lh_start, lh_end)) + list(
        range(rh_offset + lh_start, rh_offset + lh_end)
    )


# Each entry: roi_name -> list of vertex indices to mean over.
ROI_VERTICES: dict[str, list[int]] = {
    "visual": _bilateral(1000, 1500),                           # occipital pole
    "language_vwfa": _bilateral(2500, 3000),                    # ventral occipitotemporal
    "dorsal_attention": _bilateral(6500, 7000),                 # IPS / FEF
    "multiple_demand": _bilateral(7000, 7500),                  # dlPFC
    "salience": _bilateral(5500, 5800),                         # insula proxy
    "valuation_proxy": _bilateral(5800, 6000),                  # OFC / vmPFC proxy
    "dmn": (
        _bilateral(8000, 8200)        # medial PFC
        + _bilateral(3000, 3200)      # PCC
        + _bilateral(4000, 4200)      # angular gyrus
    ),
}


# ── Request / response schemas ──────────────────────────────────────────────


class PredictRequest(BaseModel):
    text: str = Field(..., description="Free-form text. TRIBE will TTS+transcribe.")
    candidate_id: str | None = None
    max_segments: int = Field(48, gt=0, le=240)


class Activation(BaseModel):
    segment_index: int
    start_second: float
    end_second: float
    vertex_activation: list[float]  # length-20484 normalised to [0, 1]


class PredictResponse(BaseModel):
    model: str
    n_vertices: int
    n_segments: int
    segments: list[Activation]


class ROIFeatures(BaseModel):
    auc: float
    peak: float | None = None
    variance: float | None = None
    suppression: float | None = None


class ROIResponse(BaseModel):
    model: str
    mode: str = "tribe_v2"
    n_vertices: int
    n_segments: int
    candidate_id: str | None = None
    # Raw mean activation (T, 20484) collapsed across time → length-20484, [0,1].
    vertex_activation: list[float]
    # 7-channel ROI summary in NeuroUI Judge's schema.
    roi_features: dict[str, ROIFeatures]
    notes: str = (
        "TRIBE v2 prediction averaged over kept segments and projected to "
        "approximate fsaverage5 ROI ranges. Aesthetic / valuation_proxy is "
        "still treated as low-confidence by the consumer."
    )


# ── Model singleton ─────────────────────────────────────────────────────────


_MODEL: Any | None = None
_MODEL_LOCK = threading.Lock()
_LOAD_ERROR: str | None = None
_LOAD_STARTED_AT: float | None = None
_LOAD_FINISHED_AT: float | None = None


def _get_model() -> Any:
    """Lazy singleton load. Raises HTTPException(503) if TRIBE isn't installed."""
    global _MODEL, _LOAD_ERROR, _LOAD_STARTED_AT, _LOAD_FINISHED_AT

    if _MODEL is not None:
        return _MODEL
    if _LOAD_ERROR is not None:
        raise HTTPException(status_code=503, detail=_LOAD_ERROR)

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from tribev2 import TribeModel  # type: ignore[import-not-found]
        except ImportError as exc:
            _LOAD_ERROR = (
                "tribev2 package is not importable in this Python "
                "environment. Activate the conda env and install it: "
                "`conda activate neuroui-tribe && pip install -e external/tribev2`. "
                f"({exc})"
            )
            raise HTTPException(status_code=503, detail=_LOAD_ERROR) from exc

        cache_dir = os.environ.get(
            "TRIBE_CACHE_DIR", str(Path.home() / ".cache" / "tribev2")
        )
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Loading facebook/tribev2 (cache=%s) …", cache_dir)
        _LOAD_STARTED_AT = time.monotonic()
        try:
            _MODEL = TribeModel.from_pretrained(
                "facebook/tribev2", cache_folder=cache_dir
            )
            _LOAD_FINISHED_AT = time.monotonic()
            logger.info(
                "TRIBE v2 ready in %.1fs.",
                _LOAD_FINISHED_AT - _LOAD_STARTED_AT,
            )
        except Exception as exc:
            _LOAD_ERROR = f"TRIBE v2 load failed: {exc}"
            raise HTTPException(status_code=503, detail=_LOAD_ERROR) from exc

    return _MODEL


# ── Inference helpers ──────────────────────────────────────────────────────


def _predict_array(text: str) -> np.ndarray:
    """Run TRIBE on a text string, return ``(n_segments, n_vertices)``."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    model = _get_model()

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        text_path = tmp.name
    try:
        df = model.get_events_dataframe(text_path=text_path)
        preds, _segments = model.predict(events=df)
    finally:
        Path(text_path).unlink(missing_ok=True)

    arr = np.asarray(preds, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[-1] != N_VERTICES:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unexpected TRIBE output shape {arr.shape}; "
                f"expected last dim {N_VERTICES}."
            ),
        )
    return arr


def _normalise_per_segment(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise each (vertex) row to [0, 1] independently per segment."""
    a_min = arr.min(axis=1, keepdims=True)
    a_max = arr.max(axis=1, keepdims=True)
    denom = np.maximum(a_max - a_min, 1e-6)
    return (arr - a_min) / denom


def _downsample(
    arr: np.ndarray, max_segments: int
) -> np.ndarray:
    """Linearly subsample rows to at most ``max_segments``."""
    n = arr.shape[0]
    if n <= max_segments:
        return arr
    idx = np.linspace(0, n - 1, max_segments).astype(int)
    return arr[idx]


def _summarise_rois(arr_norm_collapsed: np.ndarray, arr_norm_per_segment: np.ndarray) -> dict[str, dict[str, float]]:
    """
    Build the 7-channel ROI summary expected by reward_model.

    Args:
        arr_norm_collapsed: length-20484 mean activation vector in [0, 1].
        arr_norm_per_segment: ``(n_segments, 20484)`` for variance computation.
    """
    out: dict[str, dict[str, float]] = {}
    for roi, idxs in ROI_VERTICES.items():
        idx = np.asarray(idxs, dtype=np.int64)
        idx = idx[idx < arr_norm_collapsed.shape[0]]
        if idx.size == 0:
            out[roi] = {"auc": 0.5, "peak": 0.5, "variance": 0.0}
            continue
        roi_mean = float(arr_norm_collapsed[idx].mean())
        roi_peak = float(arr_norm_collapsed[idx].max())
        # Temporal variance per ROI averaged over vertices in that ROI.
        roi_var = float(arr_norm_per_segment[:, idx].std(axis=0).mean())
        entry: dict[str, float] = {
            "auc": roi_mean,
            "peak": roi_peak,
            "variance": roi_var,
        }
        if roi == "dmn":
            entry["suppression"] = float(max(0.0, 1.0 - roi_mean))
        out[roi] = entry
    return out


# ── App ────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="NeuroUI Judge — TRIBE v2 sidecar",
    description="FastAPI wrapper around facebook/tribev2 for cortical predictions.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + load state. Cheap; main app polls this on startup."""
    loaded = _MODEL is not None
    load_seconds: float | None = None
    if _LOAD_STARTED_AT is not None and _LOAD_FINISHED_AT is not None:
        load_seconds = round(_LOAD_FINISHED_AT - _LOAD_STARTED_AT, 2)
    return {
        "status": "ok",
        "model": "facebook/tribev2",
        "loaded": loaded,
        "load_seconds": load_seconds,
        "load_error": _LOAD_ERROR,
        "n_vertices": N_VERTICES,
        "rois": list(ROI_VERTICES.keys()),
    }


@app.post("/predict-text", response_model=PredictResponse)
def predict_text(req: PredictRequest) -> PredictResponse:
    arr = _predict_array(req.text)
    arr_norm = _normalise_per_segment(arr)
    arr_norm = _downsample(arr_norm, req.max_segments)

    activations = [
        Activation(
            segment_index=i,
            start_second=float(i),
            end_second=float(i + 1),
            vertex_activation=arr_norm[i].astype(float).tolist(),
        )
        for i in range(arr_norm.shape[0])
    ]
    return PredictResponse(
        model="facebook/tribev2",
        n_vertices=int(arr_norm.shape[1]),
        n_segments=int(arr_norm.shape[0]),
        segments=activations,
    )


@app.post("/predict-text-with-rois", response_model=ROIResponse)
def predict_text_with_rois(req: PredictRequest) -> ROIResponse:
    arr = _predict_array(req.text)
    arr_norm = _normalise_per_segment(arr)
    arr_norm = _downsample(arr_norm, req.max_segments)

    # Mean across kept segments → single length-20484 activation map.
    collapsed = arr_norm.mean(axis=0)
    # Re-normalise to [0, 1] for the heatmap.
    c_min, c_max = float(collapsed.min()), float(collapsed.max())
    if c_max - c_min < 1e-6:
        collapsed_norm = np.zeros_like(collapsed)
    else:
        collapsed_norm = (collapsed - c_min) / (c_max - c_min)

    rois = _summarise_rois(collapsed_norm, arr_norm)
    return ROIResponse(
        model="facebook/tribev2",
        mode="tribe_v2",
        n_vertices=int(collapsed_norm.shape[0]),
        n_segments=int(arr_norm.shape[0]),
        candidate_id=req.candidate_id,
        vertex_activation=collapsed_norm.astype(float).tolist(),
        roi_features={k: ROIFeatures(**v) for k, v in rois.items()},
    )
