"""
Bradley–Terry preference model + reward-weight calibration.

We treat reward as a linear combination of normalised features extracted
from the candidate report:

    R(u) = sum_m w_m * subscore_m(u)

and the probability that A is preferred over B as

    P(A > B) = sigmoid((R(A) - R(B)) / tau)

We fit `w` to a set of `PairwisePreference` records by minimising mean
binary cross-entropy with L2 regularisation toward the default weights.

This is intentionally minimal — pure NumPy, no torch, no sklearn — so the
MVP runs in any Python environment and stays auditable. We use full-batch
gradient descent because real preference datasets are tiny (≤ 10⁴ pairs).
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any

import numpy as np

from .reward_model import DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)


METRIC_ORDER: list[str] = list(DEFAULT_WEIGHTS.keys())


def _featurise_report(report: dict[str, Any]) -> np.ndarray:
    """Extract the metric vector in canonical order."""
    subs = report["subscores"]
    return np.array([float(subs[m]) for m in METRIC_ORDER], dtype=np.float64)


def _build_dataset(
    preferences: list[dict[str, Any]],
    reports_by_id: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_diff, y) where X_diff = phi(A) - phi(B), y = 1 if A wins."""
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    for p in preferences:
        a_id, b_id = p["ui_a_id"], p["ui_b_id"]
        if a_id not in reports_by_id or b_id not in reports_by_id:
            continue
        winner = p.get("winner")
        if winner == "tie":
            # Encode ties as two soft examples (0.5 each direction).
            phi_a = _featurise_report(reports_by_id[a_id])
            phi_b = _featurise_report(reports_by_id[b_id])
            rows_x.append(phi_a - phi_b)
            rows_y.append(0.5)
            rows_x.append(phi_b - phi_a)
            rows_y.append(0.5)
            continue
        if winner not in ("a", "b"):
            continue
        phi_a = _featurise_report(reports_by_id[a_id])
        phi_b = _featurise_report(reports_by_id[b_id])
        rows_x.append(phi_a - phi_b)
        rows_y.append(1.0 if winner == "a" else 0.0)
    if not rows_x:
        return np.zeros((0, len(METRIC_ORDER))), np.zeros((0,))
    return np.stack(rows_x), np.array(rows_y, dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _bce(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-9
    return float(
        -np.mean(
            y_true * np.log(y_pred + eps)
            + (1.0 - y_true) * np.log(1.0 - y_pred + eps)
        )
    )


def _accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Pairwise accuracy: ignore soft (tie) labels."""
    mask = (y_true == 1.0) | (y_true == 0.0)
    if mask.sum() == 0:
        return float("nan")
    return float(((y_pred[mask] >= 0.5) == (y_true[mask] >= 0.5)).mean())


def fit_preference_weights(
    preferences: list[dict[str, Any]],
    reports_by_id: dict[str, dict[str, Any]],
    *,
    tau: float = 1.0,
    learning_rate: float = 0.5,
    n_steps: int = 600,
    l2: float = 0.05,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Fit reward weights to pairwise preferences.

    Returns:
        dict with keys:
            weights: dict[str, float] (sum-normalised, non-negative)
            metrics: PreferenceModelMetrics-shaped dict
            weights_version: str
            tau: float
    """
    X, y = _build_dataset(preferences, reports_by_id)
    n = X.shape[0]
    if n < 4:
        # Not enough data — return default weights with a warning.
        logger.warning(
            "Only %d usable pairwise rows; returning default weights.", n
        )
        return {
            "weights": dict(DEFAULT_WEIGHTS),
            "weights_version": f"default-{uuid.uuid4().hex[:6]}",
            "metrics": {
                "pairwise_accuracy": float("nan"),
                "train_loss": float("nan"),
                "val_loss": None,
                "n_train": int(n),
                "n_val": 0,
                "spearman": None,
                "kendall": None,
            },
            "tau": tau,
        }

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    split = max(1, int(n * (1 - val_fraction)))
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    w = np.array([DEFAULT_WEIGHTS[m] for m in METRIC_ORDER], dtype=np.float64)
    w_prior = w.copy()

    train_loss = float("nan")
    for step in range(n_steps):
        z = (X_tr @ w) / tau
        p = _sigmoid(z)
        # Gradient of BCE wrt z: (p - y); through logit -> X / tau
        grad_z = (p - y_tr) / tau
        grad = X_tr.T @ grad_z / X_tr.shape[0]
        # L2 toward prior keeps weights interpretable when data is sparse.
        grad += 2.0 * l2 * (w - w_prior)
        w -= learning_rate * grad

        if step % 100 == 0 or step == n_steps - 1:
            train_loss = _bce(p, y_tr) + l2 * float(((w - w_prior) ** 2).sum())

    # Project to non-negative + L1-normalise to keep weights as a convex combo.
    w_clipped = np.clip(w, 0.0, None)
    s = w_clipped.sum()
    if s <= 1e-9:
        w_norm = np.array([1.0 / len(METRIC_ORDER)] * len(METRIC_ORDER))
    else:
        w_norm = w_clipped / s

    p_tr = _sigmoid((X_tr @ w_norm) / tau)
    p_va = _sigmoid((X_va @ w_norm) / tau) if X_va.shape[0] > 0 else np.zeros((0,))

    val_loss = _bce(p_va, y_va) if X_va.shape[0] > 0 else None
    pairwise_acc = _accuracy(p_tr, y_tr)
    val_acc = _accuracy(p_va, y_va) if X_va.shape[0] > 0 else float("nan")

    weights_dict = {m: float(w_norm[i]) for i, m in enumerate(METRIC_ORDER)}
    metrics = {
        "pairwise_accuracy": float(
            val_acc if not math.isnan(val_acc) else pairwise_acc
        ),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_va.shape[0]),
        "spearman": None,
        "kendall": None,
    }

    return {
        "weights": weights_dict,
        "weights_version": f"calibrated-{uuid.uuid4().hex[:6]}",
        "metrics": metrics,
        "tau": float(tau),
    }


def predict_pairwise_probability(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    weights: dict[str, float],
    tau: float = 1.0,
) -> float:
    """P(A preferred over B) under the linear reward model."""
    phi_a = _featurise_report(report_a)
    phi_b = _featurise_report(report_b)
    w = np.array([weights.get(m, 0.0) for m in METRIC_ORDER], dtype=np.float64)
    return float(_sigmoid(np.array([(phi_a - phi_b) @ w / tau]))[0])


def calibration_curve(
    preferences: list[dict[str, Any]],
    reports_by_id: dict[str, dict[str, Any]],
    weights: dict[str, float],
    tau: float = 1.0,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """
    Reliability-style calibration curve: bin predicted P(A>B) and compute
    empirical fraction of A-wins per bin. Used by the dashboard.
    """
    preds: list[float] = []
    truths: list[float] = []
    for p in preferences:
        a_id, b_id = p["ui_a_id"], p["ui_b_id"]
        if a_id not in reports_by_id or b_id not in reports_by_id:
            continue
        if p["winner"] not in ("a", "b"):
            continue
        prob = predict_pairwise_probability(
            reports_by_id[a_id], reports_by_id[b_id], weights, tau
        )
        preds.append(prob)
        truths.append(1.0 if p["winner"] == "a" else 0.0)
    if not preds:
        return {"bin_centers": [], "empirical": [], "predicted": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, emp, pred = [], [], []
    a_preds = np.array(preds)
    a_truth = np.array(truths)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (a_preds >= lo) & (a_preds < hi if i < n_bins - 1 else a_preds <= hi)
        if mask.sum() == 0:
            continue
        centers.append(float((lo + hi) / 2))
        emp.append(float(a_truth[mask].mean()))
        pred.append(float(a_preds[mask].mean()))
    return {"bin_centers": centers, "empirical": emp, "predicted": pred}
