"""Validation metrics: MSE, RMSE, R-squared (PINN vs numerical baseline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def evaluate_predictions(
    baseline: Dict[str, np.ndarray],
    prediction: Dict[str, np.ndarray],
    variables: tuple[str, ...] = ("f", "theta_f", "theta_s", "phi"),
) -> Dict[str, Dict[str, float]]:
    """Evaluate PINN predictions against the BVP baseline.

    Interpolates predictions onto baseline eta if grids differ.

    Args:
        baseline: Dict with eta + reference fields.
        prediction: Dict with eta + predicted fields.
        variables: Field names to score.

    Returns:
        Nested dict {var: {mse, rmse, r2}} plus _mean aggregate.
    """
    eta_ref = np.asarray(baseline["eta"], dtype=float).reshape(-1)
    eta_pred = np.asarray(prediction["eta"], dtype=float).reshape(-1)
    metrics: Dict[str, Dict[str, float]] = {}
    for var in variables:
        y_true = np.asarray(baseline[var], dtype=float)
        y_raw = np.asarray(prediction[var], dtype=float)
        if y_raw.shape != y_true.shape or not np.allclose(eta_pred, eta_ref):
            y_raw = np.interp(eta_ref, eta_pred, y_raw)
        metrics[var] = {
            "mse": mse(y_true, y_raw),
            "rmse": rmse(y_true, y_raw),
            "r2": r2_score(y_true, y_raw),
        }
    metrics["_mean"] = {
        "mse": float(np.mean([metrics[v]["mse"] for v in variables])),
        "rmse": float(np.mean([metrics[v]["rmse"] for v in variables])),
        "r2": float(np.mean([metrics[v]["r2"] for v in variables])),
    }
    return metrics


def save_metrics(
    metrics: Dict[str, Dict[str, float]], config: Dict[str, Any]
) -> Path:
    """Persist metrics JSON to the processed data directory."""
    out_dir = Path(config.get("outputs", {}).get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = config.get("outputs", {}).get("metrics_file", "metrics.json")
    path = out_dir / fname
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return path


def meets_accuracy_gate(metrics: Dict[str, Dict[str, float]], thr: float = 0.95) -> bool:
    """Check whether mean R^2 exceeds the target accuracy."""
    return bool(metrics.get("_mean", {}).get("r2", 0.0) >= thr)
