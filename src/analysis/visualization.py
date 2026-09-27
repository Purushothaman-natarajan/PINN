"""Matplotlib visualisation: loss curves, profiles, regression plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_loss(history: Dict[str, List[float]], outdir: str | Path) -> Path:
    """Plot total/PDE/BC loss curves (log scale).

    Args:
        history: Dict with total, pde, bc lists.
        outdir: Directory for the PNG output.

    Returns:
        Path to the saved figure.
    """
    outdir = _ensure_dir(outdir)
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for key in ("total", "pde", "bc"):
        if key in history and len(history[key]) > 0:
            plt.semilogy(history[key], label=key)
            plotted = True
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("PINN training loss")
    if plotted:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No training history", ha="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    path = outdir / "loss_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_profiles(
    baseline: Dict[str, np.ndarray],
    prediction: Dict[str, np.ndarray],
    outdir: str | Path,
) -> Path:
    """Overlay PINN vs BVP profiles for f, theta_f, theta_s, phi."""
    outdir = _ensure_dir(outdir)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    pairs = [
        ("f", "Velocity f"),
        ("theta_f", "Fluid temp θf"),
        ("theta_s", "Solid temp θs"),
        ("phi", "Concentration φ"),
    ]
    for ax, (var, title) in zip(axes.ravel(), pairs):
        ax.plot(baseline["eta"], baseline[var], "k-", label="BVP baseline")
        ax.plot(
            prediction["eta"],
            prediction[var],
            "r--",
            label="PINN",
        )
        ax.set_title(title)
        ax.set_xlabel("η")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend()
    fig.suptitle("PINN vs numerical baseline profiles")
    fig.tight_layout()
    path = outdir / "profiles.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_regression(
    baseline: Dict[str, np.ndarray],
    prediction: Dict[str, np.ndarray],
    outdir: str | Path,
) -> Path:
    """Parity (predicted vs true) scatter for each field."""
    from src.analysis.validation import r2_score

    outdir = _ensure_dir(outdir)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, var in zip(
        axes.ravel(), ("f", "theta_f", "theta_s", "phi")
    ):
        y_t = np.asarray(baseline[var]).reshape(-1)
        y_p = np.asarray(prediction[var]).reshape(-1)
        if y_p.shape != y_t.shape:
            y_p = np.interp(baseline["eta"], prediction["eta"], y_p)
        r2 = r2_score(y_t, y_p)
        lo = float(min(y_t.min(), y_p.min()))
        hi = float(max(y_t.max(), y_p.max()))
        ax.scatter(y_t, y_p, s=8, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{var} (R²={r2:.4f})")
        ax.set_xlabel("Baseline")
        ax.set_ylabel("PINN")
        ax.grid(alpha=0.3)
    fig.suptitle("Regression parity plots")
    fig.tight_layout()
    path = outdir / "regression.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_all(
    history: Dict[str, List[float]] | None,
    baseline: Dict[str, Any] | None,
    prediction: Dict[str, Any] | None,
    outdir: str | Path,
) -> List[Path]:
    """Generate every standard figure that has data available."""
    paths: List[Path] = []
    if history and any(len(history.get(k, [])) > 0 for k in ("total", "pde", "bc")):
        paths.append(plot_loss(history, outdir))
    if baseline is not None and prediction is not None:
        paths.append(plot_profiles(baseline, prediction, outdir))
        paths.append(plot_regression(baseline, prediction, outdir))
    return paths
