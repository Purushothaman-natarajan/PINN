"""SHAP-based parameter sensitivity for the trained PINN.

Two complementary modes:
  1. Spatial sensitivity: SHAP values of outputs w.r.t. eta using a
     gradient/background explainer on the torch model.
  2. Physical-parameter sensitivity: cartesian sweep over physics params
     (M, Rd, Fr, Hs, ...) from ``ablation_study_config.yaml``; a lightweight
     surrogate maps params -> QoIs (wall gradients, mean temps) and a
     ``shap.KernelExplainer`` ranks parameter importance.

Falls back gracefully when ``shap`` is not installed.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def _try_import_shap():  # type: ignore[no-untyped-def]
    try:
        import shap  # type: ignore[import-not-found]

        return shap
    except Exception:  # pragma: no cover - optional dependency
        return None


def spatial_shap(
    trainer: Any,
    eta_grid: np.ndarray,
    nsamples: int = 50,
) -> Dict[str, Any]:
    """Explain PINN outputs as a function of eta.

    Uses ``shap.DeepExplainer`` when available, else gradient magnitudes.

    Args:
        trainer: Fitted PINNTrainer (exposes .model, .device, .dtype).
        eta_grid: 1-D eta coordinates to explain.
        nsamples: Background sample count.

    Returns:
        Dict with values, base_values, feature names and method used.
    """
    import torch

    shap = _try_import_shap()
    eta_grid = np.asarray(eta_grid, dtype=float).reshape(-1, 1)
    bg_idx = np.linspace(0, len(eta_grid) - 1, min(nsamples, len(eta_grid))).astype(int)
    result: Dict[str, Any] = {"features": ["eta"], "eta": eta_grid.reshape(-1)}

    if shap is None:
        # Fallback: normalized input-gradient magnitudes as proxy importance.
        trainer.model.eval()
        eta_t = torch.tensor(
            eta_grid, dtype=trainer.dtype, device=trainer.device, requires_grad=True
        )
        out = trainer.model(eta_t)
        imps = []
        for j in range(out.shape[1]):
            g = torch.autograd.grad(
                out[:, j].sum(), eta_t, retain_graph=True, create_graph=False
            )[0]
            imps.append(g.detach().cpu().numpy().reshape(-1))
        result["values"] = np.stack(imps, axis=1)
        result["method"] = "input_gradient_fallback"
        return result

    try:
        background = torch.tensor(
            eta_grid[bg_idx], dtype=trainer.dtype, device=trainer.device
        )
        test = torch.tensor(eta_grid, dtype=trainer.dtype, device=trainer.device)
        explainer = shap.DeepExplainer(trainer.model, background)  # type: ignore[attr-defined]
        shap_values = explainer.shap_values(test)
        result["values"] = np.asarray(shap_values)
        result["method"] = "shap.DeepExplainer"
    except Exception as exc:  # pragma: no cover - robustness
        result["values"] = np.zeros((len(eta_grid), 4))
        result["method"] = f"failed:{exc}"
    return result


def parameter_sensitivity_sweep(
    base_config: Dict[str, Any],
    param_grid: Dict[str, Sequence[float]],
    qoi_fn: Any | None = None,
) -> Dict[str, Any]:
    """Run a cartesian parameter sweep and explain QoIs with SHAP.

    Each grid point rebuilds coefficients + BVP baseline (fast) and
    evaluates scalar QoIs: ``Cf`` ~ f''(0), ``Nu_f`` ~ -theta_f'(0),
    ``Sh`` ~ -phi'(0), plus field means. A KernelExplainer over a
    nearest-neighbour surrogate then ranks parameters.

    Args:
        base_config: Full base config dict.
        param_grid: Mapping param name -> list of values.
        qoi_fn: Optional custom QoI callable(baseline)->vector.

    Returns:
        Dict with params, X, qoi_names, Y, shap_values, ranking.
    """
    import copy

    from src.core.fluid_properties import compute_coefficients
    from src.solvers.numerical_rk45 import solve_bvp_baseline

    shap = _try_import_shap()
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[list(param_grid[k]) for k in keys]))
    rows: List[List[float]] = []
    qois: List[List[float]] = []
    qoi_names = ["Cf_proxy", "Nu_f", "Nu_s", "Sh", "mean_theta_f"]

    for combo in combos:
        cfg = copy.deepcopy(base_config)
        for k, v in zip(keys, combo):
            cfg["physics"][k] = float(v)
        coeffs = compute_coefficients(cfg)
        try:
            base = solve_bvp_baseline(cfg, coeffs, n_points=200)
        except Exception:
            continue
        eta = base["eta"]
        deta = float(eta[1] - eta[0]) if len(eta) > 1 else 1.0
        if qoi_fn is not None:
            q = list(qoi_fn(base))
        else:
            cf = float((base["fp"][1] - base["fp"][0]) / deta)
            nuf = float(-(base["theta_f"][1] - base["theta_f"][0]) / deta)
            nus = float(-(base["theta_s"][1] - base["theta_s"][0]) / deta)
            sh = float(-(base["phi"][1] - base["phi"][0]) / deta)
            q = [cf, nuf, nus, sh, float(np.mean(base["theta_f"]))]
        rows.append([float(v) for v in combo])
        qois.append(q)

    X = np.asarray(rows, dtype=float)
    Y = np.asarray(qois, dtype=float)
    out: Dict[str, Any] = {
        "params": keys,
        "qoi_names": qoi_names,
        "X": X,
        "Y": Y,
        "method": "none",
        "shap_values": None,
        "ranking": [],
    }
    if X.size == 0:
        return out
    if shap is None:
        # Variance-based ranking fallback.
        stds = np.std(Y, axis=0)
        out["method"] = "variance_fallback"
        out["ranking"] = sorted(
            zip(keys, [float(np.std(X[:, j])) for j in range(len(keys))]),
            key=lambda t: t[1],
            reverse=True,
        )
        return out
    try:

        def _surrogate(x: np.ndarray) -> np.ndarray:
            # Nearest-neighbour surrogate over sweep table.
            preds = []
            for row in np.atleast_2d(x):
                d = np.sum((X - row) ** 2, axis=1)
                preds.append(Y[int(np.argmin(d))])
            return np.asarray(preds)

        background = shap.sample(X, min(20, len(X)))  # type: ignore[attr-defined]
        explainer = shap.KernelExplainer(_surrogate, background)  # type: ignore[attr-defined]
        sv = explainer.shap_values(X[: min(20, len(X))], nsamples=50)
        out["shap_values"] = np.asarray(sv)
        out["method"] = "shap.KernelExplainer"
        mean_abs = np.mean(np.abs(np.asarray(sv)), axis=(0, 2)) if np.asarray(sv).ndim == 3 else np.mean(
            np.abs(np.asarray(sv)), axis=0
        )
        order = list(np.argsort(mean_abs)[::-1])
        out["ranking"] = [(keys[i], float(mean_abs[i])) for i in order]
    except Exception as exc:  # pragma: no cover
        out["method"] = f"failed:{exc}"
    return out


def save_shap_summary(result: Dict[str, Any], outdir: str | Path) -> Path:
    """Save SHAP summary arrays and try to render a beeswarm/bar plot."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "shap_values.npz",
        **{k: np.asarray(v) for k, v in result.items() if isinstance(v, np.ndarray)},
    )
    try:
        import matplotlib.pyplot as plt

        shap = _try_import_shap()
        if shap is not None and result.get("shap_values") is not None:
            plt.figure()
            shap.summary_plot(
                np.asarray(result["shap_values"])[..., 0]
                if np.asarray(result["shap_values"]).ndim == 3
                else np.asarray(result["shap_values"]),
                result["X"],
                feature_names=result.get("params", None),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(outdir / "shap_summary.png", dpi=150)
            plt.close()
    except Exception:
        pass
    return outdir / "shap_values.npz"
