"""LIME-based local explainability for the trained PINN.

Complements :mod:`src.analysis.shap_explainer` (global attributions) with
local surrogate explanations:

  1. Spatial explanations: for selected ``eta`` probe points, fit a
     ``lime_tabular.LimeTabularExplainer`` on perturbed ``eta`` samples
     mapping ``eta -> [f, theta_f, theta_s, phi]`` via the torch model,
     yielding per-output local weights and intercepts.
  2. Physical-parameter explanations: reuse the cartesian sweep table
     (physics params -> BVP QoIs) with a nearest-neighbour surrogate and
     explain each QoI locally, then aggregate mean-absolute weights into a
     global importance ranking comparable to the SHAP ranking.

Falls back to input-gradient proxies when ``lime`` is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


OUTPUT_NAMES = ("f", "theta_f", "theta_s", "phi")


def _try_import_lime():  # type: ignore[no-untyped-def]
    try:
        from lime import lime_tabular  # type: ignore[import-not-found]

        return lime_tabular
    except Exception:  # pragma: no cover - optional dependency
        return None


def _model_predict_fn(trainer: Any):  # type: ignore[no-untyped-def]
    """Build a numpy -> numpy predict function from a PINNTrainer."""
    import torch

    def _predict(x: np.ndarray) -> np.ndarray:
        trainer.model.eval()
        arr = np.asarray(x, dtype=float).reshape(-1, 1)
        with torch.no_grad():
            eta_t = torch.tensor(arr, dtype=trainer.dtype, device=trainer.device)
            out = trainer.model(eta_t).detach().cpu().numpy()
        return np.asarray(out, dtype=float)

    return _predict


def spatial_lime(
    trainer: Any,
    eta_probes: Sequence[float] | np.ndarray,
    num_samples: int = 1000,
    num_features: int = 1,
) -> Dict[str, Any]:
    """Explain PINN outputs locally around probe ``eta`` points.

    Args:
        trainer: Fitted PINNTrainer (exposes .model, .device, .dtype).
        eta_probes: 1-D probe coordinates to explain.
        num_samples: Perturbation samples per explanation.
        num_features: Features passed to LIME (always 1 for eta).

    Returns:
        Dict with per-probe, per-output ``(weight, intercept)`` lists,
        the probe coordinates, output names and method used.
    """
    lime_tabular = _try_import_lime()
    probes = np.asarray(list(eta_probes), dtype=float).reshape(-1)
    predict_fn = _model_predict_fn(trainer)
    result: Dict[str, Any] = {
        "eta_probes": probes,
        "output_names": list(OUTPUT_NAMES),
        "explanations": [],
        "method": "none",
    }
    if lime_tabular is None:
        # Fallback: input-gradient weights as local linear proxies.
        import torch

        trainer.model.eval()
        weights = []
        for probe in probes:
            eta_t = torch.tensor(
                [[float(probe)]],
                dtype=trainer.dtype,
                device=trainer.device,
                requires_grad=True,
            )
            out = trainer.model(eta_t)
            row = []
            for j in range(out.shape[1]):
                g = torch.autograd.grad(
                    out[0, j], eta_t, retain_graph=True, create_graph=False
                )[0]
                row.append([float(g.detach().cpu().reshape(-1)[0]), float(out[0, j].detach().cpu())])
            weights.append(row)
        result["explanations"] = weights
        result["method"] = "input_gradient_fallback"
        return result

    eta0 = float(getattr(trainer, "eta0", 4.0))
    training = np.linspace(0.0, eta0, 200).reshape(-1, 1)
    explainer = lime_tabular.LimeTabularExplainer(
        training,
        feature_names=["eta"],
        mode="regression",
        verbose=False,
        random_state=0,
    )
    explanations: List[List[List[float]]] = []
    for probe in probes:
        per_output: List[List[float]] = []
        for j, _name in enumerate(OUTPUT_NAMES):

            def _scalar_predict(x: np.ndarray, _j: int = j) -> np.ndarray:
                return predict_fn(x)[:, _j]

            exp = explainer.explain_instance(
                np.array([float(probe)]),
                _scalar_predict,
                num_features=num_features,
                num_samples=num_samples,
            )
            weight = float(dict(exp.local_exp[1])[0]) if exp.local_exp else 0.0
            per_output.append([weight, float(exp.intercept[1])])
        explanations.append(per_output)
    result["explanations"] = explanations
    result["method"] = "lime.LimeTabularExplainer"
    return result


def parameter_lime(
    base_config: Dict[str, Any],
    param_grid: Dict[str, Sequence[float]],
    num_samples: int = 1000,
    qoi_index: int | None = None,
) -> Dict[str, Any]:
    """Explain BVP QoIs locally in physical-parameter space with LIME.

    Builds the same sweep table as
    :func:`src.analysis.shap_explainer.parameter_sensitivity_sweep`
    (params -> ``[Cf, Nu_f, Nu_s, Sh, mean_theta_f]``) and fits one local
    linear surrogate per QoI around each sampled table row.

    Args:
        base_config: Full base config dict.
        param_grid: Mapping param name -> list of values.
        num_samples: Perturbation samples per explanation.
        qoi_index: Explain a single QoI index, or all when None.

    Returns:
        Dict with params, qoi_names, X, Y, per-QoI local weights,
        aggregated global ranking and method used.
    """
    import copy

    from src.core.fluid_properties import compute_coefficients
    from src.solvers.numerical_rk45 import solve_bvp_baseline

    lime_tabular = _try_import_lime()
    keys = list(param_grid.keys())
    qoi_names = ["Cf_proxy", "Nu_f", "Nu_s", "Sh", "mean_theta_f"]

    import itertools

    rows: List[List[float]] = []
    qois: List[List[float]] = []
    for combo in itertools.product(*[list(param_grid[k]) for k in keys]):
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
        cf = float((base["fp"][1] - base["fp"][0]) / deta)
        nuf = float(-(base["theta_f"][1] - base["theta_f"][0]) / deta)
        nus = float(-(base["theta_s"][1] - base["theta_s"][0]) / deta)
        sh = float(-(base["phi"][1] - base["phi"][0]) / deta)
        rows.append([float(v) for v in combo])
        qois.append([cf, nuf, nus, sh, float(np.mean(base["theta_f"]))])

    X = np.asarray(rows, dtype=float)
    Y = np.asarray(qois, dtype=float)
    out: Dict[str, Any] = {
        "params": keys,
        "qoi_names": qoi_names,
        "X": X,
        "Y": Y,
        "local_weights": [],
        "ranking": [],
        "method": "none",
    }
    if X.size == 0:
        return out
    targets = [qoi_index] if qoi_index is not None else list(range(Y.shape[1]))
    if lime_tabular is None:
        # Fallback: standardized-regression coefficients as proxies.
        stds = np.std(X, axis=0) + 1e-12
        weights = []
        for j in targets:
            y = Y[:, j]
            coef = []
            for d in range(X.shape[1]):
                x = X[:, d]
                denom = float(np.std(x)) + 1e-12
                coef.append(float(np.cov(x, y, bias=True)[0, 1] / denom))
            weights.append(coef)
        out["local_weights"] = weights
        mean_abs = np.mean(np.abs(np.asarray(weights)), axis=0)
        out["ranking"] = [
            (keys[i], float(mean_abs[i]))
            for i in list(np.argsort(mean_abs)[::-1])
        ]
        out["method"] = "covariance_fallback"
        return out

    def _surrogate(x: np.ndarray) -> np.ndarray:
        preds = []
        for row in np.atleast_2d(np.asarray(x, dtype=float)):
            d = np.sum((X - row) ** 2, axis=1)
            preds.append(Y[int(np.argmin(d))])
        return np.asarray(preds)

    explainer = lime_tabular.LimeTabularExplainer(
        X,
        feature_names=keys,
        mode="regression",
        verbose=False,
        random_state=0,
    )
    explain_idx = list(range(min(10, len(X))))
    all_weights: List[List[List[float]]] = []
    for j in targets:

        def _qoi_predict(x: np.ndarray, _j: int = j) -> np.ndarray:
            return _surrogate(x)[:, _j]

        per_row: List[List[float]] = []
        for i in explain_idx:
            exp = explainer.explain_instance(
                X[i],
                _qoi_predict,
                num_features=len(keys),
                num_samples=num_samples,
            )
            wmap = dict(exp.local_exp[1]) if exp.local_exp else {}
            per_row.append([float(wmap.get(d, 0.0)) for d in range(len(keys))])
        all_weights.append(per_row)
    out["local_weights"] = all_weights
    out["explained_rows"] = explain_idx
    out["explained_qois"] = [qoi_names[j] for j in targets]
    mean_abs = np.mean(np.abs(np.asarray(all_weights)), axis=(0, 1))
    out["ranking"] = [
        (keys[i], float(mean_abs[i])) for i in list(np.argsort(mean_abs)[::-1])
    ]
    out["method"] = "lime.LimeTabularExplainer"
    return out


def save_lime_summary(result: Dict[str, Any], outdir: str | Path) -> Path:
    """Persist LIME weights and render global-importance bar plots.

    Args:
        result: Output of :func:`spatial_lime` or :func:`parameter_lime`.
        outdir: Directory for ``lime_values.npz`` and PNG figures.

    Returns:
        Path of the saved ``.npz`` archive.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in result.items():
        if k in ("params", "qoi_names", "output_names", "method"):
            continue
        if isinstance(v, (np.ndarray, list)):
            try:
                arrays[k] = np.asarray(v, dtype=float)
            except (ValueError, TypeError):
                continue
    path = outdir / "lime_values.npz"
    if arrays:
        np.savez_compressed(path, **arrays)
    try:
        import matplotlib.pyplot as plt

        if result.get("ranking"):
            names = [n for n, _ in result["ranking"]]
            vals = [v for _, v in result["ranking"]]
            plt.figure(figsize=(6, 3.5))
            plt.barh(names[::-1], vals[::-1])
            plt.xlabel("Mean |LIME weight|")
            plt.title("LIME global parameter importance")
            plt.tight_layout()
            plt.savefig(outdir / "lime_importance.png", dpi=150)
            plt.close()
        if result.get("explanations"):
            arr = np.asarray(result["explanations"], dtype=float)
            # arr shape: (n_probes, n_outputs, 2); plot weights.
            weights = arr[:, :, 0]
            x = np.asarray(result.get("eta_probes"), dtype=float)
            plt.figure(figsize=(7, 4))
            for j, name in enumerate(result.get("output_names", [])):
                plt.plot(x, weights[:, j], marker="o", ms=3, label=f"d{name}/deta")
            plt.xlabel("eta probe")
            plt.ylabel("LIME local weight")
            plt.title("LIME spatial explanations")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "lime_spatial.png", dpi=150)
            plt.close()
    except Exception:
        pass
    return path
