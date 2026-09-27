"""CLI entry point: train PINN, run BVP baseline, validate, SHAP, plots.

Examples:
    python main.py --config configs/default_trihybrid_ltne.yaml --mode all
    python main.py --config configs/default_trihybrid_ltne.yaml --mode train
    python main.py --config configs/default_trihybrid_ltne.yaml --mode baseline
    python main.py --config configs/default_trihybrid_ltne.yaml --mode validate
    python main.py --config configs/ablation_study_config.yaml --mode sweep
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def _load_full_config(path: str) -> Dict[str, Any]:
    from src.core.config_loader import load_config

    return load_config(path)


def cmd_train(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Train the PINN and return artefacts."""
    import torch

    from src.core.config_loader import resolve_device
    from src.core.fluid_properties import compute_coefficients
    from src.models.pinn_architecture import build_pinn
    from src.solvers.pinn_trainer import PINNTrainer

    seed = int(config["domain"].get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    coeffs = compute_coefficients(config)
    device = resolve_device(str(config["training"].get("device", "auto")))
    model = build_pinn(config["model"])
    trainer = PINNTrainer(model, config, coeffs, device=device)
    ckpt = args.checkpoint
    if ckpt and Path(ckpt).is_file():
        trainer.load(ckpt)
        print(f"Resumed from {ckpt}")
    history = trainer.train()
    return {"trainer": trainer, "coeffs": coeffs, "history": history}


def cmd_baseline(config: Dict[str, Any], n_points: int = 400) -> Dict[str, Any]:
    """Run the numerical baseline and persist it."""
    from src.core.fluid_properties import compute_coefficients
    from src.solvers.numerical_rk45 import save_baseline, solve_bvp_baseline

    coeffs = compute_coefficients(config)
    baseline = solve_bvp_baseline(config, coeffs, n_points=n_points)
    path = save_baseline(baseline, config)
    print(f"Baseline saved to {path}")
    return {"baseline": baseline, "coeffs": coeffs, "path": path}


def cmd_validate(
    config: Dict[str, Any], trainer: Any | None = None, baseline: Any | None = None
) -> Dict[str, Any]:
    """Evaluate PINN vs baseline, print MSE/RMSE/R2, save metrics."""
    from src.analysis.validation import (
        evaluate_predictions,
        meets_accuracy_gate,
        save_metrics,
    )
    from src.analysis.visualization import plot_all
    from src.core.fluid_properties import compute_coefficients
    from src.solvers.numerical_rk45 import solve_bvp_baseline

    coeffs = compute_coefficients(config)
    if baseline is None:
        raw = Path(config.get("outputs", {}).get("raw_dir", "data/raw")) / config.get(
            "outputs", {}
        ).get("baseline_file", "baseline.npz")
        if raw.is_file():
            with np.load(raw) as z:
                baseline = {k: z[k] for k in z.files}
        else:
            baseline = solve_bvp_baseline(config, coeffs)
    if trainer is None:
        from src.core.config_loader import resolve_device
        from src.models.pinn_architecture import build_pinn
        from src.solvers.pinn_trainer import PINNTrainer

        device = resolve_device(str(config["training"].get("device", "auto")))
        model = build_pinn(config["model"])
        trainer = PINNTrainer(model, config, coeffs, device=device)
        final = Path(config["training"].get("checkpoint_dir", "checkpoints")) / "pinn_final.pt"
        if final.is_file():
            trainer.load(final)
    prediction = trainer.predict(baseline["eta"])
    metrics = evaluate_predictions(baseline, prediction)
    mpath = save_metrics(metrics, config)
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {mpath}")
    ok = meets_accuracy_gate(metrics, 0.95)
    print(f"Accuracy gate R2>0.95: {'PASS' if ok else 'FAIL'}")
    outdir = Path(config.get("outputs", {}).get("processed_dir", "data/processed"))
    plot_all(trainer.history, baseline, prediction, outdir / "figures")
    return {"metrics": metrics, "prediction": prediction, "baseline": baseline}


def cmd_sweep(sweep_cfg_path: str, args: argparse.Namespace) -> None:
    """Cartesian sweep from ablation_study_config.yaml with BVP QoIs."""
    import pandas as pd

    with open(sweep_cfg_path, encoding="utf-8") as fh:
        sweep_cfg = yaml.safe_load(fh)
    base = _load_full_config(sweep_cfg["base_config"])
    grid = sweep_cfg.get("sweep", {})
    opts = sweep_cfg.get("options", {})
    epochs_override = opts.get("epochs_override", None)

    keys = list(grid.keys())
    combos = list(itertools.product(*[list(grid[k]) for k in keys]))
    records = []
    for combo in combos:
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            cfg["physics"][k] = float(v)
        if epochs_override:
            cfg["training"]["epochs"] = int(epochs_override)
        # Fast path: BVP QoIs + optional short PINN train for metrics.
        from src.core.fluid_properties import compute_coefficients
        from src.solvers.numerical_rk45 import solve_bvp_baseline

        coeffs = compute_coefficients(cfg)
        try:
            b = solve_bvp_baseline(cfg, coeffs, n_points=150)
            rec = {k: float(v) for k, v in zip(keys, combo)}
            rec.update(
                {
                    "mean_theta_f": float(np.mean(b["theta_f"])),
                    "mean_phi": float(np.mean(b["phi"])),
                    "f_outer_fp": float(b["fp"][-1]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rec = {k: float(v) for k, v in zip(keys, combo)}
            rec["error"] = str(exc)
        records.append(rec)
        print(rec)
    out_csv = Path(opts.get("output_csv", "data/processed/ablation_results.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Sweep saved to {out_csv}")


def cmd_shap(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run SHAP spatial + parameter sensitivity analyses."""
    from src.analysis.shap_explainer import (
        parameter_sensitivity_sweep,
        save_shap_summary,
        spatial_shap,
    )
    from src.core.config_loader import resolve_device
    from src.core.fluid_properties import compute_coefficients
    from src.models.pinn_architecture import build_pinn
    from src.solvers.pinn_trainer import PINNTrainer

    coeffs = compute_coefficients(config)
    device = resolve_device(str(config["training"].get("device", "auto")))
    model = build_pinn(config["model"])
    trainer = PINNTrainer(model, config, coeffs, device=device)
    final = Path(config["training"].get("checkpoint_dir", "checkpoints")) / "pinn_final.pt"
    if final.is_file():
        trainer.load(final)
        print(f"Loaded checkpoint {final}")
    eta = np.linspace(0.0, float(config["domain"].get("eta0", 4.0)), 100)
    spat = spatial_shap(trainer, eta)
    print(f"Spatial SHAP method: {spat.get('method')}")
    outdir = Path(args.outdir or "data/processed/shap")
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "spatial_shap.npz", values=np.asarray(spat["values"]))
    grid = {"M": [0.0, 1.0, 2.0], "Rd": [0.0, 0.5, 1.0], "Fr": [0.0, 0.2, 0.5]}
    param_res = parameter_sensitivity_sweep(config, grid)
    print(f"Param SHAP method: {param_res.get('method')}")
    print(f"Ranking: {param_res.get('ranking')}")
    save_shap_summary(param_res, outdir)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="Trihybrid LTNE PINN (PyTorch)")
    p.add_argument("--config", default="configs/default_trihybrid_ltne.yaml")
    p.add_argument(
        "--mode",
        default="all",
        choices=["train", "baseline", "validate", "all", "sweep", "shap", "plot"],
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--n-points", type=int, default=400)
    return p


def main() -> None:
    """CLI dispatcher."""
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "sweep":
        cmd_sweep(args.config, args)
        return
    config = _load_full_config(args.config)
    if args.mode == "train":
        cmd_train(config, args)
    elif args.mode == "baseline":
        cmd_baseline(config, n_points=args.n_points)
    elif args.mode == "validate":
        cmd_validate(config)
    elif args.mode == "shap":
        cmd_shap(config, args)
    elif args.mode == "plot":
        cmd_validate(config)
    elif args.mode == "all":
        trained = cmd_train(config, args)
        base = cmd_baseline(config, n_points=args.n_points)
        cmd_validate(
            config, trainer=trained["trainer"], baseline=base["baseline"]
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
