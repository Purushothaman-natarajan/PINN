"""CLI entry point: train PINN, run BVP baseline, validate, explain, plots.

Examples:
    python main.py --config configs/default_trihybrid_ltne.yaml --mode all
    python main.py --config configs/default_trihybrid_ltne.yaml --mode train
    python main.py --config configs/default_trihybrid_ltne.yaml --mode baseline
    python main.py --config configs/default_trihybrid_ltne.yaml --mode validate
    python main.py --config configs/default_trihybrid_ltne.yaml --mode shap
    python main.py --config configs/default_trihybrid_ltne.yaml --mode lime
    python main.py --config configs/default_trihybrid_ltne.yaml --mode explain
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


def cmd_lime(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run LIME spatial + parameter sensitivity analyses."""
    from src.analysis.lime_explainer import (
        parameter_lime,
        save_lime_summary,
        spatial_lime,
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
    eta0 = float(config["domain"].get("eta0", 4.0))
    probes = np.linspace(0.0, eta0, 5)
    spat = spatial_lime(trainer, probes)
    print(f"Spatial LIME method: {spat.get('method')}")
    for probe, per_out in zip(spat["eta_probes"], spat["explanations"]):
        print(f"  eta={float(probe):.2f}: " + ", ".join(
            f"{n} w={w[0]:+.3e}" for n, w in zip(spat["output_names"], per_out)
        ))
    outdir = Path(args.outdir or "data/processed/lime")
    outdir.mkdir(parents=True, exist_ok=True)
    save_lime_summary(spat, outdir)
    grid = {"M": [0.0, 1.0, 2.0], "Rd": [0.0, 0.5, 1.0], "Fr": [0.0, 0.2, 0.5]}
    param_res = parameter_lime(config, grid)
    print(f"Param LIME method: {param_res.get('method')}")
    print(f"Ranking: {param_res.get('ranking')}")
    save_lime_summary(param_res, outdir)


def cmd_mock(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Generate -> clean -> train -> validate for 1000-pt mock datasets.

    For each parameter combination in the ``param_grid`` block (or the
    single default point when the grid is empty): solve the BVP on a dense
    grid, clean the dataset deterministically, train a PINN, and validate
    it against the cleaned mock data. Stages whose outputs already exist
    are skipped unless ``--force`` is given. A summary CSV is written last.
    """
    import json

    import pandas as pd
    import torch

    from src.analysis.validation import evaluate_predictions, meets_accuracy_gate
    from src.analysis.visualization import plot_all
    from src.core.config_loader import resolve_device
    from src.core.fluid_properties import compute_coefficients
    from src.data.cleaning import clean_mock, save_clean_mock
    from src.data.mock_data import (
        combo_name,
        generate_mock_data,
        load_mock,
        override_physics,
        param_combos,
        save_mock,
    )
    from src.models.pinn_architecture import build_pinn
    from src.solvers.pinn_trainer import PINNTrainer

    mock_cfg = config.get("mock", {})
    n_points = int(mock_cfg.get("n_points", 1000))
    noise = float(mock_cfg.get("noise", 0.0))
    mock_seed = int(mock_cfg.get("seed", 0))
    clean_cfg = config.get("cleaning", {})
    grid = config.get("param_grid", {}) or {}
    combos = param_combos(grid) if grid else [{}]
    force = bool(getattr(args, "force", False))
    start = int(getattr(args, "start", 0) or 0)
    end = getattr(args, "end", None)
    end = int(end) if end is not None else len(combos)
    combos = combos[start:end]

    raw_dir = Path(config.get("outputs", {}).get("raw_dir", "data/raw"))
    processed_dir = Path(
        config.get("outputs", {}).get("processed_dir", "data/processed")
    )
    base_ckpt = config["training"].get("checkpoint_dir", "checkpoints")
    seed = int(config["domain"].get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Mock pipeline: {len(combos)} combo(s), {n_points} points each")
    records: list[Dict[str, Any]] = []
    for idx, combo in enumerate(combos):
        stem = combo_name(combo) if combo else "mock_default"
        print(f"[{idx + 1}/{len(combos)}] {stem} {combo}")
        record: Dict[str, Any] = {k: float(v) for k, v in combo.items()}
        try:
            # --- Generate (or reuse) ---
            raw_path = raw_dir / f"{stem}.npz"
            solver_relaxed = False
            if raw_path.is_file() and not force:
                raw = load_mock(raw_path)
                print(f"  reuse raw {raw_path}")
            else:
                cfg_c = override_physics(config, combo)
                coeffs = compute_coefficients(cfg_c)
                try:
                    raw = generate_mock_data(
                        cfg_c, coeffs, n_points=n_points, noise=noise, seed=mock_seed
                    )
                except RuntimeError as exc:
                    # Stiff corner (e.g. strong LTNE coupling): retry once
                    # with a relaxed mesh/tolerance and record provenance.
                    print(f"  BVP stiff ({exc}); retrying relaxed")
                    cfg_c["solver"] = dict(cfg_c["solver"])
                    cfg_c["solver"]["max_nodes"] = max(
                        int(cfg_c["solver"].get("max_nodes", 2000)), 8000
                    )
                    cfg_c["solver"]["tol"] = 1e-6
                    raw = generate_mock_data(
                        cfg_c, coeffs, n_points=n_points, noise=noise, seed=mock_seed
                    )
                    solver_relaxed = True
                save_mock(raw, raw_dir, stem)
                print(f"  generated raw {raw_path} relaxed={solver_relaxed}")
            record["n_raw"] = int(len(raw["eta"]))

            # --- Clean (or reuse) ---
            clean_path = processed_dir / f"{stem}.npz"
            report_path = processed_dir / f"{stem}_cleaning_report.json"
            if clean_path.is_file() and report_path.is_file() and not force:
                clean = load_mock(clean_path)
                with report_path.open(encoding="utf-8") as fh:
                    report = json.load(fh)
                print(f"  reuse clean {clean_path}")
            else:
                clean, report = clean_mock(
                    raw,
                    clip=bool(clean_cfg.get("clip", False)),
                    smooth_window=int(clean_cfg.get("smooth_window", 0)),
                )
                report["solver_relaxed"] = bool(solver_relaxed)
                save_clean_mock(clean, report, processed_dir, stem)
                print(f"  cleaned {clean_path} {report}")
            record["n_clean"] = int(report.get("n_out", len(clean["eta"])))
            record["n_dropped"] = int(report.get("n_dropped_nonfinite", 0))
            record["solver_relaxed"] = bool(report.get("solver_relaxed", False))

            # --- Train (or reuse checkpoint) ---
            cfg_c = override_physics(config, combo)
            cfg_c["training"] = dict(cfg_c["training"])
            cfg_c["training"]["checkpoint_dir"] = str(Path(base_ckpt) / stem)
            coeffs = compute_coefficients(cfg_c)
            device = resolve_device(str(cfg_c["training"].get("device", "auto")))
            trainer = PINNTrainer(build_pinn(cfg_c["model"]), cfg_c, coeffs, device=device)
            final_ckpt = Path(cfg_c["training"]["checkpoint_dir"]) / "pinn_final.pt"
            if final_ckpt.is_file() and not force:
                trainer.load(final_ckpt)
                history = {"total": [], "pde": [], "bc": []}
                print(f"  reuse checkpoint {final_ckpt}")
            else:
                history = trainer.train()

            # --- Validate against cleaned mock ---
            prediction = trainer.predict(clean["eta"])
            metrics = evaluate_predictions(clean, prediction)
            ok = meets_accuracy_gate(metrics, 0.95)
            with (processed_dir / f"metrics_{stem}.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
            plot_all(history, clean, prediction, processed_dir / "figures" / stem)
            record["mean_r2"] = float(metrics["_mean"]["r2"])
            for var in ("f", "theta_f", "theta_s", "phi"):
                record[f"r2_{var}"] = float(metrics[var]["r2"])
            if history["total"]:
                record["final_loss"] = float(history["total"][-1])
            record["gate"] = "PASS" if ok else "FAIL"
            print(f"  mean R2={record['mean_r2']:.4f} gate={record['gate']}")
        except Exception as exc:  # noqa: BLE001 - per-combo isolation
            record["error"] = str(exc)
            print(f"  ERROR: {exc}")
        records.append(record)

    summary = Path(
        config.get("outputs", {}).get("mock_summary", "mock_sweep_summary.csv")
    )
    if not summary.is_absolute():
        summary = processed_dir / summary.name
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(summary, index=False)
    print(f"Mock summary saved to {summary}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="Trihybrid LTNE PINN (PyTorch)")
    p.add_argument("--config", default="configs/default_trihybrid_ltne.yaml")
    p.add_argument(
        "--mode",
        default="all",
        choices=["train", "baseline", "validate", "all", "sweep", "shap", "lime", "explain", "plot", "mock"],
    )
    p.add_argument("--force", action="store_true", help="Regenerate existing artefacts")
    p.add_argument("--start", type=int, default=0, help="Combo slice start (mock mode)")
    p.add_argument("--end", type=int, default=None, help="Combo slice end (mock mode)")
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
    elif args.mode == "lime":
        cmd_lime(config, args)
    elif args.mode == "explain":
        cmd_shap(config, args)
        cmd_lime(config, args)
    elif args.mode == "plot":
        cmd_validate(config)
    elif args.mode == "mock":
        cmd_mock(config, args)
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
