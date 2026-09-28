# Parameter sweeps

Sweeps evaluate the **fast BVP baseline** (not full PINN retraining) over a
cartesian grid — ideal for sensitivity mapping, XAI input tables, and
pre-screening before expensive training.

## Ablation config

```yaml
base_config: configs/default_trihybrid_ltne.yaml
sweep:
  M: [0.0, 1.0, 2.0]
  Rd: [0.0, 0.5, 1.0]
  Fr: [0.0, 0.2, 0.5]
  Hs: [0.5, 1.0, 2.0]
options:
  mode: train
  epochs_override: 5000   # light retrains; null => base-config epochs
  output_csv: data/processed/ablation_results.csv
  parallel: false
```

Run with `python main.py --config configs/ablation_study_config.yaml --mode sweep`.

- **Any** `physics:` key can be swept — add `Ec`, `Sc`, `Kr`, `Sr`, `Q`,
  `gamma_curv`, `slip`, `omega_ratio` freely (schema-validated on the base).
- Each row records the swept values plus QoIs: `mean_theta_f`, `mean_phi`,
  `f_outer_fp` (failed points record `error` instead of QoIs).
- Cost scales multiplicatively with grid size: start coarse (2–3 values),
  refine around interesting transitions.

## Scaling up

- Keep `n_points: 150`-style coarse BVP resolution for sweeps (as in
  `cmd_sweep`); use full resolution for final baselines.
- `parallel: false` is the current default; flip to a process pool only via
  custom code (each BVP solve is independent — embarrassingly parallel).
- Sweep CSVs double as training data for surrogate models feeding SHAP/LIME
  (see [Explainability](explainability.md)).
