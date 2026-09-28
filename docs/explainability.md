# Explainability (SHAP + LIME)

Two complementary lenses. Use **SHAP** for global "which parameter matters
overall" claims and **LIME** for local "why this prediction here" stories;
`--mode explain` runs both.

## SHAP (`src/analysis/shap_explainer.py`)

| Function | Scope | Method |
|---|---|---|
| `spatial_shap(trainer, eta_grid)` | Outputs vs η | `shap.DeepExplainer` on the torch model; input-gradient fallback without `shap` |
| `parameter_sensitivity_sweep(config, grid)` | Physics params → QoIs | Cartesian BVP sweep → nearest-neighbour surrogate → `shap.KernelExplainer` → ranking |

QoIs: `Cf_proxy ≈ f″(0)`, `Nu_f ≈ −θf′(0)`, `Nu_s ≈ −θs′(0)`,
`Sh ≈ −φ′(0)`, `mean_theta_f`. Outputs: `data/processed/shap/` with
`spatial_shap.npz`, `shap_values.npz`, `shap_summary.png`.

## LIME (`src/analysis/lime_explainer.py`)

| Function | Scope | Method |
|---|---|---|
| `spatial_lime(trainer, eta_probes)` | Local `d(output)/dη` at probe points | `LimeTabularExplainer` per output; gradient fallback |
| `parameter_lime(config, grid)` | Local param effects per QoI + aggregated ranking | Same sweep table, `LimeTabularExplainer` per QoI |

Outputs: `data/processed/lime/` with `lime_values.npz`,
`lime_importance.png` (global bar), `lime_spatial.png` (weights vs η).

## Reading results

- **Spatial:** weight sign flips across the boundary layer are normal
  (fields rise then fall toward ambient). Compare against `profiles.png`.
- **Parameter ranking:** expect `M` and `Rd` near the top for this problem
  class (magnetic drag + radiation dominate); a surprising top rank usually
  means the sweep range for that parameter is disproportionately wide —
  normalize ranges before drawing conclusions.
- **Cost:** SHAP `KernelExplainer` and LIME sampling are the slowest XAI
  steps; shrink the sweep grid or `num_samples` for iteration, expand for
  final figures.

## Reusing for new parameters/QoIs

Any `physics:` key can join the sweep grid ([Sweeps](sweeps.md)); new QoIs
follow [Add a QoI](extending.md#add-a-qoi-or-cli-mode) and automatically flow
into both SHAP and LIME rankings.
