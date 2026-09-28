# Validation

PINN predictions are checked against an independent mesh-based solver on the
**same** term-switched equations — validation is meaningful only because of
that parity ([Equation catalog](equations.md)).

## Baseline method

`src/solvers/numerical_rk45.py::solve_bvp_baseline` rewrites the similarity
ODEs as a 9-state first-order system
`y = [f, f′, f″, θf, θf′, θs, θs′, φ, φ′]` and solves it with
`scipy.integrate.solve_bvp` (collocation, RK-based) on an adaptive mesh
(`bvp_mesh`, `tol`, `max_nodes` from config). Dirichlet/Robin/slip variants
are mirrored. Output: `data/raw/baseline.npz`.

## Metrics

`evaluate_predictions()` interpolates the PINN onto the baseline η-grid and
scores each field:

- **MSE** = mean((true − pred)²), **RMSE** = √MSE (same units as the field)
- **R²** = 1 − SS_res/SS_tot per field, plus a `_mean` aggregate
- Saved to `data/processed/metrics.json`; the run passes iff mean R² ≥ 0.95
  (`meets_accuracy_gate`), printed as `PASS`/`FAIL`

## Figures (`data/processed/figures/`)

| Figure | Reads as |
|---|---|
| `loss_curve.png` | Total/PDE/BC loss (log scale); PDE and BC should fall together |
| `profiles.png` | PINN (dashed) vs BVP (solid) for f, θf, θs, φ |
| `regression.png` | Parity scatter per field with R²; points should hug the diagonal |

## Interpreting failures

- **One field poor, rest fine**: usually a coefficient-scale or term-switch
  issue — recheck the term's catalog row and the parameter's range.
- **All fields poor**: under-training (raise epochs), LR too high, or wrong
  `eta0` (profiles clipped at the outer wall).
- **`solve_bvp` fails**: degenerate term combination (see catalog warnings),
  mesh too coarse (`bvp_mesh`), or tolerance too tight — relax stepwise.
