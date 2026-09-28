# FAQ & troubleshooting

## Training

**Loss stalls, all fields poor.**
Under-training or LR too high. Check `loss_curve.png`: if both PDE and BC
plateau early, extend `epochs` / add the StepLR scheduler; if oscillating,
cut `lr` 10×. See [Training](training.md#tuning-guide).

**Wall values drift (e.g. θf(0) ≠ 1).**
Raise `bc_weight` (try 50–100). The default 10× works for the baseline
regime; stiff corners need more.

**NaNs within a few hundred steps.**
Almost always invalid inputs: particle `phi` > 0.2, non-positive base
properties, or `lr` far too high. The schema rejects most of these on load —
read the `ValueError` path it prints.

## Baseline (`solve_bvp` fails)

1. Suspect a **degenerate term combination** first ([catalog warnings](equations.md#solid-energy)):
   solid `conduction` off, or `diffusion` off, while keeping both BCs.
2. Coarsen then refine: lower `bvp_mesh`, loosen `tol` to `1e-6`, confirm
   convergence, then tighten.
3. Check `eta0`: if the initial-guess profiles can't satisfy decay, enlarge it.

## Validation

**R² gate FAILs but profiles look close.**
Check the η-grids: `evaluate_predictions` interpolates, but a coarse
prediction grid over sharp layers loses R². Re-run `baseline` with higher
`--n-points` and re-validate.

**One field (often θs) much worse.**
LTNE splitting with small `Hs/Hsg` is genuinely harder to learn; confirm the
BVP itself shows the split, then train longer or in `float64`.

## Config / CLI

**`ValueError: Invalid config at '...'`**
The schema path names the exact key. Common culprits: activation typo,
`hidden_layers` containing `0`, `phi` outside [0, 0.2], unknown term key in
`equations:`.

**`--mode validate` finds no checkpoint.**
It looks for `checkpoints/pinn_final.pt` (or `--checkpoint`). Train first or
pass the path explicitly.

## XAI

**SHAP KernelExplainer is very slow.**
Reduce the sweep grid and `nsamples` (50 is the default); use LIME for
iteration and reserve full SHAP for final figures.

**LIME/SHAP ranking contradicts physics intuition.**
Rankings conflate sensitivity with sweep *range width*. Equalize ranges
(e.g. min-max normalize `X`) before comparing importances.
