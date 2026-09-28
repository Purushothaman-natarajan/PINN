# Training

## Loss formulation

Per iteration the trainer samples `n_collocation` fresh interior points and
forms a dynamically weighted sum (weights from YAML, never hardcoded):

```text
L = w_pde · mean(Rm² + Ref² + Res² + Rc²) + w_bc · mean(R_bc²)
```

- Residuals `Rm/Ref/Res/Rc` come from `pde_residuals()` (up to 3rd-order
  derivatives via nested `torch.autograd.grad`).
- `R_bc` stacks 5 inner + 4 outer wall residuals from `bc_residual()`.
- Gradients are clipped (norm 1.0) before the Adam/LBFGS step.

## Key settings

| Knob | Guidance |
|---|---|
| `epochs` | 20 000 default; smoke-test with ~200 first |
| `lr` | `1e-3` Adam default; reduce 10× if loss oscillates |
| `optimizer: lbfgs` | Strong finisher after Adam; slower per step (`max_iter: 20`) |
| `scheduler: {type: step, step_size: 5000, gamma: 0.5}` | Decay LR mid-run for stiff cases |
| `pde_weight` / `bc_weight` | BCs need upweighting (`10.0` default); raise `bc_weight` if wall values drift |
| `n_collocation` | 512 default; raise for sharp layers, lower for smoke runs |
| `dtype: float64` | Use for large `M/Kp/Hs/Pr`; ~2× cost, often decisive |
| `device` | `auto` → CUDA if available; override with `cpu`/`cuda` |
| `save_every` | Intermediate `pinn_epoch{N}.pt`; `pinn_final.pt` always saved |

## Tuning guide

1. **Wall values wrong** (e.g. θf(0) ≠ 1): raise `bc_weight` (try 50–100).
2. **PDE loss stalls high**: check `baseline` profiles first — the physics
   may be stiff; try `float64`, more collocation, StepLR schedule.
3. **One field lags** (often θs/φ): inspect per-term scales in the
   [Equation catalog](equations.md); consider normalizing via `Hs`/`Sc`.
4. **NaNs**: lower `lr`, confirm `phi` values ≤ 0.2 and positive properties.

## Checkpoints and reproducibility

- `checkpoints/pinn_final.pt` is the artifact `validate`/`shap`/`lime` load.
- `--checkpoint <path>` resumes training or pins evaluation to a file.
- `domain.seed` seeds torch + numpy in `main.py`.
- History dict (`total/pde/bc`) is plotted by `plot_loss`; an empty history
  (e.g. pure `validate` runs) skips the loss figure by design.
