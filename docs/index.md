# Trihybrid LTNE PINN

A **config-driven**, modular Physics-Informed Neural Network (PyTorch) for
simulating **fluid flow, heat transfer and mass transfer** of a **trihybrid
nanofluid in a coaxial cylinder**, with a **Darcy–Forchheimer porous
medium**, a **local thermal non-equilibrium (LTNE)** model and a **transverse
magnetic field**.

[Get started](quickstart.md){ .md-button .md-button--primary }
[Learn the methods](learn.md){ .md-button }
[GitHub](https://github.com/Purushothaman-natarajan/PINN){ .md-button }

The headline idea: **the equations live in the YAML config, not in the
code.** Toggle physical terms, swap nanoparticles, resize the network or
retune training — all without editing Python. See
[Equation catalog](equations.md) and [Extending the repo](extending.md).

## What you get

| Capability | Implementation |
|---|---|
| PINN solver | PyTorch, autograd to 3rd order, Adam/LBFGS, weighted PDE+BC loss |
| Network default | 3 × 128, Tanh, Xavier — fully config-driven |
| Outputs | `f` velocity, `θf` fluid temp, `θs` solid temp, `φ` concentration |
| Numerical baseline | `scipy.integrate.solve_bvp` mirror of the same term switches |
| Validation | MSE / RMSE / R² per field, `R² > 0.95` quality gate, parity plots |
| Explainability | SHAP (global attributions) + LIME (local surrogates) |
| Sweeps | Cartesian ablation over any physics parameters → CSV + rankings |

## Where to go next

- **New here?** → [Quickstart](quickstart.md) (5 minutes, baseline first)
- **Reuse with new parameters?** → [Configuration reference](configuration.md),
  then [Parameter guide](parameters.md)
- **Reuse with new equations?** → [Equation catalog](equations.md),
  then the [extension recipes](extending.md)
- **Understand the data?** → [Datasets & mock data](datasets.md)
  (generate → clean → train → validate lifecycle)
- **Add a nanoparticle / fluid?** → [Extending: new material](extending.md#add-a-nanoparticle-or-base-fluid)
- **Tune or debug training?** → [Training](training.md) + [FAQ](faq.md)
- **Code API?** → [API reference](api/index.md)

## Repository map

```text
configs/
  default_trihybrid_ltne.yaml   # full physics + model + training setup
  ablation_study_config.yaml    # sweep definition (grid over physics keys)
  schema.json                   # machine-readable config schema (validated on load)
src/
  core/        # config_loader (YAML + schema validation), fluid_properties (A1–A7)
  models/      # pinn_architecture (config-driven MLP)
  physics/     # governing_equations (term-switched residuals), boundary_conditions
  solvers/     # pinn_trainer (Adam/LBFGS loop), numerical_rk45 (solve_bvp baseline)
  analysis/    # validation, shap_explainer, lime_explainer, visualization
tests/         # physics, models, explainability
main.py        # CLI: train | baseline | validate | shap | lime | explain | sweep | plot | all
docs/          # this site
```

!!! warning "Physics source of truth"
    The similarity ODEs implemented here are a documented, standard-form
    proposal for the stated problem class (see [Equation catalog](equations.md)
    for exact per-term mathematics). If you have a reference paper with
    different coefficients, follow [Extending: new equation term](extending.md#add-a-new-equation-term)
    — it is a ~15-minute change by design.
