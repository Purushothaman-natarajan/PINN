# Trihybrid Nanofluid LTNE PINN (PyTorch)

Physics-informed neural network for **trihybrid nanofluid flow in a coaxial
cylinder** with Darcy–Forchheimer porous medium, **local thermal
non-equilibrium (LTNE)**, and a transverse magnetic field.

- **Framework:** PyTorch (autograd up to 3rd order)
- **Config-driven:** fluid parameters, equations, layers, epochs all in YAML
- **Outputs:** `f` (velocity), `theta_f` (fluid temp), `theta_s` (solid temp),
  `phi` (concentration)
- **Baseline:** `scipy.integrate.solve_bvp` mesh solver for validation
- **Metrics:** MSE / RMSE / R² with `R² > 0.95` gate
- **Explainability:** SHAP (global) + LIME (local) spatial & parameter sensitivity

## Structure

```text
configs/  default_trihybrid_ltne.yaml, ablation_study_config.yaml
src/core/ config_loader.py, fluid_properties.py (A1-A7)
src/models/ pinn_architecture.py (default 3x128 Tanh)
src/physics/ governing_equations.py, boundary_conditions.py
src/solvers/ pinn_trainer.py, numerical_rk45.py
src/analysis/ validation.py, shap_explainer.py, lime_explainer.py, visualization.py
tests/ test_physics.py, test_models.py, test_explain.py
main.py  CLI
```

## Equations are given in the config

`configs/default_trihybrid_ltne.yaml` contains an `equations` block:

```yaml
equations:
  momentum_terms: [viscous, inertia, magnetic, darcy, forchheimer, curvature]
  fluid_energy_terms: [conduction, radiation, advection, interphase, joule, heat_source]
  solid_energy_terms: [conduction, interphase]
  concentration_terms: [diffusion, advection, reaction, soret]
```

`src/physics/governing_equations.py::pde_residuals` dynamically sums only the
listed terms. Change the physics by editing YAML — no Python edits needed.
The BVP solver mirrors the same term switches.

## Quickstart

```bash
pip install -r requirements.txt

# Full pipeline: train + baseline + validate + plots
python main.py --config configs/default_trihybrid_ltne.yaml --mode all

# Individual stages
python main.py --config configs/default_trihybrid_ltne.yaml --mode train
python main.py --config configs/default_trihybrid_ltne.yaml --mode baseline
python main.py --config configs/default_trihybrid_ltne.yaml --mode validate
python main.py --config configs/default_trihybrid_ltne.yaml --mode shap
python main.py --config configs/default_trihybrid_ltne.yaml --mode lime
python main.py --config configs/default_trihybrid_ltne.yaml --mode explain  # SHAP + LIME

# Parameter sweep (ablation)
python main.py --config configs/ablation_study_config.yaml --mode sweep

# Tests
pytest -q
```

## Loss

```text
Loss = w_pde * mean(R_MHD^2 + R_fluid^2 + R_solid^2 + R_conc^2)
     + w_bc  * mean(R_bc_inner^2 + R_bc_outer^2)
```

Weights `pde_weight` / `bc_weight` come from YAML. Gradients use nested
`torch.autograd.grad` for 1st–3rd order derivatives.

## Defaults

- Architecture: 3 hidden layers × 128, Tanh, Xavier, Adam 1e-3
- Domain: η ∈ [0, 4], 512 collocation points
- Nanofluid: Al₂O₃ + Cu + TiO₂ in water (1% each), Brinkman + Maxwell stepwise
