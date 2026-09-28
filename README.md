# Trihybrid Nanofluid LTNE PINN (PyTorch)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![PyTorch 2.0+](https://img.shields.io/badge/torch-2.0%2B-ee4c2c)](requirements.txt)
[![Tests](https://img.shields.io/badge/pytest-passing-green)](tests/)

Physics-informed neural network for **trihybrid nanofluid flow in a coaxial
cylinder** with Darcy–Forchheimer porous medium, **local thermal
non-equilibrium (LTNE)**, and a transverse magnetic field.

> **The equations live in the YAML config, not in the code.** Toggle
> physical terms, swap nanoparticles, resize the network, retune training —
> all without editing Python. Start with the
> [documentation](https://Purushothaman-natarajan.github.io/PINN/)
> (or `docs/` locally via `mkdocs serve`).

- **Framework:** PyTorch (autograd up to 3rd order), Adam/LBFGS
- **Outputs:** `f` (velocity), `theta_f` (fluid temp), `theta_s` (solid temp),
  `phi` (concentration)
- **Baseline:** `scipy.integrate.solve_bvp` mirror of the same equations
- **Metrics:** MSE / RMSE / R² with `R² > 0.95` quality gate
- **Explainability:** SHAP (global) + LIME (local) sensitivity analysis

## Quickstart

```bash
pip install -r requirements.txt
python main.py --config configs/default_trihybrid_ltne.yaml --mode baseline  # seconds
python main.py --config configs/default_trihybrid_ltne.yaml --mode all       # full pipeline
pytest -q
```

## Documentation

| I want to… | Read |
|---|---|
| Run my first case | [Quickstart](docs/quickstart.md) |
| Understand the design | [Architecture](docs/architecture.md) |
| Reuse with new parameters | [Configuration reference](docs/configuration.md), [Parameter guide](docs/parameters.md) |
| Reuse with new equations | [Equation catalog](docs/equations.md), [Extending](docs/extending.md) |
| Tune / debug training | [Training](docs/training.md), [FAQ](docs/faq.md) |
| Validate / explain / sweep | [Validation](docs/validation.md), [Explainability](docs/explainability.md), [Sweeps](docs/sweeps.md) |
| Call the code | [API reference](docs/api/index.md) |

Full site: `pip install -e ".[docs]"` then `mkdocs serve`, or
[GitHub Pages](https://Purushothaman-natarajan.github.io/PINN/) (see
`docs/` + `mkdocs.yml`).

## Repository structure

```text
configs/  default_trihybrid_ltne.yaml, ablation_study_config.yaml, schema.json
src/core/ config_loader.py (YAML + schema validation), fluid_properties.py (A1-A7)
src/models/ pinn_architecture.py (default 3x128 Tanh)
src/physics/ governing_equations.py, boundary_conditions.py
src/solvers/ pinn_trainer.py, numerical_rk45.py
src/analysis/ validation.py, shap_explainer.py, lime_explainer.py, visualization.py
tests/ test_physics.py, test_models.py, test_explain.py
docs/ full documentation site (MkDocs Material)
main.py  CLI: train | baseline | validate | shap | lime | explain | sweep | plot | all
```

## Loss

```text
Loss = w_pde * mean(R_MHD^2 + R_fluid^2 + R_solid^2 + R_conc^2)
     + w_bc  * mean(R_bc_inner^2 + R_bc_outer^2)
```

Weights `pde_weight` / `bc_weight` come from YAML. See
[Equation catalog](docs/equations.md) for the per-term mathematics.

## Defaults

- Architecture: 3 hidden layers × 128, Tanh, Xavier, Adam 1e-3
- Domain: η ∈ [0, 4], 512 collocation points
- Nanofluid: Al₂O₃ + Cu + TiO₂ in water (1% each), Brinkman + Maxwell stepwise
