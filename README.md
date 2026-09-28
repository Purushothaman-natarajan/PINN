# PINN — Trihybrid Nanofluid Flow, Heat & Mass Transfer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![PyTorch 2.0+](https://img.shields.io/badge/torch-2.0%2B-ee4c2c)](requirements.txt)
[![Tests passing](https://img.shields.io/badge/pytest-19_passing-green)](tests/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-material-blue)](https://purushothaman-natarajan.github.io/PINN/)

A **Physics-Informed Neural Network (PyTorch)** that simulates fluid flow,
heat transfer and mass transfer for a **trihybrid nanofluid in a coaxial
cylinder** — with a Darcy–Forchheimer porous medium, local thermal
non-equilibrium (LTNE) and a transverse magnetic field.

**The core idea:** the physics lives in a YAML config file, not in code.
Change parameters, switch equation terms, swap nanoparticles, resize the
network — no Python edits needed.

## What it predicts

From the similarity coordinate `η ∈ [0, η₀]`, the network outputs four fields:

| Output | Meaning |
|---|---|
| `f` | Velocity (stream function) |
| `θf` | Fluid temperature (LTNE fluid phase) |
| `θs` | Solid temperature (LTNE solid phase) |
| `φ` | Nanoparticle concentration |

Predictions are validated against an independent `solve_bvp` numerical
solver on the *same* equations (target: mean R² > 0.95), and explained with
SHAP (global) + LIME (local) analysis.

## Quickstart

> **Never used a terminal?** Start with the click-by-click
> [Start here guide](docs/start-here.md) — no experience assumed.

```bash
git clone https://github.com/Purushothaman-natarajan/PINN.git
cd PINN
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**1. Numerical baseline first** (seconds — proves your setup works):

```bash
python main.py --config configs/default_trihybrid_ltne.yaml --mode baseline
```

**2. Mock-data pipeline** (generate 1000-point reference data → clean →
train → validate, over a parameter grid):

```bash
python main.py --config configs/mock_train_1000.yaml --mode mock
```

Flags: `--start N --end M` trains only combos N–M (resumable),
`--force` regenerates existing artefacts.

**External data:** train on your own CSV/Excel measurements with the
`data:` config block (or `--data file.csv --data-weight 1.0`) — adds a
supervised misfit term to the loss. Convert formats with
`--mode export`. Details: [Datasets & mock data](docs/datasets.md).

**3. Individual stages**, all driven by `--config`:

```bash
python main.py --config configs/default_trihybrid_ltne.yaml --mode train     # train PINN
python main.py --config configs/default_trihybrid_ltne.yaml --mode validate  # MSE/RMSE/R² + plots
python main.py --config configs/default_trihybrid_ltne.yaml --mode explain   # SHAP + LIME
python main.py --config configs/ablation_study_config.yaml --mode sweep      # parameter grid
pytest -q                                                                     # test suite
```

| Mode | Does | Produces |
|---|---|---|
| `train` | Trains the PINN (Adam/LBFGS, PDE+BC loss) | `checkpoints/` |
| `baseline` | Reference solution via `solve_bvp` | `data/raw/` |
| `mock` | Generate → clean → train → validate per parameter combo | `data/`, `checkpoints/mock1000/`, summary CSV |
| `validate` | PINN vs reference: metrics + figures | `data/processed/` |
| `shap` / `lime` / `explain` | Parameter + spatial sensitivity analysis | `data/processed/shap\|lime/` |
| `sweep` | Fast BVP-only ablation grid | `data/processed/*.csv` |
| `plot` | Re-render figures from saved artefacts | `data/processed/figures/` |
| `all` | `train` → `baseline` → `validate` | everything above |

## How configuration works

Everything — physics, equations, network, training — comes from YAML
(validated against [`configs/schema.json`](configs/schema.json) on load):

```yaml
physics:
  M: 1.0    # magnetic parameter → Lorentz drag + Joule heating
  Rd: 0.5   # thermal radiation → extra conductivity
equations:
  momentum_terms: [viscous, inertia, magnetic, darcy, forchheimer, curvature]
model:
  hidden_layers: [128, 128, 128]   # default 3×128, Tanh
  activation: tanh
training:
  epochs: 5000
```

Toggling a term name changes the PDE solved by **both** the PINN and the
baseline solver, so validation always compares like with like.

## Project layout

```text
configs/
  default_trihybrid_ltne.yaml   # flagship physics + model + training setup
  mock_train_1000.yaml          # 1000-pt mock-data grid (M × Rd × Fr × Hs = 81 combos)
  ablation_study_config.yaml    # fast BVP-only sweep definition
  schema.json                   # config schema, enforced automatically
src/
  core/     # config loading/validation, trihybrid fluid properties (A1–A7)
  data/     # mock-data generation (BVP grids) + deterministic cleaning
  models/   # config-driven PyTorch MLP: η → (f, θf, θs, φ)
  physics/  # term-switched PDE residuals (autograd ≤ 3rd order) + wall BCs
  solvers/  # PINN trainer (weighted PDE+BC loss) + solve_bvp baseline
  analysis/ # validation metrics, SHAP, LIME, figures
tests/      # physics, models, data pipeline, explainability
docs/       # full documentation site (MkDocs Material)
main.py     # CLI entry point
```

> **Note:** generated data (`data/raw`, `data/processed`) and trained
> models (`checkpoints/`) are deliberately **not** committed — they are
> reproducible with the commands above. Only source, configs, docs and tests
> live in git.

## Results so far

- Pilot combo (M=1, Rd=0.5, Fr=0.2, Hs=1, 5000 epochs CPU): **mean R² = 0.9997** ✅
- 81-combo mock grid: all BVP datasets converge and pass physicality checks
  (fields in [0, 1], monotone decay, visible LTNE splitting)
- Full 81-model training sweep in progress — per-combo metrics land in
  `data/processed/mock_sweep_summary.csv`

## License and author

Open source under the **[MIT License](LICENSE)** — free to use, modify and
distribute with attribution.

Developed by **[Purushothaman Natarajan](https://purushothaman-natarajan.github.io/)**.
Bug reports and ideas are welcome via
[GitHub issues](https://github.com/Purushothaman-natarajan/PINN/issues).

## Documentation

Full site: **[purushothaman-natarajan.github.io/PINN](https://purushothaman-natarajan.github.io/PINN/)**
(local: `pip install -e ".[docs]"` then `mkdocs serve`)

| I want to… | Read |
|---|---|
| Run my first case, step by step from zero | [Start here](docs/start-here.md) |
| Run my first case | [Quickstart](docs/quickstart.md) |
| Learn the methods + papers behind them | [Learn the methods](docs/learn.md) |
| Understand the design | [Architecture](docs/architecture.md) |
| Change parameters | [Configuration reference](docs/configuration.md), [Parameter guide](docs/parameters.md) |
| Change equations | [Equation catalog](docs/equations.md), [Extending](docs/extending.md) |
| Tune or debug training | [Training](docs/training.md), [FAQ](docs/faq.md) |
| Understand the data (generate → clean → train) | [Datasets & mock data](docs/datasets.md) |
| Call functions directly | [API reference](docs/api/index.md) |
