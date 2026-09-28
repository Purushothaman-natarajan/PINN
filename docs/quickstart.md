# Quickstart

## 1. Install

```bash
git clone https://github.com/Purushothaman-natarajan/PINN.git
cd PINN
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q        # expect: all pass
```

Optional extras:

```bash
pip install -e ".[docs]"   # MkDocs site tooling (this documentation)
pip install -e ".[dev]"    # lint/format/test tooling
```

Requires Python ≥ 3.10 and a CPU; CUDA is used automatically when
`training.device: auto` resolves to GPU (see [Training](training.md)).

## 2. Run the fast path first: numerical baseline

The BVP baseline solves in seconds and validates your environment before any
expensive training:

```bash
python main.py --config configs/default_trihybrid_ltne.yaml --mode baseline
# → data/raw/baseline.npz
```

## 3. Smoke-train (2 minutes) before full training

Full training defaults to 20 000 epochs. For a first end-to-end check, copy
the default config and shrink it:

```bash
cp configs/default_trihybrid_ltne.yaml configs/dev_smoke.yaml
# edit dev_smoke.yaml: training.epochs: 200, training.log_every: 50
python main.py --config configs/dev_smoke.yaml --mode train
python main.py --config configs/dev_smoke.yaml --mode validate
# → data/processed/metrics.json, data/processed/figures/
```

`validate` prints per-field MSE/RMSE/R² and the `R² > 0.95` gate verdict.
The smoke run will *fail* the gate — that is expected; it only proves the
pipeline. The full run is next.

## 4. Full pipeline

```bash
python main.py --config configs/default_trihybrid_ltne.yaml --mode all
```

This trains, solves the baseline, validates, and renders loss/profile/parity
figures. Then explain:

```bash
python main.py --config configs/default_trihybrid_ltne.yaml --mode explain  # SHAP + LIME
python main.py --config configs/ablation_study_config.yaml --mode sweep     # parameter grid
```

## CLI modes

| Mode | What it does | Reads | Writes |
|---|---|---|---|
| `train` | Trains PINN, saves checkpoints | config | `checkpoints/*.pt` |
| `baseline` | `solve_bvp` reference solution | config | `data/raw/baseline.npz` |
| `validate` | PINN vs baseline metrics + figures | config, checkpoint, baseline | `data/processed/` |
| `shap` | Global attributions (spatial + params) | config, checkpoint | `data/processed/shap/` |
| `lime` | Local surrogate explanations | config, checkpoint | `data/processed/lime/` |
| `explain` | `shap` then `lime` | as above | as above |
| `sweep` | Ablation grid → CSV of QoIs | ablation config | `data/processed/*.csv` |
| `plot` | Alias: re-validate + re-render figures | as `validate` | `data/processed/figures/` |
| `all` | `train` → `baseline` → `validate` | config | all of the above |

Common flags: `--checkpoint <path>` (resume or evaluate a specific model),
`--outdir <dir>` (SHAP/LIME output location), `--n-points <int>` (baseline
resolution).
