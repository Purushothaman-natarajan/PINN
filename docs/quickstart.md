# Quickstart

> **New to terminals?** Start with [Start here](start-here.md) — it covers
> installing Python, opening a terminal, and your first run click by click.
> This page is the same workflow with the *why* behind each step.

## 1. Install

```bash
git clone https://github.com/Purushothaman-natarajan/PINN.git
cd PINN
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q        # expect: all pass
```

What this does: downloads the project, steps into it, creates an isolated
library workspace (`.venv`), installs everything listed in
`requirements.txt`, then runs the test suite. The workspace stays active
while your terminal is open (look for the `(.venv)` prefix); re-run the
`activate` line if you open a new terminal.

Optional extras:

```bash
pip install -e ".[docs]"   # this documentation site (mkdocs serve)
pip install -e ".[dev]"    # lint/format tooling (black, ruff)
```

Requires Python ≥ 3.10 and a CPU; an NVIDIA GPU is used automatically when
`training.device: auto` finds CUDA (see [Training](training.md)).

## 2. Baseline first (seconds, always works)

The numerical baseline solves the equations with a traditional mesh method.
Run it before any training — if this fails, the problem is your setup, not
the neural network:

```bash
python main.py --config configs/default_trihybrid_ltne.yaml --mode baseline
# → data/raw/baseline.npz  (400-pt reference: eta, f, theta_f, theta_s, phi)
```

## 3. Practice training (minutes, expected to “fail”)

Full training is 20 000 epochs (hours on CPU). Prove the whole loop first
with a 200-epoch copy:

=== "Windows"

    ```bash
    copy configs\default_trihybrid_ltne.yaml configs\dev_smoke.yaml
    ```

=== "macOS / Linux"

    ```bash
    cp configs/default_trihybrid_ltne.yaml configs/dev_smoke.yaml
    ```

Edit `dev_smoke.yaml` → `training.epochs: 200`, `training.log_every: 50`,
then:

```bash
python main.py --config configs/dev_smoke.yaml --mode train
python main.py --config configs/dev_smoke.yaml --mode validate
# → data/processed/metrics.json, data/processed/figures/
```

`validate` prints per-field MSE/RMSE/R² plus the verdict line
`Accuracy gate R2>0.95: PASS/FAIL`. The 200-epoch run will say **FAIL** —
that is the *correct* outcome for a practice run (our full run reaches
R² = 0.9997). What you are checking: numbers print, files appear,
`profiles.png` shows curves roughly tracking the baseline.

## 4. The real runs

```bash
# Everything for one config: train → baseline → validate
python main.py --config configs/default_trihybrid_ltne.yaml --mode all

# 1000-pt mock datasets + training across 81 parameter combos (resumable slices)
python main.py --config configs/mock_train_1000.yaml --mode mock --start 0 --end 7

# Explain a trained model / sweep cheaply without training
python main.py --config configs/default_trihybrid_ltne.yaml --mode explain  # SHAP + LIME
python main.py --config configs/ablation_study_config.yaml --mode sweep     # BVP-only grid

# Train on your own measurements
python main.py --config configs/default_trihybrid_ltne.yaml --mode train \
  --data data/measurements.csv --data-weight 1.0
```

How the mock pipeline works and what each file means is covered in
[Datasets & mock data](datasets.md). The `data:` config block alternative
is documented in [Configuration reference](configuration.md#data-optional).

## CLI modes

| Mode | What it does | Reads | Writes |
|---|---|---|---|
| `train` | Trains PINN, saves checkpoints | config (+ optional `--data`) | `checkpoints/*.pt` |
| `baseline` | `solve_bvp` reference solution | config | `data/raw/baseline.npz` |
| `mock` | Generate → clean → train → validate per combo | mock config | `data/`, `checkpoints/mock1000/`, summary CSV |
| `export` | Convert datasets `.npz` ↔ `.csv`/`.xlsx` | `--data` file | `--outdir` |
| `validate` | PINN vs baseline metrics + figures | config, checkpoint, baseline | `data/processed/` |
| `shap` | Global attributions (spatial + params) | config, checkpoint | `data/processed/shap/` |
| `lime` | Local surrogate explanations | config, checkpoint | `data/processed/lime/` |
| `explain` | `shap` then `lime` | as above | as above |
| `sweep` | Ablation grid → CSV of QoIs | ablation config | `data/processed/*.csv` |
| `plot` | Alias: re-validate + re-render figures | as `validate` | `data/processed/figures/` |
| `all` | `train` → `baseline` → `validate` | config | all of the above |

Common flags: `--checkpoint <path>` (resume or evaluate a specific model),
`--outdir <dir>` (output location), `--n-points <int>` (baseline
resolution), `--data PATH` + `--data-weight W` (supervised CSV/Excel),
`--start/--end` (mock slice), `--force` (regenerate), `--format` (export).
