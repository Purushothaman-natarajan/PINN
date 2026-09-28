# Changelog

All notable changes to this project are recorded here.

## [Unreleased]

### Added

- External CSV/Excel training data: `src/data/tabular.py`
  (load/export/aliases), optional supervised misfit loss in `PINNTrainer`
  (`data_weight`, `data_batch`, `data` history curve), `data:` config block
  + schema, `--data/--data-weight/--format` flags, `--mode export`
- `openpyxl` dependency for Excel I/O

### Added

- Full documentation site (`docs/`, MkDocs Material): quickstart,
  architecture, configuration reference, equation catalog, parameter guide,
  extension recipes, training/validation/XAI/sweep guides, FAQ, API reference
- `configs/schema.json`: machine-readable config schema, enforced on load
  via `validate_config()` in `src/core/config_loader.py`
- `Bi_f` (Robin wall) and `f_outer` pass-through in
  `combined_coefficients()`; Robin mirrored in the BVP `_bc_fun` for
  PINN/BVP parity

## [0.1.0] — 2026-09-28

### Added

- Config-driven PyTorch PINN for trihybrid nanofluid, coaxial cylinder,
  Darcy–Forchheimer, LTNE, magnetic field
- Term-switched governing equations + boundary conditions from YAML
- `solve_bvp` numerical baseline; MSE/RMSE/R² validation with 0.95 gate
- SHAP (global) and LIME (local) explainability; ablation sweeps
- CLI (`train|baseline|validate|shap|lime|explain|sweep|plot|all`), pytest suite
