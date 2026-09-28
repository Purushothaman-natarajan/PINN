# Extending the repo

All extensions follow one rule: **YAML first, code second, tests always.**
Each recipe lists the exact files to touch in order.

## Add a new scalar physics parameter

Example: thermophoresis coefficient `Nt` scaling a new concentration term
$S_c N_t\,\theta_f'\phi'/\theta_f$.

1. **Schema** — `configs/schema.json`: add `"Nt": {"type": "number",
   "minimum": 0}` under `physics.properties`, and document it in
   [Configuration reference](configuration.md) + [Parameter guide](parameters.md).
2. **Coefficients** — `src/core/fluid_properties.py::combined_coefficients`:
   add `"Nt": float(physics.get("Nt", 0.0))`.
3. **PINN residual** — `src/physics/governing_equations.py::pde_residuals`:
   read `nt = float(_get(coeffs, "Nt", 0.0))`, add the term under a new key
   (e.g. `thermophoresis`) in `concentration_terms`, and register the key in
   the schema enum + [Equation catalog](equations.md).
4. **BVP mirror** — `src/solvers/numerical_rk45.py::_first_order_rhs`: add the
   identical contribution to `c_rhs` under the same key. Parity is mandatory.
5. **Config** — add `Nt:` to `configs/default_trihybrid_ltne.yaml`
   (`physics:`) with a sane default (usually `0.0` = off).
6. **Test** — `tests/test_physics.py`: assert the residual is zero when
   `Nt: 0.0` (backward compatibility) and nonzero/non-degenerate for a small
   positive value; run the BVP smoke test with the term enabled.

## Add a new equation term

Same as above minus step 1's parameter (if the term only recombines existing
coefficients): pick a key name, implement it in **both**
`pde_residuals()` and `_first_order_rhs()` under that key, extend the schema
enum for the corresponding `*_terms` list, document the math row in
[Equation catalog](equations.md), and cover it in tests.

!!! danger "Keep the highest derivatives"
    Never remove the highest derivative of a field (`f'''`, `θf''`,
    `θs''`, `φ''`) through term switches — the 9-state BVP structure and the
    BC count depend on it. See the degenerate-combination notes in the
    [Equation catalog](equations.md).

## Add a nanoparticle or base fluid

- **New particle:** append to `nanofluid.particles` in your config
  (`name`, `rho`, `k`, `cp`, `sigma`, `phi`). Up to 3 are supported and
  schema-validated; mixtures apply automatically. No code change.
- **4th particle or new mixture rule:** extend `TrihybridProperties`
  (rules are per-particle loops, so this is usually a cap change + tests),
  update the schema `maxItems`, and document the rule in
  [Parameter guide](parameters.md#nanofluid-mixtures).
- **New base fluid (e.g. ethylene glycol):** new config file with updated
  `rho_f/mu_f/k_f/cp_f/sigma_f` and an adjusted `Pr`. Copy
  `default_trihybrid_ltne.yaml` — never edit it in place for a new study.

## Add an activation, optimizer or scheduler

- Activation: extend `_ACTIVATIONS` in `pinn_architecture.py`, add the enum
  value to the schema, test `build_pinn` with it.
- Optimizer/scheduler: extend `PINNTrainer.__init__`, update schema enums,
  document in [Training](training.md).

## Add a QoI or CLI mode

- QoI (e.g. skin friction): compute from baseline/prediction dicts in the
  sweep or XAI module, name it in `qoi_names`, add a CSV column in `cmd_sweep`.
- CLI mode: add `cmd_*` in `main.py`, register the choice in `build_parser`,
  document the row in [Quickstart](quickstart.md#cli-modes).
