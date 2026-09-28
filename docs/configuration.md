# Configuration reference

Every runtime choice lives in YAML. `load_config()` validates the file in
two stages: required top-level blocks, then the machine-readable
[`configs/schema.json`](https://github.com/Purushothaman-natarajan/PINN/blob/master/configs/schema.json)
(types, ranges, enums). A schema violation raises `ValueError` naming the
offending key — fix the YAML, don't touch code.

## `domain`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `eta0` | float > 0 | `4.0` | Outer similarity coordinate; domain is η ∈ [0, η₀] |
| `n_collocation` | int ≥ 8 | `512` | Interior collocation points resampled every iteration |
| `dtype` | `float32` \| `float64` | `float32` | Compute precision (`float64` helps stiff regimes; ~2× cost) |
| `seed` | int | `0` | Seeds torch + numpy in `main.py` |

## `physics`

Dimensionless groups and wall parameters. The middle column shows the
**coefficient key** each entry maps to in the flat dict built by
`compute_coefficients()` (YAML names differ in three cases).

| YAML key | Coeff key | Type | Default | Meaning |
|---|---|---|---|---|
| `M` | `M` | float ≥ 0 | `1.0` | Magnetic parameter |
| `Fr` | `Fr` | float ≥ 0 | `0.2` | Forchheimer (non-Darcy inertia) coefficient |
| `Kp` | `Kp` | float ≥ 0 | `0.5` | Porous parameter, `1/Da` |
| `Rd` | `Rd` | float ≥ 0 | `0.5` | Thermal radiation parameter (adds to conduction) |
| `Pr` | `Pr` | float > 0 | `6.2` | Prandtl number |
| `Ec` | `Ec` | float ≥ 0 | `0.1` | Eckert number (Joule/viscous dissipation scale) |
| `Hs` | `Hs` | float ≥ 0 | `1.0` | LTNE interphase coefficient, fluid side |
| `Hsg` | `Hsg` | float ≥ 0 | `1.0` | LTNE interphase coefficient, solid side |
| `Sc` | `Sc` | float > 0 | `0.6` | Schmidt number |
| `Kr` | `Kr` | float ≥ 0 | `0.2` | Chemical reaction rate |
| `Sr` | `Sr` | float ≥ 0 | `0.1` | Soret coupling coefficient |
| `Q` | `Q` | float | `0.05` | Volumetric heat source (+) / sink (−) |
| `gamma_curv` | `gamma` | float ≥ 0 | `0.1` | Cylinder curvature; `0` recovers flat plate |
| `slip` | `slip` | float ≥ 0 | `0.0` | Inner-wall velocity slip |
| `Bi_f` | `Bi_f` | float ≥ 0 \| `null` | `null` | Biot number; `null` → Dirichlet θf(0)=1, number → Robin |
| `omega_ratio` | `omega` | float | `0.0` | Outer-wall `f′(η₀)` rotation ratio |
| `f_outer` | `f_outer` | float \| `null` | `0.0` | Reserved; outer `f` is left free (rate only) |

**Adding a new scalar parameter?** Follow the
[recipe](extending.md#add-a-new-scalar-physics-parameter) — YAML key →
`combined_coefficients()` → residual term → BVP mirror → test.

## `nanofluid`

| Key | Type | Meaning |
|---|---|---|
| `base_fluid` | string | Label only (no physics attached yet) |
| `rho_f`, `mu_f`, `k_f`, `cp_f` | float > 0 | Base-fluid density, viscosity, conductivity, heat capacity |
| `sigma_f` | float ≥ 0 | Base-fluid electrical conductivity (magnetic term scale) |
| `particles` | list, 1–3 entries | Each: `name`, `rho`, `k`, `cp`, `sigma` (defaults 1e6), `phi` ∈ [0, 0.2] |

Set a particle's `phi: 0.0` to recover mono/hybrid nanofluid or pure fluid.
Mixture rules (Brinkman stepwise, Maxwell-Garnett stepwise, linear
ρ/ρcp) are detailed in [Parameter guide](parameters.md#nanofluid-mixtures)
and produce the A1–A7 ratios consumed by the PDEs.

!!! warning "Scientific notation in YAML"
    PyYAML only parses exponents **with an explicit sign** as floats:
    write `3.5e+6`, never `3.5e6` (the latter loads as a string and fails
    schema validation). The schema error names the exact key path.

## `equations`

Four term lists; only listed terms enter the residuals. The full
term → mathematics → coefficient contract is the
[Equation catalog](equations.md). Any subset is legal, but see the catalog's
*degenerate combinations* notes (e.g. dropping solid `conduction` turns the
solid equation algebraic).

## `model`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `input_dim` | `1` (const) | `1` | Similarity coordinate η |
| `output_dim` | `4` (const) | `4` | `[f, θf, θs, φ]` — order is contractual |
| `hidden_layers` | int list, non-empty | `[128, 128, 128]` | Width per hidden layer |
| `activation` | tanh/relu/gelu/silu/swish/elu/sigmoid | `tanh` | Hidden activation |
| `initialization` | xavier/kaiming/default | `xavier` | Linear-layer init |
| `output_activation` | `linear` (const) | `linear` | Final layer is always linear |

## `training`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `epochs` | int ≥ 1 | `20000` | Optimizer steps (each resamples collocation) |
| `lr` | float > 0 | `0.001` | Adam/LBFGS learning rate |
| `optimizer` | `adam` \| `lbfgs` | `adam` | LBFGS takes `max_iter: 20` inner steps |
| `scheduler` | `null` or `{type: step, step_size, gamma}` | `null` | `StepLR` decay; see [Training](training.md) |
| `pde_weight` / `bc_weight` | float ≥ 0 | `1.0` / `10.0` | Loss balance (BCs need upweighting) |
| `log_every` / `save_every` | int | `500` / `2000` | Console cadence / checkpoint cadence |
| `device` | `auto` \| `cpu` \| `cuda` | `auto` | `auto` uses CUDA when available |
| `checkpoint_dir` | string | `checkpoints` | `pinn_epoch{N}.pt` + `pinn_final.pt` |

## `solver`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `bvp_mesh` | int ≥ 10 | `200` | Initial `solve_bvp` mesh nodes |
| `tol` | float > 0 | `1e-8` | BVP tolerance |
| `max_nodes` | int | `2000` | Adaptive mesh cap |

## `data` (optional)

Supervised CSV/Excel dataset for training. Absent or `source: null` means
pure physics training (today's default).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `source` | string \| `null` | `null` | Path to `.csv`/`.xlsx`/`.xls` with `eta,f,theta_f,theta_s,phi` columns |
| `sheet` | int \| string | `0` | Excel sheet index or name |
| `columns` | object \| `null` | `null` | Alias map, e.g. `{theta_f: T_fluid}` |
| `weight` | float ≥ 0 | `0.0` | Supervised loss weight (`0` = term disabled) |
| `batch` | int ≥ 1 \| `null` | `null` | Minibatch rows per epoch (`null` = full batch) |

CLI equivalents: `--data PATH`, `--data-weight W`. Full format spec and
tuning advice: [Datasets & mock data](datasets.md#external-csvexcel-data).

## `outputs`

`raw_dir`, `processed_dir`, `baseline_file`, `metrics_file` — directory and
filename overrides for artefacts. Unknown top-level blocks are ignored
(`additionalProperties: true`), so experiment metadata can ride along.
