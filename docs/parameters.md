# Parameter guide

What each knob means, which equation terms it feeds, and how it moves the
solution. Coeff keys are the flat names in `compute_coefficients()` output.

## Dimensionless groups

| Param | Terms fed | Increase → |
|---|---|---|
| `M` magnetic | momentum `magnetic`, energy `joule` | Thinner momentum layer (Lorentz drag), hotter fluid via Joule heating |
| `Fr` Forchheimer | momentum `forchheimer` | Extra quadratic drag, reduced $f'$ |
| `Kp` = 1/Da | momentum `darcy` | Linear porous drag, reduced $f'$ |
| `Rd` radiation | fluid `conduction`/`radiation` | Larger effective conductivity, hotter/flatter $\theta_f$ |
| `Pr` Prandtl | fluid `advection` | Thinner thermal layer (water ≈ 6.2) |
| `Ec` Eckert | fluid `joule` | Stronger dissipation heating near the wall |
| `Hs` / `Hsg` LTNE | fluid/solid `interphase` | Larger → $\theta_s \to \theta_f$ (equilibrium limit); small → visible splitting |
| `Sc` Schmidt | concentration `advection`, `reaction` | Thinner concentration layer |
| `Kr` reaction | concentration `reaction` | Consumes $\phi$ (destructive, first-order) |
| `Sr` Soret | concentration `soret` | Couples $\theta_f''$ into $\phi$ |
| `Q` source | fluid `heat_source` | $Q>0$ heats, $Q<0$ cools the fluid layer |
| `gamma_curv` | momentum `curvature` ($\kappa = 1+2\gamma\eta$) | Larger cylinder curvature effect; `0` = flat plate |
| `slip` | BC #2 | Wall slip raises $f'(0)$ above 1 for given $f''(0)$ |
| `Bi_f` | BC #3 | `null` = isothermal wall; finite = convective (Robin) wall |
| `omega_ratio` | BC #6 | Outer rotation rate $f'(\eta_0)$ |

Sweep-ready parameters (used by SHAP/LIME rankings and the ablation config)
are `M`, `Rd`, `Fr`, `Hs` — but **any** `physics:` key can be swept, see
[Sweeps](sweeps.md).

## Nanofluid mixtures

`TrihybridProperties` applies, per particle in listed order:

- **Viscosity (Brinkman, stepwise):**
  $\mu \leftarrow \mu / (1-\varphi_i)^{2.5}$
- **Conductivity (Maxwell-Garnett, stepwise):** each particle upgrades the
  running medium conductivity.
- **Electrical conductivity:** same Maxwell form with $\sigma$.
- **Density / heat capacity (linear):**
  $\rho = (1-\varphi)\rho_f + \sum \varphi_i\rho_i$ (likewise $\rho c_p$).

Effective ratios feed the A-coefficients:

| Coeff | Definition | Used by |
|---|---|---|
| `A1` | $\mu_r / \rho_r$ (kinematic-viscosity ratio) | momentum diffusion scale |
| `A2` | $\sigma_r / \rho_r$ | magnetic term scale |
| `A3` | $(\rho c_p)_r$ (thermal-inertia ratio) | documented; advection uses `rho_cp_ratio` |
| `A4` | $k_r$ (conductivity ratio) | documented; energy uses `k_ratio` |
| `A5` | $\mu_r$ | documented |
| `A6` | $\sigma_r$ | documented; Joule uses `sigma_ratio` |
| `A7` | $\rho_r$ | documented |

!!! tip "Recovering simpler fluids"
    Set any particle's `phi: 0.0` to step down trihybrid → hybrid → mono →
    pure base fluid. The math handles it exactly (ratios → 1).

## Choosing values for a new study

1. Start from `configs/default_trihybrid_ltne.yaml` (water + Al₂O₃/Cu/TiO₂).
2. Change **one** group at a time; re-run `baseline` first (seconds) and
   inspect profiles before training.
3. Keep `eta0` large enough that outer BCs flatten ($\theta,\phi \to 0$);
   if profiles clip at $\eta_0$, increase it.
4. Stiff corners (large `M`, `Kp`, `Hs`, high `Pr`): prefer `float64`,
   more collocation points, and higher `bc_weight` — see
    [Training](training.md#tuning-guide).
