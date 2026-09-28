# Equation catalog

This page is the **contract** between your YAML config and the mathematics.
Each toggle name in the `equations:` block contributes one additive term to
a residual; the PINN drives all residuals to zero and the BVP solver solves
the rearranged equivalent. Both solvers consume the same term sets — if you
toggle a term, **both** change together.

Notation: primes are $d/d\eta$, curvature factor
$\kappa(\eta) = 1 + 2\gamma\eta$. Coefficients (`A1`, `M`, …) come from
[Configuration reference](configuration.md) and [Parameter guide](parameters.md).

!!! warning "Source of truth"
    These are standard-form similarity equations for the stated problem
    class, not transcriptions of a specific paper. If your reference uses
    different scalings, see
    [Add a new equation term](extending.md#add-a-new-equation-term).

## Momentum: $R_m = 0$ (3rd order in $f$)

| Term key | Contribution to $R_m$ | Coeffs used |
|---|---|---|
| `viscous` | $f'''/A_1$, or $\kappa f'''/A_1$ when `curvature` is also on | `A1` |
| `curvature` | $2\gamma f''/A_1$ (cylindrical correction) | `gamma`, `A1` |
| `inertia` | $f\,f'' - (f')^2$ | — |
| `magnetic` | $-A_2 M f'$ (Lorentz drag) | `A2`, `M` |
| `darcy` | $-K_p f'$ (porous drag) | `Kp` |
| `forchheimer` | $-F_r (f')^2$ (non-Darcy inertia) | `Fr` |

$$R_m = \frac{[\kappa]\,f'''}{A_1} + \frac{[2\gamma f'']}{A_1}
+ [f f''-(f')^2] - [A_2 M f'] - [K_p f'] - [F_r(f')^2]$$

Bracketed groups enter only when their term key is listed.

!!! note "Known behavior: `viscous` off"
    With `viscous` absent the code still adds the bare $f'''/A_1$
    (unscaled by $\kappa$). The momentum equation therefore never loses its
    highest derivative — the BVP stays 9-state and well-posed. Treat
    `viscous` as "curvature-scaled diffusion on/off" rather than full removal.

## Fluid energy (LTNE): $R_{ef} = 0$ (2nd order in $\theta_f$)

| Term key | Contribution to $R_{ef}$ | Coeffs used |
|---|---|---|
| `conduction` / `radiation` | $(k_r + R_d)\,\theta_f''$ (either key enables it) | `k_ratio`, `Rd` |
| `advection` | $P_r\,r_{cp}\,f\,\theta_f'$ | `Pr`, `rho_cp_ratio` |
| `interphase` | $H_s(\theta_s - \theta_f)$ (LTNE coupling) | `Hs` |
| `joule` | $E_c M \sigma_r (f')^2$ (Joule heating) | `Ec`, `M`, `sigma_ratio` |
| `heat_source` | $Q\,\theta_f$ | `Q` |

$$R_{ef} = (k_r+R_d)\theta_f'' + P_r r_{cp} f\theta_f'
+ H_s(\theta_s-\theta_f) + E_c M\sigma_r(f')^2 + Q\theta_f$$

## Solid energy (LTNE): $R_{es} = 0$ (2nd order in $\theta_s$) {#solid-energy}

| Term key | Contribution to $R_{es}$ | Coeffs used |
|---|---|---|
| `conduction` | $\theta_s''$ | — |
| `interphase` | $H_{sg}(\theta_f - \theta_s)$ | `Hsg` |

$$R_{es} = \theta_s'' + H_{sg}(\theta_f - \theta_s)$$

!!! danger "Degenerate combination"
    Dropping solid `conduction` while keeping `interphase` reduces the solid
    equation to the **algebraic** constraint $\theta_s = \theta_f$. The BVP
    solver sets $\theta_s'' = 0$ in that case, which generally conflicts with
    the two solid BCs — `solve_bvp` may fail to converge. Keep `conduction`
    unless you also relax the solid BCs (custom code).

## Concentration: $R_c = 0$ (2nd order in $\phi$)

| Term key | Contribution to $R_c$ | Coeffs used |
|---|---|---|
| `diffusion` | $\phi''$ | — |
| `advection` | $S_c f \phi'$ | `Sc` |
| `reaction` | $-S_c K_r \phi$ | `Sc`, `Kr` |
| `soret` | $S_r\,\theta_f''$ (thermal-diffusion coupling) | `Sr` |

$$R_c = \phi'' + S_c f\phi' - S_c K_r\phi + S_r\theta_f''$$

Dropping `diffusion` leaves a first-order equation but both $\phi$ BCs are
still enforced — expect convergence trouble. Same guidance as above.

## Boundary conditions (9 residuals)

Inner wall $\eta = 0$ (stretching cylinder), 5 conditions:

| # | Residual | Note |
|---|---|---|
| 1 | $f(0) = 0$ | no penetration |
| 2 | $f'(0) = 1 + \mathrm{slip}\cdot f''(0)$ | stretching + slip |
| 3 | $\theta_f(0) = 1$, or $\theta_f'(0) + B_i(\theta_f(0)-1) = 0$ if `Bi_f` set | Dirichlet vs Robin |
| 4 | $\theta_s(0) = 1$ | solid wall temp |
| 5 | $\phi(0) = 1$ | wall concentration |

Outer wall $\eta = \eta_0$ (rotating cylinder), 4 conditions:

| # | Residual | Note |
|---|---|---|
| 6 | $f'(\eta_0) = \omega$ | rotation ratio |
| 7–9 | $\theta_f = \theta_s = \phi = 0$ | ambient |

Outer $f(\eta_0)$ is intentionally **free** (rate-only constraint); the
`f_outer` key is reserved. The BVP `_bc_fun` mirrors Dirichlet/Robin/slip
exactly — parity is tested in `tests/test_physics.py`.

## Worked examples

**Pure-fluid flat plate, no porous/magnetic effects:**

```yaml
physics:
  M: 0.0
  Kp: 0.0
  Fr: 0.0
  gamma_curv: 0.0
equations:
  momentum_terms: [viscous, inertia]   # magnetic/darcy/forchheimer/curvature gone
```

**Thermal equilibrium instead of LTNE** (single temperature): set
`Hs`/`Hsg` large (e.g. `50.0`) so $\theta_s \to \theta_f$, keeping both
equations — no code change.

**No Soret coupling:** drop `soret` from `concentration_terms`.
