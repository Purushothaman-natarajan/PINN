# Learn the methods

Every algorithmic and modelling technique used in this repository —
what it is, where it lives in the code, and the paper it comes from.
The **Status** column tells you honestly whether a technique is
**implemented here** or is **related work** (useful context, not used).

## At a glance

| # | Technique | Status here | Key reference |
|---|---|---|---|
| 1 | Physics-Informed Neural Networks | ✅ Used — the core solver | Raissi, Perdikaris & Karniadakis (2019) |
| 2 | Automatic differentiation | ✅ Used — PyTorch autograd to 3rd order | Baydin et al. (2018) |
| 3 | Collocation residual minimization | ✅ Used — random η-points each epoch | Raissi, Perdikaris & Karniadakis (2019) |
| 4 | Fixed PDE/BC loss weighting | ✅ Used (`pde_weight`, `bc_weight`) | — (cf. adaptive: Wang, Teng & Perdikaris, 2021) |
| 5 | Adam optimizer | ✅ Used (default) | Kingma & Ba (2015) |
| 6 | L-BFGS optimizer | ✅ Used (option) | Liu & Nocedal (1989) |
| 7 | Xavier init + Tanh MLP | ✅ Used (defaults) | Glorot & Bengio (2010) |
| 8 | Similarity reduction of boundary layers | ✅ Used — ODEs in η | Blasius (1908); Schlichting & Gersten |
| 9 | Brinkman viscosity model | ✅ Used — stepwise, per particle | Brinkman (1952) |
| 10 | Maxwell–Garnett conductivity model | ✅ Used — stepwise, per particle | Maxwell (1873); Garnett (1904) |
| 11 | Darcy–Forchheimer porous drag | ✅ Used — switchable terms | Darcy (1856); Forchheimer (1901) |
| 12 | LTNE two-temperature model | ✅ Used — coupled θf/θs | Nield & Bejan |
| 13 | MHD Lorentz drag + Joule heating | ✅ Used — switchable terms | Hartmann (1937) |
| 14 | Rosseland radiation approximation | ✅ Used — folded into conductivity | Rosseland (1936) |
| 15 | Soret (thermal-diffusion) coupling | ✅ Used — switchable term | Soret (1879) |
| 16 | BVP collocation baseline (`solve_bvp`) | ✅ Used — independent validation | Kierzenka & Shampine (2001) |
| 17 | SHAP global attributions | ✅ Used — parameter ranking | Lundberg & Lee (2017) |
| 18 | LIME local surrogates | ✅ Used — per-point explanations | Ribeiro, Singh & Guestrin (2016) |

## 1. Physics-Informed Neural Networks

A neural network $u_\theta(\eta)$ approximates the solution fields, and
training minimizes the **PDE residual evaluated through the network** plus
boundary-condition residuals — no labelled data required:

```text
L = w_pde · mean(R_PDE²) + w_bc · mean(R_BC²)
```

Implemented in `src/solvers/pinn_trainer.py` (`PINNTrainer`), residuals in
`src/physics/governing_equations.py`. This is the formulation of:

- M. Raissi, P. Perdikaris, G.E. Karniadakis, *"Physics-informed neural
  networks"*, J. Comput. Phys. 378 (2019). The paper that defined the
  residual-loss PINN used here (parts I & II).

## 2. Automatic differentiation

All spatial derivatives ($f'$, $f''$, $f'''$, $\theta''$, …) are exact
derivatives of the network computed by reverse-mode autodiff
(`torch.autograd.grad`, nested), never finite differences. Implemented in
`compute_derivatives()`.

- A.G. Baydin et al., *"Automatic differentiation in machine learning:
  a survey"*, J. Mach. Learn. Res. 18 (2018). Background on the technique
  PyTorch's autograd implements.

## 3. Collocation on resampled points

Each epoch draws `n_collocation` fresh uniform points in $(0, \eta_0)$
(`_sample_collocation()`), so the PDE is enforced over the whole domain
rather than a fixed mesh. Related work also uses Latin-hypercube or
adaptive sampling; uniform resampling is the Raissi et al. default and is
what we implement.

## 4. Loss weighting (fixed)

PDE and BC terms compete during optimization, so the loss carries static
weights from YAML (`pde_weight: 1.0`, `bc_weight: 10.0` — walls need
upweighting or they drift). Adaptive schemes (NTK weighting, learning-rate
annealing) exist but are **not** implemented; if training stalls, tune the
static weights per [Training](training.md#tuning-guide):

- S. Wang, Y. Teng, P. Perdikaris, *"Understanding and mitigating gradient
  flow pathologies in physics-informed neural networks"*, SIAM J. Sci.
  Comput. (2021). Diagnoses why the weighting matters.

## 5–6. Adam and L-BFGS

`optimizer: adam` (default, lr $10^{-3}$) with gradient clipping; `lbfgs`
(20 inner iterations) as a strong finisher. Scheduler: optional StepLR.

- D.P. Kingma, J. Ba, *"Adam: A method for stochastic optimization"*,
  ICLR (2015).
- D.C. Liu, J. Nocedal, *"On the limited memory BFGS method for large
  scale optimization"*, Math. Programming (1989).

## 7. Network: Tanh MLP with Xavier init

Default 3 × 128 Tanh, Xavier-normal weights — smooth activations matter
because the loss needs up to 3rd derivatives (ReLU would give zero
higher derivatives). Fully config-driven in `src/models/pinn_architecture.py`.

- X. Glorot, Y. Bengio, *"Understanding the difficulty of training deep
  feedforward neural networks"*, AISTATS (2010).

## 8. Similarity reduction

The cylindrical Navier–Stokes/energy system is reduced by similarity
variables to ODEs in $\eta$ with unknowns $f, \theta_f, \theta_s, \phi$.
This classical boundary-layer reduction is what makes a 1-D PINN
sufficient. See [Equation catalog](equations.md) for the exact terms.

- H. Blasius, *"Grenzschichten in Flüssigkeiten mit kleiner Reibung"*,
  Z. Math. Phys. (1908). The original similarity solution.
- H. Schlichting, K. Gersten, *Boundary-Layer Theory* (textbook).

## 9–10. Trihybrid effective properties

Per particle, applied stepwise in `src/core/fluid_properties.py`:

- **Viscosity (Brinkman):** $\mu \leftarrow \mu/(1-\varphi_i)^{2.5}$ —
  H.C. Brinkman, *"The viscosity of concentrated suspensions and
  solutions"*, J. Chem. Phys. (1952).
- **Conductivity (Maxwell–Garnett):** Maxwell's effective-medium formula
  applied per species — J.C. Maxwell, *A Treatise on Electricity and
  Magnetism* (1873); J.C.M. Garnett, *"Colours in metal glasses…"*,
  Phil. Trans. R. Soc. (1904).
- Density/heat capacity use linear mixtures; the ratios form the A1–A7
  coefficients (see [Parameter guide](parameters.md)).

## 11. Darcy–Forchheimer porous medium

Linear Darcy drag $-K_p f'$ plus quadratic Forchheimer inertia
$-F_r(f')^2$, each a YAML-switchable momentum term:

- H. Darcy, *Les fontaines publiques de la ville de Dijon* (1856).
- P. Forchheimer, *"Wasserbewegung durch Boden"*, Z. Ver. Deutsch. Ing. (1901).

## 12. LTNE (two-temperature) model

Fluid and solid phases carry separate temperatures $\theta_f, \theta_s$
coupled by interphase coefficients $H_s, H_{sg}$ (large values recover
thermal equilibrium). Implemented as two coupled energy residuals.

- D.A. Nield, A. Bejan, *Convection in Porous Media* (Springer).
  The standard reference for porous-medium convection and LTNE modelling.

## 13. MHD effects

Transverse magnetic field: Lorentz drag $-A_2 M f'$ in momentum plus
Joule heating $E_c M \sigma_r (f')^2$ in fluid energy.

- J. Hartmann, *"Hg-dynamics I: Theory of the laminar flow of an
  electrically conductive liquid…"* (1937).

## 14. Thermal radiation

Rosseland diffusion approximation folded into the effective conductivity
$k_{\text{eff}} = k_r + R_d$:

- S. Rosseland, *Theoretical Astrophysics* (1936).

## 15. Soret effect

Thermal-diffusion coupling $S_r\,\theta_f''$ in the concentration
equation (switchable `soret` term):

- C. Soret, *"Sur l'état d'équilibre…"*, Arch. Sci. Phys. Nat. (1879).

## 16. Independent BVP baseline

`scipy.integrate.solve_bvp` — collocation on an adaptive mesh — solves the
*same* term-switched equations as a 9-state first-order system, giving a
solver-independent reference for MSE/RMSE/R² validation:

- J. Kierzenka, L.F. Shampine, *"A BVP solver based on residual control
  and the MATLAB PSE"*, ACM Trans. Math. Softw. (2001). The algorithm
  behind `solve_bvp`.

## 17–18. Explainability: SHAP and LIME

- **SHAP** (global): which parameters matter overall —
  S.M. Lundberg, S.-I. Lee, *"A unified approach to interpreting model
  predictions"*, NeurIPS (2017).
- **LIME** (local): why this prediction at this point, via interpretable
  surrogates — M.T. Ribeiro, S. Singh, C. Guestrin, *"'Why should I trust
  you?': Explaining the predictions of any classifier"*, KDD (2016).

See [Explainability](explainability.md) for how both are applied here
(spatial + parameter modes, fallbacks when the packages are absent).
