"""Config-driven governing equations (PyTorch autograd).

The PDE system is assembled dynamically from the ``equations`` config block,
so the "equation is given in the config": toggle term names to change the
physics without editing Python code.

Similarity ODEs on eta in [0, eta0] with unknowns:
    f (stream function), theta_f (fluid temp), theta_s (solid temp),
    phi (concentration).

Momentum (3rd order), fluid/solid energy (2nd order, LTNE coupled),
concentration (2nd order). Cylindrical curvature, Darcy-Forchheimer,
magnetic, radiation, Joule, Soret terms are individually switchable.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def _get(
    d: Dict[str, Any], key: str, default: float = 0.0
) -> torch.Tensor | float:
    """Fetch coefficient supporting float or tensor values."""
    return d.get(key, default)


def compute_derivatives(
    model: torch.nn.Module, eta: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """Compute up to 3rd-order derivatives w.r.t. eta via autograd.

    Args:
        model: PINN mapping (N,1) -> (N,4).
        eta: Collocation points (N,1) with requires_grad=True.

    Returns:
        Tuple (f, tf, ts, ph, fp, tfp, tsp, php, fpp, tfpp, tspp, phpp,
        fppp) where p denotes d/deta.
    """
    eta.requires_grad_(True)
    out = model(eta)
    f = out[:, 0:1]
    t_f = out[:, 1:2]
    t_s = out[:, 2:3]
    ph = out[:, 3:4]

    def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """First derivative with graph retention for higher orders."""
        return torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]

    f_p = grad(f, eta)
    tf_p = grad(t_f, eta)
    ts_p = grad(t_s, eta)
    ph_p = grad(ph, eta)

    f_pp = grad(f_p, eta)
    tf_pp = grad(tf_p, eta)
    ts_pp = grad(ts_p, eta)
    ph_pp = grad(ph_p, eta)

    f_ppp = grad(f_pp, eta)
    return (f, t_f, t_s, ph, f_p, tf_p, ts_p, ph_p, f_pp, tf_pp, ts_pp, ph_pp, f_ppp)


def pde_residuals(
    eta: torch.Tensor,
    comps: Dict[str, torch.Tensor],
    derivs: Dict[str, torch.Tensor],
    coeffs: Dict[str, Any],
    equations: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Assemble PDE residuals from enabled config terms.

    Args:
        eta: Collocation coordinates (N,1).
        comps: Dict with f, theta_f, theta_s, phi tensors (N,1).
        derivs: Dict with fp, fpp, fppp, tfp, tfpp, tsp, tspp, php, phpp.
        coeffs: Flat coefficient dict from fluid_properties.
        equations: The ``equations`` config block listing active terms.

    Returns:
        Dict with residual tensors Rm, Ref, Res, Rc each (N,1).
    """
    f = comps["f"]
    t_f = comps["theta_f"]
    t_s = comps["theta_s"]
    ph = comps["phi"]
    f_p = derivs["fp"]
    f_pp = derivs["fpp"]
    f_ppp = derivs["fppp"]
    tf_p = derivs["tfp"]
    tf_pp = derivs["tfpp"]
    ts_pp = derivs["tspp"]
    ph_p = derivs["php"]
    ph_pp = derivs["phpp"]

    m_terms = set(equations.get("momentum_terms", []))
    e_terms = set(equations.get("fluid_energy_terms", []))
    s_terms = set(equations.get("solid_energy_terms", []))
    c_terms = set(equations.get("concentration_terms", []))

    a1 = float(_get(coeffs, "A1", 1.0))
    a2 = float(_get(coeffs, "A2", 1.0))
    m_mag = float(_get(coeffs, "M", 0.0))
    kp = float(_get(coeffs, "Kp", 0.0))
    fr = float(_get(coeffs, "Fr", 0.0))
    gamma = float(_get(coeffs, "gamma", 0.0))
    curv = 1.0 + 2.0 * gamma * eta

    # --- Momentum residual ---
    inv_a1 = 1.0 / a1 if a1 != 0 else 1.0
    r_m = torch.zeros_like(eta)
    if "viscous" in m_terms:
        # Effective viscous diffusion scaled by 1/A1 (kinematic ratio).
        visc = f_ppp * inv_a1 if "curvature" not in m_terms else curv * f_ppp * inv_a1
        r_m = r_m + visc
    else:
        r_m = r_m + f_ppp * inv_a1
    if "curvature" in m_terms:
        r_m = r_m + 2.0 * gamma * f_pp * inv_a1
    if "inertia" in m_terms:
        r_m = r_m + (f * f_pp - f_p**2)
    if "magnetic" in m_terms:
        r_m = r_m - a2 * m_mag * f_p
    if "darcy" in m_terms:
        r_m = r_m - kp * f_p
    if "forchheimer" in m_terms:
        r_m = r_m - fr * f_p**2

    # --- Fluid energy residual (LTNE) ---
    k_r = float(_get(coeffs, "k_ratio", 1.0))
    rd = float(_get(coeffs, "Rd", 0.0))
    pr = float(_get(coeffs, "Pr", 6.2))
    rcp = float(_get(coeffs, "rho_cp_ratio", 1.0))
    h_s = float(_get(coeffs, "Hs", 1.0))
    ec = float(_get(coeffs, "Ec", 0.0))
    sig_r = float(_get(coeffs, "sigma_ratio", 1.0))
    q_src = float(_get(coeffs, "Q", 0.0))
    k_eff = k_r + rd
    r_ef = torch.zeros_like(eta)
    if "conduction" in e_terms or "radiation" in e_terms:
        r_ef = r_ef + k_eff * tf_pp
    if "advection" in e_terms:
        r_ef = r_ef + pr * rcp * f * tf_p
    if "interphase" in e_terms:
        r_ef = r_ef + h_s * (t_s - t_f)
    if "joule" in e_terms:
        r_ef = r_ef + ec * m_mag * sig_r * f_p**2
    if "heat_source" in e_terms:
        r_ef = r_ef + q_src * t_f

    # --- Solid energy residual (LTNE) ---
    hsg = float(_get(coeffs, "Hsg", 1.0))
    r_es = torch.zeros_like(eta)
    if "conduction" in s_terms:
        r_es = r_es + ts_pp
    if "interphase" in s_terms:
        r_es = r_es + hsg * (t_f - t_s)

    # --- Concentration residual ---
    sc = float(_get(coeffs, "Sc", 0.6))
    kr = float(_get(coeffs, "Kr", 0.0))
    sr = float(_get(coeffs, "Sr", 0.0))
    r_c = torch.zeros_like(eta)
    if "diffusion" in c_terms:
        r_c = r_c + ph_pp
    if "advection" in c_terms:
        r_c = r_c + sc * f * ph_p
    if "reaction" in c_terms:
        r_c = r_c - sc * kr * ph
    if "soret" in c_terms:
        r_c = r_c + sr * tf_pp

    return {"Rm": r_m, "Ref": r_ef, "Res": r_es, "Rc": r_c}
