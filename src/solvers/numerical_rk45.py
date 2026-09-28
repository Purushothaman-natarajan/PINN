"""Traditional numerical baseline via scipy ``solve_bvp`` (RK-based collocation).

Mirrors the config-driven PDE terms from
:mod:`src.physics.governing_equations` as a first-order system so PINN
predictions can be validated against a mesh-based solver.

State vector (9 components):
    y = [f, fp, fpp, tf, tfp, ts, tsp, ph, php]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.integrate import solve_bvp


def _first_order_rhs(
    eta: np.ndarray,
    y: np.ndarray,
    coeffs: Dict[str, float],
    equations: Dict[str, Any],
) -> np.ndarray:
    """Right-hand side of the first-order BVP system.

    Args:
        eta: Mesh points (M,).
        y: State matrix (9, M).
        coeffs: Flat coefficient dict.
        equations: Equation-selection block.

    Returns:
        Derivatives dy/deta with shape (9, M).
    """
    f, fp, fpp, tf, tfp, ts, tsp, ph, php = y
    m_terms = set(equations.get("momentum_terms", []))
    e_terms = set(equations.get("fluid_energy_terms", []))
    s_terms = set(equations.get("solid_energy_terms", []))
    c_terms = set(equations.get("concentration_terms", []))

    a1 = float(coeffs.get("A1", 1.0))
    a2 = float(coeffs.get("A2", 1.0))
    m_mag = float(coeffs.get("M", 0.0))
    kp = float(coeffs.get("Kp", 0.0))
    fr = float(coeffs.get("Fr", 0.0))
    gamma = float(coeffs.get("gamma", 0.0))
    inv_a1 = 1.0 / a1 if a1 != 0 else 1.0
    curv = 1.0 + 2.0 * gamma * eta

    # Momentum: solve for fppp from R_m = 0.
    fppp = np.zeros_like(eta, dtype=float)
    if "viscous" in m_terms and "curvature" in m_terms:
        fppp = fppp - 2.0 * gamma * fpp * inv_a1 / np.maximum(curv, 1e-12)
    elif "viscous" not in m_terms:
        pass  # fppp still assembled from remaining terms below
    # General rearrangement: fppp*inv_a1*(curv if enabled) + rest = 0.
    denom = inv_a1 * (curv if ("viscous" in m_terms and "curvature" in m_terms) else 1.0)
    denom = np.maximum(np.abs(denom), 1e-12) * np.sign(denom + 1e-16)
    rest = np.zeros_like(eta, dtype=float)
    if "viscous" in m_terms and "curvature" in m_terms:
        rest = rest  # curvature viscous part moved to denom side already
    if "inertia" in m_terms:
        rest = rest + (f * fpp - fp**2)
    if "magnetic" in m_terms:
        rest = rest - a2 * m_mag * fp
    if "darcy" in m_terms:
        rest = rest - kp * fp
    if "forchheimer" in m_terms:
        rest = rest - fr * fp**2
    if "curvature" in m_terms and "viscous" in m_terms:
        # fppp*inv_a1*curv + 2g*fpp*inv_a1 + rest = 0
        rest = rest + 2.0 * gamma * fpp * inv_a1
        fppp = -rest / np.maximum(curv * inv_a1, 1e-12)
    else:
        # fppp*inv_a1 + rest = 0  (or fppp alone if viscous disabled)
        scale = inv_a1 if "viscous" in m_terms else 1.0
        fppp = -rest / max(abs(scale), 1e-12)

    # Fluid energy: (k+ Rd) tfpp + Pr*rcp*f*tfp + Hs(ts-tf) + ... = 0.
    k_eff = float(coeffs.get("k_ratio", 1.0)) + float(coeffs.get("Rd", 0.0))
    k_eff = max(abs(k_eff), 1e-12)
    pr = float(coeffs.get("Pr", 6.2))
    rcp = float(coeffs.get("rho_cp_ratio", 1.0))
    h_s = float(coeffs.get("Hs", 1.0))
    ec = float(coeffs.get("Ec", 0.0))
    sig_r = float(coeffs.get("sigma_ratio", 1.0))
    q_src = float(coeffs.get("Q", 0.0))
    tf_rhs = np.zeros_like(eta, dtype=float)
    if "advection" in e_terms:
        tf_rhs = tf_rhs + pr * rcp * f * tfp
    if "interphase" in e_terms:
        tf_rhs = tf_rhs + h_s * (ts - tf)
    if "joule" in e_terms:
        tf_rhs = tf_rhs + ec * m_mag * sig_r * fp**2
    if "heat_source" in e_terms:
        tf_rhs = tf_rhs + q_src * tf
    tfpp = -tf_rhs / k_eff if ("conduction" in e_terms or "radiation" in e_terms) else -tf_rhs

    # Solid energy: R_es = ts'' + Hsg*(tf - ts) = 0  ->  ts'' = Hsg*(ts - tf).
    # Must match src/physics/governing_equations.py (no extra negation:
    # the term below is already the rearranged right-hand side).
    hsg = float(coeffs.get("Hsg", 0.0))
    if "conduction" in s_terms:
        if "interphase" in s_terms:
            tspp = hsg * (ts - tf)
        else:
            tspp = np.zeros_like(eta, dtype=float)
    else:
        tspp = np.zeros_like(eta, dtype=float)

    # Concentration: phpp + Sc f php - Sc Kr ph + Sr tfpp = 0.
    sc = float(coeffs.get("Sc", 0.6))
    kr = float(coeffs.get("Kr", 0.0))
    sr = float(coeffs.get("Sr", 0.0))
    c_rhs = np.zeros_like(eta, dtype=float)
    if "advection" in c_terms:
        c_rhs = c_rhs + sc * f * php
    if "reaction" in c_terms:
        c_rhs = c_rhs - sc * kr * ph
    if "soret" in c_terms:
        c_rhs = c_rhs + sr * tfpp
    phpp = -c_rhs if "diffusion" in c_terms else -c_rhs

    dydx = np.vstack([fp, fpp, fppp, tfp, tfpp, tsp, tspp, php, phpp])
    return dydx


def _bc_fun(
    ya: np.ndarray,
    yb: np.ndarray,
    coeffs: Dict[str, float],
) -> np.ndarray:
    """Boundary residuals: 9 conditions for the 9-state system.

    Mirrors :mod:`src.physics.boundary_conditions`: Dirichlet
    ``theta_f(0) = 1`` by default, Robin ``theta_f'(0) + Bi_f*(theta_f(0)-1)``
    when ``Bi_f`` is set.
    """
    slip = float(coeffs.get("slip", 0.0))
    omega = float(coeffs.get("omega", 0.0))
    bi_f = coeffs.get("Bi_f", None)
    if bi_f is None:
        tf_inner = ya[3] - 1.0  # theta_f(0) = 1
    else:
        tf_inner = ya[4] + float(bi_f) * (ya[3] - 1.0)  # Robin
    return np.array(
        [
            ya[0] - 0.0,  # f(0) = 0
            ya[1] - (1.0 + slip * ya[2]),  # f'(0) = 1 + slip f''(0)
            tf_inner,
            ya[5] - 1.0,  # theta_s(0) = 1
            ya[7] - 1.0,  # phi(0) = 1
            yb[1] - omega,  # f'(eta0) = omega
            yb[3] - 0.0,  # theta_f(eta0) = 0
            yb[5] - 0.0,  # theta_s(eta0) = 0
            yb[7] - 0.0,  # phi(eta0) = 0
        ]
    )


def _initial_guess(
    eta: np.ndarray, coeffs: Dict[str, float]
) -> np.ndarray:
    """Physically sensible initial profiles for the BVP solver."""
    eta0 = float(coeffs.get("eta0", 4.0))
    s = np.clip(eta / max(eta0, 1e-12), 0.0, 1.0)
    omega = float(coeffs.get("omega", 0.0))
    # Blending from stretching (fp=1) to outer rotation (fp=omega).
    fp = (1.0 - s) * 1.0 + s * omega
    f = eta * (1.0 - 0.5 * s * (1.0 - omega))
    fpp = np.full_like(eta, -0.5)
    tf = 1.0 - s
    tfp = np.full_like(eta, -1.0 / max(eta0, 1e-12))
    ts = 1.0 - s
    tsp = np.full_like(eta, -1.0 / max(eta0, 1e-12))
    ph = 1.0 - s
    php = np.full_like(eta, -1.0 / max(eta0, 1e-12))
    return np.vstack([f, fp, fpp, tf, tfp, ts, tsp, ph, php])


def solve_bvp_baseline(
    config: Dict[str, Any],
    coeffs: Dict[str, float],
    n_points: int = 400,
) -> Dict[str, np.ndarray]:
    """Solve the baseline BVP with scipy ``solve_bvp``.

    Args:
        config: Full config dict (domain, solver, equations blocks).
        coeffs: Flat coefficient dict.
        n_points: Output resolution for profiles.

    Returns:
        Dict with eta, f, fp, theta_f, theta_s, phi arrays.

    Raises:
        RuntimeError: If the BVP solver fails to converge.
    """
    equations = config["equations"]
    eta0 = float(config["domain"].get("eta0", 4.0))
    solver_cfg = config.get("solver", {})
    n_mesh = int(solver_cfg.get("bvp_mesh", 200))
    tol = float(solver_cfg.get("tol", 1e-8))
    max_nodes = int(solver_cfg.get("max_nodes", 2000))

    x_init = np.linspace(0.0, eta0, n_mesh)
    y_init = _initial_guess(x_init, coeffs)

    sol = solve_bvp(
        lambda x, y: _first_order_rhs(x, y, coeffs, equations),
        lambda ya, yb: _bc_fun(ya, yb, coeffs),
        x_init,
        y_init,
        tol=tol,
        max_nodes=max_nodes,
        verbose=0,
    )
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")
    eta = np.linspace(0.0, eta0, n_points)
    y = sol.sol(eta)
    return {
        "eta": eta,
        "f": y[0],
        "fp": y[1],
        "fpp": y[2],
        "theta_f": y[3],
        "theta_s": y[5],
        "phi": y[7],
    }


def save_baseline(
    baseline: Dict[str, np.ndarray], config: Dict[str, Any]
) -> Path:
    """Save baseline profiles to ``data/raw/baseline.npz``.

    Args:
        baseline: Output of :func:`solve_bvp_baseline`.
        config: Full config dict with outputs block.

    Returns:
        Path of the saved file.
    """
    out_dir = Path(config.get("outputs", {}).get("raw_dir", "data/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = config.get("outputs", {}).get("baseline_file", "baseline.npz")
    path = out_dir / fname
    np.savez_compressed(path, **baseline)
    return path
