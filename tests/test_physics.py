"""Physics unit tests: residuals vanish for manufactured states."""

import numpy as np
import torch

from src.physics.governing_equations import pde_residuals


def _manifolds():
    # Linear profiles satisfying a reduced equation set (no source terms).
    n = 16
    eta = torch.linspace(0, 1, n).reshape(-1, 1)
    f = torch.zeros_like(eta)
    tf = 1.0 - eta
    ts = 1.0 - eta
    ph = 1.0 - eta
    comps = {"f": f, "theta_f": tf, "theta_s": ts, "phi": ph}
    derivs = {
        "fp": torch.zeros_like(eta),
        "fpp": torch.zeros_like(eta),
        "fppp": torch.zeros_like(eta),
        "tfp": -torch.ones_like(eta),
        "tfpp": torch.zeros_like(eta),
        "tsp": -torch.ones_like(eta),
        "tspp": torch.zeros_like(eta),
        "php": -torch.ones_like(eta),
        "phpp": torch.zeros_like(eta),
    }
    return eta, comps, derivs


def test_momentum_zero_flow_residual():
    """Zero flow with all momentum sources disabled gives ~0 residual."""
    eta, comps, derivs = _manifolds()
    coeffs = {"A1": 1.0, "A2": 0.0, "M": 0.0, "Kp": 0.0, "Fr": 0.0, "gamma": 0.0}
    equations = {
        "momentum_terms": ["viscous", "inertia"],
        "fluid_energy_terms": [],
        "solid_energy_terms": [],
        "concentration_terms": [],
    }
    res = pde_residuals(eta, comps, derivs, coeffs, equations)
    assert torch.mean(res["Rm"] ** 2).item() < 1e-12


def test_ltne_equilibrium_residual():
    """theta_f == theta_s with no sources gives ~0 LTNE residuals."""
    eta, comps, derivs = _manifolds()
    coeffs = {
        "k_ratio": 1.0,
        "Rd": 0.0,
        "Pr": 6.2,
        "rho_cp_ratio": 1.0,
        "Hs": 2.0,
        "Hsg": 2.0,
        "Ec": 0.0,
        "M": 0.0,
        "sigma_ratio": 1.0,
        "Q": 0.0,
    }
    equations = {
        "momentum_terms": [],
        "fluid_energy_terms": ["conduction", "interphase"],
        "solid_energy_terms": ["conduction", "interphase"],
        "concentration_terms": [],
    }
    res = pde_residuals(eta, comps, derivs, coeffs, equations)
    assert torch.mean(res["Ref"] ** 2).item() < 1e-12
    assert torch.mean(res["Res"] ** 2).item() < 1e-12


def test_bc_inner_outer_shapes():
    """BC residual concatenates 5 inner + 4 outer conditions."""
    from src.core.config_loader import resolve_device
    from src.models.pinn_architecture import build_pinn

    model = build_pinn(
        {"input_dim": 1, "hidden_layers": [16, 16], "output_dim": 4,
         "activation": "tanh", "initialization": "xavier"}
    )
    from src.physics.boundary_conditions import bc_residual

    coeffs = {"eta0": 4.0, "slip": 0.0, "omega": 0.0}
    bc = bc_residual(model, coeffs, resolve_device("cpu"), torch.float32)
    assert bc["inner"].numel() == 5
    assert bc["outer"].numel() == 4
    assert bc["total"].numel() == 9


def test_bvp_baseline_runs():
    """solve_bvp baseline converges on default config physics."""
    import yaml

    from src.core.fluid_properties import compute_coefficients
    from src.solvers.numerical_rk45 import solve_bvp_baseline

    with open("configs/default_trihybrid_ltne.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    coeffs = compute_coefficients(cfg)
    base = solve_bvp_baseline(cfg, coeffs, n_points=50)
    for k in ("f", "theta_f", "theta_s", "phi"):
        assert k in base
        assert np.all(np.isfinite(base[k]))
    # Dirichlet ends.
    assert abs(base["theta_f"][0] - 1.0) < 1e-3
    assert abs(base["theta_f"][-1]) < 1e-3
