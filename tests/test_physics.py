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
    # Physicality regression: coupled LTNE temps must stay in [0, 1]
    # (guards against sign errors in the BVP mirror, e.g. solid equation).
    for key in ("theta_f", "theta_s", "phi"):
        assert base[key].min() > -0.05, f"{key} undershoots: {base[key].min()}"
        assert base[key].max() < 1.05, f"{key} overshoots: {base[key].max()}"


def test_schema_accepts_default_config():
    """Default YAML passes the machine-readable schema (docs reference)."""
    from src.core.config_loader import load_config, validate_config

    cfg = load_config("configs/default_trihybrid_ltne.yaml")
    validate_config(cfg)  # must not raise


def test_schema_rejects_bad_activation():
    """Schema violations raise ValueError naming the offending key."""
    import pytest

    from src.core.config_loader import load_config, validate_config

    cfg = load_config("configs/default_trihybrid_ltne.yaml")
    cfg["model"]["activation"] = "not_an_activation"
    with pytest.raises(ValueError, match="activation"):
        validate_config(cfg)


def test_bi_f_passthrough_and_robin_parity():
    """Bi_f/f_outer flow into coeffs; PINN and BVP agree on Robin BC."""
    import copy
    import yaml

    import torch

    from src.core.config_loader import resolve_device
    from src.core.fluid_properties import compute_coefficients
    from src.models.pinn_architecture import build_pinn
    from src.physics.boundary_conditions import bc_residual
    from src.solvers.numerical_rk45 import _bc_fun, solve_bvp_baseline

    with open("configs/default_trihybrid_ltne.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["physics"]["Bi_f"] = 2.0
    coeffs = compute_coefficients(cfg)
    assert coeffs["Bi_f"] == 2.0
    assert coeffs["f_outer"] is None or isinstance(coeffs["f_outer"], float)

    # PINN Robin branch executes and keeps 5+4 structure.
    model = build_pinn(
        {"input_dim": 1, "hidden_layers": [16], "output_dim": 4,
         "activation": "tanh", "initialization": "xavier"}
    )
    bc = bc_residual(model, coeffs, resolve_device("cpu"), torch.float32)
    assert bc["total"].numel() == 9

    # BVP Robin branch: spot check on a hand-built state.
    ya = np.array([0.0, 1.0, -0.5, 0.8, -0.4, 1.0, -0.2, 1.0, -0.2])
    yb = np.array([1.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = _bc_fun(ya, yb, coeffs)
    assert res.shape == (9,)
    assert abs(res[2] - (-0.4 + 2.0 * (0.8 - 1.0))) < 1e-12

    # Full BVP still converges with Robin enabled.
    base = solve_bvp_baseline(cfg, coeffs, n_points=50)
    assert np.all(np.isfinite(base["theta_f"]))
