"""Mock-data generation and cleaning tests."""

import numpy as np
import pytest
import yaml

from src.data.cleaning import clean_mock
from src.data.mock_data import (
    combo_name,
    generate_mock_data,
    override_physics,
    param_combos,
)
from src.core.fluid_properties import compute_coefficients


def _base_config():
    with open("configs/mock_train_1000.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_combo_name_stable():
    """Combo names are deterministic and filesystem-safe."""
    assert combo_name({"M": 1.0, "Rd": 0.5}) == combo_name({"Rd": 0.5, "M": 1.0})
    assert "." not in combo_name({"M": 1.0})
    assert combo_name({"M": 1.0}) == "mock_M1p0"


def test_param_combos_cartesian():
    """Grid expansion yields the full cartesian product."""
    combos = param_combos({"M": [0.0, 1.0], "Rd": [0.0, 0.5]})
    assert len(combos) == 4
    assert {"M": 1.0, "Rd": 0.5} in combos


def test_override_physics_no_mutation():
    """Overrides apply to a copy; the base config is untouched."""
    cfg = _base_config()
    before = cfg["physics"]["M"]
    cfg_c = override_physics(cfg, {"M": 9.0})
    assert cfg_c["physics"]["M"] == 9.0
    assert cfg["physics"]["M"] == before


def test_generate_mock_data_shapes():
    """Generated mock data has 1000 finite points and BC endpoints."""
    cfg = _base_config()
    coeffs = compute_coefficients(cfg)
    data = generate_mock_data(cfg, coeffs, n_points=1000)
    for key in ("eta", "f", "theta_f", "theta_s", "phi"):
        assert key in data
        assert len(data[key]) == 1000
        assert np.all(np.isfinite(data[key]))
    assert abs(data["theta_f"][0] - 1.0) < 1e-3
    assert abs(data["theta_f"][-1]) < 1e-3


def test_clean_mock_drops_and_sorts():
    """Cleaning drops NaNs, sorts eta, and reports counts."""
    eta = np.array([3.0, 1.0, 2.0, 0.0, 1.0])
    fields = {
        "eta": eta,
        "f": np.array([0.5, 0.2, np.nan, 0.0, 0.2]),
        "theta_f": np.array([0.2, 0.6, 0.4, 1.0, 0.6]),
        "theta_s": np.array([0.2, 0.6, 0.4, 1.0, 0.6]),
        "phi": np.array([0.2, 0.6, 0.4, 1.0, 0.6]),
    }
    clean, report = clean_mock(fields)
    assert report["n_in"] == 5
    assert report["n_dropped_nonfinite"] == 1
    assert report["n_deduped"] == 1
    assert report["n_out"] == 3
    assert bool(np.all(np.diff(clean["eta"]) > 0))


def test_clean_mock_rejects_missing_keys():
    """Cleaning raises on missing required fields."""
    with pytest.raises(ValueError, match="missing keys"):
        clean_mock({"eta": np.array([0.0, 1.0])})
