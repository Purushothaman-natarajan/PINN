"""CSV/Excel ingestion and supervised-loss tests."""

import numpy as np
import pytest
import torch
import yaml

from src.data.tabular import detect_format, load_tabular, save_tabular


def _sample_dataset(n: int = 50):
    eta = np.linspace(0.0, 4.0, n)
    return {
        "eta": eta,
        "f": 0.5 * (1.0 - np.exp(-2.0 * eta)),
        "theta_f": np.exp(-2.0 * eta),
        "theta_s": np.exp(-1.5 * eta),
        "phi": np.exp(-2.5 * eta),
    }


def test_detect_format():
    """Extensions map to csv/excel; unknown extensions raise."""
    assert detect_format("a.csv") == "csv"
    assert detect_format("b.XLSX") == "excel"
    with pytest.raises(ValueError, match="Unsupported"):
        detect_format("c.npz")


def test_csv_roundtrip(tmp_path):
    """Save to CSV and reload without loss."""
    data = _sample_dataset()
    path = save_tabular(data, tmp_path / "sample.csv")
    back = load_tabular(path)
    for key in ("eta", "f", "theta_f", "theta_s", "phi"):
        assert np.allclose(back[key], data[key])


def test_excel_roundtrip(tmp_path):
    """Save to Excel and reload without loss."""
    data = _sample_dataset()
    path = save_tabular(data, tmp_path / "sample.xlsx", fmt="excel")
    back = load_tabular(path)
    for key in ("eta", "f", "theta_f", "theta_s", "phi"):
        assert np.allclose(back[key], data[key])


def test_alias_map_and_missing_column(tmp_path):
    """Aliases rename file columns; missing required columns raise."""
    import pandas as pd

    data = _sample_dataset()
    frame = pd.DataFrame({"eta": data["eta"], "T_fluid": data["theta_f"]})
    path = tmp_path / "aliased.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_tabular(path)
    frame["f"] = data["f"]
    frame["theta_s"] = data["theta_s"]
    frame["phi"] = data["phi"]
    frame.to_csv(path, index=False)
    back = load_tabular(path, columns={"theta_f": "T_fluid"})
    assert np.allclose(back["theta_f"], data["theta_f"])


def _tiny_config():
    with open("configs/default_trihybrid_ltne.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["model"]["hidden_layers"] = [16, 16]
    cfg["training"]["epochs"] = 3
    return cfg


def test_supervised_weight_zero_matches_legacy():
    """weight=0 with data attached behaves like no data (zero data loss)."""
    from src.core.config_loader import resolve_device
    from src.core.fluid_properties import compute_coefficients
    from src.models.pinn_architecture import build_pinn
    from src.solvers.pinn_trainer import PINNTrainer

    cfg = _tiny_config()
    coeffs = compute_coefficients(cfg)
    device = resolve_device("cpu")
    torch.manual_seed(0)
    plain = PINNTrainer(build_pinn(cfg["model"]), cfg, coeffs, device=device)
    torch.manual_seed(0)
    with_data = PINNTrainer(
        build_pinn(cfg["model"]),
        cfg,
        coeffs,
        device=device,
        supervised_data=_sample_dataset(),
        data_weight=0.0,
    )
    assert with_data._data_loss().item() == 0.0
    assert plain._data_loss().item() == 0.0


def test_supervised_loss_pulls_to_data():
    """Positive weight produces nonzero data loss that training reduces."""
    from src.core.config_loader import resolve_device
    from src.core.fluid_properties import compute_coefficients
    from src.models.pinn_architecture import build_pinn
    from src.solvers.pinn_trainer import PINNTrainer

    cfg = _tiny_config()
    cfg["training"]["epochs"] = 30
    coeffs = compute_coefficients(cfg)
    device = resolve_device("cpu")
    torch.manual_seed(1)
    trainer = PINNTrainer(
        build_pinn(cfg["model"]),
        cfg,
        coeffs,
        device=device,
        supervised_data=_sample_dataset(),
        data_weight=10.0,
    )
    before = trainer._data_loss().item()
    assert before > 0.0
    trainer.train()
    after = trainer._data_loss().item()
    assert after < before
    assert trainer.history["data"] and trainer.history["data"][-1] <= before
