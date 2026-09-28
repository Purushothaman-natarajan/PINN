"""Explainability tests: LIME spatial + parameter sensitivity smoke tests."""

import numpy as np
import yaml

from src.analysis.lime_explainer import parameter_lime, save_lime_summary, spatial_lime


def _tiny_trainer():
    from src.core.config_loader import load_config, resolve_device
    from src.core.fluid_properties import compute_coefficients
    from src.models.pinn_architecture import build_pinn
    from src.solvers.pinn_trainer import PINNTrainer

    config = load_config("configs/default_trihybrid_ltne.yaml")
    config["model"]["hidden_layers"] = [16, 16]
    coeffs = compute_coefficients(config)
    model = build_pinn(config["model"])
    trainer = PINNTrainer(model, config, coeffs, device=resolve_device("cpu"))
    return trainer, config


def test_spatial_lime_shapes():
    """Spatial LIME returns one (weight, intercept) pair per output."""
    trainer, _ = _tiny_trainer()
    res = spatial_lime(trainer, [0.0, 2.0, 4.0], num_samples=200)
    assert res["method"].startswith("lime.") or res["method"] == "input_gradient_fallback"
    assert len(res["explanations"]) == 3
    for per_out in res["explanations"]:
        assert len(per_out) == 4  # f, theta_f, theta_s, phi
        for weight, intercept in per_out:
            assert np.isfinite(weight) and np.isfinite(intercept)


def test_parameter_lime_ranking():
    """Parameter LIME ranks every swept parameter once."""
    with open("configs/default_trihybrid_ltne.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    res = parameter_lime(config, {"M": [0.0, 1.0], "Rd": [0.0, 0.5]}, num_samples=200)
    assert res["X"].shape[1] == 2
    assert res["Y"].shape[1] == 5
    ranked = [n for n, _ in res["ranking"]]
    assert sorted(ranked) == ["M", "Rd"]


def test_save_lime_summary(tmp_path):
    """LIME artefacts persist to disk with importance plots."""
    trainer, _ = _tiny_trainer()
    res = spatial_lime(trainer, [1.0, 3.0], num_samples=200)
    path = save_lime_summary(res, tmp_path)
    assert path.is_file()
