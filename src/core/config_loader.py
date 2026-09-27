"""YAML configuration loading with validation and device resolution.

All fluid parameters, layer sizes, and training hyper-parameters must come
from YAML files. This module is the single entry point for configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import yaml


REQUIRED_TOP_KEYS = (
    "domain",
    "physics",
    "nanofluid",
    "equations",
    "model",
    "training",
    "solver",
)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required top-level keys are missing.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}
    missing = [k for k in REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config {cfg_path} missing keys: {missing}")
    _validate_model(cfg["model"])
    _validate_equations(cfg["equations"])
    return cfg


def _validate_model(model: Dict[str, Any]) -> None:
    """Validate the model block of the config."""
    if "hidden_layers" not in model:
        raise ValueError("model.hidden_layers is required, e.g. [128, 128, 128]")
    layers = model["hidden_layers"]
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("model.hidden_layers must be a non-empty list")
    if any((not isinstance(n, int) or n <= 0) for n in layers):
        raise ValueError("model.hidden_layers entries must be positive ints")


def _validate_equations(equations: Dict[str, Any]) -> None:
    """Validate the equation-selection block."""
    expected = (
        "momentum_terms",
        "fluid_energy_terms",
        "solid_energy_terms",
        "concentration_terms",
    )
    for key in expected:
        if key not in equations:
            raise ValueError(f"equations.{key} is required in config")
        if not isinstance(equations[key], list):
            raise ValueError(f"equations.{key} must be a list of term names")


def resolve_device(name: str | None = "auto") -> torch.device:
    """Resolve a config device string to a torch device.

    Args:
        name: One of ``auto``, ``cpu``, ``cuda``.

    Returns:
        Resolved torch device.
    """
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
