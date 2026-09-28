"""YAML configuration loading with validation and device resolution.

All fluid parameters, layer sizes, and training hyper-parameters must come
from YAML files. This module is the single entry point for configs.

Validation is two-stage: required top-level blocks are always checked, then
``configs/schema.json`` is enforced when the ``jsonschema`` package is
available (it is a declared dependency). See ``docs/configuration.md``.
"""

from __future__ import annotations

import json
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

_SCHEMA_CACHE: Dict[str, Any] | None = None


def _load_schema() -> Dict[str, Any] | None:
    """Load ``configs/schema.json`` located next to the ``configs`` dir.

    Returns ``None`` when the schema file or ``jsonschema`` is unavailable,
    in which case only the structural checks below apply.
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    try:
        schema_path = Path(__file__).resolve().parents[2] / "configs" / "schema.json"
        with schema_path.open("r", encoding="utf-8") as fh:
            _SCHEMA_CACHE = json.load(fh)
    except (OSError, json.JSONDecodeError):
        _SCHEMA_CACHE = None
    return _SCHEMA_CACHE


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate a config dict against ``configs/schema.json``.

    Args:
        cfg: Configuration dictionary (as returned by YAML parsing).

    Raises:
        ValueError: If the config violates the schema, with the offending
            key path and constraint in the message.
    """
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        return  # structural checks in load_config still apply
    schema = _load_schema()
    if schema is None:
        return
    try:
        jsonschema.validate(cfg, schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise ValueError(f"Invalid config at '{location}': {exc.message}") from exc


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required top-level keys are missing or any value
            violates ``configs/schema.json``.
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
    validate_config(cfg)
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
