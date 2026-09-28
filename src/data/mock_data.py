"""Mock-data generation from the numerical (BVP) baseline.

Produces labelled 1000-point reference datasets per physics-parameter
combination: solves the same term-switched BVP used for validation on a
dense uniform grid and optionally adds seeded Gaussian noise to mimic
measurement data. Deterministic given (config, combo, seed).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


def combo_name(params: Dict[str, float]) -> str:
    """Build a stable dataset name from a parameter combination.

    Args:
        params: Mapping of swept physics keys to values, e.g.
            ``{"M": 1.0, "Rd": 0.5}``.

    Returns:
        Filename stem like ``mock_M1p0_Rd0p5`` (dots become ``p`` so the
        stem stays filesystem- and glob-friendly).
    """
    parts = [f"{k}{str(float(v)).replace('.', 'p').replace('-', 'm')}" for k, v in sorted(params.items())]
    return "mock_" + "_".join(parts)


def generate_mock_data(
    config: Dict[str, Any],
    coeffs: Dict[str, Any],
    n_points: int = 1000,
    noise: float = 0.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Solve the BVP baseline on a dense grid, optionally with noise.

    Args:
        config: Full config dict (domain, equations, solver blocks).
        coeffs: Flat coefficient dict for this parameter combination.
        n_points: Uniform grid resolution (default 1000).
        noise: Std-dev of Gaussian noise added to each field
            (0.0 = clean reference data).
        seed: RNG seed for noise (ignored when noise is 0).

    Returns:
        Dict with ``eta, f, fp, theta_f, theta_s, phi`` arrays of length
        ``n_points``.

    Raises:
        RuntimeError: If the underlying BVP solve fails to converge.
    """
    from src.solvers.numerical_rk45 import solve_bvp_baseline

    baseline = solve_bvp_baseline(config, coeffs, n_points=n_points)
    if noise and noise > 0.0:
        rng = np.random.default_rng(seed)
        for key in ("f", "theta_f", "theta_s", "phi"):
            baseline[key] = np.asarray(baseline[key], dtype=float) + rng.normal(
                0.0, float(noise), size=n_points
            )
    return baseline


def override_physics(
    config: Dict[str, Any], params: Dict[str, float]
) -> Dict[str, Any]:
    """Return a deep copy of config with ``physics`` keys overridden.

    Args:
        config: Base config dict (never mutated).
        params: Parameter combination to apply.

    Returns:
        New config dict with overrides applied.
    """
    cfg = copy.deepcopy(config)
    for key, value in params.items():
        cfg["physics"][key] = float(value)
    return cfg


def save_mock(
    data: Dict[str, np.ndarray], raw_dir: str | Path, stem: str
) -> Path:
    """Persist a mock dataset to ``<raw_dir>/<stem>.npz``.

    Args:
        data: Dataset dict from :func:`generate_mock_data`.
        raw_dir: Raw data directory (created if missing).
        stem: Filename stem without extension.

    Returns:
        Path of the saved file.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{stem}.npz"
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in data.items()})
    return path


def load_mock(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a mock dataset from ``.npz`` into plain numpy arrays.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        Dict of numpy arrays.
    """
    with np.load(Path(path)) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def param_combos(param_grid: Dict[str, Sequence[float]]) -> list[Dict[str, float]]:
    """Expand a parameter grid into a list of combinations.

    Args:
        param_grid: Mapping of physics key to list of values.

    Returns:
        List of ``{key: value}`` dicts (cartesian product, sorted keys).
    """
    import itertools

    keys = sorted(param_grid.keys())
    return [
        {key: float(value) for key, value in zip(keys, combo)}
        for combo in itertools.product(*[list(param_grid[key]) for key in keys])
    ]
