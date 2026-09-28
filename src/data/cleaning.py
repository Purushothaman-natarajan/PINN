"""Deterministic cleaning for mock datasets.

Pipeline (logged per dataset): drop non-finite rows -> sort by eta ->
dedupe eta within tolerance -> optional physical clipping -> optional
light smoothing -> schema check (required keys, matching lengths,
monotone eta). Every action is counted in the returned report, which is
persisted next to the cleaned data for auditability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


REQUIRED_KEYS = ("eta", "f", "theta_f", "theta_s", "phi")

# Permissive physical bounds applied only when clipping is enabled.
# f may exceed 1 slightly (entrainment); temperatures/concentration
# live in [0, 1] for the Dirichlet problem class.
DEFAULT_BOUNDS: Dict[str, tuple[float, float]] = {
    "f": (-0.5, 3.0),
    "theta_f": (0.0, 1.5),
    "theta_s": (0.0, 1.5),
    "phi": (0.0, 1.5),
}


def clean_mock(
    data: Dict[str, np.ndarray],
    clip: bool = False,
    smooth_window: int = 0,
    eta_tol: float = 1e-12,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Clean a mock dataset deterministically.

    Args:
        data: Raw dataset dict with at least ``REQUIRED_KEYS``.
        clip: Clip fields to ``DEFAULT_BOUNDS`` when True.
        smooth_window: Odd moving-average window applied to fields
            (0 or 1 disables smoothing).
        eta_tol: Minimum eta spacing; closer points are deduped.

    Returns:
        Tuple of (cleaned dict, report dict). The report records
        ``n_in``, ``n_out``, ``n_dropped_nonfinite``,
        ``n_deduped``, ``n_clipped`` and ``warnings``.

    Raises:
        ValueError: If required keys are missing, lengths mismatch, or no
            valid rows remain.
    """
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Mock data missing keys: {missing}")
    arrays = {key: np.asarray(data[key], dtype=float).reshape(-1) for key in data}
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Mock data length mismatch: {sorted(lengths)}")
    n_in = len(arrays["eta"])
    report: Dict[str, Any] = {"n_in": int(n_in), "warnings": []}

    # 1. Drop rows with any non-finite value in required fields.
    mask = np.ones(n_in, dtype=bool)
    for key in REQUIRED_KEYS:
        mask &= np.isfinite(arrays[key])
    n_dropped = int(n_in - np.count_nonzero(mask))
    report["n_dropped_nonfinite"] = n_dropped
    if n_dropped:
        report["warnings"].append(f"dropped {n_dropped} non-finite rows")
    arrays = {key: values[mask] for key, values in arrays.items()}
    if len(arrays["eta"]) == 0:
        raise ValueError("No valid rows remain after non-finite filtering")

    # 2. Sort by eta (BVP grids are ascending, noise never reorders, but be safe).
    order = np.argsort(arrays["eta"], kind="stable")
    arrays = {key: values[order] for key, values in arrays.items()}

    # 3. Dedupe eta within tolerance (keep first occurrence).
    eta = arrays["eta"]
    keep = np.ones(len(eta), dtype=bool)
    keep[1:] = np.diff(eta) > eta_tol
    n_deduped = int(len(eta) - np.count_nonzero(keep))
    report["n_deduped"] = n_deduped
    if n_deduped:
        report["warnings"].append(f"deduped {n_deduped} eta points")
    arrays = {key: values[keep] for key, values in arrays.items()}

    # 4. Optional physical clipping (fields only, never eta).
    n_clipped = 0
    if clip:
        for key, (lo, hi) in DEFAULT_BOUNDS.items():
            if key in arrays:
                before = arrays[key]
                arrays[key] = np.clip(before, lo, hi)
                n_clipped += int(np.count_nonzero((before < lo) | (before > hi)))
    report["n_clipped"] = n_clipped
    if n_clipped:
        report["warnings"].append(f"clipped {n_clipped} field values")

    # 5. Optional light smoothing (centered moving average, edges preserved).
    if smooth_window and smooth_window > 1:
        if smooth_window % 2 == 0:
            raise ValueError("smooth_window must be odd")
        kernel = np.ones(smooth_window) / smooth_window
        for key in ("f", "theta_f", "theta_s", "phi"):
            if key in arrays:
                padded = np.pad(arrays[key], smooth_window // 2, mode="edge")
                arrays[key] = np.convolve(padded, kernel, mode="valid")
        report["warnings"].append(f"applied smoothing window {smooth_window}")
    report["smoothing"] = int(smooth_window or 0)

    # 6. Final schema check: monotone eta.
    if not bool(np.all(np.diff(arrays["eta"]) > 0)):
        raise ValueError("Cleaned eta grid is not strictly increasing")
    report["n_out"] = int(len(arrays["eta"]))
    return arrays, report


def save_clean_mock(
    clean: Dict[str, np.ndarray],
    report: Dict[str, Any],
    processed_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """Persist cleaned data and its cleaning report.

    Args:
        clean: Cleaned dataset dict from :func:`clean_mock`.
        report: Cleaning report dict from :func:`clean_mock`.
        processed_dir: Processed data directory (created if missing).
        stem: Filename stem without extension.

    Returns:
        Tuple of (data path, report path).
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_path = processed_dir / f"{stem}.npz"
    report_path = processed_dir / f"{stem}_cleaning_report.json"
    np.savez_compressed(data_path, **{k: np.asarray(v) for k, v in clean.items()})
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return data_path, report_path
