"""Tabular data ingestion and export (CSV / Excel) for supervised training.

Bridges external measurement files and the internal dataset dict format
(``eta, f, theta_f, theta_s, phi`` + optional ``fp``) consumed by
:mod:`src.data.cleaning`, validation and plotting. Files with different
column names are handled through an alias map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

REQUIRED_COLUMNS = ("eta", "f", "theta_f", "theta_s", "phi")
OPTIONAL_COLUMNS = ("fp",)

TABLE_EXTENSIONS = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
}


def detect_format(path: str | Path) -> str:
    """Detect tabular format from the file extension.

    Args:
        path: Input file path.

    Returns:
        ``"csv"`` or ``"excel"``.

    Raises:
        ValueError: For unsupported extensions.
    """
    ext = Path(path).suffix.lower()
    if ext not in TABLE_EXTENSIONS:
        raise ValueError(
            f"Unsupported tabular extension '{ext}' for {path}; "
            f"expected one of {sorted(TABLE_EXTENSIONS)}"
        )
    return TABLE_EXTENSIONS[ext]


def load_tabular(
    path: str | Path,
    columns: Mapping[str, str] | None = None,
    sheet: int | str = 0,
    fmt: str | None = None,
) -> Dict[str, np.ndarray]:
    """Load a CSV/Excel measurement file into a dataset dict.

    Args:
        path: Path to ``.csv``, ``.xlsx`` or ``.xls`` file.
        columns: Optional alias map from canonical names to file column
            names, e.g. ``{"theta_f": "T_fluid"}``. Unmapped canonical
            names are looked up verbatim.
        sheet: Excel sheet index or name (ignored for CSV).
        fmt: Force ``"csv"`` or ``"excel"``; auto-detected when None.

    Returns:
        Dict of float arrays keyed by canonical column names. Optional
        columns (``fp``) are included only when present in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing.
    """
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tabular data file not found: {path}")
    fmt = fmt or detect_format(path)
    if fmt == "csv":
        frame = pd.read_csv(path)
    elif fmt == "excel":
        frame = pd.read_excel(path, sheet_name=sheet)
    else:  # pragma: no cover - guarded by detect_format
        raise ValueError(f"Unknown tabular format '{fmt}'")

    aliases = dict(columns or {})
    data: Dict[str, np.ndarray] = {}
    missing = []
    for name in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        col = aliases.get(name, name)
        if col not in frame.columns:
            if name in REQUIRED_COLUMNS:
                missing.append(f"{name} (looked for column '{col}')")
            continue
        data[name] = np.asarray(frame[col], dtype=float).reshape(-1)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Available: {list(frame.columns)}. "
            "Use the data.columns alias map to rename file columns."
        )
    return data


def save_tabular(
    data: Dict[str, np.ndarray],
    path: str | Path,
    fmt: str | None = None,
    sheet: str = "profiles",
) -> Path:
    """Export a dataset dict to CSV or Excel.

    Args:
        data: Dataset dict (e.g. from BVP baseline or mock generation).
        path: Output path; extension selects the format when ``fmt`` is None.
        fmt: Force ``"csv"`` or ``"excel"``.
        sheet: Excel sheet name.

    Returns:
        Path of the written file.
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or detect_format(path)
    order = [k for k in ("eta", "f", "fp", "theta_f", "theta_s", "phi") if k in data]
    frame = pd.DataFrame({k: np.asarray(data[k]).reshape(-1) for k in order})
    if fmt == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, sheet_name=sheet, index=False)
    return path


def load_and_clean(
    path: str | Path,
    columns: Mapping[str, str] | None = None,
    sheet: int | str = 0,
    clip: bool = False,
    smooth_window: int = 0,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load a tabular file and run it through the standard cleaning.

    Args:
        path: CSV/Excel file path.
        columns: Alias map (see :func:`load_tabular`).
        sheet: Excel sheet index or name.
        clip: Clip fields to physical bounds.
        smooth_window: Odd moving-average window (0/1 disables).

    Returns:
        Tuple of (cleaned dict, cleaning report).
    """
    from src.data.cleaning import clean_mock

    raw = load_tabular(path, columns=columns, sheet=sheet)
    return clean_mock(raw, clip=clip, smooth_window=smooth_window)
