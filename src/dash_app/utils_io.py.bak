"""
Safe I/O utilities for loading data files.

This module provides resilient data loading functions that gracefully
handle missing or malformed CSV files without crashing the application.
"""
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd


def load_csv_safe(
    path: Union[str, Path],
    parse_dates: Optional[List[str]] = None,
    expected_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Safely load a CSV file with graceful error handling.

    If the file doesn't exist or can't be read, returns an empty
    DataFrame with the expected columns (if provided).

    Parameters
    ----------
    path : str or Path
        Path to the CSV file
    parse_dates : list of str, optional
        Column names to parse as datetime
    expected_cols : list of str, optional
        Expected column names (used for empty DataFrame if file missing)

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame, or empty DataFrame with expected_cols if loading fails

    Examples
    --------
    >>> df = load_csv_safe("data.csv", parse_dates=["timestamp"])
    >>> df = load_csv_safe("missing.csv", expected_cols=["id", "value"])
    """
    p = Path(path)

    if not p.exists():
        if expected_cols:
            return pd.DataFrame(columns=expected_cols)
        return pd.DataFrame()

    try:
        df = pd.read_csv(p, parse_dates=parse_dates)

        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        if expected_cols:
            return pd.DataFrame(columns=expected_cols)
        return pd.DataFrame()


def save_csv_safe(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> None:
    """Persist a dataframe to CSV, ensuring parent directories exist."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(p, index=index)
    except Exception as exc:
        print(f"⚠️ Error saving {p}: {exc}")
