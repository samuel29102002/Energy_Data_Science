from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

PV_COLUMN_KEYWORDS = ("pv", "photovoltaic")
DEMAND_COLUMN_KEYWORDS = ("demand",)
TIMESTAMP_COLUMN = "timestamp"


@dataclass(frozen=True)
class CleaningResult:
    cleaned: pd.DataFrame
    log: List[Dict[str, object]]
    dropped_rows: int


def _select_columns(columns: Iterable[str], keywords: Tuple[str, ...]) -> List[str]:
    lower_map = {col.lower(): col for col in columns}
    selected: List[str] = []
    for key in keywords:
        for lower, original in lower_map.items():
            if key in lower:
                selected.append(original)
    return list(dict.fromkeys(selected))


def clean_dataset(df: pd.DataFrame) -> CleaningResult:
    if df.empty:
        return CleaningResult(df.copy(), [], 0)

    work = df.copy()
    log: List[Dict[str, object]] = []

    if TIMESTAMP_COLUMN in work.columns:
        work[TIMESTAMP_COLUMN] = pd.to_datetime(work[TIMESTAMP_COLUMN], errors="coerce")
        invalid_ts = work[TIMESTAMP_COLUMN].isna().sum()
        if invalid_ts:
            log.append({"step": "drop invalid timestamps", "count": int(invalid_ts)})
            work = work.dropna(subset=[TIMESTAMP_COLUMN])
        work = work.sort_values(TIMESTAMP_COLUMN)
        work = work.reset_index(drop=True)
        work = work.set_index(TIMESTAMP_COLUMN, drop=False)
    elif isinstance(work.index, pd.DatetimeIndex):
        work = work.sort_index()
        log.append({"step": "used existing datetime index", "count": int(work.shape[0])})
    else:
        log.append({"step": "timestamp missing", "count": 0})

    numeric_cols = work.select_dtypes(include=["number"]).columns.tolist()
    non_numeric = [c for c in work.columns if c not in numeric_cols and c != TIMESTAMP_COLUMN]
    for col in non_numeric:
        coerced = pd.to_numeric(work[col], errors="coerce")
        if coerced.notna().sum() > 0:
            numeric_cols.append(col)
            work[col] = coerced
    if numeric_cols:
        log.append({"step": "coerced numeric columns", "columns": numeric_cols})

    missing_counts = work.isna().sum()
    if missing_counts.any():
        log.append({"step": "missing counts", "detail": missing_counts[missing_counts > 0].to_dict()})

    pv_cols = _select_columns(work.columns, PV_COLUMN_KEYWORDS)
    demand_cols = _select_columns(work.columns, DEMAND_COLUMN_KEYWORDS)

    if isinstance(work.index, pd.DatetimeIndex):
        if pv_cols:
            total_interpolated = 0
            for col in pv_cols:
                before = work[col].isna().sum()
                work[col] = work[col].interpolate(method="time", limit=4, limit_direction="both")
                after = work[col].isna().sum()
                total_interpolated += max(before - after, 0)
            log.append({"step": "interpolated pv", "count": int(total_interpolated)})
        if demand_cols:
            total_ffill = 0
            for col in demand_cols:
                before = work[col].isna().sum()
                work[col] = work[col].fillna(method="ffill", limit=3)
                after = work[col].isna().sum()
                total_ffill += max(before - after, 0)
            log.append({"step": "forward-filled demand", "count": int(total_ffill)})
    else:
        if pv_cols:
            log.append({"step": "skipped pv interpolation (no datetime index)", "count": 0})
        if demand_cols:
            log.append({"step": "skipped demand forward fill (no datetime index)", "count": 0})

    clipped_total = 0
    for col in pv_cols + demand_cols:
        if col in work.columns:
            before = work[col].lt(0).sum()
            if before:
                work[col] = work[col].clip(lower=0)
                clipped_total += int(before)
    if clipped_total:
        log.append({"step": "clipped negatives", "count": clipped_total})

    dropped_rows = 0
    if demand_cols:
        mask = work[demand_cols].isna().any(axis=1)
        dropped_rows = int(mask.sum())
        if dropped_rows:
            work = work[~mask]
            log.append({"step": "dropped rows with long demand gaps", "count": dropped_rows})

    work = work.sort_index()
    work = work.reset_index(drop=True)

    if TIMESTAMP_COLUMN in work.columns and ptypes.is_datetime64tz_dtype(work[TIMESTAMP_COLUMN]):
        work[TIMESTAMP_COLUMN] = work[TIMESTAMP_COLUMN].dt.tz_localize(None)

    return CleaningResult(work, log, dropped_rows)
