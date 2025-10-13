#!/usr/bin/env python3
"""Generate dashboard-ready visuals and KPI tables from notebooks 01-07.

This utility executes the early exploration notebooks (01-07) and exports
all core plots/tables required by the Dash UI. If the expected PNG/CSV assets
already exist the script skips heavy recomputation.

The workflow:
    1. Execute notebooks 01_*.ipynb-07_*.ipynb (unless --skip-exec is used).
    2. Load raw and cleaned datasets, derive missing KPIs.
    3. Build Plotly figures and persist them as static PNGs (fallback to HTML).
    4. Persist supporting CSV tables for feature insights, missingness, and KPIs.

The script is intentionally defensive: missing dependencies fall back to HTML
exports, and missing source files yield empty/no-op outputs while logging a
warning instead of crashing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nbformat
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

try:  # seasonal decomposition is optional but highly recommended
    from statsmodels.tsa.seasonal import STL
except Exception:  # pragma: no cover - statsmodels may be absent in CI
    STL = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
NOTEBOOK_DIR = ROOT / "notebooks"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"

for directory in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR / "validation"):
    directory.mkdir(parents=True, exist_ok=True)

from src.dash_app.utils_io import load_csv_safe, save_csv_safe  # noqa: E402
from src.dash_app.data_processing import CleaningResult, clean_dataset  # noqa: E402
from src.dash_app.utils_figures import (  # noqa: E402
    make_demand_pv_timeseries,
    make_demand_seasonality,
    make_missingness_heatmap,
    save_plotly_static,
)


NOTEBOOK_SEQUENCE: Sequence[Tuple[str, str]] = (
    ("01_visualization.ipynb", "01"),
    ("02_project_planning.ipynb", "02"),
    ("03_visualization.ipynb", "03"),
    ("04_pv_cleaning.ipynb", "04"),
    ("05_feature_engineering.ipynb", "05"),
    ("06_ts_decomposition.ipynb", "06"),
    ("07_stats_models_ARMA.ipynb", "07"),
)

FIGURE_TARGETS: Dict[str, str] = {
    "timeseries_main": "01_demand_pv_timeseries.png",
    "timeseries_daily": "01_demand_pv_daily_sample.png",
    "diurnal_profile": "02_demand_diurnal_profile.png",
    "weekday_heatmap": "02_demand_weekday_heatmap.png",
    "seasonality_stl": "03_demand_seasonality_stl.png",
    "missingness_heatmap": "04_missingness_heatmap.png",
    "feature_importance": "05_feature_importance.png",
    "baseline_diagnostics": "06_baseline_diagnostics.png",
}

TABLE_TARGETS: Dict[str, str] = {
    "pv_gap_stats": "04_pv_gap_stats.csv",
    "feature_stats": "05_feature_stats.csv",
    "kpis": "07_kpis.csv",
}


@dataclass
class AssetStatus:
    name: str
    path: Path
    generated: bool
    notes: List[str]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def execute_notebook(nb_path: Path, timeout: int = 900) -> List[str]:
    """Execute a notebook in-place, returning error messages if they occur."""
    logging.info("Executing notebook %s", nb_path.name)
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except FileNotFoundError:
        msg = f"Notebook not found: {nb_path}"
        logging.warning(msg)
        return [msg]

    resources = {"metadata": {"path": str(nb_path.parent)}}
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources=resources)
    try:
        client.execute()
        nbformat.write(nb, nb_path)
        return []
    except CellExecutionError as exc:  # pragma: no cover - runtime safeguard
        tb_lines = [line for line in str(exc).splitlines() if line.strip()]
        message = tb_lines[-1] if tb_lines else str(exc)
        logging.error("Notebook %s failed: %s", nb_path.name, message)
        return [message]
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logging.exception("Unexpected error while running %s", nb_path)
        return [str(exc)]


def execute_notebooks(sequence: Sequence[Tuple[str, str]], skip: bool) -> Dict[str, List[str]]:
    reports: Dict[str, List[str]] = {}
    if skip:
        logging.info("Notebook execution skipped via --skip-exec")
        return {name: [] for name, _ in sequence}
    for notebook_name, _ in sequence:
        nb_path = NOTEBOOK_DIR / notebook_name
        reports[notebook_name] = execute_notebook(nb_path)
    return reports


def ensure_dataframe(df: pd.DataFrame, expected_columns: Iterable[str]) -> pd.DataFrame:
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def derive_clean_dataset(raw_df: pd.DataFrame) -> CleaningResult:
    if raw_df.empty:
        logging.warning("Raw dataset is empty; returning empty cleaning result")
        return CleaningResult(pd.DataFrame(), [], 0)
    return clean_dataset(raw_df)


def sample_daily_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    start = df["timestamp"].min()
    if pd.isna(start):
        return df
    end = start + pd.Timedelta(days=days)
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()


def make_diurnal_profile(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    work = df.copy()
    work["hour"] = work["timestamp"].dt.hour
    grouped = work.groupby("hour")["Demand"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["hour"],
            y=grouped["Demand"],
            mode="lines+markers",
            name="Demand",
            line=dict(color="#1F77B4", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="Average Demand by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Demand (kWh)",
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(dtick=1)
    return fig


def make_weekday_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    work = df.copy()
    work["hour"] = work["timestamp"].dt.hour
    work["weekday"] = work["timestamp"].dt.day_name()
    pivot = (
        work.pivot_table(
            index="weekday",
            columns="hour",
            values="Demand",
            aggfunc="mean",
        )
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            axis=0,
        )
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Blues",
            colorbar=dict(title="kWh"),
        )
    )
    fig.update_layout(
        title="Demand Heatmap by Weekday/Hour",
        xaxis_title="Hour",
        yaxis_title="Weekday",
        template="plotly_white",
        autosize=True,
        height=420,
        margin=dict(l=60, r=20, t=60, b=80),
    )
    return fig


def make_feature_importance(df: pd.DataFrame, target: str = "Demand") -> go.Figure:
    if df.empty or target not in df.columns:
        return go.Figure()
    numeric = df.select_dtypes(include=["number", "float", "int"])
    correlations = (
        numeric.corr(method="pearson")[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
        .head(12)
    )
    fig = go.Figure(
        go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation="h",
            marker=dict(color="#1F77B4"),
        )
    )
    fig.update_layout(
        title="Feature correlation strength vs Demand",
        xaxis_title="|Pearson correlation|",
        yaxis_title="Feature",
        template="plotly_white",
        height=420,
        margin=dict(l=140, r=40, t=60, b=40),
    )
    return fig


def make_baseline_diagnostics(df: pd.DataFrame) -> go.Figure:
    if df.empty or "Demand" not in df.columns:
        return go.Figure()
    work = df.sort_values("timestamp").copy()
    work["baseline"] = work["Demand"].shift(24)
    mask = work["baseline"].notna()
    if mask.sum() < 24:
        return go.Figure()
    work = work.loc[mask].copy()
    work["error"] = work["Demand"] - work["baseline"]
    daily_rmse = work.set_index("timestamp")["error"].resample("D").apply(lambda x: np.sqrt((x**2).mean()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"],
            y=work["Demand"],
            name="Actual",
            mode="lines",
            line=dict(color="#1F77B4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"],
            y=work["baseline"],
            name="Lag-24 Baseline",
            mode="lines",
            line=dict(color="#FF7F0E", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily_rmse.index,
            y=daily_rmse.values,
            name="Daily RMSE",
            mode="lines",
            yaxis="y2",
            line=dict(color="#2CA02C", width=2),
        )
    )
    fig.update_layout(
        title="Baseline Forecast Diagnostics",
        xaxis_title="Timestamp",
        yaxis=dict(title="Demand (kWh)", side="left", color="#1F77B4"),
        yaxis2=dict(title="RMSE", overlaying="y", side="right", color="#2CA02C"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=50, r=60, t=60, b=40),
    )
    return fig


def compute_pv_gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_pct", "zero_pct", "mean", "std"])
    pv_cols = [col for col in df.columns if "pv" in col.lower()]
    rows: List[Dict[str, float]] = []
    total_rows = len(df)
    for col in pv_cols:
        column = df[col]
        rows.append(
            {
                "column": col,
                "missing_pct": float(column.isna().mean() * 100.0),
                "zero_pct": float((column == 0).mean() * 100.0),
                "mean": float(column.mean() or 0.0),
                "std": float(column.std() or 0.0),
                "nonzero_hours": int((column > 0).sum()),
                "total_rows": total_rows,
            }
        )
    return pd.DataFrame(rows)


def compute_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["feature", "mean", "std", "correlation_with_demand"])
    numeric = df.select_dtypes(include=["number", "float", "int"])
    corr = numeric.corr().get("Demand") if "Demand" in numeric.columns else pd.Series(dtype=float)
    stats = []
    for col in numeric.columns:
        if col == "Demand":
            continue
        stats.append(
            {
                "feature": col,
                "mean": float(numeric[col].mean() or 0.0),
                "std": float(numeric[col].std() or 0.0),
                "correlation_with_demand": float(abs(corr[col])) if corr is not None and col in corr else np.nan,
            }
        )
    result = pd.DataFrame(stats)
    return result.sort_values("correlation_with_demand", ascending=False).reset_index(drop=True)


def compute_kpis(clean_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    def add_kpi(name: str, value: object, unit: str, description: str) -> None:
        rows.append({"kpi": name, "value": value, "unit": unit, "description": description})

    if clean_df.empty:
        return pd.DataFrame(rows)

    mean_demand = clean_df["Demand"].mean()
    add_kpi("Mean demand", round(float(mean_demand), 3), "kWh", "Average hourly demand across cleaned dataset")

    clean_df["hour"] = clean_df["timestamp"].dt.hour
    peak_hour = int(clean_df.groupby("hour")["Demand"].mean().idxmax())
    add_kpi("Peak demand hour", f"{peak_hour:02d}:00", "hour", "Hour of day with highest mean demand")

    if "pv" in clean_df.columns and clean_df["Demand"].gt(0).any():
        pv_share = float((clean_df["pv"].clip(lower=0).sum() / clean_df["Demand"].clip(lower=0).sum()) * 100.0)
        add_kpi("PV self-consumption", round(pv_share, 2), "%", "Share of demand covered by PV output")

    missing_ratio = float(raw_df.isna().sum().sum() / raw_df.size * 100.0) if not raw_df.empty else float("nan")
    add_kpi("Overall missingness", round(missing_ratio, 2), "%", "Share of missing entries in raw dataset")

    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, target_name: str) -> AssetStatus:
    target_path = TABLES_DIR / target_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv_safe(df, target_path)
    return AssetStatus(target_name, target_path, generated=True, notes=[])


def maybe_generate_figure(fig: go.Figure, target_name: str) -> AssetStatus:
    target_path = FIGURES_DIR / target_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if fig.data:
        result = save_plotly_static(fig, target_path)
        notes = [] if result.generated else ["Saved HTML fallback"]
        return AssetStatus(target_name, result.path, result.generated, notes)
    if target_path.exists():
        return AssetStatus(target_name, target_path, generated=False, notes=["Figure already present"])
    fallback = target_path.with_suffix(".html")
    return AssetStatus(target_name, fallback, generated=False, notes=["No data available"])


def generate_assets(raw_df: pd.DataFrame, clean_result: CleaningResult, forecast_df: pd.DataFrame, features_df: pd.DataFrame) -> Dict[str, AssetStatus]:
    assets: Dict[str, AssetStatus] = {}
    cleaned_df = clean_result.cleaned

    logging.info("Generating demand vs PV timeseries figures")
    fig_main = make_demand_pv_timeseries(cleaned_df if not cleaned_df.empty else raw_df)
    assets["timeseries_main"] = maybe_generate_figure(fig_main, FIGURE_TARGETS["timeseries_main"])

    sample = sample_daily_window(cleaned_df if not cleaned_df.empty else raw_df, days=7)
    fig_sample = make_demand_pv_timeseries(sample)
    fig_sample.update_layout(title="Demand vs PV (first 7 days)")
    assets["timeseries_daily"] = maybe_generate_figure(fig_sample, FIGURE_TARGETS["timeseries_daily"])

    logging.info("Building seasonality visuals")
    fig_diurnal = make_diurnal_profile(cleaned_df)
    assets["diurnal_profile"] = maybe_generate_figure(fig_diurnal, FIGURE_TARGETS["diurnal_profile"])

    fig_heatmap = make_weekday_heatmap(cleaned_df)
    assets["weekday_heatmap"] = maybe_generate_figure(fig_heatmap, FIGURE_TARGETS["weekday_heatmap"])

    if STL is not None and not cleaned_df.empty:
        logging.info("Running STL decomposition for seasonality")
        demand_series = cleaned_df.set_index("timestamp")["Demand"].asfreq("H").interpolate(limit=6)
        stl = STL(demand_series, period=24 * 7, robust=True)
        result = stl.fit()
        fig_stl = go.Figure()
        fig_stl.add_trace(go.Scatter(x=demand_series.index, y=result.trend, name="Trend", line=dict(color="#1F77B4")))
        fig_stl.add_trace(go.Scatter(x=demand_series.index, y=result.seasonal, name="Seasonal", line=dict(color="#2CA02C")))
        fig_stl.add_trace(go.Scatter(x=demand_series.index, y=result.resid, name="Residual", line=dict(color="#FF7F0E")))
        fig_stl.update_layout(
            title="STL Decomposition of Demand",
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=20, t=60, b=40),
        )
    else:
        logging.warning("STL unavailable or dataset empty; creating placeholder seasonality figure")
        fig_stl = make_demand_seasonality(cleaned_df)
    assets["seasonality_stl"] = maybe_generate_figure(fig_stl, FIGURE_TARGETS["seasonality_stl"])

    logging.info("Creating missingness heatmap")
    fig_missing = make_missingness_heatmap(raw_df)
    assets["missingness_heatmap"] = maybe_generate_figure(fig_missing, FIGURE_TARGETS["missingness_heatmap"])

    logging.info("Computing feature importances and statistics")
    feature_stats = compute_feature_stats(features_df)
    assets["feature_stats"] = save_dataframe(feature_stats, TABLE_TARGETS["feature_stats"])

    fig_feature = make_feature_importance(features_df)
    assets["feature_importance"] = maybe_generate_figure(fig_feature, FIGURE_TARGETS["feature_importance"])

    logging.info("Baseline diagnostics")
    fig_baseline = make_baseline_diagnostics(forecast_df)
    assets["baseline_diagnostics"] = maybe_generate_figure(fig_baseline, FIGURE_TARGETS["baseline_diagnostics"])

    logging.info("PV gap stats")
    pv_stats = compute_pv_gap_stats(raw_df)
    assets["pv_gap_stats"] = save_dataframe(pv_stats, TABLE_TARGETS["pv_gap_stats"])

    logging.info("KPI export")
    kpi_df = compute_kpis(cleaned_df, raw_df)
    assets["kpis"] = save_dataframe(kpi_df, TABLE_TARGETS["kpis"])

    return assets


def summarize_assets(assets: Dict[str, AssetStatus], notebook_errors: Dict[str, List[str]], output: Optional[Path]) -> None:
    summary = {
        "notebook_errors": notebook_errors,
        "assets": {
            name: {
                "path": str(status.path),
                "generated": status.generated,
                "notes": status.notes,
            }
            for name, status in assets.items()
        },
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Export summary:\n%s", json.dumps(summary, indent=2))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-exec", action="store_true", help="Skip executing notebooks and only regenerate assets")
    parser.add_argument("--summary", type=Path, default=REPORTS_DIR / "validation" / "export_visuals_summary.json")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_logging(args.verbose)

    notebook_errors = execute_notebooks(NOTEBOOK_SEQUENCE, skip=args.skip_exec)

    logging.info("Loading datasets")
    raw_df = load_csv_safe(RAW_DIR / "train_252145.csv", parse_dates=["timestamp"])
    forecast_df = load_csv_safe(RAW_DIR / "forecast.csv", parse_dates=["timestamp"])

    try:
        features_df = pd.read_parquet(PROCESSED_DIR / "task5_features.parquet")
        features_df = features_df.reset_index().rename(columns={"index": "timestamp"}) if "timestamp" not in features_df.columns else features_df
        if "timestamp" in features_df.columns:
            features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], errors="coerce")
    except Exception:
        logging.warning("Unable to load processed feature set; falling back to cleaned raw data")
        features_df = pd.DataFrame()

    clean_result = derive_clean_dataset(raw_df)

    assets = generate_assets(raw_df, clean_result, forecast_df, features_df)
    summarize_assets(assets, notebook_errors, args.summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
