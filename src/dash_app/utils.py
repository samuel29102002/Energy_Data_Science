from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.dash_app.data_processing import CleaningResult, clean_dataset

ROOT = Path(__file__).resolve().parents[2]
TABLES_PATH = ROOT / "reports" / "tables"


@dataclass(frozen=True)
class DashboardData:
    tables_path: Path
    forecast_predictions: pd.DataFrame
    forecast_metrics_summary: pd.DataFrame
    ml_split_metrics: pd.DataFrame
    ml_split_predictions: pd.DataFrame
    ml_feature_importance: pd.DataFrame
    storage_summary: pd.DataFrame
    forecast_models: List[str]
    ml_models: List[str]
    storage_scenarios: List[str]
    best_forecast_model: Optional[str]
    overview_metrics: Dict[str, Optional[float]]
    summary_context: Dict[str, Any]
    raw_dataset: pd.DataFrame
    cleaned_dataset: pd.DataFrame
    cleaning_log: List[Dict[str, object]]
    descriptive_stats: pd.DataFrame
    raw_schema: pd.DataFrame


def _read_csv(path: Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()


def _clean_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned


def _compute_forecast_models(predictions: pd.DataFrame, summary: pd.DataFrame) -> List[str]:
    models: List[str] = []
    if "model_name" in predictions.columns:
        models = predictions["model_name"].dropna().unique().tolist()
    if not models and "model_name" in summary.columns:
        models = summary["model_name"].dropna().unique().tolist()
    return sorted(models)


def _compute_ml_models(predictions: pd.DataFrame) -> List[str]:
    excluded = {"timestamp", "Actual", "actual", "y_true"}
    return [col for col in predictions.columns if col not in excluded]


def _compute_storage_scenarios(summary: pd.DataFrame) -> List[str]:
    if "Scenario" not in summary.columns:
        return []
    return summary["Scenario"].dropna().unique().tolist()


def _safe_round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return float(round(value, digits))


def load_dashboard_data(tables_path: Optional[Path] = None) -> DashboardData:
    path = tables_path or TABLES_PATH

    forecast_predictions = _read_csv(path / "forecast_predictions.csv", parse_dates=["timestamp", "forecast_day"])
    if not forecast_predictions.empty:
        forecast_predictions = forecast_predictions.sort_values(["timestamp", "model_name"]).reset_index(drop=True)

    raw_data_path = ROOT / "data" / "raw" / "train_252145.csv"
    raw_dataset = _read_csv(raw_data_path, parse_dates=["timestamp"])
    if not raw_dataset.empty and "timestamp" in raw_dataset.columns:
        timestamps = pd.to_datetime(raw_dataset["timestamp"], errors="coerce", utc=True)
        raw_dataset["timestamp"] = timestamps.dt.tz_convert(None)
        raw_dataset = raw_dataset.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    cleaning: CleaningResult
    if raw_dataset.empty:
        cleaning = CleaningResult(pd.DataFrame(), [], 0)
    else:
        cleaning = clean_dataset(raw_dataset)
    cleaned_dataset = cleaning.cleaned.copy()
    if not cleaned_dataset.empty and "timestamp" in cleaned_dataset.columns and pd.api.types.is_datetime64tz_dtype(cleaned_dataset["timestamp"]):
        cleaned_dataset["timestamp"] = cleaned_dataset["timestamp"].dt.tz_localize(None)

    forecast_metrics_summary = _read_csv(path / "forecast_metrics_summary.csv")
    forecast_metrics_summary = _clean_numeric(
        forecast_metrics_summary,
        [
            "MAE_mean",
            "MAE_std",
            "MAE_median",
            "RMSE_mean",
            "RMSE_std",
            "RMSE_median",
            "nRMSE_mean",
            "nRMSE_std",
            "nRMSE_median",
        ],
    )

    ml_split_metrics = _read_csv(path / "ml_split_metrics.csv")
    ml_split_metrics = _clean_numeric(ml_split_metrics, ["MAE", "RMSE", "nRMSE"])

    ml_split_predictions = _read_csv(path / "ml_split_predictions.csv", parse_dates=["timestamp"])
    if not ml_split_predictions.empty:
        ml_split_predictions = ml_split_predictions.sort_values("timestamp").reset_index(drop=True)

    ml_feature_importance = _read_csv(path / "ml_feature_importance.csv")
    ml_feature_importance = _clean_numeric(ml_feature_importance, ["importance"])
    if not ml_feature_importance.empty:
        ml_feature_importance = ml_feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)

    storage_summary = _read_csv(path / "storage_optimization_summary.csv")
    storage_summary = _clean_numeric(
        storage_summary,
        [
            "Total_cost_EUR",
            "Energy_bought_kWh",
            "Energy_sold_kWh",
            "Battery_cycles",
            "SOC_max_kWh",
            "SOC_min_kWh",
        ],
    )

    forecast_models = _compute_forecast_models(forecast_predictions, forecast_metrics_summary)
    ml_models = _compute_ml_models(ml_split_predictions)
    storage_scenarios = _compute_storage_scenarios(storage_summary)

    best_model = None
    best_rmse = None
    best_nrmse = None
    if not forecast_metrics_summary.empty and "RMSE_mean" in forecast_metrics_summary.columns:
        ordered = forecast_metrics_summary.dropna(subset=["RMSE_mean"]).sort_values("RMSE_mean")
        if not ordered.empty:
            best_row = ordered.iloc[0]
            best_model = str(best_row.get("model_name", "")).strip() or None
            best_rmse = float(best_row.get("RMSE_mean")) if pd.notna(best_row.get("RMSE_mean")) else None
            best_nrmse = float(best_row.get("nRMSE_mean")) if pd.notna(best_row.get("nRMSE_mean")) else None

    avg_rmse = None
    if not forecast_metrics_summary.empty and "RMSE_mean" in forecast_metrics_summary.columns:
        avg_values = forecast_metrics_summary["RMSE_mean"].dropna()
        if not avg_values.empty:
            avg_rmse = float(avg_values.mean())

    cost_savings = None
    if not storage_summary.empty and "Total_cost_EUR" in storage_summary.columns:
        costs = storage_summary["Total_cost_EUR"].dropna()
        if len(costs) >= 2:
            cost_savings = float(costs.max() - costs.min())

    forecast_accuracy = None
    if best_nrmse is not None:
        forecast_accuracy = max(0.0, 100.0 - best_nrmse * 100.0)

    overview_metrics = {
        "average_rmse": _safe_round(avg_rmse, 3),
        "best_model_rmse": _safe_round(best_rmse, 3),
        "cost_savings": _safe_round(cost_savings, 2),
        "forecast_accuracy": _safe_round(forecast_accuracy, 1),
        "best_model": best_model,
    }

    start_date = None
    end_date = None
    if not raw_dataset.empty and "timestamp" in raw_dataset.columns:
        start_date = raw_dataset["timestamp"].min()
        end_date = raw_dataset["timestamp"].max()
    elif not forecast_predictions.empty and "timestamp" in forecast_predictions.columns:
        start_date = forecast_predictions["timestamp"].min()
        end_date = forecast_predictions["timestamp"].max()

    summary_context: Dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "model_count": len(forecast_models),
        "scenario_count": len(storage_scenarios),
        "tables_path": str(path),
    }

    descriptive_stats = _read_csv(path / "task5_descriptive_stats.csv")
    if not descriptive_stats.empty:
        first_col = descriptive_stats.columns[0]
        if first_col.startswith("Unnamed") or first_col == "":
            descriptive_stats = descriptive_stats.rename(columns={first_col: "variable"})
        elif first_col != "variable":
            descriptive_stats = descriptive_stats.rename(columns={first_col: "variable"})
    raw_schema = pd.DataFrame()
    if not raw_dataset.empty:
        missing_pct = raw_dataset.isna().mean() * 100.0
        raw_schema = pd.DataFrame(
            {
                "column": raw_dataset.columns,
                "dtype": raw_dataset.dtypes.astype(str).values,
                "missing_pct": missing_pct.values,
            }
        )

    return DashboardData(
        tables_path=path,
        forecast_predictions=forecast_predictions,
        forecast_metrics_summary=forecast_metrics_summary,
        ml_split_metrics=ml_split_metrics,
        ml_split_predictions=ml_split_predictions,
        ml_feature_importance=ml_feature_importance,
        storage_summary=storage_summary,
        forecast_models=forecast_models,
        ml_models=ml_models,
        storage_scenarios=storage_scenarios,
        best_forecast_model=best_model,
        overview_metrics=overview_metrics,
        summary_context=summary_context,
        raw_dataset=raw_dataset,
        cleaned_dataset=cleaned_dataset,
        cleaning_log=cleaning.log,
        descriptive_stats=descriptive_stats,
        raw_schema=raw_schema,
    )


def format_number(value: Optional[float], suffix: str = "", precision: int = 2, as_percent: bool = False) -> str:
    if value is None:
        return "--"
    if as_percent:
        return f"{round(value, precision)}%"
    return f"{round(value, precision)}{suffix}"


__all__ = ["DashboardData", "TABLES_PATH", "load_dashboard_data", "format_number"]
