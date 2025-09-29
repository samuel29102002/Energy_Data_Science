from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.modeling_ml import (
    align_by_index,
    build_ml_dataset,
    evaluate_forecast as evaluate_ml,
    predict_xgboost,
    train_xgboost,
)

try:  # Prefer project stat helpers when available.
    from src.modeling_stats import fit_arima, forecast_arima
except Exception:  # pragma: no cover - fallback only used when helpers missing.
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def fit_arima(
        y: pd.Series,
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs: Any,
    ) -> Any:
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            **kwargs,
        )
        return model.fit(disp=False)

    def forecast_arima(model: Any, steps: int) -> np.ndarray:
        return model.forecast(steps=steps)


# Placeholder that the notebook configures at runtime.
FEATURE_COLUMNS: List[str] = []


def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Dataframe must include a 'timestamp' column")
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    data["timestamp"] = data["timestamp"].dt.tz_convert(None)
    data.reset_index(drop=True, inplace=True)
    return data


def select_target(df: pd.DataFrame, target: str = "Demand") -> str:
    if target in df.columns:
        return target
    lowered = target.lower()
    if lowered in df.columns:
        return lowered
    raise ValueError(f"Target column '{target}' not present in dataframe")


def build_forecast_features(
    df: pd.DataFrame,
    target: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    cols = list(feature_cols) if feature_cols else None
    return build_ml_dataset(df, target=target, feature_cols=cols, dropna=True)


def naive_baseline(last_value: float, horizon: int = 24) -> np.ndarray:
    if np.isnan(last_value):
        return np.array([], dtype=np.float32)
    return np.full(horizon, float(last_value), dtype=np.float32)


def seasonal_naive_baseline(
    history: pd.Series,
    horizon: int = 24,
    season: int = 24,
) -> np.ndarray:
    cleaned = history.dropna()
    if cleaned.empty:
        return np.array([], dtype=np.float32)
    values = cleaned.iloc[-season:].values if len(cleaned) >= season else cleaned.values
    repeats = int(np.ceil(horizon / len(values)))
    tiled = np.tile(values, repeats)
    return tiled[:horizon].astype(np.float32)


def fit_stat_model(
    train_series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> Any:
    return fit_arima(train_series, order=order, seasonal_order=seasonal_order)


def predict_stat(model: Any, steps: int) -> np.ndarray:
    preds = forecast_arima(model, steps=steps)
    return np.asarray(preds, dtype=np.float32)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return evaluate_ml(y_true, y_pred)


def split_7_consecutive_days(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    data = ensure_timestamp(df)
    unique_days = data["timestamp"].dt.floor("D").drop_duplicates().sort_values()
    if unique_days.empty:
        return []
    selected = unique_days.iloc[-7:] if len(unique_days) >= 7 else unique_days
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for day in selected:
        start = pd.Timestamp(day)
        end = start + pd.Timedelta(hours=23)
        ranges.append((start, end))
    return ranges


def forecast_one_day(
    full_df: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon: int,
    target: str,
    stat_spec: Optional[Dict[str, Any]] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    data = ensure_timestamp(full_df)
    target_col = select_target(data, target)
    numeric_cols = list(feature_cols) if feature_cols else [
        col for col in data.select_dtypes(include=[np.number]).columns if col != target_col
    ]
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

    train_df = data[data["timestamp"] <= train_end].copy()
    if train_df.empty:
        raise ValueError("Training window contains no observations")

    future_hours = pd.date_range(train_end + pd.Timedelta(hours=1), periods=horizon, freq="H")
    future_df = data[data["timestamp"].isin(future_hours)].copy()

    # Statistical model
    train_series = train_df.set_index("timestamp")[target_col].dropna()
    if train_series.empty:
        raise ValueError("Training target series is empty after dropping NaN values")

    stat_kwargs = stat_spec or {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 24)}
    stat_model = fit_stat_model(
        train_series,
        order=stat_kwargs.get("order", (1, 1, 1)),
        seasonal_order=stat_kwargs.get("seasonal_order", (1, 1, 1, 24)),
    )
    stat_pred = predict_stat(stat_model, horizon)

    # Machine learning model
    X_train, y_train, _ = build_forecast_features(train_df, target_col, numeric_cols)
    ml_params = dict(xgb_params or {})
    ml_model, _ = train_xgboost(X_train, y_train, params=ml_params, seed=ml_params.get("random_state", 42))

    X_future, y_future, idx_future = build_forecast_features(future_df, target_col, numeric_cols)
    ml_pred = predict_xgboost(ml_model, X_future)

    # Baselines
    last_value = float(train_series.iloc[-1]) if not train_series.empty else np.nan
    naive_pred = naive_baseline(last_value, horizon)
    snaive_pred = seasonal_naive_baseline(train_series, horizon=horizon, season=24)

    true_series = future_df.set_index("timestamp")[target_col].astype(float)

    records: List[pd.DataFrame] = []

    def append_model(name: str, preds: np.ndarray, pred_index: Sequence[pd.Timestamp]) -> None:
        if preds.size == 0 or len(pred_index) == 0:
            return
        common_idx, aligned_true, aligned_pred = align_by_index(
            true_series.values,
            preds,
            true_series.index,
            pred_index,
        )
        if common_idx.empty:
            return
        frame = pd.DataFrame(
            {
                "timestamp": common_idx,
                "model_name": name,
                "y_true": aligned_true.values,
                "y_pred": aligned_pred.values,
            }
        )
        records.append(frame)

    append_model("BestStat", stat_pred, future_hours)
    append_model("XGBoost", ml_pred, idx_future)
    append_model("Naive", naive_pred, future_hours)
    append_model("SeasonalNaive", snaive_pred, future_hours)

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame(columns=["timestamp", "model_name", "y_true", "y_pred"])


def rolling_forecast_7days(
    df: pd.DataFrame,
    target: str = "Demand",
    horizon: int = 24,
    stat_spec: Optional[Dict[str, Any]] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        empty_preds = pd.DataFrame(columns=["timestamp", "day_idx", "model_name", "y_true", "y_pred"])
        empty_metrics = pd.DataFrame(columns=["day_idx", "model_name", "MAE", "RMSE", "nRMSE"])
        empty_summary = pd.DataFrame(columns=["model_name", "MAE_mean", "MAE_std", "MAE_median", "RMSE_mean", "RMSE_std", "RMSE_median", "nRMSE_mean", "nRMSE_std", "nRMSE_median"])
        return empty_preds, empty_metrics, empty_summary

    day_ranges = split_7_consecutive_days(df)
    if not day_ranges:
        empty_preds = pd.DataFrame(columns=["timestamp", "day_idx", "model_name", "y_true", "y_pred"])
        empty_metrics = pd.DataFrame(columns=["day_idx", "model_name", "MAE", "RMSE", "nRMSE"])
        empty_summary = pd.DataFrame(columns=["model_name", "MAE_mean", "MAE_std", "MAE_median", "RMSE_mean", "RMSE_std", "RMSE_median", "nRMSE_mean", "nRMSE_std", "nRMSE_median"])
        return empty_preds, empty_metrics, empty_summary

    predictions = []
    metrics_rows = []

    for idx, (day_start, _) in enumerate(day_ranges, start=1):
        train_end = day_start - pd.Timedelta(hours=1)
        day_preds = forecast_one_day(
            df,
            train_end=train_end,
            horizon=horizon,
            target=target,
            stat_spec=stat_spec,
            xgb_params=xgb_params,
            feature_cols=feature_cols,
        )
        if day_preds.empty:
            continue
        day_preds = day_preds.copy()
        day_preds["day_idx"] = idx
        day_preds["forecast_day"] = day_start.date()
        predictions.append(day_preds)

    if not predictions:
        empty_preds = pd.DataFrame(columns=["timestamp", "day_idx", "model_name", "y_true", "y_pred"])
        empty_metrics = pd.DataFrame(columns=["day_idx", "model_name", "MAE", "RMSE", "nRMSE"])
        empty_summary = pd.DataFrame(columns=["model_name", "MAE_mean", "MAE_std", "MAE_median", "RMSE_mean", "RMSE_std", "RMSE_median", "nRMSE_mean", "nRMSE_std", "nRMSE_median"])
        return empty_preds, empty_metrics, empty_summary

    predictions_df = pd.concat(predictions, ignore_index=True)

    for (day_idx, model_name), group in predictions_df.groupby(["day_idx", "model_name"]):
        mask = group["y_true"].notna() & group["y_pred"].notna()
        if not mask.any():
            continue
        metrics = evaluate_forecast(group.loc[mask, "y_true"].values, group.loc[mask, "y_pred"].values)
        metrics_rows.append({
            "day_idx": int(day_idx),
            "model_name": model_name,
            **metrics,
        })

    metrics_day_df = pd.DataFrame(metrics_rows)

    summary_rows: List[Dict[str, Any]] = []
    if not metrics_day_df.empty:
        for model_name, group in metrics_day_df.groupby("model_name"):
            summary_rows.append(
                {
                    "model_name": model_name,
                    "MAE_mean": group["MAE"].mean(),
                    "MAE_std": group["MAE"].std(ddof=0),
                    "MAE_median": group["MAE"].median(),
                    "RMSE_mean": group["RMSE"].mean(),
                    "RMSE_std": group["RMSE"].std(ddof=0),
                    "RMSE_median": group["RMSE"].median(),
                    "nRMSE_mean": group["nRMSE"].mean(),
                    "nRMSE_std": group["nRMSE"].std(ddof=0),
                    "nRMSE_median": group["nRMSE"].median(),
                }
            )

    metrics_summary_df = pd.DataFrame(summary_rows)

    return predictions_df, metrics_day_df, metrics_summary_df


__all__ = [
    "FEATURE_COLUMNS",
    "build_forecast_features",
    "naive_baseline",
    "seasonal_naive_baseline",
    "forecast_one_day",
    "rolling_forecast_7days",
    "evaluate_forecast",
]
