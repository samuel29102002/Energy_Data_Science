"""Rolling forecasting utilities for Task 9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .modeling_stats import fit_arima, forecast_arima, evaluate_forecast as evaluate_stat_forecast
from .modeling_ml import train_xgboost, predict_xgboost, evaluate_forecast as evaluate_ml_forecast


DEFAULT_STAT_SPEC = {
    "model_name": "SARIMA(1,1,1)(1,1,1,24)",
    "order": (1, 1, 1),
    "seasonal_order": (1, 1, 1, 24),
}

DEFAULT_ML_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.06,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_lambda": 1.2,
    "random_state": 42,
}

FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "cooling_degree",
    "heating_degree",
    "temp_irradiance_interaction",
    "Temperature",
    "Pressure (hPa)",
    "Cloud_cover (%)",
    "Wind_speed_10m (km/h)",
    "Shortwave_radiation (W/m²)",
    "direct_radiation (W/m²)",
    "diffuse_radiation (W/m²)",
    "direct_normal_irradiance (W/m²)",
    "Price",
]


def ensure_numeric_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    result = df.copy()
    for col in cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def build_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    data["hour"] = data["timestamp"].dt.hour
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["is_weekend"] = (data["timestamp"].dt.dayofweek >= 5).astype(int)
    data["cooling_degree"] = np.clip(data.get("Temperature", 0) - 18, 0, None)
    data["heating_degree"] = np.clip(18 - data.get("Temperature", 0), 0, None)
    data["temp_irradiance_interaction"] = data.get("Temperature", 0) * data.get("Shortwave_radiation (W/m²)", 0)
    data = ensure_numeric_cols(data, FEATURE_COLUMNS + ["Demand"])
    return data


def _evaluate_forecast(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    # Use either evaluation helper (same implementation) for clarity.
    return evaluate_stat_forecast(y_true, y_pred)


def _naive_predictions(train: pd.Series, horizon: int) -> np.ndarray:
    if train.empty:
        return np.array([])
    last_value = train.iloc[-1]
    return np.repeat(last_value, horizon)


def _seasonal_naive_predictions(train: pd.Series, horizon: int, season: int = 24) -> np.ndarray:
    if len(train) < season:
        return _naive_predictions(train, horizon)
    season_values = train.iloc[-season:]
    reps = int(np.ceil(horizon / season))
    repeated = np.tile(season_values.values, reps)
    return repeated[:horizon]


def _prepare_train_val(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    subset = data.dropna(subset=list(feature_cols) + [target])
    X = subset[list(feature_cols)].astype(np.float32)
    y = subset[target].astype(np.float32)
    return X, y


def rolling_forecast_7days(
    df: pd.DataFrame,
    target: str = "Demand",
    horizon: int = 24,
    stat_spec: Optional[Dict] = None,
    ml_params: Optional[Dict] = None,
    include_baselines: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or target not in df.columns:
        empty_preds = pd.DataFrame(columns=["timestamp", "day_idx", "model_name", "y_true", "y_pred"])
        empty_metrics = pd.DataFrame(columns=["day_idx", "model_name", "MAE", "RMSE", "nRMSE"])
        empty_summary = pd.DataFrame(columns=["model_name", "MAE_mean", "RMSE_mean", "nRMSE_mean"])
        return empty_preds, empty_metrics, empty_summary

    stat_spec = stat_spec or DEFAULT_STAT_SPEC
    ml_params = ml_params or DEFAULT_ML_PARAMS

    data = build_forecast_features(df)
    if data.empty:
        return (
            pd.DataFrame(columns=["timestamp", "day_idx", "model_name", "y_true", "y_pred"]),
            pd.DataFrame(columns=["day_idx", "model_name", "MAE", "RMSE", "nRMSE"]),
            pd.DataFrame(columns=["model_name", "MAE_mean", "RMSE_mean", "nRMSE_mean"]),
        )

    data = data.dropna(subset=[target])
    unique_days = np.array(sorted(data["timestamp"].dt.normalize().unique()))
    if unique_days.size == 0:
        return (
            pd.DataFrame(columns=["timestamp", "day_idx", "model_name", "y_true", "y_pred"]),
            pd.DataFrame(columns=["day_idx", "model_name", "MAE", "RMSE", "nRMSE"]),
            pd.DataFrame(columns=["model_name", "MAE_mean", "RMSE_mean", "nRMSE_mean"]),
        )

    selected_days = unique_days[-7:] if unique_days.size >= 7 else unique_days

    predictions_records = []
    metrics_records = []

    for idx, day_start in enumerate(selected_days, start=1):
        day_start = pd.Timestamp(day_start)
        day_end = day_start + pd.Timedelta(days=1)

        train_mask = data["timestamp"] < day_start
        forecast_mask = (data["timestamp"] >= day_start) & (data["timestamp"] < day_end)

        train_df = data.loc[train_mask]
        forecast_df = data.loc[forecast_mask].head(horizon)

        if train_df.empty or forecast_df.empty:
            continue

        y_true = forecast_df[target].values
        timestamps = forecast_df["timestamp"].values
        actual_len = len(y_true)
        if actual_len == 0:
            continue

        def _align_arrays(*arrays):
            converted = [np.asarray(arr) for arr in arrays if arr is not None]
            min_len = min(len(arr) for arr in converted)
            return [arr[:min_len] for arr in converted]

        # Baselines
        if include_baselines:
            naive_preds = _naive_predictions(train_df[target], actual_len)
            seasonal_preds = _seasonal_naive_predictions(train_df[target], actual_len)
            for name, preds in ("Naive", naive_preds), ("SeasonalNaive", seasonal_preds):
                aligned_ts, aligned_true, aligned_pred = _align_arrays(timestamps, y_true, preds)
                metrics = _evaluate_forecast(aligned_true, aligned_pred)
                metrics_records.append({"day_idx": idx, "model_name": name, **metrics})
                predictions_records.append(
                    {
                        "timestamp": aligned_ts.tolist(),
                        "day_idx": idx,
                        "model_name": name,
                        "y_true": aligned_true.tolist(),
                        "y_pred": aligned_pred.tolist(),
                    }
                )

        # Statistical model
        try:
            train_series = train_df.set_index("timestamp")[target]
            stat_model = fit_arima(train_series, order=stat_spec["order"], seasonal_order=stat_spec["seasonal_order"])
            stat_forecast = forecast_arima(stat_model, horizon=actual_len, index=timestamps)
            stat_values = stat_forecast.values if hasattr(stat_forecast, "values") else np.asarray(stat_forecast)
        except Exception:
            stat_values = np.full(actual_len, np.nan)
        aligned_ts, aligned_true, aligned_stat = _align_arrays(timestamps, y_true, stat_values)
        metrics = _evaluate_forecast(aligned_true, aligned_stat)
        metrics_records.append({"day_idx": idx, "model_name": stat_spec["model_name"], **metrics})
        predictions_records.append(
            {
                "timestamp": aligned_ts.tolist(),
                "day_idx": idx,
                "model_name": stat_spec["model_name"],
                "y_true": aligned_true.tolist(),
                "y_pred": aligned_stat.tolist(),
            }
        )

        # ML model
        X_train, y_train = _prepare_train_val(train_df, FEATURE_COLUMNS, target)
        X_forecast, _ = _prepare_train_val(forecast_df, FEATURE_COLUMNS, target)
        X_forecast = X_forecast.head(actual_len)
        if X_train.empty or X_forecast.empty:
            ml_preds = np.full(actual_len, np.nan)
        else:
            # Internal split for early stopping
            if len(X_train) > 72:
                cutoff_ts = train_df["timestamp"].iloc[-72]
                train_internal = train_df[train_df["timestamp"] < cutoff_ts]
                val_internal = train_df[train_df["timestamp"] >= cutoff_ts]
                X_train_internal, y_train_internal = _prepare_train_val(train_internal, FEATURE_COLUMNS, target)
                X_val_internal, y_val_internal = _prepare_train_val(val_internal, FEATURE_COLUMNS, target)
                model_ml, _ = train_xgboost(
                    X_train_internal,
                    y_train_internal,
                    X_val=X_val_internal,
                    y_val=y_val_internal,
                    params=ml_params,
                )
            else:
                model_ml, _ = train_xgboost(X_train, y_train, params=ml_params)
            ml_preds = predict_xgboost(model_ml, X_forecast)
        aligned_ts, aligned_true, aligned_ml = _align_arrays(timestamps, y_true, ml_preds)
        metrics = evaluate_ml_forecast(aligned_true, aligned_ml)
        metrics_records.append({"day_idx": idx, "model_name": "XGBoost", **metrics})
        predictions_records.append(
            {
                "timestamp": aligned_ts.tolist(),
                "day_idx": idx,
                "model_name": "XGBoost",
                "y_true": aligned_true.tolist(),
                "y_pred": aligned_ml.tolist(),
            }
        )

    if not predictions_records:
        empty_preds = pd.DataFrame(columns=["timestamp", "day_idx", "model_name", "y_true", "y_pred"])
        empty_metrics = pd.DataFrame(columns=["day_idx", "model_name", "MAE", "RMSE", "nRMSE"])
        empty_summary = pd.DataFrame(columns=["model_name", "MAE_mean", "RMSE_mean", "nRMSE_mean"])
        return empty_preds, empty_metrics, empty_summary

    # Expand predictions list into tidy DataFrame
    pred_frames = []
    for record in predictions_records:
        df_pred = pd.DataFrame(
            {
                "timestamp": record["timestamp"],
                "day_idx": record["day_idx"],
                "model_name": record["model_name"],
                "y_true": record["y_true"],
                "y_pred": record["y_pred"],
            }
        )
        pred_frames.append(df_pred)
    predictions_df = pd.concat(pred_frames, ignore_index=True)

    metrics_df = pd.DataFrame(metrics_records)

    summary = pd.DataFrame()
    if not metrics_df.empty:
        summary = (
            metrics_df.groupby("model_name").agg(
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
                MAE_median=("MAE", "median"),
                RMSE_mean=("RMSE", "mean"),
                RMSE_std=("RMSE", "std"),
                RMSE_median=("RMSE", "median"),
                nRMSE_mean=("nRMSE", "mean"),
                nRMSE_std=("nRMSE", "std"),
                nRMSE_median=("nRMSE", "median"),
            )
        ).reset_index()

    return predictions_df, metrics_df, summary


__all__ = [
    "ensure_numeric_cols",
    "build_forecast_features",
    "rolling_forecast_7days",
    "DEFAULT_STAT_SPEC",
    "DEFAULT_ML_PARAMS",
    "FEATURE_COLUMNS",
]
