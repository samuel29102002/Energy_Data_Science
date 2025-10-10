"""Utility functions for statistical time-series modeling (ARIMA/SARIMA)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


NumericSeries = pd.Series


def ensure_numeric(series: Optional[pd.Series]) -> pd.Series:
    """Return numeric series with NaNs dropped. Gracefully handles None/empty input."""

    if series is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    return numeric


def acf_pacf(series: pd.Series, nlags: int = 48) -> Dict[str, pd.DataFrame]:
    """Return ACF and PACF values up to nlags for a numeric series."""

    clean_series = ensure_numeric(series)
    if clean_series.empty:
        empty = pd.DataFrame({"lag": [], "value": []})
        return {"acf": empty, "pacf": empty}

    acf_vals = acf(clean_series, nlags=nlags, fft=True)
    pacf_vals = pacf(clean_series, nlags=nlags, method="ywmle")
    lags = np.arange(len(acf_vals))
    acf_df = pd.DataFrame({"lag": lags, "value": acf_vals})
    pacf_df = pd.DataFrame({"lag": lags, "value": pacf_vals})
    return {"acf": acf_df, "pacf": pacf_df}


def stationarity_checks(series: pd.Series) -> pd.DataFrame:
    """Compute ADF and KPSS tests for stationarity."""

    clean_series = ensure_numeric(series)
    if clean_series.empty:
        return pd.DataFrame(
            [
                {"test": "ADF", "statistic": np.nan, "p_value": np.nan, "lag": np.nan},
                {"test": "KPSS", "statistic": np.nan, "p_value": np.nan, "lag": np.nan},
            ]
        )

    results = []
    try:
        adf_stat, adf_p, adf_usedlag, *_ = adfuller(clean_series, autolag="AIC")
        results.append({"test": "ADF", "statistic": adf_stat, "p_value": adf_p, "lag": adf_usedlag})
    except Exception:  # pragma: no cover - defensive fallback
        results.append({"test": "ADF", "statistic": np.nan, "p_value": np.nan, "lag": np.nan})

    try:
        kpss_stat, kpss_p, kpss_lags, *_ = kpss(clean_series, regression="c", nlags="auto")
        results.append({"test": "KPSS", "statistic": kpss_stat, "p_value": kpss_p, "lag": kpss_lags})
    except Exception:  # pragma: no cover - defensive fallback
        results.append({"test": "KPSS", "statistic": np.nan, "p_value": np.nan, "lag": np.nan})

    return pd.DataFrame(results)


def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[SARIMAXResults]:
    """Fit an (S)ARIMA model on the provided series."""

    clean_series = ensure_numeric(series)
    if clean_series.empty:
        return None

    if seasonal_order is None:
        seasonal_order = (0, 0, 0, 0)

    try:
        model = SARIMAX(
            clean_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend="n",
        )
        result = model.fit(disp=False)
        return result
    except Exception:  # pragma: no cover - modeling may fail for invalid orders
        return None


def forecast_arima(
    fitted_model: SARIMAXResults,
    horizon: int,
    index: Optional[pd.Index] = None,
) -> pd.Series:
    """Forecast using a fitted ARIMA/SARIMA model."""

    if fitted_model is None:
        return pd.Series(dtype=float)

    forecast = fitted_model.get_forecast(steps=horizon)
    mean = forecast.predicted_mean
    if index is not None and len(index) == len(mean):
        mean.index = pd.Index(index)
    mean.name = "y_pred"
    return mean


def evaluate_forecast(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Compute MAE, RMSE, and normalized RMSE as defined for the project."""

    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "nRMSE": np.nan}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.max(y_true) - np.min(y_true))
    if np.isclose(denom, 0.0):
        nrmse = np.nan
    else:
        nrmse = rmse / denom
    return {"MAE": mae, "RMSE": rmse, "nRMSE": nrmse}


def walk_forward_daily(
    df: pd.DataFrame,
    target: str = "Demand",
    days: int = 7,
    horizon: int = 24,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Daily walk-forward evaluation for the last `days` days."""

    if "timestamp" not in df.columns or target not in df.columns:
        return pd.DataFrame(columns=["day_idx", "timestamp", "y_true", "y_pred"]), pd.DataFrame(
            columns=["day_idx", "MAE", "RMSE", "nRMSE"]
        )

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", target])
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target]).sort_values("timestamp")

    if data.empty:
        return (
            pd.DataFrame(columns=["day_idx", "timestamp", "y_true", "y_pred"]),
            pd.DataFrame(columns=["day_idx", "MAE", "RMSE", "nRMSE"]),
        )

    unique_days = np.array(sorted(pd.to_datetime(data["timestamp"]).dt.normalize().unique()))
    if unique_days.size == 0:
        return (
            pd.DataFrame(columns=["day_idx", "timestamp", "y_true", "y_pred"]),
            pd.DataFrame(columns=["day_idx", "MAE", "RMSE", "nRMSE"]),
        )

    selected_days = unique_days[-days:]

    predictions = []
    metrics = []

    for idx, day in enumerate(selected_days, start=1):
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(days=1)

        train_mask = data["timestamp"] < day_start
        test_mask = (data["timestamp"] >= day_start) & (data["timestamp"] < day_end)

        train_df = data.loc[train_mask]
        test_df = data.loc[test_mask]

        if len(train_df) < horizon or len(test_df) < horizon:
            continue

        train_series = train_df.set_index("timestamp")[target]
        model = fit_arima(train_series, order=order, seasonal_order=seasonal_order)
        if model is None:
            continue

        test_index = test_df["timestamp"].tolist()
        forecast = forecast_arima(model, horizon=horizon, index=test_index)
        evaluation = evaluate_forecast(test_df[target].values, forecast.values)

        metrics.append({"day_idx": idx, **evaluation})
        pred_df = pd.DataFrame(
            {
                "day_idx": idx,
                "timestamp": test_index,
                "y_true": test_df[target].values,
                "y_pred": forecast.values,
            }
        )
        predictions.append(pred_df)

    if predictions:
        predictions_df = pd.concat(predictions, ignore_index=True)
    else:
        predictions_df = pd.DataFrame(columns=["day_idx", "timestamp", "y_true", "y_pred"])

    metrics_df = pd.DataFrame(metrics)
    return predictions_df, metrics_df


__all__ = [
    "ensure_numeric",
    "acf_pacf",
    "stationarity_checks",
    "fit_arima",
    "forecast_arima",
    "evaluate_forecast",
    "walk_forward_daily",
]
