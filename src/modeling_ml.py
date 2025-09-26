"""Machine learning utilities for demand forecasting."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple
import inspect

import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "XGBoost failed to load because libomp is missing. On macOS run: brew install libomp"
    ) from e
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "XGBoost library could not be loaded. Make sure libomp is installed (brew install libomp on macOS)."
    ) from e
from packaging import version


RANDOM_SEED = 42


def build_ml_dataset(
    df: pd.DataFrame,
    target: str = "Demand",
    feature_cols: Optional[Sequence[str]] = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Prepare ML-ready feature matrix and target series."""

    if df is None or df.empty or target not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Index([])

    data = df.copy()
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c not in {target, "timestamp"}]

    data = data[feature_cols + [target]].apply(pd.to_numeric, errors="coerce")
    if dropna:
        data = data.dropna()

    X = data[feature_cols].astype(np.float32)
    y = data[target].astype(np.float32)

    if "timestamp" in df.columns:
        idx = pd.Index(df.loc[data.index, "timestamp"], name="timestamp")
    else:
        idx = pd.Index(data.index)

    return X, y, idx


def evaluate_forecast(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "nRMSE": np.nan}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.max(y_true) - np.min(y_true))
    nrmse = rmse / denom if not np.isclose(denom, 0.0) else np.nan
    return {"MAE": mae, "RMSE": rmse, "nRMSE": nrmse}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, float]] = None,
    seed: int = RANDOM_SEED,
):
    if X_train is None or y_train is None or len(X_train) == 0 or len(y_train) == 0:
        return None, {}

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    use_validation = (
        X_val is not None
        and y_val is not None
        and len(X_val) > 0
        and len(y_val) > 0
    )
    if use_validation:
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32).ravel()

    defaults = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
    )
    params = params or {}
    p = {**defaults, **params}
    model = xgb.XGBRegressor(**p)
    if "eval_metric" not in p:
        model.set_params(eval_metric="rmse")

    fit_kwargs: Dict[str, object] = {}
    if use_validation:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]

    # Feature-detect support for callbacks / early_stopping_rounds
    fit_sig = inspect.signature(xgb.XGBRegressor.fit)
    supports_callbacks = "callbacks" in fit_sig.parameters
    supports_es_rounds = "early_stopping_rounds" in fit_sig.parameters

    if use_validation:
        if version.parse(xgb.__version__) >= version.parse("2.0.0") and supports_callbacks:
            fit_kwargs["callbacks"] = [
                xgb.callback.EarlyStopping(rounds=50, save_best=True, maximize=False)
            ]
        elif supports_es_rounds:
            fit_kwargs["early_stopping_rounds"] = 50

    # Final fit with robust fallback if callbacks not accepted
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError as te:
        # Retry without callbacks if older xgboost rejects it
        if "callbacks" in str(te) and "callbacks" in fit_kwargs:
            fit_kwargs.pop("callbacks", None)
            if supports_es_rounds and use_validation:
                fit_kwargs["early_stopping_rounds"] = 50
            model.fit(X_train, y_train, **fit_kwargs)
        else:
            raise

    evals_result = getattr(model, "evals_result", lambda: {})()
    return model, evals_result


def predict_xgboost(model, X: pd.DataFrame) -> np.ndarray:
    if model is None or X is None or len(X) == 0:
        return np.array([])
    X = np.asarray(X, dtype=np.float32)
    return model.predict(X)


def walk_forward_daily_ml(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str = "Demand",
    days: int = 7,
    horizon: int = 24,
    model_params: Optional[Dict[str, float]] = None,
):
    if df is None or df.empty or target not in df.columns:
        return (
            pd.DataFrame(columns=["day_idx", "timestamp", "y_true", "y_pred", "model", "model_name"]),
            pd.DataFrame(columns=["day_idx", "MAE", "RMSE", "nRMSE", "model_name"]),
        )

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", target])
    data = data.sort_values("timestamp")
    data[feature_cols] = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=feature_cols + [target])

    unique_days = np.array(sorted(data["timestamp"].dt.normalize().unique()))
    selected_days = unique_days[-days:]

    predictions = []
    metrics_rows = []

    for idx, day_start in enumerate(selected_days, start=1):
        day_start = pd.Timestamp(day_start)
        day_end = day_start + pd.Timedelta(days=1)

        train_mask = data["timestamp"] < day_start
        test_mask = (data["timestamp"] >= day_start) & (data["timestamp"] < day_end)

        train_df = data.loc[train_mask]
        test_df = data.loc[test_mask]

        if len(train_df) < horizon or len(test_df) < horizon:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test = test_df[feature_cols]
        y_test = test_df[target]

        model, _ = train_xgboost(X_train, y_train, params=model_params)
        y_pred = predict_xgboost(model, X_test)
        timestamps = test_df["timestamp"].values
        min_len = min(len(y_test), len(y_pred), len(timestamps))
        if min_len == 0:
            continue
        y_true_vals = y_test.values[:min_len]
        y_pred_vals = y_pred[:min_len]
        ts_vals = timestamps[:min_len]

        metrics = evaluate_forecast(y_true_vals, y_pred_vals)
        metrics_rows.append({"day_idx": idx, **metrics, "model_name": "XGBoost"})

        pred_df = pd.DataFrame(
            {
                "day_idx": idx,
                "timestamp": ts_vals.tolist(),
                "y_true": y_true_vals.tolist(),
                "y_pred": y_pred_vals.tolist(),
                "model_name": "XGBoost",
            }
        )
        predictions.append(pred_df)

    predictions_df = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=["day_idx", "timestamp", "y_true", "y_pred", "model_name"])
    )
    metrics_df = (
        pd.DataFrame(metrics_rows)
        if metrics_rows
        else pd.DataFrame(columns=["day_idx", "MAE", "RMSE", "nRMSE", "model_name"])
    )
    return predictions_df, metrics_df


__all__ = [
    "build_ml_dataset",
    "train_xgboost",
    "predict_xgboost",
    "evaluate_forecast",
    "walk_forward_daily_ml",
]
