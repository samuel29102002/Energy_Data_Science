from __future__ import annotations

import random
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from packaging import version
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


__all__ = [
    "set_seed",
    "build_ml_dataset",
    "train_xgboost",
    "train_regressor_with_fallback",
    "predict_any",
    "evaluate_forecast",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_ml_dataset(
    df: pd.DataFrame,
    target: str = "Demand",
    feature_cols: Optional[Sequence[str]] = None,
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")

    data = df.copy()
    if "timestamp" not in data.columns:
        if data.index.name == "timestamp" or np.issubdtype(data.index.dtype, np.datetime64):
            data = data.reset_index().rename(columns={"index": "timestamp"})
        else:
            raise ValueError("Dataframe must contain a 'timestamp' column")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")

    if feature_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target]
    else:
        missing = set(feature_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found")

    selected = list(feature_cols) + [target]
    data[selected] = data[selected].apply(pd.to_numeric, errors="coerce")
    if dropna:
        data = data.dropna(subset=selected)

    X = data[feature_cols].to_numpy(dtype=np.float32)
    y = data[target].to_numpy(dtype=np.float32)
    idx = pd.Index(data["timestamp"].to_numpy(), name="timestamp")

    if not (len(X) == len(y) == len(idx)):
        raise ValueError("Feature matrix, target vector, and index must be the same length")

    return X, y, idx


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    if X_train is None or y_train is None or len(X_train) == 0:
        raise ValueError("Training data is empty")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()

    used_val = X_val is not None and y_val is not None and len(X_val) > 0
    if used_val:
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32).ravel()

    defaults: Dict[str, Any] = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        objective="reg:squarederror",
    )
    params = params or {}
    full_params = {**defaults, **params}
    model = xgb.XGBRegressor(**full_params)

    if "eval_metric" not in full_params:
        model.set_params(eval_metric="rmse")

    fit_kwargs: Dict[str, Any] = {}
    if used_val:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train)
        fit_kwargs = {}

    evals_result: Dict[str, Any] = {}
    if used_val and hasattr(model, "evals_result_"):
        evals_result = model.evals_result_

    return model, evals_result


def train_regressor_with_fallback(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[Tuple[Any, Dict[str, Any]], str]:
    try:
        model, history = train_xgboost(X_train, y_train, X_val=X_val, y_val=y_val, params=params, seed=seed)
        return (model, history), "xgboost"
    except Exception:
        rf = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).ravel()
        rf.fit(X_train, y_train)
        return (rf, {}), "random_forest"


def predict_any(model: Any, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return model.predict(X)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "nRMSE": np.nan}
    y_true = y_true[:m]
    y_pred = y_pred[:m]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.max(y_true) - np.min(y_true))
    nrmse = rmse / denom if not np.isclose(denom, 0.0) else np.nan
    return {"MAE": mae, "RMSE": rmse, "nRMSE": nrmse}
