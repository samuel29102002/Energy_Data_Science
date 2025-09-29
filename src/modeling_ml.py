from __future__ import annotations

import random
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb


ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]
IndexLike = Union[pd.Index, Sequence]


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def build_ml_dataset(
    df: pd.DataFrame,
    target: str = "Demand",
    feature_cols: Optional[Sequence[str]] = None,
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """Prepare feature matrix, target vector, and aligned timestamps for modelling.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe containing a timestamp column (or index) and features.
    target: str, default "Demand"
        Name of the target column.
    feature_cols: sequence of str, optional
        Explicit feature columns to use. If None, numeric columns (except timestamp/target)
        are selected automatically.
    dropna: bool, default True
        Whether to drop rows containing NA in selected columns.
    """

    if df is None or df.empty:
        raise ValueError("Input dataframe must not be empty")

    data = df.copy()
    if "timestamp" not in data.columns:
        if data.index.name == "timestamp" or np.issubdtype(data.index.dtype, np.datetime64):
            data = data.reset_index()
        else:
            data = data.reset_index(drop=False)

        if "timestamp" not in data.columns:
            rename_map = {data.columns[0]: "timestamp"}
            data = data.rename(columns=rename_map)

        if "timestamp" not in data.columns:
            raise ValueError("Dataframe must include a 'timestamp' column or index")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    data.reset_index(drop=True, inplace=True)

    if target not in data.columns and target.lower() in data.columns:
        target = target.lower()
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    if feature_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target]
    else:
        missing = [col for col in feature_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        feature_cols = list(feature_cols)

    selected_cols = feature_cols + [target]
    data[selected_cols] = data[selected_cols].apply(pd.to_numeric, errors="coerce")

    if dropna:
        data = data.dropna(subset=selected_cols)

    idx = pd.Index(data["timestamp"].to_numpy(), name="timestamp")
    X = data[feature_cols].to_numpy(dtype=np.float32)
    y = data[target].to_numpy(dtype=np.float32)

    if not (len(X) == len(y) == len(idx)):
        raise ValueError("Feature matrix, target vector, and index must share the same length")

    return X, y, idx


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Train an XGBoost regressor with robust, version-agnostic defaults."""

    if X_train is None or y_train is None or len(X_train) == 0:
        raise ValueError("Training data must not be empty")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()

    fit_kwargs: Dict[str, Any] = {}
    used_val = False

    if X_val is not None and y_val is not None:
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32).ravel()
        if len(X_val) != len(y_val):
            raise ValueError("Validation feature matrix and target must have equal length")
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]
        used_val = True

    defaults: Dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "tree_method": "hist",
        "objective": "reg:squarederror",
    }
    full_params = {**defaults, **(params or {})}

    model = xgb.XGBRegressor(**full_params)
    if "eval_metric" not in full_params:
        model.set_params(eval_metric="rmse")

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train)
        used_val = False

    eval_history: Dict[str, Any] = {}
    if used_val and hasattr(model, "evals_result_"):
        eval_history = getattr(model, "evals_result_", {})

    return model, eval_history


def predict_xgboost(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Return predictions from a trained XGBoost regressor."""

    X = np.asarray(X, dtype=np.float32)
    return model.predict(X)


def evaluate_forecast(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Compute MAE, RMSE, and normalised RMSE."""

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    m = min(len(y_true_arr), len(y_pred_arr))
    if m == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "nRMSE": float("nan")}

    y_true_arr = y_true_arr[:m]
    y_pred_arr = y_pred_arr[:m]

    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    denom = float(np.max(y_true_arr) - np.min(y_true_arr))
    nrmse = float(rmse / denom) if denom > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "nRMSE": nrmse}


def align_by_index(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    idx_true: IndexLike,
    idx_pred: IndexLike,
) -> Tuple[pd.Index, pd.Series, pd.Series]:
    """Align two series by their indices and return the intersection.

    Parameters
    ----------
    y_true, y_pred: array-like
        Arrays of true and predicted values.
    idx_true, idx_pred: sequence or pd.Index
        Indices associated with the true and predicted series.
    """

    true_index = pd.Index(idx_true, name="timestamp")
    pred_index = pd.Index(idx_pred, name="timestamp")

    s_true = pd.Series(np.asarray(y_true, dtype=np.float64), index=true_index)
    s_pred = pd.Series(np.asarray(y_pred, dtype=np.float64), index=pred_index)

    common_index = true_index.intersection(pred_index)
    s_true = s_true.loc[common_index].astype(float)
    s_pred = s_pred.loc[common_index].astype(float)

    return common_index, s_true, s_pred


__all__ = [
    "set_seed",
    "build_ml_dataset",
    "train_xgboost",
    "predict_xgboost",
    "evaluate_forecast",
    "align_by_index",
]
