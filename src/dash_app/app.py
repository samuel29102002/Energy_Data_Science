from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dash import Dash

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "train_252145.csv"
REPORTS_PATH = ROOT / "reports" / "tables"


def _load_plotly_template() -> None:
    template_path = Path(__file__).resolve().parent / "assets" / "plotly_template.json"
    if not template_path.exists():
        return
    with template_path.open() as fp:
        template_data = json.load(fp)
    template_name = "energy_dashboard"
    pio.templates[template_name] = template_data
    pio.templates.default = template_name
    px.defaults.template = template_name
    px.defaults.color_discrete_sequence = template_data["layout"].get("colorway", [])


def _safe_read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except FileNotFoundError:
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    for col in (cols or result.columns):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _difference_for_acf(series: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> pd.Series:
    diffed = series.copy()
    d = order[1]
    if d > 0:
        diffed = diffed.diff(d)
    D = seasonal_order[1]
    s = seasonal_order[3]
    if D > 0 and s > 0:
        for _ in range(D):
            diffed = diffed - diffed.shift(s)
    return diffed.dropna()


def build_context() -> SimpleNamespace:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.impute import KNNImputer
    from sklearn.inspection import permutation_importance
    from statsmodels.tsa.seasonal import STL

    from src.modeling_stats import acf_pacf

    ctx = SimpleNamespace()

    raw_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    df = raw_df.set_index("timestamp")
    df["timestamp"] = df.index
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.day_name()
    df["is_weekend"] = df.index.dayofweek >= 5
    df["date"] = df.index.date

    ctx.df = df

    numeric_options = [
        {"label": "PV (kW)", "value": "pv"},
        {"label": "Demand (kW)", "value": "Demand"},
        {"label": "Price (€/kWh)", "value": "Price"},
        {"label": "Temperature (°C)", "value": "Temperature"},
    ]
    label_lookup = {opt["value"]: opt["label"] for opt in numeric_options}
    ctx.numeric_options = numeric_options
    ctx.label_lookup = label_lookup
    ctx.default_series = ["pv", "Demand"]
    ctx.date_min = df["timestamp"].min().date()
    ctx.date_max = df["timestamp"].max().date()
    ctx.day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ctx.month_order = list(pd.date_range("2023-01-01", periods=12, freq="MS").month_name())

    colorway = ["#1F77B4", "#2CA02C", "#F9A825", "#6E7781", "#9467BD", "#17BECF"]
    series_colors = {
        "pv": "#F9A825",
        "Demand": "#2CA02C",
        "Price": "#1F77B4",
        "Temperature": "#9467BD",
        "pv_mod1": "#F9A825",
        "pv_mod1_simple": "#E07A5F",
        "pv_mod1_univariate": "#3D405B",
        "pv_mod1_multivariate": "#81B29A",
    }
    colors = {
        "demand": series_colors.get("Demand", "#2CA02C"),
        "pv": series_colors.get("pv", "#F9A825"),
        "price": series_colors.get("Price", "#1F77B4"),
        "temperature": series_colors.get("Temperature", "#9467BD"),
    }
    ctx.colorway = colorway
    ctx.series_colors = series_colors
    ctx.colors = colors

    demand_series = pd.to_numeric(df["Demand"], errors="coerce")
    hourly_demand = demand_series.resample("H").mean()
    hourly_demand = hourly_demand.interpolate(method="time", limit_direction="both").dropna()

    stl_daily = STL(hourly_demand, period=24, robust=True).fit()
    stl_weekly = STL(hourly_demand, period=168, seasonal=35, robust=True).fit()
    daily_avg = hourly_demand.resample("D").mean()
    stl_annual = STL(daily_avg, period=365, seasonal=31, robust=True).fit()

    def seasonality_strength(residual: pd.Series, component: pd.Series) -> float:
        resid_var = float(np.nanvar(residual))
        combined_var = float(np.nanvar(residual + component))
        if np.isclose(combined_var, 0.0):
            return float("nan")
        return float(np.clip(1.0 - resid_var / combined_var, 0.0, 1.0))

    seasonality_strength_df = pd.DataFrame(
        [
            {"period": "24h", "strength_type": "Seasonal", "value": seasonality_strength(stl_daily.resid, stl_daily.seasonal)},
            {"period": "24h", "strength_type": "Trend", "value": seasonality_strength(stl_daily.resid, stl_daily.trend)},
            {"period": "7d", "strength_type": "Seasonal", "value": seasonality_strength(stl_weekly.resid, stl_weekly.seasonal)},
            {"period": "7d", "strength_type": "Trend", "value": seasonality_strength(stl_weekly.resid, stl_weekly.trend)},
            {"period": "365d", "strength_type": "Seasonal", "value": seasonality_strength(stl_annual.resid, stl_annual.seasonal)},
            {"period": "365d", "strength_type": "Trend", "value": seasonality_strength(stl_annual.resid, stl_annual.trend)},
        ]
    )
    seasonality_strength_path = REPORTS_PATH / "seasonality_strength.csv"
    if seasonality_strength_path.exists():
        try:
            file_df = pd.read_csv(seasonality_strength_path)
            if {"period", "strength_type", "value"}.issubset(file_df.columns):
                seasonality_strength_df = file_df.copy()
        except Exception:
            pass
    seasonality_strength_df["value"] = pd.to_numeric(seasonality_strength_df["value"], errors="coerce").round(3)

    ctx.seasonality_strength_records = seasonality_strength_df.to_dict("records")
    ctx.decomposition_components = {
        "daily": pd.DataFrame({
            "trend": stl_daily.trend,
            "seasonal": stl_daily.seasonal,
            "resid": stl_daily.resid,
        }),
        "weekly": pd.DataFrame({
            "trend": stl_weekly.trend,
            "seasonal": stl_weekly.seasonal,
            "resid": stl_weekly.resid,
        }),
    }
    ctx.decomposition_labels = {"daily": "Daily (24h)", "weekly": "Weekly (168h)"}
    ctx.decomposition_period_filter = {"daily": "24h", "weekly": "7d"}
    ctx.demand_series = hourly_demand

    stat_model_configs = {
        "ARIMA(2,1,2)": {"order": (2, 1, 2), "seasonal_order": (0, 0, 0, 0)},
        "SARIMA(1,1,1)(1,1,1,24)": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 24)},
        "SARIMA(2,1,1)(0,1,1,24)": {"order": (2, 1, 1), "seasonal_order": (0, 1, 1, 24)},
    }
    ctx.stat_model_configs = stat_model_configs
    ctx.stat_model_options = [{"label": name, "value": name} for name in stat_model_configs]

    stat_model_acf: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name, cfg in stat_model_configs.items():
        diffed_series = _difference_for_acf(hourly_demand, cfg["order"], cfg["seasonal_order"])
        try:
            stat_model_acf[name] = acf_pacf(diffed_series, nlags=72)
        except Exception:
            stat_model_acf[name] = {
                "acf": pd.DataFrame(columns=["lag", "value"]),
                "pacf": pd.DataFrame(columns=["lag", "value"]),
            }
    ctx.stat_model_acf = stat_model_acf

    model_metrics_df = _safe_read_csv(REPORTS_PATH / "model_candidates_metrics.csv")
    walkforward_summary_df = _safe_read_csv(REPORTS_PATH / "walkforward_metrics_summary.csv")
    if not walkforward_summary_df.empty and "evaluation" not in walkforward_summary_df.columns:
        walkforward_summary_df["evaluation"] = "Walk-forward mean"
    walkforward_metrics_df = _safe_read_csv(REPORTS_PATH / "walkforward_per_day_metrics.csv")
    single_split_predictions_df = _safe_read_csv(REPORTS_PATH / "stats_single_split_predictions.csv", parse_dates=["timestamp"])
    walkforward_predictions_df = _safe_read_csv(REPORTS_PATH / "walkforward_predictions.csv", parse_dates=["timestamp"])

    stat_model_metrics = pd.concat([model_metrics_df, walkforward_summary_df], ignore_index=True, sort=False)
    if "evaluation" not in stat_model_metrics.columns:
        stat_model_metrics["evaluation"] = "Whole-train split"

    ctx.stat_model_metrics = stat_model_metrics
    ctx.stat_walkforward_metrics = walkforward_metrics_df
    ctx.stat_single_split_predictions = single_split_predictions_df
    ctx.stat_walkforward_predictions = walkforward_predictions_df

    ml_split_metrics_df = _safe_read_csv(REPORTS_PATH / "ml_split_metrics.csv")
    ml_walkforward_metrics_df = _safe_read_csv(REPORTS_PATH / "ml_walkforward_per_day_metrics.csv")
    ml_walkforward_predictions_df = _safe_read_csv(REPORTS_PATH / "ml_walkforward_predictions.csv", parse_dates=["timestamp"])
    ml_feature_importance_df = _safe_read_csv(REPORTS_PATH / "ml_feature_importance.csv")
    ml_learning_curve_df = _safe_read_csv(REPORTS_PATH / "ml_learning_curve.csv")
    ml_split_predictions_df = _safe_read_csv(REPORTS_PATH / "ml_split_predictions.csv", parse_dates=["timestamp"])
    best_stat_vs_ml_metrics_df = _safe_read_csv(REPORTS_PATH / "best_stat_vs_ml_metrics.csv")

    ctx.ml_split_metrics = ml_split_metrics_df
    ctx.ml_walkforward_metrics = ml_walkforward_metrics_df
    ctx.ml_walkforward_predictions = ml_walkforward_predictions_df
    ctx.ml_feature_importance = ml_feature_importance_df
    ctx.ml_learning_curve = ml_learning_curve_df
    ctx.ml_split_predictions = ml_split_predictions_df
    ctx.best_stat_vs_ml = best_stat_vs_ml_metrics_df
    ctx.best_stat_model = (
        model_metrics_df.sort_values("nRMSE").iloc[0]["model_name"]
        if not model_metrics_df.empty and "nRMSE" in model_metrics_df.columns
        else None
    )

    forecast_predictions_df = _safe_read_csv(REPORTS_PATH / "forecast_predictions.csv", parse_dates=["timestamp"])
    forecast_metrics_day_df = _safe_read_csv(REPORTS_PATH / "forecast_metrics_per_day.csv")
    forecast_metrics_summary_df = _safe_read_csv(REPORTS_PATH / "forecast_metrics_summary.csv")

    ctx.forecast_predictions = forecast_predictions_df
    ctx.forecast_metrics_day = forecast_metrics_day_df
    ctx.forecast_metrics_summary = forecast_metrics_summary_df

    pv_mod1 = pd.to_numeric(df.get("pv_mod1"), errors="coerce")
    pv_simple = pv_mod1.interpolate(method="time")
    stl = STL(pv_mod1, period=24, robust=True)
    stl_result = stl.fit()
    seasonal = stl_result.seasonal
    trend = stl_result.trend
    resid = pv_mod1 - seasonal - trend
    resid_interp = resid.interpolate(method="time")
    pv_univariate = seasonal + trend + resid_interp

    knn_features = df[["pv_mod2", "pv_mod3", "Shortwave_radiation (W/m²)", "Temperature"]].copy()
    knn_features["pv_mod1"] = pv_mod1
    knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
    knn_array = knn_imputer.fit_transform(knn_features)
    pv_multivariate = pd.Series(knn_array[:, -1], index=df.index, name="pv_mod1_knn")

    pv_imputed = pd.DataFrame(
        {
            "original": pv_mod1,
            "simple": pv_simple,
            "univariate": pv_univariate,
            "multivariate": pv_multivariate,
        }
    )
    pv_imputed = _coerce_numeric(pv_imputed)
    imputation_summary = pv_imputed.agg(["mean", "std", "min", "max"]).T
    imputation_summary["% missing"] = [pv_mod1.isna().mean() * 100, 0.0, 0.0, 0.0]
    imputation_summary = imputation_summary.round({"mean": 3, "std": 3, "min": 3, "max": 3, "% missing": 2})
    imputation_summary.reset_index(inplace=True)
    imputation_summary.rename(columns={"index": "series"}, inplace=True)

    ctx.pv_imputed = pv_imputed
    ctx.imputation_summary = imputation_summary
    ctx.imputation_summary_records = imputation_summary.to_dict("records")
    ctx.imputation_options = [
        {"label": "Original (raw)", "value": "original"},
        {"label": "Simple interpolation", "value": "simple"},
        {"label": "Seasonal (STL)", "value": "univariate"},
        {"label": "KNN multivariate", "value": "multivariate"},
    ]
    ctx.imputation_labels = {opt["value"]: opt["label"] for opt in ctx.imputation_options}

    features_df = df.copy()
    features_df['hour'] = features_df.index.hour
    features_df['dayofweek'] = features_df.index.dayofweek
    features_df['is_weekend'] = (features_df['dayofweek'] >= 5).astype(int)
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    features_df['cooling_degree'] = np.clip(features_df['Temperature'] - 18, 0, None)
    features_df['heating_degree'] = np.clip(18 - features_df['Temperature'], 0, None)
    features_df['temp_irradiance_interaction'] = features_df['Temperature'] * features_df['Shortwave_radiation (W/m²)']

    feature_columns = [
        'Demand', 'hour_sin', 'hour_cos', 'is_weekend', 'cooling_degree', 'heating_degree',
        'temp_irradiance_interaction', 'Temperature', 'Pressure (hPa)', 'Cloud_cover (%)',
        'Wind_speed_10m (km/h)', 'Shortwave_radiation (W/m²)', 'direct_radiation (W/m²)',
        'diffuse_radiation (W/m²)', 'direct_normal_irradiance (W/m²)', 'Price'
    ]
    features_df = features_df[feature_columns]
    features_df = _coerce_numeric(features_df, feature_columns)
    features_df = features_df.dropna(how="all")
    engineered_only = [
        feat for feat in ['hour_sin', 'hour_cos', 'is_weekend', 'cooling_degree', 'heating_degree', 'temp_irradiance_interaction']
        if feat in features_df.columns
    ]

    X_all = features_df.drop(columns=['Demand']).apply(pd.to_numeric, errors='coerce')
    X_all = X_all.dropna(axis=1, how='all')
    y_all = pd.to_numeric(features_df['Demand'], errors='coerce')
    valid_rows = X_all.dropna().index.intersection(y_all.dropna().index)
    if not valid_rows.empty:
        X_valid = X_all.loc[valid_rows]
        y_valid = y_all.loc[valid_rows]
        mi_all = mutual_info_regression(X_valid, y_valid, random_state=42)
        mi_all_series = pd.Series(mi_all, index=X_valid.columns, name='mutual_information').sort_values(ascending=False)
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_model.fit(X_valid, y_valid)
        perm_all = permutation_importance(rf_model, X_valid, y_valid, n_repeats=5, random_state=42, n_jobs=-1)
        perm_all_series = pd.Series(perm_all.importances_mean, index=X_valid.columns, name='permutation').sort_values(ascending=False)
    else:
        mi_all_series = pd.Series(dtype=float)
        perm_all_series = pd.Series(dtype=float)

    ctx.features_df = features_df
    ctx.ranking_data = {'mi': mi_all_series, 'permutation': perm_all_series}
    ctx.feature_options = [{'label': col, 'value': col} for col in X_all.columns.tolist()]
    ctx.engineered_only = engineered_only

    ctx.summary_stats = {
        "pv_total_mwh": df["pv"].sum() / 1000,
        "demand_total_mwh": df["Demand"].sum() / 1000,
        "avg_price": df["Price"].mean(),
        "peak_demand": df["Demand"].max(),
        "last_timestamp": df.index.max(),
    }

    ctx.safe_read_csv = _safe_read_csv
    ctx.coerce_numeric = _coerce_numeric

    return ctx


_load_plotly_template()
context = build_context()

from .layout import create_layout  # noqa: E402
from .callbacks import register_callbacks  # noqa: E402

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Energy Saving Plan",
)
app.layout = create_layout(context)
register_callbacks(app, context)

server = app.server


if __name__ == "__main__":
    app.run_server(debug=True)
