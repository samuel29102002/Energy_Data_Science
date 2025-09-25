from __future__ import annotations

from plotly.subplots import make_subplots

from pathlib import Path
from typing import Iterable, List
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, dcc, html, dash_table
import dash_bootstrap_components as dbc
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
DATA_PATH = ROOT / "data" / "raw" / "train_252145.csv"

from src.plotting import (
    plot_stl_components,
    plot_typical_profiles_weekday_weekend,
    plot_typical_profiles_monthly,
)

# Load and enrich the dataset once at startup.
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
df = df.set_index("timestamp")
df["timestamp"] = df.index
df["hour"] = df.index.hour
df["day_of_week"] = df.index.day_name()
df["is_weekend"] = df.index.dayofweek >= 5
df["date"] = df.index.date

NUMERIC_OPTIONS = [
    {"label": "PV (kW)", "value": "pv"},
    {"label": "Demand (kW)", "value": "Demand"},
    {"label": "Price (€/kWh)", "value": "Price"},
    {"label": "Temperature (°C)", "value": "Temperature"},
]
LABEL_LOOKUP = {opt["value"]: opt["label"] for opt in NUMERIC_OPTIONS}
DEFAULT_SERIES = ["pv", "Demand"]
DATE_MIN = df["timestamp"].min().date()
DATE_MAX = df["timestamp"].max().date()
DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTH_ORDER = list(pd.date_range("2023-01-01", periods=12, freq="MS").month_name())

COLORWAY = ["#FFA500", "#2ca02c", "#1f77b4", "#7f7f7f", "#FF7F0E"]
SERIES_COLORS = {
    "pv": "#FFA500",
    "Demand": "#2ca02c",
    "Price": "#1f77b4",
    "Temperature": "#FF7F0E",
    "pv_mod1": "#FFA500",
    "pv_mod1_simple": "#E07A5F",
    "pv_mod1_univariate": "#3D405B",
    "pv_mod1_multivariate": "#81B29A",
}

COLORS = {
    "demand": SERIES_COLORS.get("Demand", "#2ca02c"),
    "pv": SERIES_COLORS.get("pv", "#FFA500"),
    "price": SERIES_COLORS.get("Price", "#1f77b4"),
    "temperature": SERIES_COLORS.get("Temperature", "#FF7F0E"),
}

PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Open Sans, sans-serif", color="#212529"),
        colorway=COLORWAY,
        title=dict(font=dict(color="#212529", size=20)),
        xaxis=dict(gridcolor="#e9ecef", zerolinecolor="#e9ecef"),
        yaxis=dict(gridcolor="#e9ecef", zerolinecolor="#e9ecef"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#dee2e6", borderwidth=1),
        hoverlabel=dict(bgcolor="#f8f9fa", font=dict(color="#212529")),
    )
)
pio.templates["energy_light"] = PLOTLY_TEMPLATE
pio.templates.default = "energy_light"
px.defaults.template = "energy_light"
px.defaults.color_discrete_sequence = COLORWAY


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str] | None = None) -> pd.DataFrame:
    """Return a copy with selected columns coerced to numeric (invalid values become NaN)."""
    result = df.copy()
    if cols is None:
        cols = result.columns
    for col in cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def run_stl_decomposition(series: pd.Series, period: int, seasonal: int | None = None) -> tuple[STL, pd.DataFrame]:
    """Compute STL decomposition with optional smoother length."""
    kwargs = {} if seasonal is None else {"seasonal": seasonal}
    stl_instance = STL(series, period=period, robust=True, **kwargs)
    result = stl_instance.fit()
    components = pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        },
        index=series.index,
    )
    return result, components


def seasonality_strength(residual: pd.Series, component: pd.Series) -> float:
    resid_var = float(np.nanvar(residual))
    combined_var = float(np.nanvar(residual + component))
    if np.isclose(combined_var, 0.0):
        return float("nan")
    strength = 1.0 - resid_var / combined_var
    return float(np.clip(strength, 0.0, 1.0))

# Pre-compute PV_mod1 imputations for cleaning tab.
df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
hourly_demand_series = df["Demand"].resample("H").mean()
hourly_interpolated_points = int(hourly_demand_series.isna().sum())
hourly_demand_series = hourly_demand_series.interpolate(method="time", limit_direction="both").dropna()

stl_daily_result, stl_daily_components = run_stl_decomposition(hourly_demand_series, period=24)
stl_weekly_result, stl_weekly_components = run_stl_decomposition(hourly_demand_series, period=168, seasonal=35)
daily_avg_demand_series = hourly_demand_series.resample("D").mean()
stl_annual_result, _ = run_stl_decomposition(daily_avg_demand_series, period=365, seasonal=31)

seasonality_strength_df = pd.DataFrame(
    [
        {"period": "24h", "strength_type": "Seasonal", "value": seasonality_strength(stl_daily_result.resid, stl_daily_result.seasonal)},
        {"period": "24h", "strength_type": "Trend", "value": seasonality_strength(stl_daily_result.resid, stl_daily_result.trend)},
        {"period": "7d", "strength_type": "Seasonal", "value": seasonality_strength(stl_weekly_result.resid, stl_weekly_result.seasonal)},
        {"period": "7d", "strength_type": "Trend", "value": seasonality_strength(stl_weekly_result.resid, stl_weekly_result.trend)},
        {"period": "365d", "strength_type": "Seasonal", "value": seasonality_strength(stl_annual_result.resid, stl_annual_result.seasonal)},
        {"period": "365d", "strength_type": "Trend", "value": seasonality_strength(stl_annual_result.resid, stl_annual_result.trend)},
    ]
)
seasonality_strength_df["value"] = pd.to_numeric(seasonality_strength_df["value"], errors="coerce").round(3)

seasonality_strength_path = ROOT / "reports" / "tables" / "seasonality_strength.csv"
if seasonality_strength_path.exists():
    try:
        file_strength_df = pd.read_csv(seasonality_strength_path)
        if {"period", "strength_type", "value"}.issubset(file_strength_df.columns):
            seasonality_strength_df = file_strength_df.copy()
            seasonality_strength_df["value"] = pd.to_numeric(seasonality_strength_df["value"], errors="coerce").round(3)
    except Exception:  # pragma: no cover - defensive fallback
        pass

seasonality_strength_records = seasonality_strength_df.to_dict("records")
DECOMPOSITION_COMPONENTS = {
    "daily": stl_daily_components,
    "weekly": stl_weekly_components,
}
DECOMPOSITION_LABELS = {
    "daily": "Daily (24h)",
    "weekly": "Weekly (168h)",
}
DECOMPOSITION_PERIOD_FILTER = {
    "daily": "24h",
    "weekly": "7d",
}

DEMAND_SERIES = hourly_demand_series

# Pre-compute PV_mod1 imputations for cleaning tab.
pv_mod1 = df["pv_mod1"]
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
pv_imputed = coerce_numeric(pv_imputed, pv_imputed.columns)

imputation_summary = pv_imputed.agg(["mean", "std", "min", "max"]).T
imputation_summary["% missing"] = [pv_mod1.isna().mean() * 100, 0.0, 0.0, 0.0]
imputation_summary = imputation_summary.round({"mean": 3, "std": 3, "min": 3, "max": 3, "% missing": 2})
imputation_summary.reset_index(inplace=True)
imputation_summary.rename(columns={"index": "series"}, inplace=True)

IMPUTATION_OPTIONS = [
    {"label": "Original (raw)", "value": "original"},
    {"label": "Simple interpolation", "value": "simple"},
    {"label": "Seasonal (STL)", "value": "univariate"},
    {"label": "KNN multivariate", "value": "multivariate"},
]

imputation_summary_records = imputation_summary.to_dict('records')


# Feature engineering dataset
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
features_df = coerce_numeric(features_df, feature_columns)
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
    X_valid = pd.DataFrame(columns=X_all.columns)
    y_valid = pd.Series(dtype=float)
    mi_all_series = pd.Series(dtype=float)
    perm_all_series = pd.Series(dtype=float)

ranking_data = {
    'mi': mi_all_series,
    'permutation': perm_all_series,
}

numeric_feature_columns = X_valid.columns.tolist() if not X_valid.empty else X_all.columns.tolist()
feature_options = [{'label': col, 'value': col} for col in numeric_feature_columns]

IMPUTATION_LABELS = {opt['value']: opt['label'] for opt in IMPUTATION_OPTIONS}


def _self_check() -> None:
    """Lightweight sanity checks for numeric consistency of ranking tables and PV imputations."""
    try:
        ranking_df = pd.concat(ranking_data.values(), axis=1)
        if not ranking_df.empty:
            for col in ranking_df.columns:
                series = pd.to_numeric(ranking_df[col], errors='coerce')
                if series.dropna().empty:
                    raise ValueError(f"Ranking column '{col}' has no numeric values")
        window = coerce_numeric(pv_imputed[['original', 'simple', 'univariate', 'multivariate']]).iloc[: 24 * 7]
        if window.dropna(how='all').empty:
            raise ValueError("PV imputation sample window is empty")
    except Exception as exc:  # pragma: no cover - logging only
        print(f"Dash app self-check warning: {exc}")


_self_check()

# Pre-compute summary statistics for the info cards.
summary_stats = {
    "pv_total_mwh": df["pv"].sum() / 1000,
    "demand_total_mwh": df["Demand"].sum() / 1000,
    "avg_price": df["Price"].mean(),
    "peak_demand": df["Demand"].max(),
}

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=False,
    title="HEMS Data Explorer",
)
server = app.server

card_style = {
    "border": "1px solid #dee2e6",
    "borderRadius": "0.75rem",
    "padding": "1rem",
    "backgroundColor": "#ffffff",
    "boxShadow": "0 0.5rem 1rem rgba(0,0,0,0.05)",
}

sidebar_style = {
    "backgroundColor": "#f8f9fa",
    "border": "1px solid #dee2e6",
    "borderRadius": "0.75rem",
    "padding": "1.5rem",
    "boxShadow": "0 0.25rem 0.5rem rgba(0,0,0,0.05)",
}

overview_layout = [
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [html.H6("Total PV yield", className="text-muted"), html.H3(f"{summary_stats['pv_total_mwh']:.1f} MWh")],
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [html.H6("Total demand", className="text-muted"), html.H3(f"{summary_stats['demand_total_mwh']:.1f} MWh")],
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [html.H6("Average price", className="text-muted"), html.H3(f"€{summary_stats['avg_price']:.03f}/kWh")],
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [html.H6("Peak demand", className="text-muted"), html.H3(f"{summary_stats['peak_demand']:.2f} kW")],
                    style=card_style,
                ),
                md=3,
            ),
        ],
        className="gy-3",
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Hr(),
                    html.H4("Controls", className="mb-3"),
                    dbc.Label("Date range"),
                    dcc.DatePickerRange(
                        id="date-picker",
                        display_format="YYYY-MM-DD",
                        start_date=DATE_MIN,
                        end_date=DATE_MAX,
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        className="mb-3",
                    ),
                    dbc.Label("Variables"),
                    dcc.Dropdown(
                        id="series-dropdown",
                        options=NUMERIC_OPTIONS,
                        value=DEFAULT_SERIES,
                        multi=True,
                        clearable=False,
                        className="mb-3",
                    ),
                    dbc.Label("Resampling"),
                    dcc.RadioItems(
                        id="aggregation-radio",
                        options=[
                            {"label": "Hourly", "value": "H"},
                            {"label": "Daily", "value": "D"},
                            {"label": "Weekly", "value": "W"},
                        ],
                        value="H",
                        inline=True,
                        className="mb-3",
                    ),
                ],
                md=3,
            ),
            dbc.Col(dcc.Graph(id="timeseries-chart", config={"displayModeBar": False, "displaylogo": False}), md=9),
        ],
        className="align-items-start my-4",
    ),
    dbc.Row(
        [
            dbc.Col(dcc.Graph(id="correlation-heatmap", config={"displayModeBar": False, "displaylogo": False}), md=6),
            dbc.Col(dcc.Graph(id="pv-demand-scatter", config={"displayModeBar": False, "displaylogo": False}), md=6),
        ],
        className="gy-4",
    ),
    dbc.Row(
        [dbc.Col(dcc.Graph(id="hourly-heatmap", config={"displayModeBar": False, "displaylogo": False}), md=12)],
        className="gy-4 mb-5",
    ),
]

visualisation_layout = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H5("Explore visualisations", className="fw-semibold mb-3"),
                    dbc.Label("Date range"),
                    dcc.DatePickerRange(
                        id="viz-date-picker",
                        display_format="YYYY-MM-DD",
                        start_date=DATE_MIN,
                        end_date=DATE_MAX,
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        className="mb-3",
                    ),
                    dbc.Label("Variable(s)"),
                    dcc.Dropdown(
                        id="viz-variable-dropdown",
                        options=NUMERIC_OPTIONS,
                        value=["pv"],
                        multi=True,
                        clearable=False,
                        className="mb-3",
                    ),
                    dbc.Label("Plot type"),
                    dcc.RadioItems(
                        id="viz-plot-radio",
                        options=[
                            {"label": "Timeseries", "value": "timeseries"},
                            {"label": "Distribution", "value": "distribution"},
                            {"label": "Boxplot (hour-of-day)", "value": "boxplot"},
                            {"label": "Correlation heatmap", "value": "heatmap"},
                            {"label": "Typical profiles", "value": "profiles"},
                        ],
                        value="timeseries",
                        className="mb-3",
                    ),
                ],
                md=3,
                style=sidebar_style,
            ),
            dbc.Col(
                dcc.Graph(id="viz-graph", config={"displayModeBar": True, "displaylogo": False}),
                md=9,
            ),
        ],
        className="gy-4",
    ),
]


decomposition_layout = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H5("Decomposition Controls", className="fw-semibold mb-3"),
                    dbc.Label("Date range"),
                    dcc.DatePickerRange(
                        id="decomp-date-picker",
                        display_format="YYYY-MM-DD",
                        start_date=DATE_MIN,
                        end_date=DATE_MAX,
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        className="mb-3",
                    ),
                    dbc.Label("Seasonality period"),
                    dcc.RadioItems(
                        id="decomp-period-radio",
                        options=[
                            {"label": "Daily (24h)", "value": "daily"},
                            {"label": "Weekly (168h)", "value": "weekly"},
                        ],
                        value="daily",
                        className="mb-3",
                    ),
                    dbc.Label("Profile view"),
                    dcc.Dropdown(
                        id="decomp-profile-dropdown",
                        options=[
                            {"label": "Weekday vs Weekend", "value": "weekday_weekend"},
                            {"label": "Monthly profiles", "value": "monthly"},
                        ],
                        value="weekday_weekend",
                        clearable=False,
                        className="mb-3",
                    ),
                ],
                md=3,
                style=sidebar_style,
            ),
            dbc.Col(
                [
                    dcc.Graph(id="decomp-components-figure", config={"displayModeBar": True, "displaylogo": False}),
                    dcc.Graph(id="decomp-profile-figure", config={"displayModeBar": True, "displaylogo": False}, className="mt-4"),
                    html.Div(
                        dash_table.DataTable(
                            id="decomp-strength-table",
                            columns=[
                                {"name": "Period", "id": "period"},
                                {"name": "Component", "id": "strength_type"},
                                {"name": "Strength", "id": "value"},
                            ],
                            data=seasonality_strength_records,
                            style_as_list_view=True,
                            style_table={
                                "overflowX": "auto",
                                "border": "1px solid #e6e6e6",
                                "borderRadius": "8px",
                            },
                            style_header={
                                "backgroundColor": "#F7F9FB",
                                "fontWeight": "600",
                                "borderBottom": "1px solid #d9d9d9",
                            },
                            style_cell={
                                "padding": "8px",
                                "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial",
                                "fontSize": 13,
                                "textAlign": "center",
                            },
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"},
                            ],
                            page_size=6,
                            fill_width=True,
                            export_format="csv",
                            export_headers="display",
                        ),
                        className="mt-4",
                    ),
                ],
                md=9,
            ),
        ],
        className="gy-4",
    ),
]


pv_cleaning_layout = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H5("PV Cleaning Controls", className="fw-semibold mb-3"),
                    dbc.Label("Date range"),
                    dcc.DatePickerRange(
                        id="pv-date-picker",
                        display_format="YYYY-MM-DD",
                        start_date=DATE_MIN,
                        end_date=DATE_MAX,
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        className="mb-3",
                    ),
                    dbc.Label("Imputation strategy"),
                    dcc.Dropdown(
                        id="pv-strategy-dropdown",
                        options=IMPUTATION_OPTIONS,
                        value="multivariate",
                        clearable=False,
                        className="mb-3",
                    ),
                ],
                md=3,
                style=sidebar_style,
            ),
            dbc.Col(
                [
                    dcc.Graph(id="pv-cleaning-chart", config={"displayModeBar": True, "displaylogo": False}),
                    html.Div(
                        dash_table.DataTable(
                            id="pv-summary-table",
                            columns=[{"name": c, "id": c} for c in imputation_summary.columns],
                            data=imputation_summary_records,
                            style_as_list_view=True,
                            style_table={
                                "overflowX": "auto",
                                "border": "1px solid #e6e6e6",
                                "borderRadius": "8px",
                            },
                            style_header={
                                "backgroundColor": "#F7F9FB",
                                "fontWeight": "600",
                                "borderBottom": "1px solid #d9d9d9",
                            },
                            style_cell={
                                "padding": "8px",
                                "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial",
                                "fontSize": 13,
                                "textAlign": "center",
                            },
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"},
                            ],
                            page_size=10,
                            fill_width=True,
                            export_format="csv",
                            export_headers="display",
                        ),
                        className="mt-4",
                    ),
                ],
                md=9,
            ),
        ],
        className="gy-4",
    ),
]


features_layout = [
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H5("Feature Analysis Controls", className="fw-semibold mb-3"),
                    dbc.Checklist(
                        id="features-include-weather",
                        options=[{"label": "Include weather features", "value": "weather"}],
                        value=["weather"],
                        switch=True,
                        className="mb-3",
                    ),
                    dbc.Label("Ranking method"),
                    dcc.RadioItems(
                        id="features-ranking-method",
                        options=[
                            {"label": "Mutual information", "value": "mi"},
                            {"label": "Permutation importance", "value": "permutation"},
                        ],
                        value="mi",
                        className="mb-3",
                    ),
                    dbc.Label("Distribution variable"),
                   dcc.Dropdown(
                       id="features-variable-dropdown",
                       options=feature_options,
                        value=feature_options[0]['value'] if feature_options else None,
                        clearable=False,
                        className="mb-3",
                    ),
                ],
                md=3,
                style=sidebar_style,
            ),
            dbc.Col(
                [
                    html.Div(
                        dash_table.DataTable(
                            id="features-ranking-table",
                            columns=[{"name": "Feature", "id": "feature"}, {"name": "Score", "id": "score"}],
                            data=[],
                            style_as_list_view=True,
                            style_table={
                                "overflowX": "auto",
                                "border": "1px solid #e6e6e6",
                                "borderRadius": "8px",
                            },
                            style_header={
                                "backgroundColor": "#F7F9FB",
                                "fontWeight": "600",
                                "borderBottom": "1px solid #d9d9d9",
                            },
                            style_cell={
                                "padding": "8px",
                                "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial",
                                "fontSize": 13,
                                "textAlign": "center",
                            },
                            style_data_conditional=[
                                {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"},
                            ],
                            page_size=15,
                            fill_width=True,
                            export_format="csv",
                            export_headers="display",
                        ),
                        className="features-ranking-table-wrapper mb-4",
                    ),
                    dcc.Graph(id="features-ranking-chart", config={"displayModeBar": False, "displaylogo": False}),
                    dcc.Graph(id="features-distribution", config={"displayModeBar": True, "displaylogo": False}, className="mt-4"),
                ],
                md=9,
            ),
        ],
        className="gy-4",
    ),
]

app.layout = dbc.Container(
    [
        html.H1("HEMS Data Explorer", className="text-center my-4"),
        dbc.Tabs(
            [
                dbc.Tab(overview_layout, label="Overview", tab_id="overview", tab_style={"padding": "0.75rem"}, active_tab_style={"backgroundColor": "#ffffff"}),
                dbc.Tab(visualisation_layout, label="Visualisation Studio", tab_id="visualisation", tab_style={"padding": "0.75rem"}, active_tab_style={"backgroundColor": "#ffffff"}),
                dbc.Tab(decomposition_layout, label="Decomposition", tab_id="decomposition", tab_style={"padding": "0.75rem"}, active_tab_style={"backgroundColor": "#ffffff"}),
                dbc.Tab(pv_cleaning_layout, label="PV Cleaning", tab_id="pv_cleaning", tab_style={"padding": "0.75rem"}, active_tab_style={"backgroundColor": "#ffffff"}),
                dbc.Tab(features_layout, label="Features", tab_id="features", tab_style={"padding": "0.75rem"}, active_tab_style={"backgroundColor": "#ffffff"}),
            ],
            id="main-tabs",
            active_tab="overview",
            className="mb-4",
        ),
    ],
    fluid=True,
    className="pb-5",
)


def _filter_range(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Return dataframe slice for the given date range."""
    if start_date is None or end_date is None:
        return df.copy()
    mask = (df["date"] >= pd.to_datetime(start_date).date()) & (df["date"] <= pd.to_datetime(end_date).date())
    subset = df.loc[mask]
    if subset.empty:
        return df.head(0).copy()
    return subset.copy()


def _slice_by_date_index(obj: pd.DataFrame | pd.Series, start_date: str | None, end_date: str | None) -> pd.DataFrame | pd.Series:
    if start_date is None and end_date is None:
        return obj.copy()
    start = pd.to_datetime(start_date).date() if start_date else obj.index.min().date()
    end = pd.to_datetime(end_date).date() if end_date else obj.index.max().date()
    mask = (obj.index.date >= start) & (obj.index.date <= end)
    return obj.loc[mask]


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": message, "x": 0.5},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
    )
    return fig


def _normalize_variables(values: Iterable[str] | str | None) -> List[str]:
    if values is None:
        return DEFAULT_SERIES
    if isinstance(values, str):
        return [values]
    if not values:
        return DEFAULT_SERIES
    return list(values)


@app.callback(
    Output("timeseries-chart", "figure"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
    Input("series-dropdown", "value"),
    Input("aggregation-radio", "value"),
)
def update_timeseries(start_date: str, end_date: str, series: Iterable[str], aggregation: str):
    series = _normalize_variables(series)
    filtered = _filter_range(start_date, end_date)
    if filtered.empty:
        return _empty_figure("No data in selected range")
    value_cols = [col for col in series if col in filtered.columns]
    if not value_cols:
        return _empty_figure("No data in selected range")
    numeric_frame = filtered[['timestamp'] + value_cols].copy()
    numeric_frame = coerce_numeric(numeric_frame, value_cols).dropna(subset=value_cols, how='all')
    if numeric_frame.empty:
        return _empty_figure("No data in selected range")
    resampled = (
        numeric_frame.set_index('timestamp')[value_cols]
        .resample(aggregation)
        .mean()
        .dropna(how='all')
        .reset_index()
    )
    if resampled.empty:
        return _empty_figure("No data in selected range")
    fig = go.Figure()
    added = False
    for idx, column in enumerate(series):
        if column not in resampled.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=resampled['timestamp'],
                y=resampled[column],
                name=LABEL_LOOKUP.get(column, column),
                mode="lines",
                line=dict(color=SERIES_COLORS.get(column, COLORWAY[idx % len(COLORWAY)]), width=2.2),
            )
        )
        added = True
    if not added:
        return _empty_figure("No data in selected range")
    fig.update_layout(title="Time series overview", xaxis_title="Timestamp", yaxis_title="Value")
    return fig


@app.callback(
    Output("correlation-heatmap", "figure"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
)
def update_correlation(start_date: str, end_date: str):
    filtered = _filter_range(start_date, end_date)
    if filtered.empty:
        return _empty_figure("No data in selected range")
    cols = ["pv", "Demand", "Price", "Temperature", "Wind_speed_10m (km/h)", "Shortwave_radiation (W/m²)"]
    numeric = coerce_numeric(filtered[cols], cols).dropna(how='all')
    if numeric.empty:
        return _empty_figure("No data in selected range")
    corr = numeric.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=cols,
            y=cols,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(title="Correlation heatmap")
    return fig


@app.callback(
    Output("pv-demand-scatter", "figure"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
)
def update_scatter(start_date: str, end_date: str):
    filtered = _filter_range(start_date, end_date)
    if filtered.empty:
        return _empty_figure("No data in selected range")
    numeric = coerce_numeric(filtered[['Demand', 'pv', 'Price']], ['Demand', 'pv', 'Price']).dropna()
    if numeric.empty:
        return _empty_figure("No data in selected range")
    sample = numeric.sample(n=min(3000, len(numeric)), random_state=7)
    fig = px.scatter(
        sample,
        x="Demand",
        y="pv",
        color="Price",
        title="PV vs demand coloured by price",
        labels={"Demand": "Demand (kW)", "pv": "PV (kW)", "Price": "Price (€/kWh)"},
        color_continuous_scale="Viridis",
    )
    fig.update_traces(mode="markers", marker=dict(size=7, opacity=0.7))
    fig.update_coloraxes(colorbar=dict(title="Price (€/kWh)"))
    return fig


@app.callback(
    Output("hourly-heatmap", "figure"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
)
def update_hourly_heatmap(start_date: str, end_date: str):
    filtered = _filter_range(start_date, end_date)
    if filtered.empty:
        return _empty_figure("No data in selected range")
    subset = filtered[["day_of_week", "hour", "pv"]].copy()
    subset = coerce_numeric(subset, ["pv"]).dropna(subset=["pv"])
    if subset.empty:
        return _empty_figure("No data in selected range")
    pivot = (
        subset.pivot_table(values="pv", index="day_of_week", columns="hour", aggfunc="mean")
        .reindex(DAY_ORDER)
        .fillna(0.0)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="YlGnBu",
            colorbar=dict(title="PV (kW)"),
        )
    )
    fig.update_layout(title="Average PV availability by day and hour", xaxis_title="Hour", yaxis_title="Day of week")
    return fig


@app.callback(
    Output("decomp-components-figure", "figure"),
    Output("decomp-profile-figure", "figure"),
    Output("decomp-strength-table", "data"),
    Output("decomp-strength-table", "style_data_conditional"),
    Input("decomp-period-radio", "value"),
    Input("decomp-profile-dropdown", "value"),
    Input("decomp-date-picker", "start_date"),
    Input("decomp-date-picker", "end_date"),
)
def update_decomposition_tab(period_value: str, profile_view: str, start_date: str, end_date: str):
    period_value = period_value or "daily"
    profile_view = profile_view or "weekday_weekend"

    components = DECOMPOSITION_COMPONENTS.get(period_value)
    if components is None or components.empty:
        components_fig = _empty_figure("No data for current selection")
    else:
        subset = _slice_by_date_index(components, start_date, end_date)
        if subset.empty:
            components_fig = _empty_figure("No data for current selection")
        else:
            label = DECOMPOSITION_LABELS.get(period_value, "").strip()
            title = f"Demand STL decomposition – {label}" if label else "Demand STL decomposition"
            components_fig = plot_stl_components(
                timestamps=subset.index,
                trend=subset["trend"],
                seasonal=subset["seasonal"],
                resid=subset["resid"],
                title=title,
                style="dashboard",
            )
            components_fig.update_layout(height=700)

    series_subset = _slice_by_date_index(DEMAND_SERIES, start_date, end_date)
    series_subset = pd.to_numeric(series_subset, errors="coerce").dropna()
    if series_subset.empty:
        profile_fig = _empty_figure("No data for current selection")
    else:
        profile_df = series_subset.to_frame(name="Demand")
        profile_df["hour"] = profile_df.index.hour
        profile_df["is_weekend"] = profile_df.index.dayofweek >= 5
        if profile_view == "monthly":
            profile_df["month"] = pd.Categorical(profile_df.index.month_name(), categories=MONTH_ORDER, ordered=True)
            monthly_profile = (
                profile_df.groupby(["month", "hour"])["Demand"].mean().reset_index().rename(columns={"Demand": "value"})
            )
            if monthly_profile.empty:
                profile_fig = _empty_figure("No data for current selection")
            else:
                profile_fig = plot_typical_profiles_monthly(
                    monthly_profile,
                    value_label="Demand (kW)",
                    style="dashboard",
                )
                profile_fig.update_layout(title="Monthly typical hourly demand")
        else:
            weekday_profile = profile_df.loc[~profile_df["is_weekend"]].groupby("hour")["Demand"].mean()
            weekend_profile = profile_df.loc[profile_df["is_weekend"]].groupby("hour")["Demand"].mean()
            if weekday_profile.dropna().empty or weekend_profile.dropna().empty:
                profile_fig = _empty_figure("No data for current selection")
            else:
                profile_fig = plot_typical_profiles_weekday_weekend(
                    weekday_profile,
                    weekend_profile,
                    value_label="Demand (kW)",
                    style="dashboard",
                )
                profile_fig.update_layout(title="Typical hourly demand – weekday vs weekend")

    table_df = seasonality_strength_df.copy()
    table_records = table_df.to_dict("records")
    highlight_period = DECOMPOSITION_PERIOD_FILTER.get(period_value)
    style_conditional = [
        {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"},
    ]
    if highlight_period:
        style_conditional.append(
            {"if": {"filter_query": f'{{period}} = "{highlight_period}"'}, "backgroundColor": "#fff3cd"}
        )

    return components_fig, profile_fig, table_records, style_conditional


@app.callback(
    Output("pv-cleaning-chart", "figure"),
    Input("pv-date-picker", "start_date"),
    Input("pv-date-picker", "end_date"),
    Input("pv-strategy-dropdown", "value"),
)
def update_pv_cleaning_chart(start_date: str, end_date: str, strategy: str):
    strategy = strategy or "multivariate"
    filtered = _filter_range(start_date, end_date)
    if filtered.empty:
        return _empty_figure("No data in selected range")
    subset = coerce_numeric(pv_imputed.loc[filtered.index], ['original', 'simple', 'univariate', 'multivariate'])
    subset = subset.sort_index()
    subset = subset.dropna(how='all')
    if subset.empty:
        return _empty_figure("No data in selected range")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=subset.index,
            y=subset["original"],
            name="Original",
            mode="lines",
            line=dict(color=SERIES_COLORS["pv_mod1"], width=1.6),
        )
    )
    if strategy in subset.columns and strategy != "original":
        fig.add_trace(
            go.Scatter(
                x=subset.index,
                y=subset[strategy],
                name=IMPUTATION_LABELS.get(strategy, strategy.title()),
                mode="lines",
                line=dict(color=SERIES_COLORS.get(f"pv_mod1_{strategy}", COLORWAY[0]), width=2.2),
            )
        )
    fig.update_layout(
        title="PV_mod1 cleaning comparison",
        xaxis_title="Timestamp",
        yaxis_title="PV_mod1 (kW)",
    )
    return fig



@app.callback(
    Output("features-ranking-table", "data"),
    Output("features-ranking-chart", "figure"),
    Output("features-variable-dropdown", "options"),
    Output("features-variable-dropdown", "value"),
    Input("features-include-weather", "value"),
    Input("features-ranking-method", "value"),
    State("features-variable-dropdown", "value"),
)
def update_feature_ranking(include_weather, method, current_variable):
    include_weather = include_weather or []
    method = method or "mi"
    available_features = list(ranking_data[method].index)
    if "weather" not in include_weather:
        filtered_features = [f for f in available_features if f in engineered_only]
    else:
        filtered_features = available_features
    options = [{'label': feat, 'value': feat} for feat in filtered_features]
    if not filtered_features:
        return [], _empty_figure("No features available"), options, None
    ranking_series = ranking_data[method].reindex(filtered_features).dropna()
    if ranking_series.empty:
        return [], _empty_figure("No features available"), options, None
    table_data = [
        {"feature": feature, "score": float(score)}
        for feature, score in ranking_series.items()
    ]
    top = ranking_series.head(15)
    fig = go.Figure(
        go.Bar(
            x=top.values[::-1],
            y=top.index[::-1],
            orientation="h",
            marker_color=COLORS['demand'],
        )
    )
    fig.update_layout(
        title=f"Top features ({'Mutual information' if method == 'mi' else 'Permutation importance'})",
        xaxis_title="Score",
        yaxis_title="Feature",
    )
    valid_values = [opt['value'] for opt in options]
    if current_variable not in valid_values:
        current_variable = valid_values[0] if valid_values else None
    return table_data, fig, options, current_variable


@app.callback(
    Output("features-distribution", "figure"),
    Input("features-variable-dropdown", "value"),
)
def update_feature_distribution(variable):
    if not variable or variable not in features_df.columns:
        return _empty_figure("Select a feature")
    series = pd.to_numeric(features_df[variable], errors='coerce').dropna()
    if series.empty:
        return _empty_figure("No data available")
    hist = go.Histogram(x=series, nbinsx=60, name=variable, marker_color=COLORS['demand'], opacity=0.7)
    qq_theoretical, qq_sample = stats.probplot(series, dist='norm')[0]
    qq = go.Scatter(x=qq_theoretical, y=qq_sample, mode='markers', name='Q-Q', marker=dict(color='#1f77b4', size=5, opacity=0.7))
    line = go.Scatter(x=qq_theoretical, y=qq_theoretical, mode='lines', name='45° line', line=dict(color='#aaa', dash='dash'))
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{variable} histogram', f'{variable} Q-Q plot'))
    fig.add_trace(hist, row=1, col=1)
    fig.add_trace(qq, row=1, col=2)
    fig.add_trace(line, row=1, col=2)
    fig.update_xaxes(title_text=variable, row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_xaxes(title_text='Theoretical quantiles', row=1, col=2)
    fig.update_yaxes(title_text='Sample quantiles', row=1, col=2)
    fig.update_layout(showlegend=False)
    return fig


@app.callback(
    Output("pv-summary-table", "data"),
    Output("pv-summary-table", "style_data_conditional"),
    Input("pv-strategy-dropdown", "value"),
)
def update_pv_summary_table(strategy: str):
    strategy = strategy or "multivariate"
    ordered = sorted(
        imputation_summary_records,
        key=lambda row: (0 if row["series"] == strategy else (1 if row["series"] == "original" else 2)),
    )
    styles = [
        {"if": {"row_index": "odd"}, "backgroundColor": "rgba(0,0,0,0.02)"},
    ]
    styles.append({"if": {"filter_query": f'{{series}} = "{strategy}"'}, "backgroundColor": "#fff3cd"})
    styles.append({"if": {"filter_query": '{series} = "original"'}, "backgroundColor": "#e9ecef"})
    style = styles
    return ordered, style

@app.callback(
    Output("viz-graph", "figure"),
    Input("viz-date-picker", "start_date"),
    Input("viz-date-picker", "end_date"),
    Input("viz-variable-dropdown", "value"),
    Input("viz-plot-radio", "value"),
)
def update_visualisation(start_date: str, end_date: str, variables: Iterable[str], plot_type: str):
    variables = _normalize_variables(variables)
    filtered = _filter_range(start_date, end_date)
    if filtered.empty:
        return _empty_figure("No data in selected range")

    if plot_type == "timeseries":
        value_cols = [col for col in variables if col in filtered.columns]
        if not value_cols:
            return _empty_figure("No data in selected range")
        numeric_frame = filtered[['timestamp'] + value_cols].copy()
        numeric_frame = coerce_numeric(numeric_frame, value_cols).dropna(subset=value_cols, how='all')
        if numeric_frame.empty:
            return _empty_figure("No data in selected range")
        resampled = (
            numeric_frame.set_index('timestamp')[value_cols]
            .resample("H")
            .mean()
            .dropna(how='all')
            .reset_index()
        )
        if resampled.empty:
            return _empty_figure("No data in selected range")
        fig = go.Figure()
        added = False
        for idx, column in enumerate(variables):
            if column not in resampled.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=resampled['timestamp'],
                    y=resampled[column],
                    name=LABEL_LOOKUP.get(column, column),
                    mode="lines",
                    line=dict(color=SERIES_COLORS.get(column, COLORWAY[idx % len(COLORWAY)]), width=2),
                )
            )
            added = True
        if not added:
            return _empty_figure("No data in selected range")
        fig.update_layout(title="Timeseries overlay", xaxis_title="Timestamp", yaxis_title="Value")
        return fig

    if plot_type == "distribution":
        fig = go.Figure()
        added = False
        for idx, column in enumerate(variables):
            if column not in filtered.columns:
                continue
            series = pd.to_numeric(filtered[column], errors='coerce').dropna()
            if series.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=series,
                    name=LABEL_LOOKUP.get(column, column),
                    nbinsx=60,
                    histnorm="probability density",
                    opacity=0.6,
                    marker_color=SERIES_COLORS.get(column, COLORWAY[idx % len(COLORWAY)]),
                )
            )
            added = True
        if not added:
            return _empty_figure("No data in selected range")
        fig.update_layout(barmode="overlay", title="Distribution", xaxis_title="Value", yaxis_title="Density")
        return fig

    if plot_type == "boxplot":
        variable = variables[0]
        if variable not in filtered.columns:
            return _empty_figure("No data in selected range")
        data = filtered[["hour", variable]].copy()
        data[variable] = pd.to_numeric(data[variable], errors='coerce')
        data = data.dropna(subset=[variable])
        if data.empty:
            return _empty_figure("No data in selected range")
        fig = px.box(
            data,
            x="hour",
            y=variable,
            color_discrete_sequence=[SERIES_COLORS.get(variable, COLORWAY[0])],
            title=f"{LABEL_LOOKUP.get(variable, variable)} by hour of day",
            labels={"hour": "Hour of day", variable: LABEL_LOOKUP.get(variable, variable)},
        )
        return fig

    if plot_type == "heatmap":
        cols = ["pv", "Demand", "Price", "Temperature", "Wind_speed_10m (km/h)", "Shortwave_radiation (W/m²)"]
        corr = filtered[cols].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=cols,
                y=cols,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                colorbar=dict(title="ρ"),
            )
        )
        fig.update_layout(title="Correlation heatmap")
        return fig

    if plot_type == "profiles":
        variable = variables[0]
        if variable not in filtered.columns:
            return _empty_figure("No data in selected range")
        profile_source = filtered[["is_weekend", "hour", variable]].copy()
        profile_source[variable] = pd.to_numeric(profile_source[variable], errors='coerce')
        profile_source = profile_source.dropna(subset=[variable])
        if profile_source.empty:
            return _empty_figure("No data in selected range")
        profile = profile_source.groupby(["is_weekend", "hour"])[variable].mean().reset_index()
        weekday = profile[profile["is_weekend"] == False]
        weekend = profile[profile["is_weekend"] == True]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=weekday["hour"],
                y=weekday[variable],
                name=f"Weekday {LABEL_LOOKUP.get(variable, variable)}",
                mode="lines",
                line=dict(color=SERIES_COLORS.get(variable, COLORWAY[0]), width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=weekend["hour"],
                y=weekend[variable],
                name=f"Weekend {LABEL_LOOKUP.get(variable, variable)}",
                mode="lines",
                line=dict(color=SERIES_COLORS.get(variable, COLORWAY[0]), width=2, dash="dash"),
            )
        )
        fig.update_layout(title=f"Typical hourly profile – {LABEL_LOOKUP.get(variable, variable)}", xaxis_title="Hour of day", yaxis_title=LABEL_LOOKUP.get(variable, variable))
        return fig

    return _empty_figure("Unsupported plot type")


if __name__ == "__main__":
    app.run_server(debug=True)
