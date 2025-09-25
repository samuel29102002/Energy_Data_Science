"""Interactive Dash application for exploring the HEMS training dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "train_252145.csv"

# Load and enrich the dataset once at startup.
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()
df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
df["date"] = df["timestamp"].dt.date

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

COLORWAY = ["#FFA500", "#2ca02c", "#1f77b4", "#7f7f7f", "#FF7F0E"]
SERIES_COLORS = {
    "pv": "#FFA500",
    "Demand": "#2ca02c",
    "Price": "#1f77b4",
    "Temperature": "#FF7F0E",
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

app.layout = dbc.Container(
    [
        html.H1("HEMS Data Explorer", className="text-center my-4"),
        dbc.Tabs(
            [
                dbc.Tab(overview_layout, label="Overview", tab_id="overview", tab_style={"padding": "0.75rem"}, active_tab_style={"backgroundColor": "#ffffff"}),
                dbc.Tab(visualisation_layout, label="Visualisation Studio", tab_id="visualisation", tab_style={"padding": "0.75rem"}, active_tab_style={"backgroundColor": "#ffffff"}),
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


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=message)
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
    resampled = filtered.set_index("timestamp").resample(aggregation).mean().dropna(how="all")
    fig = go.Figure()
    for idx, column in enumerate(series):
        if column not in resampled.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=resampled.index,
                y=resampled[column],
                name=LABEL_LOOKUP.get(column, column),
                mode="lines",
                line=dict(color=SERIES_COLORS.get(column, COLORWAY[idx % len(COLORWAY)]), width=2.2),
            )
        )
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


@app.callback(
    Output("pv-demand-scatter", "figure"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
)
def update_scatter(start_date: str, end_date: str):
    filtered = _filter_range(start_date, end_date)
    if filtered.empty:
        return _empty_figure("No data in selected range")
    sample = filtered.sample(n=min(3000, len(filtered)), random_state=7)
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
    pivot = (
        filtered.pivot_table(values="pv", index="day_of_week", columns="hour", aggfunc="mean")
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
        resampled = filtered.set_index("timestamp").resample("H").mean().dropna(how="all")
        fig = go.Figure()
        for idx, column in enumerate(variables):
            if column not in resampled.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=resampled.index,
                    y=resampled[column],
                    name=LABEL_LOOKUP.get(column, column),
                    mode="lines",
                    line=dict(color=SERIES_COLORS.get(column, COLORWAY[idx % len(COLORWAY)]), width=2),
                )
            )
        fig.update_layout(title="Timeseries overlay", xaxis_title="Timestamp", yaxis_title="Value")
        return fig

    if plot_type == "distribution":
        fig = go.Figure()
        for idx, column in enumerate(variables):
            fig.add_trace(
                go.Histogram(
                    x=filtered[column],
                    name=LABEL_LOOKUP.get(column, column),
                    nbinsx=60,
                    histnorm="probability density",
                    opacity=0.6,
                    marker_color=SERIES_COLORS.get(column, COLORWAY[idx % len(COLORWAY)]),
                )
            )
        fig.update_layout(barmode="overlay", title="Distribution", xaxis_title="Value", yaxis_title="Density")
        return fig

    if plot_type == "boxplot":
        variable = variables[0]
        data = filtered[["hour", variable]]
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
        profile = filtered.groupby(["is_weekend", "hour"])[variable].mean().reset_index()
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
