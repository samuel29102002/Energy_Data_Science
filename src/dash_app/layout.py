from __future__ import annotations

from typing import Dict

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

NAV_ITEMS = [
    ("overview", "Overview"),
    ("visualisation", "Visualisation"),
    ("decomposition", "Decomposition"),
    ("stat_models", "Stat Models"),
    ("ml_models", "ML Models"),
    ("forecasting", "Forecasting"),
    ("pv_cleaning", "PV Cleaning"),
    ("features", "Features"),
]


def _nav_link(page_id: str, label: str) -> html.Li:
    return html.Li(
        html.A(
            label,
            href=f"/{page_id}",
            className="nav-link",
            id=f"nav-{page_id}",
        ),
        className="nav-item",
    )


def _kpi_card(label: str, value_id: str) -> html.Div:
    return html.Div(
        [
            html.Span(label, className="label"),
            html.Span("--", id=value_id, className="value"),
        ],
        className="card kpi",
    )


def _graph_card(graph_id: str, **kwargs) -> html.Div:
    return html.Div(
        dcc.Graph(id=graph_id, **kwargs, className="rounded"),
        className="card graph-card",
    )


def _datatable_card(table) -> html.Div:
    return html.Div(table, className="card graph-card")


def overview_layout(ctx) -> html.Div:
    controls = html.Div(
        [
            html.Div("Controls", className="section-title"),
            html.Div(
                [
                    html.Label("Date range"),
                    dcc.DatePickerRange(
                        id="date-picker",
                        display_format="YYYY-MM-DD",
                        start_date=ctx.date_min,
                        end_date=ctx.date_max,
                        min_date_allowed=ctx.date_min,
                        max_date_allowed=ctx.date_max,
                        className="rounded",
                    ),
                ],
                className="controls",
            ),
            html.Div(
                [
                    html.Label("Variables"),
                    dcc.Dropdown(
                        id="series-dropdown",
                        options=ctx.numeric_options,
                        value=ctx.default_series,
                        multi=True,
                        clearable=False,
                        className="rounded",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Resampling"),
                    dcc.RadioItems(
                        id="aggregation-radio",
                        options=[
                            {"label": "Hourly", "value": "H"},
                            {"label": "Daily", "value": "D"},
                        ],
                        value="H",
                        className="controls",
                    ),
                ]
            ),
        ],
        className="panel controls",
    )

    graphs = html.Div(
        [
            _graph_card("timeseries-chart"),
            _graph_card("correlation-heatmap"),
            _graph_card("pv-demand-scatter"),
            _graph_card("hourly-heatmap"),
        ],
        className="page-wrapper",
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(controls, lg=3, xs=12),
                    dbc.Col(graphs, lg=9, xs=12),
                ],
                className="g-4",
            )
        ],
        className="page-wrapper",
    )


def visualisation_layout(ctx) -> html.Div:
    controls = html.Div(
        [
            html.Div("Visualisation Controls", className="section-title"),
            html.Div(
                [
                    html.Label("Plot type"),
                    dcc.Dropdown(
                        id="viz-plot-type",
                        options=[
                            {"label": "Timeseries", "value": "timeseries"},
                            {"label": "Distribution", "value": "distribution"},
                            {"label": "Boxplot", "value": "boxplot"},
                            {"label": "Correlation heatmap", "value": "heatmap"},
                            {"label": "Typical profiles", "value": "profiles"},
                        ],
                        value="timeseries",
                        clearable=False,
                    ),
                ],
                className="controls",
            ),
            html.Div(
                [
                    html.Label("Variables"),
                    dcc.Dropdown(
                        id="viz-variable-dropdown",
                        options=ctx.numeric_options,
                        value=ctx.default_series,
                        multi=True,
                        clearable=False,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Date range"),
                    dcc.DatePickerRange(
                        id="viz-date-picker",
                        display_format="YYYY-MM-DD",
                        start_date=ctx.date_min,
                        end_date=ctx.date_max,
                        min_date_allowed=ctx.date_min,
                        max_date_allowed=ctx.date_max,
                    ),
                ]
            ),
        ],
        className="panel controls",
    )

    content = html.Div(
        [
            _graph_card("viz-graph", config={"displayModeBar": True, "displaylogo": False}),
            html.Div(
                html.Div(
                    dcc.Markdown(id="viz-reflection", className="mt-3"),
                    className="panel",
                ),
                className="page-wrapper",
            ),
        ],
        className="page-wrapper",
    )

    return dbc.Row(
        [
            dbc.Col(controls, lg=3, xs=12),
            dbc.Col(content, lg=9, xs=12),
        ],
        className="g-4",
    )


def decomposition_layout(ctx) -> html.Div:
    controls = html.Div(
        [
            html.Div("Decomposition Controls", className="section-title"),
            dcc.RadioItems(
                id="decomp-period-radio",
                options=[
                    {"label": label, "value": key}
                    for key, label in ctx.decomposition_labels.items()
                ],
                value="daily",
                className="controls",
            ),
            dcc.Dropdown(
                id="decomp-profile-dropdown",
                options=[
                    {"label": "Weekday vs Weekend", "value": "weekday_weekend"},
                    {"label": "Monthly profiles", "value": "monthly"},
                ],
                value="weekday_weekend",
                clearable=False,
            ),
            dcc.DatePickerRange(
                id="decomp-date-picker",
                display_format="YYYY-MM-DD",
                start_date=ctx.date_min,
                end_date=ctx.date_max,
                min_date_allowed=ctx.date_min,
                max_date_allowed=ctx.date_max,
            ),
        ],
        className="panel controls",
    )

    content = html.Div(
        [
            _graph_card("decomp-stl-figure"),
            _graph_card("decomp-profile-figure"),
            _datatable_card(
                dbc.Table.from_dataframe(
                    pd.DataFrame(ctx.seasonality_strength_records),
                    striped=True,
                    hover=True,
                    bordered=False,
                    className="rounded",
                )
            ),
        ],
        className="page-wrapper",
    )

    return dbc.Row(
        [
            dbc.Col(controls, lg=3, xs=12),
            dbc.Col(content, lg=9, xs=12),
        ],
        className="g-4",
    )


def stat_models_layout(ctx) -> html.Div:
    controls = html.Div(
        [
            html.Div("Statistical Models", className="section-title"),
            html.Div(
                [
                    html.Label("Model"),
                    dcc.Dropdown(
                        id="stat-model-dropdown",
                        options=ctx.stat_model_options,
                        value=ctx.stat_model_options[0]["value"] if ctx.stat_model_options else None,
                        clearable=False,
                    ),
                ],
                className="controls",
            ),
            dcc.RadioItems(
                id="stat-eval-radio",
                options=[
                    {"label": "Whole-train split", "value": "split"},
                    {"label": "Last-week walk-forward", "value": "walk"},
                ],
                value="split",
            ),
            dcc.Dropdown(
                id="stat-day-dropdown",
                options=[],
                value=None,
                clearable=False,
                disabled=True,
            ),
            html.Div(
                "Select a model and evaluation mode to inspect residual correlation, forecasts, and error statistics.",
                className="alert alert-secondary",
            ),
        ],
        className="panel controls",
    )

    table = dbc.Table(
        id="stat-metrics-summary",
        bordered=False,
        hover=True,
        striped=True,
        className="rounded",
    )

    return dbc.Row(
        [
            dbc.Col(controls, lg=3, xs=12),
            dbc.Col(
                html.Div(
                    [
                        _graph_card("stat-acf-figure"),
                        _graph_card("stat-forecast-figure"),
                        _datatable_card(
                            html.Div(
                                dbc.Table(id="stat-metrics-table", bordered=False, hover=True, striped=True, className="rounded"),
                                className="table-responsive",
                            )
                        ),
                        _datatable_card(
                            html.Div(
                                dbc.Table(id="stat-daily-table", bordered=False, hover=True, striped=True, className="rounded"),
                                className="table-responsive",
                            )
                        ),
                    ],
                    className="page-wrapper",
                ),
                lg=9,
                xs=12,
            ),
        ],
        className="g-4",
    )


def pv_cleaning_layout(ctx) -> html.Div:
    controls = html.Div(
        [
            html.Div("PV Cleaning", className="section-title"),
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="pv-date-picker",
                display_format="YYYY-MM-DD",
                start_date=ctx.date_min,
                end_date=ctx.date_max,
                min_date_allowed=ctx.date_min,
                max_date_allowed=ctx.date_max,
            ),
            html.Label("Imputation strategy"),
            dcc.Dropdown(
                id="pv-strategy-dropdown",
                options=ctx.imputation_options,
                value="multivariate",
                clearable=False,
            ),
        ],
        className="panel controls",
    )

    table_columns = [{"name": c, "id": c} for c in ctx.imputation_summary.columns]

    table = dbc.Table(id="pv-summary-table", bordered=False, hover=True, striped=True, className="rounded")

    content = html.Div(
        [
            _graph_card("pv-cleaning-chart"),
            _datatable_card(table),
        ],
        className="page-wrapper",
    )

    return dbc.Row(
        [
            dbc.Col(controls, lg=3, xs=12),
            dbc.Col(content, lg=9, xs=12),
        ],
        className="g-4",
    )


def features_layout(ctx) -> html.Div:
    controls = html.Div(
        [
            html.Div("Feature Analysis", className="section-title"),
            dbc.Checklist(
                id="features-include-weather",
                options=[{"label": "Include weather features", "value": "weather"}],
                value=["weather"],
                switch=True,
                className="controls",
            ),
            dcc.RadioItems(
                id="features-ranking-method",
                options=[
                    {"label": "Mutual information", "value": "mi"},
                    {"label": "Permutation importance", "value": "permutation"},
                ],
                value="mi",
            ),
            dcc.Dropdown(
                id="features-variable-dropdown",
                options=ctx.feature_options,
                value=ctx.feature_options[0]["value"] if ctx.feature_options else None,
                clearable=False,
            ),
        ],
        className="panel controls",
    )

    return dbc.Row(
        [
            dbc.Col(controls, lg=3, xs=12),
            dbc.Col(
                html.Div(
                    [
                        _datatable_card(
                            dbc.Table(id="features-ranking-table", bordered=False, hover=True, striped=True, className="rounded")
                        ),
                        _graph_card("features-ranking-chart"),
                        _graph_card("features-distribution"),
                    ],
                    className="page-wrapper",
                ),
                lg=9,
                xs=12,
            ),
        ],
        className="g-4",
    )


def ml_models_layout(ctx) -> html.Div:
    controls = html.Div(
        [
            html.Div("ML Models", className="section-title"),
            dcc.Dropdown(
                id="ml-model-dropdown",
                options=[{"label": "XGBoost", "value": "XGBoost"}],
                value="XGBoost",
                clearable=False,
            ),
            dcc.RadioItems(
                id="ml-eval-radio",
                options=[
                    {"label": "Whole-train split", "value": "split"},
                    {"label": "Last-week walk-forward", "value": "walk"},
                ],
                value="split",
            ),
            dcc.Dropdown(
                id="ml-walk-day-dropdown",
                options=[{"label": f"Day {i}", "value": i} for i in range(1, 8)],
                value=1,
                clearable=False,
                disabled=True,
            ),
            dbc.Checklist(
                id="ml-show-importance",
                options=[{"label": "Show feature importance", "value": "importance"}],
                value=["importance"],
                switch=True,
            ),
            html.Div(
                [
                    html.Button("Download metrics", id="ml-download-metrics", className="btn-outline"),
                    html.Button("Download predictions", id="ml-download-preds", className="btn-outline", style={"marginLeft": "8px"}),
                ],
                className="d-flex flex-wrap gap-2",
            ),
            html.Small("nRMSE = RMSE / (max(y_true) - min(y_true))", className="text-muted"),
        ],
        className="panel controls",
    )

    content = html.Div(
        [
            _graph_card("ml-forecast-figure"),
            _graph_card("ml-metrics-figure"),
            _graph_card("ml-importance-figure"),
            _datatable_card(
                html.Div(
                    dbc.Table(id="ml-metrics-table", bordered=False, hover=True, striped=True, className="rounded"),
                    className="table-responsive",
                )
            ),
        ],
        className="page-wrapper",
    )

    return dbc.Row(
        [
            dbc.Col(controls, lg=3, xs=12),
            dbc.Col(content, lg=9, xs=12),
        ],
        className="g-4",
    )


def forecasting_layout(ctx) -> html.Div:
    controls = html.Div(
        [
            html.Div("Forecasting", className="section-title"),
            dcc.RadioItems(
                id="forecast-model-radio",
                options=[
                    {"label": "Best ML", "value": "XGBoost"},
                    {"label": "Best Stat", "value": "BestStat"},
                    {"label": "Naive", "value": "Naive"},
                    {"label": "Seasonal Naive", "value": "SeasonalNaive"},
                ],
                value="XGBoost",
            ),
            dcc.Dropdown(
                id="forecast-day-dropdown",
                options=[{"label": f"Day {i}", "value": i} for i in range(1, 8)],
                value=1,
                clearable=False,
            ),
            dbc.Checklist(
                id="forecast-show-summary",
                options=[{"label": "Show metrics summary", "value": "summary"}],
                value=["summary"],
                switch=True,
            ),
            html.Div(
                [
                    html.Button("Download all predictions", id="forecast-download-preds", className="btn-outline"),
                    html.Button("Download metrics", id="forecast-download-metrics", className="btn-outline", style={"marginLeft": "8px"}),
                ],
                className="d-flex flex-wrap gap-2",
            ),
        ],
        className="panel controls",
    )

    content = html.Div(
        [
            _graph_card("forecast-day-figure"),
            _graph_card("forecast-week-figure"),
            _datatable_card(
                html.Div(
                    dbc.Table(id="forecast-metrics-table", bordered=False, hover=True, striped=True, className="rounded"),
                    className="table-responsive",
                )
            ),
        ],
        className="page-wrapper",
    )

    return dbc.Row(
        [
            dbc.Col(controls, lg=3, xs=12),
            dbc.Col(content, lg=9, xs=12),
        ],
        className="g-4",
    )


PAGE_FACTORIES = {
    "overview": overview_layout,
    "visualisation": visualisation_layout,
    "decomposition": decomposition_layout,
    "stat_models": stat_models_layout,
    "ml_models": ml_models_layout,
    "forecasting": forecasting_layout,
    "pv_cleaning": pv_cleaning_layout,
    "features": features_layout,
}


def create_layout(ctx) -> html.Div:
    sidebar = html.Div(
        [
            html.Div("Energy Saving Plan", className="logo"),
            html.Ul([_nav_link(page_id, label) for page_id, label in NAV_ITEMS], className="sidebar-nav"),
        ],
        className="sidebar",
    )

    header = html.Div(
        [
            html.Div(
                [
                    html.Div("Overview", className="section-title", id="page-title"),
                    html.Div(
                        [
                            _kpi_card("Last data timestamp", "kpi-last-timestamp"),
                            _kpi_card("Active window (days)", "kpi-window-length"),
                            _kpi_card("Selected model", "kpi-selected-model"),
                        ],
                        className="kpi-row",
                    ),
                ],
                className="header",
            ),
        ]
    )

    content = html.Div(id="page-content", className="page-wrapper")

    main = html.Div(
        [
            header,
            content,
        ],
        className="main-container",
    )

    return html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="shared-selections", data={}),
            html.Div([sidebar, main], className="app-shell"),
        ]
    )
