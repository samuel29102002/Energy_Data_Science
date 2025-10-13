from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from src.dash_app.ids import IDS
from src.dash_app.utils import DashboardData, format_number

NAV_ITEMS = (
    {"href": "/overview", "label": "Overview", "nav_id": IDS["navigation"]["overview"]},
    {"href": "/ml", "label": "ML Models", "nav_id": IDS["navigation"]["ml"]},
    {"href": "/forecast", "label": "Forecasting", "nav_id": IDS["navigation"]["forecast"]},
    {"href": "/optimization", "label": "Optimization", "nav_id": IDS["navigation"]["optimization"]},
    {"href": "/data", "label": "Data / Cleaning", "nav_id": IDS["navigation"]["data"]},
    {"href": "/about", "label": "About / Info", "nav_id": IDS["navigation"]["about"]},
)


def _format_date(value) -> str:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert(None)
        return value.strftime("%d.%m.%Y")
    if hasattr(value, "strftime"):
        return value.strftime("%d.%m.%Y")
    if value:
        return str(value)
    return "--"


def _nav_links() -> dbc.Nav:
    links = [
        dbc.NavLink(
            item["label"],
            href=item["href"],
            id=item["nav_id"],
            active="exact",
            className="nav-link-modern",
        )
        for item in NAV_ITEMS
    ]
    return dbc.Nav(links, pills=True, className="top-nav", justified=False)


def _topbar() -> html.Header:
    return html.Header(
        [
            html.Div(
                [
                    html.Div("Energy Data Science", className="topbar-title"),
                    html.Div("Insights Dashboard", className="topbar-subtitle"),
                ],
                className="topbar-brand",
            ),
            html.Div(_nav_links(), className="topbar-links"),
            html.Div(
                [
                    html.Span("Palette", className="topbar-footnote"),
                    html.Div(
                        [
                            html.Div(style={"background": "#1E90FF"}),
                            html.Div(style={"background": "#00B386"}),
                            html.Div(style={"background": "#F4B400"}),
                        ],
                        className="topbar-palette",
                    ),
                ],
                className="topbar-meta",
            ),
        ],
        className="topbar",
    )


def _kpi_card(title: str, value_id: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="kpi-title"),
            html.Div("--", id=value_id, className="kpi-value"),
        ]),
        className="kpi-card",
    )


def _stat_card(title: str, value_id: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="stat-title"),
            html.Div("--", id=value_id, className="stat-value"),
        ]),
        className="stat-card",
    )


def _loading(component) -> dcc.Loading:
    return dcc.Loading(component, type="circle", color="#1E90FF")


def _overview_hero(data: DashboardData) -> html.Div:
    context = data.summary_context
    start_str = _format_date(context.get("start_date"))
    end_str = _format_date(context.get("end_date"))

    raw_rows = context.get("raw_rows", 0)
    clean_rows = context.get("clean_rows", 0)
    feature_count = context.get("feature_count", 0)
    missing_ratio = context.get("missing_ratio")
    clean_steps = context.get("clean_steps", 0)
    scenario_count = context.get("scenario_count", 0)
    model_count = context.get("model_count", 0)

    cleaned_pct = (clean_rows / raw_rows) * 100 if raw_rows else None
    missing_pct = (missing_ratio * 100) if missing_ratio is not None else None

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Projektstatus", className="hero-label"),
                    html.H2("Aktueller Überblick", className="hero-title"),
                    html.P(
                        f"Zeitraum {start_str} bis {end_str} mit {model_count} Prognosemodellen und {scenario_count} Batterieszenarien.",
                        className="hero-text",
                    ),
                    html.Div(
                        [
                            html.Span(f"Bereinigungsschritte: {clean_steps}", className="hero-chip"),
                            html.Span(
                                f"Datensätze gesäubert: {format_number(cleaned_pct, precision=1, as_percent=True)}",
                                className="hero-chip",
                            )
                            if cleaned_pct is not None
                            else html.Span("Datensätze gesäubert: --", className="hero-chip"),
                        ],
                        className="hero-chips",
                    ),
                ],
                className="hero-copy",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Bereinigte Zeilen", className="hero-stat-label"),
                            html.Span(format_number(clean_rows, precision=0), className="hero-stat-value"),
                            html.Span(
                                f"von {format_number(raw_rows, precision=0)} Rohdaten", className="hero-stat-footer"
                            ),
                        ],
                        className="hero-stat-card",
                    ),
                    html.Div(
                        [
                            html.Span("Merkmalsumfang", className="hero-stat-label"),
                            html.Span(str(feature_count), className="hero-stat-value"),
                            html.Span("verwendete Features", className="hero-stat-footer"),
                        ],
                        className="hero-stat-card",
                    ),
                    html.Div(
                        [
                            html.Span("Missing-Anteil", className="hero-stat-label"),
                            html.Span(
                                format_number(missing_pct, precision=1, as_percent=True)
                                if missing_pct is not None
                                else "--",
                                className="hero-stat-value",
                            ),
                            html.Span("in den Rohdaten", className="hero-stat-footer"),
                        ],
                        className="hero-stat-card",
                    ),
                ],
                className="hero-stats",
            ),
        ],
        className="overview-hero",
    )


def _overview_insight_list(data: DashboardData) -> html.Ul:
    df = data.cleaned_dataset.copy()
    if df.empty or "timestamp" not in df.columns:
        return html.Ul([html.Li("Keine bereinigten Daten verfügbar.")], className="insight-list")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    insights: List[html.Li] = []

    if "Demand" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        hourly = df.groupby("hour")["Demand"].mean().dropna()
        if not hourly.empty:
            peak_hour = int(hourly.idxmax())
            peak_value = format_number(float(hourly.max()), precision=2)
            insights.append(html.Li(f"Höchste Nachfrage um {peak_hour:02d}:00 Uhr mit {peak_value} kWh."))

        df["month"] = df["timestamp"].dt.month
        monthly = df.groupby("month")["Demand"].mean().dropna()
        if not monthly.empty:
            top_month = int(monthly.idxmax())
            month_name = pd.Timestamp(year=2000, month=top_month, day=1).strftime("%B")
            insights.append(
                html.Li(
                    f"Stärkster Nachfrage-Monat: {month_name} mit {format_number(float(monthly.max()), precision=2)} kWh durchschnittlich."
                )
            )

    if {"Demand", "pv"}.issubset(df.columns):
        demand_sum = df["Demand"].sum()
        pv_sum = df["pv"].sum()
        if demand_sum > 0:
            pv_share = (pv_sum / demand_sum) * 100
            insights.append(
                html.Li(
                    f"PV deckt {format_number(pv_share, precision=1, as_percent=True)} des aggregierten Energiebedarfs."
                )
            )

    if {"Demand", "Temperature"}.issubset(df.columns):
        corr = df[["Demand", "Temperature"]].corr().iloc[0, 1]
        if pd.notna(corr):
            insights.append(html.Li(f"Korrelation Demand/Temperatur: {format_number(float(corr), precision=2)}."))

    if not insights:
        insights.append(html.Li("Keine zusätzlichen Insights berechenbar."))

    return html.Ul(insights, className="insight-list")


def overview_page(data: DashboardData) -> html.Div:
    dropdown_options = [{"label": model, "value": model} for model in data.forecast_models]
    default_value = data.best_forecast_model or (dropdown_options[0]["value"] if dropdown_options else None)

    return html.Div(
        [
            _overview_hero(data),
            html.Div(
                [
                    _kpi_card("Average RMSE", IDS["overview"]["kpi_rmse"]),
                    _kpi_card("Best Model", IDS["overview"]["kpi_best_model"]),
                    _kpi_card("Cost Savings", IDS["overview"]["kpi_cost_savings"]),
                    _kpi_card("Forecast Accuracy", IDS["overview"]["kpi_accuracy"]),
                ],
                className="kpi-row",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div("Model selection", className="control-label"),
                                dcc.Dropdown(
                                    id=IDS["overview"]["model_dropdown"],
                                    options=dropdown_options,
                                    value=default_value,
                                    clearable=False,
                                    className="dropdown-modern",
                                ),
                            ],
                            className="control-row",
                        ),
                        html.Div(
                            _loading(dcc.Graph(id=IDS["overview"]["timeseries_graph"], responsive=True)),
                            className="graph-wrapper",
                        ),
                    ]
                ),
                className="card-section",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Model RMSE comparison", className="section-title"),
                                    _loading(dcc.Graph(id=IDS["overview"]["metrics_graph"], responsive=True)),
                                ]
                            ),
                            className="card-section",
                        ),
                        lg=6,
                        md=12,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("System summary & insights", className="section-title"),
                                    html.Div(
                                        id=IDS["overview"]["summary_text"],
                                        className="summary-text summary-container",
                                    ),
                                ]
                            ),
                            className="card-section",
                        ),
                        lg=6,
                        md=12,
                    ),
                ],
                className="mt-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Demand seasonality", className="section-title"),
                                    _loading(
                                        dcc.Graph(
                                            id=IDS["overview"]["seasonality_graph"],
                                            responsive=True,
                                            config={"displayModeBar": False},
                                        )
                                    ),
                                ]
                            ),
                            className="card-section",
                        ),
                        lg=6,
                        md=12,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Demand vs PV profiles", className="section-title"),
                                    _loading(
                                        dcc.Graph(
                                            id=IDS["overview"]["profile_graph"],
                                            responsive=True,
                                            config={"displayModeBar": False},
                                        )
                                    ),
                                ]
                            ),
                            className="card-section",
                        ),
                        lg=6,
                        md=12,
                    ),
                ],
                className="mt-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Explorative Highlights", className="section-title"),
                        _overview_insight_list(data),
                    ]
                ),
                className="card-section mt-4 insight-card",
            ),
        ],
        className="page-body",
    )


def ml_page(data: DashboardData) -> html.Div:
    dropdown_options = [{"label": model, "value": model} for model in data.ml_models]
    default_value = dropdown_options[0]["value"] if dropdown_options else None

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Model selection", className="control-label"),
                        dcc.Dropdown(
                            id=IDS["ml"]["model_dropdown"],
                            options=dropdown_options,
                            value=default_value,
                            clearable=False,
                            className="dropdown-modern",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    _loading(dcc.Graph(id=IDS["ml"]["feature_importance"], responsive=True)),
                                    lg=6,
                                    md=12,
                                ),
                                dbc.Col(
                                    _loading(dcc.Graph(id=IDS["ml"]["residual_graph"], responsive=True)),
                                    lg=6,
                                    md=12,
                                ),
                            ],
                            className="mt-4",
                        ),
                    ]
                ),
                className="card-section",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Evaluation metrics", className="section-title"),
                        _loading(
                            dash_table.DataTable(
                                id=IDS["ml"]["metrics_table"],
                                columns=[
                                    {"name": "Metric", "id": "metric"},
                                    {"name": "Value", "id": "value"},
                                ],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_as_list_view=True,
                                style_cell={
                                    "padding": "0.75rem",
                                    "fontFamily": "Inter, system-ui",
                                    "fontSize": 14,
                                },
                                style_header={
                                    "backgroundColor": "#f1f5f9",
                                    "fontWeight": "600",
                                    "border": "none",
                                },
                                style_data={"border": "none"},
                            )
                        ),
                    ]
                ),
                className="card-section mt-4",
            ),
        ],
        className="page-body",
    )


def forecasting_page(data: DashboardData) -> html.Div:
    dropdown_options = [{"label": model, "value": model} for model in data.forecast_models]
    default_value = data.best_forecast_model or (dropdown_options[0]["value"] if dropdown_options else None)

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Select model", className="control-label"),
                        dcc.Dropdown(
                            id=IDS["forecast"]["model_dropdown"],
                            options=dropdown_options,
                            value=default_value,
                            clearable=False,
                            className="dropdown-modern",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    _loading(dcc.Graph(id=IDS["forecast"]["timeseries_graph"], responsive=True)),
                                    lg=8,
                                    md=12,
                                ),
                                dbc.Col(
                                    _loading(dcc.Graph(id=IDS["forecast"]["daily_error_graph"], responsive=True)),
                                    lg=4,
                                    md=12,
                                ),
                            ],
                            className="mt-4",
                        ),
                    ]
                ),
                className="card-section",
            ),
        ],
        className="page-body",
    )


def optimization_page(data: DashboardData) -> html.Div:
    dropdown_options = [{"label": scenario, "value": scenario} for scenario in data.storage_scenarios]
    default_value = dropdown_options[0]["value"] if dropdown_options else None

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Scenario", className="control-label"),
                        dcc.Dropdown(
                            id=IDS["optimization"]["scenario_dropdown"],
                            options=dropdown_options,
                            value=default_value,
                            clearable=False,
                            className="dropdown-modern",
                        ),
                        html.Div(
                            [
                                _kpi_card("Total cost", IDS["optimization"]["kpi_cost"]),
                                _kpi_card("Energy bought", IDS["optimization"]["kpi_bought"]),
                                _kpi_card("Energy sold", IDS["optimization"]["kpi_sold"]),
                                _kpi_card("Battery cycles", IDS["optimization"]["kpi_cycles"]),
                            ],
                            className="kpi-row",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    _loading(dcc.Graph(id=IDS["optimization"]["soc_graph"], responsive=True)),
                                    lg=6,
                                    md=12,
                                ),
                                dbc.Col(
                                    _loading(dcc.Graph(id=IDS["optimization"]["energy_graph"], responsive=True)),
                                    lg=6,
                                    md=12,
                                ),
                            ],
                            className="mt-4",
                        ),
                    ]
                ),
                className="card-section",
            ),
        ],
        className="page-body",
    )


def data_page(data: DashboardData) -> html.Div:
    start_date = data.summary_context.get("start_date")
    end_date = data.summary_context.get("end_date")

    default_columns: List[str] = []
    if not data.raw_dataset.empty:
        numeric_cols = data.raw_dataset.select_dtypes(include=["number"]).columns.tolist()
        default_columns = [col for col in numeric_cols if col not in {"timestamp", "model_name"}]
    variable_options = [{"label": col, "value": col} for col in default_columns]
    default_selection = [value for value in default_columns if value.lower() in {"demand", "pv"}][:2]
    if not default_selection and default_columns:
        default_selection = default_columns[:2]

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Filters", className="section-title compact"),
                                    dcc.DatePickerRange(
                                        id=IDS["data"]["date_range"],
                                        start_date=start_date,
                                        end_date=end_date,
                                        display_format="YYYY-MM-DD",
                                        className="date-picker",
                                    ),
                                    html.Div("Variables", className="control-label mt-3"),
                                    dcc.Dropdown(
                                        id=IDS["data"]["variable_select"],
                                        options=variable_options,
                                        value=default_selection,
                                        multi=True,
                                        placeholder="Select variables",
                                        className="dropdown-modern",
                                    ),
                                    html.Div("Show", className="control-label mt-3"),
                                    dcc.RadioItems(
                                        id=IDS["data"]["show_mode"],
                                        options=[
                                            {"label": "Raw", "value": "raw"},
                                            {"label": "Cleaned", "value": "cleaned"},
                                            {"label": "Overlay", "value": "overlay"},
                                        ],
                                        value="overlay",
                                        className="radio-toggle",
                                        labelStyle={"display": "inline-block", "marginRight": "12px"},
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Download cleaned CSV",
                                                id=IDS["data"]["download_cleaned"],
                                                color="primary",
                                                className="mt-4",
                                            ),
                                            dcc.Download(id=IDS["data"]["download_target"]),
                                        ],
                                        className="mt-2",
                                    ),
                                ]
                            ),
                            className="card-section control-card",
                        ),
                        lg=4,
                        md=12,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    _stat_card("Total rows", IDS["data"]["totals_rows"]),
                                    _stat_card("Date range", IDS["data"]["totals_range"]),
                                    _stat_card("Missing values", IDS["data"]["totals_missing"]),
                                ],
                                className="stat-row",
                            ),
                            html.Div(
                                [
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.Div("Time-series comparison", className="section-title"),
                                                _loading(
                                                    dcc.Graph(
                                                        id=IDS["data"]["time_series"],
                                                        responsive=True,
                                                        config={"responsive": True},
                                                        style={"width": "100%", "height": "100%"},
                                                    )
                                                ),
                                            ]
                                        ),
                                        className="card-section figure-card",
                                    ),
                                ],
                                className="data-figures mt-3",
                            ),
                        ],
                        lg=8,
                        md=12,
                    ),
                ],
                className="page-row",
            ),
            html.Div(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Missingness overview", className="section-title"),
                                _loading(
                                    dcc.Graph(
                                        id=IDS["data"]["missing_heatmap"],
                                        responsive=True,
                                        config={"responsive": True},
                                        style={"width": "100%", "height": "100%"},
                                    )
                                ),
                                html.Div(id=IDS["data"]["missing_summary"], className="summary-text mt-3"),
                            ]
                        ),
                        className="card-section figure-card",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Summary statistics", className="section-title"),
                                _loading(
                                    dash_table.DataTable(
                                        id=IDS["data"]["summary_stats"],
                                        data=[],
                                        columns=[],
                                        style_as_list_view=True,
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "padding": "0.65rem",
                                            "fontFamily": "Inter, system-ui",
                                            "fontSize": 13,
                                        },
                                        style_header={
                                            "backgroundColor": "#f1f5f9",
                                            "fontWeight": "600",
                                            "border": "none",
                                        },
                                        style_data={"border": "none"},
                                    )
                                ),
                                html.Hr(),
                                html.Div("Schema", className="section-title compact"),
                                _loading(
                                    dash_table.DataTable(
                                        id=IDS["data"]["schema_table"],
                                        data=[],
                                        columns=[],
                                        style_as_list_view=True,
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "padding": "0.55rem",
                                            "fontFamily": "Inter, system-ui",
                                            "fontSize": 13,
                                        },
                                        style_header={
                                            "backgroundColor": "#f8fafc",
                                            "fontWeight": "600",
                                            "border": "none",
                                        },
                                        style_data={"border": "none"},
                                    )
                                ),
                            ]
                        ),
                        className="card-section figure-card",
                    ),
                ],
                className="data-figures mt-3",
            ),
            html.Div(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Raw data sample", className="section-title"),
                                _loading(
                                    dash_table.DataTable(
                                        id=IDS["data"]["raw_head"],
                                        data=[],
                                        columns=[],
                                        style_as_list_view=True,
                                        style_table={"overflowX": "auto"},
                                        page_size=10,
                                        style_cell={
                                            "padding": "0.55rem",
                                            "fontFamily": "Inter, system-ui",
                                            "fontSize": 13,
                                        },
                                        style_header={
                                            "backgroundColor": "#f1f5f9",
                                            "fontWeight": "600",
                                            "border": "none",
                                        },
                                        style_data={"border": "none"},
                                    )
                                ),
                            ]
                        ),
                        className="card-section figure-card",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Cleaning steps", className="section-title"),
                                _loading(html.Ul(id=IDS["data"]["clean_log"], className="clean-log")),
                            ]
                        ),
                        className="card-section figure-card",
                    ),
                ],
                className="data-figures mt-3",
            ),
        ],
        className="page-body page-body-compact",
        id=IDS["data"]["page"],
    )


def about_page(_: DashboardData) -> html.Div:
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Project summary", className="section-title"),
                        html.P(
                            "This interactive dashboard summarises the Energy Data Science project, combining forecasting, "
                            "machine learning, and optimisation outputs into a single view.",
                            className="about-text",
                        ),
                        html.P(
                            "All metrics and charts are refreshed from the latest CSV exports inside reports/tables.",
                            className="about-text",
                        ),
                        html.Hr(),
                        html.Div("Team focus areas", className="section-title"),
                        html.Ul(
                            [
                                html.Li("Data ingestion & cleaning"),
                                html.Li("Forecasting & ML modelling"),
                                html.Li("Storage optimisation"),
                            ],
                            className="about-list",
                        ),
                    ]
                ),
                className="card-section",
            ),
        ],
        className="page-body",
    )


PAGE_BUILDERS: Dict[str, Callable[[DashboardData], html.Div]] = {
    "/": overview_page,
    "/overview": overview_page,
    "/ml": ml_page,
    "/forecast": forecasting_page,
    "/optimization": optimization_page,
    "/data": data_page,
    "/about": about_page,
}


def create_layout(data: DashboardData) -> html.Div:
    return html.Div(
        [
            dcc.Location(id=IDS["routing"]["location"], refresh=False),
            html.Div(
                [
                    _topbar(),
                    html.Div(
                        overview_page(data),
                        id=IDS["routing"]["content"],
                        className="content",
                    ),
                ],
                className="layout",
            ),
        ],
        className="app-root",
    )


def render_page(pathname: str, data: DashboardData) -> html.Div:
    builder = PAGE_BUILDERS.get(pathname, overview_page)
    return builder(data)
