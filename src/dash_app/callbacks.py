from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update

from src.dash_app.ids import IDS
from src.dash_app.layout import render_page
from src.dash_app.utils import DashboardData, format_number
from src.dash_app.data_processing import (
    compute_missing_summary,
    compute_schema_table,
    compute_totals_summary,
    filter_time_range,
)
from src.dash_app.utils_figures import (
    make_demand_pv_timeseries,
    make_demand_seasonality,
    make_missingness_heatmap,
    placeholder_fig,
)
from src.dash_app.utils_io import load_csv_safe


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


@dataclass(frozen=True)
class _OverviewSummary:
    average_rmse: str
    best_model: str
    cost_savings: str
    accuracy: str
    summary: html.Div


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="#6b7280", size=16),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        height=400,
        font=dict(family="Inter, system-ui, -apple-system, sans-serif"),
        title_font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=18, color="#94a3b8"),
    )
    return fig


def _apply_fig_style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        uirevision=True,
        margin=dict(l=50, r=30, t=80, b=50),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        autosize=True,
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=12, color="#333"),
        title_font=dict(family="CMU Serif, 'Times New Roman', serif", size=22, color="#1a1a1a"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


def _make_profile_overlay(df: pd.DataFrame) -> go.Figure:
    if df.empty or not {"timestamp", "Demand", "pv"}.issubset(df.columns):
        return placeholder_fig("No profile data available")
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    work["hour"] = work["timestamp"].dt.hour
    aggregated = (
        work.groupby("hour")[["Demand", "pv"]]
        .mean()
        .reset_index()
        .sort_values("hour")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=aggregated["hour"],
            y=aggregated["Demand"],
            name="Demand",
            mode="lines+markers",
            line=dict(color="#1E90FF", width=2.4),
            marker=dict(size=7, color="#1E90FF"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=aggregated["hour"],
            y=aggregated["pv"],
            name="PV",
            mode="lines+markers",
            line=dict(color="#2CA02C", width=2.2, dash="dash"),
            marker=dict(size=7, color="#2CA02C"),
        )
    )
    fig.update_layout(
        title="Demand vs PV hourly profile",
        xaxis_title="Hour of day",
        yaxis_title="kWh",
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(dtick=1)
    return fig


def _load_table(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name
    df = load_csv_safe(path)
    if df.empty:
        return df
    return df.dropna(how="all").reset_index(drop=True)


def _format_kpi_value(value: object, unit: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, (int, float)) and unit.strip() == "%":
        return f"{value:.2f}%"
    if isinstance(value, (int, float)) and unit:
        return f"{value:.2f} {unit}".rstrip()
    if isinstance(value, (int, float)):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value} {unit}".strip()


def _build_overview_kpis(data: DashboardData) -> tuple[str, str, str, str]:
    metrics = data.overview_metrics
    return (
        format_number(metrics.get("average_rmse"), precision=3),
        metrics.get("best_model") or "--",
        format_number(metrics.get("cost_savings"), suffix=" €"),
        format_number(metrics.get("forecast_accuracy"), precision=1, as_percent=True),
    )


def _build_insights(clean_df: pd.DataFrame, limit: int = 5) -> html.Ul:
    kpi_df = _load_table("07_kpis.csv")
    feature_df = _load_table("05_feature_stats.csv")
    items: List[html.Li] = []

    if not kpi_df.empty:
        kpi_map = {row["kpi"]: row for row in kpi_df.to_dict("records")}
        if "Peak demand hour" in kpi_map:
            val = kpi_map["Peak demand hour"].get("value")
            items.append(html.Li(f"Peak demand occurs around {val}."))
        if "Mean demand" in kpi_map:
            val = _format_kpi_value(kpi_map["Mean demand"].get("value"), kpi_map["Mean demand"].get("unit", ""))
            items.append(html.Li(f"Mean hourly demand is approximately {val}."))
        if "PV self-consumption" in kpi_map:
            val = _format_kpi_value(kpi_map["PV self-consumption"].get("value"), kpi_map["PV self-consumption"].get("unit", ""))
            items.append(html.Li(f"PV covers about {val} of the demand."))
        if "Overall missingness" in kpi_map:
            val = _format_kpi_value(kpi_map["Overall missingness"].get("value"), kpi_map["Overall missingness"].get("unit", ""))
            items.append(html.Li(f"Overall data missingness is {val}."))

    if not feature_df.empty:
        top_row = feature_df.iloc[0]
        feature_name = top_row.get("feature")
        corr_val = top_row.get("correlation_with_demand")
        if feature_name:
            if isinstance(corr_val, (int, float)) and not np.isnan(corr_val):
                items.append(html.Li(f"{feature_name} shows the strongest correlation with demand ({corr_val:.2f})."))
            else:
                items.append(html.Li(f"{feature_name} is among the most informative features for demand prediction."))

    if not clean_df.empty:
        work = clean_df.copy()
        if "timestamp" in work.columns:
            work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
            work = work.dropna(subset=["timestamp"])  # type: ignore[call-arg]
            work["weekday"] = work["timestamp"].dt.day_name()
            work["hour"] = work["timestamp"].dt.hour
            if {"Demand", "pv"}.issubset(work.columns):
                midday = work[(work["hour"] >= 10) & (work["hour"] <= 16)]
                if not midday.empty:
                    pv_share_midday = midday["pv"].sum() / midday["Demand"].sum() if midday["Demand"].sum() else np.nan
                    if pv_share_midday and not np.isnan(pv_share_midday):
                        items.append(html.Li(f"Midday PV covers roughly {pv_share_midday * 100:.1f}% of demand."))
            weekday_load = work.groupby("weekday")["Demand"].mean().sort_values(ascending=False)
            if not weekday_load.empty:
                items.append(html.Li(f"{weekday_load.index[0]} shows the highest average demand."))

    unique_items: List[str] = []
    deduped: List[html.Li] = []
    for li in items:
        text = li.children if isinstance(li.children, str) else " ".join(map(str, li.children))
        if text not in unique_items:
            unique_items.append(text)
            deduped.append(li)
    if len(deduped) < 1:
        deduped.append(html.Li("No insights available."))
    return html.Ul(deduped[:limit], className="insight-list")


def _format_schema_table(schema_df: pd.DataFrame, selected: Optional[List[str]] = None) -> tuple[list[dict], list[dict]]:
    if schema_df.empty:
        return [], []
    df = schema_df.copy()
    df["missing_pct"] = df["missing_pct"].fillna(0.0)
    df["missing_pct"] = df["missing_pct"].round(2)
    if selected:
        selected_set = set(selected)
        df["_selected"] = df["column"].isin(selected_set)
        df = df.sort_values(["_selected", "missing_pct"], ascending=[False, False]).drop(columns=["_selected"])
    else:
        df = df.sort_values("missing_pct", ascending=False)
    data = df.to_dict("records")
    columns = [
        {"name": "Column", "id": "column"},
        {"name": "Dtype", "id": "dtype"},
        {"name": "Missing %", "id": "missing_pct", "type": "numeric", "format": {"specifier": ".2f"}},
    ]
    return data, columns


def _format_clean_log(log: List[Dict[str, object]]) -> List[html.Li]:
    if not log:
        return [html.Li("No cleaning steps required.")]
    items: List[html.Li] = []
    for entry in log:
        step = entry.get("step", "")
        detail_parts: List[str] = []
        for key, value in entry.items():
            if key == "step" or value in (None, ""):
                continue
            detail_parts.append(f"{key}: {value}")
        detail = " | ".join(map(str, detail_parts)) if detail_parts else ""
        text = f"{step}" if not detail else f"{step} ({detail})"
        items.append(html.Li(text))
    return items


def _build_overview_summary(data: DashboardData) -> _OverviewSummary:
    metrics = data.overview_metrics
    avg_rmse = format_number(metrics.get("average_rmse"), precision=3)
    best_model = metrics.get("best_model") or "--"
    best_model_rmse = format_number(metrics.get("best_model_rmse"), precision=3)
    cost_savings = format_number(metrics.get("cost_savings"), suffix=" €")
    accuracy = format_number(metrics.get("forecast_accuracy"), precision=1, as_percent=True)

    start = data.summary_context.get("start_date")
    end = data.summary_context.get("end_date")
    start_str = start.strftime("%Y-%m-%d") if isinstance(start, pd.Timestamp) else "--"
    end_str = end.strftime("%Y-%m-%d") if isinstance(end, pd.Timestamp) else "--"

    summary = html.Div(
        [
            html.Div(f"Date range: {start_str} → {end_str}"),
            html.Div(f"Models compared: {data.summary_context.get('model_count', 0)}"),
            html.Div(f"Storage scenarios: {data.summary_context.get('scenario_count', 0)}"),
            html.Div(f"Tables path: {data.summary_context.get('tables_path', '--')}", className="summary-path"),
            html.Div(f"Best model RMSE: {best_model_rmse}"),
        ],
        className="summary-stack",
    )

    return _OverviewSummary(
        average_rmse=avg_rmse,
        best_model=best_model,
        cost_savings=cost_savings,
        accuracy=accuracy,
        summary=summary,
    )


def _get_model_predictions(data: DashboardData, model: Optional[str]) -> pd.DataFrame:
    if data.forecast_predictions.empty:
        return pd.DataFrame()
    if model and model in data.forecast_models:
        filtered = data.forecast_predictions[data.forecast_predictions["model_name"] == model]
        if not filtered.empty:
            return filtered
    if data.best_forecast_model:
        fallback = data.forecast_predictions[data.forecast_predictions["model_name"] == data.best_forecast_model]
        if not fallback.empty:
            return fallback
    return data.forecast_predictions


def _forecast_metrics_bar(data: DashboardData) -> go.Figure:
    metrics = data.forecast_metrics_summary
    if metrics.empty:
        return _empty_figure("No forecast metrics available")

    # Determine RMSE column
    rmse_col = "RMSE_mean" if "RMSE_mean" in metrics.columns else "RMSE"
    if rmse_col not in metrics.columns:
        return _empty_figure("No RMSE data available")

    display = metrics.dropna(subset=["model_name", rmse_col]).copy()
    if display.empty:
        return _empty_figure("No forecast metrics available")

    display.sort_values(rmse_col, inplace=True)
    fig = px.bar(
        display,
        x="model_name",
        y=rmse_col,
        color=rmse_col,
        color_continuous_scale=["#00B386", "#1E90FF"],
        labels={"model_name": "Model", rmse_col: "RMSE"},
        title="Mean RMSE by model",
    )
    return _apply_fig_style(fig, height=360)


def register_callbacks(app: Dash, data: DashboardData) -> None:
    overview_summary = _build_overview_summary(data)

    @app.callback(
        Output(IDS["routing"]["content"], "children"),
        Input(IDS["routing"]["location"], "pathname"),
    )
    def _render_page(pathname: str) -> html.Div:
        path = pathname or "/overview"
        return render_page(path, data)

    @app.callback(
        Output(IDS["overview"]["kpi_rmse"], "children"),
        Output(IDS["overview"]["kpi_best_model"], "children"),
        Output(IDS["overview"]["kpi_cost_savings"], "children"),
        Output(IDS["overview"]["kpi_accuracy"], "children"),
        Output(IDS["overview"]["summary_text"], "children"),
        Input(IDS["routing"]["location"], "pathname"),
    )
    def _populate_overview_kpis(pathname: str):
        mean_demand, peak_hour, pv_share, missing = _build_overview_kpis(data)
        summary_block = html.Div(
            [
                overview_summary.summary,
                html.Div("Key insights", className="summary-subtitle"),
                _build_insights(data.cleaned_dataset),
            ],
            className="summary-wrapper",
        )
        return mean_demand, peak_hour, pv_share, missing, summary_block

    @app.callback(
        Output(IDS["overview"]["timeseries_graph"], "figure"),
        Input(IDS["overview"]["model_dropdown"], "value"),
    )
    def _overview_timeseries(selected_range: Optional[str]):
        clean_df = data.cleaned_dataset.copy()
        if clean_df.empty:
            return _empty_figure("No cleaned dataset available")
        clean_df = clean_df.copy()
        if "timestamp" in clean_df.columns:
            clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
            clean_df = clean_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if selected_range in {"last_30", "last_90"} and "timestamp" in clean_df.columns:
            end = clean_df["timestamp"].max()
            delta = 30 if selected_range == "last_30" else 90
            start = end - pd.Timedelta(days=delta)
            mask = clean_df["timestamp"] >= start
            clean_df = clean_df.loc[mask]
        figure = make_demand_pv_timeseries(clean_df)
        return figure

    @app.callback(
        Output(IDS["overview"]["metrics_graph"], "figure"),
        Input(IDS["routing"]["location"], "pathname"),
    )
    def _overview_metrics(_: str):
        return _forecast_metrics_bar(data)

    @app.callback(
        Output(IDS["ml"]["feature_importance"], "figure"),
        Output(IDS["ml"]["residual_graph"], "figure"),
        Output(IDS["ml"]["metrics_table"], "data"),
        Input(IDS["ml"]["model_dropdown"], "value"),
    )
    def _ml_section(model: Optional[str]):
        importance = data.ml_feature_importance
        if importance.empty:
            importance_fig = _empty_figure("No feature importance data")
        else:
            importance_fig = px.bar(
                importance.head(15),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#00B386", "#1E90FF"],
                title="Feature importance",
            )
            importance_fig.update_yaxes(categoryorder="total ascending")
            importance_fig = _apply_fig_style(importance_fig, height=480)

        predictions = data.ml_split_predictions.copy()
        residual_fig = _empty_figure("No predictions available")
        if not predictions.empty and model and model in predictions.columns:
            actual_candidates = ["Actual", "actual", "y_true", "target"]
            actual_ml_col = next((col for col in actual_candidates if col in predictions.columns), None)
            if actual_ml_col:
                predictions["Actual"] = pd.to_numeric(predictions[actual_ml_col], errors="coerce")
                predictions["Forecast"] = pd.to_numeric(predictions[model], errors="coerce")
                predictions.dropna(subset=["Actual", "Forecast"], inplace=True)
                if not predictions.empty:
                    predictions["Residual"] = predictions["Forecast"] - predictions["Actual"]
                    residual_fig = go.Figure()
                    residual_fig.add_trace(
                        go.Scatter(
                            x=predictions["timestamp"],
                            y=predictions["Residual"],
                            mode="lines",
                            name="Residual",
                            line=dict(color="#F4B400"),
                        )
                    )
                    residual_fig.add_hline(y=0, line_color="#6b7280", line_dash="dot")
                    residual_fig.update_layout(title=f"Residuals over time — {model}")
                    residual_fig = _apply_fig_style(residual_fig, height=360)

        metrics = data.ml_split_metrics
        table_data: List[dict] = []
        if not metrics.empty:
            filtered = metrics if "model_name" not in metrics.columns or not model else metrics[metrics["model_name"] == model]
            if filtered.empty:
                filtered = metrics
            top_row = filtered.iloc[0]
            table_data = [
                {"metric": "Model", "value": top_row.get("model_name", model or "--")},
                {"metric": "MAE", "value": format_number(top_row.get("MAE"))},
                {"metric": "RMSE", "value": format_number(top_row.get("RMSE"))},
                {"metric": "nRMSE", "value": format_number(top_row.get("nRMSE"))},
            ]
        return importance_fig, residual_fig, table_data

    @app.callback(
        Output(IDS["forecast"]["timeseries_graph"], "figure"),
        Output(IDS["forecast"]["daily_error_graph"], "figure"),
        Input(IDS["forecast"]["model_dropdown"], "value"),
    )
    def _forecast_section(model: Optional[str]):
        df = _get_model_predictions(data, model)
        if df.empty:
            return _empty_figure("No forecast predictions available"), _empty_figure("No forecast metrics available")

        fig = go.Figure()
        actual_candidates = ["Actual", "actual", "y_true", "target"]
        actual_forecast_col = next((col for col in actual_candidates if col in df.columns), None)
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=pd.to_numeric(df[actual_forecast_col], errors="coerce") if actual_forecast_col else pd.Series(dtype=float),
                name="Actual",
                marker_color="#1E90FF",
                opacity=0.75,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=pd.to_numeric(df["y_pred"], errors="coerce"),
                name="Predicted",
                mode="lines",
                line=dict(color="#F4B400", width=2),
            )
        )
        fig.update_layout(title=f"Daily forecast overlay — {df['model_name'].iloc[0] if 'model_name' in df.columns else model}")
        fig = _apply_fig_style(fig, height=460)

        daily = df.copy()
        if "forecast_day" in daily.columns and actual_forecast_col:
            daily["forecast_day"] = pd.to_datetime(daily["forecast_day"])
            error_group = (
                daily.groupby("forecast_day")
                .apply(
                    lambda g: np.sqrt(
                        np.nanmean(
                            (pd.to_numeric(g["y_pred"], errors="coerce") - pd.to_numeric(g[actual_forecast_col], errors="coerce"))
                            ** 2
                        )
                    )
                )
                .dropna()
            )
        else:
            error_group = pd.Series(dtype=float)
        if error_group.empty:
            error_fig = _empty_figure("No daily error data available")
        else:
            error_fig = px.bar(
                error_group.reset_index(name="RMSE"),
                x="forecast_day",
                y="RMSE",
                labels={"forecast_day": "Forecast day", "RMSE": "RMSE"},
                title="Daily RMSE",
                color="RMSE",
                color_continuous_scale=["#00B386", "#1E90FF"],
            )
            error_fig = _apply_fig_style(error_fig, height=360)
        return fig, error_fig

    @app.callback(
        Output(IDS["overview"]["seasonality_graph"], "figure"),
        Output(IDS["overview"]["profile_graph"], "figure"),
        Input(IDS["routing"]["location"], "pathname"),
    )
    def _overview_seasonality(_: str):
        clean_df = data.cleaned_dataset.copy()
        if clean_df.empty:
            empty = placeholder_fig("No data available")
            return empty, empty
        if "timestamp" in clean_df.columns:
            clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
            clean_df = clean_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        seasonality = make_demand_seasonality(clean_df)
        weekly_fig = seasonality.get("weekly") or placeholder_fig("No seasonality data")
        profile_fig = _make_profile_overlay(clean_df)
        return weekly_fig, profile_fig

    @app.callback(
        Output(IDS["optimization"]["kpi_cost"], "children"),
        Output(IDS["optimization"]["kpi_bought"], "children"),
        Output(IDS["optimization"]["kpi_sold"], "children"),
        Output(IDS["optimization"]["kpi_cycles"], "children"),
        Output(IDS["optimization"]["soc_graph"], "figure"),
        Output(IDS["optimization"]["energy_graph"], "figure"),
        Input(IDS["optimization"]["scenario_dropdown"], "value"),
    )
    def _optimization_section(scenario: Optional[str]):
        summary = data.storage_summary
        if summary.empty:
            empty_fig = _empty_figure("No storage optimisation data available")
            return ("--", "--", "--", "--", empty_fig, empty_fig)
        if scenario and scenario in summary["Scenario"].values:
            scenario_row = summary[summary["Scenario"] == scenario].iloc[0]
        else:
            scenario_row = summary.iloc[0]

        cost = format_number(scenario_row.get("Total_cost_EUR"), suffix=" €")
        bought = format_number(scenario_row.get("Energy_bought_kWh"), suffix=" kWh")
        sold = format_number(scenario_row.get("Energy_sold_kWh"), suffix=" kWh")
        cycles = format_number(scenario_row.get("Battery_cycles"))

        soc_fig = px.bar(
            summary,
            x="Scenario",
            y=["SOC_min_kWh", "SOC_max_kWh"],
            title="Battery state of charge",
            labels={"value": "Energy (kWh)", "Scenario": "Scenario"},
        )
        soc_fig.update_layout(barmode="group")
        if "Scenario" in summary.columns and not summary.empty:
            highlight_mask = summary["Scenario"] == scenario_row.get("Scenario")
            highlight_colors = ["#1E90FF" if flag else "#cbd5f5" for flag in highlight_mask]
            for trace in soc_fig.data:
                trace.update(marker_color=highlight_colors)
        soc_fig = _apply_fig_style(soc_fig, height=360)

        energy_fig = px.bar(
            summary,
            x="Scenario",
            y=["Energy_bought_kWh", "Energy_sold_kWh"],
            title="Grid import vs export",
            labels={"value": "Energy (kWh)", "Scenario": "Scenario"},
        )
        energy_fig.update_layout(barmode="group")
        if "Scenario" in summary.columns and not summary.empty:
            highlight_mask = summary["Scenario"] == scenario_row.get("Scenario")
            highlight_colors = ["#00B386" if flag else "#c4f5e5" for flag in highlight_mask]
            for trace in energy_fig.data:
                trace.update(marker_color=highlight_colors)
        energy_fig = _apply_fig_style(energy_fig, height=360)

        return cost, bought, sold, cycles, soc_fig, energy_fig

    @app.callback(
        Output(IDS["data"]["time_series"], "figure"),
        Output(IDS["data"]["missing_heatmap"], "figure"),
        Output(IDS["data"]["missing_summary"], "children"),
        Output(IDS["data"]["summary_stats"], "data"),
        Output(IDS["data"]["summary_stats"], "columns"),
        Output(IDS["data"]["schema_table"], "data"),
        Output(IDS["data"]["schema_table"], "columns"),
        Output(IDS["data"]["raw_head"], "data"),
        Output(IDS["data"]["raw_head"], "columns"),
        Output(IDS["data"]["totals_rows"], "children"),
        Output(IDS["data"]["totals_range"], "children"),
        Output(IDS["data"]["totals_missing"], "children"),
        Output(IDS["data"]["clean_log"], "children"),
        Input(IDS["data"]["date_range"], "start_date"),
        Input(IDS["data"]["date_range"], "end_date"),
        Input(IDS["data"]["variable_select"], "value"),
        Input(IDS["data"]["show_mode"], "value"),
    )
    def _data_page_update(start_date: Optional[str], end_date: Optional[str], variables, mode: Optional[str]):
        raw_df = data.raw_dataset.copy()
        clean_df = data.cleaned_dataset.copy()

        if raw_df.empty:
            empty_fig = _empty_figure("No source data available")
            clean_log = _format_clean_log(data.cleaning_log)
            return (
                empty_fig,
                empty_fig,
                html.Span("No data loaded."),
                [],
                [],
                [],
                [],
                [],
                [],
                "0",
                "--",
                "0",
                clean_log,
            )

        if isinstance(variables, str):
            selected = [variables]
        elif isinstance(variables, list):
            selected = [str(v) for v in variables if isinstance(v, str)]
        else:
            selected = []

        available_columns = [col for col in raw_df.columns if col != "timestamp"]
        if not selected:
            priority = [col for col in available_columns if col.lower() in {"demand", "pv"}]
            selected = priority or available_columns[: min(3, len(available_columns))]

        start_ts = pd.to_datetime(start_date) if start_date else None
        end_ts = pd.to_datetime(end_date) if end_date else None
        raw_filtered = filter_time_range(raw_df, start_ts, end_ts)
        clean_filtered = filter_time_range(clean_df, start_ts, end_ts)
        overall_missing_info = compute_missing_summary(raw_filtered)
        missing_summary_dict = overall_missing_info.copy()

        show_mode = (mode or "overlay").lower()

        # --- Time series figure -------------------------------------------------
        time_series_fig: go.Figure
        if raw_filtered.empty and (show_mode != "cleaned" or clean_filtered.empty):
            time_series_fig = _empty_figure("No data for selected period")
        else:
            time_series_fig = go.Figure()
            palette = ["#1E90FF", "#00B386", "#F4B400", "#EF6C00", "#8E24AA"]
            for idx, col in enumerate(selected):
                color = palette[idx % len(palette)]
                if show_mode in {"raw", "overlay"} and col in raw_filtered.columns:
                    time_series_fig.add_trace(
                        go.Scatter(
                            x=raw_filtered["timestamp"],
                            y=pd.to_numeric(raw_filtered[col], errors="coerce"),
                            name=f"{col} (raw)",
                            mode="lines",
                            line=dict(color=color, width=2.2, dash="solid"),
                        )
                    )
                if show_mode in {"cleaned", "overlay"} and col in clean_filtered.columns:
                    time_series_fig.add_trace(
                        go.Scatter(
                            x=clean_filtered["timestamp"],
                            y=pd.to_numeric(clean_filtered[col], errors="coerce"),
                            name=f"{col} (cleaned)",
                            mode="lines",
                            line=dict(color=color, width=2.0, dash="dot"),
                        )
                    )
            time_series_fig.update_layout(title="Time Series Comparison")
            time_series_fig = _apply_fig_style(time_series_fig, height=440)
            time_series_fig.update_layout(uirevision="data-timeseries")

        # --- Missing heatmap ----------------------------------------------------
        subset_cols = [col for col in selected if col in raw_filtered.columns]
        if not subset_cols:
            subset_cols = [col for col in available_columns if col in raw_filtered.columns][:3]
        if raw_filtered.empty or not subset_cols:
            missing_fig = _empty_figure("No data for missingness analysis")
            missing_summary = html.Span("No missing values in selected period.")
        else:
            subset_df = raw_filtered[["timestamp"] + subset_cols] if "timestamp" in raw_filtered.columns else raw_filtered[subset_cols].copy()
            missing_fig = make_missingness_heatmap(subset_df, export_path=None)
            missing_summary_dict = compute_missing_summary(raw_filtered[subset_cols])
            summary_items = [
                html.Li(
                    f"{col}: {count} values ({(count / missing_summary_dict['total_rows'] * 100):.2f}%)"
                    if missing_summary_dict["total_rows"]
                    else f"{col}: {count} missing values"
                )
                for col, count in missing_summary_dict["columns_with_missing"].items()
            ]
            if not summary_items:
                summary_items = [html.Li("No missing values in selected columns.")]
            missing_summary = html.Ul(summary_items, className="clean-log")

        # --- Summary statistics -------------------------------------------------
        summary_data: List[dict]
        summary_columns: List[dict]
        stats_df = data.descriptive_stats.copy()
        if stats_df.empty:
            summary_data = []
            summary_columns = []
        else:
            working = stats_df.copy()
            if "variable" in working.columns and selected:
                filtered = working[working["variable"].isin(selected)]
                if not filtered.empty:
                    working = filtered
            working = working.reset_index(drop=True)
            numeric_cols = working.select_dtypes(include=["number"]).columns
            if not working.empty and len(numeric_cols) > 0:
                working.loc[:, numeric_cols] = working.loc[:, numeric_cols].round(3)
            summary_data = working.to_dict("records")
            summary_columns = [{"name": col.replace("_", " ").title(), "id": col} for col in working.columns]

        # --- Schema table -------------------------------------------------------
        schema_source = compute_schema_table(raw_filtered if not raw_filtered.empty else raw_df)
        schema_data, schema_columns = _format_schema_table(schema_source, selected)

        # --- Data sample --------------------------------------------------------
        table_source = clean_filtered if show_mode == "cleaned" else raw_filtered
        table_columns_available = table_source.columns.tolist()
        table_display_cols: List[str] = []
        if "timestamp" in table_columns_available:
            table_display_cols.append("timestamp")
        table_display_cols.extend([col for col in selected if col in table_columns_available and col not in table_display_cols])
        if not table_display_cols:
            table_display_cols = table_columns_available[: min(5, len(table_columns_available))]
        table_df = table_source.loc[:, table_display_cols].head(10).copy() if table_display_cols else table_source.head(10).copy()
        numeric_cols = table_df.select_dtypes(include=["number"]).columns
        table_df.loc[:, numeric_cols] = table_df.loc[:, numeric_cols].round(3)
        table_data = table_df.to_dict("records")
        table_columns = [{"name": col, "id": col} for col in table_df.columns]
        # --- Totals -------------------------------------------------------------
        totals_info = compute_totals_summary(raw_filtered)
        raw_rows = totals_info["rows"]
        clean_rows = len(clean_filtered)
        totals_rows = f"{raw_rows:,} raw | {clean_rows:,} cleaned".replace(",", " ")
        totals_range = totals_info.get("date_range", "--")
        missing_count = overall_missing_info.get("total_missing", 0)
        totals_missing = f"{int(missing_count):,}".replace(",", " ") if missing_count else "0"

        clean_log_children = _format_clean_log(data.cleaning_log)

        return (
            time_series_fig,
            missing_fig,
            missing_summary,
            summary_data,
            summary_columns,
            schema_data,
            schema_columns,
            table_data,
            table_columns,
            totals_rows,
            totals_range,
            totals_missing,
            clean_log_children,
        )

    @app.callback(
        Output(IDS["data"]["download_target"], "data"),
        Input(IDS["data"]["download_cleaned"], "n_clicks"),
        prevent_initial_call=True,
    )
    def _download_cleaned_dataset(n_clicks):
        if not n_clicks or data.cleaned_dataset.empty:
            return no_update
        return dcc.send_data_frame(data.cleaned_dataset.to_csv, "cleaned_dataset.csv", index=False)

    @app.callback(
        [
            Output(IDS["pipeline"]["pipeline_graph"], "figure"),
            Output(IDS["pipeline"]["pipeline_metrics"], "figure"),
            Output(IDS["pipeline"]["exog_graph"], "figure"),
            Output(IDS["pipeline"]["exog_metrics"], "figure"),
            Output(IDS["pipeline"]["exog_importance"], "figure"),
        ],
        [Input(IDS["pipeline"]["tabs"], "active_tab")],
    )
    def _update_pipeline_graphs(active_tab):
        # Pipeline Graph
        if data.pipeline_predictions.empty:
            fig_pipe = _empty_figure("No pipeline predictions available")
        else:
            df = data.pipeline_predictions
            fig_pipe = go.Figure()
            if "Actual" in df.columns:
                fig_pipe.add_trace(go.Scatter(x=df["timestamp"], y=df["Actual"], name="Actual", line=dict(color="black", width=1)))
            for col in df.columns:
                if col not in ["timestamp", "Actual"]:
                    fig_pipe.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=col, mode="lines", opacity=0.7))
            fig_pipe = _apply_fig_style(fig_pipe)

        # Pipeline Metrics
        if data.pipeline_metrics.empty:
            fig_pipe_metrics = _empty_figure("No pipeline metrics available")
        else:
            df = data.pipeline_metrics.copy()
            if "model_name" not in df.columns:
                df["model_name"] = df.index

            # Determine available metrics
            y_cols = []
            for metric in ["RMSE", "MAE", "nRMSE"]:
                if metric in df.columns:
                    y_cols.append(metric)
                elif f"{metric}_mean" in df.columns:
                    y_cols.append(f"{metric}_mean")

            if not y_cols:
                fig_pipe_metrics = _empty_figure("No valid metrics columns found")
            else:
                fig_pipe_metrics = px.bar(df, x="model_name", y=y_cols, barmode="group")
                fig_pipe_metrics = _apply_fig_style(fig_pipe_metrics)

        # Exog Graph
        if data.exog_predictions.empty:
            fig_exog = _empty_figure("No exogenous predictions available")
        else:
            df = data.exog_predictions
            fig_exog = go.Figure()
            if "Actual" in df.columns:
                fig_exog.add_trace(go.Scatter(x=df["timestamp"], y=df["Actual"], name="Actual", line=dict(color="black", width=1)))
            for col in df.columns:
                if col not in ["timestamp", "Actual"]:
                    fig_exog.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=col, mode="lines", opacity=0.7))
            fig_exog = _apply_fig_style(fig_exog)

        # Exog Metrics
        if data.exog_metrics.empty:
            fig_exog_metrics = _empty_figure("No exogenous metrics available")
        else:
            df = data.exog_metrics.copy()
            if "model_name" not in df.columns:
                df["model_name"] = df.index

            # Determine available metrics
            y_cols = []
            for metric in ["RMSE", "MAE", "nRMSE"]:
                if metric in df.columns:
                    y_cols.append(metric)
                elif f"{metric}_mean" in df.columns:
                    y_cols.append(f"{metric}_mean")

            if not y_cols:
                fig_exog_metrics = _empty_figure("No valid metrics columns found")
            else:
                fig_exog_metrics = px.bar(df, x="model_name", y=y_cols, barmode="group")
                fig_exog_metrics = _apply_fig_style(fig_exog_metrics)

        # Exog Importance
        if data.exog_importance.empty:
            fig_exog_imp = _empty_figure("No feature importance available")
        else:
            df = data.exog_importance.head(20).copy()
            fig_exog_imp = px.bar(df, x="importance", y="feature", orientation="h")
            fig_exog_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            fig_exog_imp = _apply_fig_style(fig_exog_imp)

        return fig_pipe, fig_pipe_metrics, fig_exog, fig_exog_metrics, fig_exog_imp

    @app.callback(
        [
            Output(IDS["optimization"]["summary_table_11"], "data"),
            Output(IDS["optimization"]["summary_table_11"], "columns"),
            Output(IDS["optimization"]["sensitivity_graph_11"], "figure"),
        ],
        [Input(IDS["optimization"]["scenario_dropdown"], "value")],  # Dummy input to trigger on load
    )
    def _update_optimization_11(_):
        # Summary Table
        if data.optim_summary_11.empty:
            table_data = []
            table_columns = []
        else:
            df = data.optim_summary_11.copy()
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].round(2)
            table_data = df.to_dict("records")
            table_columns = [{"name": col.replace("_", " "), "id": col} for col in df.columns]

        # Sensitivity Graph
        if data.optim_sensitivity_11.empty:
            fig_sens = _empty_figure("No sensitivity data available")
        else:
            df_sens = data.optim_sensitivity_11.sort_values("Battery_capacity_kWh")
            fig_sens = px.line(
                df_sens,
                x="Battery_capacity_kWh",
                y="Total_cost",
                markers=True,
                title="Cost Sensitivity to Battery Capacity"
            )
            fig_sens = _apply_fig_style(fig_sens)

        return table_data, table_columns, fig_sens

    app.logger.info("Registered %d callbacks", len(app.callback_map))
