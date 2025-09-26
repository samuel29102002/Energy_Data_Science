from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

Style = Literal["academic", "dashboard"]

# Shared colour palette for energy-themed visuals
ENERGY_COLORS = {
    "solar": "#FFA500",
    "grid": "#1f77b4",
    "battery": "#2ca02c",
    "storage": "#7f7f7f",
    "residual": "#6c757d",
}


def _apply_style(fig: go.Figure, style: Style = "academic") -> go.Figure:
    """Apply consistent styling to a Plotly figure."""
    if style == "academic":
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="CMU Serif, 'Times New Roman', serif", size=14, color="#222"),
            legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#d0d0d0", borderwidth=1),
            margin=dict(t=60, r=30, b=50, l=70),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5", zeroline=False)
    else:
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, 'Open Sans', sans-serif", size=13, color="#212529"),
            legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor="#e3e6ea", borderwidth=1),
            margin=dict(t=60, r=30, b=50, l=60),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f1f3f5", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f3f5", zeroline=False)
    return fig


def plot_stl_components(
    timestamps: Iterable[pd.Timestamp] | pd.Index,
    trend: Iterable[float],
    seasonal: Iterable[float],
    resid: Iterable[float],
    title: str,
    style: Style = "academic",
) -> go.Figure:
    """Return a figure with STL components stacked vertically."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Trend", "Seasonal", "Residual"))
    fig.add_trace(
        go.Scatter(x=list(timestamps), y=list(trend), mode="lines", name="Trend", line=dict(color=ENERGY_COLORS["grid"], width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=list(timestamps), y=list(seasonal), mode="lines", name="Seasonal", line=dict(color=ENERGY_COLORS["solar"], width=2)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=list(timestamps), y=list(resid), mode="lines", name="Residual", line=dict(color=ENERGY_COLORS["residual"], width=1.5)),
        row=3,
        col=1,
    )
    fig.update_layout(title=title, height=900, hovermode="x unified")
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    fig.update_xaxes(title_text="Timestamp", row=3, col=1)
    return _apply_style(fig, style)


def plot_typical_profiles_weekday_weekend(
    weekday: pd.Series,
    weekend: pd.Series,
    value_label: str = "Demand (kW)",
    style: Style = "academic",
) -> go.Figure:
    """Return typical hourly profiles for weekday vs weekend."""
    hour_index = np.arange(24)
    weekday = weekday.reindex(hour_index).sort_index()
    weekend = weekend.reindex(hour_index).sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=hour_index, y=weekday.values, mode="lines", name="Weekday", line=dict(color=ENERGY_COLORS["grid"], width=2.4))
    )
    fig.add_trace(
        go.Scatter(x=hour_index, y=weekend.values, mode="lines", name="Weekend", line=dict(color=ENERGY_COLORS["battery"], width=2.4, dash="dash"))
    )
    fig.update_layout(title="Typical hourly demand profile", hovermode="x unified")
    fig.update_xaxes(title_text="Hour of day")
    fig.update_yaxes(title_text=value_label)
    return _apply_style(fig, style)


def plot_typical_profiles_monthly(
    profile_df: pd.DataFrame,
    value_label: str = "Demand (kW)",
    style: Style = "academic",
) -> go.Figure:
    """Return a multi-line plot of monthly hourly demand profiles.

    Expects a dataframe with columns ['month', 'hour', 'value'].
    """
    if not {"month", "hour", "value"}.issubset(profile_df.columns):
        raise ValueError("profile_df must contain columns 'month', 'hour', 'value'")

    month_order = list(pd.date_range("2023-01-01", periods=12, freq="MS").month_name())
    fig = go.Figure()
    for idx, month in enumerate(month_order):
        month_data = profile_df[profile_df["month"] == month]
        if month_data.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=month_data["hour"],
                y=month_data["value"],
                mode="lines",
                name=month,
                line=dict(color=_monthly_color(idx), width=1.8),
            )
        )
    fig.update_layout(title="Monthly typical hourly demand profiles", hovermode="x unified")
    fig.update_xaxes(title_text="Hour of day")
    fig.update_yaxes(title_text=value_label)
    return _apply_style(fig, style)


def _monthly_color(idx: int) -> str:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#ffa600",
        "#6a51a3",
    ]
    return palette[idx % len(palette)]


__all__ = [
    "ENERGY_COLORS",
    "plot_stl_components",
    "plot_typical_profiles_weekday_weekend",
    "plot_typical_profiles_monthly",
    "plot_acf_pacf",
    "plot_forecast_overlay",
    "plot_walkforward_panels",
    "plot_metrics_bar",
    "plot_feature_importance",
    "plot_forecast_overlay_multimodel",
    "plot_metrics_comparison",
    "plot_learning_curve",
    "plot_forecast_overlay_day",
    "plot_forecast_overlay_week",
    "plot_forecast_metrics",
]


def _empty_plot(title: str, style: Style = "academic") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[
            {
                "text": title,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ],
    )
    return _apply_style(fig, style)


def plot_acf_pacf(
    acf_df: pd.DataFrame,
    pacf_df: pd.DataFrame,
    title: str,
    style: Style = "academic",
) -> go.Figure:
    if acf_df.empty or pacf_df.empty:
        return _empty_plot("ACF/PACF unavailable", style=style)

    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    fig.add_trace(
        go.Bar(
            x=acf_df["lag"],
            y=acf_df["value"],
            marker_color=ENERGY_COLORS["grid"],
            name="ACF",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=pacf_df["lag"],
            y=pacf_df["value"],
            marker_color=ENERGY_COLORS["solar"],
            name="PACF",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Partial correlation", row=1, col=2)
    fig.update_layout(title=title, showlegend=False)
    return _apply_style(fig, style)


def plot_forecast_overlay(
    df24: pd.DataFrame,
    title: str,
    style: Style = "academic",
) -> go.Figure:
    if df24 is None or df24.empty:
        return _empty_plot("No forecast data", style=style)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df24["timestamp"],
            y=df24["y_true"],
            mode="lines+markers",
            name="Observed",
            line=dict(color=ENERGY_COLORS["grid"], width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df24["timestamp"],
            y=df24["y_pred"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=ENERGY_COLORS["solar"], width=2, dash="dash"),
            marker=dict(size=6),
        )
    )
    fig.update_layout(title=title, hovermode="x unified")
    fig.update_xaxes(title_text="Timestamp")
    default_label = "Demand (kW)"
    if "value_label" in df24.columns:
        label_series = df24["value_label"].dropna()
        y_label = label_series.iloc[0] if not label_series.empty else default_label
    else:
        y_label = default_label
    fig.update_yaxes(title_text=str(y_label))
    return _apply_style(fig, style)


def plot_walkforward_panels(
    pred_df: pd.DataFrame,
    style: Style = "academic",
) -> go.Figure:
    if pred_df is None or pred_df.empty:
        return _empty_plot("Walk-forward data unavailable", style=style)

    from plotly.subplots import make_subplots

    unique_days = sorted(pred_df["day_idx"].unique())
    n_days = len(unique_days)
    cols = 2
    rows = int(np.ceil(n_days / cols))

    fig = make_subplots(rows=rows, cols=cols, shared_yaxes=True, subplot_titles=[f"Day {d}" for d in unique_days])

    for i, day in enumerate(unique_days):
        row = i // cols + 1
        col = i % cols + 1
        subset = pred_df[pred_df["day_idx"] == day]
        fig.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=subset["y_true"],
                mode="lines",
                name=f"Observed {day}",
                line=dict(color=ENERGY_COLORS["grid"], width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=subset["y_pred"],
                mode="lines",
                name=f"Forecast {day}",
                line=dict(color=ENERGY_COLORS["solar"], width=2, dash="dash"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(title="Last-week walk-forward forecasts", hovermode="x unified")
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Demand (kW)")
    return _apply_style(fig, style)


def plot_metrics_bar(
    metrics_df: pd.DataFrame,
    style: Style = "academic",
) -> go.Figure:
    if metrics_df is None or metrics_df.empty:
        return _empty_plot("Metrics unavailable", style=style)

    if {"model_name", "metric", "value"}.issubset(metrics_df.columns):
        formatted = metrics_df.copy()
    else:
        id_vars = [c for c in metrics_df.columns if c not in {"MAE", "RMSE", "nRMSE"}]
        formatted = metrics_df.melt(id_vars=id_vars, value_vars=["MAE", "RMSE", "nRMSE"], var_name="metric", value_name="value")

    color_map = {
        "MAE": ENERGY_COLORS["grid"],
        "RMSE": ENERGY_COLORS["battery"],
        "nRMSE": ENERGY_COLORS["solar"],
    }

    fig = go.Figure()
    for metric, group in formatted.groupby("metric"):
        fig.add_trace(
            go.Bar(
                x=group["model_name"],
                y=group["value"],
                name=metric,
                marker_color=color_map.get(metric, ENERGY_COLORS["storage"]),
            )
        )

    fig.update_layout(
        title="Model performance comparison",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Score",
    )
    return _apply_style(fig, style)


def plot_feature_importance(
    importances_df: pd.DataFrame,
    top_n: int = 15,
    style: Style = "academic",
) -> go.Figure:
    if importances_df is None or importances_df.empty:
        return _empty_plot("Feature importance unavailable", style=style)

    df_sorted = importances_df.sort_values("importance", ascending=False).head(top_n)
    fig = go.Figure(
        go.Bar(
            x=df_sorted["importance"][::-1],
            y=df_sorted["feature"][::-1],
            orientation="h",
            marker_color=ENERGY_COLORS["solar"],
        )
    )
    fig.update_layout(title="Top feature importances", xaxis_title="Importance", yaxis_title="Feature")
    return _apply_style(fig, style)


def plot_forecast_overlay_multimodel(
    df24: pd.DataFrame,
    style: Style = "academic",
) -> go.Figure:
    if df24 is None or df24.empty:
        return _empty_plot("No forecast data", style=style)

    fig = go.Figure()
    color_map = {
        "Actual": ENERGY_COLORS["grid"],
        "XGBoost": ENERGY_COLORS["solar"],
        "Statistical": ENERGY_COLORS["battery"],
    }

    for series_name in ["Actual", "XGBoost", "Statistical"]:
        if series_name not in df24.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df24["timestamp"],
                y=df24[series_name],
                mode="lines+markers",
                name=series_name,
                line=dict(
                    color=color_map.get(series_name, ENERGY_COLORS["storage"]),
                    width=2,
                    dash="dash" if series_name != "Actual" else None,
                ),
                marker=dict(size=6),
            )
        )

    fig.update_layout(title="24h forecast comparison", hovermode="x unified")
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Demand (kW)")
    return _apply_style(fig, style)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    style: Style = "academic",
) -> go.Figure:
    if metrics_df is None or metrics_df.empty:
        return _empty_plot("Metrics unavailable", style=style)

    required = {"model", "metric", "value"}
    if not required.issubset(metrics_df.columns):
        raise ValueError("metrics_df must contain columns 'model', 'metric', 'value'")

    fig = go.Figure()
    color_map = {
        "MAE": ENERGY_COLORS["grid"],
        "RMSE": ENERGY_COLORS["battery"],
        "nRMSE": ENERGY_COLORS["solar"],
    }

    for metric, group in metrics_df.groupby("metric"):
        fig.add_trace(
            go.Bar(
                x=group["model"],
                y=group["value"],
                name=metric,
                marker_color=color_map.get(metric, ENERGY_COLORS["storage"]),
            )
        )
    fig.update_layout(
        title="ML vs Statistical metrics",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Score",
    )
    return _apply_style(fig, style)


def plot_learning_curve(
    curve_df: pd.DataFrame,
    style: Style = "academic",
) -> go.Figure:
    if curve_df is None or curve_df.empty:
        return _empty_plot("Learning curve unavailable", style=style)

    fig = go.Figure()
    for column in curve_df.columns:
        if column == "iteration":
            continue
        fig.add_trace(
            go.Scatter(
                x=curve_df["iteration"],
                y=curve_df[column],
                mode="lines",
                name=column,
            )
        )

    fig.update_layout(title="Training history", xaxis_title="Iteration", yaxis_title="Metric")
    return _apply_style(fig, style)


def plot_forecast_overlay_day(
    df24: pd.DataFrame,
    style: Style = "academic",
) -> go.Figure:
    if df24 is None or df24.empty:
        return _empty_plot("No forecast data", style=style)

    fig = go.Figure()
    columns = [col for col in df24.columns if col != "timestamp"]
    color_cycle = [
        ENERGY_COLORS["grid"],
        ENERGY_COLORS["solar"],
        ENERGY_COLORS["battery"],
        ENERGY_COLORS["storage"],
        "#17becf",
    ]
    for idx, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=df24["timestamp"],
                y=df24[col],
                mode="lines+markers",
                name=col,
                line=dict(color=color_cycle[idx % len(color_cycle)], width=2, dash=None if col == "Actual" else "dash"),
                marker=dict(size=6),
            )
        )
    fig.update_layout(title="24h forecast overlay", hovermode="x unified")
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Demand (kW)")
    return _apply_style(fig, style)


def plot_forecast_overlay_week(
    df_week: pd.DataFrame,
    style: Style = "academic",
) -> go.Figure:
    if df_week is None or df_week.empty:
        return _empty_plot("No week-long data", style=style)

    fig = go.Figure()
    columns = [col for col in df_week.columns if col != "timestamp"]
    palette = {
        "Actual": ENERGY_COLORS["grid"],
        "BestStat": ENERGY_COLORS["battery"],
        "BestML": ENERGY_COLORS["solar"],
    }
    for idx, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=df_week["timestamp"],
                y=df_week[col],
                mode="lines",
                name=col,
                line=dict(width=2, color=palette.get(col, palette.get(col.title(), None))),
            )
        )
    fig.update_layout(title="7-day rolling forecast overlay", hovermode="x unified")
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Demand (kW)")
    return _apply_style(fig, style)


def plot_forecast_metrics(
    metrics_df: pd.DataFrame,
    style: Style = "academic",
) -> go.Figure:
    if metrics_df is None or metrics_df.empty:
        return _empty_plot("Metrics unavailable", style=style)

    required = {"model_name", "metric", "value"}
    if not required.issubset(metrics_df.columns):
        raise ValueError("metrics_df must contain model_name, metric, value")

    color_map = {
        "MAE": ENERGY_COLORS["grid"],
        "RMSE": ENERGY_COLORS["battery"],
        "nRMSE": ENERGY_COLORS["solar"],
    }

    fig = go.Figure()
    for metric, group in metrics_df.groupby("metric"):
        fig.add_trace(
            go.Bar(
                x=group["model_name"],
                y=group["value"],
                name=metric,
                marker_color=color_map.get(metric, ENERGY_COLORS["storage"]),
            )
        )

    fig.update_layout(
        title="Forecast metrics comparison",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Score",
    )
    return _apply_style(fig, style)
