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
]

