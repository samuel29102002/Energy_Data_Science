"""Reusable Plotly figure helpers used across the dashboard and export scripts."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

DEFAULT_TEMPLATE = "plotly_white"
DEFAULT_FONT = "Inter, system-ui, -apple-system, sans-serif"
TITLE_FONT = "CMU Serif, 'Times New Roman', serif"


@dataclass(frozen=True)
class SavedFigure:
    path: Path
    generated: bool
    fallback: Optional[Path] = None


def _ensure_datetime(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _basic_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family=TITLE_FONT, size=22, color="#1a1a1a")),
        template=DEFAULT_TEMPLATE,
        hovermode="x unified",
        autosize=True,
        height=height,
        margin=dict(l=50, r=30, t=70, b=50),
        font=dict(family=DEFAULT_FONT, size=12, color="#333"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


def placeholder_fig(title: str = "No data yet") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        showarrow=False,
        font=dict(size=16, color="#666"),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="middle",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", uirevision=True)
    return fig


def save_plotly_static(fig: go.Figure, path: Union[str, Path], width: int = 1200, height: int = 520, scale: int = 2) -> SavedFigure:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(target), width=width, height=height, scale=scale, engine="kaleido")
        return SavedFigure(target, True)
    except Exception:  # pragma: no cover - kaleido optional
        fallback = target.with_suffix(".html")
        fig.write_html(str(fallback))
        return SavedFigure(fallback, False, fallback)


def load_saved_figure(path: Union[str, Path]) -> Union[go.Figure, "html.Img", None]:
    from dash import html  # imported lazily to avoid Dash dependency during scripts

    p = Path(path)
    if not p.exists():
        return None
    suffix = p.suffix.lower()
    if suffix in {".json"}:
        return pio.read_json(p)
    if suffix in {".png", ".jpg", ".jpeg"}:
        encoded = base64.b64encode(p.read_bytes()).decode("ascii")
        return html.Img(src=f"data:image/{suffix[1:]};base64,{encoded}", style={"width": "100%", "height": "100%", "objectFit": "contain"})
    if suffix in {".html"}:
        return html.Iframe(srcDoc=p.read_text(encoding="utf-8"), style={"width": "100%", "height": "100%", "border": "0"})
    # default attempt to load as plotly figure
    try:
        return pio.read_json(p)
    except Exception:  # pragma: no cover - best effort
        return None


def make_demand_pv_timeseries(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    demand_col: str = "Demand",
    pv_col: str = "pv",
    export_path: Union[str, Path, None] = None,
) -> go.Figure:
    if df.empty or demand_col not in df.columns:
        return placeholder_fig("No demand data available")
    work = _ensure_datetime(df)
    if start is not None:
        work = work[work["timestamp"] >= start]
    if end is not None:
        work = work[work["timestamp"] <= end]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["timestamp"],
            y=work[demand_col],
            name="Demand",
            mode="lines",
            line=dict(color="#1F77B4", width=2.2),
        )
    )
    if pv_col in work.columns:
        fig.add_trace(
            go.Scatter(
                x=work["timestamp"],
                y=work[pv_col],
                name="PV",
                mode="lines",
                line=dict(color="#2CA02C", width=2, dash="solid"),
            )
        )
    _basic_layout(fig, "Demand vs PV over time")
    fig.update_yaxes(title="kWh")
    if export_path is not None:
        save_plotly_static(fig, export_path)
    return fig


def make_demand_seasonality(
    df: pd.DataFrame,
    demand_col: str = "Demand",
    export_dir: Union[str, Path, None] = None,
) -> Dict[str, go.Figure]:
    if df.empty or demand_col not in df.columns:
        return {"diurnal": placeholder_fig("No seasonality data"), "weekly": placeholder_fig("No seasonality data")}
    work = _ensure_datetime(df)
    work["hour"] = work["timestamp"].dt.hour
    work["weekday"] = work["timestamp"].dt.day_name()

    diurnal = work.groupby("hour")[demand_col].mean().reset_index()
    fig_diurnal = go.Figure(
        go.Scatter(
            x=diurnal["hour"],
            y=diurnal[demand_col],
            mode="lines+markers",
            marker=dict(size=7, color="#1F77B4"),
            line=dict(width=2.5, color="#1F77B4"),
        )
    )
    _basic_layout(fig_diurnal, "Average demand by hour")
    fig_diurnal.update_xaxes(dtick=1)
    fig_diurnal.update_yaxes(title="kWh")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = (
        work.pivot_table(index="weekday", columns="hour", values=demand_col, aggfunc="mean")
        .reindex(weekday_order, axis=0)
    )
    fig_weekly = px.imshow(
        heatmap_data,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig_weekly.update_layout(coloraxis_colorbar=dict(title="kWh"))
    _basic_layout(fig_weekly, "Demand heatmap by weekday/hour")
    fig_weekly.update_layout(margin=dict(l=60, r=20, t=60, b=80))

    if export_dir is not None:
        export_dir = Path(export_dir)
        save_plotly_static(fig_diurnal, export_dir / "02_demand_diurnal_profile.png")
        save_plotly_static(fig_weekly, export_dir / "02_demand_weekday_heatmap.png")

    return {"diurnal": fig_diurnal, "weekly": fig_weekly}


def make_missingness_heatmap(
    df: pd.DataFrame,
    freq: str = "D",
    export_path: Union[str, Path, None] = None,
) -> go.Figure:
    if df.empty:
        return placeholder_fig("No data for missingness")
    work = _ensure_datetime(df)
    if "timestamp" not in work.columns or work["timestamp"].isna().all():
        missing_pct = df.isna().mean().sort_values(ascending=False)
        fig = px.bar(
            missing_pct.reset_index(),
            x="index",
            y=0,
            labels={"index": "Feature", "0": "Missing %"},
            title="Missingness per feature",
        )
        fig.update_yaxes(ticksuffix="%")
        fig.update_layout(template=DEFAULT_TEMPLATE, height=420, margin=dict(l=60, r=20, t=60, b=80))
        if export_path is not None:
            save_plotly_static(fig, export_path)
        return fig

    work = work.set_index("timestamp")
    missing_matrix = work.isna().astype(float)
    heatmap = missing_matrix.resample(freq).mean() * 100.0
    heatmap = heatmap.transpose()
    fig = px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale="Reds",
        labels=dict(x="Date", y="Feature", color="Missing %"),
    )
    _basic_layout(fig, "Missingness heatmap")
    fig.update_layout(margin=dict(l=120, r=20, t=60, b=80))
    fig.update_coloraxes(colorbar=dict(ticksuffix="%"))
    if export_path is not None:
        save_plotly_static(fig, export_path)
    return fig


def apply_dashboard_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(uirevision=True, hovermode="x unified")
    return fig


def make_responsive_figure(
    fig: go.Figure,
    uirevision: Union[bool, str] = True,
    autosize: bool = True,
    margin: Optional[Dict[str, int]] = None,
) -> go.Figure:
    if margin is None:
        margin = {"l": 40, "r": 40, "t": 60, "b": 40}
    fig.update_layout(uirevision=uirevision, autosize=autosize, margin=margin)
    return fig


def apply_energy_theme(
    fig: go.Figure,
    template: str = DEFAULT_TEMPLATE,
    font_family: str = DEFAULT_FONT,
    colorway: Optional[Iterable[str]] = None,
) -> go.Figure:
    if colorway is None:
        colorway = (
            "#1F77B4",
            "#2CA02C",
            "#FF7F0E",
            "#D62728",
            "#9467BD",
            "#8C564B",
            "#E377C2",
            "#7F7F7F",
        )
    fig.update_layout(template=template, font=dict(family=font_family), colorway=list(colorway))
    return fig


def create_empty_figure(message: str = "No data available") -> go.Figure:
    return placeholder_fig(message)
