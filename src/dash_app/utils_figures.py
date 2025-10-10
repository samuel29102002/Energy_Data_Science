"""
Figure utilities for Dash 2.15.0+ compatibility.

This module provides helper functions to ensure figure objects
are properly configured for Dash 2.15.0+, where uirevision must
be set in the figure layout, not as a Graph component parameter.
"""
from typing import Any, Union, Optional, Dict
import plotly.graph_objects as go


def placeholder_fig(title: str = "No data yet") -> go.Figure:
    """
    Create a placeholder figure for empty states.

    Parameters
    ----------
    title : str, default="No data yet"
        Message to display in the placeholder

    Returns
    -------
    go.Figure
        Empty figure with centered message
    """
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
        yanchor="middle"
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        uirevision=True
    )
    return fig


def apply_dashboard_style(fig: go.Figure) -> go.Figure:
    """
    Apply standard dashboard styling to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to style

    Returns
    -------
    go.Figure
        Styled figure with uirevision and hovermode
    """
    fig.update_layout(
        uirevision=True,
        hovermode="x unified"
    )
    return fig


def make_responsive_figure(
    fig: go.Figure,
    uirevision: Union[bool, str] = True,
    autosize: bool = True,
    margin: Optional[Dict[str, int]] = None
) -> go.Figure:
    """
    Configure a figure for responsive display with UI state preservation.

    This helper ensures Dash 2.15.0+ compatibility by setting uirevision
    inside the figure layout (not as a Graph component argument).

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to configure
    uirevision : bool or str, default=True
        Preserve user interactions (zoom, pan) across updates.
        - True: Preserve state using default key
        - str: Preserve state using custom key
        - False: Reset state on every update
    autosize : bool, default=True
        Enable automatic figure resizing
    margin : dict, optional
        Custom margins. Defaults to {'l': 40, 'r': 40, 't': 60, 'b': 40}

    Returns
    -------
    go.Figure
        The modified figure with responsive layout settings

    Examples
    --------
    >>> fig = go.Figure(data=[go.Scatter(x=[1,2,3], y=[4,5,6])])
    >>> fig = make_responsive_figure(fig)
    >>> # Figure now has uirevision=True in layout

    >>> # Use custom margin
    >>> fig = make_responsive_figure(fig, margin={'l': 20, 'r': 20, 't': 40, 'b': 20})
    """
    if margin is None:
        margin = {'l': 40, 'r': 40, 't': 60, 'b': 40}

    fig.update_layout(
        uirevision=uirevision,
        autosize=autosize,
        margin=margin,
    )

    return fig


def apply_energy_theme(
    fig: go.Figure,
    template: str = "plotly_white",
    font_family: str = "Inter, system-ui, -apple-system, sans-serif",
    colorway: Optional[list] = None
) -> go.Figure:
    """
    Apply energy-themed styling to a figure.

    Parameters
    ----------
    fig : go.Figure
        The figure to style
    template : str, default="plotly_white"
        Base Plotly template
    font_family : str, optional
        CSS font family string
    colorway : list of str, optional
        Custom color sequence. Defaults to energy-themed palette.

    Returns
    -------
    go.Figure
        The styled figure
    """
    if colorway is None:
        # Energy-themed colorway
        colorway = [
            '#1F77B4',  # Primary blue
            '#2CA02C',  # Green (renewable)
            '#FF7F0E',  # Orange (energy)
            '#D62728',  # Red (demand)
            '#9467BD',  # Purple
            '#8C564B',  # Brown
            '#E377C2',  # Pink
            '#7F7F7F',  # Gray
        ]

    fig.update_layout(
        template=template,
        font=dict(family=font_family),
        colorway=colorway,
    )

    return fig


def create_empty_figure(message: str = "No data available") -> go.Figure:
    """
    Create an empty placeholder figure with a message.

    Parameters
    ----------
    message : str, default="No data available"
        Message to display in the empty figure

    Returns
    -------
    go.Figure
        Empty figure with centered message
    """
    fig = go.Figure()
    fig.update_layout(
        title={'text': message, 'x': 0.5, 'xanchor': 'center'},
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[
            {
                'text': message,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 16, 'color': '#666'},
                'xanchor': 'center',
                'yanchor': 'middle',
            }
        ],
    )
    return fig
