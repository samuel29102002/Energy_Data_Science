from __future__ import annotations

import sys
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dash_app.callbacks import register_callbacks  # noqa: E402
from src.dash_app.layout import create_layout  # noqa: E402
from src.dash_app.utils import TABLES_PATH, DashboardData, load_dashboard_data  # noqa: E402


def create_app(tables_path: Path | None = None) -> Dash:
    data: DashboardData = load_dashboard_data(tables_path)
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Energy Insights Dashboard",
    )
    app.layout = create_layout(data)
    register_callbacks(app, data)
    return app


def main() -> Dash:
    return create_app(TABLES_PATH)


app = main()
server = app.server


if __name__ == "__main__":
    app.run(debug=True)
