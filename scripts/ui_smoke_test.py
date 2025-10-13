from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dash.development.base_component import Component
from dash.dependencies import Output

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dash_app.app import create_app  # noqa: E402
from src.dash_app.ids import IDS  # noqa: E402
from src.dash_app.layout import render_page  # noqa: E402
from src.dash_app.utils import DashboardData, load_dashboard_data  # noqa: E402

REPORTS_DIR = ROOT / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DEFAULT_OUTPUT = VALIDATION_DIR / "ui_integration_report.json"

EXPECTED_FIGURES: Sequence[str] = (
    "01_demand_pv_timeseries.png",
    "01_demand_pv_daily_sample.png",
    "02_demand_diurnal_profile.png",
    "02_demand_weekday_heatmap.png",
    "03_demand_seasonality_stl.png",
    "04_missingness_heatmap.png",
    "05_feature_importance.png",
    "06_baseline_diagnostics.png",
)

EXPECTED_TABLES: Sequence[str] = (
    "04_pv_gap_stats.csv",
    "05_feature_stats.csv",
    "07_kpis.csv",
)

OPTIONAL_COMPONENT_IDS: Sequence[str] = ("data-toggle-raw-clean",)
EXPECTED_COMPONENT_IDS: Tuple[str, ...] = tuple(
    sorted(
        {
            value
            for section in IDS.values()
            for value in section.values()
            if value not in OPTIONAL_COMPONENT_IDS
        }
    )
)


@dataclass
class CheckResult:
    name: str
    status: str
    details: str


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def discover_routes() -> List[str]:
    return [
        "/",
        "/overview",
        "/ml",
        "/forecast",
        "/optimization",
        "/data",
        "/about",
    ]


def run_http_checks(app) -> Dict[str, object]:
    client = app.server.test_client()
    results: Dict[str, object] = {"responses": []}
    for route in discover_routes():
        response = client.get(route)
        results["responses"].append(
            {
                "route": route,
                "status": response.status_code,
                "ok": 200 <= response.status_code < 400,
            }
        )
    return results


def collect_component_ids(component: Component) -> List[str]:
    accumulator: List[str] = []

    def _walk(node: Component | Iterable | str | None) -> None:
        if node is None:
            return
        if isinstance(node, Component):
            component_id = getattr(node, "id", None)
            if component_id:
                accumulator.append(str(component_id))
            children = getattr(node, "children", None)
            if children is not None:
                _walk(children)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)
        elif isinstance(node, dict):
            for value in node.values():
                _walk(value)

    _walk(component)
    return accumulator


def check_files(base_dir: Path, filenames: Sequence[str]) -> Tuple[List[str], List[str]]:
    present: List[str] = []
    missing: List[str] = []
    for name in filenames:
        path = base_dir / name
        if path.exists():
            present.append(str(path.relative_to(ROOT)))
        else:
            missing.append(str(path.relative_to(ROOT)))
    return present, missing


def collect_app_facts(app, additional_ids: Optional[Sequence[str]] = None) -> Dict[str, object]:
    layout_present = app.layout is not None
    layout_ids: List[str] = []
    if layout_present:
        layout_ids = sorted(set(collect_component_ids(app.layout)))

    combined_ids = layout_ids
    if additional_ids:
        combined_ids = sorted(set(layout_ids).union(additional_ids))

    callback_map = app.callback_map
    callback_count = len(callback_map)
    registered_outputs: List[str] = []
    for data in callback_map.values():
        output = data.get("output")
        if isinstance(output, Output):
            registered_outputs.append(str(output.component_id))
        elif isinstance(output, list):
            registered_outputs.extend(str(item.component_id) for item in output if isinstance(item, Output))

        outputs_list = data.get("outputs_list")
        if isinstance(outputs_list, list):
            for item in outputs_list:
                if isinstance(item, dict) and "id" in item:
                    registered_outputs.append(str(item["id"]))

    registered_outputs = sorted(set(registered_outputs))
    return {
        "layout_present": layout_present,
        "callback_count": callback_count,
        "layout_component_ids": layout_ids,
        "all_component_ids": combined_ids,
        "registered_callback_outputs": registered_outputs,
        "callback_map_keys": sorted(callback_map.keys()),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight smoke test against the Dash UI.")
    parser.add_argument("--tables-path", type=Path, help="Optional override path for dashboard tables")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the JSON summary report",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv or sys.argv[1:])


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    logging.info("Creating Dash application")
    data: DashboardData = load_dashboard_data(args.tables_path)
    app = create_app(args.tables_path)

    page_component_ids: List[str] = []
    try:
        for route in discover_routes():
            page = render_page(route, data)
            page_component_ids.extend(collect_component_ids(page))
    except Exception:  # pragma: no cover - defensive safeguard
        logging.exception("Unable to render individual pages for component discovery")

    logging.info("Running HTTP checks")
    http_results = run_http_checks(app)
    any_route_failure = any(not result["ok"] for result in http_results["responses"])

    logging.info("Collecting callback metadata")
    app_facts = collect_app_facts(app, page_component_ids)

    checks: List[CheckResult] = []
    overall_success = True

    if any_route_failure:
        overall_success = False
        failed_routes = [r["route"] for r in http_results["responses"] if not r["ok"]]
        checks.append(CheckResult("http_routes", "fail", f"Routes failing: {', '.join(failed_routes)}"))
    else:
        checks.append(CheckResult("http_routes", "pass", "All routes returned <400 status"))

    if not app_facts["layout_present"]:
        overall_success = False
        checks.append(CheckResult("layout", "fail", "App layout is missing"))
    else:
        missing_ids = sorted(set(EXPECTED_COMPONENT_IDS) - set(app_facts["all_component_ids"]))
        if missing_ids:
            overall_success = False
            checks.append(CheckResult("layout_ids", "fail", "Missing IDs: " + ", ".join(missing_ids)))
        else:
            checks.append(
                CheckResult(
                    "layout_ids",
                    "pass",
                    f"All {len(EXPECTED_COMPONENT_IDS)} expected component IDs present",
                )
            )

    for name, directory, expected in (
        ("figure_assets", FIGURES_DIR, EXPECTED_FIGURES),
        ("table_assets", TABLES_DIR, EXPECTED_TABLES),
    ):
        present, missing = check_files(directory, expected)
        if missing:
            overall_success = False
            checks.append(CheckResult(name, "fail", "Missing assets: " + ", ".join(missing)))
        else:
            checks.append(CheckResult(name, "pass", f"All {len(expected)} assets present"))

    required_outputs = {
        IDS["overview"]["timeseries_graph"],
        IDS["overview"]["metrics_graph"],
        IDS["overview"]["seasonality_graph"],
        IDS["overview"]["profile_graph"],
        IDS["data"]["missing_heatmap"],
        IDS["data"]["download_target"],
    }
    missing_outputs = sorted(required_outputs - set(app_facts["registered_callback_outputs"]))
    if missing_outputs:
        overall_success = False
        checks.append(CheckResult("callback_outputs", "fail", "Missing callback outputs: " + ", ".join(missing_outputs)))
    else:
        checks.append(
            CheckResult(
                "callback_outputs",
                "pass",
                f"Callback map exposes {app_facts['callback_count']} outputs",
            )
        )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if overall_success else "fail",
        "http": http_results,
        "app": app_facts,
        "checks": [check.__dict__ for check in checks],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Smoke test summary written to %s", args.output)

    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
