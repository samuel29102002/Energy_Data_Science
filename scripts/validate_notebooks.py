import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
CHECKS_DIR = REPORTS_DIR / "checks"

for path in (FIGURES_DIR, TABLES_DIR, CHECKS_DIR):
    path.mkdir(parents=True, exist_ok=True)

TARGET_NOTEBOOKS = [
    ("08_ml_models.ipynb", "08"),
    ("09_forecasting_pipeline.ipynb", "09"),
    ("10_exogenous_models.ipynb", "10"),
    ("11_optim_storage.ipynb", "11"),
]


def execute_notebook(nb_path: Path) -> list[str]:
    errors: list[str] = []
    for attempt in (1, 2):
        try:
            nb = nbformat.read(nb_path, as_version=4)
            client = NotebookClient(
                nb,
                timeout=900,
                kernel_name="python3",
                resources={"metadata": {"path": nb_path.parent}},
            )
            client.execute()
            nbformat.write(nb, nb_path)
            return errors
        except CellExecutionError as exc:
            tb = traceback.format_exc(limit=2)
            errors.append(f"Attempt {attempt}: Cell execution error in {nb_path.name}: {exc}\n{tb}")
        except Exception as exc:  # pragma: no cover - safeguard
            tb = traceback.format_exc(limit=2)
            errors.append(f"Attempt {attempt}: Unexpected error in {nb_path.name}: {exc}\n{tb}")
    return errors


def collect_outputs(prefix: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    fig_matches = sorted(FIGURES_DIR.glob(f"{prefix}_*.png"))
    table_matches = sorted(TABLES_DIR.glob(f"{prefix}_*.csv"))
    status = "ok"
    if not fig_matches:
        warnings.append(f"No figures saved with prefix {prefix}_")
        status = "warning"
    if not table_matches:
        warnings.append(f"No tables saved with prefix {prefix}_")
        status = "warning"
    return status, warnings


def write_summary(prefix: str, status: str, errors: list[str], warnings: list[str]) -> None:
    summary_path = CHECKS_DIR / f"{prefix}_summary.txt"
    lines = [
        f"Notebook prefix: {prefix}",
        f"Status: {status}",
    ]
    if errors:
        lines.append("Errors:")
        lines.extend(errors)
    if warnings:
        lines.append("Warnings:")
        lines.extend(warnings)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report: dict[str, dict[str, object]] = {}
    passed = failed = warned = 0

    for notebook_name, prefix in TARGET_NOTEBOOKS:
        nb_path = NOTEBOOK_DIR / notebook_name
        errors = execute_notebook(nb_path)
        if errors:
            status = "error"
            failed += 1
            warnings: list[str] = []
        else:
            status, warnings = collect_outputs(prefix)
            if status == "ok":
                passed += 1
            else:
                warned += 1
        write_summary(prefix, status, errors, warnings)
        report[notebook_name] = {
            "status": status,
            "errors": errors,
            "warnings": warnings,
        }

    report["summary"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "passed": passed,
        "failed": failed,
        "warnings": warned,
    }

    validation_report_path = CHECKS_DIR / "validation_report.json"
    validation_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Validation finished:")
    for notebook_name, details in report.items():
        if notebook_name == "summary":
            continue
        print(f"  {notebook_name}: {details['status']}")
        if details["errors"]:
            print("    Errors:")
            for err in details["errors"]:
                print(f"      - {err.splitlines()[0]}")
        if details["warnings"]:
            print("    Warnings:")
            for warn in details["warnings"]:
                print(f"      - {warn}")
    summary = report["summary"]
    print(f"Totals → passed: {summary['passed']}, warnings: {summary['warnings']}, failed: {summary['failed']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
