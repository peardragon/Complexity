from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
BASE_ROOT = SCRIPT_PATH.parents[1]
DNN_ROOT = SCRIPT_PATH.parents[2]

RUN_DATASET = "gaussian_random_90_dataset"
RUN_REFERENCE = "gaussian_random_90_dataset_30_reference"
RANGE_NAME = "d_0.01_to_2.50_dense"

EXPECTED_DATASETS = 90
EXPECTED_REFS_PER_DATASET = 30
EXPECTED_MANIFEST_ROWS = EXPECTED_DATASETS * EXPECTED_REFS_PER_DATASET
EXPECTED_RADII = 250
EXPECTED_SAMPLING_UNITS = EXPECTED_MANIFEST_ROWS * EXPECTED_RADII

DATASET_RAW = BASE_ROOT / "raw_outputs" / "01_dataset_gen" / RUN_DATASET / "raw_datasets"
COMPLEXITY_REPORT = (
    BASE_ROOT
    / "raw_outputs"
    / "02_complexity_measure"
    / RUN_REFERENCE
    / "complexity_diagnostics_report.md"
)
REFERENCE_SELECTED = (
    BASE_ROOT
    / "raw_outputs"
    / "03_reference_search"
    / RUN_REFERENCE
    / "selected_references"
)
REFERENCE_POOL = (
    BASE_ROOT
    / "raw_outputs"
    / "04_sampling"
    / "reference_pool"
    / RUN_REFERENCE
    / "selected_reference_pool"
)
MANIFEST = BASE_ROOT / "manifests" / "reference_manifest_gaussian_90_30.csv"
SHELL_ROOT = (
    BASE_ROOT
    / "raw_outputs"
    / "04_sampling"
    / "shell_pool"
    / RUN_REFERENCE
    / RANGE_NAME
)
PROXY_RAW_ROOT = BASE_ROOT / "raw_outputs" / "05_proxy_local_entropy" / RUN_REFERENCE / RANGE_NAME
OVERLAY_ROOT = BASE_ROOT / "figures" / "gaussian_overlay"
PHI_ANALYSIS_JSON = BASE_ROOT / "analysis" / "gaussian_vs_spin_phi_analysis.json"
PHI_ANALYSIS_MD = BASE_ROOT / "analysis" / "gaussian_vs_spin_phi_analysis.md"
SPIN_DETAIL = (
    DNN_ROOT
    / "05_proxy_local_entropy"
    / "figures"
    / "high_beta_energy_derivatives_ci_30_60_90"
    / "energy_phi_d1_d2_ci_detail.csv"
)
EVENTS_JSONL = BASE_ROOT / "progress" / "full_pipeline" / "events.jsonl"


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def csv_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def count_selected_reference_units(root: Path) -> dict[str, Any]:
    counts: list[int] = []
    partials: list[dict[str, Any]] = []
    for path in sorted(root.glob("cell_beta_*/dataset_*/width_048/selected_refs.json")):
        payload = load_json(path, {})
        count = len(payload.get("selected_refs", []))
        counts.append(count)
        if count < EXPECTED_REFS_PER_DATASET:
            partials.append(
                {
                    "dataset": path.parents[1].name,
                    "selected_refs": count,
                }
            )
    return {
        "files": len(counts),
        "valid_units": sum(count >= EXPECTED_REFS_PER_DATASET for count in counts),
        "min_refs": min(counts) if counts else 0,
        "max_refs": max(counts) if counts else 0,
        "partials": partials[:20],
    }


def latest_pipeline_driver_event() -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not EVENTS_JSONL.exists():
        return latest
    with EVENTS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stage") == "full_pipeline_driver":
                latest = row
    return latest


def has_pipeline_completed() -> bool:
    if not EVENTS_JSONL.exists():
        return False
    with EVENTS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stage") == "pipeline_completed" and row.get("state") == "completed":
                return True
    return False


def check_spin_detail() -> dict[str, Any]:
    columns = csv_columns(SPIN_DETAIL)
    required = {"metric", "beta", "radius", "dataset_count", "mean", "ci95_low", "ci95_high"}
    phi90_rows = 0
    dataset_counts: set[int] = set()
    metrics: set[str] = set()
    if SPIN_DETAIL.exists():
        with SPIN_DETAIL.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                metrics.add(row.get("metric", ""))
                try:
                    dataset_count = int(float(row.get("dataset_count", "nan")))
                    dataset_counts.add(dataset_count)
                except ValueError:
                    dataset_count = -1
                if row.get("metric") == "phi_energy" and dataset_count == EXPECTED_DATASETS:
                    phi90_rows += 1
    return {
        "path": str(SPIN_DETAIL),
        "exists": SPIN_DETAIL.exists(),
        "columns": columns,
        "required_columns_present": required.issubset(columns),
        "metrics": sorted(metrics),
        "dataset_counts": sorted(dataset_counts),
        "phi_energy_90_rows": phi90_rows,
    }


def build_report() -> dict[str, Any]:
    selected = count_selected_reference_units(REFERENCE_SELECTED)
    pool = count_selected_reference_units(REFERENCE_POOL)
    aggregate = load_json(SHELL_ROOT / "logs" / "aggregate_status.json", {})
    preflight = load_json(SHELL_ROOT / "logs" / "gaussian_preflight_acceptance.json", {})
    driver = latest_pipeline_driver_event()
    gaussian_curve = PROXY_RAW_ROOT / "summary_tables" / "high_beta_curve_comparison.csv"
    overlay_report = load_json(OVERLAY_ROOT / "overlay_report.json", {})
    overlay_figure = OVERLAY_ROOT / "phi_energy_high_beta_spin_90_with_gaussian_baseline_dmax_0p3.png"
    phi_analysis = load_json(PHI_ANALYSIS_JSON, {})

    checks = {
        "dataset_count_90": len(list(DATASET_RAW.glob("cell_beta_*/dataset_*/dataset.npz"))) == EXPECTED_DATASETS,
        "complexity_report_exists": COMPLEXITY_REPORT.exists() and COMPLEXITY_REPORT.stat().st_size > 0,
        "selected_references_90x30": selected["valid_units"] == EXPECTED_DATASETS,
        "reference_pool_90x30": pool["valid_units"] == EXPECTED_DATASETS,
        "manifest_2700_rows": csv_row_count(MANIFEST) == EXPECTED_MANIFEST_ROWS,
        "sampling_preflight_accepted": bool(preflight.get("accepted")),
        "sampling_completed_675000_units": (
            int(aggregate.get("completed") or 0) >= EXPECTED_SAMPLING_UNITS
            and int(aggregate.get("total") or 0) == EXPECTED_SAMPLING_UNITS
            and int(aggregate.get("failed") or 0) == 0
            and str(aggregate.get("event", "")).lower() == "completed"
        ),
        "gaussian_curve_exists": gaussian_curve.exists() and csv_row_count(gaussian_curve) > 0,
        "overlay_figure_exists": overlay_figure.exists() and overlay_figure.stat().st_size > 0,
        "overlay_report_exists": bool(overlay_report),
        "phi_analysis_ready": (
            PHI_ANALYSIS_JSON.exists()
            and PHI_ANALYSIS_MD.exists()
            and PHI_ANALYSIS_MD.stat().st_size > 0
            and bool(phi_analysis.get("gaussian_curve_available"))
            and int(phi_analysis.get("comparison_rows") or 0) > 0
            and int(phi_analysis.get("all_beta_comparison_rows") or 0) > 0
            and phi_analysis.get("complexity_nearest_beta") is not None
        ),
        "spin_detail_compatible": check_spin_detail()["required_columns_present"] and check_spin_detail()["phi_energy_90_rows"] > 0,
        "pipeline_completed_event": has_pipeline_completed(),
        "resource_limits_recorded": driver.get("cpu_affinity_cpus") == "0-15"
        and str(driver.get("cuda_visible_devices", "")) == "2,3",
    }

    return {
        "complete": all(checks.values()),
        "checks": checks,
        "counts": {
            "datasets": len(list(DATASET_RAW.glob("cell_beta_*/dataset_*/dataset.npz"))),
            "selected_references": selected,
            "reference_pool": pool,
            "manifest_rows": csv_row_count(MANIFEST),
            "gaussian_curve_rows": csv_row_count(gaussian_curve),
        },
        "sampling": {
            "preflight": preflight,
            "aggregate_status": aggregate,
        },
        "overlay": {
            "figure": str(overlay_figure),
            "figure_size_bytes": overlay_figure.stat().st_size if overlay_figure.exists() else 0,
            "report": overlay_report,
        },
        "phi_analysis": {
            "json": str(PHI_ANALYSIS_JSON),
            "markdown": str(PHI_ANALYSIS_MD),
            "json_exists": PHI_ANALYSIS_JSON.exists(),
            "markdown_exists": PHI_ANALYSIS_MD.exists(),
            "report": phi_analysis,
        },
        "spin_detail": check_spin_detail(),
        "latest_driver_event": driver,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["complete"] or args.allow_incomplete:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
