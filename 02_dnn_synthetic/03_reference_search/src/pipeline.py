from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2].resolve()
REF_SRC = SCRIPT_DIR
if str(REF_SRC) not in sys.path:
    sys.path.insert(0, str(REF_SRC))

from io_utils import load_json, save_csv, save_json
from rescue import summarize_and_select_reference_candidates
import simple_pipeline


EXCLUDED_SUMMARY_OUTPUT_NAMES = {
    "candidate_refs.json",
    "valid_refs_manifest.json",
    "invalid_for_sampling.json",
    "invalid_refs_manifest.json",
}


def _record_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _resolve_recorded_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _load_attempt_records(width_dir: Path) -> list[dict[str, Any]]:
    attempt_csv = width_dir / "attempt_results.csv"
    if not attempt_csv.exists():
        return []
    rows = list(csv.DictReader(attempt_csv.read_text(encoding="utf-8").splitlines()))
    records: list[dict[str, Any]] = []
    for row in rows:
        theta_path = _resolve_recorded_path(row["theta_final_path"])
        theta_init_path = _resolve_recorded_path(row["theta_init_path"])
        summary_path = _resolve_recorded_path(row["summary_path"])
        if not theta_path.exists() or not theta_init_path.exists() or not summary_path.exists():
            continue
        summary = load_json(summary_path, {}) or {}
        if "final_train_accuracy" not in summary:
            cls_err = float(row["final_cls_err"])
            summary["final_cls_err"] = cls_err
            summary["final_train_accuracy"] = float(row["final_train_accuracy"])
            summary["final_train_loss"] = float(row["final_train_loss"])
            summary["is_exact_solution"] = bool(str(row["is_exact_solution"]).lower() == "true")
        records.append(
            {
                "attempt_id": int(row["attempt_id"]),
                "theta": np.load(theta_path).astype(np.float64),
                "theta_init": np.load(theta_init_path).astype(np.float64),
                "theta_path": _record_path(theta_path),
                "theta_init_path": _record_path(theta_init_path),
                "summary_path": _record_path(summary_path),
                "summary": summary,
                "is_rescue": False,
            }
        )
    return records


def _postprocess_selection(summary_root: Path, config: dict[str, Any]) -> None:
    target_count = int(config.get("selected_refs_per_dataset", 10))
    require_exact = bool(config.get("require_exact_selected_refs", False))
    fail_on_insufficient = bool(config.get("fail_on_insufficient_selected_refs", False))
    min_train_accuracy = float(config.get("fallback_reference_min_train_accuracy", 0.95))
    topk = int(config.get("selection_topk", 8))
    dedup_scale = float(config.get("selection_dedup_scale", 0.25))
    coverage_rows: list[dict[str, Any]] = []
    insufficient_rows: list[dict[str, Any]] = []
    for width_dir in sorted(summary_root.glob("cell_*/*/width_*")):
        attempt_records = _load_attempt_records(width_dir)
        if not attempt_records:
            continue
        cell_id = width_dir.parent.parent.name
        dataset_tag = width_dir.parent.name
        width = int(width_dir.name.replace("width_", ""))
        selection = summarize_and_select_reference_candidates(
            attempt_records,
            cell_id=cell_id,
            dataset_tag=dataset_tag,
            width=width,
            min_train_accuracy=min_train_accuracy,
            target_valid_count=target_count,
            max_selected_count=target_count,
            topk=topk,
            dedup_scale=dedup_scale,
            require_exact=require_exact,
            rescue_enabled=False,
            rescue_policy_name="none",
        )
        retained_rows: list[dict[str, Any]] = []
        payload_root = width_dir / "selected_ref_payloads"
        for selected_row in selection["selected_rows"]:
            retained_row = dict(selected_row)
            ref_id = int(retained_row["ref_id"])
            ref_dir = payload_root / f"ref_{ref_id:03d}"
            ref_dir.mkdir(parents=True, exist_ok=True)
            theta_path = ref_dir / "theta.npy"
            theta_init_path = ref_dir / "theta_init.npy"
            summary_path = ref_dir / "train_summary.json"
            source_theta = _resolve_recorded_path(retained_row["theta_path"])
            source_theta_init = _resolve_recorded_path(retained_row["theta_init_path"])
            source_summary = _resolve_recorded_path(retained_row["summary_path"])
            if not theta_path.exists():
                np.save(theta_path, np.load(source_theta).astype(np.float64))
            if not theta_init_path.exists():
                np.save(theta_init_path, np.load(source_theta_init).astype(np.float64))
            if not summary_path.exists():
                save_json(summary_path, load_json(source_summary, {}) or {})
            retained_row["theta_path"] = _record_path(theta_path)
            retained_row["theta_init_path"] = _record_path(theta_init_path)
            retained_row["summary_path"] = _record_path(summary_path)
            retained_rows.append(retained_row)
        save_json(width_dir / "selected_refs.json", {"cell_id": cell_id, "dataset_tag": dataset_tag, "width": width, "selected_refs": retained_rows})
        invalid_payload = selection.get("invalid_payload")
        if invalid_payload is not None:
            insufficient_rows.append(
                {
                    "cell_id": cell_id,
                    "dataset_tag": dataset_tag,
                    "width": int(width),
                    "required_selected_refs": int(target_count),
                    "selected_ref_count": int(len(retained_rows)),
                    "exact_count": int(selection["manifest_payload"]["exact_count"]),
                    "require_exact_selected_refs": bool(require_exact),
                    "reason": str(invalid_payload.get("reason", "insufficient_distinct_sampling_eligible_references")),
                }
            )
        for diagnostic_name in EXCLUDED_SUMMARY_OUTPUT_NAMES:
            diagnostic_path = width_dir / diagnostic_name
            if diagnostic_path.exists():
                diagnostic_path.unlink()
        coverage_rows.append(
            {
                "cell_id": cell_id,
                "dataset_tag": dataset_tag,
                "width": width,
                "selected_ref_count": int(len(retained_rows)),
                "required_selected_refs": int(target_count),
                "require_exact_selected_refs": bool(require_exact),
                "exact_count": int(selection["manifest_payload"]["exact_count"]),
                "relaxed_count": int(selection["manifest_payload"]["relaxed_count"]),
                "fake_count": int(selection["manifest_payload"]["fake_count"]),
                "sampling_eligible_count": int(selection["manifest_payload"]["sampling_eligible_count"]),
                "dedup_threshold": float(selection["manifest_payload"]["dedup_threshold"]),
            }
        )
    if coverage_rows:
        summary_tables = summary_root / "summary_tables"
        summary_tables.mkdir(parents=True, exist_ok=True)
        save_csv(
            summary_tables / "selected_ref_coverage.csv",
            coverage_rows,
            [
                "cell_id",
                "dataset_tag",
                "width",
                "selected_ref_count",
                "required_selected_refs",
                "require_exact_selected_refs",
                "exact_count",
                "relaxed_count",
                "fake_count",
                "sampling_eligible_count",
                "dedup_threshold",
            ],
        )
        if insufficient_rows:
            save_csv(
                summary_tables / "insufficient_selected_refs.csv",
                insufficient_rows,
                [
                    "cell_id",
                    "dataset_tag",
                    "width",
                    "required_selected_refs",
                    "selected_ref_count",
                    "exact_count",
                    "require_exact_selected_refs",
                    "reason",
                ],
            )
    manifest_path = summary_root / "manifest.json"
    manifest = load_json(manifest_path, {}) or {}
    summary_outputs = {
        _record_path(_resolve_recorded_path(str(path)))
        for path in manifest.get("summary_outputs", [])
    }
    for path in summary_root.rglob("*"):
        if path.is_file() and path.name not in {"manifest.json", "run_config.json", *EXCLUDED_SUMMARY_OUTPUT_NAMES}:
            summary_outputs.add(_record_path(path))
    manifest["summary_outputs"] = sorted(
        path for path in summary_outputs if _resolve_recorded_path(str(path)).name not in EXCLUDED_SUMMARY_OUTPUT_NAMES
    )
    if insufficient_rows and fail_on_insufficient:
        manifest["status"] = "failed"
        manifest["failure_reason"] = "insufficient_selected_references"
        manifest["failed_selected_ref_units"] = int(len(insufficient_rows))
        save_json(manifest_path, manifest)
        raise RuntimeError(
            "reference search produced insufficient selected references for "
            f"{len(insufficient_rows)} dataset-width units; see {summary_root / 'summary_tables' / 'insufficient_selected_refs.csv'}"
        )
    save_json(manifest_path, manifest)


def run_pipeline(*, part_root: Path, config_path: Path, upstream_manifest: Path, force: bool, verbose: bool = False) -> Path:
    manifest_path = simple_pipeline.run_pipeline(
        part_root=part_root,
        config_path=config_path,
        upstream_manifest=upstream_manifest,
        force=force,
        verbose=verbose,
    )
    summary_root = Path(manifest_path).parent
    config = simple_pipeline.merged_config(config_path, upstream_manifest, force)
    _postprocess_selection(summary_root, config)
    return Path(manifest_path)


__all__ = ["run_pipeline"]
