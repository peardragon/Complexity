from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _reference_status(summary: dict[str, Any], *, min_train_accuracy: float) -> tuple[str, bool]:
    cls_err = float(summary.get("final_cls_err", float("inf")))
    train_accuracy = float(summary.get("final_train_accuracy", 0.0))
    if cls_err == 0.0 or bool(summary.get("is_exact_solution", False)):
        return "exact_solution", True
    if train_accuracy >= float(min_train_accuracy):
        return "high_accuracy_reference", True
    return "fake_reference", False


def _sort_key(record: dict[str, Any]) -> tuple[float, float, int]:
    summary = dict(record.get("summary", {}))
    return (
        float(summary.get("final_cls_err", float("inf"))),
        float(summary.get("final_train_loss", float("inf"))),
        int(record.get("attempt_id", 0)),
    )


def _dedup_threshold(attempt_records: Sequence[dict[str, Any]], *, topk: int, dedup_scale: float) -> float:
    sorted_records = sorted(list(attempt_records), key=_sort_key)[: max(2, int(topk))]
    if len(sorted_records) < 2:
        return 0.0
    theta_best = np.asarray(sorted_records[0]["theta"], dtype=np.float64).reshape(-1)
    distances = [
        float(np.linalg.norm(np.asarray(row["theta"], dtype=np.float64).reshape(-1) - theta_best))
        for row in sorted_records[1:]
    ]
    distances = [value for value in distances if np.isfinite(value) and value > 1.0e-12]
    if not distances:
        return 1.0e-3 * float(np.sqrt(max(1, theta_best.size)))
    return float(max(1.0e-3 * np.sqrt(max(1, theta_best.size)), float(dedup_scale) * float(np.median(distances))))


def summarize_and_select_reference_candidates(
    attempt_records: Sequence[dict[str, Any]],
    *,
    cell_id: str,
    dataset_tag: str,
    width: int,
    min_train_accuracy: float,
    target_valid_count: int,
    max_selected_count: int,
    topk: int,
    dedup_scale: float,
    rescue_enabled: bool,
    rescue_policy_name: str,
    require_exact: bool = False,
) -> dict[str, Any]:
    threshold = _dedup_threshold(attempt_records, topk=topk, dedup_scale=dedup_scale)
    sorted_records = sorted(list(attempt_records), key=_sort_key)
    selected_records: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    exact_count = 0
    relaxed_count = 0
    fake_count = 0

    for candidate_id, record in enumerate(sorted_records):
        summary = dict(record.get("summary", {}))
        status, eligible = _reference_status(summary, min_train_accuracy=min_train_accuracy)
        if bool(require_exact) and status != "exact_solution":
            eligible = False
        if status == "exact_solution":
            exact_count += 1
        elif status == "high_accuracy_reference":
            relaxed_count += 1
        else:
            fake_count += 1
        summary["reference_status"] = status
        summary["sampler_eligible"] = bool(eligible)
        record["summary"] = summary
        candidate_rows.append(
            {
                "candidate_id": int(candidate_id),
                "attempt_id": int(record["attempt_id"]),
                "theta_path": str(record["theta_path"]),
                "theta_init_path": str(record["theta_init_path"]),
                "summary_path": str(record["summary_path"]),
                "reference_status": status,
                "sampler_eligible": bool(eligible),
                "final_cls_err": float(summary.get("final_cls_err", float("inf"))),
                "final_train_loss": float(summary.get("final_train_loss", float("inf"))),
                "final_train_accuracy": float(summary.get("final_train_accuracy", 0.0)),
            }
        )
        if not eligible:
            continue
        if len(selected_records) >= min(int(target_valid_count), int(max_selected_count)):
            continue
        theta = np.asarray(record["theta"], dtype=np.float64).reshape(-1)
        if all(float(np.linalg.norm(theta - np.asarray(prev["theta"], dtype=np.float64).reshape(-1))) >= threshold for prev in selected_records):
            selected_records.append(record)

    selected_rows = [
        {
            "ref_id": int(ref_id),
            "attempt_id": int(record["attempt_id"]),
            "theta_path": str(record["theta_path"]),
            "theta_init_path": str(record["theta_init_path"]),
            "summary_path": str(record["summary_path"]),
            "reference_status": str(record["summary"]["reference_status"]),
            "sampler_eligible": True,
        }
        for ref_id, record in enumerate(selected_records)
    ]
    manifest_payload = {
        "cell_id": str(cell_id),
        "dataset_tag": str(dataset_tag),
        "width": int(width),
        "attempt_count": int(len(sorted_records)),
        "base_attempt_count": int(len(sorted_records)),
        "rescue_attempt_count": 0,
        "rescue_enabled": bool(rescue_enabled),
        "rescue_policy_name": str(rescue_policy_name),
        "rescue_target_valid_count": int(min(int(target_valid_count), int(max_selected_count))),
        "exact_count": int(exact_count),
        "relaxed_count": int(relaxed_count),
        "fake_count": int(fake_count),
        "sampling_eligible_count": int(len(selected_rows)),
        "valid_count": int(len(selected_rows)),
        "selected_refs_max_count": int(max_selected_count),
        "dedup_threshold": float(threshold),
    }
    invalid_payload = None
    if len(selected_rows) < min(int(target_valid_count), int(max_selected_count)):
        invalid_payload = {
            "cell_id": str(cell_id),
            "dataset_tag": str(dataset_tag),
            "width": int(width),
            "reason": "insufficient_distinct_sampling_eligible_references",
            "selected_ref_count": int(len(selected_rows)),
        }
    return {
        "manifest_payload": manifest_payload,
        "selected_rows": selected_rows,
        "candidate_rows": candidate_rows,
        "invalid_payload": invalid_payload,
    }
