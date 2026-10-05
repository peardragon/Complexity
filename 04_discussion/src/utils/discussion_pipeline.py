#!/usr/bin/env python3
"""Deterministic post-hoc adapters for the Discussion evidence tree.

This module reads validated upstream summaries.  It never generates a dataset,
trains a reference, evaluates a model, or runs shell SMC.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from utils.antipodal_summary import aggregate_raw_antipodal


DISCUSSION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = DISCUSSION_ROOT.parent

STAGE_DIRS = {
    "01": DISCUSSION_ROOT / "01_frozen_inputs",
    "02": DISCUSSION_ROOT / "02_normalized_radial_geometry",
    "03": DISCUSSION_ROOT / "03_antipodal_geometry",
    "04": DISCUSSION_ROOT / "04_random_label_reentrance",
}

def stage_dir(stage_key: str, output_dir: Path | None = None) -> Path:
    """Return the canonical stage or its mirrored location below output_dir."""
    if output_dir is None:
        return STAGE_DIRS[stage_key]
    return Path(output_dir) / STAGE_DIRS[stage_key].name


def stage_outputs(stage_key: str, output_dir: Path | None = None) -> list[Path]:
    _, config = _stage_config(stage_key)
    return [stage_dir(stage_key, output_dir) / name for name in config["outputs"]]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV input: {path}")
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV has no header: {path}") from error
        return sum(1 for _ in reader)


def _stage_config(stage_key: str) -> tuple[Path, dict[str, Any]]:
    path = STAGE_DIRS[stage_key] / "config" / "default.json"
    return path, load_json(path)


def _write_csv_if_needed(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
    *,
    force: bool,
) -> None:
    if path.is_file() and not force:
        print(f"{path.name}: skipped_existing")
        return
    write_csv(path, fieldnames, rows)


def _write_json_if_needed(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.is_file() and not force:
        print(f"{path.name}: skipped_existing")
        return
    write_json(path, payload)


def run_stage_01(output_dir: Path | None = None, force: bool = False) -> None:
    stage = stage_dir("01", output_dir)
    if not force and all(path.is_file() for path in stage_outputs("01", output_dir)):
        print("01_frozen_inputs: skipped_existing")
        return
    _, config = _stage_config("01")
    inventory_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for item in config["repository_relative_sources"]:
        path = REPOSITORY_ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen input {item['source_id']}: {path}")
        suffix = path.suffix.lower().lstrip(".")
        row_count: int | str = _csv_row_count(path) if suffix == "csv" else ""
        record = {
            "source_id": item["source_id"],
            "role": item["role"],
            "path": item["path"],
            "format": suffix,
            "bytes": path.stat().st_size,
            "row_count": row_count,
        }
        inventory_rows.append(record)
        source_records.append(record.copy())

    inventory_path = stage / "summarized_outputs" / "input_inventory.csv"
    receipt_path = stage / "summarized_outputs" / "input_receipt.json"
    _write_csv_if_needed(
        inventory_path,
        ["source_id", "role", "path", "format", "bytes", "row_count"],
        inventory_rows,
        force=force,
    )
    _write_json_if_needed(
        receipt_path,
        {
            "schema_version": "complexity.discussion.input_receipt.v1",
            "stage_id": config["stage_id"],
            "repository_root": ".",
            "source_count": len(source_records),
            "sources": source_records,
            "scope": "read-only post-hoc summaries and frozen Study-22 audit",
        },
        force=force,
    )


def source_inventory(output_dir: Path | None = None) -> dict[str, dict[str, str]]:
    path = stage_dir("01", output_dir) / "summarized_outputs" / "input_inventory.csv"
    rows = read_csv(path)
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        source_id = row["source_id"]
        if source_id in mapping:
            raise ValueError(f"duplicate source id in input inventory: {source_id}")
        source_path = REPOSITORY_ROOT / row["path"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        mapping[source_id] = row
    return mapping


def source_path(inventory: dict[str, dict[str, str]], source_id: str) -> Path:
    if source_id not in inventory:
        raise KeyError(f"source id not frozen by Stage 01: {source_id}")
    return REPOSITORY_ROOT / inventory[source_id]["path"]


def _positive_runs(rows: list[dict[str, Any]], value_key: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        positive = float(row[value_key]) > 0.0
        if positive and start is None:
            start = index
        if not positive and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(rows) - 1))
    return runs


def _zero_crossing(left: dict[str, Any], right: dict[str, Any], value_key: str) -> float:
    x0, x1 = float(left["radius"]), float(right["radius"])
    y0, y1 = float(left[value_key]), float(right[value_key])
    if y0 == y1:
        raise ValueError("cannot interpolate a zero crossing from equal endpoint values")
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def _profile_phase_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = rows[0]
    peak_index = max(range(len(rows)), key=lambda index: float(rows[index]["h_mean"]))
    peak = rows[peak_index]
    baseline_h = float(baseline["h_mean"])
    recovery = ""
    for row in rows[peak_index + 1 :]:
        if float(row["h_mean"]) <= baseline_h:
            recovery = float(row["radius"])
            break

    runs = _positive_runs(rows, "g_mean")
    crossings: list[float] = []
    signs: list[int] = []
    for row in rows:
        value = float(row["g_mean"])
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign and (not signs or signs[-1] != sign):
            signs.append(sign)
    for left, right in zip(rows[:-1], rows[1:]):
        y0, y1 = float(left["g_mean"]), float(right["g_mean"])
        if y0 * y1 < 0:
            crossings.append(_zero_crossing(left, right, "g_mean"))

    return {
        "baseline_radius": float(baseline["radius"]),
        "baseline_h": baseline_h,
        "peak_radius": float(peak["radius"]),
        "peak_h": float(peak["h_mean"]),
        "hardening_amplitude": float(peak["h_mean"]) - baseline_h,
        "postpeak_recovery_to_baseline_radius": recovery,
        "max_g": max(float(row["g_mean"]) for row in rows),
        "positive_interval_count": len(runs),
        "first_positive_radius": float(rows[runs[0][0]]["radius"]) if runs else "",
        "last_positive_radius": float(rows[runs[-1][1]]["radius"]) if runs else "",
        "g_sign_sequence": "/".join("positive" if sign > 0 else "negative" for sign in signs),
        "g_zero_crossings_linear": ";".join(f"{value:.12g}" for value in crossings),
        "radius_count": len(rows),
    }


def run_stage_02(output_dir: Path | None = None, force: bool = False) -> None:
    stage = stage_dir("02", output_dir)
    if not force and all(path.is_file() for path in stage_outputs("02", output_dir)):
        print("02_normalized_radial_geometry: skipped_existing")
        return
    _, config = _stage_config("02")
    inventory = source_inventory(output_dir)
    normalized_rows: list[dict[str, Any]] = []

    for source_id in config["canonical_profile_source_ids"]:
        path = source_path(inventory, source_id)
        for row in read_csv(path):
            radius = float(row["radius"])
            if radius <= 0:
                raise ValueError(f"non-positive shell radius in {path}: {radius}")
            g_mean = float(row["dphi_energetic_dr_direct_mean"])
            g_se = float(row["dphi_energetic_dr_direct_se_across_datasets"])
            if not all(math.isfinite(value) for value in (radius, g_mean, g_se)) or g_se < 0.0:
                raise ValueError(f"invalid radial value in {path}: {row}")
            normalized_rows.append(
                {
                    "domain": row["domain"],
                    "condition": row["condition"],
                    "condition_order": int(row["condition_order"]),
                    "radius_index": int(row["radius_index"]),
                    "radius": radius,
                    "dataset_count": int(row["dataset_count"]),
                    "references_per_dataset": int(row["references_per_dataset"]),
                    "g_mean": g_mean,
                    "g_se_across_datasets": g_se,
                    "h_mean": -g_mean / radius,
                    "h_se_across_datasets": g_se / radius,
                    "source_id": source_id,
                }
            )

    normalized_rows.sort(
        key=lambda row: (
            str(row["domain"]),
            int(row["condition_order"]),
            int(row["radius_index"]),
        )
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        grouped[(str(row["domain"]), str(row["condition"]))].append(row)

    phase_rows: list[dict[str, Any]] = []
    for (domain, condition), rows in grouped.items():
        rows.sort(key=lambda row: float(row["radius"]))
        expected_indices = list(range(len(rows)))
        observed_indices = [int(row["radius_index"]) for row in rows]
        if observed_indices != expected_indices:
            raise ValueError(f"non-contiguous radius grid for {domain}/{condition}")
        metrics = _profile_phase_metrics(rows)
        phase_rows.append(
            {
                "domain": domain,
                "condition": condition,
                "condition_order": int(rows[0]["condition_order"]),
                "dataset_count": int(rows[0]["dataset_count"]),
                **metrics,
            }
        )
    phase_rows.sort(key=lambda row: (str(row["domain"]), int(row["condition_order"])))

    profile_path = stage / "summarized_outputs" / "normalized_radial_profiles.csv"
    phase_path = stage / "summarized_outputs" / "radial_phase_metrics.csv"
    all_harden = all(float(row["hardening_amplitude"]) > 0 for row in phase_rows)
    if not all_harden:
        raise ValueError("at least one condition mean lacks inner hardening")
    _write_csv_if_needed(
        profile_path,
        [
            "domain",
            "condition",
            "condition_order",
            "radius_index",
            "radius",
            "dataset_count",
            "references_per_dataset",
            "g_mean",
            "g_se_across_datasets",
            "h_mean",
            "h_se_across_datasets",
            "source_id",
        ],
        normalized_rows,
        force=force,
    )
    _write_csv_if_needed(
        phase_path,
        [
            "domain",
            "condition",
            "condition_order",
            "dataset_count",
            "baseline_radius",
            "baseline_h",
            "peak_radius",
            "peak_h",
            "hardening_amplitude",
            "postpeak_recovery_to_baseline_radius",
            "max_g",
            "positive_interval_count",
            "first_positive_radius",
            "last_positive_radius",
            "g_sign_sequence",
            "g_zero_crossings_linear",
            "radius_count",
        ],
        phase_rows,
        force=force,
    )


def _with_common_metadata(
    rows: list[dict[str, str]], condition_labels: dict[str, str]
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        condition = row.get("condition", "")
        item["condition_label"] = condition_labels.get(condition, condition)
        item["source_study"] = "arxiv_2608.22361v1_appendix_d"
        enriched.append(item)
    return enriched


def _stage_03_source_rows(
    config: dict[str, Any],
    inventory: dict[str, dict[str, str]],
    raw_dir: Path | None,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    raw_root = Path(raw_dir) if raw_dir is not None else STAGE_DIRS["03"] / "raw_outputs"
    raw_paths = {
        condition: raw_root / f"antipodal_{condition}.npz"
        for condition in config["design"]["conditions"]
    }
    existing = {condition: path.is_file() for condition, path in raw_paths.items()}
    if any(existing.values()) and not all(existing.values()):
        missing = [condition for condition, present in existing.items() if not present]
        raise FileNotFoundError(
            f"partial Appendix D raw pair in {raw_root}; missing {missing}"
        )
    if all(existing.values()):
        return aggregate_raw_antipodal(raw_paths, config), "raw_npz"

    return (
        {
            "profiles": read_csv(source_path(inventory, "study22_fresh_label_K_profiles")),
            "summary": read_csv(source_path(inventory, "study22_fresh_label_K_summary")),
            "contrast": read_csv(
                source_path(inventory, "study22_fresh_label_K_random_minus_clean")
            ),
            "peak": read_csv(source_path(inventory, "study22_fresh_label_K_peak_contrast")),
        },
        "compact_frozen_csv",
    )


def _validate_stage_03_source_rows(
    rows: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> None:
    conditions = tuple(str(value) for value in config["design"]["conditions"])
    radii = tuple(float(value) for value in config["evaluation"]["radii"])
    dataset_count = int(config["design"]["datasets_per_condition"])
    profile_coordinates: set[tuple[str, float]] = set()
    for row in rows["profiles"]:
        _finite(
            row,
            ("radius", "mean_K", "se_K", "ci95_K_low", "ci95_K_high"),
            "Appendix D source profiles",
        )
        coordinate = (str(row["condition"]), float(row["radius"]))
        if coordinate in profile_coordinates:
            raise ValueError(f"duplicate Appendix D source profile coordinate: {coordinate}")
        profile_coordinates.add(coordinate)
    expected_profiles = {(condition, radius) for condition in conditions for radius in radii}
    if profile_coordinates != expected_profiles:
        raise ValueError("Appendix D source profile coverage differs from config")

    nonbaseline = radii[1:]
    summary_coordinates: set[tuple[str, float, float]] = set()
    for row in rows["summary"]:
        _finite(
            row,
            (
                "near_radius", "far_radius", "mean_delta_K", "se_delta_K",
                "ci95_delta_K_low", "ci95_delta_K_high", "positive_dataset_count",
                "exact_two_sided_sign_flip_p", "mean_hardening_direction_fraction",
                "ci95_hardening_fraction_low", "ci95_hardening_fraction_high",
                "pooled_delta_K_q05_descriptive", "pooled_delta_K_median_descriptive",
                "pooled_delta_K_q95_descriptive",
            ),
            "Appendix D source summary",
        )
        coordinate = (
            str(row["condition"]),
            float(row["near_radius"]),
            float(row["far_radius"]),
        )
        if coordinate in summary_coordinates:
            raise ValueError(f"duplicate Appendix D source summary coordinate: {coordinate}")
        summary_coordinates.add(coordinate)
        if not 0 <= int(row["positive_dataset_count"]) <= dataset_count:
            raise ValueError(f"Appendix D source dataset count is invalid: {row}")
        if not 0.0 <= float(row["mean_hardening_direction_fraction"]) <= 1.0:
            raise ValueError(f"Appendix D source direction fraction is invalid: {row}")
    expected_summaries = {
        (condition, radii[0], radius)
        for condition in conditions
        for radius in nonbaseline
    }
    if summary_coordinates != expected_summaries:
        raise ValueError("Appendix D source summary coverage differs from config")

    contrast_coordinates: set[tuple[float, float]] = set()
    for row in rows["contrast"]:
        _finite(
            row,
            (
                "near_radius", "far_radius", "mean_extra_delta_K", "se_extra_delta_K",
                "ci95_low", "ci95_high", "positive_dataset_count",
                "exact_two_sided_sign_flip_p",
            ),
            "Appendix D source contrast",
        )
        coordinate = (float(row["near_radius"]), float(row["far_radius"]))
        if coordinate in contrast_coordinates:
            raise ValueError(f"duplicate Appendix D source contrast coordinate: {coordinate}")
        contrast_coordinates.add(coordinate)
    if contrast_coordinates != {(radii[0], radius) for radius in nonbaseline}:
        raise ValueError("Appendix D source contrast coverage differs from config")

    expected_metrics = {"peak_radius", "peak_absolute_K", "peak_hardening_delta_K"}
    if len(rows["peak"]) != len(expected_metrics) or {
        str(row["metric"]) for row in rows["peak"]
    } != expected_metrics:
        raise ValueError("Appendix D source peak metrics differ from config")
    for row in rows["peak"]:
        _finite(
            row,
            (
                "mean_difference", "se_difference", "ci95_low", "ci95_high",
                "positive_dataset_count", "negative_dataset_count",
                "exact_two_sided_sign_flip_p",
            ),
            "Appendix D source peak contrast",
        )


def run_stage_03(
    output_dir: Path | None = None,
    force: bool = False,
    raw_dir: Path | None = None,
) -> None:
    stage = stage_dir("03", output_dir)
    if not force and all(path.is_file() for path in stage_outputs("03", output_dir)):
        print("03_antipodal_geometry: skipped_existing")
        return
    _, config = _stage_config("03")
    inventory = source_inventory(output_dir)
    labels = {
        "noise_eta_0p00": "clean labels",
        "noise_eta_0p50": "50% randomized labels",
    }

    source_rows, source_mode = _stage_03_source_rows(config, inventory, raw_dir)
    _validate_stage_03_source_rows(source_rows, config)
    profile_rows = _with_common_metadata(source_rows["profiles"], labels)
    all_summary_rows = source_rows["summary"]
    near = float(config["near_radius_contrast"]["baseline_radius"])
    far = float(config["near_radius_contrast"]["comparison_radius"])
    summary_rows = _with_common_metadata(
        [
            row
            for row in all_summary_rows
            if math.isclose(float(row["near_radius"]), near, abs_tol=1e-12)
            and math.isclose(float(row["far_radius"]), far, abs_tol=1e-12)
        ],
        labels,
    )
    if {row["condition"] for row in summary_rows} != set(labels):
        raise ValueError("missing clean/random near-radius K summary")
    contrast_rows = [
        {**row, "source_study": config["source_study"]}
        for row in source_rows["contrast"]
        if math.isclose(float(row["near_radius"]), near, abs_tol=1e-12)
        and math.isclose(float(row["far_radius"]), far, abs_tol=1e-12)
    ]
    if len(contrast_rows) != 1:
        raise ValueError("expected one clean/random near-radius paired contrast")
    peak_rows = [
        {**row, "source_study": config["source_study"]} for row in source_rows["peak"]
    ]

    profile_path = stage / "summarized_outputs" / "matched_antipodal_profiles.csv"
    summary_path = stage / "summarized_outputs" / "matched_antipodal_near_radius_summary.csv"
    contrast_path = stage / "summarized_outputs" / "matched_antipodal_random_minus_clean.csv"
    peak_path = stage / "summarized_outputs" / "matched_antipodal_peak_contrast.csv"
    dataset_count = int(config["design"]["datasets_per_condition"])
    if not all(int(row["positive_dataset_count"]) == dataset_count for row in summary_rows):
        raise ValueError("near-radius antipodal increase is not positive in all datasets")
    if int(contrast_rows[0]["positive_dataset_count"]) != dataset_count:
        raise ValueError("random-minus-clean K amplification is not positive in all datasets")
    _write_csv_if_needed(
        profile_path,
        [
            "condition",
            "condition_label",
            "radius",
            "mean_K",
            "se_K",
            "ci95_K_low",
            "ci95_K_high",
            "source_study",
        ],
        profile_rows,
        force=force,
    )
    summary_fields = list(all_summary_rows[0].keys()) + ["condition_label", "source_study"]
    _write_csv_if_needed(summary_path, summary_fields, summary_rows, force=force)
    _write_csv_if_needed(
        contrast_path, list(contrast_rows[0].keys()), contrast_rows, force=force
    )
    _write_csv_if_needed(peak_path, list(peak_rows[0].keys()), peak_rows, force=force)

    print(f"03_antipodal_geometry: source={source_mode}")


def run_stage_04(output_dir: Path | None = None, force: bool = False) -> None:
    stage = stage_dir("04", output_dir)
    if not force and all(path.is_file() for path in stage_outputs("04", output_dir)):
        print("04_random_label_reentrance: skipped_existing")
        return
    _, config = _stage_config("04")
    inventory = source_inventory(output_dir)
    canonical_path = source_path(inventory, config["canonical_profile_source_id"])
    source_rows = read_csv(canonical_path)
    sign_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        radius = float(row["radius"])
        g_mean = float(row["dphi_energetic_dr_direct_mean"])
        g_se = float(row["dphi_energetic_dr_direct_se_across_datasets"])
        if radius <= 0.0 or g_se < 0.0 or not all(
            math.isfinite(value) for value in (radius, g_mean, g_se)
        ):
            raise ValueError(f"invalid label-noise radial value: {row}")
        item = {
            "domain": row["domain"],
            "condition": row["condition"],
            "condition_order": int(row["condition_order"]),
            "noise_eta": float(row["noise_eta"]),
            "radius_index": int(row["radius_index"]),
            "radius": radius,
            "g_mean": g_mean,
            "g_se_across_datasets": g_se,
            "sign": 1 if g_mean > 0 else (-1 if g_mean < 0 else 0),
            "h_mean": -g_mean / radius,
            "h_se_across_datasets": g_se / radius,
        }
        sign_rows.append(item)
        grouped[row["condition"]].append(item)

    interval_rows: list[dict[str, Any]] = []
    topologies: dict[str, str] = {}
    for condition, rows in grouped.items():
        rows.sort(key=lambda row: float(row["radius"]))
        observed_indices = [int(row["radius_index"]) for row in rows]
        if observed_indices != list(range(len(rows))):
            raise ValueError(f"non-contiguous label-noise radius grid for {condition}")
        observed_radii = [float(row["radius"]) for row in rows]
        if any(right <= left for left, right in zip(observed_radii, observed_radii[1:])):
            raise ValueError(f"non-increasing label-noise radius grid for {condition}")
        runs = _positive_runs(rows, "g_mean")
        signs: list[int] = []
        for row in rows:
            sign = int(row["sign"])
            if sign and (not signs or signs[-1] != sign):
                signs.append(sign)
        topology = "/".join("positive" if value > 0 else "negative" for value in signs)
        topologies[condition] = topology
        if not runs:
            interval_rows.append(
                {
                    "domain": rows[0]["domain"],
                    "condition": condition,
                    "condition_order": rows[0]["condition_order"],
                    "noise_eta": rows[0]["noise_eta"],
                    "g_sign_sequence": topology,
                    "interval_index": "",
                    "entry_zero_crossing_linear": "",
                    "first_positive_grid_radius": "",
                    "last_positive_grid_radius": "",
                    "exit_zero_crossing_linear": "",
                    "positive_grid_count": 0,
                    "max_g_in_interval": "",
                    "max_g_radius": "",
                }
            )
            continue
        for interval_index, (start, stop) in enumerate(runs):
            positive_rows = rows[start : stop + 1]
            maximum = max(positive_rows, key=lambda row: float(row["g_mean"]))
            entry = _zero_crossing(rows[start - 1], rows[start], "g_mean") if start > 0 else ""
            exit_radius = (
                _zero_crossing(rows[stop], rows[stop + 1], "g_mean")
                if stop + 1 < len(rows)
                else ""
            )
            interval_rows.append(
                {
                    "domain": rows[0]["domain"],
                    "condition": condition,
                    "condition_order": rows[0]["condition_order"],
                    "noise_eta": rows[0]["noise_eta"],
                    "g_sign_sequence": topology,
                    "interval_index": interval_index,
                    "entry_zero_crossing_linear": entry,
                    "first_positive_grid_radius": positive_rows[0]["radius"],
                    "last_positive_grid_radius": positive_rows[-1]["radius"],
                    "exit_zero_crossing_linear": exit_radius,
                    "positive_grid_count": len(positive_rows),
                    "max_g_in_interval": maximum["g_mean"],
                    "max_g_radius": maximum["radius"],
                }
            )

    expected = {
        "noise_eta_0p00": "negative",
        "noise_eta_0p05": "negative",
        "noise_eta_0p15": "negative",
        "noise_eta_0p25": "negative/positive/negative",
        "noise_eta_0p50": "negative/positive/negative",
    }
    if topologies != expected:
        raise ValueError(f"unexpected label-noise sign topology: {topologies}")

    sign_rows.sort(key=lambda row: (int(row["condition_order"]), int(row["radius_index"])))
    interval_rows.sort(
        key=lambda row: (
            int(row["condition_order"]),
            -1 if row["interval_index"] == "" else int(row["interval_index"]),
        )
    )
    sign_path = stage / "summarized_outputs" / "label_noise_radial_signs.csv"
    interval_path = stage / "summarized_outputs" / "reentrant_intervals.csv"
    _write_csv_if_needed(
        sign_path,
        [
            "domain",
            "condition",
            "condition_order",
            "noise_eta",
            "radius_index",
            "radius",
            "g_mean",
            "g_se_across_datasets",
            "sign",
            "h_mean",
            "h_se_across_datasets",
        ],
        sign_rows,
        force=force,
    )
    _write_csv_if_needed(
        interval_path,
        [
            "domain",
            "condition",
            "condition_order",
            "noise_eta",
            "g_sign_sequence",
            "interval_index",
            "entry_zero_crossing_linear",
            "first_positive_grid_radius",
            "last_positive_grid_radius",
            "exit_zero_crossing_linear",
            "positive_grid_count",
            "max_g_in_interval",
            "max_g_radius",
        ],
        interval_rows,
        force=force,
    )


def run_all(output_dir: Path | None = None, force: bool = False) -> None:
    """Run all four post-hoc stages with per-output filename reuse."""
    for run in (run_stage_01, run_stage_02, run_stage_03, run_stage_04):
        run(output_dir=output_dir, force=force)


def _require_columns(
    rows: list[dict[str, str]], required: set[str], label: str
) -> None:
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def _finite(row: dict[str, str], fields: Iterable[str], label: str) -> None:
    for field in fields:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label}: invalid {field}: {row.get(field)!r}") from error
        if not math.isfinite(value):
            raise ValueError(f"{label}: non-finite {field}: {row}")


def _strict_grid(
    rows: list[dict[str, str]], *, group_fields: tuple[str, ...], label: str
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    coordinates: set[tuple[str, ...]] = set()
    for row in rows:
        group = tuple(row[field] for field in group_fields)
        coordinate = (*group, row["radius_index"])
        if coordinate in coordinates:
            raise ValueError(f"{label}: duplicate coordinate {coordinate}")
        coordinates.add(coordinate)
        grouped[group].append(row)
    for group, values in grouped.items():
        values.sort(key=lambda row: int(row["radius_index"]))
        indices = [int(row["radius_index"]) for row in values]
        radii = [float(row["radius"]) for row in values]
        if indices != list(range(len(values))):
            raise ValueError(f"{label}: non-contiguous radius indices for {group}")
        if any(right <= left for left, right in zip(radii, radii[1:])):
            raise ValueError(f"{label}: radius grid is not strictly increasing for {group}")
    return grouped


def validate_all(
    output_dir: Path | None = None,
    *,
    check_published_benchmarks: bool | None = None,
    write_report: bool = False,
) -> dict[str, Any]:
    """Validate tables read-only; write_report is accepted for old callers only."""
    if check_published_benchmarks is None:
        check_published_benchmarks = output_dir is None
    output_paths = [
        path for key in ("01", "02", "03", "04") for path in stage_outputs(key, output_dir)
    ]
    missing_outputs = [str(path) for path in output_paths if not path.is_file()]
    if missing_outputs:
        raise FileNotFoundError(f"missing Discussion outputs: {missing_outputs}")

    inventory = source_inventory(output_dir)
    allowed_roots = {"01_theory", "02_dnn_synthetic", "03_dnn_mnist", "04_discussion"}
    for source_id, record in inventory.items():
        path = Path(record["path"])
        if path.is_absolute() or not path.parts or path.parts[0] not in allowed_roots:
            raise ValueError(f"{source_id}: source escapes release roots: {path}")
        if not (REPOSITORY_ROOT / path).is_file():
            raise FileNotFoundError(REPOSITORY_ROOT / path)

    profiles = read_csv(
        stage_dir("02", output_dir)
        / "summarized_outputs"
        / "normalized_radial_profiles.csv"
    )
    _require_columns(
        profiles,
        {
            "domain", "condition", "condition_order", "radius_index", "radius",
            "dataset_count", "references_per_dataset", "g_mean",
            "g_se_across_datasets", "h_mean", "h_se_across_datasets", "source_id",
        },
        "normalized radial profiles",
    )
    radial_groups = _strict_grid(
        profiles, group_fields=("domain", "condition"), label="normalized radial profiles"
    )
    expected_domain_counts = {"synthetic": 18, "mnist_label_noise": 5, "mnist_digit_pair": 12}
    observed_domain_counts: dict[str, int] = defaultdict(int)
    for domain, _ in radial_groups:
        observed_domain_counts[domain] += 1
    if dict(observed_domain_counts) != expected_domain_counts:
        raise ValueError(f"unexpected domain/condition coverage: {dict(observed_domain_counts)}")
    for row in profiles:
        _finite(
            row,
            (
                "condition_order", "radius_index", "radius", "dataset_count",
                "references_per_dataset", "g_mean", "g_se_across_datasets",
                "h_mean", "h_se_across_datasets",
            ),
            "normalized radial profiles",
        )
        radius = float(row["radius"])
        if radius <= 0.0 or float(row["g_se_across_datasets"]) < 0.0:
            raise ValueError(f"invalid radial coordinate/SEM: {row}")
        if not math.isclose(
            float(row["h_mean"]),
            -float(row["g_mean"]) / radius,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"h=-g/r validation failed: {row}")
        if not math.isclose(
            float(row["h_se_across_datasets"]),
            float(row["g_se_across_datasets"]) / radius,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"h SEM validation failed: {row}")

    phase = read_csv(
        stage_dir("02", output_dir) / "summarized_outputs" / "radial_phase_metrics.csv"
    )
    if len(phase) != len(radial_groups):
        raise ValueError("radial phase table does not cover every condition")
    phase_keys: set[tuple[str, str]] = set()
    for row in phase:
        key = (row["domain"], row["condition"])
        if key in phase_keys or key not in radial_groups:
            raise ValueError(f"invalid radial phase key: {key}")
        phase_keys.add(key)
        _finite(
            row,
            (
                "condition_order", "dataset_count", "baseline_radius", "baseline_h",
                "peak_radius", "peak_h", "hardening_amplitude", "max_g",
                "positive_interval_count", "radius_count",
            ),
            "radial phase metrics",
        )
        if float(row["hardening_amplitude"]) <= 0.0:
            raise ValueError(f"condition lacks finite-distance hardening: {key}")
        if not math.isclose(
            float(row["hardening_amplitude"]),
            float(row["peak_h"]) - float(row["baseline_h"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"hardening amplitude differs from peak-baseline: {key}")
        if int(row["radius_count"]) != len(radial_groups[key]):
            raise ValueError(f"phase radius count differs for {key}")

    antipodal_config = _stage_config("03")[1]
    conditions = tuple(antipodal_config["design"]["conditions"])
    expected_radii = tuple(float(value) for value in antipodal_config["evaluation"]["radii"])
    dataset_count = int(antipodal_config["design"]["datasets_per_condition"])
    appendix_root = stage_dir("03", output_dir) / "summarized_outputs"
    antipodal_profiles = read_csv(appendix_root / "matched_antipodal_profiles.csv")
    if len(antipodal_profiles) != len(conditions) * len(expected_radii):
        raise ValueError("Appendix D profile row count differs from the configured design")
    profile_coordinates: set[tuple[str, float]] = set()
    profile_values: dict[str, dict[float, float]] = defaultdict(dict)
    for row in antipodal_profiles:
        _finite(
            row,
            ("radius", "mean_K", "se_K", "ci95_K_low", "ci95_K_high"),
            "Appendix D profiles",
        )
        coordinate = (row["condition"], float(row["radius"]))
        if coordinate in profile_coordinates:
            raise ValueError(f"duplicate Appendix D profile coordinate: {coordinate}")
        profile_coordinates.add(coordinate)
        if row["condition"] not in conditions or float(row["radius"]) not in expected_radii:
            raise ValueError(f"unexpected Appendix D profile coordinate: {coordinate}")
        if float(row["se_K"]) < 0.0 or float(row["ci95_K_low"]) > float(row["ci95_K_high"]):
            raise ValueError(f"invalid Appendix D uncertainty: {row}")
        profile_values[row["condition"]][float(row["radius"])] = float(row["mean_K"])

    near = float(antipodal_config["near_radius_contrast"]["baseline_radius"])
    far = float(antipodal_config["near_radius_contrast"]["comparison_radius"])
    near_summary = read_csv(appendix_root / "matched_antipodal_near_radius_summary.csv")
    if {row["condition"] for row in near_summary} != set(conditions) or len(
        near_summary
    ) != len(conditions):
        raise ValueError("Appendix D near-radius summary condition coverage differs")
    for row in near_summary:
        _finite(
            row,
            (
                "near_radius", "far_radius", "mean_delta_K", "se_delta_K",
                "ci95_delta_K_low", "ci95_delta_K_high", "positive_dataset_count",
                "exact_two_sided_sign_flip_p", "mean_hardening_direction_fraction",
                "ci95_hardening_fraction_low", "ci95_hardening_fraction_high",
                "pooled_delta_K_q05_descriptive", "pooled_delta_K_median_descriptive",
                "pooled_delta_K_q95_descriptive",
            ),
            "Appendix D near-radius summary",
        )
        if not math.isclose(
            float(row["near_radius"]), near, abs_tol=1e-12
        ) or not math.isclose(float(row["far_radius"]), far, abs_tol=1e-12):
            raise ValueError(f"Appendix D near-radius coordinate differs: {row}")
        if int(row["positive_dataset_count"]) != dataset_count:
            raise ValueError(f"Appendix D hardening is not positive in every dataset: {row}")
        if not 0.0 <= float(row["mean_hardening_direction_fraction"]) <= 1.0:
            raise ValueError(f"Appendix D direction fraction is outside [0,1]: {row}")
        profile_delta = (
            profile_values[row["condition"]][far]
            - profile_values[row["condition"]][near]
        )
        if not math.isclose(
            float(row["mean_delta_K"]),
            profile_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Appendix D summary/profile delta differs: {row}")

    contrast = read_csv(appendix_root / "matched_antipodal_random_minus_clean.csv")
    if len(contrast) != 1:
        raise ValueError("expected one configured Appendix D near-radius contrast")
    _finite(
        contrast[0],
        (
            "near_radius", "far_radius", "mean_extra_delta_K", "se_extra_delta_K",
            "ci95_low", "ci95_high", "positive_dataset_count", "exact_two_sided_sign_flip_p",
        ),
        "Appendix D paired contrast",
    )
    if int(contrast[0]["positive_dataset_count"]) != dataset_count:
        raise ValueError("random-minus-clean Appendix D contrast is not positive in every dataset")
    if not math.isclose(
        float(contrast[0]["near_radius"]), near, abs_tol=1e-12
    ) or not math.isclose(float(contrast[0]["far_radius"]), far, abs_tol=1e-12):
        raise ValueError("Appendix D paired contrast uses the wrong radius pair")
    paired_delta = (
        profile_values[conditions[1]][far] - profile_values[conditions[1]][near]
        - profile_values[conditions[0]][far] + profile_values[conditions[0]][near]
    )
    if not math.isclose(
        float(contrast[0]["mean_extra_delta_K"]),
        paired_delta,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Appendix D paired contrast differs from the profiles")

    peak = read_csv(appendix_root / "matched_antipodal_peak_contrast.csv")
    expected_metrics = {"peak_radius", "peak_absolute_K", "peak_hardening_delta_K"}
    if len(peak) != 3 or {row["metric"] for row in peak} != expected_metrics:
        raise ValueError("Appendix D peak-contrast metric coverage differs")
    for row in peak:
        _finite(
            row,
            (
                "mean_difference", "se_difference", "ci95_low", "ci95_high",
                "positive_dataset_count", "negative_dataset_count", "exact_two_sided_sign_flip_p",
            ),
            "Appendix D peak contrast",
        )
        if not 0 <= int(row["positive_dataset_count"]) <= dataset_count or not 0 <= int(
            row["negative_dataset_count"]
        ) <= dataset_count:
            raise ValueError(f"Appendix D peak sign count is outside the dataset range: {row}")

    if check_published_benchmarks:
        benchmark = {
            conditions[0]: (0.20, 0.4168902802263177),
            conditions[1]: (0.12, 3.5075451635345933),
        }
        for condition, (peak_radius, expected_delta) in benchmark.items():
            observed = profile_values[condition][peak_radius] - profile_values[condition][near]
            if not math.isclose(observed, expected_delta, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"published Appendix D benchmark differs for {condition}: {observed}"
                )

    sign_rows = read_csv(
        stage_dir("04", output_dir) / "summarized_outputs" / "label_noise_radial_signs.csv"
    )
    sign_groups = _strict_grid(
        sign_rows,
        group_fields=("condition",),
        label="label-noise radial signs",
    )
    expected_topology = {
        "noise_eta_0p00": "negative", "noise_eta_0p05": "negative",
        "noise_eta_0p15": "negative", "noise_eta_0p25": "negative/positive/negative",
        "noise_eta_0p50": "negative/positive/negative",
    }
    if {key[0] for key in sign_groups} != set(expected_topology):
        raise ValueError("label-noise radial condition coverage differs")
    for row in sign_rows:
        _finite(
            row,
            (
                "condition_order", "noise_eta", "radius_index", "radius", "g_mean",
                "g_se_across_datasets", "sign", "h_mean", "h_se_across_datasets",
            ),
            "label-noise radial signs",
        )
        g_mean = float(row["g_mean"])
        sign = 1 if g_mean > 0.0 else (-1 if g_mean < 0.0 else 0)
        if int(row["sign"]) != sign:
            raise ValueError(f"label-noise sign differs from g: {row}")
        if (
            float(row["radius"]) <= 0.0
            or float(row["g_se_across_datasets"]) < 0.0
            or float(row["h_se_across_datasets"]) < 0.0
        ):
            raise ValueError(f"invalid label-noise coordinate/SEM: {row}")
        if not math.isclose(
            float(row["h_mean"]),
            -g_mean / float(row["radius"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"label-noise h=-g/r differs: {row}")

    intervals = read_csv(
        stage_dir("04", output_dir) / "summarized_outputs" / "reentrant_intervals.csv"
    )
    topology = {row["condition"]: row["g_sign_sequence"] for row in intervals}
    if len(intervals) != len(expected_topology) or topology != expected_topology:
        raise ValueError(f"label-noise reentrant topology differs: {topology}")
    for row in intervals:
        _finite(
            row,
            ("condition_order", "noise_eta", "positive_grid_count"),
            "label-noise reentrant intervals",
        )
        if row["g_sign_sequence"] == "negative":
            if int(row["positive_grid_count"]) != 0:
                raise ValueError(f"negative-only condition has a positive interval: {row}")
            continue
        _finite(
            row,
            (
                "interval_index", "entry_zero_crossing_linear",
                "first_positive_grid_radius", "last_positive_grid_radius",
                "exit_zero_crossing_linear", "max_g_in_interval", "max_g_radius",
            ),
            "label-noise reentrant intervals",
        )
        entry = float(row["entry_zero_crossing_linear"])
        first = float(row["first_positive_grid_radius"])
        maximum = float(row["max_g_radius"])
        last = float(row["last_positive_grid_radius"])
        exit_radius = float(row["exit_zero_crossing_linear"])
        if not entry < first <= maximum <= last < exit_radius:
            raise ValueError(f"invalid reentrant interval coordinate order: {row}")
        if int(row["positive_grid_count"]) <= 0 or float(row["max_g_in_interval"]) <= 0.0:
            raise ValueError(f"invalid positive reentrant interval: {row}")

    return {
        "schema_version": "complexity.discussion.pipeline_validation.v2",
        "status": "pass",
        "checked_stages": 4,
        "checked_output_files": len(output_paths),
        "published_benchmarks_checked": bool(check_published_benchmarks),
        "semantic_checks": {
            "finite_numeric_outputs": True,
            "unique_complete_radius_grids": True,
            "h_equals_minus_g_over_r": True,
            "all_35_condition_means_have_inner_hardening": True,
            "appendix_d_full_schema_and_coordinates": True,
            "appendix_d_matched_near_radius_positive_in_all_datasets": True,
            "label_noise_topologies_match_results_and_discussion": True,
            "all_inputs_are_within_release_roots_01_to_04": True,
        },
    }
