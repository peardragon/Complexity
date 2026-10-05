#!/usr/bin/env python3
"""Summarize and validate MNIST shell-sampler quality diagnostics."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = Path(__file__).resolve().parents[2]
MNIST_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = STAGE_ROOT / "config" / "default.json"
METRIC_FIELDS = [
    "sampling_unit_count", "reference_count", "radius_count", "independent_splits",
    "split_logz_difference_per_parameter_mean", "split_logz_difference_per_parameter_max",
    "minimum_step_overlap_min", "minimum_pool_overlap_pre_resample_min",
    "mean_mh_acceptance", "mean_temperature_step_count", "mean_resample_count",
    "direct_derivative_finite_fraction",
]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = "label_noise_sweep" if PROTOCOL_ROOT.name == "label_noise_sweep" else "digit_pairwise_sweep"
    if config.get("protocol") != expected:
        raise ValueError("config protocol disagrees with experiment directory")
    sampling = config["sampling"]
    if int(sampling["particles_total"]) != int(sampling["independent_splits"]) * int(sampling["particles_per_split"]):
        raise ValueError("particle counts are inconsistent")
    if int(sampling["radii_count"]) <= 0 or float(sampling["radii_start"]) <= 0 or float(sampling["radii_stop"]) < float(sampling["radii_start"]):
        raise ValueError("invalid radius grid")
    indices = [int(value) for value in config["production_replicates"]["dataset_indices"]]
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError("dataset indices must be unique non-negative integers")
    return config


def radii(config: Mapping[str, Any]) -> np.ndarray:
    sampling = config["sampling"]
    return np.linspace(float(sampling["radii_start"]), float(sampling["radii_stop"]), int(sampling["radii_count"]), dtype=np.float64)


def condition_specs(config: Mapping[str, Any], config_path: Path) -> list[dict[str, Any]]:
    if config["protocol"] == "label_noise_sweep":
        specs = []
        for order, eta in enumerate(config["dataset"]["etas"]):
            value = float(eta)
            specs.append({"condition": f"noise_eta_{value:.2f}".replace(".", "p"), "condition_order": order, "noise_eta": value})
    else:
        del config_path
        value = Path(str(config["dataset"]["frozen_manifest"]))
        path = value if value.is_absolute() else STAGE_ROOT / value
        frozen = json.loads(path.read_text(encoding="utf-8"))
        specs = [
            {"condition": str(row["pair_id"]), "condition_order": order, "digit_a": int(row["digit_a"]), "digit_b": int(row["digit_b"]), "pair_rank": int(row["rank"])}
            for order, row in enumerate(frozen["expected_selected_pairs"])
        ]
    names = [row["condition"] for row in specs]
    if not names or len(names) != len(set(names)):
        raise ValueError("condition list is empty or duplicated")
    return specs


def sampling_root(config: Mapping[str, Any]) -> Path:
    value = Path(str(config["paths"]["sampling_shard_root"]))
    resolved = value if value.is_absolute() else MNIST_ROOT / value
    return resolved.parent


def output_path(config: Mapping[str, Any], output_dir: Path | None) -> Path:
    configured = Path(str(config["summary"]["output"]))
    return output_dir / configured.name if output_dir is not None else (configured if configured.is_absolute() else STAGE_ROOT / configured)


def sampling_path(root: Path, dataset_index: int, condition: str) -> Path:
    dataset_root = root / f"dataset_{dataset_index:03d}"
    semantic = dataset_root / f"{condition}.jsonl.gz"
    if semantic.is_file():
        return semantic
    matches: list[Path] = []
    for candidate in sorted(dataset_root.glob("shard_*.jsonl.gz")):
        with gzip.open(candidate, "rt", encoding="utf-8") as handle:
            first = next((json.loads(line) for line in handle if json.loads(line).get("row_type") == "sampling_result"), None)
        if first is not None and str(first["condition"]) == condition:
            matches.append(candidate)
    if len(matches) != 1:
        raise FileNotFoundError(f"{dataset_root}: expected one shard for {condition}, found {matches}")
    return matches[0]


def read_results(path: Path, condition: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if json.loads(line).get("row_type") == "sampling_result"]
    expected_refs = int(config["reference_search"]["references_per_condition"])
    grid = radii(config)
    expected = {(ref, radius) for ref in range(expected_refs) for radius in range(grid.size)}
    coordinates = [(int(row["reference_index"]), int(row["radius_index"])) for row in rows]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"{path}: duplicate reference/radius coordinate")
    if set(coordinates) != expected:
        raise ValueError(f"{path}: incomplete reference/radius grid")
    split_count = int(config["sampling"]["independent_splits"])
    for row in rows:
        if str(row["condition"]) != condition:
            raise ValueError(f"{path}: condition drift")
        radius_index = int(row["radius_index"])
        if not math.isclose(float(row["radius"]), float(grid[radius_index]), rel_tol=0.0, abs_tol=2e-15):
            raise ValueError(f"{path}: radius drift")
        if len(row["splits"]) != split_count:
            raise ValueError(f"{path}: independent split count drift")
    return rows


def summarize_file(path: Path, condition: str, config: Mapping[str, Any]) -> dict[str, object]:
    rows = read_results(path, condition, config)
    parameter_count = int(config["model"]["parameter_count"])
    split_delta: list[float] = []; minimum_step: list[float] = []; minimum_pool: list[float] = []
    acceptance: list[float] = []; temperature_steps: list[float] = []; resamples: list[float] = []; derivative_finite: list[bool] = []
    for row in rows:
        splits = list(row["splits"])
        split_delta.append(abs(float(splits[0]["logz_angular"]) - float(splits[1]["logz_angular"])) / parameter_count)
        minimum_step.extend(float(split["minimum_step_overlap"]) for split in splits)
        minimum_pool.extend(float(split["minimum_pool_overlap_pre_resample"]) for split in splits)
        acceptance.extend(float(split["mean_mh_acceptance"]) for split in splits)
        temperature_steps.extend(float(split["temperature_step_count"]) for split in splits)
        resamples.extend(float(split["resample_count"]) for split in splits)
        derivative_finite.append(math.isfinite(float(row["dlogz_dr_direct"])))
    return {
        "sampling_unit_count": len(rows),
        "reference_count": int(config["reference_search"]["references_per_condition"]),
        "radius_count": int(config["sampling"]["radii_count"]),
        "independent_splits": int(config["sampling"]["independent_splits"]),
        "split_logz_difference_per_parameter_mean": float(np.mean(split_delta)),
        "split_logz_difference_per_parameter_max": float(np.max(split_delta)),
        "minimum_step_overlap_min": float(np.min(minimum_step)),
        "minimum_pool_overlap_pre_resample_min": float(np.min(minimum_pool)),
        "mean_mh_acceptance": float(np.mean(acceptance)),
        "mean_temperature_step_count": float(np.mean(temperature_steps)),
        "mean_resample_count": float(np.mean(resamples)),
        "direct_derivative_finite_fraction": float(np.mean(derivative_finite)),
    }


def build_rows(config: Mapping[str, Any], config_path: Path, raw_root: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    specs = condition_specs(config, config_path)
    pair = config["protocol"] == "digit_pairwise_sweep"
    coordinate_fields = ["digit_a", "digit_b", "pair_rank"] if pair else ["noise_eta"]
    for spec in specs:
        for dataset_index in config["production_replicates"]["dataset_indices"]:
            condition = str(spec["condition"]); index = int(dataset_index)
            coordinates = {key: spec[key] for key in coordinate_fields}
            rows.append({
                "domain": "mnist_digit_pair" if pair else "mnist_label_noise",
                "scale_id": str(config["thermodynamic_scale"]["scale_id"]),
                "shell_inv_temp_beta": float(config["thermodynamic_scale"]["shell_inverse_temperature_beta"]),
                "condition": condition,
                "condition_order": int(spec["condition_order"]),
                **coordinates,
                "dataset_index": index,
                **summarize_file(sampling_path(raw_root, index, condition), condition, config),
            })
    fields = ["domain", "scale_id", "shell_inv_temp_beta", "condition", "condition_order", *coordinate_fields, "dataset_index", *METRIC_FIELDS]
    validate_rows(rows, config, config_path)
    return rows, fields


def validate_rows(rows: list[Mapping[str, Any]], config: Mapping[str, Any], config_path: Path) -> None:
    specs = condition_specs(config, config_path)
    by_name = {str(row["condition"]): row for row in specs}
    expected = {(name, int(index)) for name in by_name for index in config["production_replicates"]["dataset_indices"]}
    actual = [(str(row["condition"]), int(row["dataset_index"])) for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError("sampling QC contains duplicate condition/dataset coordinates")
    if set(actual) != expected:
        raise ValueError("sampling QC has missing or unexpected coordinates")
    expected_units = int(config["reference_search"]["references_per_condition"]) * int(config["sampling"]["radii_count"])
    for row in rows:
        spec = by_name[str(row["condition"])]
        if int(row["condition_order"]) != int(spec["condition_order"]):
            raise ValueError("sampling QC condition order drift")
        if int(row["sampling_unit_count"]) != expected_units:
            raise ValueError("sampling QC unit count drift")
        for key in METRIC_FIELDS:
            if not math.isfinite(float(row[key])):
                raise ValueError(f"sampling QC contains non-finite {key}")
        for key, bounds in config["summary"]["finite_ranges"].items():
            value = float(row[key]); low, high = map(float, bounds)
            if not low <= value <= high:
                raise ValueError(f"sampling QC {key} outside [{low}, {high}]")
        for key in ("noise_eta", "digit_a", "digit_b", "pair_rank"):
            if key in spec and float(row[key]) != float(spec[key]):
                raise ValueError(f"sampling QC {key} drift")


def atomic_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true"); parser.add_argument("--execute", action="store_true"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.check_only and (args.execute or args.force): parser.error("--check-only cannot be combined with --execute or --force")
    config_path = args.config.resolve(); config = load_config(config_path); output = output_path(config, args.output_dir)
    if args.force or (args.execute and not output.is_file()):
        rows, fields = build_rows(config, config_path, sampling_root(config)); atomic_csv(output, rows, fields)
    if not output.is_file(): raise FileNotFoundError(f"{output}; run with --execute")
    rows = read_csv(output); validate_rows(rows, config, config_path)
    print({"status": "validated", "path": str(output), "rows": len(rows)}); return 0


if __name__ == "__main__": raise SystemExit(main())
