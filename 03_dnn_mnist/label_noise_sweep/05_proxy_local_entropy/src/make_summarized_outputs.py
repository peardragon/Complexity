#!/usr/bin/env python3
"""Build and validate compact MNIST energetic profiles and condition metrics."""

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
DEFAULT_CONFIG = STAGE_ROOT / "config" / "default.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = "label_noise_sweep" if PROTOCOL_ROOT.name == "label_noise_sweep" else "digit_pairwise_sweep"
    if config.get("schema_version") != "mnist.proxy-local-entropy-summary.v1" or config.get("protocol") != expected:
        raise ValueError("unsupported proxy-local-entropy summary config")
    indices = [int(value) for value in config["dataset_indices"]]
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError("dataset_indices must be unique non-negative integers")
    grid = radius_grid(config)
    baseline = float(config["radius_grid"]["start"])
    if not math.isclose(float(grid[0]), baseline, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("baseline radius must be the first radius")
    if not math.isclose(float(grid[-1]), 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("r1 weighted accuracy requires the terminal radius to equal 1.0")
    if int(config["references_per_dataset"]) <= 0 or int(config["parameter_count"]) <= 0:
        raise ValueError("reference and parameter counts must be positive")
    return config


def radius_grid(config: Mapping[str, Any]) -> np.ndarray:
    spec = config["radius_grid"]
    count = int(spec["count"])
    if count <= 1 or float(spec["start"]) <= 0 or float(spec["stop"]) <= float(spec["start"]):
        raise ValueError("invalid radius grid")
    decimals = int(spec["decimals"])
    if not 0 <= decimals <= 15:
        raise ValueError("radius-grid decimals must lie in [0,15]")
    return np.round(
        np.linspace(float(spec["start"]), float(spec["stop"]), count, dtype=np.float64),
        decimals=decimals,
    )


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config["protocol"] == "label_noise_sweep":
        specs = [dict(row) for row in config["conditions"]]
    else:
        value = Path(str(config["pair_selection"]["frozen_manifest"]))
        path = value if value.is_absolute() else STAGE_ROOT / value
        frozen = json.loads(path.read_text(encoding="utf-8"))
        specs = [
            {"condition": str(row["pair_id"]), "digit_a": int(row["digit_a"]), "digit_b": int(row["digit_b"]), "pair_rank": int(row["rank"])}
            for row in frozen["expected_selected_pairs"]
        ]
    for order, row in enumerate(specs): row["condition_order"] = order
    names = [str(row["condition"]) for row in specs]
    if not names or len(names) != len(set(names)):
        raise ValueError("conditions must be non-empty and unique")
    return specs


def protocol_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROTOCOL_ROOT / path


def resolve_output(config: Mapping[str, Any], key: str, output_dir: Path | None) -> Path:
    configured = Path(str(config["outputs"][key]))
    return output_dir / configured.name if output_dir is not None else (configured if configured.is_absolute() else STAGE_ROOT / configured)


def mean_and_se(values: np.ndarray, dataset_count: int, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[axis] != dataset_count:
        raise ValueError(f"dataset-first summary expected {dataset_count} values")
    return np.mean(array, axis=axis), np.std(array, axis=axis, ddof=1) / math.sqrt(dataset_count)


def sampling_path(root: Path, dataset_index: int, condition: str) -> Path:
    dataset_root = root / f"dataset_{dataset_index:03d}"
    semantic = dataset_root / f"{condition}.jsonl.gz"
    if semantic.is_file(): return semantic
    matches: list[Path] = []
    for candidate in sorted(dataset_root.glob("shard_*.jsonl.gz")):
        with gzip.open(candidate, "rt", encoding="utf-8") as handle:
            first = None
            for line in handle:
                row = json.loads(line)
                if row.get("row_type") == "sampling_result": first = row; break
        if first is not None and str(first["condition"]) == condition: matches.append(candidate)
    if len(matches) != 1: raise FileNotFoundError(f"{dataset_root}: expected one shard for {condition}, found {matches}")
    return matches[0]


def read_curves(path: Path, condition: str, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    refs = int(config["references_per_dataset"]); grid = radius_grid(config)
    logz = np.full((refs, grid.size), np.nan); direct = np.full_like(logz, np.nan)
    accuracy = np.full(refs, np.nan); accuracy_presence: list[bool] = []
    seen: set[tuple[int, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("row_type") != "sampling_result": continue
            if str(row["condition"]) != condition: raise ValueError(f"{path}: condition drift")
            ref = int(row["reference_index"]); radius_index = int(row["radius_index"]); key = (ref, radius_index)
            if key in seen: raise ValueError(f"{path}: duplicate unit {key}")
            if not 0 <= ref < refs or not 0 <= radius_index < grid.size: raise ValueError(f"{path}: coordinate outside configured grid")
            if not math.isclose(float(row["radius"]), float(grid[radius_index]), rel_tol=0.0, abs_tol=2e-15): raise ValueError(f"{path}: radius drift")
            has_accuracy = "weighted_training_accuracy" in row
            if has_accuracy and radius_index != grid.size - 1: raise ValueError(f"{path}: training accuracy present away from terminal radius")
            if radius_index == grid.size - 1:
                accuracy_presence.append(has_accuracy)
                if has_accuracy:
                    value = float(row["weighted_training_accuracy"])
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0: raise ValueError(f"{path}: invalid weighted training accuracy")
                    accuracy[ref] = value
            seen.add(key); logz[ref, radius_index] = float(row["logz_angular"]); direct[ref, radius_index] = float(row["dlogz_dr_direct"])
    expected = {(ref, radius) for ref in range(refs) for radius in range(grid.size)}
    if seen != expected or not np.all(np.isfinite(logz)) or not np.all(np.isfinite(direct)): raise ValueError(f"{path}: incomplete or non-finite shell grid")
    if all(accuracy_presence): return logz, direct, accuracy
    if not any(accuracy_presence): return logz, direct, None
    raise ValueError(f"{path}: mixed fresh and archival terminal-accuracy fields")


def load_accuracy_authority(config: Mapping[str, Any], expected_conditions: set[str]) -> dict[str, Any]:
    path = protocol_path(config["paths"]["accuracy_authority"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scale_id") != config["scale_id"] or int(payload["dataset_count"]) != len(config["dataset_indices"]) or int(payload["references_per_dataset"]) != int(config["references_per_dataset"]):
        raise ValueError("archival accuracy authority contract drift")
    if set(payload.get("conditions", {})) != expected_conditions: raise ValueError("archival accuracy authority condition coverage drift")
    return payload


def build_outputs(config: Mapping[str, Any]) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]], list[str], dict[str, str]]:
    specs = condition_specs(config); condition_names = {str(row["condition"]) for row in specs}
    authority: dict[str, Any] | None = None
    grid = radius_grid(config); dataset_indices = [int(value) for value in config["dataset_indices"]]; dataset_count = len(dataset_indices)
    parameter_count = float(config["parameter_count"]); shell_beta = float(config["shell_inv_temp_beta"])
    pair = config["protocol"] == "digit_pairwise_sweep"; coordinate_fields = ["digit_a", "digit_b", "pair_rank"] if pair else ["noise_eta"]
    profiles: list[dict[str, object]] = []; metrics: list[dict[str, object]] = []; sources: dict[str, str] = {}
    raw_root = protocol_path(config["paths"]["sampling_root"])
    for spec in specs:
        condition = str(spec["condition"]); coordinates = {key: spec[key] for key in coordinate_fields}
        dataset_phi: list[np.ndarray] = []; dataset_direct: list[np.ndarray] = []; dataset_atv: list[float] = []; dataset_accuracy: list[float] = []
        availability: list[bool] = []
        for dataset_index in dataset_indices:
            logz, direct, terminal_accuracy = read_curves(sampling_path(raw_root, dataset_index, condition), condition, config)
            phi = (logz - logz[:, [0]]) / parameter_count; dphi = direct / parameter_count
            dataset_phi.append(np.mean(phi, axis=0)); dataset_direct.append(np.mean(dphi, axis=0))
            dataset_atv.append(float(np.mean(np.sum(np.abs(np.diff(dphi, axis=1)), axis=1) / shell_beta)))
            availability.append(terminal_accuracy is not None)
            if terminal_accuracy is not None: dataset_accuracy.append(float(np.mean(terminal_accuracy)))
        if all(availability):
            accuracy_mean, accuracy_se = mean_and_se(np.asarray(dataset_accuracy), dataset_count); sources[condition] = "fresh_terminal_particles"
        elif not any(availability):
            if authority is None:
                authority = load_accuracy_authority(config, condition_names)
            row = authority["conditions"][condition]; accuracy_mean = np.asarray(float(row["mean"])); accuracy_se = np.asarray(float(row["se_across_datasets"])); sources[condition] = "archival_r1_weighted_accuracy"
        else:
            raise ValueError(f"{condition}: refusing mixed fresh and archival accuracy sources")
        phi_mean, phi_se = mean_and_se(np.asarray(dataset_phi), dataset_count); direct_mean, direct_se = mean_and_se(np.asarray(dataset_direct), dataset_count)
        if not (phi_mean[0] == 0.0 and phi_se[0] == 0.0): raise ValueError(f"{condition}: baseline anchoring is not exact")
        for radius_index, radius in enumerate(grid):
            profiles.append({
                "domain": "mnist_digit_pair" if pair else "mnist_label_noise", "scale_id": config["scale_id"], "shell_inv_temp_beta": shell_beta,
                "condition": condition, "condition_order": int(spec["condition_order"]), **coordinates,
                "radius_index": radius_index, "radius": float(radius), "baseline_radius": float(grid[0]), "dataset_count": dataset_count,
                "references_per_dataset": int(config["references_per_dataset"]), "phi_energetic_mean": float(phi_mean[radius_index]),
                "phi_energetic_se_across_datasets": float(phi_se[radius_index]), "dphi_energetic_dr_direct_mean": float(direct_mean[radius_index]),
                "dphi_energetic_dr_direct_se_across_datasets": float(direct_se[radius_index]),
            })
        atv_mean, atv_se = mean_and_se(np.asarray(dataset_atv), dataset_count)
        metrics.append({
            "domain": "mnist_digit_pair" if pair else "mnist_label_noise", "scale_id": config["scale_id"], "shell_inv_temp_beta": shell_beta,
            "condition": condition, "condition_order": int(spec["condition_order"]), **coordinates, "A_TV_mean": float(atv_mean),
            "A_TV_se_across_datasets": float(atv_se), "r1_weighted_accuracy_mean": float(accuracy_mean),
            "r1_weighted_accuracy_se_across_datasets": float(accuracy_se),
        })
    profile_fields = ["domain", "scale_id", "shell_inv_temp_beta", "condition", "condition_order", *coordinate_fields, "radius_index", "radius", "baseline_radius", "dataset_count", "references_per_dataset", "phi_energetic_mean", "phi_energetic_se_across_datasets", "dphi_energetic_dr_direct_mean", "dphi_energetic_dr_direct_se_across_datasets"]
    metric_fields = ["domain", "scale_id", "shell_inv_temp_beta", "condition", "condition_order", *coordinate_fields, "A_TV_mean", "A_TV_se_across_datasets", "r1_weighted_accuracy_mean", "r1_weighted_accuracy_se_across_datasets"]
    validate_outputs(profiles, metrics, config); return profiles, profile_fields, metrics, metric_fields, sources


def validate_outputs(profiles: list[Mapping[str, Any]], metrics: list[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    specs = condition_specs(config); by_name = {str(row["condition"]): row for row in specs}; grid = radius_grid(config)
    expected_profile = {(name, index) for name in by_name for index in range(grid.size)}
    actual_profile = [(str(row["condition"]), int(row["radius_index"])) for row in profiles]
    if len(actual_profile) != len(set(actual_profile)): raise ValueError("profiles contain duplicate condition/radius coordinates")
    if set(actual_profile) != expected_profile: raise ValueError("profiles have missing or unexpected condition/radius coordinates")
    metric_names = [str(row["condition"]) for row in metrics]
    if len(metric_names) != len(set(metric_names)) or set(metric_names) != set(by_name): raise ValueError("metrics have missing or duplicate conditions")
    expected_domain = "mnist_digit_pair" if config["protocol"] == "digit_pairwise_sweep" else "mnist_label_noise"
    for row in [*profiles, *metrics]:
        spec = by_name[str(row["condition"])]
        if str(row["domain"]) != expected_domain or int(row["condition_order"]) != int(spec["condition_order"]): raise ValueError("summary condition metadata drift")
        for key in ("noise_eta", "digit_a", "digit_b", "pair_rank"):
            if key in spec and float(row[key]) != float(spec[key]): raise ValueError(f"summary {key} drift")
        for value in row.values():
            try: number = float(value)
            except (TypeError, ValueError): continue
            if not math.isfinite(number): raise ValueError("summary contains a non-finite numeric value")
    for row in profiles:
        index = int(row["radius_index"])
        if not math.isclose(float(row["radius"]), float(grid[index]), rel_tol=0.0, abs_tol=2e-15): raise ValueError("profile radius drift")
        if int(row["dataset_count"]) != len(config["dataset_indices"]) or int(row["references_per_dataset"]) != int(config["references_per_dataset"]): raise ValueError("profile replicate-count drift")
    for row in metrics:
        accuracy = float(row["r1_weighted_accuracy_mean"]); accuracy_se = float(row["r1_weighted_accuracy_se_across_datasets"])
        if not 0.0 <= accuracy <= 1.0 or accuracy_se < 0.0: raise ValueError("invalid accuracy metric")


def atomic_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true"); parser.add_argument("--execute", action="store_true"); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    if args.check_only and (args.execute or args.force): parser.error("--check-only cannot be combined with --execute or --force")
    config = load_config(args.config.resolve()); profile_path = resolve_output(config, "energetic_profiles", args.output_dir); metric_path = resolve_output(config, "condition_metrics", args.output_dir)
    missing = [path for path in (profile_path, metric_path) if not path.is_file()]; sources: dict[str, str] = {}
    if args.force or (args.execute and missing):
        profiles, profile_fields, metrics, metric_fields, sources = build_outputs(config)
        if args.force or not profile_path.is_file(): atomic_csv(profile_path, profiles, profile_fields)
        if args.force or not metric_path.is_file(): atomic_csv(metric_path, metrics, metric_fields)
    if not profile_path.is_file() or not metric_path.is_file(): raise FileNotFoundError("summary output missing; run with --execute")
    profiles = read_csv(profile_path); metrics = read_csv(metric_path); validate_outputs(profiles, metrics, config)
    print({"status": "validated", "profiles": str(profile_path), "profile_rows": len(profiles), "metrics": str(metric_path), "metric_rows": len(metrics), "accuracy_sources": sources or "existing_compact_outputs_not_reconstructed"}); return 0


if __name__ == "__main__": raise SystemExit(main())
