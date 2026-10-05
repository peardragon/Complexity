#!/usr/bin/env python3
"""Build or verify full-resolution Synthetic energetic summaries.

The ``--execute`` path rebuilds the two compact production tables directly
from the complete odd/even shell shards.  Without ``--execute`` the same
schema and coverage checks verify the existing tables without rewriting them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


MAX_CPU_WORKERS = 24
STAGE_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = SYNTHETIC_ROOT / "04_sampling" / "raw_outputs"
PROFILE_OUTPUT = STAGE_ROOT / "summarized_outputs" / "energetic_profiles.csv"
METRIC_OUTPUT = STAGE_ROOT / "summarized_outputs" / "condition_metrics.csv"
ACCURACY_CONFIG = STAGE_ROOT / "summarized_outputs" / "r1_weighted_accuracy.json"
CONFIG_PATH = STAGE_ROOT / "config/default.json"
ACCURACY_SOURCE = "not_rebuilt"


def configure(config_path=None, output_dir=None, raw_root=None):
    global CONFIG_PATH, BETAS, DATASETS_PER_CONDITION, REFERENCES_PER_DATASET, RADII, BASELINE_RADIUS, SCALE_ID, SHELL_BETA, PARAMETER_COUNT, ATV_MAX_RADIUS, RAW_ROOT, PROFILE_OUTPUT, METRIC_OUTPUT
    CONFIG_PATH = Path(config_path or CONFIG_PATH).resolve()
    config = json.loads(CONFIG_PATH.read_text())
    BETAS = tuple(map(float,config["beta_values"]))
    DATASETS_PER_CONDITION = int(config["selection"]["datasets_per_condition"])
    REFERENCES_PER_DATASET = int(config["selection"]["references_per_dataset"])
    spec = config["radius_grid"]
    RADII = tuple(round(float(spec["start"])+i*float(spec["step"]),12) for i in range(int(spec["count"])))
    BASELINE_RADIUS = float(spec["baseline_radius"])
    if RADII[0] != BASELINE_RADIUS: raise ValueError("baseline must be first measured radius")
    if not any(math.isclose(r,1.0,rel_tol=0,abs_tol=1e-12) for r in RADII):
        raise ValueError("r=1 must be present for the weighted-accuracy metric")
    SHELL_BETA = float(config["shell_inv_temp_beta"])
    PARAMETER_COUNT = int(config["parameter_count"])
    ATV_MAX_RADIUS = float(config["aggregation"]["atv_max_radius"])
    SCALE_ID = f"shell_beta_{SHELL_BETA:g}"
    RAW_ROOT = Path(raw_root) if raw_root else SYNTHETIC_ROOT / "04_sampling/raw_outputs"
    output = Path(output_dir) if output_dir else STAGE_ROOT / "summarized_outputs"
    PROFILE_OUTPUT, METRIC_OUTPUT = output / "energetic_profiles.csv", output / "condition_metrics.csv"

configure()
PROFILE_FIELDS = (
    "domain",
    "scale_id",
    "shell_inv_temp_beta",
    "condition",
    "condition_order",
    "data_beta",
    "radius_index",
    "radius",
    "baseline_radius",
    "dataset_count",
    "references_per_dataset",
    "phi_energetic_mean",
    "phi_energetic_se_across_datasets",
    "dphi_energetic_dr_direct_mean",
    "dphi_energetic_dr_direct_se_across_datasets",
)
METRIC_FIELDS = (
    "domain",
    "scale_id",
    "shell_inv_temp_beta",
    "condition",
    "condition_order",
    "data_beta",
    "A_TV_mean",
    "A_TV_se_across_datasets",
    "r1_weighted_accuracy_mean",
    "r1_weighted_accuracy_se_across_datasets",
)


def condition_slug(data_beta: float) -> str:
    return f"data_beta_{data_beta:.2f}".replace(".", "p")


def _atomic_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _accuracy_map() -> dict[float, tuple[float, float]]:
    payload = json.loads(ACCURACY_CONFIG.read_text(encoding="utf-8"))
    if (
        payload.get("artifact_id")
        != "synthetic_r1_weighted_accuracy_shell_beta_100_v1"
        or float(payload.get("radius", -1.0)) != 1.0
        or int(payload.get("dataset_count", -1)) != DATASETS_PER_CONDITION
        or int(payload.get("references_per_dataset", -1))
        != REFERENCES_PER_DATASET
    ):
        raise RuntimeError("r=1 weighted-accuracy contract differs")
    result = {
        round(float(row["data_beta"]), 2): (
            float(row["mean"]),
            float(row["se_across_datasets"]),
        )
        for row in payload["rows"]
    }
    if tuple(sorted(result)) != BETAS:
        raise RuntimeError("r=1 weighted-accuracy condition coverage differs")
    return result


def _scan_dataset_job(job_index: int) -> dict[str, Any]:
    phi = np.full((REFERENCES_PER_DATASET, len(RADII)), np.nan, dtype=np.float64)
    derivative = np.full_like(phi, np.nan)
    accuracy = np.full(REFERENCES_PER_DATASET, np.nan)
    data_beta: float | None = None
    dataset_id: int | None = None
    for shell_pass in ("odd", "even"):
        path = (
            RAW_ROOT
            / f"{shell_pass}_radius"
            / f"dataset_job_{job_index:04d}.jsonl.gz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        unit_count = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("record_type") != "shell_unit":
                    continue
                if (
                    row.get("derivative_method")
                    != "direct_autograd_radial_score"
                    or bool(row.get("finite_difference_first_derivative_used"))
                ):
                    raise RuntimeError(f"non-direct derivative in {path}")
                if float(row["gamma_ce"]) != 100.0 or float(row["lambda_reg"]) != 1.0:
                    raise RuntimeError(f"effective shell coefficients differ in {path}")
                reference_index = int(row["ref_id"])
                radius_index = int(row["radius_index"])
                if not (
                    0 <= reference_index < REFERENCES_PER_DATASET
                    and 0 <= radius_index < len(RADII)
                ):
                    raise RuntimeError(f"shell coordinate outside contract in {path}")
                if math.isfinite(float(phi[reference_index, radius_index])):
                    raise ValueError(f"duplicate reference/radius coordinate: {path}")
                if not math.isclose(float(row["radius"]),RADII[radius_index],abs_tol=1e-12) or not row.get("smc_completed",False):
                    raise ValueError(f"invalid radius/incomplete SMC: {path}")
                if math.isclose(float(row["radius"]),1.0,abs_tol=1e-12) and "weighted_training_accuracy" in row:
                    accuracy[reference_index] = float(row["weighted_training_accuracy"])
                phi[reference_index, radius_index] = float(row["phi_energy_absolute"])
                derivative[reference_index, radius_index] = float(
                    row["dphi_energy_dr_direct"]
                )
                current_beta = round(float(row["data_beta"]), 2)
                current_dataset = int(row["dataset_id"])
                if data_beta is None:
                    data_beta, dataset_id = current_beta, current_dataset
                elif (data_beta, dataset_id) != (current_beta, current_dataset):
                    raise RuntimeError(f"multiple datasets in {path}")
                unit_count += 1
        if unit_count != REFERENCES_PER_DATASET * (len(range(0 if shell_pass == "odd" else 1,len(RADII),2))):
            raise RuntimeError(f"unit-count mismatch in {path}")
    if data_beta is None or dataset_id is None:
        raise RuntimeError(f"empty shell dataset job {job_index}")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(derivative)):
        raise RuntimeError(f"incomplete odd/even radius union for job {job_index}")
    centered_phi = phi - phi[:, [0]]
    window = np.asarray(RADII) <= ATV_MAX_RADIUS + 1e-12
    normalized_r1_derivative = (derivative[:, window] - derivative[:, [0]]) / SHELL_BETA
    atv_by_reference = np.sum(
        np.abs(np.diff(normalized_r1_derivative, axis=1)), axis=1
    )
    return {
        "weighted_training_accuracy": float(np.mean(accuracy)) if np.isfinite(accuracy).all() else None,
        "accuracy_partial": bool(np.isfinite(accuracy).any() and not np.isfinite(accuracy).all()),
        "data_beta": data_beta,
        "dataset_id": dataset_id,
        "phi": np.mean(centered_phi, axis=0),
        "derivative": np.mean(derivative, axis=0),
        "A_TV": float(np.mean(atv_by_reference)),
    }


def _mean_and_se(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[axis] != DATASETS_PER_CONDITION:
        raise RuntimeError("dataset-first uncertainty requires exactly 60 datasets")
    return (
        np.mean(values, axis=axis),
        np.std(values, axis=axis, ddof=1) / math.sqrt(DATASETS_PER_CONDITION),
    )


def _build_from_raw(workers: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    with ProcessPoolExecutor(max_workers=workers, initializer=configure, initargs=(CONFIG_PATH, None, RAW_ROOT)) as executor:
        scanned = list(
            executor.map(
                _scan_dataset_job,
                range(len(BETAS) * DATASETS_PER_CONDITION),
            )
        )
    grouped: dict[float, list[dict[str, Any]]] = {beta: [] for beta in BETAS}
    for row in scanned:
        grouped[round(float(row["data_beta"]), 2)].append(row)
    profile_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    global ACCURACY_SOURCE
    archival_accuracy = None
    ACCURACY_SOURCE = "raw_terminal_particles"
    for condition_order, data_beta in enumerate(BETAS):
        datasets = sorted(grouped[data_beta], key=lambda row: int(row["dataset_id"]))
        if [int(row["dataset_id"]) for row in datasets] != list(
            range(DATASETS_PER_CONDITION)
        ):
            raise RuntimeError(f"{data_beta}: dataset coverage differs")
        phi_mean, phi_se = _mean_and_se(
            np.stack([np.asarray(row["phi"]) for row in datasets])
        )
        derivative_mean, derivative_se = _mean_and_se(
            np.stack([np.asarray(row["derivative"]) for row in datasets])
        )
        for radius_index, radius in enumerate(RADII):
            profile_rows.append(
                {
                    "domain": "synthetic",
                    "scale_id": SCALE_ID,
                    "shell_inv_temp_beta": SHELL_BETA,
                    "condition": condition_slug(data_beta),
                    "condition_order": condition_order,
                    "data_beta": data_beta,
                    "radius_index": radius_index,
                    "radius": radius,
                    "baseline_radius": BASELINE_RADIUS,
                    "dataset_count": DATASETS_PER_CONDITION,
                    "references_per_dataset": REFERENCES_PER_DATASET,
                    "phi_energetic_mean": float(phi_mean[radius_index]),
                    "phi_energetic_se_across_datasets": float(phi_se[radius_index]),
                    "dphi_energetic_dr_direct_mean": float(
                        derivative_mean[radius_index]
                    ),
                    "dphi_energetic_dr_direct_se_across_datasets": float(
                        derivative_se[radius_index]
                    ),
                }
            )
        atv = np.asarray([float(row["A_TV"]) for row in datasets])
        atv_mean, atv_se = _mean_and_se(atv)
        fresh = [row["weighted_training_accuracy"] for row in datasets]
        if any(row["accuracy_partial"] for row in datasets) or (any(value is None for value in fresh) and not all(value is None for value in fresh)):
            raise ValueError("partial raw accuracy coverage; do not mix raw and archival values")
        if all(value is None for value in fresh):
            archival_accuracy = archival_accuracy or _accuracy_map()
            accuracy_mean, accuracy_se = archival_accuracy[data_beta]
            ACCURACY_SOURCE = "archival_r1_authority_for_retained_raw"
        else:
            values = np.asarray(fresh,dtype=float)
            if np.any((values < 0) | (values > 1)): raise ValueError("accuracy outside [0,1]")
            accuracy_mean, accuracy_se = _mean_and_se(values)
        metric_rows.append(
            {
                "domain": "synthetic",
                "scale_id": SCALE_ID,
                "shell_inv_temp_beta": SHELL_BETA,
                "condition": condition_slug(data_beta),
                "condition_order": condition_order,
                "data_beta": data_beta,
                "A_TV_mean": float(atv_mean),
                "A_TV_se_across_datasets": float(atv_se),
                "r1_weighted_accuracy_mean": accuracy_mean,
                "r1_weighted_accuracy_se_across_datasets": accuracy_se,
            }
        )
    return profile_rows, metric_rows


def verify() -> None:
    with PROFILE_OUTPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PROFILE_FIELDS:
            raise RuntimeError("energetic_profiles.csv column contract differs")
        profiles = list(reader)
    with METRIC_OUTPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise RuntimeError("condition_metrics.csv column contract differs")
        metrics = list(reader)
    if len(profiles) != len(BETAS) * len(RADII) or len(metrics) != len(BETAS):
        raise RuntimeError("Synthetic production summary row counts differ")
    keys = {
        (int(row["condition_order"]), int(row["radius_index"])) for row in profiles
    }
    if keys != {
        (condition_order, radius_index)
        for condition_order in range(len(BETAS))
        for radius_index in range(len(RADII))
    }:
        raise RuntimeError("Synthetic profile coordinate coverage differs")
    for row in profiles:
        if (
            row["scale_id"] != SCALE_ID
            or float(row["shell_inv_temp_beta"]) != SHELL_BETA
        ):
            raise RuntimeError("Synthetic profile scale contract differs")
        for field in (
            "phi_energetic_mean",
            "phi_energetic_se_across_datasets",
            "dphi_energetic_dr_direct_mean",
            "dphi_energetic_dr_direct_se_across_datasets",
        ):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite profile field {field}")
    for row in metrics:
        if (
            row["scale_id"] != SCALE_ID
            or float(row["shell_inv_temp_beta"]) != SHELL_BETA
        ):
            raise RuntimeError("Synthetic metric scale contract differs")
        for field in (
            "A_TV_mean",
            "A_TV_se_across_datasets",
            "r1_weighted_accuracy_mean",
            "r1_weighted_accuracy_se_across_datasets",
        ):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite metric field {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="rebuild from the complete production odd/even raw shards",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs from raw inputs.")
    parser.add_argument("--check-only", action="store_true", help="Check existing compact outputs.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    configure(args.config, args.output_dir, args.raw_root)
    if args.check_only and (args.execute or args.force):
        parser.error("--check-only cannot be combined with --execute or --force")
    if not 1 <= args.workers <= MAX_CPU_WORKERS:
        raise ValueError(
            f"--workers must lie in [1,{MAX_CPU_WORKERS}] under the CPU cap"
        )
    if args.force or (args.execute and not (PROFILE_OUTPUT.is_file() and METRIC_OUTPUT.is_file())):
        profiles, metrics = _build_from_raw(args.workers)
        _atomic_csv(PROFILE_OUTPUT, PROFILE_FIELDS, profiles)
        _atomic_csv(METRIC_OUTPUT, METRIC_FIELDS, metrics)
    verify()
    print(f"{PROFILE_OUTPUT} rows={len(BETAS) * len(RADII)}")
    print(f"{METRIC_OUTPUT} rows={len(BETAS)} accuracy_source={ACCURACY_SOURCE}")


if __name__ == "__main__":
    main()
