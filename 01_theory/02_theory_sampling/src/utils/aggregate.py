"""Deterministic nested aggregation of completed 01 sampling shards."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from schedule import (
    build_rows,
    radii_from_config,
    sha256_file,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "01_theory"
    / "02_theory_sampling"
    / "config"
    / "default.json"
)


def aggregation_code_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("aggregate.py", "run_sampling_shard.py"):
        path = SOURCE_DIR / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_unit_rows(shard_root: Path, shard_count: int) -> list[dict[str, Any]]:
    """Read complete scalar shards without code/environment identity gates."""
    rows = []
    for index in range(shard_count):
        path = shard_root / f"shard_{index:04d}_of_{shard_count:04d}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records or records[0].get("record_type") != "shard_metadata":
            raise ValueError(f"missing shard header: {path}")
        units = records[1:]
        if len(units) != int(records[0]["unit_count"]):
            raise ValueError(f"incomplete shard: {path}")
        for row in units:
            if row.get("record_type") != "sampling_unit" or int(row["unit_id"]) % shard_count != index:
                raise ValueError(f"invalid sampling coordinate: {path}")
            if not row.get("smc_completed") or row.get("derivative_method") != "direct_particle_radial_score":
                raise ValueError(f"unfinished/non-direct sampling result: {path}")
            for key in ("logZ_angular_full", "dlogZ_dr_direct", "weighted_ce_sum", "weighted_error", "split_logZ_per_N_difference", "split_dlogZ_dr_per_N_difference"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"nonfinite {key}: {path}")
        rows.extend(units)
    rows.sort(key=lambda row: int(row["unit_id"]))
    if len({int(row["unit_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate sampling unit")
    return rows


def load_completed_sampling(config_path: Path):
    config_path = project_path(config_path)
    config = json.loads(config_path.read_text())
    validate_config(config)
    expected = build_rows(config, "")
    rows = load_unit_rows(project_path(config["artifacts"]["shard_output_directory"]), int(config["artifacts"]["default_shard_count"]))
    if len(rows) != len(expected):
        raise ValueError(f"expected {len(expected)} units, found {len(rows)}")
    fields = ("unit_id", "N", "dataset_id", "reference_id", "radius_index", "total_particles")
    for actual, planned in zip(rows, expected):
        if any(int(actual[key]) != int(planned[key]) for key in fields):
            raise ValueError(f"sampling coordinate mismatch: {actual['unit_id']}")
        if not math.isclose(float(actual["radius"]), float(planned["radius"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError("sampling radius differs from configured grid")
        if actual["split_seeds"] != planned["split_seeds"]:
            raise ValueError("sampling seeds differ from the paper coordinate scheme")
    return config, rows, {"input_unit_count": len(rows)}, {"input_shard_root": str(config["artifacts"]["shard_output_directory"])}


def mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("aggregate contains missing or non-finite values")
    sem = float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return float(np.mean(array)), sem


def nested_summary(rows: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Average references within dataset, then datasets within each condition."""
    config = config or json.loads(DEFAULT_CONFIG.read_text())
    dataset_count = int(config["problem"]["datasets_per_dimension"])
    reference_count = int(config["problem"]["references_per_dataset"])
    fields = (
        "logZ_angular_full",
        "dlogZ_dr_direct",
        "weighted_ce_sum",
        "weighted_error",
        "split_logZ_per_N_difference",
        "split_dlogZ_dr_per_N_difference",
    )
    grouped: dict[tuple[int, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["N"]),
            int(row["total_particles"]),
            int(row["dataset_id"]),
            float(row["radius"]),
        )
        grouped[key].append(row)
    dataset_means: dict[tuple[int, int, int, float], dict[str, float]] = {}
    for key, group in grouped.items():
        if len(group) != reference_count or {int(row["reference_id"]) for row in group} != set(range(reference_count)):
            raise RuntimeError(f"incomplete 10-reference dataset cell: {key}")
        dataset_means[key] = {
            field: float(np.mean([float(row[field]) for row in group]))
            for field in fields
        }

    conditions = sorted(
        {
            (int(row["N"]), int(row["total_particles"]))
            for row in rows
        }
    )
    radii_by_condition: dict[tuple[int, int], list[float]] = {}
    for condition in conditions:
        radii_by_condition[condition] = sorted(
            {
                radius
                for n_value, particles, _, radius in dataset_means
                if (n_value, particles) == condition
            }
        )

    output: list[dict[str, Any]] = []
    for (dimension, total_particles), radii in conditions_with_radii(
        radii_by_condition
    ):
        baseline_radius = radii[0]
        if not math.isclose(baseline_radius, 0.15, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("expected r0=0.15")
        for radius in radii:
            energy_by_dataset: list[float] = []
            shell_by_dataset: list[float] = []
            derivative_energy_by_dataset: list[float] = []
            derivative_shell_by_dataset: list[float] = []
            diagnostics: dict[str, list[float]] = defaultdict(list)
            for dataset_id in range(dataset_count):
                key = (dimension, total_particles, dataset_id, radius)
                base_key = (
                    dimension,
                    total_particles,
                    dataset_id,
                    baseline_radius,
                )
                if key not in dataset_means or base_key not in dataset_means:
                    raise RuntimeError(f"incomplete 10-dataset condition cell: {key}")
                current = dataset_means[key]
                baseline = dataset_means[base_key]
                energy_relative = (
                    current["logZ_angular_full"]
                    - baseline["logZ_angular_full"]
                ) / dimension
                shell_relative = energy_relative + (
                    (dimension - 2.0) / dimension
                ) * math.log(radius / baseline_radius)
                dphi_energy = current["dlogZ_dr_direct"] / dimension
                dphi_shell = dphi_energy + (
                    dimension - 2.0
                ) / (dimension * radius)
                energy_by_dataset.append(energy_relative)
                shell_by_dataset.append(shell_relative)
                derivative_energy_by_dataset.append(dphi_energy)
                derivative_shell_by_dataset.append(dphi_shell)
                for field in fields[2:]:
                    diagnostics[field].append(current[field])
            phi_energy, phi_energy_sem = mean_sem(energy_by_dataset)
            phi_shell, phi_shell_sem = mean_sem(shell_by_dataset)
            dphi_energy, dphi_energy_sem = mean_sem(
                derivative_energy_by_dataset
            )
            dphi_shell, dphi_shell_sem = mean_sem(
                derivative_shell_by_dataset
            )
            output.append(
                {
                    "N": dimension,
                    "total_particles": total_particles,
                    "particles_per_split": total_particles // 2,
                    "radius": radius,
                    "baseline_radius": baseline_radius,
                    "dataset_count": dataset_count,
                    "references_per_dataset": reference_count,
                    "phi_energy_rel": phi_energy,
                    "phi_energy_rel_dataset_sem": phi_energy_sem,
                    "phi_shell_rel": phi_shell,
                    "phi_shell_rel_dataset_sem": phi_shell_sem,
                    "dphi_energy_dr_direct": dphi_energy,
                    "dphi_energy_dr_direct_dataset_sem": dphi_energy_sem,
                    "dphi_shell_dr_direct": dphi_shell,
                    "dphi_shell_dr_direct_dataset_sem": dphi_shell_sem,
                    "weighted_ce_sum": float(
                        np.mean(diagnostics["weighted_ce_sum"])
                    ),
                    "weighted_error": float(
                        np.mean(diagnostics["weighted_error"])
                    ),
                    "max_dataset_mean_split_logZ_per_N_difference": float(
                        np.max(
                            diagnostics["split_logZ_per_N_difference"]
                        )
                    ),
                    "max_dataset_mean_split_dlogZ_per_N_difference": float(
                        np.max(
                            diagnostics[
                                "split_dlogZ_dr_per_N_difference"
                            ]
                        )
                    ),
                    "first_derivative_method": "direct_particle_radial_score",
                }
            )
    return output


def conditions_with_radii(
    mapping: dict[tuple[int, int], list[float]]
) -> list[tuple[tuple[int, int], list[float]]]:
    return [(condition, mapping[condition]) for condition in sorted(mapping)]


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot write empty summary")
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def validate_summary(path: Path, config: dict[str, Any]) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    problem, pool = config["problem"], config["pool_design"]
    expected = {(int(n), int(p), round(r, 12)) for n in problem["dimensions"] for p in (pool["convergence_total_particles"] if int(n) == int(pool["convergence_dimension"]) else [pool["thermodynamic_total_particles"]]) for r in radii_from_config(config)}
    observed = [(int(row["N"]), int(row["total_particles"]), round(float(row["radius"]), 12)) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("theory summary has missing/duplicate coordinates")
    for row in rows:
        for key, value in row.items():
            if key == "first_derivative_method":
                if value != "direct_particle_radial_score": raise ValueError("non-direct derivative")
            elif not math.isfinite(float(value)):
                raise ValueError(f"nonfinite {key}")
        n, r, r0 = float(row["N"]), float(row["radius"]), float(row["baseline_radius"])
        expected_shell = float(row["phi_energy_rel"]) + (n-2)/n * math.log(r/r0)
        expected_derivative = float(row["dphi_energy_dr_direct"]) + (n-2)/(n*r)
        if not math.isclose(float(row["phi_shell_rel"]), expected_shell, rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(float(row["dphi_shell_dr_direct"]), expected_derivative, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("full-shell columns do not use the squared-distance delta convention")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(project_path(args.config).read_text())
    output = args.output_dir or project_path(config["artifacts"]["summary_output_directory"])
    destination = output / "phi_by_sampling.csv"
    if args.check_only or (destination.is_file() and not args.force):
        validate_summary(destination, config)
        print(f"{destination}: {'validated' if args.check_only else 'skipped_existing'}")
        return
    config, rows, _, _ = load_completed_sampling(args.config)
    summary = nested_summary(rows, config)
    atomic_write(destination, csv_bytes(summary))
    validate_summary(destination, config)
    print(f"{destination}: written {len(summary)} rows from {len(rows)} sampling units")


if __name__ == "__main__":
    main()
