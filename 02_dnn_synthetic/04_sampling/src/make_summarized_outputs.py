#!/usr/bin/env python3
"""Build compact shell_beta_100 sampling-quality diagnostics from both passes."""

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


MAX_CPU_WORKERS = 24
STAGE_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = STAGE_ROOT / "raw_outputs"
OUTPUT = STAGE_ROOT / "summarized_outputs" / "logz_sampling_quality.csv"
CONFIG_PATH = STAGE_ROOT / "config/default.json"


def configure(config_path=None, output_dir=None, raw_root=None):
    global CONFIG_PATH, PASSES, BETAS, SHARDS_PER_CONDITION, UNITS_PER_SHARD, REFERENCES, RADII, RAW_ROOT, OUTPUT
    CONFIG_PATH = Path(config_path or CONFIG_PATH).resolve()
    config = json.loads(CONFIG_PATH.read_text())
    BETAS = tuple(map(float, config["dataset"]["beta_values"]))
    SHARDS_PER_CONDITION = int(config["dataset"]["datasets_per_beta"])
    REFERENCES = int(config["reference_search"]["references_per_dataset"])
    spec = config["shell"]["radii"]
    RADII = tuple(round(float(spec["start"])+i*float(spec["step"]),12) for i in range(int(spec["count"])))
    PASSES = tuple(config["shell"]["execution"]["pass_order"])
    UNITS_PER_SHARD = REFERENCES * int(config["shell"]["execution"]["radii_per_pass"])
    RAW_ROOT = Path(raw_root) if raw_root else STAGE_ROOT / "raw_outputs"
    OUTPUT = (Path(output_dir) if output_dir else STAGE_ROOT / "summarized_outputs") / "logz_sampling_quality.csv"


configure()
FIELDS = (
    "domain",
    "scale_id",
    "condition",
    "condition_order",
    "data_beta",
    "shell_inv_temp_beta",
    "dataset_count",
    "references_per_dataset",
    "radius_count",
    "odd_shard_count",
    "even_shard_count",
    "shell_unit_count",
    "split_logz_per_parameter_difference_mean",
    "split_logz_per_parameter_difference_max",
    "split_dlogz_per_parameter_difference_mean",
    "split_dlogz_per_parameter_difference_max",
    "ce_replay_max_abs_difference",
    "finite_difference_first_derivative_used",
)


def condition_slug(data_beta: float) -> str:
    return f"data_beta_{data_beta:.2f}".replace(".", "p")


def _scan_one(item: tuple[str, int]) -> dict[str, Any]:
    shell_pass, job_index = item
    path = (
        RAW_ROOT
        / f"{shell_pass}_radius"
        / f"dataset_job_{job_index:04d}.jsonl.gz"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    header: dict[str, Any] | None = None
    trailer: dict[str, Any] | None = None
    count = 0
    seen = set()
    radius_indices = set(range(0 if shell_pass == "odd" else 1, len(RADII), 2))
    logz_sum = 0.0
    logz_max = 0.0
    dlogz_sum = 0.0
    dlogz_max = 0.0
    replay_max = 0.0
    data_beta: float | None = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_type = row.get("record_type")
            if record_type == "shard_header":
                header = row
                continue
            if record_type == "shard_trailer":
                trailer = row
                continue
            if record_type != "shell_unit":
                raise RuntimeError(f"unknown record type in {path}")
            if row.get("shell_pass") != shell_pass:
                raise RuntimeError(f"shell-pass mismatch in {path}")
            if (
                row.get("derivative_method") != "direct_autograd_radial_score"
                or bool(row.get("finite_difference_first_derivative_used"))
            ):
                raise RuntimeError(f"non-direct derivative in {path}")
            key = (int(row["ref_id"]), int(row["radius_index"]))
            if key in seen or key[0] not in range(REFERENCES) or key[1] not in radius_indices:
                raise ValueError(f"duplicate/invalid sampling coordinate: {path}")
            seen.add(key)
            if int(row["dataset_id"]) != job_index % SHARDS_PER_CONDITION or not math.isclose(float(row["data_beta"]), BETAS[job_index // SHARDS_PER_CONDITION],abs_tol=1e-12) or not math.isclose(float(row["radius"]), RADII[key[1]],abs_tol=1e-12):
                raise ValueError(f"sampling dataset/radius metadata differs: {path}")
            if not row.get("smc_completed", False): raise ValueError(f"incomplete SMC: {path}")
            value = float(row["split_logZ_per_parameter_difference"])
            derivative_value = float(row["split_dlogZ_per_parameter_difference"])
            replay_value = float(row["ce_replay_max_abs_difference"])
            logz_sum += value
            logz_max = max(logz_max, value)
            dlogz_sum += derivative_value
            dlogz_max = max(dlogz_max, derivative_value)
            replay_max = max(replay_max, replay_value)
            current_beta = round(float(row["data_beta"]), 12)
            if data_beta is None:
                data_beta = current_beta
            elif data_beta != current_beta:
                raise RuntimeError(f"multiple data_beta values in {path}")
            count += 1
    if header is None or trailer is None or data_beta is None:
        raise RuntimeError(f"incomplete compact shard: {path}")
    if count != UNITS_PER_SHARD or int(trailer.get("unit_count", -1)) != count:
        raise RuntimeError(f"unit-count mismatch in {path}")
    return {
        "shell_pass": shell_pass,
        "data_beta": data_beta,
        "unit_count": count,
        "logz_sum": logz_sum,
        "logz_max": logz_max,
        "dlogz_sum": dlogz_sum,
        "dlogz_max": dlogz_max,
        "replay_max": replay_max,
    }


def build(workers: int) -> list[dict[str, object]]:
    tasks = [
        (shell_pass, job_index)
        for shell_pass in PASSES
        for job_index in range(len(BETAS) * SHARDS_PER_CONDITION)
    ]
    with ProcessPoolExecutor(max_workers=workers, initializer=configure, initargs=(CONFIG_PATH, None, RAW_ROOT)) as executor:
        scanned = list(executor.map(_scan_one, tasks))
    grouped: dict[float, list[dict[str, Any]]] = {beta: [] for beta in BETAS}
    for row in scanned:
        grouped[round(float(row["data_beta"]), 2)].append(row)
    output: list[dict[str, object]] = []
    for condition_order, data_beta in enumerate(BETAS):
        rows = grouped[data_beta]
        expected = 2 * SHARDS_PER_CONDITION
        if len(rows) != expected:
            raise RuntimeError(f"{data_beta}: expected {expected} shards, found {len(rows)}")
        unit_count = sum(int(row["unit_count"]) for row in rows)
        output.append(
            {
                "domain": "synthetic",
                "scale_id": "shell_beta_100",
                "condition": condition_slug(data_beta),
                "condition_order": condition_order,
                "data_beta": data_beta,
                "shell_inv_temp_beta": 100.0,
                "dataset_count": SHARDS_PER_CONDITION,
                "references_per_dataset": REFERENCES,
                "radius_count": len(RADII),
                "odd_shard_count": sum(row["shell_pass"] == "odd" for row in rows),
                "even_shard_count": sum(row["shell_pass"] == "even" for row in rows),
                "shell_unit_count": unit_count,
                "split_logz_per_parameter_difference_mean": sum(
                    float(row["logz_sum"]) for row in rows
                )
                / unit_count,
                "split_logz_per_parameter_difference_max": max(
                    float(row["logz_max"]) for row in rows
                ),
                "split_dlogz_per_parameter_difference_mean": sum(
                    float(row["dlogz_sum"]) for row in rows
                )
                / unit_count,
                "split_dlogz_per_parameter_difference_max": max(
                    float(row["dlogz_max"]) for row in rows
                ),
                "ce_replay_max_abs_difference": max(
                    float(row["replay_max"]) for row in rows
                ),
                "finite_difference_first_derivative_used": False,
            }
        )
    return output


def atomic_write(rows: list[dict[str, object]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{OUTPUT.name}.", suffix=".tmp", dir=OUTPUT.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def verify() -> None:
    with OUTPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise RuntimeError("logz_sampling_quality.csv column contract differs")
        rows = list(reader)
    if len(rows) != len(BETAS):
        raise RuntimeError(f"expected 18 quality rows, found {len(rows)}")
    keys = [int(row["condition_order"]) for row in rows]
    if len(set(keys)) != len(BETAS) or set(keys) != set(range(len(BETAS))): raise ValueError("duplicate/missing QC conditions")
    for row in rows:
        beta = BETAS[int(row["condition_order"])]
        if row["condition"] != condition_slug(beta) or not math.isclose(float(row["data_beta"]),beta,abs_tol=1e-12): raise ValueError("QC condition metadata differs")
        if (
            row["scale_id"] != "shell_beta_100"
            or float(row["shell_inv_temp_beta"]) != 100.0
            or
            int(row["radius_count"]) != len(RADII)
            or int(row["odd_shard_count"]) != SHARDS_PER_CONDITION
            or int(row["even_shard_count"]) != SHARDS_PER_CONDITION
            or int(row["shell_unit_count"]) != SHARDS_PER_CONDITION * REFERENCES * len(RADII)
            or row["finite_difference_first_derivative_used"].lower() not in {"false", "0"}
        ):
            raise RuntimeError("sampling-quality coverage contract differs")
        for field in (
            "split_logz_per_parameter_difference_mean",
            "split_logz_per_parameter_difference_max",
            "split_dlogz_per_parameter_difference_mean",
            "split_dlogz_per_parameter_difference_max",
            "ce_replay_max_abs_difference",
        ):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite sampling diagnostic {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="scan both compact raw passes")
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
    if args.force or (args.execute and not (OUTPUT.is_file())):
        atomic_write(build(args.workers))
    verify()
    print(f"{OUTPUT} rows={len(BETAS)}")


if __name__ == "__main__":
    main()
