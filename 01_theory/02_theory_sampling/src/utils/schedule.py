"""Config-driven 33,600-unit sampling manifest and injective seed scheme."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "01_theory"
    / "02_theory_sampling"
    / "config"
    / "default.json"
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def radii_from_config(config: dict[str, Any]) -> tuple[float, ...]:
    problem = config["problem"]
    start_cents = int(round(100 * float(problem["radius_start"])))
    stop_cents = int(round(100 * float(problem["radius_stop"])))
    step_cents = int(round(100 * float(problem["radius_step"])))
    radii = tuple(value / 100.0 for value in range(start_cents, stop_cents + 1, step_cents))
    if len(radii) != int(problem["radius_count"]):
        raise ValueError("radius specification does not produce radius_count")
    return radii


def validate_config(config: dict[str, Any]) -> None:
    objective = config["objective"]
    frozen = (
        objective["domain"] == "perceptron_theory"
        and objective["reference_measure"] == "hard_feasible_truncated_gaussian"
        and objective["ce_reduction"] == "sum"
        and float(objective["beta"]) == 1.0
        and float(objective["lambda_ref"]) == 1.0
        and float(objective["lambda_shell"]) == 1.0
        and objective["dnn_h_0p01_applies"] is False
    )
    if not frozen:
        raise ValueError("01 objective differs from the frozen perceptron construction")
    problem = config["problem"]
    if list(map(int, problem["dimensions"])) != [40, 80, 160, 320]:
        raise ValueError("dimensions must be [40,80,160,320]")
    if int(problem["datasets_per_dimension"]) != 10:
        raise ValueError("datasets_per_dimension must be 10")
    if int(problem["references_per_dataset"]) != 10:
        raise ValueError("references_per_dataset must be 10")
    radii_from_config(config)
    design = config["pool_design"]
    ladder = list(map(int, design["convergence_total_particles"]))
    if ladder != [2048, 4096, 8192, 16384, 32768]:
        raise ValueError("particle ladder differs from the approved design")
    if int(design["split_count"]) != 2 or any(value % 2 for value in ladder):
        raise ValueError("all total particle counts must split equally in two")
    smc = config["smc"]
    expected_smc = {
        "target_step_overlap": 0.85,
        "resample_pool_overlap": 0.85,
        "max_temperature_events": 160,
        "minimum_temperature_increment": 5.0e-5,
        "bisection_iterations": 36,
        "mh_sweeps_per_event": 2,
        "move_kappa_factor": 80.0,
    }
    for key, expected in expected_smc.items():
        if float(smc[key]) != float(expected):
            raise ValueError(f"unexpected SMC setting {key}={smc[key]!r}")


def seed_for(
    config: dict[str, Any],
    *,
    dimension_index: int,
    dataset_id: int,
    reference_id: int,
    radius_index: int,
    particle_ladder_index: int,
    split_id: int,
) -> int:
    """Perfect mixed-radix packing over the declared finite key domain."""
    if not 0 <= dimension_index < 4:
        raise ValueError("dimension_index out of range")
    if not 0 <= dataset_id < 10 or not 0 <= reference_id < 10:
        raise ValueError("dataset/reference ID out of range")
    if not 0 <= radius_index < 42:
        raise ValueError("radius_index out of range")
    if not 0 <= particle_ladder_index < 5:
        raise ValueError("particle_ladder_index out of range")
    if split_id not in (0, 1):
        raise ValueError("split_id must be 0 or 1")
    packed = dimension_index
    packed = packed * 10 + dataset_id
    packed = packed * 10 + reference_id
    packed = packed * 42 + radius_index
    packed = packed * 5 + particle_ladder_index
    packed = packed * 2 + split_id
    return int(config["seeding"]["base_seed"]) + packed


def build_rows(config: dict[str, Any], config_sha256: str) -> list[dict[str, Any]]:
    validate_config(config)
    problem = config["problem"]
    design = config["pool_design"]
    dimensions = tuple(int(value) for value in problem["dimensions"])
    radii = radii_from_config(config)
    ladder = tuple(int(value) for value in design["convergence_total_particles"])
    maximum = int(design["thermodynamic_total_particles"])
    convergence_dimension = int(design["convergence_dimension"])
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, int, int, int]] = set()
    for dimension_index, dimension in enumerate(dimensions):
        particle_counts = ladder if dimension == convergence_dimension else (maximum,)
        for dataset_id in range(int(problem["datasets_per_dimension"])):
            for reference_id in range(int(problem["references_per_dataset"])):
                for radius_index, radius in enumerate(radii):
                    for total_particles in particle_counts:
                        key = (
                            dimension,
                            dataset_id,
                            reference_id,
                            radius_index,
                            total_particles,
                        )
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        particle_ladder_index = ladder.index(total_particles)
                        split_seeds = [
                            seed_for(
                                config,
                                dimension_index=dimension_index,
                                dataset_id=dataset_id,
                                reference_id=reference_id,
                                radius_index=radius_index,
                                particle_ladder_index=particle_ladder_index,
                                split_id=split_id,
                            )
                            for split_id in range(2)
                        ]
                        rows.append(
                            {
                                "unit_id": -1,
                                "N": dimension,
                                "M": int(round(float(problem["alpha"]) * dimension)),
                                "dimension_index": dimension_index,
                                "dataset_id": dataset_id,
                                "reference_id": reference_id,
                                "radius_index": radius_index,
                                "radius": radius,
                                "total_particles": total_particles,
                                "particles_per_split": total_particles // 2,
                                "particle_ladder_index": particle_ladder_index,
                                "split_seeds": split_seeds,
                                "retain_full_payload": False,
                                "dataset_path": (
                                    "01_theory/02_theory_sampling/raw_outputs/dataset_pool/"
                                    f"N_{dimension}/dataset_{dataset_id + 1:03d}/dataset.npz"
                                ),
                                "reference_path": (
                                    "01_theory/02_theory_sampling/raw_outputs/reference_pool/"
                                    f"N_{dimension}/dataset_{dataset_id + 1:03d}/"
                                    f"ref_{reference_id + 1:03d}/reference.npz"
                                ),
                                "config_sha256": config_sha256,
                            }
                        )
    rows.sort(
        key=lambda row: (
            row["N"],
            row["dataset_id"],
            row["reference_id"],
            row["radius_index"],
            row["total_particles"],
        )
    )
    for unit_id, row in enumerate(rows):
        row["unit_id"] = unit_id
    return rows


def validate_rows(config: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    expected = int(config["pool_design"]["expected_unique_units"])
    if len(materialized) != expected:
        raise RuntimeError(f"expected {expected} units, found {len(materialized)}")
    unit_ids = [int(row["unit_id"]) for row in materialized]
    if unit_ids != list(range(expected)):
        raise RuntimeError("unit_id values are not contiguous and sorted")
    keys = [
        (
            int(row["N"]),
            int(row["dataset_id"]),
            int(row["reference_id"]),
            int(row["radius_index"]),
            int(row["total_particles"]),
        )
        for row in materialized
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate sampling unit key")
    seeds = [
        int(seed)
        for row in materialized
        for seed in row["split_seeds"]
    ]
    if len(seeds) != 2 * expected or len(set(seeds)) != len(seeds):
        raise RuntimeError("split seeds are not globally unique")
    return {
        "unit_count": len(materialized),
        "split_seed_count": len(seeds),
        "unique_split_seed_count": len(set(seeds)),
        "thermodynamic_units": sum(
            int(row["total_particles"])
            == int(config["pool_design"]["thermodynamic_total_particles"])
            for row in materialized
        ),
        "convergence_dimension_units": sum(
            int(row["N"]) == int(config["pool_design"]["convergence_dimension"])
            for row in materialized
        ),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_schedule(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(config_path)
    rows = build_rows(config, config_sha)
    summary = validate_rows(config, rows)
    manifest_path = project_path(config["artifacts"]["task_manifest"])
    summary_path = project_path(config["artifacts"]["task_manifest_summary"])
    manifest_bytes = "".join(canonical_json(row) + "\n" for row in rows).encode(
        "utf-8"
    )
    atomic_write(manifest_path, manifest_bytes)
    summary_payload = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": config_sha,
        **summary,
    }
    atomic_write(
        summary_path, (canonical_json(summary_payload) + "\n").encode("utf-8")
    )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build and validate in memory without writing manifests.",
    )
    args = parser.parse_args()
    config_path = project_path(args.config)
    if args.check_only:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = build_rows(config, sha256_file(config_path))
        summary = validate_rows(config, rows)
    else:
        summary = write_schedule(config_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
