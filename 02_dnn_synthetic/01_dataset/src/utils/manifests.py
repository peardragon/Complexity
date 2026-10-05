from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import (
    radii_from_config,
    stage_config_fingerprint,
    stage_source_fingerprint,
)
from .fingerprints import NOT_APPLICABLE_FINGERPRINT, strict_fingerprints
from .io import atomic_json
from .seeds import dataset_seed, seed_proof, shell_split_seed


def beta_slug(beta: float) -> str:
    return f"data_beta_{float(beta):.2f}".replace(".", "p")


def dataset_tag(dataset_id: int) -> str:
    return f"dataset_{int(dataset_id):03d}"


@dataclass(frozen=True)
class DatasetTask:
    task_index: int
    beta_index: int
    beta: float
    dataset_id: int
    seed: int


@dataclass(frozen=True)
class ReferenceTask:
    task_index: int
    beta_index: int
    beta: float
    dataset_id: int


@dataclass(frozen=True)
class ShellTask:
    task_index: int
    beta_index: int
    beta: float
    dataset_id: int
    ref_id: int
    radius_index: int
    radius: float
    shell_pass: str
    pass_radius_position: int
    split_seeds: tuple[int, int]


SHELL_PASSES = ("odd", "even")


def dataset_count(config: Mapping[str, Any]) -> int:
    return len(config["dataset"]["beta_values"]) * int(config["dataset"]["datasets_per_beta"])


def reference_count(config: Mapping[str, Any]) -> int:
    return dataset_count(config) * int(config["reference_search"]["references_per_dataset"])


def shell_count(config: Mapping[str, Any]) -> int:
    return reference_count(config) * len(radii_from_config(config))


def shell_pass_order(config: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(str(value) for value in config["shell"]["execution"]["pass_order"])
    if values != ("odd", "even"):
        raise ValueError("the full-resolution Synthetic pass order must be ('odd', 'even')")
    return values


def shell_radius_indices(config: Mapping[str, Any], shell_pass: str) -> tuple[int, ...]:
    selected = str(shell_pass)
    if selected not in shell_pass_order(config):
        raise ValueError(f"unknown shell pass {selected!r}; expected one of {SHELL_PASSES!r}")
    radius_count = len(radii_from_config(config))
    start = 0 if selected == "odd" else 1
    indices = tuple(range(start, radius_count, 2))
    expected_count = int(config["shell"]["execution"]["radii_per_pass"])
    if len(indices) != expected_count:
        raise ValueError(
            f"shell pass {selected!r} contains {len(indices)} radii, expected {expected_count}"
        )
    return indices


def shell_pass_unit_count(config: Mapping[str, Any], shell_pass: str) -> int:
    return int(config["reference_search"]["references_per_dataset"]) * len(
        shell_radius_indices(config, shell_pass)
    )


def decode_dataset_task(config: dict, task_index: int) -> DatasetTask:
    per_beta = int(config["dataset"]["datasets_per_beta"])
    count = dataset_count(config)
    index = int(task_index)
    if not 0 <= index < count:
        raise IndexError(f"dataset task index {index} outside [0,{count})")
    beta_index, dataset_id = divmod(index, per_beta)
    beta = float(config["dataset"]["beta_values"][beta_index])
    return DatasetTask(index, beta_index, beta, dataset_id, dataset_seed(config, beta_index, dataset_id))


def decode_reference_task(config: dict, task_index: int) -> ReferenceTask:
    dataset_task = decode_dataset_task(config, int(task_index))
    return ReferenceTask(
        dataset_task.task_index,
        dataset_task.beta_index,
        dataset_task.beta,
        dataset_task.dataset_id,
    )


def decode_shell_task(config: dict, task_index: int) -> ShellTask:
    n_radius = len(radii_from_config(config))
    n_ref = int(config["reference_search"]["references_per_dataset"])
    n_dataset = int(config["dataset"]["datasets_per_beta"])
    count = shell_count(config)
    index = int(task_index)
    if not 0 <= index < count:
        raise IndexError(f"shell task index {index} outside [0,{count})")
    prefix, radius_index = divmod(index, n_radius)
    prefix, ref_id = divmod(prefix, n_ref)
    beta_index, dataset_id = divmod(prefix, n_dataset)
    beta = float(config["dataset"]["beta_values"][beta_index])
    radius = float(radii_from_config(config)[radius_index])
    shell_pass = "odd" if radius_index % 2 == 0 else "even"
    if shell_pass not in shell_pass_order(config):
        raise ValueError(f"radius index {radius_index} maps to disabled shell pass {shell_pass!r}")
    seeds = (
        shell_split_seed(config, beta_index, dataset_id, ref_id, radius_index, 0),
        shell_split_seed(config, beta_index, dataset_id, ref_id, radius_index, 1),
    )
    return ShellTask(
        task_index=index,
        beta_index=beta_index,
        beta=beta,
        dataset_id=dataset_id,
        ref_id=ref_id,
        radius_index=radius_index,
        radius=radius,
        shell_pass=shell_pass,
        pass_radius_position=radius_index // 2,
        split_seeds=seeds,
    )


def encode_shell_task(
    config: Mapping[str, Any],
    beta_index: int,
    dataset_id: int,
    ref_id: int,
    radius_index: int,
) -> int:
    n_dataset = int(config["dataset"]["datasets_per_beta"])
    n_ref = int(config["reference_search"]["references_per_dataset"])
    n_radius = len(radii_from_config(config))
    coordinates = (int(beta_index), int(dataset_id), int(ref_id), int(radius_index))
    bounds = (len(config["dataset"]["beta_values"]), n_dataset, n_ref, n_radius)
    if any(value < 0 or value >= bound for value, bound in zip(coordinates, bounds)):
        raise IndexError(f"shell coordinates {coordinates} outside bounds {bounds}")
    return (((int(beta_index) * n_dataset + int(dataset_id)) * n_ref + int(ref_id)) * n_radius) + int(
        radius_index
    )


def iter_dataset_tasks(config: dict, start: int = 0, stop: int | None = None) -> Iterator[DatasetTask]:
    final = dataset_count(config) if stop is None else min(int(stop), dataset_count(config))
    for task_index in range(int(start), final):
        yield decode_dataset_task(config, task_index)


def iter_reference_tasks(config: dict, start: int = 0, stop: int | None = None) -> Iterator[ReferenceTask]:
    final = dataset_count(config) if stop is None else min(int(stop), dataset_count(config))
    for task_index in range(int(start), final):
        yield decode_reference_task(config, task_index)


def iter_shell_tasks(config: dict, start: int = 0, stop: int | None = None) -> Iterator[ShellTask]:
    final = shell_count(config) if stop is None else min(int(stop), shell_count(config))
    for task_index in range(int(start), final):
        yield decode_shell_task(config, task_index)


def iter_shell_pass_tasks(
    config: dict,
    dataset_job_index: int,
    shell_pass: str,
) -> Iterator[ShellTask]:
    dataset_task = decode_dataset_task(config, int(dataset_job_index))
    n_ref = int(config["reference_search"]["references_per_dataset"])
    for ref_id in range(n_ref):
        for radius_index in shell_radius_indices(config, shell_pass):
            task_index = encode_shell_task(
                config,
                dataset_task.beta_index,
                dataset_task.dataset_id,
                ref_id,
                radius_index,
            )
            task = decode_shell_task(config, task_index)
            if task.shell_pass != str(shell_pass):
                raise RuntimeError("shell pass iterator produced a task from the wrong pass")
            yield task


def dataset_path(synthetic_root: Path, task: DatasetTask | ReferenceTask | ShellTask) -> Path:
    return (
        Path(synthetic_root)
        / "01_dataset"
        / "raw_outputs"
        / beta_slug(task.beta)
        / dataset_tag(task.dataset_id)
        / "dataset.npz"
    )


def dataset_metadata_path(synthetic_root: Path, task: DatasetTask | ReferenceTask | ShellTask) -> Path:
    return dataset_path(synthetic_root, task).with_name("dataset_meta.json")


def reference_pack_path(synthetic_root: Path, task: ReferenceTask | ShellTask) -> Path:
    return (
        Path(synthetic_root)
        / "03_reference_search"
        / "raw_outputs"
        / beta_slug(task.beta)
        / dataset_tag(task.dataset_id)
        / "references.npz"
    )


def reference_metadata_path(synthetic_root: Path, task: ReferenceTask | ShellTask) -> Path:
    return reference_pack_path(synthetic_root, task).with_name("references_meta.json")


def manifest_paths(synthetic_root: Path) -> dict[str, Path]:
    return {
        "dataset": Path(synthetic_root)
        / "01_dataset"
        / "raw_outputs"
        / "manifests"
        / "dataset_manifest.json",
        "reference": Path(synthetic_root)
        / "03_reference_search"
        / "raw_outputs"
        / "manifests"
        / "reference_manifest.json",
        "shell": Path(synthetic_root)
        / "04_sampling"
        / "raw_outputs"
        / "manifests"
        / "shell_manifest.json",
    }


def manifest_task_projection(config: dict, stage: str) -> dict[str, Any]:
    if stage == "dataset":
        count = dataset_count(config)
        first = asdict(decode_dataset_task(config, 0))
        last = asdict(decode_dataset_task(config, count - 1))
        return {
            "manifest_stage": "dataset",
            "task_interval_half_open": [0, count],
            "task_count": count,
            "axes": {
                "beta_index": [0, len(config["dataset"]["beta_values"]) - 1],
                "data_beta": list(config["dataset"]["beta_values"]),
                "dataset_id": [0, int(config["dataset"]["datasets_per_beta"]) - 1],
            },
            "index_order_slowest_to_fastest": ["beta_index", "dataset_id"],
            "decoder": "decode_dataset_task",
            "first_task": first,
            "last_task": last,
        }
    if stage == "reference":
        count = dataset_count(config)
        first = asdict(decode_reference_task(config, 0))
        last = asdict(decode_reference_task(config, count - 1))
        return {
            "manifest_stage": "reference",
            "dataset_job_interval_half_open": [0, count],
            "dataset_job_count": count,
            "selected_reference_count": reference_count(config),
            "axes": {
                "beta_index": [0, len(config["dataset"]["beta_values"]) - 1],
                "data_beta": list(config["dataset"]["beta_values"]),
                "dataset_id": [0, int(config["dataset"]["datasets_per_beta"]) - 1],
                "attempt_id": [
                    0,
                    int(config["reference_search"]["max_attempts_per_dataset"]) - 1,
                ],
                "selected_ref_id": [
                    0,
                    int(config["reference_search"]["references_per_dataset"]) - 1,
                ],
            },
            "index_order_slowest_to_fastest": [
                "beta_index",
                "dataset_id",
                "attempt_id",
            ],
            "decoder": "decode_reference_task + reference_seed",
            "first_dataset_job": first,
            "last_dataset_job": last,
        }
    if stage == "shell":
        count = dataset_count(config) * sum(
            shell_pass_unit_count(config, shell_pass)
            for shell_pass in shell_pass_order(config)
        )
        first_task = next(iter_shell_pass_tasks(config, 0, "odd"))
        last_task = list(
            iter_shell_pass_tasks(config, dataset_count(config) - 1, "odd")
        )[-1]
        first = asdict(first_task)
        last = asdict(last_task)
        first["split_seeds"] = list(first["split_seeds"])
        last["split_seeds"] = list(last["split_seeds"])
        pass_rows: list[dict[str, Any]] = []
        for shell_pass in shell_pass_order(config):
            first_pass = next(iter_shell_pass_tasks(config, 0, shell_pass))
            last_pass = list(
                iter_shell_pass_tasks(config, dataset_count(config) - 1, shell_pass)
            )[-1]
            pass_rows.append(
                {
                    "shell_pass": shell_pass,
                    "radius_indices": list(shell_radius_indices(config, shell_pass)),
                    "radii": [
                        float(radii_from_config(config)[index])
                        for index in shell_radius_indices(config, shell_pass)
                    ],
                    "dataset_job_interval_half_open": [0, dataset_count(config)],
                    "units_per_dataset_shard": shell_pass_unit_count(config, shell_pass),
                    "first_task": {
                        **asdict(first_pass),
                        "split_seeds": list(first_pass.split_seeds),
                    },
                    "last_task": {
                        **asdict(last_pass),
                        "split_seeds": list(last_pass.split_seeds),
                    },
                }
            )
        return {
            "manifest_stage": "shell",
            "canonical_task_index_bounds": [
                int(first_task.task_index),
                int(last_task.task_index),
            ],
            "task_count": count,
            "execution_pass_order": list(shell_pass_order(config)),
            "execution_passes": pass_rows,
            "atomic_shard_count": dataset_count(config) * len(shell_pass_order(config)),
            "axes": {
                "beta_index": [0, len(config["dataset"]["beta_values"]) - 1],
                "data_beta": list(config["dataset"]["beta_values"]),
                "dataset_id": [0, int(config["dataset"]["datasets_per_beta"]) - 1],
                "ref_id": [
                    0,
                    int(config["reference_search"]["references_per_dataset"]) - 1,
                ],
                "radius_index": list(shell_radius_indices(config, "odd")),
                "radius": [
                    float(radii_from_config(config)[index])
                    for index in shell_radius_indices(config, "odd")
                ],
                "split_id": [0, 1],
            },
            "index_order_slowest_to_fastest": [
                "beta_index",
                "dataset_id",
                "ref_id",
                "radius_index",
            ],
            "execution_order_slowest_to_fastest": [
                "shell_pass",
                "beta_index",
                "dataset_id",
                "ref_id",
                "pass_radius_position",
            ],
            "decoder": "decode_shell_task",
            "pass_iterator": "iter_shell_pass_tasks",
            "first_task": first,
            "last_task": last,
        }
    raise ValueError(f"unknown manifest stage: {stage}")


def manifest_seed_projection(config: dict, stage: str) -> dict[str, Any]:
    proof = seed_proof(config)
    if stage not in proof["intervals"]:
        raise ValueError(f"unknown manifest seed stage: {stage}")
    coordinate_names = {
        "dataset": ["beta_index", "dataset_id"],
        "reference": ["beta_index", "dataset_id", "attempt_id"],
        "shell": [
            "beta_index",
            "dataset_id",
            "ref_id",
            "radius_index",
            "split_id",
        ],
    }
    return {
        "manifest_stage": stage,
        "scheme": proof["scheme"],
        "global_seed": int(config["seeds"]["global_seed"]),
        "coordinate_order": coordinate_names[stage],
        "injective_interval": dict(proof["intervals"][stage]),
        "all_seeds_positive_uint63": bool(proof["all_seeds_positive_uint63"]),
    }


def manifest_descriptor(config: dict, stage: str) -> dict[str, Any]:
    common = {
        "schema_version": 1,
        "record_type": "cartesian_manifest_descriptor",
        "stage": str(stage),
        "protocol_id": config["protocol_id"],
        "protocol_sha256": stage_config_fingerprint(config, stage),
        "source_sha256": stage_source_fingerprint(stage),
        "stage_config_sha256": stage_config_fingerprint(config, stage),
        "stage_source_sha256": stage_source_fingerprint(stage),
        "seed_scheme": config["seeds"]["scheme"],
        "global_seed": int(config["seeds"]["global_seed"]),
    }
    if stage == "dataset":
        common.update(
            {
                "task_count": dataset_count(config),
                "axes": {
                    "beta": list(config["dataset"]["beta_values"]),
                    "dataset_id": [0, int(config["dataset"]["datasets_per_beta"]) - 1],
                },
                "decoder": "decode_dataset_task",
            }
        )
    elif stage == "reference":
        common.update(
            {
                "task_count": dataset_count(config),
                "selected_reference_count": reference_count(config),
                "attempt_seed_coordinates": ["beta_index", "dataset_id", "attempt_id"],
                "decoder": "decode_reference_task",
            }
        )
    elif stage == "shell":
        pass_projection = [
            {
                "shell_pass": shell_pass,
                "radius_indices": list(shell_radius_indices(config, shell_pass)),
                "radii": [
                    float(radii_from_config(config)[index])
                    for index in shell_radius_indices(config, shell_pass)
                ],
                "units_per_dataset_shard": shell_pass_unit_count(config, shell_pass),
            }
            for shell_pass in shell_pass_order(config)
        ]
        common.update(
            {
                "task_count": dataset_count(config)
                * sum(
                    shell_pass_unit_count(config, shell_pass)
                    for shell_pass in shell_pass_order(config)
                ),
                "atomic_shard_count": dataset_count(config) * len(shell_pass_order(config)),
                "axes": {
                    "beta": list(config["dataset"]["beta_values"]),
                    "dataset_id": [0, int(config["dataset"]["datasets_per_beta"]) - 1],
                    "ref_id": [0, int(config["reference_search"]["references_per_dataset"]) - 1],
                    "radius": [
                        float(radii_from_config(config)[index])
                        for index in shell_radius_indices(config, "odd")
                    ],
                    "split_id": [0, 1],
                },
                "execution_pass_order": list(shell_pass_order(config)),
                "execution_passes": pass_projection,
                "index_order_slowest_to_fastest": [
                    "beta_index",
                    "dataset_id",
                    "ref_id",
                    "radius_index",
                ],
                "execution_order_slowest_to_fastest": [
                    "shell_pass",
                    "beta_index",
                    "dataset_id",
                    "ref_id",
                    "pass_radius_position",
                ],
                "decoder": "decode_shell_task",
                "pass_iterator": "iter_shell_pass_tasks",
            }
        )
    else:
        raise ValueError(f"unknown manifest stage: {stage}")
    task_projection = manifest_task_projection(config, stage)
    seed_projection = manifest_seed_projection(config, stage)
    common["task_projection"] = task_projection
    common["seed_projection"] = seed_projection
    common["fingerprints"] = strict_fingerprints(
        config,
        dataset_fingerprint=NOT_APPLICABLE_FINGERPRINT,
        reference_fingerprint=None,
        task=task_projection,
        seed=seed_projection,
    )
    return common


def write_manifest_descriptors(config: dict, synthetic_root: Path) -> list[Path]:
    outputs: list[Path] = []
    for stage, path in manifest_paths(synthetic_root).items():
        atomic_json(path, manifest_descriptor(config, stage))
        outputs.append(path)
    return outputs
