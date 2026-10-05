"""Canonical strict-eight fingerprint records for synthetic artifacts."""

from __future__ import annotations

import sys
from typing import Any, Mapping

from .config import (
    LOCAL_SOURCE_ROOT,
    stage_config_fingerprint,
    stage_source_fingerprint,
)
from .objective import OBJECTIVE_FINGERPRINT
from .runtime import runtime_fingerprint


if str(LOCAL_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE_ROOT))

from utils.provenance import fingerprint_mapping, validate_fingerprint_record


NOT_APPLICABLE_FINGERPRINT = fingerprint_mapping({"state": "not_applicable"})


def strict_fingerprints(
    config: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
    reference_fingerprint: str | None,
    task: Mapping[str, Any],
    seed: Mapping[str, Any],
) -> dict[str, str]:
    """Build and validate the canonical objective/config/data/ref/code/task/seed/env set."""

    raw_stage = task.get("stage", task.get("manifest_stage"))
    if raw_stage is None:
        raise ValueError("strict fingerprint task projection does not identify its stage")
    record = {
        "objective_fingerprint": OBJECTIVE_FINGERPRINT,
        "resolved_config_fingerprint": stage_config_fingerprint(config, str(raw_stage)),
        "dataset_fingerprint": str(dataset_fingerprint),
        "reference_fingerprint": (
            str(reference_fingerprint)
            if reference_fingerprint is not None
            else NOT_APPLICABLE_FINGERPRINT
        ),
        "code_fingerprint": stage_source_fingerprint(str(raw_stage)),
        "task_fingerprint": fingerprint_mapping(dict(task)),
        "seed_fingerprint": fingerprint_mapping(dict(seed)),
        "environment_fingerprint": runtime_fingerprint(),
    }
    validate_fingerprint_record(record)
    return record


def dataset_task_projection(task: Any) -> dict[str, Any]:
    return {
        "stage": "synthetic_dataset",
        "task_index": int(task.task_index),
        "beta_index": int(task.beta_index),
        "data_beta": float(task.beta),
        "dataset_id": int(task.dataset_id),
    }


def dataset_seed_projection(config: Mapping[str, Any], task: Any) -> dict[str, Any]:
    return {
        "stage": "synthetic_dataset",
        "scheme": str(config["seeds"]["scheme"]),
        "global_seed": int(config["seeds"]["global_seed"]),
        "derived_seed": int(task.seed),
    }


def reference_task_projection(task: Any) -> dict[str, Any]:
    return {
        "stage": "synthetic_reference_search",
        "task_index": int(task.task_index),
        "beta_index": int(task.beta_index),
        "data_beta": float(task.beta),
        "dataset_id": int(task.dataset_id),
    }


def reference_seed_projection(config: Mapping[str, Any], task: Any) -> dict[str, Any]:
    from .seeds import reference_seed

    attempt_count = int(config["reference_search"]["max_attempts_per_dataset"])
    return {
        "stage": "synthetic_reference_search",
        "scheme": str(config["seeds"]["scheme"]),
        "global_seed": int(config["seeds"]["global_seed"]),
        "attempt_id_interval": [0, attempt_count - 1],
        "first_derived_seed": reference_seed(
            config,
            int(task.beta_index),
            int(task.dataset_id),
            0,
        ),
        "last_derived_seed": reference_seed(
            config,
            int(task.beta_index),
            int(task.dataset_id),
            attempt_count - 1,
        ),
    }


def shell_shard_task_projection(
    first_task: Any,
    last_task: Any,
    *,
    dataset_job_index: int,
    shell_pass: str,
    radius_indices: tuple[int, ...],
    unit_count: int,
) -> dict[str, Any]:
    return {
        "stage": "synthetic_shell_shard",
        "dataset_job_index": int(dataset_job_index),
        "shell_pass": str(shell_pass),
        "unit_count": int(unit_count),
        "radius_indices": [int(value) for value in radius_indices],
        "beta_index": int(first_task.beta_index),
        "data_beta": float(first_task.beta),
        "dataset_id": int(first_task.dataset_id),
        "canonical_task_index_bounds": [
            int(first_task.task_index),
            int(last_task.task_index),
        ],
        "first_coordinates": {
            "ref_id": int(first_task.ref_id),
            "radius_index": int(first_task.radius_index),
            "radius": float(first_task.radius),
        },
        "last_coordinates": {
            "ref_id": int(last_task.ref_id),
            "radius_index": int(last_task.radius_index),
            "radius": float(last_task.radius),
        },
    }


def shell_shard_seed_projection(
    config: Mapping[str, Any],
    first_task: Any,
    last_task: Any,
    *,
    dataset_job_index: int,
    shell_pass: str,
    radius_indices: tuple[int, ...],
    unit_count: int,
) -> dict[str, Any]:
    return {
        "stage": "synthetic_shell_shard",
        "scheme": str(config["seeds"]["scheme"]),
        "global_seed": int(config["seeds"]["global_seed"]),
        "dataset_job_index": int(dataset_job_index),
        "shell_pass": str(shell_pass),
        "radius_indices": [int(value) for value in radius_indices],
        "first_split_seeds": [int(value) for value in first_task.split_seeds],
        "last_split_seeds": [int(value) for value in last_task.split_seeds],
        "derived_seed_count": 2 * int(unit_count),
    }
