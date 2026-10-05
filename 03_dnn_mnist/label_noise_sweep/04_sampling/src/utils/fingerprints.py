"""Stage-scoped strict-eight fingerprint records for production artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io_utils import STAGE_ROOT
from .protocol import load_objective_contract

from .provenance import (
    fingerprint_mapping,
    runtime_fingerprint,
    sha256_file,
    validate_fingerprint_record,
)


FINGERPRINT_SCOPE_SCHEMA_VERSION = 2
VALID_SCOPES = ("dataset", "manifest", "reference", "sampling", "aggregation")


def _scope_from_task(task: Mapping[str, Any]) -> str:
    stage = str(task.get("stage", ""))
    if stage.startswith("01_dataset"):
        return "dataset"
    if stage.startswith("phase1_manifest"):
        return "manifest"
    if stage.startswith("03_reference_search"):
        return "reference"
    if stage.startswith("04_sampling"):
        return "sampling"
    if stage.startswith("05_") or stage.startswith("aggregation"):
        return "aggregation"
    raise ValueError(f"cannot infer fingerprint scope from task stage {stage!r}")


def scoped_config_projection(
    config: Mapping[str, Any],
    *,
    scope: str,
) -> dict[str, Any]:
    """Project only settings that can affect one production stage."""

    value = str(scope)
    if value not in VALID_SCOPES:
        raise ValueError(f"unknown fingerprint scope {value!r}")
    common = {
        "fingerprint_scope_schema_version": FINGERPRINT_SCOPE_SCHEMA_VERSION,
        "protocol": config["protocol"],
    }
    paths = config["paths"]
    if value == "dataset":
        return {
            **common,
            "dataset": config["dataset"],
            "paths": {
                "dataset_root": paths["dataset_root"],
                "complexity_root": paths["complexity_root"],
            },
        }
    if value == "reference":
        return {
            **common,
            "model": config["model"],
            "objective": config["objective"],
            "reference_search": config["reference_search"],
            "paths": {
                "dataset_root": paths["dataset_root"],
                "reference_root": paths["reference_root"],
                "manifest_root": paths["manifest_root"],
            },
        }
    if value == "manifest":
        dataset = config["dataset"]
        if config["protocol"] == "label_noise_sweep":
            condition_contract = {
                "etas": list(dataset["etas"]),
            }
        else:
            frozen_path = (
                STAGE_ROOT / str(dataset["frozen_manifest"])
            ).resolve()
            frozen_path.relative_to(STAGE_ROOT)
            condition_contract = {
                "selected_ranks_one_based": list(
                    dataset["selected_ranks_one_based"]
                ),
                "selected_pair_count": int(dataset["selected_pair_count"]),
                "frozen_manifest_sha256": sha256_file(frozen_path),
            }
        return {
            **common,
            "objective": config["objective"],
            "dataset": {
                "replica_index": int(dataset["replica_index"]),
                "condition_contract": condition_contract,
            },
            "reference_search": {
                key: config["reference_search"][key]
                for key in (
                    "references_per_condition",
                    "max_attempts_per_condition",
                    "seed_namespace",
                    "expected_attempt_count",
                    "expected_reference_count",
                )
            },
            "sampling": {
                key: config["sampling"][key]
                for key in (
                    "independent_splits",
                    "radii_start",
                    "radii_stop",
                    "radii_count",
                    "seed_namespace",
                    "expected_unit_count",
                )
            },
            "paths": {
                "manifest_root": paths["manifest_root"],
            },
        }
    if value == "sampling":
        return {
            **common,
            "model": config["model"],
            "objective": config["objective"],
            "reference_search": {
                "references_per_condition": config["reference_search"][
                    "references_per_condition"
                ],
                "acceptance_policy": config["reference_search"][
                    "acceptance_policy"
                ],
            },
            "sampling": config["sampling"],
            "paths": {
                "dataset_root": paths["dataset_root"],
                "reference_root": paths["reference_root"],
                "manifest_root": paths["manifest_root"],
                "sampling_shard_root": paths["sampling_shard_root"],
            },
        }
    return {
        **common,
        "model": config["model"],
        "objective": config["objective"],
        "reference_search": {
            "references_per_condition": config["reference_search"][
                "references_per_condition"
            ],
            "acceptance_policy": config["reference_search"]["acceptance_policy"],
        },
        "sampling": config["sampling"],
        "paths": {
            "dataset_root": paths["dataset_root"],
            "reference_root": paths["reference_root"],
            "manifest_root": paths["manifest_root"],
            "sampling_shard_root": paths["sampling_shard_root"],
            "summary_root": paths["summary_root"],
        },
    }


def scoped_config_fingerprint(
    config: Mapping[str, Any],
    *,
    scope: str,
) -> str:
    return fingerprint_mapping(scoped_config_projection(config, scope=scope))


def source_paths_for_scope(scope: str) -> tuple[Path, ...]:
    """List code/dependency files that can affect a stage's payload."""

    value = str(scope)
    if value not in VALID_SCOPES:
        raise ValueError(f"unknown fingerprint scope {value!r}")
    src = Path(__file__).resolve().parent
    support = Path(__file__).resolve().parent
    config_root = STAGE_ROOT / "config"
    common = (
        src / "fingerprints.py",
        src / "io_utils.py",
        src / "protocol.py",
        support / "provenance.py",
        support / "resources.py",
    )
    by_scope = {
        "dataset": (
            src / "datasets.py",
            *common,
        ),
        "manifest": (
            src / "manifests.py",
            src / "modeling.py",
            *common,
            support / "shards.py",
            config_root / "objective.json",
        ),
        "reference": (
            src / "reference_training.py",
            src / "manifests.py",
            src / "datasets.py",
            src / "modeling.py",
            *common,
            support / "objective.py",
            support / "shards.py",
            config_root / "objective.json",
        ),
        "sampling": (
            src / "shell_smc.py",
            src / "reference_training.py",
            src / "datasets.py",
            src / "modeling.py",
            src / "manifests.py",
            *common,
            support / "objective.py",
            support / "radial.py",
            support / "shards.py",
            config_root / "objective.json",
        ),
        "aggregation": (
            src / "aggregation.py",
            src / "shell_smc.py",
            src / "reference_training.py",
            src / "datasets.py",
            src / "modeling.py",
            src / "manifests.py",
            *common,
            support / "objective.py",
            support / "radial.py",
            support / "shards.py",
            config_root / "objective.json",
        ),
    }
    return tuple(dict.fromkeys(Path(path).resolve() for path in by_scope[value]))


def source_fingerprint(scope: str = "reference") -> str:
    source_paths = source_paths_for_scope(scope)
    hashes = {
        str(path.relative_to(STAGE_ROOT)): sha256_file(path)
        for path in source_paths
        if path.is_file()
    }
    return fingerprint_mapping(
        {
            "fingerprint_scope_schema_version": FINGERPRINT_SCOPE_SCHEMA_VERSION,
            "scope": str(scope),
            "files": hashes,
        }
    )


def not_applicable_fingerprint(concept: str) -> str:
    return fingerprint_mapping(
        {"schema_version": 1, "state": "not_applicable", "concept": str(concept)}
    )


def strict_fingerprints(
    config: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
    reference_fingerprint: str | None,
    task: Mapping[str, Any],
    seed: Mapping[str, Any],
    scope: str | None = None,
) -> dict[str, str]:
    resolved_scope = _scope_from_task(task) if scope is None else str(scope)
    if resolved_scope not in VALID_SCOPES:
        raise ValueError(f"unknown fingerprint scope {resolved_scope!r}")
    record = {
        "objective_fingerprint": (
            not_applicable_fingerprint("dataset_objective")
            if resolved_scope == "dataset"
            else load_objective_contract().fingerprint
        ),
        "resolved_config_fingerprint": scoped_config_fingerprint(
            config, scope=resolved_scope
        ),
        "dataset_fingerprint": str(dataset_fingerprint),
        "reference_fingerprint": (
            str(reference_fingerprint)
            if reference_fingerprint is not None
            else not_applicable_fingerprint("reference")
        ),
        "code_fingerprint": source_fingerprint(resolved_scope),
        "task_fingerprint": fingerprint_mapping(dict(task)),
        "seed_fingerprint": fingerprint_mapping(dict(seed)),
        "environment_fingerprint": runtime_fingerprint(),
    }
    validate_fingerprint_record(record)
    return record
