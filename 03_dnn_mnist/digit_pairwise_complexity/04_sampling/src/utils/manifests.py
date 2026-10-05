"""Deterministic attempt and shell-task manifests."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .protocol import condition_names, radius_grid, validate_config

from .provenance import fingerprint_mapping, require_unique, stable_seed


REFERENCE_FILENAME = "reference_attempt_manifest.jsonl.gz"
SAMPLING_FILENAME = "sampling_task_manifest.jsonl.gz"
INDEX_FILENAME = "manifest_index.json"
MANIFEST_SCHEMA_VERSION = "complexity-revised.mnist.manifest.v2"
MANIFEST_INDEX_SCHEMA_VERSION = "complexity-revised.mnist.manifest-index.v2"
MANIFEST_PRODUCER_CONTRACT_ID = (
    "complexity-revised.mnist.reference-search-manifest-producer.v1"
)


def contiguous_sampling_shard(
    rows: list[dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
    units_per_shard: int,
) -> list[dict[str, Any]]:
    """Return one contiguous production shard without changing manifest order."""

    index = int(shard_index)
    count = int(shard_count)
    units = int(units_per_shard)
    if units <= 0 or count <= 0:
        raise ValueError("shard_count and units_per_shard must be positive")
    if not 0 <= index < count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    if len(rows) != count * units:
        raise ValueError(
            f"manifest has {len(rows)} rows, expected exactly "
            f"{count} * {units} = {count * units}"
        )
    start = index * units
    stop = start + units
    selected = rows[start:stop]
    if len(selected) != units:
        raise AssertionError("internal contiguous shard partition error")
    expected_indices = list(range(start, stop))
    actual_indices = [int(row["manifest_index"]) for row in selected]
    if actual_indices != expected_indices:
        raise ValueError("sampling manifest order/index drift")
    return selected


def _unit_id(kind: str, row_key: Mapping[str, Any]) -> str:
    return f"{kind}_{fingerprint_mapping(dict(row_key))[:24]}"


def build_reference_attempt_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    namespace = str(config["reference_search"]["seed_namespace"])
    attempts = int(config["reference_search"]["max_attempts_per_condition"])
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(condition_names(config)):
        for attempt_index in range(attempts):
            key = {
                "protocol": str(config["protocol"]),
                "condition": condition,
                "attempt_index": attempt_index,
            }
            rows.append(
                {
                    "row_type": "reference_attempt",
                    "manifest_index": len(rows),
                    "attempt_id": _unit_id("attempt", key),
                    "condition_index": condition_index,
                    "condition": condition,
                    "attempt_index": attempt_index,
                    "attempt_seed": stable_seed(
                        condition,
                        attempt_index,
                        namespace=namespace,
                        bits=63,
                    ),
                }
            )
    expected = int(config["reference_search"]["expected_attempt_count"])
    if len(rows) != expected:
        raise AssertionError(f"reference attempt count {len(rows)} != {expected}")
    require_unique((row["attempt_id"] for row in rows), label="reference attempt ids")
    require_unique((row["attempt_seed"] for row in rows), label="reference attempt seeds")
    return rows


def build_sampling_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    namespace = str(config["sampling"]["seed_namespace"])
    references = int(config["reference_search"]["references_per_condition"])
    split_count = int(config["sampling"]["independent_splits"])
    radii = radius_grid(config)
    rows: list[dict[str, Any]] = []
    all_seeds: list[int] = []
    for condition_index, condition in enumerate(condition_names(config)):
        for reference_index in range(references):
            for radius_index, radius in enumerate(radii):
                key = {
                    "protocol": str(config["protocol"]),
                    "condition": condition,
                    "reference_index": reference_index,
                    "radius_index": radius_index,
                    "radius": float(radius),
                }
                unit_seed = stable_seed(
                    condition,
                    reference_index,
                    radius_index,
                    "unit",
                    namespace=namespace,
                    bits=63,
                )
                split_seeds = [
                    stable_seed(
                        condition,
                        reference_index,
                        radius_index,
                        "split",
                        split_index,
                        namespace=namespace,
                        bits=63,
                    )
                    for split_index in range(split_count)
                ]
                all_seeds.extend([unit_seed, *split_seeds])
                rows.append(
                    {
                        "row_type": "sampling_task",
                        "manifest_index": len(rows),
                        "unit_id": _unit_id("shell", key),
                        "condition_index": condition_index,
                        "condition": condition,
                        "reference_index": reference_index,
                        "radius_index": radius_index,
                        "radius": float(radius),
                        "unit_seed": unit_seed,
                        "split_seeds": split_seeds,
                        "sentinel": bool(
                            reference_index == 0 and radius_index in (0, 9, 99)
                        ),
                    }
                )
    expected = int(config["sampling"]["expected_unit_count"])
    if len(rows) != expected:
        raise AssertionError(f"sampling unit count {len(rows)} != {expected}")
    require_unique((row["unit_id"] for row in rows), label="sampling unit ids")
    require_unique(all_seeds, label="sampling seeds")
    return rows


def read_sampling_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive task order and seeds directly from the numerical configuration."""
    return build_sampling_rows(config)


def read_reference_attempt_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return build_reference_attempt_rows(config)
