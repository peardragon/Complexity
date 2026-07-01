from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .io_utils import load_json, save_json
from .layout import CONFIG_PATH, DNN_ROOT, RAW_ROOT, dataset_path, rule_dirs, source_dataset_path


if str(DNN_ROOT) not in sys.path:
    sys.path.insert(0, str(DNN_ROOT))

from _shared.mnist10_standalone import (  # noqa: E402
    DEFAULT_SPLIT_SEED,
    build_mnist10_base_payload,
    ensure_original_mnist,
    manual_rule_payload,
    write_json_atomic,
    write_npz_atomic,
)


REQUIRED_KEYS = {
    "X_train",
    "y_train",
    "X_test",
    "y_test",
    "X_train_raw10",
    "X_test_raw10",
    "digit_train",
    "digit_test",
    "train_indices",
    "test_indices",
    "standardization_mean",
    "standardization_std",
}

METADATA_FILENAME = "dataset_meta.json"
GENERATION_METADATA_FILENAME = "generation_meta.json"


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH, default=None)
    if not isinstance(config, dict):
        raise FileNotFoundError(CONFIG_PATH)
    return config


def rule_specs() -> list[dict[str, str]]:
    rules = load_config().get("rules", [])
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"{CONFIG_PATH} must define a non-empty rules list")

    specs: list[dict[str, str]] = []
    for item in rules:
        if not isinstance(item, dict):
            raise TypeError(f"Invalid rule spec: {item!r}")
        specs.append(
            {
                "rule_id": str(item["rule_id"]),
                "rule_name": str(item["rule_name"]),
                "label": str(item["label"]),
            }
        )
    return specs


def _balance_dict(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _validate_payload(path: str) -> None:
    with np.load(path) as data:
        missing = sorted(REQUIRED_KEYS.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")


def dataset_metadata(rule: dict[str, str]) -> dict[str, object]:
    rule_dir = RAW_ROOT / rule["rule_id"]
    payload = dataset_path(rule_dir)
    if not payload.exists():
        raise FileNotFoundError(payload)

    with np.load(payload) as data:
        missing = sorted(REQUIRED_KEYS.difference(data.files))
        if missing:
            raise KeyError(f"{payload} is missing keys: {missing}")

        x_train = data["X_train"]
        x_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        digit_train = data["digit_train"]
        digit_test = data["digit_test"]

        metadata: dict[str, object] = {
            "metadata_schema": "mnist_manual_rule_dataset_v2",
            "rule_id": rule["rule_id"],
            "rule_name": rule["rule_name"],
            "label": rule["label"],
            "dataset_path": source_dataset_path(rule_dir),
            "source_dataset_path": source_dataset_path(rule_dir),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "input_shape": [10, 10],
            "train_label_balance": _balance_dict(y_train),
            "test_label_balance": _balance_dict(y_test),
            "train_digit_balance": _balance_dict(digit_train),
            "test_digit_balance": _balance_dict(digit_test),
            "keys": list(data.files),
            "shape_by_key": {key: list(data[key].shape) for key in data.files},
            "dtype_by_key": {key: str(data[key].dtype) for key in data.files},
        }
        generation_metadata = load_json(rule_dir / GENERATION_METADATA_FILENAME, default=None)
        if isinstance(generation_metadata, dict):
            metadata["generation_metadata"] = generation_metadata
        return metadata


def write_dataset_metadata(rule: dict[str, str]) -> dict[str, object]:
    metadata = dataset_metadata(rule)
    rule_dir = RAW_ROOT / rule["rule_id"]
    save_json(rule_dir / METADATA_FILENAME, metadata)
    return metadata


def _generation_targets(specs: list[dict[str, str]], *, overwrite: bool) -> list[dict[str, str]]:
    if overwrite:
        return specs
    return [rule for rule in specs if not dataset_path(RAW_ROOT / rule["rule_id"]).exists()]


def _generate_rule_payloads(
    targets: list[dict[str, str]],
    *,
    raw28: np.ndarray,
    digits: np.ndarray,
    source_metadata: dict[str, Any],
) -> int:
    if not targets:
        return 0

    config = load_config()
    base_payload, split_metadata = build_mnist10_base_payload(
        raw28,
        digits,
        n_train=int(config.get("n_train", 512)),
        n_test=int(config.get("n_test", 2048)),
        split_seed=int(config.get("source", {}).get("split_seed", DEFAULT_SPLIT_SEED)),
    )

    generated = 0
    for rule in targets:
        rule_payload, rule_generation = manual_rule_payload(rule["rule_name"], base_payload)
        rule_dir = RAW_ROOT / rule["rule_id"]
        write_npz_atomic(dataset_path(rule_dir), rule_payload)
        write_json_atomic(
            rule_dir / GENERATION_METADATA_FILENAME,
            {
                "metadata_schema": "mnist_manual_rule_generation_v1",
                "rule_id": rule["rule_id"],
                "rule_name": rule["rule_name"],
                "source_mnist": source_metadata,
                "split": split_metadata,
                "rule_generation": rule_generation,
            },
        )
        generated += 1
    return generated


def build_datasets(
    *,
    overwrite: bool = False,
    source_cache: str | None = None,
    download: bool = True,
) -> dict[str, int]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    specs = rule_specs()
    expected_rule_ids = {rule["rule_id"] for rule in specs}
    unexpected = [path.name for path in rule_dirs() if path.name not in expected_rule_ids]
    if unexpected:
        raise ValueError(f"Unexpected rule directories: {unexpected}")

    raw28, digits, source_metadata = ensure_original_mnist(RAW_ROOT, source_cache=source_cache, download=download)
    generated = _generate_rule_payloads(
        _generation_targets(specs, overwrite=overwrite),
        raw28=raw28,
        digits=digits,
        source_metadata=source_metadata,
    )

    missing = 0
    metadata_written = 0
    for rule in specs:
        payload = dataset_path(RAW_ROOT / rule["rule_id"])
        if not payload.exists():
            missing += 1
            continue
        _validate_payload(str(payload))
        write_dataset_metadata(rule)
        metadata_written += 1

    if missing:
        missing_rule_ids = [rule["rule_id"] for rule in specs if not dataset_path(RAW_ROOT / rule["rule_id"]).exists()]
        raise FileNotFoundError(f"Missing dataset payloads: {missing_rule_ids}")
    return {"validated": len(specs), "metadata_written": metadata_written, "missing": missing, "generated": generated}


__all__ = [
    "METADATA_FILENAME",
    "GENERATION_METADATA_FILENAME",
    "REQUIRED_KEYS",
    "build_datasets",
    "dataset_metadata",
    "load_config",
    "rule_specs",
    "write_dataset_metadata",
]
