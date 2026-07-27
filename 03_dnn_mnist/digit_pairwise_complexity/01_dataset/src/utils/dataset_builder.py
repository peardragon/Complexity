from __future__ import annotations

import os
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io_utils import load_json, save_json
from .layout import CONFIG_PATH, PAIR_MAPPING_PATH, RAW_ROOT, dataset_path, pair_dirs, pair_id, pair_label, source_dataset_path


ORIGINAL_MNIST_DIRNAME = "original_mnist"
ORIGINAL_MNIST_FILENAME = "mnist_openml_uint8.npz"
ORIGINAL_MNIST_MANIFEST = "source_manifest.json"
INPUT_SIDE = 10
INPUT_DIM = INPUT_SIDE * INPUT_SIDE
DEFAULT_SPLIT_SEED = 20260610

REQUIRED_KEYS = {
    "X_train",
    "y_train",
    "X_test",
    "y_test",
    "X_train_raw10",
    "X_test_raw10",
    "X_train_raw",
    "X_test_raw",
    "digit_train",
    "digit_test",
    "train_indices",
    "test_indices",
    "standardization_mean",
    "standardization_std",
    "pair_digits",
}

METADATA_FILENAME = "dataset_meta.json"
GENERATION_METADATA_FILENAME = "generation_meta.json"


def write_npz_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH, default=None)
    if not isinstance(config, dict):
        raise FileNotFoundError(CONFIG_PATH)
    return config


def pair_specs() -> list[dict[str, int | str]]:
    values = load_config().get("generation", {}).get("digit_values", list(range(10)))
    digits = [int(value) for value in values]
    specs: list[dict[str, int | str]] = []
    for digit_a, digit_b in combinations(digits, 2):
        specs.append(
            {
                "pair_id": pair_id(digit_a, digit_b),
                "digit_a": int(digit_a),
                "digit_b": int(digit_b),
                "label": pair_label(digit_a, digit_b),
            }
        )
    if not specs:
        raise ValueError(f"{CONFIG_PATH} must define at least two digit_values")
    return specs


def original_mnist_path(raw_root: Path) -> Path:
    return raw_root / ORIGINAL_MNIST_DIRNAME / ORIGINAL_MNIST_FILENAME


def _load_mnist_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path) as payload:
        if "X" not in payload.files or "y" not in payload.files:
            raise KeyError(f"{cache_path} must contain X and y arrays")
        x = np.asarray(payload["X"], dtype=np.uint8).reshape(-1, 784)
        y = np.asarray(payload["y"], dtype=np.int16).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{cache_path} has mismatched X/y rows: {x.shape[0]} vs {y.shape[0]}")
    return x, y


def _manifest(cache_path: Path, *, status: str, download_performed: bool, source_cache: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cache_path": str(cache_path),
        "source": "openml_mnist_784",
        "source_status": status,
        "download_performed": bool(download_performed),
        "payload_keys": ["X", "y"],
        "payload_shape": {"X": [70000, 784], "y": [70000]},
    }
    if source_cache is not None:
        payload["source_cache"] = str(source_cache)
    return payload


def ensure_original_mnist(
    raw_root: Path,
    *,
    source_cache: str | Path | None = None,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = original_mnist_path(raw_root)
    manifest_path = cache_path.parent / ORIGINAL_MNIST_MANIFEST
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        x, y = _load_mnist_cache(cache_path)
        manifest = _manifest(cache_path, status="local_cache", download_performed=False)
        save_json(manifest_path, manifest)
        return x, y, manifest

    if source_cache is not None:
        source_path = Path(source_cache).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        _load_mnist_cache(source_path)
        shutil.copy2(source_path, cache_path)
        x, y = _load_mnist_cache(cache_path)
        manifest = _manifest(cache_path, status="copied_from_source_cache", download_performed=False, source_cache=source_path)
        save_json(manifest_path, manifest)
        return x, y, manifest

    if not download:
        raise FileNotFoundError(f"{cache_path} does not exist. Provide --source-cache or allow OpenML download.")

    try:
        from sklearn.datasets import fetch_openml

        try:
            fetched = fetch_openml(
                "mnist_784",
                version=1,
                as_frame=False,
                parser="auto",
                data_home=str(cache_path.parent / "openml"),
            )
        except TypeError:
            fetched = fetch_openml(
                "mnist_784",
                version=1,
                as_frame=False,
                data_home=str(cache_path.parent / "openml"),
            )
    except Exception as exc:  # pragma: no cover - depends on external network/OpenML.
        raise RuntimeError(f"MNIST data are not available locally and OpenML fetch failed for {cache_path}.") from exc

    x = np.asarray(fetched.data, dtype=np.uint8).reshape(-1, 784)
    y = np.asarray(fetched.target, dtype=np.int16).reshape(-1)
    write_npz_atomic(cache_path, {"X": x, "y": y})
    manifest = _manifest(cache_path, status="downloaded_from_openml", download_performed=True)
    save_json(manifest_path, manifest)
    return x, y, manifest


def box_downscale_10(x784: np.ndarray) -> np.ndarray:
    x = np.asarray(x784, dtype=np.uint8).reshape(-1, 28, 28)
    out = np.empty((x.shape[0], INPUT_DIM), dtype=np.float32)
    for i, image in enumerate(x):
        small = Image.fromarray(image).resize((INPUT_SIDE, INPUT_SIDE), Image.Resampling.BOX)
        out[i] = np.asarray(small, dtype=np.float32).reshape(-1)
    return out


def _pair_seed(split_seed: int, digit_a: int, digit_b: int) -> int:
    return int(split_seed) + 100 * int(digit_a) + int(digit_b)


def _balanced_pair_indices(
    digits: np.ndarray,
    *,
    digit_a: int,
    digit_b: int,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(n_train) % 2 != 0 or int(n_test) % 2 != 0:
        raise ValueError("pairwise splits require even n_train and n_test")

    per_train = int(n_train) // 2
    per_test = int(n_test) // 2
    rng = np.random.default_rng(int(seed))
    a_idx = rng.permutation(np.flatnonzero(np.asarray(digits) == int(digit_a)))
    b_idx = rng.permutation(np.flatnonzero(np.asarray(digits) == int(digit_b)))
    required = per_train + per_test
    if a_idx.size < required:
        raise ValueError(f"digit {digit_a} has only {a_idx.size} examples; need {required}")
    if b_idx.size < required:
        raise ValueError(f"digit {digit_b} has only {b_idx.size} examples; need {required}")

    train_idx = np.concatenate([a_idx[:per_train], b_idx[:per_train]])
    test_idx = np.concatenate([a_idx[per_train:required], b_idx[per_train:required]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def _standardize(x_train_raw: np.ndarray, x_test_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train_raw.mean(axis=0, keepdims=True)
    train_std = x_train_raw.std(axis=0, keepdims=True)
    std = np.where(train_std < 1.0e-6, 1.0, train_std)
    x_train = ((x_train_raw - mean) / std).astype(np.float32)
    x_test = ((x_test_raw - mean) / std).astype(np.float32)
    return x_train, x_test, mean.astype(np.float32), std.astype(np.float32)


def _build_pair_payload(
    *,
    raw28: np.ndarray,
    digits: np.ndarray,
    spec: dict[str, int | str],
    n_train: int,
    n_test: int,
    split_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    digit_a = int(spec["digit_a"])
    digit_b = int(spec["digit_b"])
    seed = _pair_seed(split_seed, digit_a, digit_b)
    train_idx, test_idx = _balanced_pair_indices(
        digits,
        digit_a=digit_a,
        digit_b=digit_b,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    x_train_raw = box_downscale_10(np.asarray(raw28)[train_idx])
    x_test_raw = box_downscale_10(np.asarray(raw28)[test_idx])
    x_train, x_test, mean, std = _standardize(x_train_raw, x_test_raw)
    digit_train = np.asarray(digits, dtype=np.int16)[train_idx]
    digit_test = np.asarray(digits, dtype=np.int16)[test_idx]
    y_train = np.where(digit_train == digit_a, 1, -1).astype(np.int8)
    y_test = np.where(digit_test == digit_a, 1, -1).astype(np.int8)

    payload = {
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
        "X_train_raw10": x_train_raw.astype(np.float32),
        "X_test_raw10": x_test_raw.astype(np.float32),
        "X_train_raw": x_train_raw.astype(np.float32),
        "X_test_raw": x_test_raw.astype(np.float32),
        "digit_train": digit_train.astype(np.int16),
        "digit_test": digit_test.astype(np.int16),
        "train_indices": train_idx,
        "test_indices": test_idx,
        "standardization_mean": mean,
        "standardization_std": std,
        "pair_digits": np.asarray([digit_a, digit_b], dtype=np.int16),
    }
    metadata = {
        "pair_id": str(spec["pair_id"]),
        "digit_a": digit_a,
        "digit_b": digit_b,
        "label": str(spec["label"]),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "split_seed": int(split_seed),
        "pair_seed": int(seed),
        "input_shape": [INPUT_SIDE, INPUT_SIDE],
        "label_policy": f"{digit_a} -> +1, {digit_b} -> -1",
        "train_digit_balance": _balance_dict(digit_train),
        "test_digit_balance": _balance_dict(digit_test),
    }
    return payload, metadata


def _balance_dict(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _validate_payload(path: Path) -> None:
    with np.load(path) as data:
        missing = sorted(REQUIRED_KEYS.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")


def dataset_metadata(spec: dict[str, int | str]) -> dict[str, Any]:
    pair_dir = RAW_ROOT / str(spec["pair_id"])
    payload = dataset_path(pair_dir)
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

        metadata: dict[str, Any] = {
            "metadata_schema": "mnist_digit_pairwise_dataset_v1",
            "pair_id": str(spec["pair_id"]),
            "digit_a": int(spec["digit_a"]),
            "digit_b": int(spec["digit_b"]),
            "label": str(spec["label"]),
            "dataset_path": source_dataset_path(pair_dir),
            "source_dataset_path": source_dataset_path(pair_dir),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "input_shape": [INPUT_SIDE, INPUT_SIDE],
            "train_label_balance": _balance_dict(y_train),
            "test_label_balance": _balance_dict(y_test),
            "train_digit_balance": _balance_dict(digit_train),
            "test_digit_balance": _balance_dict(digit_test),
            "keys": list(data.files),
            "shape_by_key": {key: list(data[key].shape) for key in data.files},
            "dtype_by_key": {key: str(data[key].dtype) for key in data.files},
        }
    generation_metadata = load_json(pair_dir / GENERATION_METADATA_FILENAME, default=None)
    if isinstance(generation_metadata, dict):
        metadata["generation_metadata"] = generation_metadata
    return metadata


def write_pair_mapping(specs: list[dict[str, int | str]]) -> None:
    lines = ["pair_id,digit_a,digit_b,label"]
    for spec in specs:
        lines.append(f"{spec['pair_id']},{spec['digit_a']},{spec['digit_b']},{spec['label']}")
    PAIR_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAIR_MAPPING_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_metadata(spec: dict[str, int | str]) -> dict[str, Any]:
    metadata = dataset_metadata(spec)
    save_json(RAW_ROOT / str(spec["pair_id"]) / METADATA_FILENAME, metadata)
    return metadata


def build_datasets(
    *,
    overwrite: bool = False,
    source_cache: str | None = None,
    download: bool = True,
) -> dict[str, int]:
    config = load_config()
    specs = pair_specs()
    write_pair_mapping(specs)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    expected_pair_ids = {str(spec["pair_id"]) for spec in specs}
    unexpected = [path.name for path in pair_dirs() if path.name not in expected_pair_ids]
    if unexpected:
        raise ValueError(f"Unexpected pair directories: {unexpected}")

    raw28, digits, source_metadata = ensure_original_mnist(RAW_ROOT, source_cache=source_cache, download=download)
    split_seed = int(config.get("source", {}).get("split_seed", DEFAULT_SPLIT_SEED))
    n_train = int(config.get("n_train", 512))
    n_test = int(config.get("n_test", 2048))
    generated = 0

    for spec in specs:
        pair_dir = RAW_ROOT / str(spec["pair_id"])
        payload_path = dataset_path(pair_dir)
        if payload_path.exists() and not overwrite:
            continue
        payload, generation = _build_pair_payload(
            raw28=raw28,
            digits=digits,
            spec=spec,
            n_train=n_train,
            n_test=n_test,
            split_seed=split_seed,
        )
        write_npz_atomic(payload_path, payload)
        save_json(
            pair_dir / GENERATION_METADATA_FILENAME,
            {
                "metadata_schema": "mnist_digit_pairwise_generation_v1",
                "pair_id": str(spec["pair_id"]),
                "source_mnist": source_metadata,
                "pair_generation": generation,
            },
        )
        generated += 1

    missing = 0
    metadata_written = 0
    for spec in specs:
        payload_path = dataset_path(RAW_ROOT / str(spec["pair_id"]))
        if not payload_path.exists():
            missing += 1
            continue
        _validate_payload(payload_path)
        write_dataset_metadata(spec)
        metadata_written += 1

    if missing:
        missing_pair_ids = [str(spec["pair_id"]) for spec in specs if not dataset_path(RAW_ROOT / str(spec["pair_id"])).exists()]
        raise FileNotFoundError(f"Missing dataset payloads: {missing_pair_ids}")
    return {
        "validated": len(specs),
        "metadata_written": metadata_written,
        "missing": missing,
        "generated": generated,
    }


__all__ = [
    "GENERATION_METADATA_FILENAME",
    "METADATA_FILENAME",
    "REQUIRED_KEYS",
    "build_datasets",
    "dataset_metadata",
    "load_config",
    "pair_specs",
    "write_dataset_metadata",
]
