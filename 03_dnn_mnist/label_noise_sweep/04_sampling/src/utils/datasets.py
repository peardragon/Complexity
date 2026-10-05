"""Deterministic MNIST even/odd datasets with nested label noise.

Pillow BOX downscaling, train-feature standardization, splits, and the shared
noise stream reproduce the published preprocessing and dataset identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io_utils import (
    MNIST_ROOT,
    atomic_write_json,
    atomic_write_npz,
    load_json,
    payload_sha256,
    resolve_project_path,
)

from .provenance import fingerprint_mapping, sha256_array


def box_downscale_10(images: np.ndarray) -> np.ndarray:
    """Apply Pillow's production BOX resampler, preserving uint8 semantics."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency guard.
        raise RuntimeError("Pillow is required for canonical MNIST preprocessing") from exc
    values = np.asarray(images, dtype=np.uint8)
    if values.ndim == 2 and values.shape[1] == 784:
        values = values.reshape(-1, 28, 28)
    if values.ndim != 3 or values.shape[1:] != (28, 28):
        raise ValueError(f"expected MNIST images shaped (n,28,28), got {values.shape}")
    output = np.empty((values.shape[0], 100), dtype=np.float32)
    for index, image in enumerate(values):
        resized = Image.fromarray(image).resize((10, 10), Image.Resampling.BOX)
        output[index] = np.asarray(resized, dtype=np.float32).reshape(-1)
    return output


def standardize_from_train(
    x_train_raw10: np.ndarray,
    x_test_raw10: np.ndarray,
    *,
    std_floor: float = 1.0e-6,
    floor_replacement: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(x_train_raw10, dtype=np.float32)
    test = np.asarray(x_test_raw10, dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True)
    raw_std = train.std(axis=0, keepdims=True)
    std = np.where(raw_std < float(std_floor), float(floor_replacement), raw_std)
    return (
        ((train - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def fetch_openml_mnist(dataset_cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Fetch MNIST from OpenML only when the dataset command is run."""

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:  # pragma: no cover - runtime dependency guard.
        raise RuntimeError("scikit-learn is required for MNIST generation") from exc
    result = fetch_openml(
        name=str(dataset_cfg["openml_name"]),
        version=int(dataset_cfg["openml_version"]),
        as_frame=False,
        parser="auto",
        data_home=str(resolve_project_path(dataset_cfg["openml_cache_root"])),
    )
    images = np.asarray(result.data, dtype=np.uint8).reshape(-1, 784)
    labels = np.asarray(result.target, dtype=np.int16).reshape(-1)
    if images.shape != (70000, 784) or labels.shape != (70000,):
        raise RuntimeError(
            f"unexpected OpenML MNIST shape: images={images.shape}, labels={labels.shape}"
        )
    return images, labels


def eta_token(eta: float) -> str:
    return f"{float(eta):.2f}".replace(".", "p")


def label_condition_name(eta: float) -> str:
    return f"noise_eta_{eta_token(eta)}"


def _balanced_even_odd_split(
    digits: np.ndarray,
    *,
    n_train: int,
    n_test: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_train % 2 or n_test % 2:
        raise ValueError("even/odd split sizes must be even")
    values = np.asarray(digits, dtype=np.int16).reshape(-1)
    even = np.flatnonzero(values % 2 == 0)
    odd = np.flatnonzero(values % 2 == 1)
    per_train = n_train // 2
    per_test = n_test // 2
    required = per_train + per_test
    if even.size < required or odd.size < required:
        raise ValueError("insufficient examples for balanced even/odd split")
    rng = np.random.default_rng(int(split_seed))
    even = rng.permutation(even)
    odd = rng.permutation(odd)
    train = np.concatenate([even[:per_train], odd[:per_train]])
    test = np.concatenate(
        [even[per_train:required], odd[per_train:required]]
    )
    rng.shuffle(train)
    rng.shuffle(test)
    return train.astype(np.int64), test.astype(np.int64)


def _dataset_hash_payload(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    extra: Mapping[str, Any],
) -> str:
    return payload_sha256(
        {
            "x_train": sha256_array(np.asarray(x_train)),
            "y_train": sha256_array(np.asarray(y_train)),
            "x_test": sha256_array(np.asarray(x_test)),
            "y_test": sha256_array(np.asarray(y_test)),
            "metadata": dict(extra),
        }
    )


def common_noise_masks(
    *,
    n_train: int,
    n_test: int,
    etas: list[float],
    master_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Draw one master stream in the canonical train-then-test order."""

    rng = np.random.default_rng(int(master_seed))
    train_uniform = rng.random(int(n_train))
    test_uniform = rng.random(int(n_test))
    masks: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    previous_train = np.zeros(n_train, dtype=bool)
    previous_test = np.zeros(n_test, dtype=bool)
    for eta in etas:
        train_mask = train_uniform < float(eta)
        test_mask = test_uniform < float(eta)
        if np.any(previous_train & ~train_mask) or np.any(previous_test & ~test_mask):
            raise AssertionError("common-uniform threshold masks are not nested")
        masks[float(eta)] = (train_mask, test_mask)
        previous_train, previous_test = train_mask, test_mask
    return train_uniform, test_uniform, masks


def _dataset_task_and_seed(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config["protocol"] != "label_noise_sweep":
        raise ValueError(f"unsupported label-noise protocol {config['protocol']!r}")
    task = {
        "stage": "01_dataset",
        "protocol": "label_noise_sweep",
        "condition": str(metadata["condition"]),
        "eta": float(metadata["eta"]),
    }
    seed = {
        "split_seed": int(metadata["split_seed"]),
        "flip_uniform_master_seed": int(metadata["flip_uniform_master_seed"]),
        "flip_uniform_draw_order": str(metadata["flip_uniform_draw_order"]),
    }
    return task, seed


def _payload_from_path(path: Path, config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in ("x_train", "y_train", "x_test", "y_test")}
        stored_hash = str(np.asarray(payload["dataset_hash"]).reshape(()).item())
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(()).item()))
    for split, count in (("train", int(config["dataset"]["n_train"])), ("test", int(config["dataset"]["n_test"]))):
        x, y = arrays[f"x_{split}"], arrays[f"y_{split}"]
        if x.shape != (count, 100) or y.size != count or not np.all(np.isfinite(x)):
            raise ValueError(f"invalid dataset dimensions or values: {path}")
        if not set(np.unique(y)).issubset({-1, 1}):
            raise ValueError(f"invalid binary labels: {path}")
    return arrays, metadata, stored_hash


def generate_label_noise(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    output_root = resolve_project_path(config["paths"]["dataset_root"])
    conditions_to_generate = [label_condition_name(float(eta)) for eta in dataset_cfg["etas"]]
    if all((output_root / name / "dataset.npz").is_file() for name in conditions_to_generate):
        return {"status": "skipped_existing", "conditions": conditions_to_generate}
    images, digits = fetch_openml_mnist(dataset_cfg)
    source_hashes = {
        "openml_images_sha256": sha256_array(images),
        "openml_labels_sha256": sha256_array(digits),
    }
    n_train = int(dataset_cfg["n_train"])
    n_test = int(dataset_cfg["n_test"])
    split_seed = int(dataset_cfg["split_seed"])
    train_indices, test_indices = _balanced_even_odd_split(
        digits,
        n_train=n_train,
        n_test=n_test,
        split_seed=split_seed,
    )
    x_train_raw10 = box_downscale_10(images[train_indices])
    x_test_raw10 = box_downscale_10(images[test_indices])
    x_train, x_test, mean, std = standardize_from_train(
        x_train_raw10,
        x_test_raw10,
        std_floor=float(dataset_cfg["standardization_std_floor"]),
        floor_replacement=float(dataset_cfg["standardization_floor_replacement"]),
    )
    digit_train = digits[train_indices].astype(np.int16)
    digit_test = digits[test_indices].astype(np.int16)
    # Historical production orientation: even -> +1, odd -> -1.
    y_train_clean = np.where(digit_train % 2 == 0, 1, -1).astype(np.int8)
    y_test_clean = np.where(digit_test % 2 == 0, 1, -1).astype(np.int8)

    etas = [float(value) for value in dataset_cfg["etas"]]
    train_uniform, test_uniform, masks = common_noise_masks(
        n_train=n_train,
        n_test=n_test,
        etas=etas,
        master_seed=int(dataset_cfg["flip_uniform_master_seed"]),
    )
    base_metadata = {
        "protocol": "label_noise_sweep",
        "base_task": "even_vs_odd",
        "split_seed": split_seed,
        "flip_uniform_master_seed": int(dataset_cfg["flip_uniform_master_seed"]),
        "flip_uniform_draw_order": "train_then_test",
        "downscale": str(dataset_cfg["downscale"]),
        "standardization": str(dataset_cfg["standardization"]),
        "source_hashes": source_hashes,
    }
    conditions: list[dict[str, Any]] = []
    for eta in etas:
        train_mask, test_mask = masks[eta]
        y_train = y_train_clean.copy()
        y_test = y_test_clean.copy()
        y_train[train_mask] *= -1
        y_test[test_mask] *= -1
        condition = label_condition_name(eta)
        metadata = {
            **base_metadata,
            "condition": condition,
            "eta": eta,
            "realized_train_flip_count": int(train_mask.sum()),
            "realized_test_flip_count": int(test_mask.sum()),
            "realized_train_flip_fraction": float(train_mask.mean()),
            "realized_test_flip_fraction": float(test_mask.mean()),
            "mask_definition": "common_uniform < eta",
            "noise_applies_to": "train_and_test",
        }
        dataset_hash = _dataset_hash_payload(
            x_train, y_train, x_test, y_test, metadata
        )
        metadata["dataset_hash"] = dataset_hash
        from .fingerprints import strict_fingerprints

        task, seed = _dataset_task_and_seed(config, metadata)
        metadata["fingerprints"] = strict_fingerprints(
            config,
            dataset_fingerprint=dataset_hash,
            reference_fingerprint=None,
            task=task,
            seed=seed,
        )
        path = output_root / condition / "dataset.npz"
        if not path.is_file():
            atomic_write_npz(
                path,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_train_raw10=x_train_raw10,
                x_test_raw10=x_test_raw10,
                digit_train=digit_train,
                digit_test=digit_test,
                train_indices=train_indices,
                test_indices=test_indices,
                standardization_mean=mean,
                standardization_std=std,
                flip_mask_train=train_mask,
                flip_mask_test=test_mask,
                eta=np.asarray(eta, dtype=np.float64),
                realized_train_flip_fraction=np.asarray(train_mask.mean(), dtype=np.float64),
                realized_test_flip_fraction=np.asarray(test_mask.mean(), dtype=np.float64),
                dataset_hash=np.asarray(dataset_hash),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        conditions.append(
            {**metadata, "dataset_path": str(path.relative_to(MNIST_ROOT))}
        )

    manifest = {
        "schema_version": "complexity-revised.mnist.label-dataset.v1",
        "conditions": conditions,
        "nested_train_masks_verified": True,
        "nested_test_masks_verified": True,
        "eta_zero_clean_verified": bool(
            not masks[0.0][0].any() and not masks[0.0][1].any()
        ),
        "eta_half_direct_bernoulli_threshold": True,
        "shared_preprocessing_across_eta": True,
        "source_hashes": source_hashes,
    }
    from .fingerprints import strict_fingerprints

    manifest["fingerprints"] = strict_fingerprints(
        config,
        dataset_fingerprint=fingerprint_mapping(
            {"condition_hashes": [row["dataset_hash"] for row in conditions]}
        ),
        reference_fingerprint=None,
        task={
            "stage": "01_dataset_manifest",
            "protocol": "label_noise_sweep",
            "conditions": [row["condition"] for row in conditions],
        },
        seed={
            "split_seed": split_seed,
            "flip_uniform_master_seed": int(
                dataset_cfg["flip_uniform_master_seed"]
            ),
        },
    )
    atomic_write_json(output_root / "dataset_manifest.json", manifest)
    return manifest


def load_condition_dataset(
    config: Mapping[str, Any],
    condition: str,
) -> dict[str, np.ndarray]:
    if config["protocol"] != "label_noise_sweep":
        raise ValueError(f"unsupported label-noise protocol {config['protocol']!r}")
    root = resolve_project_path(config["paths"]["dataset_root"])
    path = root / condition / "dataset.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    arrays, metadata, stored_hash = _payload_from_path(path, config)
    result = {**arrays, "dataset_hash": np.asarray(stored_hash)}
    if result["x_train"].shape != (512, 100):
        raise ValueError(f"invalid train shape in {path}: {result['x_train'].shape}")
    if result["x_test"].shape != (2048, 100):
        raise ValueError(f"invalid test shape in {path}: {result['x_test'].shape}")
    if str(result["x_train"].dtype) != "float32":
        raise ValueError(f"production features must be float32, found {result['x_train'].dtype}")
    if not np.all(np.isin(result["y_train"], (-1, 1))):
        raise ValueError(f"training labels are not +/-1 in {path}")
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        row = next(
            (value for value in manifest["conditions"] if value["condition"] == condition),
            None,
        )
        if row is None or str(row["dataset_hash"]) != stored_hash:
            raise ValueError(f"{path}: label manifest hash mismatch")
    elif str(metadata.get("condition")) != condition:
        raise ValueError(f"{path}: label condition mismatch")
    return result
