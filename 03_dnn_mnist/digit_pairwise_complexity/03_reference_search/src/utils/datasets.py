"""Fresh, deterministic MNIST generation for the two production protocols.

The data contract deliberately preserves the historical production
preprocessing (Pillow BOX downscale followed by train-feature
standardization).  Only the protocol axis is revised: one clean even/odd base
with a common nested noise stream, and one deterministic all-45 pair ranking.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .io_utils import (
    MNIST_ROOT,
    STAGE_ROOT,
    atomic_write_json,
    atomic_write_npz,
    file_sha256,
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
    """Fetch the approved OpenML payload only when the dataset command is run."""

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:  # pragma: no cover - runtime dependency guard.
        raise RuntimeError("scikit-learn is required for approved OpenML generation") from exc
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


def pair_condition_name(digit_a: int, digit_b: int) -> str:
    return f"pair_{int(digit_a)}_{int(digit_b)}"


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


def pair_seed(split_seed: int, digit_a: int, digit_b: int) -> int:
    return int(split_seed) + 100 * int(digit_a) + int(digit_b)


def _balanced_pair_split(
    digits: np.ndarray,
    *,
    digit_a: int,
    digit_b: int,
    n_train: int,
    n_test: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if n_train % 2 or n_test % 2:
        raise ValueError("pair split sizes must be even")
    per_train = n_train // 2
    per_test = n_test // 2
    required = per_train + per_test
    seed = pair_seed(split_seed, digit_a, digit_b)
    rng = np.random.default_rng(seed)
    values = np.asarray(digits, dtype=np.int16).reshape(-1)
    indices_a = rng.permutation(np.flatnonzero(values == int(digit_a)))
    indices_b = rng.permutation(np.flatnonzero(values == int(digit_b)))
    if indices_a.size < required or indices_b.size < required:
        raise ValueError(f"insufficient MNIST rows for pair {digit_a}/{digit_b}")
    train = np.concatenate([indices_a[:per_train], indices_b[:per_train]])
    test = np.concatenate(
        [indices_a[per_train:required], indices_b[per_train:required]]
    )
    rng.shuffle(train)
    rng.shuffle(test)
    return train.astype(np.int64), test.astype(np.int64), int(seed)


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
    if config["protocol"] == "label_noise_sweep":
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
    elif config["protocol"] == "digit_pairwise_sweep":
        task = {
            "stage": "01_dataset",
            "protocol": "digit_pairwise_sweep",
            "condition": str(metadata["pair_id"]),
            "rank": int(metadata["rank"]),
        }
        seed = {
            "split_seed": int(metadata["split_seed"]),
            "pair_seed": int(metadata["pair_seed"]),
            "pair_seed_policy": str(metadata["pair_seed_policy"]),
        }
    else:
        raise ValueError(f"unknown protocol {config['protocol']!r}")
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


def chance_label_disagreement(labels: np.ndarray) -> float:
    """Finite-sample chance disagreement for the observed class counts."""

    values = np.asarray(labels).reshape(-1)
    _, counts = np.unique(values, return_counts=True)
    n = int(values.size)
    if n < 2:
        raise ValueError("at least two labels are required")
    chance = float(
        (n * n - int(np.sum(counts.astype(np.int64) ** 2)))
        / (n * (n - 1))
    )
    if not math.isfinite(chance) or chance <= 0.0:
        raise ValueError("degenerate chance-label disagreement")
    return chance


def multiscale_knn_dataset_complexity(
    x: np.ndarray,
    y: np.ndarray,
    *,
    neighbor_audit_max: int = 64,
    selected_scales: int = 22,
) -> float:
    """Chance-normalized cumulative kNN disagreement averaged over scales."""

    features = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("x/y shape mismatch for kNN complexity")
    k_max = int(neighbor_audit_max)
    k_selected = int(selected_scales)
    if not 1 <= k_selected <= k_max < labels.size:
        raise ValueError("invalid multiscale kNN range")
    indices = (
        NearestNeighbors(n_neighbors=k_max + 1, n_jobs=1)
        .fit(features)
        .kneighbors(features, return_distance=False)
    )
    if not np.array_equal(
        indices[:, 0],
        np.arange(labels.size, dtype=indices.dtype),
    ):
        raise ValueError("self is not the first nearest neighbor")
    different = labels[:, None] != labels[indices[:, 1:]]
    exact_rank_disagreement = np.mean(different, axis=0)
    cumulative = np.cumsum(exact_rank_disagreement) / np.arange(
        1,
        k_max + 1,
        dtype=np.float64,
    )
    return float(
        np.mean(
            (cumulative / chance_label_disagreement(labels))[:k_selected]
        )
    )


def resolve_pair_ranking(
    complexity_rows: list[dict[str, Any]],
    frozen: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        (dict(row) for row in complexity_rows),
        key=lambda row: (
            -float(row["complexity"]),
            int(row["digit_a"]),
            int(row["digit_b"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    by_pair = {str(row["pair_id"]): row for row in ranked}
    selected: list[dict[str, Any]] = []
    for expected in frozen["expected_selected_pairs"]:
        pair_id = str(expected["pair_id"])
        if pair_id not in by_pair:
            raise RuntimeError(f"frozen pair is absent: {pair_id}")
        row = dict(by_pair[pair_id])
        row["realized_rank"] = int(row["rank"])
        row["rank"] = int(expected["rank"])
        row["selection_policy"] = (
            "frozen_pair_identity_across_independent_dataset_realizations"
        )
        selected.append(row)
    return ranked, selected


def _build_pair_payload(
    images: np.ndarray,
    digits: np.ndarray,
    dataset_cfg: Mapping[str, Any],
    *,
    digit_a: int,
    digit_b: int,
) -> dict[str, Any]:
    train_indices, test_indices, resolved_seed = _balanced_pair_split(
        digits,
        digit_a=digit_a,
        digit_b=digit_b,
        n_train=int(dataset_cfg["n_train"]),
        n_test=int(dataset_cfg["n_test"]),
        split_seed=int(dataset_cfg["split_seed"]),
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
    return {
        "x_train": x_train,
        "y_train": np.where(digit_train == digit_a, 1, -1).astype(np.int8),
        "x_test": x_test,
        "y_test": np.where(digit_test == digit_a, 1, -1).astype(np.int8),
        "x_train_raw10": x_train_raw10,
        "x_test_raw10": x_test_raw10,
        "digit_train": digit_train,
        "digit_test": digit_test,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "standardization_mean": mean,
        "standardization_std": std,
        "pair_seed": resolved_seed,
    }


def generate_digit_pairs(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    output_root = resolve_project_path(config["paths"]["dataset_root"])
    conditions_to_generate = [row["pair_id"] for row in load_json(STAGE_ROOT / dataset_cfg["frozen_manifest"])["expected_selected_pairs"]]
    if all((output_root / name / "dataset.npz").is_file() for name in conditions_to_generate):
        return {"status": "skipped_existing", "conditions": conditions_to_generate}
    images, digits = fetch_openml_mnist(dataset_cfg)
    split_seed = int(dataset_cfg["split_seed"])
    source_hashes = {
        "openml_images_sha256": sha256_array(images),
        "openml_labels_sha256": sha256_array(digits),
    }
    rows: list[dict[str, Any]] = []

    for digit_a, digit_b in itertools.combinations(range(10), 2):
        payload = _build_pair_payload(
            images,
            digits,
            dataset_cfg,
            digit_a=digit_a,
            digit_b=digit_b,
        )
        condition = pair_condition_name(digit_a, digit_b)
        complexity = multiscale_knn_dataset_complexity(
            payload["x_train"],
            payload["y_train"],
            neighbor_audit_max=int(
                dataset_cfg["complexity"]["neighbor_audit_max"]
            ),
            selected_scales=int(
                dataset_cfg["complexity"]["selected_scales"]
            ),
        )
        row = {
            "pair_id": condition,
            "digit_a": digit_a,
            "digit_b": digit_b,
            "complexity": complexity,
        }
        rows.append(row)

    frozen_path = (STAGE_ROOT / dataset_cfg["frozen_manifest"]).resolve()
    frozen = load_json(frozen_path)
    ranked, selected = resolve_pair_ranking(rows, frozen)
    conditions: list[dict[str, Any]] = []
    for row in selected:
        key = (int(row["digit_a"]), int(row["digit_b"]))
        payload = _build_pair_payload(
            images,
            digits,
            dataset_cfg,
            digit_a=key[0],
            digit_b=key[1],
        )
        metadata = {
            **row,
            "selected_for_production": True,
            "split_seed": split_seed,
            "pair_seed": int(payload["pair_seed"]),
            "pair_seed_policy": str(dataset_cfg["pair_seed_policy"]),
            "downscale": str(dataset_cfg["downscale"]),
            "standardization": str(dataset_cfg["standardization"]),
            "class_balance_train": [256, 256],
            "class_balance_test": [1024, 1024],
            "source_hashes": source_hashes,
        }
        dataset_hash = _dataset_hash_payload(
            payload["x_train"],
            payload["y_train"],
            payload["x_test"],
            payload["y_test"],
            metadata,
        )
        metadata["dataset_hash"] = dataset_hash
        row["dataset_hash"] = dataset_hash
        from .fingerprints import strict_fingerprints

        task, seed = _dataset_task_and_seed(config, metadata)
        metadata["fingerprints"] = strict_fingerprints(
            config,
            dataset_fingerprint=dataset_hash,
            reference_fingerprint=None,
            task=task,
            seed=seed,
        )
        path = output_root / str(row["pair_id"]) / "dataset.npz"
        if not path.is_file():
            atomic_write_npz(
                path,
                **payload,
                rank=np.asarray(int(row["rank"]), dtype=np.int64),
                complexity=np.asarray(float(row["complexity"]), dtype=np.float64),
                dataset_hash=np.asarray(dataset_hash),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        conditions.append(
            {**metadata, "dataset_path": str(path.relative_to(MNIST_ROOT))}
        )
    conditions.sort(key=lambda row: int(row["rank"]))
    manifest = {
        "schema_version": "complexity-revised.mnist.pair-dataset.v1",
        "all_pair_count": len(ranked),
        "selected_pair_count": len(conditions),
        "ranking": ranked,
        "selected_conditions": conditions,
        "frozen_manifest_path": str(frozen_path.relative_to(MNIST_ROOT)),
        "frozen_manifest_sha256": file_sha256(frozen_path),
        "frozen_match": True,
        "source_hashes": source_hashes,
        "full_payload_retention": "selected_12_only",
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
            "protocol": "digit_pairwise_sweep",
            "selected_conditions": [row["pair_id"] for row in conditions],
            "ranking": ranked,
        },
        seed={
            "split_seed": split_seed,
            "pair_seed_policy": str(dataset_cfg["pair_seed_policy"]),
        },
    )
    atomic_write_json(output_root / "pair_manifest.json", manifest)
    atomic_write_json(
        resolve_project_path(config["paths"]["complexity_root"])
        / "digit_pairwise_complexity_summary.json",
        {"ranking": ranked, "selected_conditions": conditions},
    )
    return manifest


def load_condition_dataset(
    config: Mapping[str, Any],
    condition: str,
) -> dict[str, np.ndarray]:
    root = resolve_project_path(config["paths"]["dataset_root"])
    if config["protocol"] == "label_noise_sweep":
        path = root / condition / "dataset.npz"
    elif config["protocol"] == "digit_pairwise_sweep":
        path = root / condition / "dataset.npz"
    else:
        raise ValueError(f"unknown protocol {config['protocol']!r}")
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
    if config["protocol"] == "label_noise_sweep":
        manifest_path = root / "dataset_manifest.json"
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            row = next(
                (
                    value
                    for value in manifest["conditions"]
                    if value["condition"] == condition
                ),
                None,
            )
            if row is None or str(row["dataset_hash"]) != stored_hash:
                raise ValueError(f"{path}: label manifest hash mismatch")
        elif str(metadata.get("condition")) != condition:
            raise ValueError(f"{path}: promoted label condition mismatch")
    else:
        manifest_path = root / "pair_manifest.json"
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            frozen_path = (
                STAGE_ROOT / config["dataset"]["frozen_manifest"]
            ).resolve()
            frozen = load_json(frozen_path)
            selected = {
                str(row["pair_id"]): row
                for row in manifest["selected_conditions"]
            }
            expected_selected = {
                str(row["pair_id"]): row
                for row in frozen["expected_selected_pairs"]
            }
            if set(selected) != set(expected_selected) or any(
                (
                    int(selected[pair_id]["rank"]),
                    int(selected[pair_id]["digit_a"]),
                    int(selected[pair_id]["digit_b"]),
                )
                != (
                    int(expected_selected[pair_id]["rank"]),
                    int(expected_selected[pair_id]["digit_a"]),
                    int(expected_selected[pair_id]["digit_b"]),
                )
                for pair_id in expected_selected
            ):
                raise ValueError("pair manifest disagrees with frozen pair identities/ranks")
            if (
                condition not in selected
                or str(selected[condition]["dataset_hash"]) != stored_hash
            ):
                raise ValueError(f"{path}: pair manifest hash mismatch")
        else:
            if str(metadata.get("pair_id")) != condition:
                raise ValueError(
                    f"{path}: promoted digit-pair condition mismatch"
                )
            frozen_path = (
                STAGE_ROOT / config["dataset"]["frozen_manifest"]
            ).resolve()
            frozen = load_json(frozen_path)
            selected = {
                str(row["pair_id"]): row
                for row in frozen["expected_selected_pairs"]
            }
            if condition not in selected:
                raise ValueError(
                    f"{path}: pair absent from frozen production selection"
                )
        if int(metadata["rank"]) != int(selected[condition]["rank"]):
            raise ValueError(f"{path}: pair rank mismatch")
    return result
