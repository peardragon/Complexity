from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import (
    SYNTHETIC_ROOT,
    stage_config_fingerprint,
    stage_source_fingerprint,
)
from .fingerprints import dataset_seed_projection, dataset_task_projection, strict_fingerprints
from .io import atomic_json, atomic_npz, sha256_array
from .manifests import DatasetTask, dataset_metadata_path, dataset_path, iter_dataset_tasks
from .runtime import runtime_fingerprint, runtime_signature


def mutual_knn_graph(x: np.ndarray, k: int) -> list[tuple[int, int]]:
    points = np.asarray(x, dtype=np.float64)
    count = int(points.shape[0])
    if not 1 <= int(k) < count:
        raise ValueError("k_graph must satisfy 1 <= k_graph < n_train")
    neighbors = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean", n_jobs=1)
    indices = neighbors.fit(points).kneighbors(points, return_distance=False)[:, 1:]
    neighbor_sets = [set(map(int, row)) for row in indices]
    edges: set[tuple[int, int]] = set()
    for left in range(count):
        for right_raw in indices[left]:
            right = int(right_raw)
            if left in neighbor_sets[right]:
                edges.add((min(left, right), max(left, right)))
    return sorted(edges)


def _neighbor_arrays(n_nodes: int, edges: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    adjacency: list[list[int]] = [[] for _ in range(int(n_nodes))]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return [np.asarray(row, dtype=np.int32) for row in adjacency]


def kawasaki_labels(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    data_beta: float,
    sweeps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_nodes = int(n_nodes)
    if n_nodes % 2:
        raise ValueError("balanced Kawasaki labels require even n_train")
    labels = np.ones(n_nodes, dtype=np.int8)
    labels[n_nodes // 2 :] = -1
    rng.shuffle(labels)
    neighbors = _neighbor_arrays(n_nodes, edges)
    edge_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
    for left, right in edges:
        edge_mask[int(left), int(right)] = True
        edge_mask[int(right), int(left)] = True
    field = np.asarray([np.sum(labels[row]) for row in neighbors], dtype=np.float64)
    for _sweep in range(int(sweeps)):
        for _proposal in range(n_nodes):
            left = int(rng.integers(0, n_nodes))
            for _ in range(20):
                right = int(rng.integers(0, n_nodes))
                if labels[left] != labels[right]:
                    break
            else:
                continue
            left_value = float(labels[left])
            right_value = float(labels[right])
            connected = bool(edge_mask[left, right])
            left_field = field[left] - (right_value if connected else 0.0)
            right_field = field[right] - (left_value if connected else 0.0)
            delta_energy = 2.0 * (left_value * left_field + right_value * right_field)
            if delta_energy <= 0.0 or rng.uniform() < math.exp(-float(data_beta) * delta_energy):
                field[neighbors[left]] -= 2.0 * left_value
                field[neighbors[right]] -= 2.0 * right_value
                labels[left] = -labels[left]
                labels[right] = -labels[right]
    return labels


def generate_arrays(config: dict, task: DatasetTask) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    spec = config["dataset"]
    rng = np.random.default_rng(int(task.seed))
    x_raw = rng.uniform(-1.0, 1.0, size=(int(spec["n_train"]), int(spec["input_dim"]))).astype(np.float64)
    edges = mutual_knn_graph(x_raw, int(spec["k_graph"]))
    labels = kawasaki_labels(
        int(spec["n_train"]),
        edges,
        float(task.beta),
        int(spec["ising_sweeps"]),
        rng,
    )
    mean = np.mean(x_raw, axis=0)
    std = np.std(x_raw, axis=0)
    std = np.where(std > 1.0e-12, std, 1.0)
    x_train = ((x_raw - mean) / std).astype(np.float64)
    values, counts = np.unique(labels, return_counts=True)
    class_counts = {str(int(value)): int(count) for value, count in zip(values, counts)}
    if class_counts != {"-1": 256, "1": 256}:
        raise RuntimeError(f"generator violated exact balance: {class_counts}")
    metadata = {
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "edge_count": len(edges),
        "class_counts": class_counts,
    }
    return x_raw, x_train, labels, metadata


def _expected_metadata(config: dict, task: DatasetTask, payload_hash: str) -> dict:
    fingerprints = strict_fingerprints(
        config,
        dataset_fingerprint=payload_hash,
        reference_fingerprint=None,
        task=dataset_task_projection(task),
        seed=dataset_seed_projection(config, task),
    )
    return {
        "schema_version": 1,
        "record_type": "synthetic_dataset",
        "generator_id": config["dataset"]["generator_id"],
        "protocol_id": config["protocol_id"],
        "protocol_sha256": stage_config_fingerprint(config, "dataset"),
        "source_sha256": stage_source_fingerprint("dataset"),
        "dataset_stage_config_sha256": stage_config_fingerprint(config, "dataset"),
        "dataset_stage_source_sha256": stage_source_fingerprint("dataset"),
        "runtime_fingerprint": runtime_fingerprint(),
        "runtime_signature": runtime_signature(),
        "beta_index": int(task.beta_index),
        "data_beta": float(task.beta),
        "dataset_id": int(task.dataset_id),
        "seed": int(task.seed),
        "seed_scheme": config["seeds"]["scheme"],
        "payload_sha256": payload_hash,
        "fingerprints": fingerprints,
    }


def generate_dataset(config: dict, task: DatasetTask, *, synthetic_root: Path = SYNTHETIC_ROOT) -> str:
    output_path = dataset_path(synthetic_root, task)
    metadata_path = dataset_metadata_path(synthetic_root, task)
    if output_path.is_file():
        return "skipped_existing"
    x_raw, x_train, labels, details = generate_arrays(config, task)
    payload_hash = sha256_array(x_raw, x_train, labels)
    expected = _expected_metadata(config, task, payload_hash)
    atomic_npz(output_path, X_raw=x_raw, X_train=x_train, y=labels)
    atomic_json(metadata_path, {**expected, **details})
    return "written"


def generate_range(
    config: dict,
    *,
    start: int,
    stop: int | None,
    synthetic_root: Path = SYNTHETIC_ROOT,
) -> dict[str, int]:
    counts = {"written": 0, "skipped_existing": 0}
    for task in iter_dataset_tasks(config, start=start, stop=stop):
        status = generate_dataset(config, task, synthetic_root=synthetic_root)
        counts[status] += 1
    return counts
