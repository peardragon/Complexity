#!/usr/bin/env python3
"""Summarize paper Synthetic C_MS from the raw datasets."""

from __future__ import annotations

import argparse
import csv
import math
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from sklearn.neighbors import NearestNeighbors


STAGE_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = SYNTHETIC_ROOT / "01_dataset" / "raw_outputs"
OUTPUT = STAGE_ROOT / "summarized_outputs" / "cms_by_dataset.csv"
CONFIG_PATH = STAGE_ROOT / "config/default.json"


def configure(config_path=None, output_dir=None, raw_root=None):
    global CONFIG_PATH, BETAS, DATASETS_PER_CONDITION, N_TRAIN, K, DATASET_ROOT, OUTPUT
    CONFIG_PATH = Path(config_path or CONFIG_PATH).resolve()
    config = json.loads(CONFIG_PATH.read_text())
    BETAS = tuple(map(float, config["beta_values"]))
    DATASETS_PER_CONDITION = int(config["datasets_per_condition"])
    N_TRAIN = int(config["n_train"])
    K = int(config["measure"]["neighbor_count"])
    DATASET_ROOT = Path(raw_root) if raw_root else (STAGE_ROOT / config["input_root"]).resolve()
    OUTPUT = (Path(output_dir) / "cms_by_dataset.csv") if output_dir else STAGE_ROOT / config["output"]

configure()
FIELDS = (
    "domain",
    "condition",
    "condition_order",
    "data_beta",
    "dataset_index",
    "C_MS",
)


def condition_slug(data_beta: float) -> str:
    return f"data_beta_{data_beta:.2f}".replace(".", "p")


def dataset_path(data_beta: float, dataset_index: int) -> Path:
    return (
        DATASET_ROOT
        / condition_slug(data_beta)
        / f"dataset_{dataset_index:03d}"
        / "dataset.npz"
    )


def c_ms(path: Path) -> float:
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["X_train"], dtype=np.float64)
        labels = np.asarray(payload["y"]).reshape(-1)
    if points.shape != (N_TRAIN, 2) or labels.shape != (N_TRAIN,):
        raise RuntimeError(f"unexpected Synthetic dataset shape: {path}")
    if not np.all(np.isfinite(points)):
        raise RuntimeError(f"non-finite Synthetic features: {path}")
    indices = (
        NearestNeighbors(n_neighbors=K + 1, n_jobs=1)
        .fit(points)
        .kneighbors(points, return_distance=False)
    )
    if not np.array_equal(indices[:, 0], np.arange(N_TRAIN, dtype=indices.dtype)):
        raise RuntimeError(f"self is not the first nearest neighbor: {path}")
    different = labels[:, None] != labels[indices[:, 1:]]
    rank_disagreement = np.mean(different, axis=0)
    cumulative_disagreement = np.cumsum(rank_disagreement) / np.arange(
        1, K + 1, dtype=np.float64
    )
    _values, counts = np.unique(labels, return_counts=True)
    chance = (
        N_TRAIN * N_TRAIN - int(np.sum(counts.astype(np.int64) ** 2))
    ) / (N_TRAIN * (N_TRAIN - 1))
    if not math.isfinite(chance) or chance <= 0.0:
        raise RuntimeError(f"degenerate chance-disagreement baseline: {path}")
    return float(np.mean(cumulative_disagreement / chance))


def build() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition_order, data_beta in enumerate(BETAS):
        condition = condition_slug(data_beta)
        for dataset_index in range(DATASETS_PER_CONDITION):
            path = dataset_path(data_beta, dataset_index)
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "domain": "synthetic",
                    "condition": condition,
                    "condition_order": condition_order,
                    "data_beta": data_beta,
                    "dataset_index": dataset_index,
                    "C_MS": c_ms(path),
                }
            )
    return rows


def atomic_write(rows: list[dict[str, object]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{OUTPUT.name}.", suffix=".tmp", dir=OUTPUT.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def verify() -> None:
    with OUTPUT.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS: raise ValueError("C_MS columns differ")
        rows = list(reader)
    keys = [(int(r["condition_order"]), int(r["dataset_index"])) for r in rows]
    expected = {(i,j) for i in range(len(BETAS)) for j in range(DATASETS_PER_CONDITION)}
    if len(keys) != len(expected) or set(keys) != expected: raise ValueError("missing/duplicate C_MS coordinates")
    for row in rows:
        beta = BETAS[int(row["condition_order"])]
        if row["domain"] != "synthetic" or row["condition"] != condition_slug(beta) or not math.isclose(float(row["data_beta"]),beta,abs_tol=1e-12): raise ValueError("C_MS condition metadata differs")
        if not math.isfinite(float(row["C_MS"])): raise ValueError("nonfinite C_MS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="recompute from raw datasets")
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs from raw inputs.")
    parser.add_argument("--check-only", action="store_true", help="Check existing compact outputs.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    configure(args.config, args.output_dir, args.raw_root)
    if args.check_only and (args.execute or args.force):
        parser.error("--check-only cannot be combined with --execute or --force")
    if args.force or (args.execute and not (OUTPUT.is_file())):
        atomic_write(build())
    verify()
    print(f"{OUTPUT} rows={len(BETAS) * DATASETS_PER_CONDITION}")


if __name__ == "__main__":
    main()
