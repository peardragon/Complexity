from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "01_dataset"
RUN_NAME = "gaussian_random_90_dataset"
RAW_ROOT = STAGE_ROOT / "raw_outputs" / RUN_NAME
DATASET_ROOT = RAW_ROOT / "raw_datasets"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / RUN_NAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _normalize_features(x_raw: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = np.mean(x_raw, axis=0)
    std = np.std(x_raw, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (x_raw - mean[None, :]) / std[None, :], {
        "x_mean": mean.astype(np.float64).tolist(),
        "x_std": std.astype(np.float64).tolist(),
    }


def _balanced_random_labels(rng: np.random.Generator, n_points: int) -> np.ndarray:
    labels = np.ones(int(n_points), dtype=np.int8)
    labels[: n_points // 2] = -1
    rng.shuffle(labels)
    return labels


def dataset_seed(dataset_id: int) -> int:
    return 606000000 + 100000 * int(dataset_id) + 1234


def build_dataset(dataset_id: int, *, n_points: int, input_dim: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    seed = dataset_seed(dataset_id)
    rng = np.random.default_rng(seed)
    x_raw = rng.normal(size=(int(n_points), int(input_dim))).astype(np.float64)
    x_train, norm_stats = _normalize_features(x_raw)
    y = _balanced_random_labels(rng, int(n_points))
    meta = {
        "created_at": _now_iso(),
        "generator": "iid_gaussian_features_balanced_random_labels_v1",
        "input_dim": int(input_dim),
        "label_mode": "balanced_random_independent",
        "n_points": int(n_points),
        "seed": int(seed),
        **norm_stats,
    }
    arrays = {"X_raw": x_raw, "X_train": x_train.astype(np.float64), "y": y}
    return arrays, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DNN synthetic Gaussian random-baseline datasets.")
    parser.add_argument("--datasets", type=int, default=90)
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for dataset_id in range(int(args.datasets)):
        canonical_dataset = f"dataset_{dataset_id + 1:03d}"
        dataset_dir = DATASET_ROOT / canonical_dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / "dataset.npz"
        meta_path = dataset_dir / "dataset_meta.json"
        if (dataset_path.exists() or meta_path.exists()) and not args.overwrite:
            raise FileExistsError(f"{dataset_dir} exists; pass --overwrite to replace generated data")
        arrays, meta = build_dataset(dataset_id, n_points=args.n_points, input_dim=args.input_dim)
        np.savez(dataset_path, **arrays)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seed = dataset_seed(dataset_id)
        rows.append(
            {
                "cell_id": "cell_beta_0p05_p_0p00",
                "series": "random_gaussian_baseline",
                "dataset_id": dataset_id,
                "seed": seed,
                "beta_ising": 0.05,
                "rewire_p": 0.0,
                "dataset_raw_path": f"02_dnn_synthetic/06_random_baseline/01_dataset/raw_outputs/{RUN_NAME}/raw_datasets/{canonical_dataset}/dataset.npz",
                "dataset_meta_path": f"02_dnn_synthetic/06_random_baseline/01_dataset/raw_outputs/{RUN_NAME}/raw_datasets/{canonical_dataset}/dataset_meta.json",
                "canonical_dataset": canonical_dataset,
            }
        )

    with (SUMMARY_ROOT / "dataset_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    run_config = {
        "datasets": int(args.datasets),
        "generator": "iid_gaussian_features_balanced_random_labels_v1",
        "run_name": RUN_NAME,
        "seed_formula": "606000000 + 100000 * dataset_id + 1234",
    }
    (RAW_ROOT / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
