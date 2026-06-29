from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "01_dataset"
DATASET_ROOT = STAGE_ROOT / "raw_outputs" / "gaussian_random_90_dataset" / "raw_datasets"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / "gaussian_random_90_dataset"
FIGURE_ROOT = STAGE_ROOT / "figures"


def main() -> None:
    dataset_name = "dataset_001"
    point_summary_path = SUMMARY_ROOT / "example_dataset_points.csv"
    if point_summary_path.exists():
        points = pd.read_csv(point_summary_path)
        x = points[["x0", "x1"]].to_numpy(dtype=np.float64)
        y = points["label"].to_numpy(dtype=np.int8)
    else:
        dataset_path = DATASET_ROOT / dataset_name / "dataset.npz"
        data = np.load(dataset_path)
        x = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.int8)
        SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "dataset": dataset_name,
                "x0": x[:, 0],
                "x1": x[:, 1],
                "label": y.astype(int),
            }
        ).to_csv(point_summary_path, index=False)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    ax.scatter(x[y < 0, 0], x[y < 0, 1], s=18, alpha=0.72, color="#2364aa", label="label -1")
    ax.scatter(x[y > 0, 0], x[y > 0, 1], s=18, alpha=0.72, color="#b42318", label="label +1")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Gaussian random baseline example dataset")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(FIGURE_ROOT / "gaussian_random_dataset_001_example.png", dpi=240)
    plt.close(fig)

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "figure": "gaussian_random_dataset_001_example.png",
            "dataset": dataset_name,
            "dataset_path": "02_dnn_synthetic/06_random_baseline/01_dataset/raw_outputs/gaussian_random_90_dataset/raw_datasets/dataset_001/dataset.npz",
            "n_points": int(x.shape[0]),
            "input_dim": int(x.shape[1]),
            "negative_label_count": int(np.sum(y < 0)),
            "positive_label_count": int(np.sum(y > 0)),
        }
    ]
    with (SUMMARY_ROOT / "example_dataset_preview.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
