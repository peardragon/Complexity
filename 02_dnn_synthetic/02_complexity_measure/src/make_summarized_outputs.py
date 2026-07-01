from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = PROJECT_ROOT / "02_dnn_synthetic" / "02_complexity_measure"
RAW_OUTPUTS_ROOT = (
    PROJECT_ROOT
    / "02_dnn_synthetic"
    / "01_dataset"
    / "raw_outputs"
)
SUMMARY_PATH = STAGE_ROOT / "summarized_outputs" / "beta_complexity_summary.csv"
DATASET_PATTERN = "beta_0p*/dataset_*/dataset.npz"


def _dataset_complexity(dataset_path: Path, k: int = 3) -> float:
    with np.load(dataset_path) as data:
        x = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y"]).reshape(-1)

    nn = NearestNeighbors(n_neighbors=int(k) + 1, algorithm="auto")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float(np.mean(y[:, None] != y[indices]))


def _dataset_paths() -> list[Path]:
    paths = sorted(RAW_OUTPUTS_ROOT.glob(DATASET_PATTERN))
    if not paths:
        raise FileNotFoundError(f"no dataset.npz files found under: {RAW_OUTPUTS_ROOT / DATASET_PATTERN}")
    return paths


def _dataset_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for dataset_path in _dataset_paths():
        meta_path = dataset_path.with_name("dataset_meta.json")
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        rows.append(
            {
                "beta": float(meta["beta_ising"]),
                "complexity": _dataset_complexity(dataset_path, k=3),
            }
        )
    return rows


def _summary_rows(rows: list[dict[str, float]]) -> list[dict[str, float | int]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["beta"], []).append(row["complexity"])

    summary_rows: list[dict[str, float | int]] = []
    for beta, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        summary_rows.append(
            {
                "beta": beta,
                "dataset_count": int(arr.size),
                "complexity_mean": float(np.mean(arr)),
                "complexity_sd": sd,
                "complexity_se": float(sd / math.sqrt(arr.size)) if arr.size else float("nan"),
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dataset_rows = _dataset_rows()
    summary_rows = _summary_rows(dataset_rows)
    _write_csv(SUMMARY_PATH, summary_rows)


if __name__ == "__main__":
    main()
