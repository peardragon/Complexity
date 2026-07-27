from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


STAGE_ROOT = Path(__file__).resolve().parents[1]
PAIRWISE_ROOT = STAGE_ROOT.parent
RAW_OUTPUTS_ROOT = PAIRWISE_ROOT / "01_dataset" / "raw_outputs"
SUMMARY_PATH = STAGE_ROOT / "summarized_outputs" / "digit_pairwise_complexity_summary.csv"
DATASET_PATTERN = "pair_*/dataset.npz"


def _dataset_complexity(dataset_path: Path, k: int = 3) -> float:
    with np.load(dataset_path) as data:
        missing = [key for key in ("X_train", "y_train") if key not in data]
        if missing:
            raise ValueError(f"{dataset_path} is missing arrays: {', '.join(missing)}")
        x = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y_train"]).reshape(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X_train/y_train length mismatch in {dataset_path}: {x.shape[0]} != {y.shape[0]}")
    if x.shape[0] <= int(k):
        raise ValueError(f"{dataset_path} has {x.shape[0]} samples, which is not enough for k={k}")

    nn = NearestNeighbors(n_neighbors=int(k) + 1, algorithm="auto")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float(np.mean(y[:, None] != y[indices]))


def _dataset_paths() -> list[Path]:
    paths = sorted(RAW_OUTPUTS_ROOT.glob(DATASET_PATTERN), key=lambda path: _pair_sort_key(path.parent.name))
    if not paths:
        raise FileNotFoundError(f"no dataset.npz files found under: {RAW_OUTPUTS_ROOT / DATASET_PATTERN}")
    return paths


def _pair_sort_key(pair_id: str) -> tuple[int, int]:
    _, a, b = pair_id.split("_")
    return int(a), int(b)


def _read_meta(dataset_path: Path) -> dict[str, Any]:
    meta_path = dataset_path.parent / "dataset_meta.json"
    if meta_path.exists():
        import json

        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    pair_id = dataset_path.parent.name
    _, a, b = pair_id.split("_")
    with np.load(dataset_path) as data:
        return {
            "pair_id": pair_id,
            "digit_a": int(a),
            "digit_b": int(b),
            "label": f"{a}/{b}",
            "dataset_path": str(dataset_path),
            "n_train": int(data["X_train"].shape[0]),
            "n_test": int(data["X_test"].shape[0]) if "X_test" in data else 0,
        }


def _dataset_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for dataset_path in _dataset_paths():
        meta = _read_meta(dataset_path)
        rows.append(
            {
                "pair_id": str(meta["pair_id"]),
                "digit_a": int(meta["digit_a"]),
                "digit_b": int(meta["digit_b"]),
                "label": str(meta["label"]),
                "dataset_path": str(meta.get("dataset_path", dataset_path)),
                "n_train": int(meta.get("n_train", 0)),
                "n_test": int(meta.get("n_test", 0)),
                "complexity": _dataset_complexity(dataset_path, k=3),
            }
        )
    return rows


def _summary_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[float]] = {}
    metadata: dict[str, dict[str, float | int | str]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        grouped.setdefault(pair_id, []).append(float(row["complexity"]))
        metadata.setdefault(
            pair_id,
            {
                "pair_id": pair_id,
                "digit_a": int(row["digit_a"]),
                "digit_b": int(row["digit_b"]),
                "label": str(row["label"]),
                "dataset_path": str(row["dataset_path"]),
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
            },
        )

    unsorted_rows: list[dict[str, float | int | str]] = []
    for pair_id, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        unsorted_rows.append(
            {
                **metadata[pair_id],
                "dataset_count": int(arr.size),
                "complexity_mean": float(np.mean(arr)),
                "complexity_sd": sd,
                "complexity_se": float(sd / math.sqrt(arr.size)) if arr.size else float("nan"),
            }
        )

    ordered = sorted(
        unsorted_rows,
        key=lambda row: (-float(row["complexity_mean"]), int(row["digit_a"]), int(row["digit_b"])),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank_complexity_desc"] = rank
    fields = [
        "rank_complexity_desc",
        "pair_id",
        "digit_a",
        "digit_b",
        "label",
        "dataset_path",
        "n_train",
        "n_test",
        "dataset_count",
        "complexity_mean",
        "complexity_sd",
        "complexity_se",
    ]
    return [{field: row[field] for field in fields} for row in ordered]


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dataset_rows = _dataset_rows()
    summary_rows = _summary_rows(dataset_rows)
    _write_csv(SUMMARY_PATH, summary_rows)
    top = summary_rows[0]
    print(
        "complexity_summary "
        f"rows={len(summary_rows)} "
        f"top_pair={top['label']} "
        f"top_complexity={float(top['complexity_mean']):.9f}"
    )


if __name__ == "__main__":
    main()
