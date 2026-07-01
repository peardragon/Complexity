from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


STAGE_ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROOT = STAGE_ROOT.parent
RAW_OUTPUTS_ROOT = MANUAL_ROOT / "01_dataset" / "raw_outputs"
RULE_MAPPING_PATH = MANUAL_ROOT / "config" / "rule_mapping.csv"
SUMMARY_PATH = STAGE_ROOT / "summarized_outputs" / "manual_rule_complexity_summary.csv"
DATASET_PATTERN = "rule_*/dataset.npz"


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


def _dataset_paths() -> dict[str, Path]:
    paths = {path.parent.name: path for path in sorted(RAW_OUTPUTS_ROOT.glob(DATASET_PATTERN))}
    if not paths:
        raise FileNotFoundError(f"no dataset.npz files found under: {RAW_OUTPUTS_ROOT / DATASET_PATTERN}")
    return paths


def _rule_rows() -> list[dict[str, int | str]]:
    if not RULE_MAPPING_PATH.exists():
        raise FileNotFoundError(RULE_MAPPING_PATH)

    required_fields = ("rule_id", "rule_name", "label")
    rows: list[dict[str, int | str]] = []
    with RULE_MAPPING_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise ValueError(f"{RULE_MAPPING_PATH} is missing columns: {', '.join(missing)}")
        for order, row in enumerate(reader, start=1):
            rows.append(
                {
                    "rule_id": str(row["rule_id"]),
                    "rule_name": str(row["rule_name"]),
                    "label": str(row["label"]),
                    "rule_order": order,
                }
            )

    if not rows:
        raise ValueError(f"no rows for {RULE_MAPPING_PATH}")
    return rows


def _dataset_rows() -> list[dict[str, float | int | str]]:
    paths = _dataset_paths()
    rows: list[dict[str, float | int | str]] = []
    for rule in _rule_rows():
        rule_id = str(rule["rule_id"])
        dataset_path = paths.get(rule_id)
        if dataset_path is None:
            raise FileNotFoundError(RAW_OUTPUTS_ROOT / rule_id / "dataset.npz")
        rows.append(
            {
                "rule_id": rule_id,
                "rule_name": str(rule["rule_name"]),
                "label": str(rule["label"]),
                "rule_order": int(rule["rule_order"]),
                "complexity": _dataset_complexity(dataset_path, k=3),
            }
        )
    return rows


def _summary_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[float]] = {}
    metadata: dict[str, dict[str, float | int | str]] = {}
    for row in rows:
        rule_id = str(row["rule_id"])
        grouped.setdefault(rule_id, []).append(float(row["complexity"]))
        metadata.setdefault(
            rule_id,
            {
                "rule_id": rule_id,
                "rule_name": str(row["rule_name"]),
                "label": str(row["label"]),
                "rule_order": int(row["rule_order"]),
            },
        )

    summary_rows: list[dict[str, float | int | str]] = []
    for rule_id, values in sorted(grouped.items(), key=lambda item: int(metadata[item[0]]["rule_order"])):
        arr = np.asarray(values, dtype=np.float64)
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        complexity_mean = float(np.mean(arr))
        summary_rows.append(
            {
                **metadata[rule_id],
                "dataset_count": int(arr.size),
                "complexity_mean": complexity_mean,
                "complexity_sd": sd,
                "complexity_se": float(sd / math.sqrt(arr.size)) if arr.size else float("nan"),
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
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
