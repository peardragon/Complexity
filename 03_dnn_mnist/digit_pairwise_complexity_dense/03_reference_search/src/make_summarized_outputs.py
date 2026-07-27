from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = STAGE_ROOT / "raw_outputs"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "reference_quality"
REFERENCE_INDEX = RAW_ROOT / "reference_index.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def build() -> dict[str, Path]:
    rows = _read_csv(REFERENCE_INDEX)
    ref_rows: list[dict[str, Any]] = []
    for row in rows:
        ref_rows.append(
            {
                "pair_id": row["pair_id"],
                "pair_label": row["pair_label"],
                "pair_order": int(float(row["pair_order"])),
                "pair_rank_complexity_desc": int(float(row["pair_rank_complexity_desc"])),
                "complexity_mean": _float(row, "complexity_mean"),
                "ref_id": int(float(row["ref_id"])),
                "attempt_seed": int(float(row["attempt_seed"])),
                "train_error": _float(row, "train_error"),
                "test_error": _float(row, "test_error"),
                "CE_mean_train": _float(row, "CE_mean_train"),
                "CE_mean_test": _float(row, "CE_mean_test"),
                "theta_norm": _float(row, "theta_norm"),
                "min_margin": _float(row, "min_margin"),
                "q05_margin": _float(row, "q05_margin"),
                "median_margin": _float(row, "median_margin"),
                "mean_margin": _float(row, "mean_margin"),
                "theta_path": row["theta_path"],
                "dataset_path": row["dataset_path"],
            }
        )
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in ref_rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    pair_rows: list[dict[str, Any]] = []
    for pair_id, group in sorted(by_pair.items(), key=lambda item: int(item[1][0]["pair_order"])):
        pair_rows.append(
            {
                "pair_id": pair_id,
                "pair_label": group[0]["pair_label"],
                "pair_order": group[0]["pair_order"],
                "pair_rank_complexity_desc": group[0]["pair_rank_complexity_desc"],
                "complexity_mean": group[0]["complexity_mean"],
                "ref_count": len(group),
                "train_error_mean": float(np.mean([row["train_error"] for row in group])),
                "test_error_mean": float(np.mean([row["test_error"] for row in group])),
                "test_error_sem": _sem([row["test_error"] for row in group]),
                "CE_mean_train_mean": float(np.mean([row["CE_mean_train"] for row in group])),
                "CE_mean_test_mean": float(np.mean([row["CE_mean_test"] for row in group])),
                "theta_norm_mean": float(np.mean([row["theta_norm"] for row in group])),
                "min_margin_mean": float(np.mean([row["min_margin"] for row in group])),
                "q05_margin_mean": float(np.mean([row["q05_margin"] for row in group])),
                "median_margin_mean": float(np.mean([row["median_margin"] for row in group])),
                "mean_margin_mean": float(np.mean([row["mean_margin"] for row in group])),
            }
        )
    FIGURE_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ref_path = FIGURE_INPUT_ROOT / "reference_quality_by_ref.csv"
    pair_path = FIGURE_INPUT_ROOT / "reference_quality_by_pair.csv"
    _write_csv(ref_path, ref_rows)
    _write_csv(pair_path, pair_rows)
    return {"reference_quality_by_ref": ref_path, "reference_quality_by_pair": pair_path}


def main() -> None:
    for path in build().values():
        print(path)


if __name__ == "__main__":
    main()
