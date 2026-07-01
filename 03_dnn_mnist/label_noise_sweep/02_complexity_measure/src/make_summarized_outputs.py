from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


STAGE_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = STAGE_ROOT.parent
RAW_OUTPUTS_ROOT = SWEEP_ROOT / "01_dataset" / "raw_outputs"
SUMMARY_PATH = STAGE_ROOT / "summarized_outputs" / "eta_complexity_summary.csv"
DATASET_PATTERN = "noise_eta_0p*/dataset.npz"


def _dataset_complexity(dataset_path: Path, k: int = 3) -> float:
    with np.load(dataset_path) as data:
        x = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y_train"]).reshape(-1)

    nn = NearestNeighbors(n_neighbors=int(k) + 1, algorithm="auto")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float(np.mean(y[:, None] != y[indices]))


def _dataset_paths() -> list[Path]:
    paths = sorted(RAW_OUTPUTS_ROOT.glob(DATASET_PATTERN))
    if not paths:
        raise FileNotFoundError(f"no dataset.npz files found under: {RAW_OUTPUTS_ROOT / DATASET_PATTERN}")
    return paths


def _eta_label(eta: float) -> str:
    return f"noise_eta_{eta:g}".replace(".", "p")


def _dataset_eta(dataset_path: Path) -> float:
    if dataset_path.parent.name.startswith("noise_eta_"):
        return float(dataset_path.parent.name.removeprefix("noise_eta_").replace("p", "."))

    with np.load(dataset_path) as data:
        if "eta" in data:
            return float(np.asarray(data["eta"]).reshape(()))

    raise ValueError(f"cannot infer eta for {dataset_path}")


def _dataset_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for dataset_path in _dataset_paths():
        eta = _dataset_eta(dataset_path)
        rows.append(
            {
                "eta": eta,
                "noise_eta": _eta_label(eta),
                "complexity": _dataset_complexity(dataset_path, k=3),
            }
        )
    return rows


def _summary_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[float, list[float]] = {}
    labels: dict[float, str] = {}
    for row in rows:
        eta = float(row["eta"])
        grouped.setdefault(eta, []).append(float(row["complexity"]))
        labels.setdefault(eta, str(row["noise_eta"]))

    summary_rows: list[dict[str, float | int | str]] = []
    for eta, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        summary_rows.append(
            {
                "eta": eta,
                "noise_eta": labels[eta],
                "dataset_count": int(arr.size),
                "complexity_mean": float(np.mean(arr)),
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
