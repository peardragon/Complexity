from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "02_complexity_measure"
DATASET_ROOT = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "01_dataset"
    / "raw_outputs"
    / "18_beta_cell_90_dataset"
    / "raw_datasets"
)
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / "18_beta_cell_90_dataset"
FIGURE_PATH = STAGE_ROOT / "figures" / "beta_complexity_figure.png"


def _dataset_complexity(dataset_path: Path, k: int = 3) -> float:
    with np.load(dataset_path) as data:
        x = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y"]).reshape(-1)
    nn = NearestNeighbors(n_neighbors=int(k) + 1, algorithm="auto")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float(np.mean(y[:, None] != y[indices]))


def _dataset_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_path in sorted(DATASET_ROOT.glob("cell_beta_*_p_0p00/dataset_*/dataset.npz")):
        meta_path = dataset_path.with_name("dataset_meta.json")
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        rows.append(
            {
                "beta": float(meta["beta_ising"]),
                "cell_id": str(meta["cell_id"]),
                "dataset_id": int(meta["dataset_id"]),
                "dataset_tag": dataset_path.parent.name,
                "complexity_3nn_disagreement": _dataset_complexity(dataset_path, k=3),
                "source_dataset": str(dataset_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(float(row["beta"]), []).append(float(row["complexity_3nn_disagreement"]))
    out: list[dict[str, object]] = []
    for beta, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        out.append(
            {
                "beta": beta,
                "dataset_count": int(arr.size),
                "complexity_mean": float(np.mean(arr)),
                "complexity_sd": sd,
                "complexity_se": float(sd / math.sqrt(arr.size)) if arr.size else float("nan"),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary_rows: list[dict[str, object]]) -> None:
    beta = np.asarray([float(row["beta"]) for row in summary_rows], dtype=np.float64)
    mean = np.asarray([float(row["complexity_mean"]) for row in summary_rows], dtype=np.float64)
    se = np.asarray([float(row["complexity_se"]) for row in summary_rows], dtype=np.float64)
    r, p_value = pearsonr(beta, mean)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.errorbar(beta, mean, yerr=se, fmt="o-", color="#284f8f", ecolor="#8aa7d6", capsize=3)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("3-NN label-disagreement complexity")
    ax.set_title(f"Beta vs complexity (Pearson r={r:.3f}, p={p_value:.1e})")
    ax.grid(True, alpha=0.25)
    fig.savefig(FIGURE_PATH, dpi=240)
    plt.close(fig)


def main() -> None:
    rows = _dataset_rows()
    summary_rows = _summary_rows(rows)
    _write_csv(SUMMARY_ROOT / "beta_complexity_per_dataset.csv", rows)
    _write_csv(SUMMARY_ROOT / "beta_complexity_summary.csv", summary_rows)
    _plot(summary_rows)


if __name__ == "__main__":
    main()
