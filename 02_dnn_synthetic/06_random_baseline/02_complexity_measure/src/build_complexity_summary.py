from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "02_complexity_measure"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / "gaussian_random_90_dataset_30_reference" / "summary_tables"
FIGURE_ROOT = STAGE_ROOT / "figures"
RANDOM_DATASET_ROOT = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "06_random_baseline"
    / "01_dataset"
    / "raw_outputs"
    / "gaussian_random_90_dataset"
    / "raw_datasets"
)
RANDOM_RUN_CONFIG = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "06_random_baseline"
    / "01_dataset"
    / "raw_outputs"
    / "gaussian_random_90_dataset"
    / "run_config.json"
)
BASELINE_CONFIG = REPO_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "config" / "default.json"
SPIN_PER_DATASET_SOURCE = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "02_complexity_measure"
    / "summarized_outputs"
    / "18_beta_cell_90_dataset"
    / "beta_complexity_per_dataset.csv"
)
SPIN_SUMMARY_SOURCE = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "02_complexity_measure"
    / "summarized_outputs"
    / "18_beta_cell_90_dataset"
    / "beta_complexity_summary.csv"
)
PER_DATASET_CSV = SUMMARY_ROOT / "dataset_complexity_per_dataset.csv"
BY_RUN_CSV = SUMMARY_ROOT / "dataset_complexity_by_run_beta.csv"
NEAREST_CSV = SUMMARY_ROOT / "nearest_spin_beta_to_gaussian_complexity.csv"
SPIN_CURVE_CSV = SUMMARY_ROOT / "spin_beta_complexity_curve.csv"
RANDOM_POINT_CSV = SUMMARY_ROOT / "random_baseline_complexity_point.csv"
MARKER_CSV = SUMMARY_ROOT / "spin_beta_curve_with_random_baseline_marker.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_beta_tag() -> float:
    if not RANDOM_RUN_CONFIG.exists():
        if BASELINE_CONFIG.exists():
            with BASELINE_CONFIG.open(encoding="utf-8") as f:
                config = json.load(f)
            return float(config.get("sampling", {}).get("source_beta_tag", 0.05))
        return 0.05
    with RANDOM_RUN_CONFIG.open(encoding="utf-8") as f:
        config = json.load(f)
    selected = config.get("selected_betas") or [0.05]
    return float(selected[0])


def _dataset_complexity(dataset_path: Path, k: int = 3) -> float:
    with np.load(dataset_path) as data:
        x = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y"]).reshape(-1)
    nn = NearestNeighbors(n_neighbors=int(k) + 1, algorithm="auto")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float(np.mean(y[:, None] != y[indices]))


def _spin_per_dataset_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _read_csv(SPIN_PER_DATASET_SOURCE):
        rows.append(
            {
                "run": "spin_dynamics_90_dataset",
                "beta_role": "spin_sweep_axis",
                "beta": float(row["beta"]),
                "source_beta_tag": "",
                "cell_id": row["cell_id"],
                "dataset_id": int(row["dataset_id"]),
                "dataset_tag": row["dataset_tag"],
                "complexity_metric": "3nn_label_disagreement",
                "knn_k": 3,
                "complexity_3nn_disagreement": float(row["complexity_3nn_disagreement"]),
                "source_dataset": row["source_dataset"],
            }
        )
    return rows


def _random_per_dataset_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_beta = _source_beta_tag()
    for dataset_path in sorted(RANDOM_DATASET_ROOT.glob("dataset_*/dataset.npz")):
        meta_path = dataset_path.with_name("dataset_meta.json")
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        rows.append(
            {
                "run": "gaussian_random_90_dataset",
                "beta_role": "source_tag_only_not_sweep_axis",
                "beta": source_beta,
                "source_beta_tag": source_beta,
                "cell_id": "gaussian_random_baseline",
                "dataset_id": int(meta["dataset_id"]),
                "dataset_tag": dataset_path.parent.name,
                "complexity_metric": "3nn_label_disagreement",
                "knn_k": 3,
                "complexity_3nn_disagreement": _dataset_complexity(dataset_path, k=3),
                "source_dataset": str(dataset_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def _spin_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _read_csv(SPIN_SUMMARY_SOURCE):
        rows.append(
            {
                "run": "spin_dynamics_90_dataset",
                "beta_role": "spin_sweep_axis",
                "beta": float(row["beta"]),
                "source_beta_tag": "",
                "dataset_count": int(row["dataset_count"]),
                "complexity_metric": "3nn_label_disagreement",
                "complexity_mean": float(row["complexity_mean"]),
                "complexity_std": float(row["complexity_sd"]),
                "complexity_sem": float(row["complexity_se"]),
            }
        )
    return rows


def _random_summary_row(rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(row["complexity_3nn_disagreement"]) for row in rows]
    spread = stdev(values) if len(values) > 1 else 0.0
    beta = float(rows[0]["source_beta_tag"])
    return {
        "run": "gaussian_random_90_dataset",
        "beta_role": "source_tag_only_not_sweep_axis",
        "beta": beta,
        "source_beta_tag": beta,
        "dataset_count": len(values),
        "complexity_metric": "3nn_label_disagreement",
        "complexity_mean": mean(values),
        "complexity_std": spread,
        "complexity_sem": spread / math.sqrt(len(values)),
    }



def _nearest_spin_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    random_rows = [row for row in summary_rows if row["run"] == "gaussian_random_90_dataset"]
    if len(random_rows) != 1:
        raise ValueError(f"expected one gaussian random row, found {len(random_rows)}")
    random_complexity = float(random_rows[0]["complexity_mean"])
    spin_rows = [row for row in summary_rows if row["run"] == "spin_dynamics_90_dataset"]
    out: list[dict[str, object]] = []
    for row in sorted(spin_rows, key=lambda item: abs(float(item["complexity_mean"]) - random_complexity)):
        clean = dict(row)
        clean["abs_gap_to_gaussian_complexity"] = abs(float(row["complexity_mean"]) - random_complexity)
        out.append(clean)
    return out


def _random_point_row(random_row: dict[str, object], nearest_row: dict[str, object]) -> dict[str, object]:
    return {
        "baseline": "gaussian_random_90_dataset",
        "beta_role": "source_tag_only_not_a_sweep_axis",
        "source_beta_tag": random_row["beta"],
        "dataset_count": random_row["dataset_count"],
        "complexity_metric": random_row["complexity_metric"],
        "complexity_mean": random_row["complexity_mean"],
        "complexity_std": random_row["complexity_std"],
        "complexity_sem": random_row["complexity_sem"],
        "nearest_spin_beta": nearest_row["beta"],
        "nearest_spin_complexity_mean": nearest_row["complexity_mean"],
        "abs_gap_to_nearest_spin_complexity": nearest_row["abs_gap_to_gaussian_complexity"],
    }


def _marker_rows(spin_rows: list[dict[str, object]], random_point: dict[str, object]) -> list[dict[str, object]]:
    nearest_beta = float(random_point["nearest_spin_beta"])
    random_complexity = float(random_point["complexity_mean"])
    random_sem = float(random_point["complexity_sem"])
    out: list[dict[str, object]] = []
    for row in spin_rows:
        clean = dict(row)
        clean["plot_role"] = "spin_beta_curve"
        clean["random_baseline_complexity_mean"] = random_complexity
        clean["random_baseline_complexity_sem"] = random_sem
        clean["random_baseline_point_beta"] = nearest_beta
        clean["gap_to_random_baseline_complexity"] = abs(float(row["complexity_mean"]) - random_complexity)
        clean["is_nearest_spin_beta"] = float(row["beta"]) == nearest_beta
        out.append(clean)
    return out


def _plot_summary(summary_rows: list[dict[str, object]]) -> None:
    random_row = next(row for row in summary_rows if row["run"] == "gaussian_random_90_dataset")
    spin_rows = sorted(
        [row for row in summary_rows if row["run"] == "spin_dynamics_90_dataset"],
        key=lambda row: float(row["beta"]),
    )
    beta = [float(row["beta"]) for row in spin_rows]
    complexity = [float(row["complexity_mean"]) for row in spin_rows]
    sem = [float(row["complexity_sem"]) for row in spin_rows]
    random_complexity = float(random_row["complexity_mean"])
    random_sem = float(random_row["complexity_sem"])
    nearest_row = min(spin_rows, key=lambda row: abs(float(row["complexity_mean"]) - random_complexity))
    nearest_beta = float(nearest_row["beta"])

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.errorbar(beta, complexity, yerr=sem, color="#2364aa", marker="o", markersize=4.0, linewidth=1.35, capsize=2.5, label="spin synthetic")
    ax.axhline(random_complexity, color="black", linewidth=1.5, linestyle="--", label="random baseline complexity")
    ax.fill_between(
        [min(beta), max(beta)],
        [random_complexity - random_sem, random_complexity - random_sem],
        [random_complexity + random_sem, random_complexity + random_sem],
        color="black",
        alpha=0.08,
        linewidth=0,
    )
    ax.scatter([nearest_beta], [random_complexity], color="black", marker="s", s=48, zorder=4)
    ax.set_xlabel("spin beta")
    ax.set_ylabel("3-NN label-disagreement complexity")
    ax.set_title("Random baseline complexity on spin beta curve")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / "random_baseline_complexity_summary.png", dpi=240)
    plt.close(fig)


def main() -> None:
    if not RANDOM_DATASET_ROOT.exists() and BY_RUN_CSV.exists():
        _plot_summary(_read_csv(BY_RUN_CSV))
        return

    spin_dataset_rows = _spin_per_dataset_rows()
    random_dataset_rows = _random_per_dataset_rows()
    rows = spin_dataset_rows + random_dataset_rows
    _write_csv(PER_DATASET_CSV, rows)

    spin_rows = _spin_summary_rows()
    random_row = _random_summary_row(random_dataset_rows)
    summary_rows = spin_rows + [random_row]
    nearest_rows = _nearest_spin_rows(summary_rows)
    random_point = _random_point_row(random_row, nearest_rows[0])
    _write_csv(BY_RUN_CSV, summary_rows)
    _write_csv(SPIN_CURVE_CSV, spin_rows)
    _write_csv(RANDOM_POINT_CSV, [random_point])
    _write_csv(MARKER_CSV, _marker_rows(spin_rows, random_point))
    _write_csv(NEAREST_CSV, nearest_rows)
    _plot_summary(summary_rows)


if __name__ == "__main__":
    main()
