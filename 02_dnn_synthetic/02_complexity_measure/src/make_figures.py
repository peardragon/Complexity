from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = PROJECT_ROOT / "02_dnn_synthetic" / "02_complexity_measure"
SUMMARY_PATH = STAGE_ROOT / "summarized_outputs" / "beta_complexity_summary.csv"
FIGURE_PATH = STAGE_ROOT / "figures" / "beta_complexity_figure.png"


def _read_summary(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"run src/make_summarized_outputs.py first: {path}")

    required_fields = ("beta", "complexity_mean", "complexity_se")
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            rows.append(
                {
                    "beta": float(row["beta"]),
                    "complexity_mean": float(row["complexity_mean"]),
                    "complexity_se": float(row["complexity_se"]),
                }
            )

    if not rows:
        raise ValueError(f"no rows for {path}")
    return rows


def _plot(summary_rows: list[dict[str, float]]) -> None:
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
    summary_rows = _read_summary(SUMMARY_PATH)
    _plot(summary_rows)


if __name__ == "__main__":
    main()
