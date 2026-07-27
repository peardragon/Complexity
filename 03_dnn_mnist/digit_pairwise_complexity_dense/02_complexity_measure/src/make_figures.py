from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = STAGE_ROOT / "summarized_outputs" / "digit_pairwise_complexity_summary.csv"
FIGURE_PATH = STAGE_ROOT / "figures" / "digit_pairwise_complexity_figure.png"


def _read_summary(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        raise FileNotFoundError(f"run src/make_summarized_outputs.py first: {path}")
    required_fields = ("rank_complexity_desc", "label", "complexity_mean")
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            rows.append(
                {
                    "rank_complexity_desc": int(row["rank_complexity_desc"]),
                    "label": str(row["label"]),
                    "complexity_mean": float(row["complexity_mean"]),
                }
            )
    if not rows:
        raise ValueError(f"no rows for {path}")
    return sorted(rows, key=lambda row: int(row["rank_complexity_desc"]))


def _plot(summary_rows: list[dict[str, float | int | str]]) -> None:
    labels = [str(row["label"]) for row in summary_rows]
    values = np.asarray([float(row["complexity_mean"]) for row in summary_rows], dtype=np.float64)
    y = np.arange(len(summary_rows), dtype=np.float64)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 10.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary_rows)))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("3-NN label-disagreement complexity")
    ax.set_ylabel("digit pair, ranked high to low")
    ax.set_title("MNIST digit-pair complexity ranking")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(FIGURE_PATH, dpi=240)
    plt.close(fig)


def main() -> None:
    rows = _read_summary(SUMMARY_PATH)
    _plot(rows)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
