from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = STAGE_ROOT / "summarized_outputs" / "figure_inputs" / "reference_quality" / "reference_quality_by_pair.csv"
FIGURE_PATH = STAGE_ROOT / "figures" / "reference_quality_by_pair.png"


def _read_rows(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        raise FileNotFoundError(f"run src/make_summarized_outputs.py first: {path}")
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "pair_label": str(row["pair_label"]),
                    "pair_order": int(float(row["pair_order"])),
                    "complexity_mean": float(row["complexity_mean"]),
                    "test_error_mean": float(row["test_error_mean"]),
                    "test_error_sem": float(row["test_error_sem"]),
                    "ref_count": int(float(row["ref_count"])),
                }
            )
    return sorted(rows, key=lambda row: int(row["pair_order"]))


def main() -> None:
    rows = _read_rows(SUMMARY_PATH)
    labels = [str(row["pair_label"]) for row in rows]
    x = np.arange(len(rows), dtype=float)
    test_error = np.asarray([float(row["test_error_mean"]) for row in rows])
    test_sem = np.asarray([float(row["test_error_sem"]) for row in rows])
    complexity = np.asarray([float(row["complexity_mean"]) for row in rows])

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_left = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax_left.errorbar(x, test_error, yerr=test_sem, marker="o", capsize=3, color="#284f8f", label="test error")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_xlabel("digit pair")
    ax_left.set_ylabel("reference test error")
    ax_left.grid(True, alpha=0.25)
    ax_right = ax_left.twinx()
    ax_right.plot(x, complexity, marker="s", color="#a23b72", label="complexity")
    ax_right.set_ylabel("3-NN complexity")
    ax_left.set_title("Digit-pair reference quality")
    lines = ax_left.get_lines() + ax_right.get_lines()
    ax_left.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
    fig.savefig(FIGURE_PATH, dpi=240)
    plt.close(fig)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
