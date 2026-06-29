from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_summary_inputs import FIGURE_INPUT_ROOT, SPLIT_LOGZ_THRESHOLD, STAGE_ROOT, build_outputs  # noqa: E402


FIGURE_ROOT = STAGE_ROOT / "figures"
COLORS = {
    "rule_001": "#0072B2",
    "rule_002": "#009E73",
    "rule_003": "#D55E00",
    "rule_004": "#CC79A7",
}


def _plot_logz(path: Path) -> None:
    frame = pd.read_csv(FIGURE_INPUT_ROOT / "logZ_split_qc_results" / "logZ_split_qc_results.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)
    for rule_id, sub in frame.groupby("rule_id", sort=True):
        sub = sub.sort_values("radius")
        label = str(sub["rule_label"].iloc[0])
        ax.plot(
            sub["radius"],
            sub["q95_split_logZ_per_P_diff"],
            marker="o",
            linewidth=1.35,
            markersize=3.2,
            color=COLORS.get(rule_id),
            label=label,
        )
    ax.axhline(SPLIT_LOGZ_THRESHOLD, color="#B42318", linestyle="--", linewidth=1.1, label="QC threshold")
    ax.set_xlabel("distance d")
    ax.set_ylabel("q95 split logZ / P diff")
    ax.set_title("MNIST manual-rule split logZ QC")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_reference_variability(path: Path) -> None:
    frame = pd.read_csv(
        FIGURE_INPUT_ROOT / "reference_variability_results" / "reference_variability_results.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    panels = [
        ("reference_se_phi_energy_raw", "SE energetic phi(d)", "Reference SE of energetic phi"),
        ("reference_se_logZ_inf_full", "SE logZ", "Reference SE of logZ"),
    ]
    for ax, (value, ylabel, title) in zip(axes, panels):
        for rule_id, sub in frame.groupby("rule_id", sort=True):
            sub = sub.sort_values("radius")
            label = str(sub["rule_label"].iloc[0])
            ax.plot(
                sub["radius"],
                sub[value],
                marker="o",
                linewidth=1.35,
                markersize=3.2,
                color=COLORS.get(rule_id),
                label=label,
            )
        ax.set_xlabel("distance d")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.24)
    axes[0].legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    build_outputs()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    _plot_logz(FIGURE_ROOT / "logZ_split_qc_results.png")
    _plot_reference_variability(FIGURE_ROOT / "reference_variability_results.png")


if __name__ == "__main__":
    main()
