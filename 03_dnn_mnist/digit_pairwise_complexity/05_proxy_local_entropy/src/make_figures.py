from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from make_summarized_outputs import FIGURE_INPUT_ROOT, STAGE_ROOT, build as build_summarized_outputs  # noqa: E402


FIGURE_ROOT = STAGE_ROOT / "figures"
COLORS = {
    "pair_7_9": "#0072B2",
    "pair_4_8": "#009E73",
    "pair_2_4": "#D55E00",
    "pair_2_9": "#CC79A7",
    "pair_6_7": "#6F4E7C",
}


def _plot_curves(frame: pd.DataFrame, value: str, sem: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for pair_id, sub in frame.groupby("pair_id", sort=False):
        sub = sub.sort_values("radius")
        label = str(sub["pair_label"].iloc[0])
        ax.plot(sub["radius"], sub[value], linewidth=1.8, color=COLORS.get(pair_id), label=label)
        if sem in sub.columns:
            lower = sub[value] - sub[sem]
            upper = sub[value] + sub[sem]
            ax.fill_between(sub["radius"], lower, upper, color=COLORS.get(pair_id), alpha=0.14, linewidth=0)
    ax.set_xlabel("distance d")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.01, 0.02, "band: mean +/- SE across references", transform=ax.transAxes, fontsize=7.5, color="0.35")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_phase(x_key: str, x_label: str, output_name: str) -> None:
    phase = pd.read_csv(FIGURE_INPUT_ROOT / output_name / f"{output_name}.csv")
    curves = pd.read_csv(FIGURE_INPUT_ROOT / output_name / "phase_derivative_curves.csv")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    for pair_id, sub in curves.groupby("pair_id", sort=False):
        sub = sub.sort_values("radius")
        label = str(sub["pair_label"].iloc[0])
        ax_left.plot(sub["radius"], sub["dphi_dr_smooth_mean"], linewidth=1.5, color=COLORS.get(pair_id), label=label)
        if "dphi_dr_smooth_sem" in sub.columns:
            x = sub["radius"].to_numpy(dtype=float)
            y = sub["dphi_dr_smooth_mean"].to_numpy(dtype=float)
            err = sub["dphi_dr_smooth_sem"].fillna(0.0).to_numpy(dtype=float)
            ax_left.fill_between(x, y - err, y + err, color=COLORS.get(pair_id), alpha=0.10, linewidth=0)
    ax_left.set_xlabel("distance d")
    ax_left.set_ylabel("energetic d phi / dd")
    ax_left.set_title("Energetic derivative")
    ax_left.grid(True, alpha=0.24)
    ax_left.legend(frameon=False, fontsize=8)
    ax_left.text(0.01, 0.02, "band: mean +/- SE across references", transform=ax_left.transAxes, fontsize=7.5, color="0.35")

    phase = phase.sort_values(x_key)
    ax_right.errorbar(
        phase[x_key],
        phase["A_kappa_mean"],
        yerr=phase["A_kappa_sem"],
        marker="o",
        linewidth=1.5,
        capsize=2.8,
        color="#4C78A8",
    )
    for _, row in phase.iterrows():
        ax_right.annotate(str(row["pair_label"]), (row[x_key], row["A_kappa_mean"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax_right.set_xlabel(x_label)
    ax_right.set_ylabel("A measure")
    ax_right.set_title("A-measure phase-like plot")
    ax_right.grid(True, alpha=0.24)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{output_name}.png", dpi=240)
    plt.close(fig)


def build() -> None:
    build_summarized_outputs()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    phi = pd.read_csv(FIGURE_INPUT_ROOT / "phi_d_curve" / "phi_d_curve.csv")
    dphi = pd.read_csv(FIGURE_INPUT_ROOT / "derivative_phi_d_curve" / "derivative_phi_d_curve.csv")

    _plot_curves(
        phi,
        "delta_phi_energy_unit_mean",
        "delta_phi_energy_unit_sem",
        "phi(d) - phi(d0)",
        "MNIST digit-pair phi(d)",
        FIGURE_ROOT / "phi_d_curve.png",
    )
    _plot_curves(
        phi,
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        "energetic phi(d)",
        "MNIST digit-pair energetic phi(d)",
        FIGURE_ROOT / "phi_energetic_d_curve.png",
    )
    _plot_curves(
        dphi,
        "d_delta_phi_energy_direct_dd_unit_mean",
        "d_delta_phi_energy_direct_dd_unit_sem",
        "d phi / dd",
        "MNIST digit-pair direct derivative of phi(d)",
        FIGURE_ROOT / "derivative_phi_d_curve.png",
    )
    _plot_curves(
        dphi,
        "d_phi_energy_direct_dd_unit_mean",
        "d_phi_energy_direct_dd_unit_sem",
        "energetic d phi / dd",
        "MNIST digit-pair direct energetic derivative",
        FIGURE_ROOT / "derivative_phi_energetic_d_curve.png",
    )
    _plot_phase("pair_order", "selected pair order", "phase_like_A_by_pair")
    _plot_phase("complexity_mean", "3-NN MNIST complexity", "phase_like_A_by_complexity")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
