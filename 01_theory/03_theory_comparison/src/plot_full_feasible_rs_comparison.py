from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BRANCH_STYLE = {
    "boundary_mixed_eta0": {
        "label": "boundary mixed (eta=0)",
        "color": "#1f5aa6",
        "linestyle": "-",
        "linewidth": 2.3,
    },
    "full_mixed_maxQ_min_s_eta": {
        "label": "full mixed",
        "color": "#d1495b",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "full_max_envelope": {
        "label": "full max envelope",
        "color": "#2a9d8f",
        "linestyle": ":",
        "linewidth": 2.4,
    },
}


def load_branch(theory: pd.DataFrame, branch: str) -> pd.DataFrame:
    out = theory[theory["branch"] == branch].copy().sort_values("r")
    if out.empty:
        raise ValueError(f"missing branch: {branch}")
    return out


def add_one_run(
    axes: np.ndarray,
    *,
    col: int,
    title: str,
    theory_csv: Path,
    sampling_csv: Path,
) -> None:
    theory = pd.read_csv(theory_csv)
    sampling = pd.read_csv(sampling_csv)
    largest_n = int(sampling["N"].max())
    sampling_largest = sampling[sampling["N"] == largest_n].copy().sort_values("r")
    boundary = load_branch(theory, "boundary_mixed_eta0")
    boundary_phi = np.interp(
        theory["r"].to_numpy(dtype=float),
        boundary["r"].to_numpy(dtype=float),
        boundary["phi_rel"].to_numpy(dtype=float),
    )

    ax_curve = axes[0, col]
    for branch, style in BRANCH_STYLE.items():
        bdf = load_branch(theory, branch)
        ax_curve.plot(
            bdf["r"],
            bdf["phi_rel"],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
    ax_curve.scatter(
        sampling_largest["r"],
        sampling_largest["phi_emp"],
        label=f"PM-SAIS N={largest_n}",
        color="#111111",
        s=16,
        zorder=5,
        alpha=0.85,
    )
    ax_curve.set_title(title)
    ax_curve.set_ylabel("Phi(r) - Phi(r0)")
    ax_curve.grid(True, alpha=0.28)
    ax_curve.legend(fontsize=8, frameon=False)

    ax_delta = axes[1, col]
    r_boundary = boundary["r"].to_numpy(dtype=float)
    y_boundary = boundary["phi_rel"].to_numpy(dtype=float)
    for branch in ("full_mixed_maxQ_min_s_eta", "full_max_envelope"):
        bdf = load_branch(theory, branch)
        delta = bdf["phi_rel"].to_numpy(dtype=float) - np.interp(
            bdf["r"].to_numpy(dtype=float),
            r_boundary,
            y_boundary,
        )
        style = BRANCH_STYLE[branch]
        ax_delta.plot(
            bdf["r"],
            delta,
            label=f"{style['label']} - boundary",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
    ax_delta.axhline(0.0, color="#222222", linewidth=0.9, alpha=0.8)
    ax_delta.set_ylabel("branch delta")
    ax_delta.grid(True, alpha=0.28)
    ax_delta.legend(fontsize=8, frameon=False)

    ax_A = axes[2, col]
    for branch in ("full_mixed_maxQ_min_s_eta", "full_max_envelope"):
        bdf = load_branch(theory, branch)
        style = BRANCH_STYLE[branch]
        ax_A.plot(
            bdf["r"],
            bdf["A"],
            label=f"{style['label']} A",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
    ax_A.axhline(0.0, color="#222222", linewidth=0.9, alpha=0.8)
    ax_A.set_xlabel("r")
    ax_A.set_ylabel("A")
    ax_A.grid(True, alpha=0.28)
    ax_A.legend(fontsize=8, frameon=False)

    # Keep the curve panel focused on the actual theory/sampling scale while
    # allowing the delta panel to show small differences.
    for ax in axes[:, col]:
        ax.set_xlim(float(boundary["r"].min()), float(boundary["r"].max()))


def make_figure(
    *,
    coarse_csv: Path,
    fine_csv: Path,
    sampling_csv: Path,
    output_png: Path,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14.5, 10.0),
        sharex="col",
        gridspec_kw={"height_ratios": [1.45, 1.0, 1.0]},
    )
    add_one_run(
        axes,
        col=0,
        title="Coarse grid: q=45, s=41, eta=21",
        theory_csv=coarse_csv,
        sampling_csv=sampling_csv,
    )
    add_one_run(
        axes,
        col=1,
        title="Fine grid: q=75, s=61, eta=31",
        theory_csv=fine_csv,
        sampling_csv=sampling_csv,
    )
    fig.suptitle("Full feasible RS branches vs PM-SAIS, alpha=0.1", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    fig.savefig(output_png, dpi=170)
    plt.close(fig)
    return output_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot full feasible RS branch comparison.")
    parser.add_argument(
        "--coarse-csv",
        type=Path,
        default=Path("01_theory/01_theory_analytic/raw_outputs/theory_full_feasible_rs_alpha0p1.csv"),
    )
    parser.add_argument(
        "--fine-csv",
        type=Path,
        default=Path("01_theory/01_theory_analytic/raw_outputs/theory_full_feasible_rs_alpha0p1_fine.csv"),
    )
    parser.add_argument(
        "--sampling-csv",
        type=Path,
        default=Path("01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("01_theory/03_theory_comparison/figures/full_feasible_rs_alpha0p1/fig01_full_feasible_branch_comparison.png"),
    )
    args = parser.parse_args()
    out = make_figure(
        coarse_csv=args.coarse_csv,
        fine_csv=args.fine_csv,
        sampling_csv=args.sampling_csv,
        output_png=args.output_png,
    )
    print(out)


if __name__ == "__main__":
    main()
