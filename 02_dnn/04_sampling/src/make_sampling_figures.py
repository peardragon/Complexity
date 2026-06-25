from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FIGURES: tuple[tuple[str, str, str], ...] = (
    ("fig_sampling_q05_ess_fraction_by_radius.png", "q05_ess_fraction", "q05 ESS fraction"),
    ("fig_sampling_split_logZ_per_P_by_radius.png", "max_split_logZ_per_P_diff", "max split logZ/P difference"),
    ("fig_smc_mh_acceptance_by_radius.png", "mean_smc_mh_acceptance", "mean SMC MH acceptance"),
    ("fig_smc_min_cess_by_radius.png", "min_smc_cess_fraction", "min SMC CESS fraction"),
    ("fig_smc_step_count_by_radius.png", "mean_smc_step_count", "mean SMC step count"),
    ("fig_weighted_accuracy_by_radius.png", "mean_weighted_accuracy", "mean weighted accuracy"),
    ("fig_weighted_ce_l2_ratio_by_radius.png", "mean_weighted_ce_l2_ratio", "mean weighted CE/L2 ratio"),
)


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, output_png: Path, range_label: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.6))

    for beta, group in df.groupby("beta", sort=True):
        group = group.sort_values("radius")
        plt.plot(group["radius"], group[metric], linewidth=0.9, alpha=0.45, color="#476a9f")

    aggregate = df.groupby("radius", as_index=False)[metric].median().sort_values("radius")
    plt.plot(aggregate["radius"], aggregate[metric], linewidth=2.2, color="#b3261e", label="median over beta")

    plt.xlabel("d")
    plt.ylabel(ylabel)
    plt.title(f"Sampling summary by radius ({range_label})")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=170)
    plt.close()


def make_figures(summary_csv: Path, figure_dir: Path, range_label: str) -> list[Path]:
    df = pd.read_csv(summary_csv)
    outputs: list[Path] = []
    for filename, metric, ylabel in FIGURES:
        if metric not in df.columns:
            continue
        output_png = figure_dir / filename
        _plot_metric(df, metric, ylabel, output_png, range_label)
        outputs.append(output_png)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render DNN shell sampling figures from compact summary tables.")
    parser.add_argument(
        "--range-root",
        type=Path,
        required=True,
        help="Range root under 02_dnn/04_sampling/raw_outputs/shell_pool/.../d_*.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        required=True,
        help="Matching figure directory under 02_dnn/04_sampling/figures/.../d_*.",
    )
    args = parser.parse_args()

    range_label = args.range_root.name
    summary_csv = args.range_root / "summary_tables" / "beta_radius_summary.csv"
    outputs = make_figures(summary_csv, args.figure_dir, range_label)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
