from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _series_value(frame: pd.DataFrame) -> pd.Series:
    if "phi_emp_rel" in frame.columns:
        return frame["phi_emp_rel"]
    base = frame.sort_values("r")["phi_emp"].iloc[0]
    return frame["phi_emp"] - base


def make_figure(input_csv: Path, output_png: Path) -> None:
    df = pd.read_csv(input_csv).sort_values(["N", "r"])
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.4, 4.8))
    for n_value, group in df.groupby("N", sort=True):
        group = group.sort_values("r")
        plt.plot(group["r"], _series_value(group), marker="o", linewidth=1.7, label=f"N={int(n_value)}")
    plt.xlabel("d")
    plt.ylabel("empirical phi(d) - phi(d0)")
    plt.title("Two-pool shell sampling, alpha=0.1")
    plt.grid(True, alpha=0.28)
    plt.legend(title="system size", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sampling-only phi(d) figure from the compact summary CSV.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("01_theory/02_theory_sampling/figures/fig01_sampling_phi_by_distance.png"),
    )
    args = parser.parse_args()
    make_figure(args.input_csv, args.output_png)
    print(args.output_png)


if __name__ == "__main__":
    main()
