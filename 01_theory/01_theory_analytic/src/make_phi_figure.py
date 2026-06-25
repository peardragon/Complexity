from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_figure(input_csv: Path, output_png: Path) -> None:
    df = pd.read_csv(input_csv).sort_values("r")
    y_col = "phi_rel" if "phi_rel" in df.columns else "phi"

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["r"], df[y_col], marker="o", linewidth=2.0, color="#2457a7")
    plt.xlabel("d")
    plt.ylabel("phi(d) - phi(d0)" if y_col == "phi_rel" else "phi(d)")
    plt.title("Analytic full-RS solution, alpha=0.1")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render analytic phi(d) figure from the canonical full-RS CSV.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("01_theory/01_theory_analytic/figures/fig01_phi_by_analytic_solution_alpha0p1.png"),
    )
    args = parser.parse_args()
    make_figure(args.input_csv, args.output_png)
    print(args.output_png)


if __name__ == "__main__":
    main()
