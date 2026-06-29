from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "01_theory"
    / "01_theory_analytic"
    / "summarized_outputs"
    / "fig01_phi_by_analytic_solution_alpha0p1.csv"
)
DEFAULT_OUTPUT_PNG = (
    PROJECT_ROOT
    / "01_theory"
    / "01_theory_analytic"
    / "figures"
    / "fig01_phi_by_analytic_solution_alpha0p1.png"
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


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
        default=DEFAULT_INPUT_CSV,
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
    )
    args = parser.parse_args()
    output_png = project_path(args.output_png)
    make_figure(project_path(args.input_csv), output_png)
    print(output_png)


if __name__ == "__main__":
    main()
