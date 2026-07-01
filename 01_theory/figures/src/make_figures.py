from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


THEORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYTIC_CSV = (
    THEORY_ROOT
    / "01_theory_analytic"
    / "summarized_outputs"
    / "phi_by_analytic_solution_alpha0p1.csv"
)
DEFAULT_SAMPLING_INPUT = (
    THEORY_ROOT
    / "02_theory_sampling"
    / "summarized_outputs"
    / "figure_inputs"
    / "phi_by_sampling"
)
DEFAULT_OUTPUT_PNG = THEORY_ROOT / "figures" / "fig01_sampling_vs_analytic_phi_by_distance_alpha0p1.png"


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else THEORY_ROOT / path


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_sampling_input(path: Path) -> pd.DataFrame:
    if path.is_file():
        return read_required_csv(path)
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(path.glob("N_*.csv"))
    if not files:
        raise FileNotFoundError(f"no N_*.csv files found under {path}")
    return pd.concat((pd.read_csv(csv_path) for csv_path in files), ignore_index=True, sort=False)


def analytic_relative_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_rel" in frame.columns:
        return frame["phi_rel"]
    ordered = frame.sort_values("r")
    return ordered["phi"] - ordered["phi"].iloc[0]


def sampling_relative_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_emp_rel" in frame.columns:
        return frame["phi_emp_rel"]
    ordered = frame.sort_values("r")
    return ordered["phi_emp"] - ordered["phi_emp"].iloc[0]


def write_combined_figure(analytic: pd.DataFrame, sampling: pd.DataFrame, output: Path) -> None:
    analytic = analytic.sort_values("r")
    sampling = sampling.sort_values(["N", "r"])

    plt.figure(figsize=(8.5, 5.1))
    plt.plot(
        analytic["r"],
        analytic_relative_phi(analytic),
        color="black",
        linewidth=2.4,
        label="analytic full-RS",
    )
    for n_value, group in sampling.groupby("N", sort=True):
        group = group.sort_values("r")
        plt.plot(
            group["r"],
            sampling_relative_phi(group),
            marker="o",
            markersize=3.2,
            linewidth=1.55,
            label=f"N={int(n_value)} sampling",
        )
    plt.xlabel("d")
    plt.ylabel("phi(d) - phi(d0)")
    plt.title("Analytic vs shell sampling, alpha=0.1")
    plt.grid(True, alpha=0.28)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the theory analytic-vs-sampling combined figure.")
    parser.add_argument("--analytic-csv", type=Path, default=DEFAULT_ANALYTIC_CSV)
    parser.add_argument("--sampling-input", "--sampling-csv", dest="sampling_input", type=Path, default=DEFAULT_SAMPLING_INPUT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    args = parser.parse_args()

    output_png = project_path(args.output_png)
    write_combined_figure(
        read_required_csv(project_path(args.analytic_csv)),
        read_sampling_input(project_path(args.sampling_input)),
        output_png,
    )
    print(output_png)


if __name__ == "__main__":
    main()
