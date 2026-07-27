from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
DEFAULT_SAMPLING_ENERGETIC_INPUT = (
    THEORY_ROOT
    / "02_theory_sampling"
    / "summarized_outputs"
    / "figure_inputs"
    / "phi_energetic_by_sampling"
)
DEFAULT_OUTPUT_PNG = THEORY_ROOT / "figures" / "fig01_sampling_vs_analytic_phi_by_distance_alpha0p1.png"
DEFAULT_ENERGETIC_OUTPUT_PNG = (
    THEORY_ROOT / "figures" / "fig01_sampling_vs_analytic_phi_energetic_by_distance_alpha0p1.png"
)


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


def analytic_relative_energetic_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_energy_rel" in frame.columns:
        return frame["phi_energy_rel"]
    if "phi_energy" in frame.columns:
        ordered = frame.sort_values("r")
        return ordered["phi_energy"] - ordered["phi_energy"].iloc[0]
    ordered = frame.sort_values("r")
    radius_ratio = ordered["r"] / float(ordered["r"].iloc[0])
    return analytic_relative_phi(ordered) - np.log(radius_ratio)


def sampling_relative_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_emp_rel" in frame.columns:
        return frame["phi_emp_rel"]
    ordered = frame.sort_values("r")
    return ordered["phi_emp"] - ordered["phi_emp"].iloc[0]


def sampling_relative_energetic_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_energy_emp_rel" in frame.columns:
        return frame["phi_energy_emp_rel"]
    if "phi_energy_emp" in frame.columns:
        return frame["phi_energy_emp"]
    ordered = frame.sort_values("r")
    radius_ratio = ordered["r"] / float(ordered["r"].iloc[0])
    n_value = float(ordered["N"].iloc[0])
    radius_term = ((n_value - 1.0) / n_value) * np.log(radius_ratio)
    return sampling_relative_phi(ordered) - radius_term


def write_combined_figure(
    analytic: pd.DataFrame,
    sampling: pd.DataFrame,
    output: Path,
    *,
    energetic_only: bool = False,
) -> None:
    analytic = analytic.sort_values("r")
    sampling = sampling.sort_values(["N", "r"])
    analytic_y = analytic_relative_energetic_phi(analytic) if energetic_only else analytic_relative_phi(analytic)

    plt.figure(figsize=(8.5, 5.1))
    plt.plot(
        analytic["r"],
        analytic_y,
        color="black",
        linewidth=2.4,
        label="analytic full-RS",
    )
    for n_value, group in sampling.groupby("N", sort=True):
        group = group.sort_values("r")
        sampling_y = sampling_relative_energetic_phi(group) if energetic_only else sampling_relative_phi(group)
        plt.plot(
            group["r"],
            sampling_y,
            marker="o",
            markersize=3.2,
            linewidth=1.55,
            label=f"N={int(n_value)} sampling",
        )
    plt.xlabel("d")
    if energetic_only:
        plt.ylabel("energetic phi(d) - energetic phi(d0)")
        plt.title("Analytic vs shell sampling energetic part, alpha=0.1")
    else:
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
    parser.add_argument("--which", choices=["full", "energetic", "all"], default="all")
    parser.add_argument("--analytic-csv", type=Path, default=DEFAULT_ANALYTIC_CSV)
    parser.add_argument("--sampling-input", "--sampling-csv", dest="sampling_input", type=Path, default=DEFAULT_SAMPLING_INPUT)
    parser.add_argument(
        "--sampling-energetic-input",
        "--sampling-energetic-csv",
        dest="sampling_energetic_input",
        type=Path,
        default=DEFAULT_SAMPLING_ENERGETIC_INPUT,
    )
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--energetic-output-png", type=Path, default=DEFAULT_ENERGETIC_OUTPUT_PNG)
    args = parser.parse_args()

    analytic = read_required_csv(project_path(args.analytic_csv))
    outputs: list[Path] = []
    if args.which in {"full", "all"}:
        output_png = project_path(args.output_png)
        write_combined_figure(
            analytic,
            read_sampling_input(project_path(args.sampling_input)),
            output_png,
        )
        outputs.append(output_png)
    if args.which in {"energetic", "all"}:
        energetic_output_png = project_path(args.energetic_output_png)
        write_combined_figure(
            analytic,
            read_sampling_input(project_path(args.sampling_energetic_input)),
            energetic_output_png,
            energetic_only=True,
        )
        outputs.append(energetic_output_png)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
