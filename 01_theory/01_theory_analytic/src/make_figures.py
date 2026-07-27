from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "01_theory"
    / "01_theory_analytic"
    / "summarized_outputs"
    / "phi_by_analytic_solution_alpha0p1.csv"
)
DEFAULT_OUTPUT_PNG = (
    PROJECT_ROOT
    / "01_theory"
    / "01_theory_analytic"
    / "figures"
    / "phi_by_analytic_solution_alpha0p1.png"
)
DEFAULT_ENERGETIC_OUTPUT_PNG = (
    PROJECT_ROOT
    / "01_theory"
    / "01_theory_analytic"
    / "figures"
    / "phi_energetic_by_analytic_solution_alpha0p1.png"
)


@dataclass(frozen=True)
class TheoryCurve:
    source: Path
    label: str
    d: np.ndarray
    phi: np.ndarray
    alpha: float
    kappa: float
    kappa_tilde: float


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def spherical_shell_entropy(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    return np.log(np.maximum(d, 1.0e-300))


def find_crossings(
    reference: TheoryCurve,
    comparison: TheoryCurve,
    *,
    d_min: float,
    d_max: float,
) -> list[dict[str, float | str]]:
    x = np.asarray(reference.d, dtype=np.float64)
    y = np.asarray(reference.phi, dtype=np.float64) - np.interp(x, comparison.d, comparison.phi)
    rows: list[dict[str, float | str]] = []
    for idx in range(len(x) - 1):
        if x[idx] < d_min or x[idx + 1] > d_max:
            continue
        y0, y1 = float(y[idx]), float(y[idx + 1])
        if y0 == 0.0:
            root = float(x[idx])
        elif y0 * y1 > 0.0:
            continue
        else:
            root = float(x[idx] - y0 * (x[idx + 1] - x[idx]) / (y1 - y0))
        rows.append(
            {
                "status": "crossing",
                "reference_kappa_tilde": float(reference.kappa_tilde),
                "comparison_kappa_tilde": float(comparison.kappa_tilde),
                "crossing_d": root,
            }
        )
    return rows


def _relative_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_rel" in frame.columns:
        return frame["phi_rel"]
    return frame["phi"] - frame["phi"].iloc[0]


def _relative_energetic_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_energy_rel" in frame.columns:
        return frame["phi_energy_rel"]
    if "phi_energy" in frame.columns:
        return frame["phi_energy"] - frame["phi_energy"].iloc[0]
    radius_ratio = frame["r"] / float(frame["r"].iloc[0])
    return _relative_phi(frame) - np.log(radius_ratio)


def make_phi_figure(input_csv: Path, output_png: Path, *, energetic_only: bool = False) -> None:
    df = pd.read_csv(input_csv).sort_values("r")
    if energetic_only:
        y_values = _relative_energetic_phi(df)
        y_label = "energetic phi(d) - energetic phi(d0)"
        title = "Analytic energetic full-RS solution, alpha=0.1"
        color = "#9a4d12"
    else:
        y_values = _relative_phi(df)
        y_label = "phi(d) - phi(d0)"
        title = "Analytic full-RS solution, alpha=0.1"
        color = "#2457a7"

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["r"], y_values, marker="o", linewidth=2.0, color=color)
    plt.xlabel("d")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def make_figure(input_csv: Path, output_png: Path) -> None:
    make_phi_figure(input_csv, output_png)


def _load_curves(path: Path) -> list[TheoryCurve]:
    df = pd.read_csv(path)
    curves: list[TheoryCurve] = []
    for key, group in df.groupby(["alpha", "kappa", "kappa_tilde"]):
        alpha, kappa, kappa_tilde = key
        curves.append(
            TheoryCurve(
                source=path,
                label=f"kt={float(kappa_tilde):g}",
                d=group["d"].to_numpy(dtype=np.float64),
                phi=group["phi"].to_numpy(dtype=np.float64),
                alpha=float(alpha),
                kappa=float(kappa),
                kappa_tilde=float(kappa_tilde),
            )
        )
    return curves


def render_phase_reference_views(
    combined_paths: list[Path],
    *,
    figure_dir: Path,
    crossing_table_path: Path,
    reference_kappa_tilde: float,
) -> tuple[list[Path], list[dict[str, float | str]]]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    curves = [curve for path in combined_paths for curve in _load_curves(path)]
    figures: list[Path] = []
    crossing_rows: list[dict[str, float | str]] = []
    by_source: dict[Path, list[TheoryCurve]] = {}
    for curve in curves:
        by_source.setdefault(curve.source, []).append(curve)
    for source, source_curves in by_source.items():
        plt.figure(figsize=(7, 4))
        for curve in source_curves:
            plt.plot(curve.d, curve.phi, marker="o", label=curve.label)
        plt.xlabel("d")
        plt.ylabel("phi")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = figure_dir / f"{source.stem}_overlay.png"
        plt.savefig(out, dpi=120)
        plt.close()
        figures.append(out)
        ref = next((curve for curve in source_curves if abs(curve.kappa_tilde - reference_kappa_tilde) < 1.0e-12), None)
        if ref is not None:
            for curve in source_curves:
                if curve is not ref:
                    crossing_rows.extend(find_crossings(ref, curve, d_min=float(np.min(ref.d)), d_max=float(np.max(ref.d))))
    crossing_table_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["status", "reference_kappa_tilde", "comparison_kappa_tilde", "crossing_d"]
    with crossing_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in crossing_rows:
            writer.writerow(row)
    if figures:
        summary = figure_dir / "phase_reference_summary.png"
        figures.append(summary)
        figures[0].replace(summary)
        figures[0] = summary
    return figures, crossing_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render analytic phi(d) figure from the canonical full-RS CSV.")
    parser.add_argument("--which", choices=["full", "energetic", "all"], default="all")
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
    parser.add_argument(
        "--energetic-output-png",
        type=Path,
        default=DEFAULT_ENERGETIC_OUTPUT_PNG,
    )
    args = parser.parse_args()
    input_csv = project_path(args.input_csv)
    outputs: list[Path] = []
    if args.which in {"full", "all"}:
        output_png = project_path(args.output_png)
        make_phi_figure(input_csv, output_png)
        outputs.append(output_png)
    if args.which in {"energetic", "all"}:
        energetic_output_png = project_path(args.energetic_output_png)
        make_phi_figure(input_csv, energetic_output_png, energetic_only=True)
        outputs.append(energetic_output_png)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
