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


def make_phi_figure(input_csv: Path, output_png: Path) -> None:
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
    make_phi_figure(project_path(args.input_csv), output_png)
    print(output_png)


if __name__ == "__main__":
    main()
