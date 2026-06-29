#!/usr/bin/env python3
"""Positive-curvature-mass composite for eta label-flip phi(d)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = STAGE_ROOT / "raw_outputs" / "eta_reference_phi_promoted_4eta_10ref_d1_n1024_cpu35_gpu0"
DEFAULT_OUT_DIR = STAGE_ROOT / "figures" / "eta_positive_curvature_mass_small_d_n1024"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def load_units(run_root: Path) -> pd.DataFrame:
    path = run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    if not path.exists():
        path = run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv"
    if not path.exists():
        raise FileNotFoundError(f"unit phi table not found under {run_root}")
    df = pd.read_csv(path)
    for col in [
        "eta",
        "ref_id",
        "radius",
        "phi_energy_raw",
        "d_phi_energy_raw_dd_unit",
        "ess_fraction",
        "split_logZ_per_P_diff",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "d_phi_energy_raw_dd_unit" not in df.columns:
        df["d_phi_energy_raw_dd_unit"] = np.nan
    return df


def compute_ref_curvature(units: pd.DataFrame, d_min: float, d_max: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window = units[(units["radius"] >= d_min - 1.0e-9) & (units["radius"] <= d_max + 1.0e-9)].copy()
    rows = []
    curve_rows = []
    for (eta, ref_id), sub in window.groupby(["eta", "ref_id"], sort=True):
        sub = sub.sort_values("radius").drop_duplicates("radius", keep="first")
        if len(sub) < 3:
            continue
        radius = sub["radius"].to_numpy(dtype=np.float64)
        phi = sub["phi_energy_raw"].to_numpy(dtype=np.float64)
        d1 = sub["d_phi_energy_raw_dd_unit"].to_numpy(dtype=np.float64)
        if not np.isfinite(d1).all():
            d1 = np.gradient(phi, radius)
        d2 = np.gradient(d1, radius)
        pos = np.maximum(d2, 0.0)
        a_kappa = trapz(pos, radius)
        min_idx = int(np.nanargmin(d2))
        max_idx = int(np.nanargmax(d2))
        rows.append(
            {
                "eta": float(eta),
                "ref_id": int(ref_id),
                "n_radii": int(len(radius)),
                "d_min": float(radius.min()),
                "d_max": float(radius.max()),
                "positive_curvature_mass": float(a_kappa),
                "min_curvature": float(d2[min_idx]),
                "min_curvature_radius": float(radius[min_idx]),
                "max_curvature": float(d2[max_idx]),
                "max_curvature_radius": float(radius[max_idx]),
            }
        )
        for r, p, g, k in zip(radius, phi, d1, d2):
            curve_rows.append(
                {
                    "eta": float(eta),
                    "ref_id": int(ref_id),
                    "radius": float(r),
                    "phi_energy_raw": float(p),
                    "d_phi_energy_raw_dd": float(g),
                    "d2_phi_energy_raw_dd2": float(k),
                    "positive_curvature": float(max(k, 0.0)),
                }
            )
    ref_metrics = pd.DataFrame(rows).sort_values(["eta", "ref_id"]) if rows else pd.DataFrame()
    curvature = pd.DataFrame(curve_rows).sort_values(["eta", "ref_id", "radius"]) if curve_rows else pd.DataFrame()
    if ref_metrics.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            ref_metrics.groupby("eta", as_index=False)
            .agg(
                n_refs=("ref_id", "nunique"),
                positive_curvature_mass_mean=("positive_curvature_mass", "mean"),
                positive_curvature_mass_sd=("positive_curvature_mass", "std"),
                positive_curvature_mass_sem=("positive_curvature_mass", sem),
                min_curvature_mean=("min_curvature", "mean"),
                min_curvature_sd=("min_curvature", "std"),
                min_curvature_sem=("min_curvature", sem),
                min_curvature_radius_mean=("min_curvature_radius", "mean"),
                max_curvature_mean=("max_curvature", "mean"),
                max_curvature_radius_mean=("max_curvature_radius", "mean"),
            )
            .sort_values("eta")
        )
    return ref_metrics, curvature, summary


def summarize_curve(curvature: pd.DataFrame) -> pd.DataFrame:
    if curvature.empty:
        return pd.DataFrame()
    return (
        curvature.groupby(["eta", "radius"], as_index=False)
        .agg(
            n_refs=("ref_id", "nunique"),
            phi_energy_raw_mean=("phi_energy_raw", "mean"),
            phi_energy_raw_sem=("phi_energy_raw", sem),
            d_phi_energy_raw_dd_mean=("d_phi_energy_raw_dd", "mean"),
            d_phi_energy_raw_dd_sem=("d_phi_energy_raw_dd", sem),
            d2_phi_energy_raw_dd2_mean=("d2_phi_energy_raw_dd2", "mean"),
            d2_phi_energy_raw_dd2_sem=("d2_phi_energy_raw_dd2", sem),
            positive_curvature_mean=("positive_curvature", "mean"),
        )
        .sort_values(["eta", "radius"])
    )


def plot_composite(curve: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, inset_eta: float | None) -> Path:
    if curve.empty or metrics.empty:
        raise RuntimeError("No curvature rows available to plot.")

    etas = np.array(sorted(curve["eta"].unique()), dtype=np.float64)
    norm = plt.Normalize(vmin=float(etas.min()), vmax=float(etas.max()))
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(10.6, 4.2), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.34)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    for eta, sub in curve.groupby("eta", sort=True):
        sub = sub.sort_values("radius")
        color = cmap(norm(float(eta)))
        x = sub["radius"].to_numpy(dtype=np.float64)
        y = sub["d_phi_energy_raw_dd_mean"].to_numpy(dtype=np.float64)
        yerr = sub["d_phi_energy_raw_dd_sem"].to_numpy(dtype=np.float64)
        ax_left.plot(x, y, lw=2.0, marker="o", ms=3.0, color=color)
        if np.isfinite(yerr).any():
            ax_left.fill_between(x, y - 1.96 * yerr, y + 1.96 * yerr, color=color, alpha=0.11, linewidth=0)
    ax_left.axhline(0.0, color="0.25", lw=0.8)
    ax_left.set_xlabel("radius d")
    ax_left.set_ylabel(r"$d\phi_E/dd$")
    ax_left.set_title("label-flip eta: small-d derivative")
    ax_left.grid(True, color="0.88", linewidth=0.7)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_left, fraction=0.045, pad=0.025)
    cbar.set_label("flip eta")

    metric = metrics.sort_values("eta")
    x_eta = metric["eta"].to_numpy(dtype=np.float64)
    y_mass = metric["positive_curvature_mass_mean"].to_numpy(dtype=np.float64)
    y_mass_sem = metric["positive_curvature_mass_sem"].fillna(0.0).to_numpy(dtype=np.float64)
    ax_right.plot(x_eta, y_mass, color="0.18", lw=1.7, zorder=1)
    ax_right.errorbar(x_eta, y_mass, yerr=1.96 * y_mass_sem, fmt="none", ecolor="0.35", elinewidth=1.0, capsize=3, zorder=2)
    ax_right.scatter(x_eta, y_mass, c=x_eta, cmap=cmap, norm=norm, s=42, edgecolor="white", linewidth=0.6, zorder=3)
    ax_right.set_xlabel("flip eta")
    ax_right.set_ylabel(r"$A_\kappa$")
    ax_right.set_title("positive curvature mass")
    ax_right.grid(True, color="0.88", linewidth=0.7)

    if inset_eta is None:
        row = metric.sort_values("positive_curvature_mass_mean", ascending=False).iloc[0]
        inset_eta = float(row["eta"])
    eta_choice = float(x_eta[np.argmin(np.abs(x_eta - float(inset_eta)))])
    sub = curve[np.isclose(curve["eta"], eta_choice)].sort_values("radius")
    inset = ax_right.inset_axes([0.46, 0.52, 0.48, 0.40])
    ix = sub["radius"].to_numpy(dtype=np.float64)
    iy = sub["d2_phi_energy_raw_dd2_mean"].to_numpy(dtype=np.float64)
    color = cmap(norm(eta_choice))
    inset.plot(ix, iy, color=color, lw=1.4)
    inset.fill_between(ix, 0.0, iy, where=iy > 0.0, color=color, alpha=0.26, interpolate=True)
    inset.axhline(0.0, color="0.25", lw=0.7)
    inset.set_xlabel("d", fontsize=7)
    inset.set_ylabel(r"$d^2\phi_E/dd^2$", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(True, color="0.9", linewidth=0.5)

    out_path = out_dir / "fig01_eta_positive_curvature_mass_composite.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--d-min", type=float, default=0.1)
    parser.add_argument("--d-max", type=float, default=1.0)
    parser.add_argument("--inset-eta", type=float, default=None)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    units = load_units(args.run_root)
    ref_metrics, curvature, eta_summary = compute_ref_curvature(units, float(args.d_min), float(args.d_max))
    curve_summary = summarize_curve(curvature)

    ref_metrics.to_csv(out_dir / "eta_positive_curvature_mass_by_ref.csv", index=False)
    curvature.to_csv(out_dir / "eta_curvature_by_eta_ref_radius.csv", index=False)
    eta_summary.to_csv(out_dir / "eta_positive_curvature_mass_by_eta.csv", index=False)
    curve_summary.to_csv(out_dir / "eta_derivative_curvature_curve_by_eta_radius.csv", index=False)

    fig_path = plot_composite(curve_summary, eta_summary, out_dir, args.inset_eta)
    write_json(
        out_dir / "run_config_resolved.json",
        {
            "run_root": args.run_root,
            "out_dir": out_dir,
            "d_min": float(args.d_min),
            "d_max": float(args.d_max),
            "inset_eta": args.inset_eta,
            "figure": fig_path,
            "n_unit_rows_loaded": int(len(units)),
            "n_ref_metric_rows": int(len(ref_metrics)),
        },
    )
    print(fig_path)
    if not eta_summary.empty:
        print(eta_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
