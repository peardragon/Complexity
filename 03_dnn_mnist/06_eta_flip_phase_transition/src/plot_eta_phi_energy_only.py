#!/usr/bin/env python3
"""Plot eta label-flip phi_E(d) curves only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = STAGE_ROOT / "raw_outputs" / "eta_reference_phi_promoted_4eta_10ref_d1_n1024_cpu35_gpu0"
DEFAULT_OUT_DIR = STAGE_ROOT / "figures" / "eta_phi_energy_small_d_n1024"


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--d-min", type=float, default=0.1)
    parser.add_argument("--d-max", type=float, default=1.0)
    args = parser.parse_args()

    unit_path = args.run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    if not unit_path.exists():
        unit_path = args.run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv"
    units = pd.read_csv(unit_path)
    for col in ["eta", "ref_id", "radius", "phi_energy_raw"]:
        units[col] = pd.to_numeric(units[col], errors="coerce")
    units = units[(units["radius"] >= args.d_min - 1.0e-9) & (units["radius"] <= args.d_max + 1.0e-9)].copy()
    units = units.dropna(subset=["eta", "ref_id", "radius", "phi_energy_raw"])

    summary = (
        units.groupby(["eta", "radius"], as_index=False)
        .agg(
            n_refs=("ref_id", "nunique"),
            phi_energy_raw_mean=("phi_energy_raw", "mean"),
            phi_energy_raw_sd=("phi_energy_raw", "std"),
            phi_energy_raw_sem=("phi_energy_raw", sem),
        )
        .sort_values(["eta", "radius"])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    units[["eta", "ref_id", "radius", "phi_energy_raw"]].sort_values(["eta", "ref_id", "radius"]).to_csv(
        args.out_dir / "eta_phi_energy_by_ref_radius.csv", index=False
    )
    summary.to_csv(args.out_dir / "eta_phi_energy_by_eta_radius.csv", index=False)

    etas = np.array(sorted(summary["eta"].unique()), dtype=float)
    norm = plt.Normalize(float(etas.min()), float(etas.max()))
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(10.8, 5.9), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[18.0, 1.0], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    for eta, sub_eta in units.groupby("eta", sort=True):
        color = cmap(norm(float(eta)))
        for _, sub_ref in sub_eta.groupby("ref_id", sort=True):
            sub_ref = sub_ref.sort_values("radius")
            ax.plot(sub_ref["radius"], sub_ref["phi_energy_raw"], color=color, lw=0.7, alpha=0.16)

    label_rows = []
    for eta, sub in summary.groupby("eta", sort=True):
        sub = sub.sort_values("radius")
        color = cmap(norm(float(eta)))
        ax.plot(sub["radius"], sub["phi_energy_raw_mean"], color=color, lw=3.2)
        ax.fill_between(
            sub["radius"].to_numpy(),
            (sub["phi_energy_raw_mean"] - 1.96 * sub["phi_energy_raw_sem"]).to_numpy(),
            (sub["phi_energy_raw_mean"] + 1.96 * sub["phi_energy_raw_sem"]).to_numpy(),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        last = sub.sort_values("radius").iloc[-1]
        label_rows.append(
            {
                "eta": float(eta),
                "x": float(last["radius"]) + 0.012 * (args.d_max - args.d_min),
                "y": float(last["phi_energy_raw_mean"]),
                "color": color,
            }
        )

    label_rows = sorted(label_rows, key=lambda row: row["y"])
    min_gap = 0.0042
    previous_y = None
    for row in label_rows:
        if previous_y is not None and row["y"] - previous_y < min_gap:
            row["y"] = previous_y + min_gap
        previous_y = row["y"]
    for row in label_rows:
        ax.text(row["x"], row["y"], f"eta {row['eta']:.2f}", color=row["color"], va="center", fontsize=10.5)

    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xlim(args.d_min - 0.02, args.d_max + 0.10 * (args.d_max - args.d_min))
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$\phi_E(d) = \log Z_{\infty,\mathrm{full}} / P$")
    ax.set_title(r"Label-flip eta: raw $\phi_E(d)$ energy")
    ax.grid(True, color="0.90", linewidth=0.7)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("flip eta")
    cbar.set_ticks(etas)
    cbar.set_ticklabels([f"{eta:.2f}" for eta in etas])

    out_path = args.out_dir / "fig01_eta_phi_energy_only.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
