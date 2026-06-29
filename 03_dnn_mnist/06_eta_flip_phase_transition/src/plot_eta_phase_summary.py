#!/usr/bin/env python3
"""Summarize eta graph complexity and eta-specific phi smoke into phase-style figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_RUN = STAGE_ROOT / "raw_outputs" / "eta_sweep_pilot_cpu35_gpu0"
DEFAULT_PHI_RUN = STAGE_ROOT / "raw_outputs" / "eta_reference_phi_4eta_3ref_d1_n128_cpu35_gpu0"
DEFAULT_OUT_DIR = STAGE_ROOT / "figures" / "eta_phase_summary_cpu35_gpu0"


def crossing_eta(df: pd.DataFrame, x_col: str, y_col: str, threshold: float, direction: str) -> float:
    df = df.sort_values(x_col)
    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    if direction == "above":
        mask = y >= threshold
    else:
        mask = y <= threshold
    if not mask.any():
        return float("nan")
    idx = int(np.argmax(mask))
    if idx == 0:
        return float(x[idx])
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    if abs(y1 - y0) < 1.0e-12:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-run", type=Path, default=DEFAULT_GRAPH_RUN)
    parser.add_argument("--phi-run", type=Path, default=DEFAULT_PHI_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph = pd.read_csv(args.graph_run / "summary_by_eta_k.csv")
    phi = pd.read_csv(args.phi_run / "06_results_figures" / "eta_reference_phi_by_eta_radius.csv")
    dphi = pd.read_csv(args.phi_run / "06_results_figures" / "eta_reference_dphi_dd_by_eta_radius.csv")

    k3 = graph[graph["k"].eq(3)].copy().sort_values("eta")
    phi_d1 = phi[np.isclose(phi["radius"], 1.0)].copy().sort_values("eta")
    if "d_phi_energy_raw_dd_unit_mean" in phi_d1.columns:
        dphi_d1 = phi_d1[
            [
                "eta",
                "d_phi_energy_raw_dd_unit_mean",
                "d_delta_phi_energy_dd_unit_mean",
                "d_phi_energy_raw_dd_unit_sem",
                "d_delta_phi_energy_dd_unit_sem",
            ]
        ].rename(
            columns={
                "d_phi_energy_raw_dd_unit_mean": "d_phi_energy_raw_dd",
                "d_delta_phi_energy_dd_unit_mean": "d_delta_phi_energy_dd",
                "d_phi_energy_raw_dd_unit_sem": "d_phi_energy_raw_dd_sem",
                "d_delta_phi_energy_dd_unit_sem": "d_delta_phi_energy_dd_sem",
            }
        )
    else:
        dphi_d1 = dphi[np.isclose(dphi["radius"], 1.0)].copy().sort_values("eta")
        dphi_d1 = dphi_d1[["eta", "d_phi_energy_raw_dd", "d_delta_phi_energy_dd"]]
    merged = (
        k3[["eta", "knn_nmstv_mean", "knn_nmstv_sem", "cut_fraction_mean"]]
        .merge(
            phi_d1[
                [
                    "eta",
                    "n_units",
                    "phi_energy_raw_mean",
                    "phi_energy_raw_sem",
                    "delta_phi_energy_mean",
                    "delta_phi_energy_sem",
                    "weighted_ce_mean",
                    "weighted_error_mean",
                    "split_logZ_per_P_diff_max",
                ]
            ],
            on="eta",
            how="inner",
        )
        .merge(dphi_d1, on="eta", how="inner")
    )
    base_eta = float(merged.sort_values("eta").iloc[0]["eta"])
    base_dphi = float(merged.sort_values("eta").iloc[0]["d_phi_energy_raw_dd"])
    merged["dphi_uplift_from_eta0"] = merged["d_phi_energy_raw_dd"] - base_dphi
    merged["dphi_uplift_positive"] = np.maximum(merged["dphi_uplift_from_eta0"], 0.0)
    merged.to_csv(args.out_dir / "eta_phase_summary_table.csv", index=False)

    eta_nmstv_09 = crossing_eta(k3, "eta", "knn_nmstv_mean", 0.90, "above")
    eta_dphi_flat = crossing_eta(merged, "eta", "d_phi_energy_raw_dd", -0.02, "above")
    eta_phi_saturation = crossing_eta(
        merged,
        "eta",
        "phi_energy_raw_mean",
        float(merged["phi_energy_raw_mean"].min() + 0.05 * (merged["phi_energy_raw_mean"].max() - merged["phi_energy_raw_mean"].min())),
        "below",
    )
    landmarks = pd.DataFrame(
        [
            {"landmark": "k3_nmstv_cross_0p90", "eta": eta_nmstv_09},
            {"landmark": "dphi_d1_cross_minus_0p02", "eta": eta_dphi_flat},
            {"landmark": "phi_d1_95pct_saturation", "eta": eta_phi_saturation},
        ]
    )
    landmarks.to_csv(args.out_dir / "eta_phase_landmarks.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=190)
    ax = axes[0]
    ax.plot(k3["eta"], k3["knn_nmstv_mean"], marker="o", lw=1.9, color="#2451a6")
    ax.fill_between(
        k3["eta"].to_numpy(),
        (k3["knn_nmstv_mean"] - 1.96 * k3["knn_nmstv_sem"]).to_numpy(),
        (k3["knn_nmstv_mean"] + 1.96 * k3["knn_nmstv_sem"]).to_numpy(),
        color="#2451a6",
        alpha=0.14,
        linewidth=0,
    )
    ax.axhline(0.90, color="0.25", ls=":", lw=1.1)
    ax.axvline(eta_nmstv_09, color="#8b2c2c", ls="--", lw=1.1)
    ax.set_xlabel("eta")
    ax.set_ylabel("k=3 NMSTV")
    ax.set_title("label graph complexity")
    ax.grid(True, color="0.88", linewidth=0.7)

    ax = axes[1]
    ax.errorbar(
        merged["eta"],
        merged["phi_energy_raw_mean"],
        yerr=1.96 * merged["phi_energy_raw_sem"],
        marker="o",
        lw=1.9,
        capsize=3,
        color="#00857a",
    )
    ax.axvline(eta_phi_saturation, color="#8b2c2c", ls="--", lw=1.1)
    ax.set_xlabel("eta")
    ax.set_ylabel("phi(d=1) energy raw")
    ax.set_title("eta-specific refs, d=1")
    ax.grid(True, color="0.88", linewidth=0.7)

    ax = axes[2]
    ax.plot(merged["eta"], merged["d_phi_energy_raw_dd"], marker="o", lw=1.9, color="#d27a00", label="d phi/dd at d=1")
    if "d_phi_energy_raw_dd_sem" in merged.columns:
        ax.fill_between(
            merged["eta"].to_numpy(),
            (merged["d_phi_energy_raw_dd"] - 1.96 * merged["d_phi_energy_raw_dd_sem"]).to_numpy(),
            (merged["d_phi_energy_raw_dd"] + 1.96 * merged["d_phi_energy_raw_dd_sem"]).to_numpy(),
            color="#d27a00",
            alpha=0.13,
            linewidth=0,
        )
    ax.plot(
        merged["eta"],
        merged["dphi_uplift_positive"],
        marker="s",
        lw=1.5,
        color="#7a5195",
        label=f"positive uplift vs eta={base_eta:.2f}",
    )
    ax.axhline(0.0, color="0.25", lw=0.9)
    ax.axhline(-0.02, color="0.25", ls=":", lw=1.1)
    ax.axvline(eta_dphi_flat, color="#8b2c2c", ls="--", lw=1.1)
    ax.set_xlabel("eta")
    ax.set_ylabel("first-derivative signal")
    ax.set_title("phase-like flattening")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Eta label-noise phase summary: graph complexity and phi(d) near 1", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig01_eta_phase_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=190)
    ax.plot(merged["knn_nmstv_mean"], merged["d_phi_energy_raw_dd"], marker="o", lw=1.9, color="#2451a6")
    for _, row in merged.iterrows():
        ax.annotate(f"{row['eta']:.2f}", (row["knn_nmstv_mean"], row["d_phi_energy_raw_dd"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.0, color="0.25", lw=0.9)
    ax.axhline(-0.02, color="0.25", ls=":", lw=1.0)
    ax.axvline(0.90, color="0.25", ls=":", lw=1.0)
    ax.set_xlabel("k=3 NMSTV")
    ax.set_ylabel("d phi/dd at d=1")
    ax.set_title("Complexity proxy vs phi first derivative")
    ax.grid(True, color="0.88", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig02_nmstv_vs_dphi_phase_plane.png")
    plt.close(fig)

    print(landmarks.to_string(index=False))


if __name__ == "__main__":
    main()
