#!/usr/bin/env python3
"""Combined advanced-rule and eta-flip raw phi_E(d) spaghetti plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
DEFAULT_ADV_RUN = LOCAL_ROOT / "04_sampling" / "raw_outputs" / "very_low_tv_spectral_teacher_refpool1024_advanced_90ref"
DEFAULT_ETA_RUN = (
    LOCAL_ROOT
    / "06_eta_flip_phase_transition"
    / "raw_outputs"
    / "eta_reference_phi_advanced_4eta_90ref_r0p1_to_2p5_step0p05_n1024_cpu35_gpu0"
)
DEFAULT_OUT_DIR = LOCAL_ROOT / "06_eta_flip_phase_transition" / "figures" / "combined_advanced_eta_phi_energy_90ref"
DEFAULT_GRAPH_RUN = LOCAL_ROOT / "06_eta_flip_phase_transition" / "raw_outputs" / "eta_sweep_pilot_cpu35_gpu0"

ADV_LABELS = {
    "very_low_tv_spectral_teacher": "adv very low tv",
    "real_even_odd": "adv even odd",
    "teacher_nn": "adv teacher nn",
    "random_label": "adv random",
}
ADV_NMSTV = {
    "very_low_tv_spectral_teacher": 0.3245703473792008,
    "real_even_odd": 0.4932864276461805,
    "teacher_nn": 0.6843772639598127,
    "random_label": 0.985558573825462,
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def eta_nmstv_map(graph_run: Path, etas: list[float]) -> dict[float, float]:
    path = graph_run / "summary_by_eta_k.csv"
    out: dict[float, float] = {}
    if path.exists():
        graph = pd.read_csv(path)
        graph = graph[pd.to_numeric(graph["k"], errors="coerce").eq(3)].copy()
        graph["eta"] = pd.to_numeric(graph["eta"], errors="coerce")
        graph["knn_nmstv_mean"] = pd.to_numeric(graph["knn_nmstv_mean"], errors="coerce")
        graph = graph.dropna(subset=["eta", "knn_nmstv_mean"]).sort_values("eta")
        x = graph["eta"].to_numpy(dtype=np.float64)
        y = graph["knn_nmstv_mean"].to_numpy(dtype=np.float64)
        if len(x) > 0:
            for eta in etas:
                value = float(eta)
                if np.isclose(x, value).any():
                    out[value] = float(y[np.argmin(np.abs(x - value))])
                else:
                    out[value] = float(np.interp(value, x, y))
    fallback = {0.25: 0.811, 0.30: 0.884, 0.35: 0.934, 0.40: 0.969}
    for eta in etas:
        out.setdefault(float(eta), float(fallback.get(round(float(eta), 2), float(eta))))
    return out


def load_advanced(run_root: Path, d_min: float, d_max: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(run_root / "06_results_figures" / "phi_energy_by_rule_radius.csv")
    summary = summary.rename(columns={"phi_energy_raw": "phi_energy_raw_mean"})
    summary["source"] = "advanced"
    summary["group"] = summary["rule"].astype(str)
    summary["label"] = summary["rule"].map(lambda r: ADV_LABELS.get(str(r), f"adv {r}"))
    summary["nmstv"] = summary["rule"].map(lambda r: ADV_NMSTV.get(str(r), np.nan))

    unit_path = run_root / "06_results_figures" / "tables" / "advanced_phi_energy_by_ref_radius.csv"
    if not unit_path.exists():
        unit_path = run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    units = pd.read_csv(unit_path)
    units = units.rename(columns={"rule": "group"})
    units["source"] = "advanced"
    units["label"] = units["group"].map(lambda r: ADV_LABELS.get(str(r), f"adv {r}"))
    units["nmstv"] = units["group"].map(lambda r: ADV_NMSTV.get(str(r), np.nan))
    units["ref_key"] = units["source"].astype(str) + ":" + units["group"].astype(str) + ":" + units["ref_id"].astype(str)

    for df in (summary, units):
        df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
        df["phi_energy_raw_mean" if "phi_energy_raw_mean" in df.columns else "phi_energy_raw"] = pd.to_numeric(
            df["phi_energy_raw_mean" if "phi_energy_raw_mean" in df.columns else "phi_energy_raw"],
            errors="coerce",
        )
        df.drop(df[(df["radius"] < d_min - 1.0e-9) | (df["radius"] > d_max + 1.0e-9)].index, inplace=True)
    return summary, units


def load_eta(run_root: Path, graph_run: Path, d_min: float, d_max: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = run_root / "06_results_figures" / "eta_reference_phi_by_eta_radius.csv"
    unit_path = run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    if not summary_path.exists() or not unit_path.exists():
        raise FileNotFoundError(f"eta run is not fully aggregated yet: {run_root}")
    summary = pd.read_csv(summary_path)
    summary["eta"] = pd.to_numeric(summary["eta"], errors="coerce")
    etas = sorted(summary["eta"].dropna().unique().tolist())
    nmstv = eta_nmstv_map(graph_run, etas)
    summary["source"] = "flip"
    summary["group"] = summary["eta"].map(lambda eta: f"eta_{float(eta):.2f}")
    summary["label"] = summary["eta"].map(lambda eta: f"flip eta {float(eta):.2f}")
    summary["nmstv"] = summary["eta"].map(lambda eta: nmstv[float(eta)])

    units = pd.read_csv(unit_path)
    units["eta"] = pd.to_numeric(units["eta"], errors="coerce")
    units["source"] = "flip"
    units["group"] = units["eta"].map(lambda eta: f"eta_{float(eta):.2f}")
    units["label"] = units["eta"].map(lambda eta: f"flip eta {float(eta):.2f}")
    units["nmstv"] = units["eta"].map(lambda eta: nmstv[float(eta)])
    units["ref_key"] = units["source"].astype(str) + ":" + units["group"].astype(str) + ":" + units["ref_id"].astype(str)

    for df in (summary, units):
        df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
        value_col = "phi_energy_raw_mean" if "phi_energy_raw_mean" in df.columns else "phi_energy_raw"
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df.drop(df[(df["radius"] < d_min - 1.0e-9) | (df["radius"] > d_max + 1.0e-9)].index, inplace=True)
    return summary, units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advanced-run-root", type=Path, default=DEFAULT_ADV_RUN)
    parser.add_argument("--eta-run-root", type=Path, default=DEFAULT_ETA_RUN)
    parser.add_argument("--graph-run-root", type=Path, default=DEFAULT_GRAPH_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--d-min", type=float, default=0.1)
    parser.add_argument("--d-max", type=float, default=2.5)
    args = parser.parse_args()

    adv_summary, adv_units = load_advanced(args.advanced_run_root, args.d_min, args.d_max)
    eta_summary, eta_units = load_eta(args.eta_run_root, args.graph_run_root, args.d_min, args.d_max)
    summary = pd.concat(
        [
            adv_summary[["source", "group", "label", "nmstv", "radius", "phi_energy_raw_mean", "phi_energy_raw_sem", "n_units"]],
            eta_summary[["source", "group", "label", "nmstv", "radius", "phi_energy_raw_mean", "phi_energy_raw_sem", "n_units"]],
        ],
        ignore_index=True,
    )
    units = pd.concat(
        [
            adv_units[["source", "group", "label", "nmstv", "ref_key", "radius", "phi_energy_raw"]],
            eta_units[["source", "group", "label", "nmstv", "ref_key", "radius", "phi_energy_raw"]],
        ],
        ignore_index=True,
    )
    summary = summary.sort_values(["nmstv", "source", "group", "radius"]).reset_index(drop=True)
    units = units.sort_values(["nmstv", "source", "group", "ref_key", "radius"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "combined_phi_energy_by_group_radius.csv", index=False)
    units.to_csv(args.out_dir / "combined_phi_energy_by_ref_radius.csv", index=False)

    nmstv_min = float(summary["nmstv"].min())
    nmstv_max = float(summary["nmstv"].max())
    norm = plt.Normalize(nmstv_min, nmstv_max)
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(15.8, 7.4), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[18.0, 1.15, 3.3], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lax = fig.add_subplot(gs[0, 2])

    for (_, group), sub in units.groupby(["source", "group"], sort=False):
        color = cmap(norm(float(sub["nmstv"].iloc[0])))
        linestyle = "-" if sub["source"].iloc[0] == "advanced" else "--"
        for _, sub_ref in sub.groupby("ref_key", sort=False):
            sub_ref = sub_ref.sort_values("radius")
            ax.plot(sub_ref["radius"], sub_ref["phi_energy_raw"], color=color, lw=0.45, alpha=0.055, ls=linestyle)

    label_rows = []
    for (_, group), sub in summary.groupby(["source", "group"], sort=False):
        sub = sub.sort_values("radius")
        color = cmap(norm(float(sub["nmstv"].iloc[0])))
        linestyle = "-" if sub["source"].iloc[0] == "advanced" else "--"
        ax.plot(sub["radius"], sub["phi_energy_raw_mean"], color=color, lw=2.7, ls=linestyle)
        last = sub.iloc[-1]
        label_rows.append(
            {
                "label": str(last["label"]),
                "source": str(last["source"]),
                "x": float(last["radius"]) + 0.035,
                "y": float(last["phi_energy_raw_mean"]),
                "color": color,
            }
        )

    label_rows = sorted(label_rows, key=lambda row: row["y"])
    min_gap = 0.0045
    prev = None
    for row in label_rows:
        if prev is not None and row["y"] - prev < min_gap:
            row["y"] = prev + min_gap
        prev = row["y"]
    for row in label_rows:
        ax.text(row["x"], row["y"], row["label"], color=row["color"], va="center", fontsize=9.2)

    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xlim(args.d_min - 0.02, args.d_max + 0.46)
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$\phi_E(d) = \log Z_{\infty,\mathrm{full}} / P$")
    ref_count = int(summary["n_units"].max()) if len(summary) else 0
    ax.set_title(rf"Advanced rules and label-flip eta: raw $\phi_E(d)$ at {ref_count} references")
    ax.grid(True, color="0.91", linewidth=0.65)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("NMSTV")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("right")

    groups = summary[["source", "group", "label", "nmstv"]].drop_duplicates().sort_values(["nmstv", "source"]).reset_index(drop=True)
    tick_values = groups["nmstv"].to_numpy(dtype=float)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.3f}" for v in tick_values])
    cbar.ax.tick_params(labelsize=7.5, pad=3)
    pad = 0.035 * max(nmstv_max - nmstv_min, 1.0e-9)
    lax.set_xlim(0.0, 1.0)
    lax.set_ylim(nmstv_min - pad, nmstv_max + pad)
    lax.axis("off")
    for _, row in groups.iterrows():
        y = float(row["nmstv"])
        color = cmap(norm(y))
        cbar.ax.hlines(y, 0.38, 0.92, color="white", lw=1.1, alpha=0.90)
        lax.hlines(y, 0.0, 0.18, color="0.35", lw=0.75)
        style = "solid" if row["source"] == "advanced" else "dashed"
        lax.plot([0.24, 0.40], [y, y], color=color, lw=2.2, ls="-" if style == "solid" else "--")
        lax.text(0.44, y, f"{row['label']} {y:.3f}", va="center", ha="left", fontsize=8.4, color="0.12")

    out_path = args.out_dir / "fig01_combined_advanced_eta_phi_energy_spaghetti.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
