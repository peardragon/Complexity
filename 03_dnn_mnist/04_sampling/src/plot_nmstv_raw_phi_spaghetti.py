#!/usr/bin/env python3
"""Plot per-reference raw phi(d) curves colored by dataset NMSTV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
WINDOWS_ROOT = Path("/home/bjyong/Complexity/windows_project")
EXISTING_REF_PHI = (
    LOCAL_ROOT
    / "04_sampling/raw_outputs/refpool1024_all_radii_90ref/06_results_figures/stability_clustering/tables/raw_phi_energy_by_ref_radius.csv"
)
EXISTING_COMPLEXITY = (
    WINDOWS_ROOT
    / "02_dnn/08_mnist/runs/final/"
    "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500/"
    "02_complexity_measure/complexity_by_rule_summary.csv"
)
SYNTHETIC_RUN = LOCAL_ROOT / "04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_90ref"
SYNTHETIC_COMPLEXITY = LOCAL_ROOT / "01_dataset_gen/raw_outputs/very_low_tv_spectral_teacher_v1/02_complexity_measure/complexity_by_rule_summary.csv"
DEFAULT_OUT = LOCAL_ROOT / "04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_90ref/06_results_figures/figures/fig_nmstv_raw_phi_energy_spaghetti.png"

RULE_ORDER = [
    "very_low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
DEPRECATED_RULES = [
    "low_tv_spectral_teacher",
]
LABELS = {
    "very_low_tv_spectral_teacher": "very low tv",
    "low_tv_spectral_teacher": "low tv",
    "real_even_odd": "even odd",
    "teacher_nn": "teacher nn",
    "random_label": "random",
}


def load_existing() -> pd.DataFrame:
    ref = pd.read_csv(EXISTING_REF_PHI)
    ref = ref.rename(columns={"phi_energy": "phi_energy_raw"})
    ref = ref[["rule", "ref_id", "radius", "phi_energy_raw"]].copy()
    ref["source_run"] = "current_90ref"
    return ref


def load_synthetic(run_root: Path) -> pd.DataFrame:
    unit_path = run_root / "05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi.csv"
    if not unit_path.exists():
        raise FileNotFoundError(f"Synthetic unit phi table missing: {unit_path}")
    unit = pd.read_csv(unit_path)
    if "phi_energy_raw" not in unit.columns:
        unit["phi_energy_raw"] = pd.to_numeric(unit["logZ_inf_full"], errors="coerce") / 2461.0
    out = unit[["rule", "ref_id", "radius", "phi_energy_raw"]].copy()
    out["source_run"] = "synthetic_90ref"
    return out


def load_complexity() -> pd.DataFrame:
    existing = pd.read_csv(EXISTING_COMPLEXITY)
    synthetic = pd.read_csv(SYNTHETIC_COMPLEXITY)
    comp = pd.concat([existing, synthetic], ignore_index=True, sort=False)
    comp = comp[comp["rule"].isin(RULE_ORDER + DEPRECATED_RULES)].copy()
    order = {rule: idx for idx, rule in enumerate(RULE_ORDER + DEPRECATED_RULES)}
    comp["_plot_order"] = comp["rule"].map(order)
    comp = comp.sort_values("_plot_order").drop(columns=["_plot_order"]).reset_index(drop=True)
    return comp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot raw phi(d) spaghetti with NMSTV colorbar.")
    parser.add_argument("--synthetic-run-root", default=str(SYNTHETIC_RUN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    comp_all = load_complexity()
    comp = comp_all[comp_all["rule"].astype(str).isin(RULE_ORDER)].copy()
    deprecated_comp = comp_all[comp_all["rule"].astype(str).isin(DEPRECATED_RULES)].copy()
    ref = pd.concat([load_existing(), load_synthetic(Path(args.synthetic_run_root))], ignore_index=True, sort=False)
    ref = ref[ref["rule"].isin(RULE_ORDER)].copy()
    ref = ref.merge(comp[["rule", "nmstv_mean", "tv_mean"]], on="rule", how="left")
    ref["rule"] = pd.Categorical(ref["rule"], RULE_ORDER, ordered=True)
    ref = ref.sort_values(["rule", "ref_id", "radius"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table_dir = out.parent.parent / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    ref.to_csv(table_dir / "nmstv_raw_phi_energy_by_ref_radius.csv", index=False)

    comp_out = comp.copy()
    comp_out["label"] = comp_out["rule"].map(LABELS)
    comp_out.to_csv(table_dir / "nmstv_values_for_raw_phi_plot.csv", index=False)
    deprecated_comp.to_csv(table_dir / "deprecated_rules_excluded_from_figures.csv", index=False)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(comp["nmstv_mean"].min()), float(comp["nmstv_mean"].max()))
    fig = plt.figure(figsize=(13.2, 7.2))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[24, 2.8], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    inline_labels = []
    for rule in RULE_ORDER:
        sub = ref[ref["rule"].astype(str).eq(rule)].copy()
        if sub.empty:
            continue
        color = cmap(norm(float(sub["nmstv_mean"].iloc[0])))
        for _ref_id, rsub in sub.groupby("ref_id", sort=False):
            rsub = rsub.sort_values("radius")
            ax.plot(rsub["radius"], rsub["phi_energy_raw"], color=color, linewidth=0.55, alpha=0.13)
        mean = sub.groupby("radius", as_index=False)["phi_energy_raw"].mean().sort_values("radius")
        ax.plot(mean["radius"], mean["phi_energy_raw"], color=color, linewidth=2.4, alpha=0.98)
        inline_labels.append(
            {
                "x": float(mean["radius"].max()),
                "y": float(mean["phi_energy_raw"].iloc[-1]),
                "label": LABELS.get(rule, rule),
                "color": color,
            }
        )

    if inline_labels:
        ordered = sorted(inline_labels, key=lambda item: item["y"])
        min_gap = 0.007
        placed = []
        for item in ordered:
            y_label = item["y"] if not placed else max(item["y"], placed[-1]["y_label"] + min_gap)
            placed.append({**item, "y_label": y_label})
        top_limit = max(item["y"] for item in inline_labels) + 0.006
        overflow = placed[-1]["y_label"] - top_limit
        if overflow > 0:
            for item in placed:
                item["y_label"] -= overflow
        for item in placed:
            if abs(item["y_label"] - item["y"]) > 1.0e-4:
                ax.plot(
                    [item["x"] - 0.02, item["x"] + 0.02],
                    [item["y"], item["y_label"]],
                    color=item["color"],
                    linewidth=0.65,
                    alpha=0.55,
                    clip_on=False,
                )
            ax.text(
                item["x"] + 0.025,
                item["y_label"],
                item["label"],
                color=item["color"],
                fontsize=9,
                va="center",
                clip_on=False,
            )

    ax.set_xlabel("radius d")
    ax.set_ylabel("phi(d) energy = logZ_inf_full / P")
    ax.set_title("Per-reference raw phi(d) energy curves; color encodes dataset NMSTV")
    ax.grid(True, linewidth=0.35, alpha=0.23)
    ax.set_xlim(0.08, 2.78)

    gradient = np.linspace(float(comp["nmstv_mean"].min()), float(comp["nmstv_mean"].max()), 256)[:, None]
    cax.imshow(gradient, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    cax.set_xticks([])
    ticks = np.linspace(0, 255, 5)
    tick_values = np.linspace(float(comp["nmstv_mean"].min()), float(comp["nmstv_mean"].max()), 5)
    cax.set_yticks(ticks)
    cax.set_yticklabels([f"{v:.3f}" for v in tick_values])
    cax.set_title("NMSTV", fontsize=10)
    for _, row in comp_out.iterrows():
        value = float(row["nmstv_mean"])
        y = (value - norm.vmin) / max(norm.vmax - norm.vmin, 1.0e-12) * 255.0
        cax.plot([0.08, 0.92], [y, y], color="white", linewidth=2.0, solid_capstyle="round")
        cax.plot([0.08, 0.92], [y, y], color="black", linewidth=0.55, solid_capstyle="round")
        cax.text(1.12, y, f"{row['label']} {value:.3f}", va="center", fontsize=8.5, transform=cax.transData)
    cax.set_ylim(0, 255)

    fig.suptitle("Raw phi(d) energy vs rule complexity after adding very-low-TV synthetic labels", y=0.985)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
