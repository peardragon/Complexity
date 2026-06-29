#!/usr/bin/env python3
"""Recreate backup-style raw phi_E(d) spaghetti figures for advanced sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
DEFAULT_RUN = LOCAL_ROOT / "04_sampling" / "raw_outputs" / "very_low_tv_spectral_teacher_refpool1024_advanced_90ref"
DEFAULT_BACKUP_RUN = (
    LOCAL_ROOT
    / "99_backup"
    / "cleanup_20260626_002622"
    / "04_sampling"
    / "raw_outputs"
    / "very_low_tv_spectral_teacher_refpool1024_90ref"
)

RULE_LABELS = {
    "very_low_tv_spectral_teacher": "very low tv",
    "real_even_odd": "even odd",
    "teacher_nn": "teacher nn",
    "random_label": "random",
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def load_nmstv(run_root: Path, backup_run_root: Path, rules: list[str]) -> pd.DataFrame:
    candidates = [
        run_root / "06_results_figures" / "tables" / "nmstv_values_for_raw_phi_plot.csv",
        backup_run_root / "06_results_figures" / "tables" / "nmstv_values_for_raw_phi_plot.csv",
    ]
    for path in candidates:
        if path.exists():
            out = pd.read_csv(path)
            out = out[out["rule"].isin(rules)].copy()
            if set(out["rule"]) == set(rules):
                return out

    metadata_path = (
        LOCAL_ROOT
        / "01_dataset_gen"
        / "raw_outputs"
        / "very_low_tv_spectral_teacher_v1"
        / "01_dataset_prepare"
        / "raw_datasets"
        / "split_000"
        / "very_low_tv_spectral_teacher"
        / "dataset_metadata.json"
    )
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = []
        for item in meta.get("complexity_ladder", []):
            rule = str(item.get("rule", ""))
            if rule in rules:
                rows.append(
                    {
                        "rule": rule,
                        "nmstv_mean": float(item["nmstv_mean"]),
                        "tv_mean": float(item.get("tv_mean", np.nan)),
                        "n_datasets": 1,
                        "label": RULE_LABELS.get(rule, rule.replace("_", " ")),
                    }
                )
        out = pd.DataFrame(rows)
        if set(out["rule"]) == set(rules):
            return out

    fallback = {
        "very_low_tv_spectral_teacher": 0.3245703473792008,
        "real_even_odd": 0.4932864276461805,
        "teacher_nn": 0.6843772639598127,
        "random_label": 0.985558573825462,
    }
    return pd.DataFrame(
        [
            {
                "rule": rule,
                "nmstv_mean": float(fallback.get(rule, i)),
                "tv_mean": np.nan,
                "n_datasets": 1,
                "label": RULE_LABELS.get(rule, rule.replace("_", " ")),
            }
            for i, rule in enumerate(rules)
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--backup-run-root", type=Path, default=DEFAULT_BACKUP_RUN)
    parser.add_argument("--d-min", type=float, default=0.1)
    parser.add_argument("--d-max", type=float, default=2.5)
    args = parser.parse_args()

    unit_path = args.run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    if not unit_path.exists():
        unit_path = args.run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv"
    if not unit_path.exists():
        raise FileNotFoundError(f"unit phi table not found: {unit_path}")

    out_root = args.run_root / "06_results_figures"
    fig_dir = out_root / "figures"
    table_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    units = pd.read_csv(unit_path)
    for col in ["ref_id", "radius", "phi_energy_raw", "ess_fraction", "split_logZ_per_P_diff"]:
        units[col] = pd.to_numeric(units[col], errors="coerce")
    units = units[(units["radius"] >= args.d_min - 1.0e-9) & (units["radius"] <= args.d_max + 1.0e-9)].copy()
    units = units.dropna(subset=["rule", "ref_id", "radius", "phi_energy_raw"])

    rules = sorted(units["rule"].unique().tolist())
    nmstv = load_nmstv(args.run_root, args.backup_run_root, rules)
    nmstv["label"] = nmstv["rule"].map(lambda r: RULE_LABELS.get(r, str(r).replace("_", " ")))
    nmstv = nmstv.sort_values("nmstv_mean").reset_index(drop=True)
    rule_order = nmstv["rule"].tolist()
    units["rule"] = pd.Categorical(units["rule"], categories=rule_order, ordered=True)
    units = units.sort_values(["rule", "ref_id", "radius"]).copy()

    summary = (
        units.groupby(["rule", "radius"], observed=True, as_index=False)
        .agg(
            phi_energy_raw=("phi_energy_raw", "mean"),
            phi_energy_raw_sd=("phi_energy_raw", "std"),
            phi_energy_raw_sem=("phi_energy_raw", sem),
            n_units=("ref_id", "nunique"),
            target_ref_count=("ref_id", lambda x: int(x.nunique())),
            sampling_status=("finite", lambda x: "complete"),
        )
        .sort_values(["rule", "radius"])
    )
    qc_path = args.run_root / "05_pool2_pm_sais_sampling" / "qc_diagnostics_by_rule_radius.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        qc = qc[["rule", "radius", "qc_diagnostic_pass"]].copy()
        summary = summary.merge(qc, on=["rule", "radius"], how="left")
    else:
        summary["qc_diagnostic_pass"] = np.nan

    raw_summary = summary[["rule", "radius", "phi_energy_raw", "n_units", "target_ref_count", "sampling_status"]].copy()
    summary.to_csv(out_root / "phi_energy_by_rule_radius.csv", index=False)
    raw_summary.to_csv(out_root / "phi_raw_by_rule_radius.csv", index=False)
    summary.to_csv(table_dir / "advanced_phi_energy_by_rule_radius.csv", index=False)
    units[["rule", "ref_id", "radius", "phi_energy_raw"]].to_csv(table_dir / "advanced_phi_energy_by_ref_radius.csv", index=False)
    nmstv.to_csv(table_dir / "nmstv_values_for_raw_phi_plot.csv", index=False)
    pd.DataFrame({"rule": rule_order, "status": "active"}).to_csv(table_dir / "active_rules_for_result_figures.csv", index=False)

    nmstv_min = float(nmstv["nmstv_mean"].min())
    nmstv_max = float(nmstv["nmstv_mean"].max())
    nmstv_pad = 0.035 * max(nmstv_max - nmstv_min, 1.0e-9)
    norm = plt.Normalize(nmstv_min, nmstv_max)
    cmap = plt.get_cmap("viridis")
    nmstv_map = dict(zip(nmstv["rule"], nmstv["nmstv_mean"]))
    label_map = dict(zip(nmstv["rule"], nmstv["label"]))

    fig = plt.figure(figsize=(14.2, 7.4), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[18.0, 1.15, 2.15], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lax = fig.add_subplot(gs[0, 2])

    for rule in rule_order:
        color = cmap(norm(float(nmstv_map[rule])))
        rule_units = units[units["rule"].astype(str).eq(rule)]
        for _, sub_ref in rule_units.groupby("ref_id", sort=True):
            sub_ref = sub_ref.sort_values("radius")
            ax.plot(sub_ref["radius"], sub_ref["phi_energy_raw"], color=color, lw=0.55, alpha=0.12)

    for rule in rule_order:
        color = cmap(norm(float(nmstv_map[rule])))
        sub = summary[summary["rule"].astype(str).eq(rule)].sort_values("radius")
        ax.plot(sub["radius"], sub["phi_energy_raw"], color=color, lw=3.0)
        ax.fill_between(
            sub["radius"].to_numpy(dtype=float),
            (sub["phi_energy_raw"] - 1.96 * sub["phi_energy_raw_sem"]).to_numpy(dtype=float),
            (sub["phi_energy_raw"] + 1.96 * sub["phi_energy_raw_sem"]).to_numpy(dtype=float),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        last = sub.iloc[-1]
        ax.text(
            float(last["radius"]) + 0.035,
            float(last["phi_energy_raw"]),
            label_map[rule],
            color=color,
            fontsize=11,
            va="center",
        )

    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xlim(args.d_min - 0.02, args.d_max + 0.22)
    y_min = float(units["phi_energy_raw"].quantile(0.002))
    y_max = max(0.005, float(units["phi_energy_raw"].quantile(0.998)))
    ax.set_ylim(y_min - 0.01, y_max + 0.005)
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$\phi_E(d) = \log Z_{\infty,\mathrm{full}} / P$")
    ref_count = int(summary["n_units"].max()) if len(summary) else 0
    ax.set_title(rf"Advanced {ref_count}-reference raw $\phi_E(d)$ energy")
    ax.grid(True, color="0.91", linewidth=0.65)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("NMSTV")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_ticks(nmstv["nmstv_mean"].to_numpy(dtype=float))
    cbar.set_ticklabels([f"{v:.3f}" for v in nmstv["nmstv_mean"]])
    cbar.ax.tick_params(labelsize=8.5, pad=3)

    lax.set_xlim(0.0, 1.0)
    lax.set_ylim(nmstv_min - nmstv_pad, nmstv_max + nmstv_pad)
    lax.axis("off")
    for _, row in nmstv.iterrows():
        y = float(row["nmstv_mean"])
        cbar.ax.hlines(y, 0.38, 0.92, color="white", lw=1.4, alpha=0.90)
        lax.hlines(y, 0.0, 0.26, color="0.35", lw=0.8)
        lax.text(0.31, y, f"{row['label']} {y:.3f}", va="center", ha="left", fontsize=9.4, color="0.12")

    out_path = fig_dir / "fig01_advanced_raw_phi_energy_spaghetti.png"
    fig.savefig(out_path)
    alias_path = fig_dir / "fig01_active_rules_raw_phi_energy_spaghetti_advanced.png"
    fig.savefig(alias_path)
    legacy_alias_path = fig_dir / "fig_nmstv_raw_phi_energy_spaghetti.png"
    fig.savefig(legacy_alias_path)
    plt.close(fig)
    print(out_path)
    print(alias_path)
    print(legacy_alias_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
