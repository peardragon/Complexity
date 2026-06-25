#!/usr/bin/env python3
"""Apply a high-loss gate to QC-passed MNIST phi selections and redraw plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
OUT = LOCAL_ROOT / "05_proxy_local_entropy" / "raw_outputs" / "qc_loss_gated_phi_dmax0p65"

INPUTS = {
    "current": LOCAL_ROOT / "05_proxy_local_entropy" / "raw_outputs" / "qc_filled_phi_dmax0p65" / "filled_unit_selection.csv",
    "strict4096": LOCAL_ROOT / "05_proxy_local_entropy" / "raw_outputs" / "qc_filled_phi_dmax0p65_strict" / "filled_unit_selection.csv",
}

RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
LABELS = {
    "low_tv_spectral_teacher": "low_tv",
    "real_even_odd": "even_odd",
    "teacher_nn": "teacher_nn",
    "random_label": "random",
}
COLORS = {
    "low_tv_spectral_teacher": "#2f6b9a",
    "real_even_odd": "#4c8c4a",
    "teacher_nn": "#b0782d",
    "random_label": "#9a3b58",
}

RADII = [
    0.010,
    0.011,
    0.012,
    0.013,
    0.014,
    0.016,
    0.018,
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,
    0.065,
    0.080,
    0.120,
    0.150,
    0.200,
    0.300,
    0.450,
    0.650,
]
CE_GATE = 1.0
ERROR_GATE = 0.3
PHI_RAW_GATE = -0.3


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    rows = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def add_loss_gate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "radius",
        "ref_id",
        "weighted_ce",
        "weighted_error",
        "phi_energy_raw",
        "delta_phi_energy",
        "delta_phi_full",
        "split_logZ_per_P_diff",
        "ess_fraction",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["loss_gate_pass"] = (
        out["weighted_ce"].lt(CE_GATE)
        & out["weighted_error"].lt(ERROR_GATE)
        & out["phi_energy_raw"].gt(PHI_RAW_GATE)
    )
    out["loss_gate_drop"] = ~out["loss_gate_pass"]
    return out.sort_values(["rule", "ref_id", "radius"]).reset_index(drop=True)


def summarize_rule_radius(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        for radius in RADII:
            sub = df[df["rule"].eq(rule) & np.isclose(df["radius"], float(radius))]
            gated = sub[sub["loss_gate_pass"]]
            rows.append(
                {
                    "policy": policy,
                    "rule": rule,
                    "radius": float(radius),
                    "qc_selected_count": int(len(sub)),
                    "loss_gate_pass_count": int(len(gated)),
                    "loss_gate_drop_count": int(len(sub) - len(gated)),
                    "loss_gate_drop_fraction": float((len(sub) - len(gated)) / len(sub)) if len(sub) else np.nan,
                    "mean_delta_phi_energy_qc": float(sub["delta_phi_energy"].mean()) if len(sub) else np.nan,
                    "mean_delta_phi_energy_gated": float(gated["delta_phi_energy"].mean()) if len(gated) else np.nan,
                    "mean_delta_phi_energy_shift": (
                        float(gated["delta_phi_energy"].mean() - sub["delta_phi_energy"].mean())
                        if len(gated) and len(sub)
                        else np.nan
                    ),
                    "mean_delta_phi_full_qc": float(sub["delta_phi_full"].mean()) if len(sub) else np.nan,
                    "mean_delta_phi_full_gated": float(gated["delta_phi_full"].mean()) if len(gated) else np.nan,
                    "mean_phi_energy_raw_qc": float(sub["phi_energy_raw"].mean()) if len(sub) else np.nan,
                    "mean_phi_energy_raw_gated": float(gated["phi_energy_raw"].mean()) if len(gated) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        sub = df[df["rule"].eq(rule)]
        traj = (
            sub.groupby(["rule", "ref_id"], as_index=False)
            .agg(
                qc_points=("radius", "size"),
                dropped_points=("loss_gate_drop", "sum"),
                gate_pass_points=("loss_gate_pass", "sum"),
            )
        )
        rows.append(
            {
                "policy": policy,
                "rule": rule,
                "qc_selected_rows": int(len(sub)),
                "loss_gate_dropped_rows": int(sub["loss_gate_drop"].sum()),
                "row_drop_fraction": float(sub["loss_gate_drop"].mean()) if len(sub) else np.nan,
                "trajectories": int(len(traj)),
                "trajectories_with_any_drop": int((traj["dropped_points"] > 0).sum()),
                "trajectory_drop_fraction": float((traj["dropped_points"] > 0).mean()) if len(traj) else np.nan,
                "trajectories_fully_preserved": int((traj["dropped_points"] == 0).sum()),
            }
        )
    total_traj = (
        df.groupby(["rule", "ref_id"], as_index=False)
        .agg(dropped_points=("loss_gate_drop", "sum"))
    )
    rows.append(
        {
            "policy": policy,
            "rule": "ALL",
            "qc_selected_rows": int(len(df)),
            "loss_gate_dropped_rows": int(df["loss_gate_drop"].sum()),
            "row_drop_fraction": float(df["loss_gate_drop"].mean()),
            "trajectories": int(len(total_traj)),
            "trajectories_with_any_drop": int((total_traj["dropped_points"] > 0).sum()),
            "trajectory_drop_fraction": float((total_traj["dropped_points"] > 0).mean()),
            "trajectories_fully_preserved": int((total_traj["dropped_points"] == 0).sum()),
        }
    )
    return pd.DataFrame(rows)


def plot_before_after(summary: pd.DataFrame, policy: str, value: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for rule in RULES:
        sub = summary[summary["rule"].eq(rule)].sort_values("radius")
        ax.plot(
            sub["radius"],
            sub[f"{value}_qc"],
            color=COLORS[rule],
            lw=1.0,
            ls="--",
            alpha=0.35,
        )
        ax.plot(
            sub["radius"],
            sub[f"{value}_gated"],
            color=COLORS[rule],
            lw=2.0,
            marker="o",
            ms=3.2,
            label=LABELS[rule],
        )
    ax.axhline(0.0, color="#666666", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("radius d")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{policy}: QC-pass mean dashed vs loss-gated mean solid")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, which="both", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_count_grid(summary: pd.DataFrame, policy: str, path: Path) -> None:
    pivot_pass = (
        summary.pivot(index="rule", columns="radius", values="loss_gate_pass_count")
        .reindex(RULES)
        .sort_index(axis=1)
        .fillna(0)
    )
    pivot_drop = (
        summary.pivot(index="rule", columns="radius", values="loss_gate_drop_count")
        .reindex(RULES)
        .sort_index(axis=1)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(11.4, 3.8))
    im = ax.imshow(pivot_pass.to_numpy(dtype=float), vmin=0, vmax=30, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(pivot_pass.index)), [LABELS[r] for r in pivot_pass.index])
    ax.set_xticks(np.arange(len(pivot_pass.columns)), [f"{x:g}" for x in pivot_pass.columns], rotation=45, ha="right")
    ax.set_xlabel("radius d")
    ax.set_title(f"{policy}: loss-gate pass count among selected QC-pass units")
    for i in range(pivot_pass.shape[0]):
        for j in range(pivot_pass.shape[1]):
            keep = int(pivot_pass.iat[i, j]) if np.isfinite(pivot_pass.iat[i, j]) else 0
            drop = int(pivot_drop.iat[i, j]) if np.isfinite(pivot_drop.iat[i, j]) else 0
            text = f"{keep}"
            if drop:
                text = f"{keep}\n(-{drop})"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if keep < 18 else "black")
    fig.colorbar(im, ax=ax, label="gate-passed units")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trajectories(df: pd.DataFrame, policy: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), sharex=True)
    axes = axes.ravel()
    for ax, rule in zip(axes, RULES):
        sub_rule = df[df["rule"].eq(rule)]
        for _, sub in sub_rule.groupby("ref_id"):
            sub = sub.sort_values("radius")
            ax.plot(sub["radius"], sub["delta_phi_energy"], color="#9a9a9a", alpha=0.16, lw=0.8)
            kept = sub[sub["loss_gate_pass"]]
            if not kept.empty:
                ax.plot(kept["radius"], kept["delta_phi_energy"], color=COLORS[rule], alpha=0.42, lw=1.0)
                ax.scatter(kept["radius"], kept["delta_phi_energy"], color=COLORS[rule], alpha=0.55, s=8)
            dropped = sub[sub["loss_gate_drop"]]
            if not dropped.empty:
                ax.scatter(dropped["radius"], dropped["delta_phi_energy"], facecolors="none", edgecolors="#8a2635", s=22, lw=0.8)
        ax.axhline(0.0, color="#666666", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{LABELS[rule]} trajectories")
        ax.set_xlabel("radius d")
        ax.set_ylabel("delta_phi_energy")
        ax.grid(True, which="both", alpha=0.22)
    fig.suptitle(f"{policy}: selected QC-pass trajectories, colored portions pass loss gate")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_policy = []
    all_rule_radius = []
    report_lines = [
        "# QC-Pass Phi With High-Loss Gate",
        "",
        f"Loss gate: `weighted_ce < {CE_GATE}`, `weighted_error < {ERROR_GATE}`, `phi_energy_raw > {PHI_RAW_GATE}`.",
        "Dashed lines are the original selected QC-pass mean; solid lines use only rows passing the loss gate.",
        "",
    ]

    for policy, input_path in INPUTS.items():
        df = add_loss_gate(pd.read_csv(input_path))
        summary = summarize_rule_radius(df, policy)
        policy_summary = summarize_policy(df, policy)
        dropped = df[df["loss_gate_drop"]].copy()

        write_csv(df, OUT / f"{policy}_selection_with_loss_gate.csv")
        write_csv(summary, OUT / f"{policy}_rule_radius_loss_gate_summary.csv")
        write_csv(policy_summary, OUT / f"{policy}_loss_gate_policy_summary.csv")
        write_csv(dropped, OUT / f"{policy}_loss_gate_dropped_units.csv")

        plot_before_after(
            summary,
            policy,
            "mean_delta_phi_energy",
            "mean delta_phi_energy",
            OUT / f"fig01_{policy}_delta_phi_energy_qc_vs_loss_gated.png",
        )
        plot_before_after(
            summary,
            policy,
            "mean_delta_phi_full",
            "mean delta_phi_full",
            OUT / f"fig02_{policy}_delta_phi_full_qc_vs_loss_gated.png",
        )
        plot_before_after(
            summary,
            policy,
            "mean_phi_energy_raw",
            "mean raw phi_energy",
            OUT / f"fig03_{policy}_raw_phi_energy_qc_vs_loss_gated.png",
        )
        plot_count_grid(summary, policy, OUT / f"fig04_{policy}_loss_gate_pass_count_grid.png")
        plot_trajectories(df, policy, OUT / f"fig05_{policy}_loss_gated_trajectories.png")

        all_policy.append(policy_summary)
        all_rule_radius.append(summary)

        report_lines.extend(
            [
                f"## {policy}",
                "",
                markdown_table(policy_summary),
                "",
                "Largest upward shifts after removing high-loss rows:",
                "",
                markdown_table(
                    summary.assign(abs_shift=summary["mean_delta_phi_energy_shift"].abs())
                    .sort_values("abs_shift", ascending=False)
                    .drop(columns=["abs_shift"])
                    [
                        [
                            "rule",
                            "radius",
                            "qc_selected_count",
                            "loss_gate_pass_count",
                            "loss_gate_drop_count",
                            "mean_delta_phi_energy_qc",
                            "mean_delta_phi_energy_gated",
                            "mean_delta_phi_energy_shift",
                        ]
                    ],
                    max_rows=10,
                ),
                "",
            ]
        )

    write_csv(pd.concat(all_policy, ignore_index=True), OUT / "loss_gate_policy_summary_all.csv")
    write_csv(pd.concat(all_rule_radius, ignore_index=True), OUT / "rule_radius_loss_gate_summary_all.csv")

    report_lines.extend(
        [
            "## Files",
            "",
            "- `loss_gate_policy_summary_all.csv`",
            "- `rule_radius_loss_gate_summary_all.csv`",
            "- `*_selection_with_loss_gate.csv`",
            "- `*_loss_gate_dropped_units.csv`",
            "- `fig01_*_delta_phi_energy_qc_vs_loss_gated.png`",
            "- `fig02_*_delta_phi_full_qc_vs_loss_gated.png`",
            "- `fig03_*_raw_phi_energy_qc_vs_loss_gated.png`",
            "- `fig04_*_loss_gate_pass_count_grid.png`",
            "- `fig05_*_loss_gated_trajectories.png`",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
