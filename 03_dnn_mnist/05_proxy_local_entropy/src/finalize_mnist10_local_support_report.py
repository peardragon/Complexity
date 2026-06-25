#!/usr/bin/env python3
"""Finalize the MNIST10 local-support PM-SAIS comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(
    "/home/bjyong/Complexity/windows_project/02_dnn/08_mnist/runs/final/"
    "local_support_dmax0p65_all_rules_resampled"
)
FIG = OUT / "06_results_figures"
SAMPLE = OUT / "05_pool2_pm_sais_sampling"
PRIOR_D085 = Path(
    "/home/bjyong/Complexity/windows_project/02_dnn/08_mnist/runs/final/"
    "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500/"
    "05_pool2_pm_sais_sampling/unit_summaries/split_000"
)
OLD_LOCAL = Path(
    "/home/bjyong/Complexity/windows_project/02_dnn/08_mnist/runs/final/"
    "local_support_dmax0p65_all_rules"
)

RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
RULE_LABELS = {
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


def fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def markdown_table(df: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    cols = list(df.columns)
    rows = [
        "| " + " | ".join(cell(col) for col in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(cell(row[col]) for col in cols) + " |")
    return "\n".join(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    fail_if_exists(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_csv_if_missing(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        return
    write_csv(df, path)


def savefig(path: Path) -> None:
    fail_if_exists(path)
    tmp = path.with_name(path.name + ".tmp.png")
    plt.savefig(tmp, dpi=180, bbox_inches="tight")
    plt.close()
    tmp.replace(path)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    phi = pd.read_csv(FIG / "phi_by_rule_radius.csv")
    qc = pd.read_csv(SAMPLE / "qc_by_rule_radius.csv")
    units = pd.read_csv(SAMPLE / "shell_summary_by_unit_with_phi.csv")
    claims = pd.read_csv(FIG / "final_claim_table.csv")
    with (OUT / "QC_STATUS.json").open() as f:
        status = json.load(f)
    return phi, qc, units, claims, status


def make_reference_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        sub = units[units["rule"] == rule].copy()
        refs = sorted(int(x) for x in sub["ref_id"].dropna().unique())
        ref0 = sub.drop_duplicates("ref_id")
        rows.append(
            {
                "selector": "dense_qc_stable_ref30",
                "rule": rule,
                "selected_ref_count": 30,
                "observed_ref_count": len(refs),
                "selected_ref_ids": " ".join(f"{x:03d}" for x in refs),
                "theta_ref_norm_mean": ref0["theta_ref_norm"].mean(),
                "theta_ref_norm_min": ref0["theta_ref_norm"].min(),
                "theta_ref_norm_max": ref0["theta_ref_norm"].max(),
                "reference_prior_log_weight_mean": ref0["reference_prior_log_weight"].mean(),
                "unit_count": int(len(sub)),
                "radius_count": int(sub["radius"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def load_complexity() -> pd.DataFrame:
    path = OLD_LOCAL / "complexity_by_rule.csv"
    comp = pd.read_csv(path)
    return comp[comp["rule"].isin(RULES)].copy()


def plot_phi(df: pd.DataFrame, value_col: str, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8.8, 5.2))
    for rule in RULES:
        sub = df[df["rule"] == rule].sort_values("radius")
        claim = sub[sub["qc_pass"] == True]
        no_claim = sub[sub["qc_pass"] != True]
        plt.plot(sub["radius"], sub[value_col], color=COLORS[rule], lw=1.5, alpha=0.65)
        plt.scatter(
            claim["radius"],
            claim[value_col],
            color=COLORS[rule],
            s=30,
            label=RULE_LABELS[rule],
            zorder=3,
        )
        plt.scatter(
            no_claim["radius"],
            no_claim[value_col],
            facecolors="white",
            edgecolors=COLORS[rule],
            s=36,
            zorder=3,
        )
    plt.axhline(0.0, color="#777777", lw=0.8)
    plt.xscale("log")
    plt.xlabel("d_raw")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, ncol=2)
    plt.grid(True, which="both", alpha=0.2)
    savefig(path)


def plot_qc_grid(qc: pd.DataFrame, path: Path) -> None:
    radii = sorted(qc["radius"].unique())
    grid = np.zeros((len(RULES), len(radii)))
    for i, rule in enumerate(RULES):
        for j, radius in enumerate(radii):
            row = qc[(qc["rule"] == rule) & (qc["radius"] == radius)]
            grid[i, j] = 1.0 if len(row) and bool(row.iloc[0]["qc_pass"]) else 0.0
    plt.figure(figsize=(10, 3.7))
    plt.imshow(grid, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#d8d8d8", "#3b8b5a"]))
    plt.yticks(range(len(RULES)), [RULE_LABELS[r] for r in RULES])
    plt.xticks(range(len(radii)), [f"{r:g}" for r in radii], rotation=45, ha="right")
    plt.title("QC pass grid, d_raw <= 0.65")
    plt.xlabel("d_raw")
    plt.colorbar(ticks=[0, 1], label="QC pass")
    savefig(path)


def plot_complexity(comp: pd.DataFrame, claims: pd.DataFrame, path: Path) -> None:
    d003 = claims[np.isclose(claims["radius"], 0.03)][
        ["rule", "delta_phi_energy", "delta_phi_full", "qc_pass"]
    ]
    merged = comp.merge(d003, on="rule", how="left")
    plt.figure(figsize=(7.4, 5.0))
    for rule in RULES:
        sub = merged[merged["rule"] == rule]
        if sub.empty:
            continue
        plt.scatter(
            sub["nmstv_mean"],
            sub["delta_phi_energy"],
            s=70,
            color=COLORS[rule],
            label=RULE_LABELS[rule],
        )
        plt.text(
            float(sub["nmstv_mean"].iloc[0]),
            float(sub["delta_phi_energy"].iloc[0]),
            "  " + RULE_LABELS[rule],
            va="center",
            fontsize=8,
        )
    plt.xlabel("label complexity, NMSTV mean")
    plt.ylabel("phi_energy at d_raw=0.03")
    plt.title("Complexity versus claimable local phi_energy")
    plt.grid(True, alpha=0.25)
    savefig(path)


def plot_ce_error(qc: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for rule in RULES:
        sub = qc[qc["rule"] == rule].sort_values("radius")
        axes[0].plot(sub["radius"], sub["weighted_ce_mean"], marker="o", ms=3, color=COLORS[rule], label=RULE_LABELS[rule])
        axes[1].plot(sub["radius"], sub["weighted_error_mean"], marker="o", ms=3, color=COLORS[rule], label=RULE_LABELS[rule])
    for ax, ylabel in zip(axes, ["weighted CE", "weighted error"]):
        ax.set_xscale("log")
        ax.set_xlabel("d_raw")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Weighted CE and error by rule")
    savefig(path)


def load_d085_prior() -> pd.DataFrame:
    rows = []
    if not PRIOR_D085.exists():
        return pd.DataFrame()
    for path in PRIOR_D085.glob("*/ref_*/r_0p8500/unit_summary.json"):
        with path.open() as f:
            data = json.load(f)
        rows.append(
            {
                "rule": data.get("rule"),
                "ref_id": data.get("ref_id"),
                "radius": data.get("radius"),
                "split_logZ_per_P_diff": data.get("split_logZ_per_P_diff"),
                "weighted_ce": data.get("weighted_ce"),
                "weighted_error": data.get("weighted_error"),
                "finite": data.get("finite"),
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def plot_d085(d085: pd.DataFrame, path: Path) -> str:
    plt.figure(figsize=(8.5, 4.6))
    if d085.empty:
        plt.text(0.5, 0.5, "No prior d_raw=0.85 unit summaries found.", ha="center", va="center")
        plt.axis("off")
        status = "only_referenced_no_prior_unit_summaries_found"
    else:
        positions = range(len(RULES))
        data = [
            d085[d085["rule"] == rule]["split_logZ_per_P_diff"].dropna().to_numpy()
            for rule in RULES
        ]
        plt.boxplot(data, positions=list(positions), widths=0.55, patch_artist=True)
        for i, rule in enumerate(RULES):
            vals = data[i]
            if len(vals):
                x = np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals))
                plt.scatter(x, vals, s=12, alpha=0.45, color=COLORS[rule])
        plt.axhline(0.004, color="#9a3b58", lw=1.2, ls="--", label="production split-logZ gate")
        plt.xticks(list(positions), [RULE_LABELS[r] for r in RULES], rotation=20, ha="right")
        plt.ylabel("split_logZ_per_P_diff at d_raw=0.85")
        plt.title("Prior d_raw=0.85 sectorization diagnostic, no production claim")
        plt.legend(frameon=False)
        plt.grid(True, axis="y", alpha=0.2)
        status = "referenced_from_prior_unit_summaries_not_reproduced"
    savefig(path)
    return status


def write_report(
    qc: pd.DataFrame,
    claims: pd.DataFrame,
    comp: pd.DataFrame,
    ref_summary: pd.DataFrame,
    status: dict,
    d085_status: str,
) -> None:
    fail_if_exists(OUT / "REPORT.md")
    common = status.get("all_rule_common_qc_pass_radii", [])
    no_claim = claims[claims["comparison_claim_status"] != "claimable_all_rule_comparison_radius"]
    pass_by_rule = (
        qc.groupby("rule")["qc_pass"].apply(lambda s: bool(s.all())).reindex(RULES).fillna(False)
    )
    max_rule_pass = (
        qc[qc["qc_pass"] == True].groupby("rule")["radius"].max().reindex(RULES)
    )
    lines = [
        "# MNIST10 PM-SAIS Local-Support Comparison",
        "",
        "Production comparison is restricted to the common QC-supported local support `d_raw <= 0.65`.",
        "`d_raw = 0.85` is diagnostic/no-claim due to prior sectorization evidence; it is not included in the main `phi(d)` comparison.",
        "",
        "## Main Scientific Question",
        "",
        "Does lower label complexity correspond to wider or less negative `phi_energy(d)` on the common local support?",
        "",
        "## Setup",
        "",
        "- Source of truth: `02_dnn/08_mnist`.",
        "- Input: MNIST-derived `28x28 -> 10x10`, `n_train = 512`, single dataset.",
        "- Model family: existing small MLP / 3NN setup with `P = 2461`.",
        "- Selector: `dense_qc_stable_ref30`, 30 references per rule.",
        "- Radius grid: `0.01, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.04, 0.05, 0.065, 0.08, 0.12, 0.15, 0.2, 0.3, 0.45, 0.65`.",
        "- PM-SAIS resampling completed units: `{}` / `{}`.".format(
            status.get("completed_units"), status.get("expected_units")
        ),
        "",
        "## QC Policy",
        "",
        "- Production split-logZ gate: `split_logZ_per_P_diff <= 0.004`.",
        "- ESS gate: existing 08_mnist threshold, recorded as `q05_ess_fraction` in `qc_by_rule_radius.csv`.",
        "- Bootstrap SE must be finite and small, and no NaN/Inf units are claimable.",
        "- Failed radii are marked `no_claim`; no interpolation is used.",
        "",
        "## Main Results",
        "",
        "- All-rule common QC-pass radii: `{}`.".format(", ".join(str(x) for x in common)),
        "- Rules passing every radius on `d_raw <= 0.65`: `{}`.".format(
            ", ".join(rule for rule, ok in pass_by_rule.items() if ok) or "none"
        ),
        "- Per-rule maximum QC-pass radius: `{}`.".format(
            ", ".join(f"{rule}={max_rule_pass[rule]:g}" for rule in RULES if pd.notna(max_rule_pass[rule]))
        ),
        "- Main all-rule comparison claims are limited to the common pass radii above.",
        "",
        "## Complexity Context",
        "",
        markdown_table(comp),
        "",
        "## Reference Summary",
        "",
        markdown_table(ref_summary[["rule", "observed_ref_count", "selected_ref_ids"]]),
        "",
        "## No-Claim Radii",
        "",
        markdown_table(no_claim[["rule", "radius", "claim_status", "comparison_claim_status"]]),
        "",
        "## d_raw=0.85 Diagnostic",
        "",
        "- Diagnostic status: `{}`.".format(d085_status),
        "- `d_raw=0.85` was not rerun for this final production comparison.",
        "- Prior `d_raw=0.85` unit summaries were used only to visualize/record the sectorization diagnostic; they were not used for reference removal or production claims.",
        "",
        "## Limitations",
        "",
        "- Single dataset.",
        "- Optimizer-induced references.",
        "- No claim beyond local support `d_raw <= 0.65`.",
        "- No post-hoc reference removal.",
        "- `d_raw=0.85` remains diagnostic/no-claim.",
        "",
        "## Files",
        "",
        "- `phi_energy_by_rule_radius.csv`",
        "- `phi_full_by_rule_radius.csv`",
        "- `qc_by_rule_radius.csv`",
        "- `reference_summary_by_rule.csv`",
        "- `complexity_by_rule.csv`",
        "- `final_claim_table.csv`",
    ]
    tmp = OUT / "REPORT.md.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(OUT / "REPORT.md")


def main() -> None:
    phi, qc, units, claims, status = read_inputs()
    if status.get("completed_units") != status.get("expected_units"):
        raise RuntimeError("Final aggregate is not complete")

    energy = phi[
        ["selector", "rule", "radius", "d0", "delta_phi_energy", "n_units", "qc_pass", "claim_status"]
    ].rename(columns={"delta_phi_energy": "phi_energy"})
    full = phi[
        ["selector", "rule", "radius", "d0", "delta_phi_full", "n_units", "qc_pass", "claim_status"]
    ].rename(columns={"delta_phi_full": "phi_full"})
    ref_summary = make_reference_summary(units)
    comp = load_complexity()
    d085 = load_d085_prior()

    write_csv_if_missing(energy, OUT / "phi_energy_by_rule_radius.csv")
    write_csv_if_missing(full, OUT / "phi_full_by_rule_radius.csv")
    write_csv_if_missing(qc, OUT / "qc_by_rule_radius.csv")
    write_csv_if_missing(ref_summary, OUT / "reference_summary_by_rule.csv")
    write_csv_if_missing(comp, OUT / "complexity_by_rule.csv")
    write_csv_if_missing(claims, OUT / "final_claim_table.csv")
    if not d085.empty:
        write_csv_if_missing(d085, OUT / "d0p85_prior_diagnostic_by_reference.csv")

    fig01 = OUT / "fig01_phi_energy_all_rules_dmax0p65.png"
    if not fig01.exists():
        plot_phi(phi, "delta_phi_energy", "PM-SAIS phi_energy by label rule, d_raw <= 0.65", "phi_energy(d) - phi_energy(d0)", fig01)
    fig02 = OUT / "fig02_phi_full_all_rules_dmax0p65.png"
    if not fig02.exists():
        plot_phi(phi, "delta_phi_full", "PM-SAIS phi_full by label rule, d_raw <= 0.65", "phi_full(d) - phi_full(d0)", fig02)
    fig03 = OUT / "fig03_qc_pass_grid_dmax0p65.png"
    if not fig03.exists():
        plot_qc_grid(qc, fig03)
    fig04 = OUT / "fig04_complexity_vs_phi_summary.png"
    if not fig04.exists():
        plot_complexity(comp, claims, fig04)
    fig05 = OUT / "fig05_weighted_ce_error_by_rule.png"
    if not fig05.exists():
        plot_ce_error(qc, fig05)
    fig06 = OUT / "fig06_d0p85_sectorization_diagnostic_if_available.png"
    if fig06.exists():
        d085_status = (
            "referenced_from_prior_unit_summaries_not_reproduced"
            if not d085.empty
            else "only_referenced_no_prior_unit_summaries_found"
        )
    else:
        d085_status = plot_d085(d085, fig06)
    write_report(qc, claims, comp, ref_summary, status, d085_status)


if __name__ == "__main__":
    main()
