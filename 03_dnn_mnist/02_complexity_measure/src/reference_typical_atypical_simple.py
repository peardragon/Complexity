#!/usr/bin/env python3
"""Single-feature typical/atypical reference split for MNIST10 local support."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN = Path(
    "/home/bjyong/Complexity/windows_project/02_dnn/08_mnist/runs/final/"
    "local_support_dmax0p65_all_rules_resampled"
)
OUT = RUN / "reference_typical_atypical_simple_phi065_mad15_v2"
UNITS = RUN / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv"

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
    "typical": "#3b7f5f",
    "atypical": "#a13f46",
}
RADIUS_FEATURE = 0.65
ROBUST_Z_CUTOFF = 1.5


def fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    fail_if_exists(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_text(text: str, path: Path) -> None:
    fail_if_exists(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def savefig(path: Path) -> None:
    fail_if_exists(path)
    tmp = path.with_name(path.name + ".tmp.png")
    plt.savefig(tmp, dpi=180, bbox_inches="tight")
    plt.close()
    tmp.replace(path)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def robust_split(feature: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    values = feature.astype(float)
    median = float(values.median())
    mad = float(np.median(np.abs(values.to_numpy() - median)))
    sigma = 1.4826 * mad
    if sigma == 0.0 or not np.isfinite(sigma):
        robust_z = pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    else:
        robust_z = (values - median) / sigma
    group = np.where(np.abs(robust_z) > ROBUST_Z_CUTOFF, "atypical", "typical")
    threshold_low = median - ROBUST_Z_CUTOFF * sigma
    threshold_high = median + ROBUST_Z_CUTOFF * sigma
    result = pd.DataFrame(
        {
            "ref_id": values.index.astype(int),
            "phi_energy_d0p65": values.to_numpy(),
            "robust_z_phi_d0p65": robust_z.to_numpy(),
            "group": group,
        }
    )
    meta = {
        "median_phi_d0p65": median,
        "mad_phi_d0p65": mad,
        "robust_sigma_phi_d0p65": sigma,
        "threshold_low": float(threshold_low),
        "threshold_high": float(threshold_high),
    }
    return result, meta


def build_tables(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = []
    summary_rows = []
    for rule in RULES:
        at_radius = units[(units["rule"].eq(rule)) & np.isclose(units["radius"], RADIUS_FEATURE)]
        feature = at_radius.set_index("ref_id")["delta_phi_energy_unit"].sort_index()
        split, meta = robust_split(feature)
        split.insert(0, "rule", rule)
        split["feature"] = "phi_energy_d0p65"
        assignments.append(split)

        atypical_refs = split.loc[split["group"].eq("atypical"), "ref_id"].astype(int).tolist()
        typical_refs = split.loc[split["group"].eq("typical"), "ref_id"].astype(int).tolist()
        summary_rows.append(
            {
                "rule": rule,
                "feature": "phi_energy_d0p65",
                "n_refs": int(len(split)),
                "n_typical": int(len(typical_refs)),
                "n_atypical": int(len(atypical_refs)),
                "median_phi_d0p65": meta["median_phi_d0p65"],
                "mad_phi_d0p65": meta["mad_phi_d0p65"],
                "threshold_low": meta["threshold_low"],
                "threshold_high": meta["threshold_high"],
                "typical_refs": " ".join(f"{x:03d}" for x in sorted(typical_refs)),
                "atypical_refs": " ".join(f"{x:03d}" for x in sorted(atypical_refs)),
            }
        )

    assign = pd.concat(assignments, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    joined = units.merge(assign[["rule", "ref_id", "group"]], on=["rule", "ref_id"], how="left")
    curve = (
        joined.groupby(["rule", "group", "radius"], as_index=False)
        .agg(
            n_refs=("ref_id", "nunique"),
            phi_energy_mean=("delta_phi_energy_unit", "mean"),
            phi_energy_median=("delta_phi_energy_unit", "median"),
            phi_energy_q25=("delta_phi_energy_unit", lambda x: float(np.quantile(x, 0.25))),
            phi_energy_q75=("delta_phi_energy_unit", lambda x: float(np.quantile(x, 0.75))),
            phi_energy_sd=("delta_phi_energy_unit", "std"),
            split_logZ_max=("split_logZ_per_P_diff", "max"),
            weighted_ce_mean=("weighted_ce", "mean"),
            weighted_error_mean=("weighted_error", "mean"),
        )
        .sort_values(["rule", "group", "radius"])
    )
    return assign.sort_values(["rule", "group", "ref_id"]), summary, curve


def plot_strip(assign: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=False)
    axes = axes.ravel()
    for ax, rule in zip(axes, RULES):
        sub = assign[assign["rule"].eq(rule)].sort_values("phi_energy_d0p65")
        y = np.zeros(len(sub))
        ax.scatter(
            sub["phi_energy_d0p65"],
            y,
            c=[COLORS[g] for g in sub["group"]],
            s=46,
            edgecolor="white",
            linewidth=0.5,
        )
        for _, row in sub.iterrows():
            if row["group"] != "atypical":
                continue
            y_text = 0.075 if row["robust_z_phi_d0p65"] > 0 else -0.075
            va = "bottom" if y_text > 0 else "top"
            ax.text(
                row["phi_energy_d0p65"],
                y_text,
                f"{int(row['ref_id']):03d}",
                rotation=90,
                ha="center",
                va=va,
                fontsize=8,
                color="#6f1d29",
            )
        meta = summary[summary["rule"].eq(rule)].iloc[0]
        ax.axvline(meta["threshold_low"], color="#777777", lw=1.0, ls="--")
        ax.axvline(meta["threshold_high"], color="#777777", lw=1.0, ls="--")
        ax.axvline(meta["median_phi_d0p65"], color="#333333", lw=1.0)
        ax.set_title(f"{LABELS[rule]}: single-feature split", pad=10)
        ax.set_xlabel("phi_energy(d_raw=0.65)")
        ax.set_ylim(-0.18, 0.18)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.2)
    fig.suptitle("Typical/atypical references from one scalar: phi_energy at d_raw=0.65", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(OUT / "fig01_phi065_typical_atypical_strip.png")


def plot_curves(curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.0), sharex=True)
    axes = axes.ravel()
    for ax, rule in zip(axes, RULES):
        sub = curve[curve["rule"].eq(rule)].copy()
        for group in ["typical", "atypical"]:
            ss = sub[sub["group"].eq(group)].sort_values("radius")
            if ss.empty:
                continue
            ax.plot(ss["radius"], ss["phi_energy_median"], marker="o", ms=3.5, lw=1.8, color=COLORS[group], label=group)
            ax.fill_between(ss["radius"], ss["phi_energy_q25"], ss["phi_energy_q75"], color=COLORS[group], alpha=0.16, linewidth=0)
        ax.set_xscale("log")
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_title(LABELS[rule])
        ax.set_xlabel("d_raw")
        ax.set_ylabel("median phi_energy")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Median phi_energy curves after simple typical/atypical split")
    savefig(OUT / "fig02_phi_curves_typical_atypical.png")


def plot_sorted(assign: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=False)
    axes = axes.ravel()
    for ax, rule in zip(axes, RULES):
        sub = assign[assign["rule"].eq(rule)].sort_values("phi_energy_d0p65").reset_index(drop=True)
        x = np.arange(len(sub))
        ax.bar(x, sub["phi_energy_d0p65"], color=[COLORS[g] for g in sub["group"]], width=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r):03d}" for r in sub["ref_id"]], rotation=90, fontsize=7)
        ax.set_title(LABELS[rule])
        ax.set_ylabel("phi_energy(d_raw=0.65)")
        ax.grid(True, axis="y", alpha=0.2)
    fig.suptitle("References sorted by the single split feature")
    savefig(OUT / "fig03_sorted_phi065_by_rule.png")


def write_report(summary: pd.DataFrame) -> None:
    display = summary[
        [
            "rule",
            "n_typical",
            "n_atypical",
            "median_phi_d0p65",
            "mad_phi_d0p65",
            "typical_refs",
            "atypical_refs",
        ]
    ].copy()
    text = "\n".join(
        [
            "# Simple Typical/Atypical Reference Split",
            "",
            "This is a diagnostic reference decomposition, not post-hoc reference removal.",
            "",
            "Method:",
            "",
            "- Single feature only: per-reference `phi_energy(d_raw=0.65)` from the existing local-support PM-SAIS units.",
            "- `d_raw=0.85` diagnostics are not used.",
            "- A reference is `atypical` if `|robust_z| > 1.5`, where robust z uses the rule-wise median and `1.4826 * MAD`.",
            "- All other references are `typical`.",
            "",
            "Summary:",
            "",
            markdown_table(display),
            "",
            "Files:",
            "",
            "- `reference_typical_atypical_assignments.csv`",
            "- `typical_atypical_summary_by_rule.csv`",
            "- `phi_energy_typical_atypical_by_rule_radius.csv`",
            "- `fig01_phi065_typical_atypical_strip.png`",
            "- `fig02_phi_curves_typical_atypical.png`",
            "- `fig03_sorted_phi065_by_rule.png`",
        ]
    )
    write_text(text + "\n", OUT / "REPORT.md")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    units = pd.read_csv(UNITS)
    assignments, summary, curve = build_tables(units)
    write_csv(assignments, OUT / "reference_typical_atypical_assignments.csv")
    write_csv(summary, OUT / "typical_atypical_summary_by_rule.csv")
    write_csv(curve, OUT / "phi_energy_typical_atypical_by_rule_radius.csv")
    plot_strip(assignments, summary)
    plot_curves(curve)
    plot_sorted(assignments)
    write_report(summary)


if __name__ == "__main__":
    main()
