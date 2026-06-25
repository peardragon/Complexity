#!/usr/bin/env python3
"""Local-only strict-4096 single-reference phi_energy diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
SOURCE_UNITS = Path(
    "/home/bjyong/Complexity/windows_project/02_dnn/08_mnist/runs/final/"
    "local_support_dmax0p65_all_rules_resampled/05_pool2_pm_sais_sampling/"
    "shell_summary_by_unit_with_phi.csv"
)
OUT = LOCAL_ROOT / "05_proxy_local_entropy" / "raw_outputs" / "strict4096_phi_energy_shape_local"
P = 2461.0
EXTREME_PHI_GATE = -0.3
HIGH_CE_GATE = 1.0
HIGH_ERROR_GATE = 0.3
SPLIT_GATE = 0.004

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
    rows = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(rows)


def compute_derivative(sub: pd.DataFrame, value_col: str) -> np.ndarray:
    sub = sub.sort_values("radius")
    x = sub["radius"].to_numpy(dtype=float)
    y = sub[value_col].to_numpy(dtype=float)
    if len(sub) < 2:
        return np.full(len(sub), np.nan)
    return np.gradient(y, x)


def load_strict_units() -> pd.DataFrame:
    units = pd.read_csv(SOURCE_UNITS)
    strict = units[units["n_samples"].eq(4096)].copy()
    if strict.empty:
        raise RuntimeError("No n_samples == 4096 rows found")
    numeric_cols = [
        "radius",
        "ref_id",
        "logZ_inf_full",
        "weighted_ce",
        "weighted_error",
        "ess_fraction",
        "split_logZ_per_P_diff",
    ]
    for col in numeric_cols:
        strict[col] = pd.to_numeric(strict[col], errors="coerce")
    strict["phi_energy"] = strict["logZ_inf_full"] / P
    strict["extreme_low_phi"] = strict["phi_energy"] < EXTREME_PHI_GATE

    rows = []
    for (rule, ref_id), sub in strict.groupby(["rule", "ref_id"], sort=False):
        sub = sub.sort_values("radius").copy()
        sub["d_phi_energy_dd_raw"] = compute_derivative(sub, "phi_energy")
        values = sub["phi_energy"].to_numpy(dtype=float)
        for idx, row in sub.reset_index(drop=True).iterrows():
            neighbors = []
            if idx > 0:
                neighbors.append(values[idx - 1])
            if idx + 1 < len(values):
                neighbors.append(values[idx + 1])
            neighbor_median = float(np.median(neighbors)) if neighbors else float("nan")
            item = row.to_dict()
            item["neighbor_phi_energy_median"] = neighbor_median
            item["phi_energy_jump_from_neighbors"] = (
                float(row["phi_energy"] - neighbor_median)
                if np.isfinite(neighbor_median)
                else float("nan")
            )
            rows.append(item)
    strict = pd.DataFrame(rows).sort_values(["rule", "ref_id", "radius"]).reset_index(drop=True)
    strict["reason_high_ce"] = strict["weighted_ce"] >= HIGH_CE_GATE
    strict["reason_high_error"] = strict["weighted_error"] >= HIGH_ERROR_GATE
    strict["reason_split_unstable"] = strict["split_logZ_per_P_diff"] > SPLIT_GATE
    strict["reason_local_phi_spike"] = strict["phi_energy_jump_from_neighbors"] < EXTREME_PHI_GATE

    def reason_code(row: pd.Series) -> str:
        reasons = []
        if bool(row["reason_high_ce"]):
            reasons.append("high_weighted_ce")
        if bool(row["reason_high_error"]):
            reasons.append("high_weighted_error")
        if bool(row["reason_split_unstable"]):
            reasons.append("split_unstable")
        if bool(row["reason_local_phi_spike"]):
            reasons.append("local_phi_spike")
        return "+".join(reasons) or "not_extreme"

    strict["reason_code"] = strict.apply(reason_code, axis=1)
    return strict


def visible_derivatives(strict: pd.DataFrame) -> pd.DataFrame:
    visible = strict[~strict["extreme_low_phi"]].copy()
    rows = []
    for (rule, ref_id), sub in visible.groupby(["rule", "ref_id"], sort=False):
        sub = sub.sort_values("radius").copy()
        sub["d_phi_energy_dd_visible"] = compute_derivative(sub, "phi_energy")
        rows.append(sub)
    return pd.concat(rows, ignore_index=True).sort_values(["rule", "ref_id", "radius"])


def robust_summary(visible: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rule, radius), sub in visible.groupby(["rule", "radius"], sort=False):
        phi = sub["phi_energy"].to_numpy(dtype=float)
        deriv = sub["d_phi_energy_dd_visible"].to_numpy(dtype=float)
        deriv = deriv[np.isfinite(deriv)]
        rows.append(
            {
                "rule": rule,
                "radius": float(radius),
                "visible_ref_count": int(sub["ref_id"].nunique()),
                "phi_energy_median": float(np.median(phi)),
                "phi_energy_q10": float(np.quantile(phi, 0.10)),
                "phi_energy_q25": float(np.quantile(phi, 0.25)),
                "phi_energy_q75": float(np.quantile(phi, 0.75)),
                "phi_energy_q90": float(np.quantile(phi, 0.90)),
                "d_phi_energy_dd_median": float(np.median(deriv)) if len(deriv) else float("nan"),
                "d_phi_energy_dd_q10": float(np.quantile(deriv, 0.10)) if len(deriv) else float("nan"),
                "d_phi_energy_dd_q25": float(np.quantile(deriv, 0.25)) if len(deriv) else float("nan"),
                "d_phi_energy_dd_q75": float(np.quantile(deriv, 0.75)) if len(deriv) else float("nan"),
                "d_phi_energy_dd_q90": float(np.quantile(deriv, 0.90)) if len(deriv) else float("nan"),
                "weighted_ce_median": float(np.median(sub["weighted_ce"].to_numpy(dtype=float))),
                "split_logZ_per_P_max_visible": float(sub["split_logZ_per_P_diff"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["rule", "radius"])


def coverage(strict: pd.DataFrame) -> pd.DataFrame:
    return (
        strict.groupby(["rule", "radius"], as_index=False)
        .agg(
            strict4096_ref_count=("ref_id", "nunique"),
            extreme_low_phi_count=("extreme_low_phi", "sum"),
        )
        .sort_values(["rule", "radius"])
    )


def outlier_tables(strict: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outliers = strict[strict["extreme_low_phi"]].copy()
    cols = [
        "rule",
        "ref_id",
        "radius",
        "phi_energy",
        "d_phi_energy_dd_raw",
        "logZ_inf_full",
        "weighted_ce",
        "weighted_error",
        "ess_fraction",
        "split_logZ_per_P_diff",
        "neighbor_phi_energy_median",
        "phi_energy_jump_from_neighbors",
        "reason_code",
    ]
    outliers = outliers[cols].sort_values("phi_energy")
    reason_summary = (
        outliers.groupby(["rule", "reason_code"], as_index=False)
        .agg(count=("ref_id", "size"), refs=("ref_id", lambda s: " ".join(f"{int(x):03d}" for x in sorted(set(s)))))
        .sort_values(["rule", "reason_code"])
    )
    return outliers, reason_summary


def plot_phi_panels(visible: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True)
    axes = axes.ravel()
    for ax, rule in zip(axes, RULES):
        rule_visible = visible[visible["rule"].eq(rule)]
        for _, ref_sub in rule_visible.groupby("ref_id"):
            ref_sub = ref_sub.sort_values("radius")
            ax.plot(ref_sub["radius"], ref_sub["phi_energy"], color=COLORS[rule], alpha=0.16, lw=0.9)
        sub = summary[summary["rule"].eq(rule)].sort_values("radius")
        ax.fill_between(sub["radius"], sub["phi_energy_q10"], sub["phi_energy_q90"], color=COLORS[rule], alpha=0.16, linewidth=0)
        ax.plot(sub["radius"], sub["phi_energy_median"], color=COLORS[rule], lw=2.0, marker="o", ms=3.5)
        ax.set_xscale("log")
        ax.set_title(f"{LABELS[rule]} phi_energy(d)")
        ax.set_xlabel("d_raw")
        ax.set_ylabel("phi_energy = logZ_inf_full / P")
        ax.grid(True, which="both", alpha=0.22)
    fig.suptitle("Readable single-reference phi_energy(d), strict-4096 visible rows")
    savefig(OUT / "fig01_readable_single_ref_phi_energy_by_rule.png")


def plot_phi_medians(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(9.8, 5.8))
    for rule in RULES:
        sub = summary[summary["rule"].eq(rule)].sort_values("radius")
        plt.fill_between(sub["radius"], sub["phi_energy_q25"], sub["phi_energy_q75"], color=COLORS[rule], alpha=0.12, linewidth=0)
        plt.plot(sub["radius"], sub["phi_energy_median"], color=COLORS[rule], lw=2.2, marker="o", ms=3.2, label=LABELS[rule])
    plt.xscale("log")
    plt.xlabel("d_raw")
    plt.ylabel("median phi_energy")
    plt.title("Rule-level phi_energy(d), strict-4096 visible rows")
    plt.legend(frameon=False, ncol=2)
    plt.grid(True, which="both", alpha=0.22)
    savefig(OUT / "fig02_rule_median_phi_energy.png")


def plot_derivative_panels(visible: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True)
    axes = axes.ravel()
    for ax, rule in zip(axes, RULES):
        rule_visible = visible[visible["rule"].eq(rule)]
        for _, ref_sub in rule_visible.groupby("ref_id"):
            ref_sub = ref_sub.sort_values("radius")
            ax.plot(ref_sub["radius"], ref_sub["d_phi_energy_dd_visible"], color=COLORS[rule], alpha=0.14, lw=0.9)
        sub = summary[summary["rule"].eq(rule)].sort_values("radius")
        ax.fill_between(sub["radius"], sub["d_phi_energy_dd_q10"], sub["d_phi_energy_dd_q90"], color=COLORS[rule], alpha=0.16, linewidth=0)
        ax.plot(sub["radius"], sub["d_phi_energy_dd_median"], color=COLORS[rule], lw=2.0, marker="o", ms=3.5)
        ax.axhline(0.0, color="#666666", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{LABELS[rule]} d phi_energy / d d_raw")
        ax.set_xlabel("d_raw")
        ax.set_ylabel("d phi_energy / d d_raw")
        ax.grid(True, which="both", alpha=0.22)
    fig.suptitle("Readable single-reference phi_energy derivative, strict-4096 visible rows")
    savefig(OUT / "fig03_readable_dphi_energy_dd_by_rule.png")


def plot_derivative_medians(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(9.8, 5.8))
    for rule in RULES:
        sub = summary[summary["rule"].eq(rule)].sort_values("radius")
        plt.fill_between(sub["radius"], sub["d_phi_energy_dd_q25"], sub["d_phi_energy_dd_q75"], color=COLORS[rule], alpha=0.12, linewidth=0)
        plt.plot(sub["radius"], sub["d_phi_energy_dd_median"], color=COLORS[rule], lw=2.2, marker="o", ms=3.2, label=LABELS[rule])
    plt.axhline(0.0, color="#666666", lw=0.8)
    plt.xscale("log")
    plt.xlabel("d_raw")
    plt.ylabel("median d phi_energy / d d_raw")
    plt.title("Rule-level phi_energy derivative, strict-4096 visible rows")
    plt.legend(frameon=False, ncol=2)
    plt.grid(True, which="both", alpha=0.22)
    savefig(OUT / "fig04_rule_median_dphi_energy_dd.png")


def plot_outliers(strict: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for rule in RULES:
        sub = strict[strict["rule"].eq(rule)]
        ax.scatter(sub["radius"], sub["phi_energy"], s=16, color=COLORS[rule], alpha=0.28, label=LABELS[rule])
        out = sub[sub["extreme_low_phi"]]
        ax.scatter(out["radius"], out["phi_energy"], s=30, color=COLORS[rule], edgecolor="black", linewidth=0.4)
    ax.axhline(EXTREME_PHI_GATE, color="#8a2635", lw=1.0, ls="--", label="extreme gate")
    ax.set_xscale("log")
    ax.set_xlabel("d_raw")
    ax.set_ylabel("raw strict-4096 phi_energy")
    ax.set_title("Raw phi_energy outliers retained as diagnostic")
    ax.grid(True, which="both", alpha=0.22)

    ax = axes[1]
    for rule in RULES:
        sub = strict[strict["rule"].eq(rule)]
        ax.scatter(sub["weighted_ce"], sub["phi_energy"], s=18, color=COLORS[rule], alpha=0.28, label=LABELS[rule])
        out = sub[sub["extreme_low_phi"]]
        ax.scatter(out["weighted_ce"], out["phi_energy"], s=34, color=COLORS[rule], edgecolor="black", linewidth=0.4)
    ax.axhline(EXTREME_PHI_GATE, color="#8a2635", lw=1.0, ls="--")
    ax.axvline(HIGH_CE_GATE, color="#8a2635", lw=1.0, ls=":")
    ax.set_xlabel("weighted CE")
    ax.set_ylabel("raw strict-4096 phi_energy")
    ax.set_title("Extreme phi_energy drops coincide with high weighted CE")
    ax.grid(True, alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    savefig(OUT / "fig05_raw_phi_energy_outlier_diagnostics.png")


def plot_coverage(cov: pd.DataFrame) -> None:
    radii = sorted(cov["radius"].unique())
    x = np.arange(len(radii))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    for idx, rule in enumerate(RULES):
        sub = cov[cov["rule"].eq(rule)].set_index("radius")
        vals = [int(sub.loc[r, "strict4096_ref_count"]) if r in sub.index else 0 for r in radii]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, color=COLORS[rule], label=LABELS[rule])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}" for r in radii], rotation=45, ha="right")
    ax.set_ylabel("strict-4096 ref count")
    ax.set_xlabel("d_raw")
    ax.set_title("Strict-4096 coverage by rule/radius")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=4)
    savefig(OUT / "fig06_strict4096_coverage.png")


def write_report(strict: pd.DataFrame, visible: pd.DataFrame, summary: pd.DataFrame, outliers: pd.DataFrame, reasons: pd.DataFrame, cov: pd.DataFrame) -> None:
    total = len(strict)
    n_extreme = int(strict["extreme_low_phi"].sum())
    counts = (
        strict.groupby("rule", as_index=False)
        .agg(
            strict_rows=("ref_id", "size"),
            strict_refs=("ref_id", "nunique"),
            extreme_phi_count=("extreme_low_phi", "sum"),
            phi_energy_median=("phi_energy", "median"),
            weighted_ce_median=("weighted_ce", "median"),
            extreme_weighted_ce_median=("weighted_ce", lambda s: float(np.median(s[strict.loc[s.index, "extreme_low_phi"]])) if strict.loc[s.index, "extreme_low_phi"].any() else float("nan")),
        )
    )
    coverage_summary = (
        cov.groupby("rule", as_index=False)
        .agg(
            min_ref_count=("strict4096_ref_count", "min"),
            max_ref_count=("strict4096_ref_count", "max"),
            radii_with_extreme_phi=("extreme_low_phi_count", lambda s: int((s > 0).sum())),
        )
    )
    text = "\n".join(
        [
            "# Local Strict-4096 Phi Energy Shape Diagnostics",
            "",
            "All outputs in this directory are under `local_project/03_dnn_mnist`; the Windows project table is read-only input evidence.",
            "",
            "Scope:",
            "",
            "- Use only rows with `n_samples == 4096`.",
            "- Ignore QC for raw diagnostics.",
            "- Focus on `phi_energy(d) = logZ_inf_full / P` and finite-difference `d phi_energy / d d_raw`.",
            "- `delta_phi` is not used for the main figures.",
            "- Extreme raw phi points are not silently removed: they are listed in `strict4096_phi_energy_outlier_cases.csv` and shown in `fig05_raw_phi_energy_outlier_diagnostics.png`.",
            "- Main shape figures exclude only `phi_energy < -0.3` high-loss events so the rule-level phi and derivative geometry is readable.",
            "",
            "Outlier Interpretation:",
            "",
            f"- Extreme gate: `phi_energy < {EXTREME_PHI_GATE}`.",
            f"- Extreme count: `{n_extreme}` / `{total}` strict-4096 rows.",
            "- Extreme rows have very high weighted CE / weighted error; several have acceptable ESS and small split-logZ, so split-logZ alone does not catch them.",
            "- Interpretation: strict-4096 sampling sometimes lands consistently in a high-loss shell sector for that ref/radius. This supports proposal/sector mismatch or very narrow low-loss mass, not necessarily a globally different landscape law.",
            "",
            "Rule Summary:",
            "",
            markdown_table(counts),
            "",
            "Coverage Summary:",
            "",
            markdown_table(coverage_summary),
            "",
            "Reason Summary:",
            "",
            markdown_table(reasons),
            "",
            "Files:",
            "",
            "- `strict4096_phi_energy_units_with_diagnostics.csv`",
            "- `strict4096_visible_phi_energy_derivatives.csv`",
            "- `strict4096_rule_radius_phi_derivative_summary.csv`",
            "- `strict4096_phi_energy_outlier_cases.csv`",
            "- `strict4096_outlier_reason_summary.csv`",
            "- `strict4096_coverage_by_rule_radius.csv`",
            "- `fig01_readable_single_ref_phi_energy_by_rule.png`",
            "- `fig02_rule_median_phi_energy.png`",
            "- `fig03_readable_dphi_energy_dd_by_rule.png`",
            "- `fig04_rule_median_dphi_energy_dd.png`",
            "- `fig05_raw_phi_energy_outlier_diagnostics.png`",
            "- `fig06_strict4096_coverage.png`",
        ]
    )
    write_text(text + "\n", OUT / "REPORT.md")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    strict = load_strict_units()
    visible = visible_derivatives(strict)
    summary = robust_summary(visible)
    cov = coverage(strict)
    outliers, reasons = outlier_tables(strict)

    write_csv(strict, OUT / "strict4096_phi_energy_units_with_diagnostics.csv")
    write_csv(visible, OUT / "strict4096_visible_phi_energy_derivatives.csv")
    write_csv(summary, OUT / "strict4096_rule_radius_phi_derivative_summary.csv")
    write_csv(outliers, OUT / "strict4096_phi_energy_outlier_cases.csv")
    write_csv(reasons, OUT / "strict4096_outlier_reason_summary.csv")
    write_csv(cov, OUT / "strict4096_coverage_by_rule_radius.csv")

    plot_phi_panels(visible, summary)
    plot_phi_medians(summary)
    plot_derivative_panels(visible, summary)
    plot_derivative_medians(summary)
    plot_outliers(strict)
    plot_coverage(cov)
    write_report(strict, visible, summary, outliers, reasons, cov)


if __name__ == "__main__":
    main()
