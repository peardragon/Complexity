#!/usr/bin/env python3
"""Plot single-reference strict-4096 phi curves with QC ignored."""

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
OUT = RUN / "single_reference_phi_strict4096_no_qc_v2"
UNITS = RUN / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv"
P = 2461.0

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


def build_strict_phi(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = units[units["n_samples"].eq(4096)].copy()
    if strict.empty:
        raise RuntimeError("No strict n_samples=4096 rows found")
    strict["radius"] = pd.to_numeric(strict["radius"], errors="coerce")
    strict["logZ_inf_full"] = pd.to_numeric(strict["logZ_inf_full"], errors="coerce")
    rows = []
    coverage_rows = []
    for rule in RULES:
        rule_sub = strict[strict["rule"].eq(rule)].copy()
        coverage = (
            rule_sub.groupby("radius", as_index=False)
            .agg(strict4096_ref_count=("ref_id", "nunique"))
            .sort_values("radius")
        )
        for _, row in coverage.iterrows():
            coverage_rows.append({"rule": rule, "radius": float(row["radius"]), "strict4096_ref_count": int(row["strict4096_ref_count"])})
        for ref_id, ref_sub in rule_sub.groupby("ref_id"):
            ref_sub = ref_sub.sort_values("radius")
            finite = ref_sub[np.isfinite(ref_sub["logZ_inf_full"])]
            if finite.empty:
                continue
            anchor = finite.iloc[0]
            anchor_radius = float(anchor["radius"])
            anchor_logz = float(anchor["logZ_inf_full"])
            for _, row in finite.iterrows():
                rows.append(
                    {
                        "rule": rule,
                        "ref_id": int(ref_id),
                        "radius": float(row["radius"]),
                        "anchor_radius": anchor_radius,
                        "logZ_inf_full_strict4096": float(row["logZ_inf_full"]),
                        "anchor_logZ_inf_full_strict4096": anchor_logz,
                        "phi_energy_strict4096_rel_anchor": (float(row["logZ_inf_full"]) - anchor_logz) / P,
                        "existing_delta_phi_energy_unit": float(row["delta_phi_energy_unit"]),
                        "split_logZ_per_P_diff": float(row["split_logZ_per_P_diff"]),
                        "ess_fraction": float(row["ess_fraction"]),
                    }
                )
    curve = pd.DataFrame(rows).sort_values(["rule", "ref_id", "radius"])
    coverage_df = pd.DataFrame(coverage_rows).sort_values(["rule", "radius"])
    return curve, coverage_df


def plot_existing_delta(curve: pd.DataFrame) -> None:
    plt.figure(figsize=(10.5, 6.6))
    for rule in RULES:
        rule_sub = curve[curve["rule"].eq(rule)]
        first = True
        for ref_id, sub in rule_sub.groupby("ref_id"):
            sub = sub.sort_values("radius")
            plt.plot(
                sub["radius"],
                sub["existing_delta_phi_energy_unit"],
                color=COLORS[rule],
                alpha=0.28,
                lw=1.25,
                label=LABELS[rule] if first else None,
            )
            first = False
    plt.axhline(0.0, color="#666666", lw=0.8)
    plt.xscale("log")
    plt.xlabel("d_raw")
    plt.ylabel("single-ref delta phi_energy(d), existing d0 anchor")
    plt.title("Single-reference phi(d), strict n_samples=4096 rows only, QC ignored")
    plt.legend(frameon=False, ncol=4)
    plt.grid(True, which="both", alpha=0.22)
    savefig(OUT / "fig01_single_reference_phi_strict4096_all_rules.png")


def plot_strict_anchor(curve: pd.DataFrame) -> None:
    plt.figure(figsize=(10.5, 6.6))
    for rule in RULES:
        rule_sub = curve[curve["rule"].eq(rule)]
        first = True
        for _, sub in rule_sub.groupby("ref_id"):
            sub = sub.sort_values("radius")
            plt.plot(
                sub["radius"],
                sub["phi_energy_strict4096_rel_anchor"],
                color=COLORS[rule],
                alpha=0.22,
                lw=1.1,
                label=LABELS[rule] if first else None,
            )
            first = False
    plt.axhline(0.0, color="#666666", lw=0.8)
    plt.xscale("log")
    plt.xlabel("d_raw")
    plt.ylabel("single-ref phi_energy relative to first strict-4096 radius")
    plt.title("Strict-4096-only reanchored diagnostic, QC ignored")
    plt.legend(frameon=False, ncol=4)
    plt.grid(True, which="both", alpha=0.22)
    savefig(OUT / "fig02_single_reference_phi_strict4096_reanchored.png")


def plot_coverage(coverage: pd.DataFrame) -> None:
    radii = sorted(coverage["radius"].unique())
    fig, ax = plt.subplots(figsize=(10.0, 3.9))
    width = 0.18
    x = np.arange(len(radii))
    for idx, rule in enumerate(RULES):
        sub = coverage[coverage["rule"].eq(rule)].set_index("radius")
        vals = [int(sub.loc[r, "strict4096_ref_count"]) if r in sub.index else 0 for r in radii]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, color=COLORS[rule], label=LABELS[rule])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}" for r in radii], rotation=45, ha="right")
    ax.set_ylabel("strict-4096 ref count")
    ax.set_xlabel("d_raw")
    ax.set_title("Coverage of strict n_samples=4096 rows")
    ax.legend(frameon=False, ncol=4)
    ax.grid(True, axis="y", alpha=0.22)
    savefig(OUT / "fig03_strict4096_coverage_by_rule_radius.png")


def write_report(curve: pd.DataFrame, coverage: pd.DataFrame) -> None:
    summary_rows = []
    for rule in RULES:
        sub = curve[curve["rule"].eq(rule)]
        counts = sub.groupby("ref_id")["radius"].nunique()
        anchors = sub.groupby("ref_id")["anchor_radius"].first()
        summary_rows.append(
            {
                "rule": rule,
                "strict_refs": int(counts.size),
                "min_points_per_ref": int(counts.min()),
                "max_points_per_ref": int(counts.max()),
                "anchor_radii": " ".join(f"{x:g}" for x in sorted(anchors.unique())),
            }
        )
    summary = pd.DataFrame(summary_rows)
    text = "\n".join(
        [
            "# Single-Reference Strict-4096 Phi Curves",
            "",
            "QC is ignored in this diagnostic plot.",
            "",
            "Only rows with `n_samples == 4096` are used. Replicate/fallback rows are excluded.",
            "`fig01` plots the existing per-unit `delta_phi_energy_unit` for those strict-4096 rows, colored by rule.",
            "Because strict `d_raw=0.01` rows are missing for some rules, `fig02` additionally shows a strict-only reanchored diagnostic with `(logZ(d) - logZ(anchor)) / P`, where the anchor is each reference's first available strict-4096 radius.",
            "",
            "Summary:",
            "",
            markdown_table(summary),
            "",
            "Files:",
            "",
            "- `single_reference_phi_strict4096.csv`",
            "- `strict4096_coverage_by_rule_radius.csv`",
            "- `fig01_single_reference_phi_strict4096_all_rules.png`",
            "- `fig02_single_reference_phi_strict4096_reanchored.png`",
            "- `fig03_strict4096_coverage_by_rule_radius.png`",
        ]
    )
    write_text(text + "\n", OUT / "REPORT.md")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    units = pd.read_csv(UNITS)
    curve, coverage = build_strict_phi(units)
    write_csv(curve, OUT / "single_reference_phi_strict4096.csv")
    write_csv(coverage, OUT / "strict4096_coverage_by_rule_radius.csv")
    plot_existing_delta(curve)
    plot_strict_anchor(curve)
    plot_coverage(coverage)
    write_report(curve, coverage)


if __name__ == "__main__":
    main()
