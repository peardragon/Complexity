#!/usr/bin/env python3
"""Local-only random-pool split diagnostics for MNIST10 PM-SAIS units."""

from __future__ import annotations

import ast
import math
import os
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
OUT = LOCAL_ROOT / "05_proxy_local_entropy" / "raw_outputs" / "random_pool_split_diagnostics_local"

SPLIT_GATE = 0.004
ESS_GATE = 0.04
RANDOM_SPLITS = 2000
RANDOM_SEED = 20260617

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


def prepare_output_dir() -> Path:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    tmp = OUT.with_name(f"{OUT.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(f"Temporary output directory already exists: {tmp}")
    tmp.mkdir(parents=True)
    (tmp / "figures").mkdir()
    return tmp


def finalize_output_dir(tmp: Path) -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    tmp.rename(OUT)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def savefig(path: Path) -> None:
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def infer_parameter_dimension(df: pd.DataFrame) -> float:
    sub = df[
        df["split0_logZ"].notna()
        & df["split1_logZ"].notna()
        & df["split_logZ_per_P_diff"].notna()
        & (df["split_logZ_per_P_diff"] > 1e-12)
    ].copy()
    p_vals = (
        (sub["split0_logZ"] - sub["split1_logZ"]).abs()
        / sub["split_logZ_per_P_diff"]
    ).replace([np.inf, -np.inf], np.nan)
    p_vals = p_vals.dropna()
    if p_vals.empty:
        return 2461.0
    return float(np.median(p_vals))


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda x: str(x).lower() == "true").fillna(False)


def parse_replicates(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed]


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    m = float(np.max(values))
    return m + float(np.log(np.mean(np.exp(values - m))))


def random_pool_diffs(logz: np.ndarray, p_dim: float, rng: np.random.Generator) -> np.ndarray:
    logz = np.asarray(logz, dtype=float)
    if len(logz) < 4 or len(logz) % 2:
        return np.asarray([], dtype=float)
    half = len(logz) // 2
    diffs = np.empty(RANDOM_SPLITS, dtype=float)
    for i in range(RANDOM_SPLITS):
        perm = rng.permutation(len(logz))
        a = logmeanexp(logz[perm[:half]])
        b = logmeanexp(logz[perm[half:]])
        diffs[i] = abs(a - b) / p_dim
    return diffs


def existing_fail_reason(row: pd.Series) -> str:
    reasons = []
    if not bool(row["finite_pass"]):
        reasons.append("non_finite")
    if not bool(row["ess_pass_0p04"]):
        reasons.append("ess_fraction_below_0p04")
    if not bool(row["existing_split_pass_0p004"]):
        reasons.append("split_logZ_per_P_diff_above_0p004")
    return "ok" if not reasons else ";".join(reasons)


def random_pool_status(row: pd.Series) -> str:
    if pd.isna(row.get("random_pool_split_diff_q95", np.nan)):
        return "not_available_no_replicate_summary"
    if bool(row["random_pool_max_pass_0p004"]):
        return "pass_all_tested"
    if bool(row["random_pool_q95_pass_0p004"]):
        return "pass_q95_but_some_tested_splits_fail"
    return "fail_q95"


def plot_existing_pass_grid(summary: pd.DataFrame, out_dir: Path) -> None:
    pivot = (
        summary.pivot(index="rule", columns="radius", values="existing_unit_pass_fraction")
        .reindex(RULES)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)), [LABELS[r] for r in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_title("Existing unit split pass fraction")
    ax.set_xlabel("radius")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val < 0.65 else "black")
    fig.colorbar(im, ax=ax, label="pass fraction")
    savefig(out_dir / "fig01_existing_unit_split_pass_grid.png")


def plot_random_pool_grid(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    pivot = (
        summary.pivot(index="rule", columns="radius", values="random_pool_q95_pass_fraction")
        .reindex(RULES)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    im = ax.imshow(masked, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)), [LABELS[r] for r in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_title("Random replicate-pool split q95 pass fraction")
    ax.set_xlabel("radius")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="q95 pass fraction")
    savefig(out_dir / "fig02_random_pool_q95_pass_grid.png")


def plot_existing_vs_random(random_df: pd.DataFrame, out_dir: Path) -> None:
    if random_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for rule in RULES:
        sub = random_df[random_df["rule"] == rule]
        if sub.empty:
            continue
        ax.scatter(
            sub["split_logZ_per_P_diff"],
            sub["random_pool_split_diff_q95"],
            s=24,
            alpha=0.72,
            color=COLORS[rule],
            label=LABELS[rule],
        )
    lim = max(
        SPLIT_GATE * 1.25,
        float(np.nanmax(random_df[["split_logZ_per_P_diff", "random_pool_split_diff_q95"]].to_numpy())) * 1.05,
    )
    ax.plot([0, lim], [0, lim], color="#777777", linewidth=1.0, linestyle="--")
    ax.axhline(SPLIT_GATE, color="#222222", linewidth=1.0, linestyle=":")
    ax.axvline(SPLIT_GATE, color="#222222", linewidth=1.0, linestyle=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("existing split logZ/P diff")
    ax.set_ylabel("random pool split logZ/P diff q95")
    ax.set_title("Existing split vs random replicate-pool split")
    ax.legend(frameon=False, fontsize=8)
    savefig(out_dir / "fig03_existing_vs_random_pool_split_diff.png")


def plot_failure_counts(unit_df: pd.DataFrame, random_df: pd.DataFrame, out_dir: Path) -> None:
    existing = (
        unit_df.loc[~unit_df["unit_pass_existing_split_and_ess"]]
        .groupby("rule")
        .size()
        .reindex(RULES, fill_value=0)
    )
    random_fail = (
        random_df.loc[~random_df["random_pool_q95_pass_0p004"]]
        .groupby("rule")
        .size()
        .reindex(RULES, fill_value=0)
        if not random_df.empty
        else pd.Series(0, index=RULES)
    )
    x = np.arange(len(RULES))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.35
    ax.bar(x - width / 2, existing.to_numpy(), width, label="existing split fail", color="#6b7f95")
    ax.bar(x + width / 2, random_fail.to_numpy(), width, label="random q95 fail", color="#b45a4d")
    ax.set_xticks(x, [LABELS[r] for r in RULES], rotation=15, ha="right")
    ax.set_ylabel("unit count")
    ax.set_title("Unit-level split failures")
    ax.legend(frameon=False)
    savefig(out_dir / "fig04_split_failure_counts_by_rule.png")


def main() -> None:
    tmp = prepare_output_dir()
    try:
        df = pd.read_csv(SOURCE_UNITS)
        p_dim = infer_parameter_dimension(df)

        unit_df = df.copy()
        unit_df["p_dimension"] = p_dim
        unit_df["finite_pass"] = bool_series(unit_df["finite"])
        unit_df["ess_pass_0p04"] = unit_df["ess_fraction"].astype(float) >= ESS_GATE
        unit_df["existing_split_pass_0p004"] = unit_df["split_logZ_per_P_diff"].astype(float) <= SPLIT_GATE
        unit_df["unit_pass_existing_split_and_ess"] = (
            unit_df["finite_pass"] & unit_df["ess_pass_0p04"] & unit_df["existing_split_pass_0p004"]
        )
        unit_df["random_pool_available"] = unit_df["replicate_summaries"].notna()
        unit_df["existing_fail_reason"] = unit_df.apply(existing_fail_reason, axis=1)
        unit_df["sample_kind"] = np.where(unit_df["replicates"].notna(), "replicate_fallback", "baseline_4096")

        rng = np.random.default_rng(RANDOM_SEED)
        random_rows: list[dict[str, object]] = []
        for _, row in unit_df[unit_df["replicate_summaries"].notna()].iterrows():
            reps = parse_replicates(row["replicate_summaries"])
            logz = np.asarray([float(rep["logZ_inf_full"]) for rep in reps], dtype=float)
            diffs = random_pool_diffs(logz, p_dim, rng)
            if diffs.size == 0:
                stats = {k: np.nan for k in ["mean", "median", "q90", "q95", "q99", "max", "pass_rate"]}
            else:
                stats = {
                    "mean": float(np.mean(diffs)),
                    "median": float(np.median(diffs)),
                    "q90": float(np.quantile(diffs, 0.90)),
                    "q95": float(np.quantile(diffs, 0.95)),
                    "q99": float(np.quantile(diffs, 0.99)),
                    "max": float(np.max(diffs)),
                    "pass_rate": float(np.mean(diffs <= SPLIT_GATE)),
                }
            random_rows.append(
                {
                    "rule": row["rule"],
                    "ref_id": int(row["ref_id"]),
                    "radius": float(row["radius"]),
                    "sample_kind": row["sample_kind"],
                    "fallback_policy_name": row.get("fallback_policy_name", ""),
                    "n_samples_each": row.get("n_samples_each", np.nan),
                    "n_samples_total": row.get("n_samples_total", np.nan),
                    "replicates": int(row["replicates"]),
                    "random_pool_splits": int(diffs.size),
                    "random_seed_base": RANDOM_SEED,
                    "p_dimension": p_dim,
                    "split_logZ_per_P_diff": row["split_logZ_per_P_diff"],
                    "existing_split_pass_0p004": bool(row["existing_split_pass_0p004"]),
                    "ess_fraction": row["ess_fraction"],
                    "weighted_ce": row["weighted_ce"],
                    "weighted_error": row["weighted_error"],
                    "replicate_logZ_per_P_range": row.get("replicate_logZ_per_P_range", np.nan),
                    "replicate_split_logZ_per_P_diff_max": row.get("replicate_split_logZ_per_P_diff_max", np.nan),
                    "random_pool_split_diff_mean": stats["mean"],
                    "random_pool_split_diff_median": stats["median"],
                    "random_pool_split_diff_q90": stats["q90"],
                    "random_pool_split_diff_q95": stats["q95"],
                    "random_pool_split_diff_q99": stats["q99"],
                    "random_pool_split_diff_max": stats["max"],
                    "random_pool_pass_rate_0p004": stats["pass_rate"],
                    "random_pool_q95_pass_0p004": bool(stats["q95"] <= SPLIT_GATE) if pd.notna(stats["q95"]) else False,
                    "random_pool_max_pass_0p004": bool(stats["max"] <= SPLIT_GATE) if pd.notna(stats["max"]) else False,
                }
            )

        random_df = pd.DataFrame(random_rows)
        if not random_df.empty:
            random_df["random_pool_status"] = random_df.apply(random_pool_status, axis=1)
            merge_cols = [
                "rule",
                "ref_id",
                "radius",
                "random_pool_split_diff_q95",
                "random_pool_split_diff_max",
                "random_pool_pass_rate_0p004",
                "random_pool_q95_pass_0p004",
                "random_pool_max_pass_0p004",
                "random_pool_status",
            ]
            unit_df = unit_df.merge(random_df[merge_cols], on=["rule", "ref_id", "radius"], how="left")
        else:
            unit_df["random_pool_status"] = "not_available_no_replicate_summary"

        unit_cols = [
            "rule",
            "ref_id",
            "radius",
            "sample_kind",
            "n_samples",
            "n_samples_each",
            "n_samples_total",
            "replicates",
            "p_dimension",
            "finite_pass",
            "ess_fraction",
            "ess_pass_0p04",
            "split_logZ_per_P_diff",
            "existing_split_pass_0p004",
            "unit_pass_existing_split_and_ess",
            "existing_fail_reason",
            "random_pool_available",
            "random_pool_split_diff_q95",
            "random_pool_split_diff_max",
            "random_pool_pass_rate_0p004",
            "random_pool_q95_pass_0p004",
            "random_pool_max_pass_0p004",
            "random_pool_status",
            "weighted_ce",
            "weighted_error",
            "fallback_policy_name",
            "unit_summary_path",
        ]
        unit_out = unit_df[unit_cols].sort_values(["rule", "ref_id", "radius"]).reset_index(drop=True)

        passed_existing = unit_out[unit_out["unit_pass_existing_split_and_ess"]].copy()
        failed_existing = unit_out[~unit_out["unit_pass_existing_split_and_ess"]].copy()

        existing_summary = (
            unit_out.groupby(["rule", "radius"], as_index=False)
            .agg(
                unit_count=("ref_id", "count"),
                existing_unit_pass_count=("unit_pass_existing_split_and_ess", "sum"),
                existing_split_fail_count=("existing_split_pass_0p004", lambda s: int((~s).sum())),
                finite_pass_count=("finite_pass", "sum"),
                ess_pass_count=("ess_pass_0p04", "sum"),
                max_split_logZ_per_P_diff=("split_logZ_per_P_diff", "max"),
                q05_ess_fraction=("ess_fraction", lambda s: float(np.quantile(s, 0.05))),
                max_weighted_ce=("weighted_ce", "max"),
                max_weighted_error=("weighted_error", "max"),
            )
            .sort_values(["rule", "radius"])
        )
        existing_summary["existing_unit_pass_fraction"] = (
            existing_summary["existing_unit_pass_count"] / existing_summary["unit_count"]
        )
        existing_summary["rule_radius_existing_all_units_pass"] = (
            existing_summary["existing_unit_pass_count"] == existing_summary["unit_count"]
        )

        random_summary = pd.DataFrame()
        if not random_df.empty:
            random_summary = (
                random_df.groupby(["rule", "radius"], as_index=False)
                .agg(
                    random_pool_unit_count=("ref_id", "count"),
                    random_pool_q95_pass_count=("random_pool_q95_pass_0p004", "sum"),
                    random_pool_max_pass_count=("random_pool_max_pass_0p004", "sum"),
                    random_pool_q95_diff_max=("random_pool_split_diff_q95", "max"),
                    random_pool_diff_max=("random_pool_split_diff_max", "max"),
                    random_pool_pass_rate_min=("random_pool_pass_rate_0p004", "min"),
                    existing_split_diff_max=("split_logZ_per_P_diff", "max"),
                )
                .sort_values(["rule", "radius"])
            )
            random_summary["random_pool_q95_pass_fraction"] = (
                random_summary["random_pool_q95_pass_count"] / random_summary["random_pool_unit_count"]
            )
            random_summary["random_pool_max_pass_fraction"] = (
                random_summary["random_pool_max_pass_count"] / random_summary["random_pool_unit_count"]
            )

        random_passed = (
            random_df[random_df["random_pool_q95_pass_0p004"]].copy()
            if not random_df.empty
            else pd.DataFrame()
        )
        random_failed = (
            random_df[~random_df["random_pool_q95_pass_0p004"]].copy()
            if not random_df.empty
            else pd.DataFrame()
        )
        random_overlap = (
            pd.crosstab(
                random_df["existing_split_pass_0p004"],
                random_df["random_pool_q95_pass_0p004"],
                rownames=["existing_split_pass_0p004"],
                colnames=["random_pool_q95_pass_0p004"],
            )
            .reset_index()
            if not random_df.empty
            else pd.DataFrame()
        )
        random_new_failures = (
            random_df[random_df["existing_split_pass_0p004"] & ~random_df["random_pool_q95_pass_0p004"]]
            .sort_values(["rule", "radius", "ref_id"])
            .copy()
            if not random_df.empty
            else pd.DataFrame()
        )
        random_rescued = (
            random_df[~random_df["existing_split_pass_0p004"] & random_df["random_pool_q95_pass_0p004"]]
            .sort_values(["rule", "radius", "ref_id"])
            .copy()
            if not random_df.empty
            else pd.DataFrame()
        )

        write_csv(unit_out, tmp / "unit_ref_radius_split_qc_table.csv")
        write_csv(passed_existing, tmp / "passed_ref_radius_combinations_existing_split.csv")
        write_csv(failed_existing, tmp / "failed_ref_radius_combinations_existing_split.csv")
        write_csv(existing_summary, tmp / "rule_radius_existing_split_summary.csv")
        write_csv(random_df, tmp / "random_pool_split_diagnostics_replicate_units.csv")
        write_csv(random_summary, tmp / "random_pool_rule_radius_summary.csv")
        write_csv(random_passed, tmp / "random_pool_passed_ref_radius_combinations_q95.csv")
        write_csv(random_failed, tmp / "random_pool_failed_ref_radius_combinations_q95.csv")
        write_csv(random_overlap, tmp / "random_pool_existing_vs_random_q95_crosstab.csv")
        write_csv(random_new_failures, tmp / "random_pool_new_failures_despite_existing_pass.csv")
        write_csv(random_rescued, tmp / "random_pool_existing_failures_rescued_by_random_q95.csv")

        plot_existing_pass_grid(existing_summary, tmp / "figures")
        plot_random_pool_grid(random_summary, tmp / "figures")
        plot_existing_vs_random(random_df, tmp / "figures")
        plot_failure_counts(unit_out, random_df, tmp / "figures")

        existing_by_kind = (
            unit_out.groupby("sample_kind")
            .agg(
                units=("ref_id", "count"),
                existing_split_pass=("existing_split_pass_0p004", "sum"),
                existing_split_fail=("existing_split_pass_0p004", lambda s: int((~s).sum())),
                unit_qc_pass=("unit_pass_existing_split_and_ess", "sum"),
            )
            .reset_index()
        )
        existing_by_rule = (
            unit_out.groupby("rule")
            .agg(
                units=("ref_id", "count"),
                existing_split_pass=("existing_split_pass_0p004", "sum"),
                existing_split_fail=("existing_split_pass_0p004", lambda s: int((~s).sum())),
                unit_qc_pass=("unit_pass_existing_split_and_ess", "sum"),
            )
            .reindex(RULES)
            .reset_index()
        )
        random_by_rule = (
            random_df.groupby("rule")
            .agg(
                random_units=("ref_id", "count"),
                q95_pass=("random_pool_q95_pass_0p004", "sum"),
                q95_fail=("random_pool_q95_pass_0p004", lambda s: int((~s).sum())),
                max_pass=("random_pool_max_pass_0p004", "sum"),
                max_fail=("random_pool_max_pass_0p004", lambda s: int((~s).sum())),
                median_pass_rate=("random_pool_pass_rate_0p004", "median"),
            )
            .reindex(RULES)
            .reset_index()
            if not random_df.empty
            else pd.DataFrame()
        )

        worst_existing = failed_existing.sort_values("split_logZ_per_P_diff", ascending=False)[
            ["rule", "ref_id", "radius", "sample_kind", "split_logZ_per_P_diff", "ess_fraction", "weighted_ce", "weighted_error"]
        ].head(12)
        worst_random = (
            random_failed.sort_values("random_pool_split_diff_q95", ascending=False)[
                [
                    "rule",
                    "ref_id",
                    "radius",
                    "replicates",
                    "n_samples_total",
                    "split_logZ_per_P_diff",
                    "random_pool_split_diff_q95",
                    "random_pool_split_diff_max",
                    "random_pool_pass_rate_0p004",
                ]
            ].head(12)
            if not random_failed.empty
            else pd.DataFrame()
        )

        report = f"""# MNIST10 Random Pool Split Diagnostics

This is a local-only diagnostic output. It reads existing MNIST10 PM-SAIS unit summaries from:

`{SOURCE_UNITS}`

and writes only under:

`{OUT}`

## What Was Tested

- Existing saved split: every unit already has `split_logZ_per_P_diff`; pass means finite, `ESS >= {ESS_GATE}`, and `split_logZ/P <= {SPLIT_GATE}`.
- Random pool split: only units with saved `replicate_summaries` can be re-split. For each such unit, replicate-level `logZ_inf_full` values were randomly divided into two balanced pools {RANDOM_SPLITS} times, combined with log-mean-exp, and compared as `abs(logZ_pool_a - logZ_pool_b) / P`.
- Inferred parameter dimension: `P = {p_dim:.0f}`.

## Important Limitation

The baseline `n_samples=4096` units do not store particle-level log weights or directions. Therefore true random re-splitting of the same particle pool is not reproducible from saved artifacts for those {int((unit_out['sample_kind'] == 'baseline_4096').sum())} baseline units. Those rows are still reported with the existing saved split only.

The random-pool diagnostic is available for the {len(random_df)} replicate/fallback units. It tests sensitivity to how replicate pools are grouped; it is not a substitute for particle-level random split unless future runs retain particle arrays or implement random split saving inside PM-SAIS.

## Existing Split Summary

| quantity | value |
| --- | ---: |
| total units | {len(unit_out)} |
| finite units | {int(unit_out['finite_pass'].sum())} |
| ESS pass units (`>= {ESS_GATE}`) | {int(unit_out['ess_pass_0p04'].sum())} |
| existing split pass units (`<= {SPLIT_GATE}`) | {int(unit_out['existing_split_pass_0p004'].sum())} |
| existing split fail units | {int((~unit_out['existing_split_pass_0p004']).sum())} |
| existing finite+ESS+split pass units | {int(unit_out['unit_pass_existing_split_and_ess'].sum())} |

### Existing split by sample kind

{markdown_table(existing_by_kind)}

### Existing split by rule

{markdown_table(existing_by_rule)}

## Random Replicate-Pool Split Summary

Pass is shown two ways:

- `q95_pass`: the 95th percentile of random split differences is `<= {SPLIT_GATE}`.
- `max_pass`: every tested random split is `<= {SPLIT_GATE}`.

{markdown_table(random_by_rule) if not random_by_rule.empty else 'No replicate summaries were available.'}

### Existing split vs random-pool q95 overlap

{markdown_table(random_overlap) if not random_overlap.empty else 'No replicate summaries were available.'}

## Worst Existing-Split Failed Units

{markdown_table(worst_existing)}

## Worst Random-Pool q95 Failed Units

{markdown_table(worst_random) if not worst_random.empty else 'No random-pool q95 failures.'}

## Files

- `unit_ref_radius_split_qc_table.csv`: all unit-level ref/radius rows with existing split pass/fail and random-pool columns when available.
- `passed_ref_radius_combinations_existing_split.csv`: unit ref/radius combinations passing finite + ESS + existing split.
- `failed_ref_radius_combinations_existing_split.csv`: unit ref/radius combinations failing finite + ESS + existing split.
- `random_pool_split_diagnostics_replicate_units.csv`: random-pool split diagnostics for replicate/fallback units.
- `random_pool_passed_ref_radius_combinations_q95.csv`: replicate/fallback ref/radius combinations passing random-pool q95.
- `random_pool_failed_ref_radius_combinations_q95.csv`: replicate/fallback ref/radius combinations failing random-pool q95.
- `random_pool_existing_vs_random_q95_crosstab.csv`: overlap of existing 2-way split pass/fail and random-pool q95 pass/fail.
- `random_pool_new_failures_despite_existing_pass.csv`: replicate/fallback units that pass existing split but fail random-pool q95.
- `random_pool_existing_failures_rescued_by_random_q95.csv`: replicate/fallback units that fail existing split but pass random-pool q95.
- `rule_radius_existing_split_summary.csv`: existing split aggregation by rule/radius.
- `random_pool_rule_radius_summary.csv`: random-pool aggregation by rule/radius.
- `figures/fig01_existing_unit_split_pass_grid.png`
- `figures/fig02_random_pool_q95_pass_grid.png`
- `figures/fig03_existing_vs_random_pool_split_diff.png`
- `figures/fig04_split_failure_counts_by_rule.png`

## Interpretation

The current saved outputs do not imply that all split-logZ is broken. They show localized instability: {int((~unit_out['existing_split_pass_0p004']).sum())} of {len(unit_out)} units fail the existing split gate, while ESS is comfortably above the existing threshold for all units. Random replicate-pool splitting asks whether those failures are tied to one arbitrary two-way grouping. When random q95 fails, the ref/radius combination is unstable under many groupings and should be treated as no-claim under split-logZ QC.
"""
        write_text(report, tmp / "REPORT.md")
        finalize_output_dir(tmp)
        print(OUT)
        print(f"existing_split_pass={int(unit_out['existing_split_pass_0p004'].sum())}/{len(unit_out)}")
        print(f"existing_split_fail={int((~unit_out['existing_split_pass_0p004']).sum())}")
        if not random_df.empty:
            print(f"random_pool_q95_pass={int(random_df['random_pool_q95_pass_0p004'].sum())}/{len(random_df)}")
            print(f"random_pool_q95_fail={int((~random_df['random_pool_q95_pass_0p004']).sum())}")
            print(f"random_pool_max_pass={int(random_df['random_pool_max_pass_0p004'].sum())}/{len(random_df)}")
    except Exception:
        if tmp.exists():
            # The temp directory is intentionally left for debugging if cleanup fails.
            import shutil

            shutil.rmtree(tmp)
        raise


if __name__ == "__main__":
    main()
