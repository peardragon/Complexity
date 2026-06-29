from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "raw_outputs"
RUNS = {
    "60ref": RAW_ROOT / "refpool1024_all_radii_60ref",
    "90ref": RAW_ROOT / "refpool1024_all_radii_90ref",
}

SPLIT_GATE = 0.004
ESS_GATE = 0.04
FINITE_FRACTION_GATE = 0.95
BOOTSTRAP_GATE = 0.012


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def join_ids(values: pd.Series) -> str:
    ids = sorted({int(value) for value in values.dropna().tolist()})
    return ",".join(f"{value:03d}" for value in ids)


def quantile(values: pd.Series, q: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def fail_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["sampling_complete"]):
        reasons.append("missing_units")
    if not bool(row["finite_fraction_ok"]):
        reasons.append("finite_fraction")
    if not bool(row["q05_ess_ok"]):
        reasons.append("q05_ess")
    if not bool(row["split_max_ok"]):
        reasons.append("max_split_logZ_per_P_diff")
    if not bool(row["bootstrap_ok"]):
        reasons.append("bootstrap_sd")
    return ";".join(reasons) if reasons else "pass"


def build_tables(tag: str, run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sampling_dir = run_root / "05_pool2_pm_sais_sampling"
    results_dir = run_root / "06_results_figures"
    unit = read_csv(sampling_dir / "shell_summary_by_unit.csv")
    qc = read_csv(sampling_dir / "qc_diagnostics_by_rule_radius.csv")
    phi = read_csv(results_dir / "phi_by_rule_radius.csv")

    unit["run"] = tag
    unit["finite_ok"] = unit["finite"].astype(bool) & np.isfinite(
        pd.to_numeric(unit["logZ_inf_full"], errors="coerce")
    )
    unit["ess_ok"] = pd.to_numeric(unit["ess_fraction"], errors="coerce") >= ESS_GATE
    unit["split_ok"] = pd.to_numeric(unit["split_logZ_per_P_diff"], errors="coerce") <= SPLIT_GATE
    unit["unit_strict_qc_pass"] = unit["finite_ok"] & unit["ess_ok"] & unit["split_ok"]

    unit_summary = (
        unit.groupby(["rule", "radius"], sort=True)
        .agg(
            n_units=("rule", "size"),
            n_ref_observed=("ref_id", "nunique"),
            n_samples_min=("n_samples", "min"),
            n_samples_max=("n_samples", "max"),
            unit_strict_qc_pass_count=("unit_strict_qc_pass", "sum"),
            unit_finite_fail_count=("finite_ok", lambda s: int((~s).sum())),
            unit_ess_fail_count=("ess_ok", lambda s: int((~s).sum())),
            unit_split_fail_count=("split_ok", lambda s: int((~s).sum())),
            split_fail_ref_ids=("ref_id", lambda s: ""),
            split_logZ_per_P_diff_min=("split_logZ_per_P_diff", "min"),
            split_logZ_per_P_diff_median=("split_logZ_per_P_diff", "median"),
            split_logZ_per_P_diff_q90=("split_logZ_per_P_diff", lambda s: quantile(s, 0.90)),
            split_logZ_per_P_diff_q95=("split_logZ_per_P_diff", lambda s: quantile(s, 0.95)),
            split_logZ_per_P_diff_q99=("split_logZ_per_P_diff", lambda s: quantile(s, 0.99)),
            split_logZ_per_P_diff_max=("split_logZ_per_P_diff", "max"),
            ess_fraction_min=("ess_fraction", "min"),
            ess_fraction_q05=("ess_fraction", lambda s: quantile(s, 0.05)),
            ess_fraction_median=("ess_fraction", "median"),
        )
        .reset_index()
    )
    bad_ids = (
        unit[~unit["split_ok"]]
        .groupby(["rule", "radius"])["ref_id"]
        .apply(join_ids)
        .rename("split_fail_ref_ids")
        .reset_index()
    )
    unit_summary = unit_summary.drop(columns=["split_fail_ref_ids"]).merge(
        bad_ids, on=["rule", "radius"], how="left"
    )
    unit_summary["split_fail_ref_ids"] = unit_summary["split_fail_ref_ids"].fillna("")

    phi_cols = [
        "rule",
        "radius",
        "phi_energy_raw",
        "delta_phi_energy",
        "delta_phi_full",
    ]
    qc_cols = [
        "rule",
        "radius",
        "target_ref_count",
        "observed_ref_count",
        "missing_ref_count",
        "finite_unit_count",
        "finite_unit_fraction",
        "q05_ess_fraction",
        "max_split_logZ_per_P_diff",
        "bootstrap_sd_delta_phi_energy",
        "qc_diagnostic_pass",
        "sampling_status",
    ]
    table = (
        qc[qc_cols]
        .merge(phi[phi_cols], on=["rule", "radius"], how="left")
        .merge(unit_summary, on=["rule", "radius"], how="left")
    )
    table.insert(0, "run", tag)
    table["sampling_complete"] = (
        table["observed_ref_count"].eq(table["target_ref_count"])
        & table["missing_ref_count"].eq(0)
        & table["n_samples_min"].eq(table["n_samples_max"])
    )
    table["finite_fraction_ok"] = table["finite_unit_fraction"] >= FINITE_FRACTION_GATE
    table["q05_ess_ok"] = table["q05_ess_fraction"] >= ESS_GATE
    table["split_max_ok"] = table["max_split_logZ_per_P_diff"] <= SPLIT_GATE
    table["bootstrap_ok"] = table["bootstrap_sd_delta_phi_energy"] <= BOOTSTRAP_GATE
    table["strict_claim_qc_pass"] = table["qc_diagnostic_pass"].astype(bool)
    table["strict_claim_fail_reason"] = table.apply(fail_reason, axis=1)
    table["diagnostic_phi_available"] = table["sampling_complete"] & table["finite_fraction_ok"]
    table["unit_split_fail_fraction"] = table["unit_split_fail_count"] / table["target_ref_count"]
    table["refs_remaining_if_drop_split_fail"] = table["target_ref_count"] - table["unit_split_fail_count"]
    table["claim_status_updated"] = np.where(
        table["strict_claim_qc_pass"],
        "strict_claim_pass",
        np.where(
            table["diagnostic_phi_available"],
            "diagnostic_only_split_instability",
            "diagnostic_incomplete",
        ),
    )

    ref_summary = (
        unit.groupby(["rule", "ref_id"], sort=True)
        .agg(
            radius_count=("radius", "nunique"),
            split_fail_radius_count=("split_ok", lambda s: int((~s).sum())),
            finite_fail_radius_count=("finite_ok", lambda s: int((~s).sum())),
            ess_fail_radius_count=("ess_ok", lambda s: int((~s).sum())),
            split_logZ_per_P_diff_max=("split_logZ_per_P_diff", "max"),
            split_logZ_per_P_diff_median=("split_logZ_per_P_diff", "median"),
            split_logZ_per_P_diff_mean=("split_logZ_per_P_diff", "mean"),
            first_split_fail_radius=("radius", lambda s: float(unit.loc[s.index[~unit.loc[s.index, "split_ok"]], "radius"].min()) if (~unit.loc[s.index, "split_ok"]).any() else float("nan")),
        )
        .reset_index()
    )
    ref_summary.insert(0, "run", tag)

    removal_rows: list[dict[str, object]] = []
    for rule, sub in ref_summary.groupby("rule", sort=True):
        bad = sub[sub["split_fail_radius_count"] > 0].copy()
        good = sub[sub["split_fail_radius_count"] == 0].copy()
        top = bad.sort_values(
            ["split_fail_radius_count", "split_logZ_per_P_diff_max"],
            ascending=[False, False],
        ).head(15)
        removal_rows.append(
            {
                "run": tag,
                "rule": rule,
                "refs_total": int(len(sub)),
                "refs_with_any_split_fail": int(len(bad)),
                "refs_to_remove_for_all_radii_strict_pass": int(len(bad)),
                "refs_remaining_after_global_drop": int(len(good)),
                "refs_fail_5plus_radii": int((bad["split_fail_radius_count"] >= 5).sum()),
                "refs_fail_10plus_radii": int((bad["split_fail_radius_count"] >= 10).sum()),
                "refs_fail_20plus_radii": int((bad["split_fail_radius_count"] >= 20).sum()),
                "bad_ref_ids": join_ids(bad["ref_id"]),
                "top_bad_refs": "; ".join(
                    f"{int(row.ref_id):03d}:{int(row.split_fail_radius_count)}fail:max{float(row.split_logZ_per_P_diff_max):.5f}"
                    for row in top.itertuples()
                ),
            }
        )
    removal = pd.DataFrame(removal_rows)

    table = table[
        [
            "run",
            "rule",
            "radius",
            "target_ref_count",
            "observed_ref_count",
            "n_samples_min",
            "n_samples_max",
            "sampling_complete",
            "diagnostic_phi_available",
            "strict_claim_qc_pass",
            "claim_status_updated",
            "strict_claim_fail_reason",
            "unit_strict_qc_pass_count",
            "unit_split_fail_count",
            "unit_split_fail_fraction",
            "refs_remaining_if_drop_split_fail",
            "split_fail_ref_ids",
            "max_split_logZ_per_P_diff",
            "split_logZ_per_P_diff_q99",
            "split_logZ_per_P_diff_q95",
            "split_logZ_per_P_diff_q90",
            "split_logZ_per_P_diff_median",
            "q05_ess_fraction",
            "finite_unit_fraction",
            "bootstrap_sd_delta_phi_energy",
            "delta_phi_energy",
            "delta_phi_full",
            "phi_energy_raw",
        ]
    ].sort_values(["rule", "radius"])

    return table, ref_summary.sort_values(["rule", "ref_id"]), removal.sort_values(["rule"])


def write_report(out_dir: Path, combined: pd.DataFrame, removals: pd.DataFrame) -> None:
    lines = [
        "# Updated QC tables",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This update separates sampling completeness from strict claim QC.",
        "",
        "- `sampling_complete`: expected rule/ref/radius units exist and have a consistent sample count.",
        "- `diagnostic_phi_available`: sampling is complete and finite enough to use the phi value as a diagnostic curve point.",
        "- `strict_claim_qc_pass`: the old/formal all-reference claim gate, including `max_split_logZ_per_P_diff <= 0.004`.",
        "- `unit_split_fail_count`: number of references at that rule/radius whose single 1024-sample unit exceeded the split gate.",
        "",
        "## Run Summary",
        "",
        "| run | rows | sampling complete | diagnostic phi available | strict claim pass | unit split fails |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, sub in combined.groupby("run", sort=True):
        lines.append(
            f"| {run} | {len(sub)} | {int(sub['sampling_complete'].sum())} | "
            f"{int(sub['diagnostic_phi_available'].sum())} | {int(sub['strict_claim_qc_pass'].sum())} | "
            f"{int(sub['unit_split_fail_count'].sum())} |"
        )
    lines.extend(["", "## Global Reference Removal Check", ""])
    lines.extend(
        [
            "| run | rule | refs total | refs with any split fail | refs remaining after global drop |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in removals.to_dict("records"):
        lines.append(
            f"| {row['run']} | {row['rule']} | {int(row['refs_total'])} | "
            f"{int(row['refs_with_any_split_fail'])} | {int(row['refs_remaining_after_global_drop'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `derived/qc_table_updated_60ref.csv`",
            "- `derived/qc_table_updated_90ref.csv`",
            "- `derived/qc_table_updated_combined_60_90ref.csv`",
            "- `derived/ref_split_fail_summary_60ref.csv`",
            "- `derived/ref_split_fail_summary_90ref.csv`",
            "- `derived/ref_split_fail_removal_summary_combined_60_90ref.csv`",
            "",
        ]
    )
    (out_dir / "QC_TABLE_UPDATED_README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    tables: list[pd.DataFrame] = []
    ref_tables: list[pd.DataFrame] = []
    removals: list[pd.DataFrame] = []
    final_out_dir = RUNS["90ref"] / "06_results_figures"
    final_derived = ensure_dir(final_out_dir / "derived")
    for tag, run_root in RUNS.items():
        table, ref_summary, removal = build_tables(tag, run_root)
        run_derived = ensure_dir(run_root / "06_results_figures" / "derived")
        table.to_csv(run_derived / f"qc_table_updated_{tag}.csv", index=False)
        ref_summary.to_csv(run_derived / f"ref_split_fail_summary_{tag}.csv", index=False)
        removal.to_csv(run_derived / f"ref_split_fail_removal_summary_{tag}.csv", index=False)
        tables.append(table)
        ref_tables.append(ref_summary)
        removals.append(removal)

    combined = pd.concat(tables, ignore_index=True)
    combined_ref = pd.concat(ref_tables, ignore_index=True)
    combined_removal = pd.concat(removals, ignore_index=True)
    combined.to_csv(final_derived / "qc_table_updated_combined_60_90ref.csv", index=False)
    combined_ref.to_csv(final_derived / "ref_split_fail_summary_combined_60_90ref.csv", index=False)
    combined_removal.to_csv(final_derived / "ref_split_fail_removal_summary_combined_60_90ref.csv", index=False)
    write_report(final_out_dir, combined, combined_removal)

    print(f"wrote updated QC tables to {final_derived}")
    print(combined.groupby("run").agg(
        rows=("run", "size"),
        sampling_complete=("sampling_complete", "sum"),
        diagnostic_phi_available=("diagnostic_phi_available", "sum"),
        strict_claim_qc_pass=("strict_claim_qc_pass", "sum"),
        unit_split_fail_count=("unit_split_fail_count", "sum"),
    ).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
