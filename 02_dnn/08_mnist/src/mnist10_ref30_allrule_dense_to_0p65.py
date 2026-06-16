from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnist10_reference_family_analysis import (
    BOOTSTRAP_SD_GATE,
    ESS_GATE,
    SPLIT_GATE,
    add_delta_phi,
    ensure_dir,
    selector_qc,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_ALLRULE_DENSE = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_dense_0p010_to_0p080"
SOURCE_ALLRULE_SPARSE = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
PILOT_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_ref30_selector_targeted_pilot"
RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_ref30_allrule_dense_to_0p650"

ANALYSIS_ROOT = SOURCE_ALLRULE_SPARSE / "07_reference_family_analysis"
SELECTOR = "dense_qc_stable_ref30"
RULES = ["low_tv_spectral_teacher", "real_even_odd", "teacher_nn", "random_label"]
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def load_unit_rows(run_root: Path, source_label: str, source_rank: int) -> pd.DataFrame:
    unit_root = run_root / "05_pool2_pm_sais_sampling" / "unit_summaries"
    rows: list[dict[str, Any]] = []
    for path in sorted(unit_root.rglob("unit_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unit_summary_path"] = rel(path)
        payload["overlay_source"] = source_label
        payload["_source_rank"] = int(source_rank)
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in [
        "split_id",
        "ref_id",
        "radius",
        "ess_fraction",
        "split_logZ_per_P_diff",
        "logZ_inf_full",
        "weighted_ce",
        "weighted_error",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rule"] = df["rule"].astype(str)
    df["radius_key"] = df["radius"].round(4)
    return df


def build_overlay_units() -> pd.DataFrame:
    sources = [
        (SOURCE_ALLRULE_DENSE, "allrule_dense_0p01_to_0p08", 0),
        (SOURCE_ALLRULE_SPARSE, "allrule_sparse_0p01_to_2p50", 1),
        (PILOT_ROOT, "targeted_selector_pilot", 2),
    ]
    frames = [load_unit_rows(root, label, rank) for root, label, rank in sources]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError("No unit_summary.json sources found for dense-to-0.65 overlay.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    radius_keys = {round(float(r), 4) for r in RADII}
    df = df[df["rule"].isin(RULES) & df["radius_key"].isin(radius_keys)].copy()
    df = df.sort_values(
        ["split_id", "rule", "ref_id", "radius_key", "_source_rank"],
        ascending=[True, True, True, True, True],
    )
    df = df.drop_duplicates(["split_id", "rule", "ref_id", "radius_key"], keep="last")
    df = add_delta_phi(df)
    return df.sort_values(["rule", "ref_id", "radius"]).reset_index(drop=True)


def selected_membership() -> pd.DataFrame:
    path = ANALYSIS_ROOT / "selector_membership.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[(df["selector"] == SELECTOR) & (df["rule"].isin(RULES))].copy()
    if df.empty:
        raise ValueError(f"No selector membership found for {SELECTOR}")
    return df.sort_values(["selector", "rule", "ref_id"]).reset_index(drop=True)


def build_missing_manifest(unit_df: pd.DataFrame, selectors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    indexed = {
        (str(row.rule), int(row.ref_id), round(float(row.radius), 4))
        for row in unit_df[["rule", "ref_id", "radius"]].dropna().itertuples()
    }
    for (selector, rule), refs_sub in selectors.groupby(["selector", "rule"]):
        for ref_id in sorted(refs_sub["ref_id"].astype(int).unique()):
            for radius in RADII:
                key = (str(rule), int(ref_id), round(float(radius), 4))
                if key in indexed:
                    continue
                rows.append(
                    {
                        "selector": selector,
                        "rule": str(rule),
                        "ref_id": int(ref_id),
                        "radius": float(radius),
                        "radius_token": f"{radius:.4f}".replace(".", "p"),
                    }
                )
    return pd.DataFrame(rows, columns=["selector", "rule", "ref_id", "radius", "radius_token"])


def plot_outputs(phi_df: pd.DataFrame, qc_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for rule, sub in phi_df[phi_df["qc_pass"]].groupby("rule"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["delta_phi_energy"], marker="o", linewidth=1.4, label=rule)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("delta phi energy")
    ax.set_title("MNIST10 ref30 all-rule phi(d)_energy QC-pass line to 0.65")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.35, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_phi_energy_qc_pass_allrule_dense_to_0p65.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=False)
    for ax, rule in zip(axes.ravel(), RULES):
        sub = phi_df[phi_df["rule"] == rule].sort_values("radius")
        if sub.empty:
            continue
        ax.plot(sub["radius"], sub["delta_phi_energy"], linewidth=1.1, alpha=0.75)
        pass_sub = sub[sub["qc_pass"]]
        missing_sub = sub[sub["claim_status"].astype(str).str.contains("missing", na=False)]
        fail_sub = sub[(~sub["qc_pass"]) & (~sub.index.isin(missing_sub.index))]
        if not pass_sub.empty:
            ax.scatter(pass_sub["radius"], pass_sub["delta_phi_energy"], s=28, marker="o", label="QC pass")
        if not missing_sub.empty:
            ax.scatter(missing_sub["radius"], missing_sub["delta_phi_energy"], s=38, marker="s", label="missing")
        if not fail_sub.empty:
            ax.scatter(fail_sub["radius"], fail_sub["delta_phi_energy"], s=44, marker="x", label="QC fail")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
        ax.set_title(rule)
        ax.set_xlabel("d_raw")
        ax.set_ylabel("delta phi energy")
        ax.grid(True, linewidth=0.35, alpha=0.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("MNIST10 ref30 all-rule phi(d)_energy dense-to-0.65 diagnostic")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_phi_energy_allrule_dense_to_0p65_diagnostic_panels.png", dpi=190)
    plt.close(fig)

    pivot = qc_df.pivot_table(index="rule", columns="radius", values="qc_pass", aggfunc="max").reindex(RULES)
    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    im = ax.imshow(pivot.fillna(False).astype(float).to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xticks(range(len(pivot.columns)), [f"{float(c):.3f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_title("Selector QC pass heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_qc_pass_heatmap_dense_to_0p65.png", dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, unit_df: pd.DataFrame, qc_df: pd.DataFrame, phi_df: pd.DataFrame, missing_df: pd.DataFrame) -> None:
    common_pass = sorted(
        set.intersection(
            *[
                set(qc_df[(qc_df["rule"] == rule) & (qc_df["qc_pass"])]["radius"].round(4).tolist())
                for rule in RULES
            ]
        )
    )
    failed = qc_df[~qc_df["qc_pass"]].sort_values(["rule", "radius"]).copy()
    output_rows = [
        "| file | rows |",
        "| --- | ---: |",
        f"| overlay_unit_summary_long.csv | {len(unit_df)} |",
        f"| selector_qc_by_rule_radius.csv | {len(qc_df)} |",
        f"| selector_phi_by_rule_radius.csv | {len(phi_df)} |",
        f"| missing_selector_units.csv | {len(missing_df)} |",
    ]
    qc_preview_cols = ["rule", "radius", "observed_ref_count", "missing_ref_count", "max_split_logZ_per_P_diff", "bootstrap_sd_phi", "claim_status"]
    if failed.empty:
        failed_preview = "none"
    else:
        preview = failed[qc_preview_cols].head(30).copy()
        table_lines = ["| " + " | ".join(qc_preview_cols) + " |", "| " + " | ".join(["---"] * len(qc_preview_cols)) + " |"]
        for row in preview.to_dict("records"):
            table_lines.append("| " + " | ".join(str(row.get(col, "")) for col in qc_preview_cols) + " |")
        failed_preview = "\n".join(table_lines)
    report = f"""# MNIST10 Ref30 All-Rule Dense To 0.65

Selector: `{SELECTOR}`

Radii: `{", ".join(f"{r:.3f}" for r in RADII)}`

Rules: `{", ".join(RULES)}`

## Current Decision

- Common QC-pass radii across all rules: `{", ".join(f"{r:.4f}" for r in common_pass) if common_pass else "none"}`
- Missing selected units: `{len(missing_df)}`
- Failed or incomplete selector/radius rows: `{len(failed)}`
- Split gate: `{SPLIT_GATE}`
- ESS gate: `{ESS_GATE}`
- Bootstrap SD gate: `{BOOTSTRAP_SD_GATE}`

## Outputs

{chr(10).join(output_rows)}

## Figures

- `figures/fig01_phi_energy_qc_pass_allrule_dense_to_0p65.png`
- `figures/fig02_phi_energy_allrule_dense_to_0p65_diagnostic_panels.png`
- `figures/fig03_qc_pass_heatmap_dense_to_0p65.png`

## First Failed Or Incomplete Rows

{failed_preview}
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def write_blocked(out_dir: Path, qc_df: pd.DataFrame, missing_df: pd.DataFrame) -> None:
    failed = qc_df[~qc_df["qc_pass"]].sort_values(["rule", "radius"]).copy()
    if not missing_df.empty:
        reason = f"{len(missing_df)} selected reference/radius units are missing for {SELECTOR}."
        next_action = (
            "Run targeted Stage05 sampling for rows in missing_selector_units.csv, then rerun this overlay script. "
            "Use the existing hard-shell PM-SAIS sampler; do not promote diagnostic figures until all selected rows pass QC."
        )
    elif not failed.empty:
        worst = failed.sort_values("max_split_logZ_per_P_diff", ascending=False).head(1).iloc[0].to_dict()
        reason = (
            "All selected units are present, but selector-level QC failed. "
            f"Worst row: rule={worst.get('rule')} radius={float(worst.get('radius')):.4f} "
            f"max_split_logZ_per_P_diff={worst.get('max_split_logZ_per_P_diff')}."
        )
        next_action = (
            "Stop promotion. Inspect failed selector/radius rows and run targeted stronger PM-SAIS only if the reference-family law remains predeclared."
        )
    else:
        return
    lines = [
        "# Stage Blocked",
        "",
        "Stage: `06_results_figures`",
        "",
        f"Exact reason: {reason}",
        "",
        f"Next safe action: {next_action}",
        "",
        "This run only writes a new overlay/report path and does not modify retained production outputs.",
        "",
    ]
    (out_dir / "STAGE_BLOCKED.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build dense-to-0.65 all-rule ref30 energetic phi(d) overlay.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Write diagnostic outputs and return success even if missing/QC-failed.")
    args = parser.parse_args(argv)

    out_dir = ensure_dir(RUN_ROOT / "06_results_figures")
    unit_df = build_overlay_units()
    selectors = selected_membership()
    missing_df = build_missing_manifest(unit_df, selectors)
    qc_df, phi_df = selector_qc(unit_df, selectors, selector_size=30)
    radius_keys = {round(float(r), 4) for r in RADII}
    qc_df = qc_df[qc_df["radius"].round(4).isin(radius_keys)].sort_values(["rule", "radius"]).reset_index(drop=True)
    phi_df = phi_df[phi_df["radius"].round(4).isin(radius_keys)].sort_values(["rule", "radius"]).reset_index(drop=True)

    write_csv(out_dir / "overlay_unit_summary_long.csv", unit_df)
    write_csv(out_dir / "selector_membership.csv", selectors)
    write_csv(out_dir / "missing_selector_units.csv", missing_df)
    write_csv(out_dir / "selector_qc_by_rule_radius.csv", qc_df)
    write_csv(out_dir / "selector_phi_by_rule_radius.csv", phi_df)
    plot_outputs(phi_df, qc_df, out_dir)
    write_report(out_dir, unit_df, qc_df, phi_df, missing_df)

    all_qc_pass = bool(missing_df.empty and len(qc_df) == len(RULES) * len(RADII) and qc_df["qc_pass"].all())
    payload = {
        "status": "pass" if all_qc_pass else "blocked",
        "selector": SELECTOR,
        "rules": RULES,
        "radii": RADII,
        "unit_rows": int(len(unit_df)),
        "missing_units": int(len(missing_df)),
        "qc_rows": int(len(qc_df)),
        "qc_pass_rows": int(qc_df["qc_pass"].sum()) if "qc_pass" in qc_df.columns else 0,
        "all_qc_pass": all_qc_pass,
        "source_runs": {
            "allrule_dense": rel(SOURCE_ALLRULE_DENSE),
            "allrule_sparse": rel(SOURCE_ALLRULE_SPARSE),
            "targeted_pilot": rel(PILOT_ROOT),
        },
    }
    write_json(out_dir / "QC_STATUS.json", payload)

    if not all_qc_pass:
        write_blocked(out_dir, qc_df, missing_df)
        print(json.dumps(payload, indent=2, sort_keys=True, default=json_default))
        return 0 if args.allow_incomplete else 2
    blocked_path = out_dir / "STAGE_BLOCKED.md"
    if blocked_path.exists():
        blocked_path.unlink()
    print(json.dumps(payload, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
