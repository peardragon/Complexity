from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mnist10_reference_family_analysis import (
    DEFAULT_RUN_ROOT,
    ESS_GATE,
    SPLIT_GATE,
    add_delta_phi,
    ensure_dir,
    selector_qc,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = DEFAULT_RUN_ROOT / "07_reference_family_analysis"
DEFAULT_PILOT_ROOT = (
    ROOT
    / "runs"
    / "final"
    / "single_dataset_10x10_box_n_train_512_ref30_selector_targeted_pilot"
)
DEFAULT_OUTPUT_DIR = DEFAULT_PILOT_ROOT / "06_overlay_selector_qc"


DERIVED_COLUMNS = {
    "logZ_r0",
    "delta_phi_energy_unit",
    "delta_phi_full_unit",
    "unit_split_pass",
    "unit_ess_pass",
    "unit_qc_pass",
}


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def load_source_units(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in DERIVED_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["overlay_source"] = "source"
    return coerce_units(df)


def load_pilot_units(pilot_root: Path) -> pd.DataFrame:
    unit_root = pilot_root / "05_pool2_pm_sais_sampling" / "unit_summaries"
    rows: list[dict[str, Any]] = []
    for path in sorted(unit_root.rglob("unit_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unit_summary_path"] = str(path)
        payload["overlay_source"] = "pilot"
        rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"No pilot unit_summary.json files found under {unit_root}")
    return coerce_units(pd.DataFrame(rows))


def coerce_units(df: pd.DataFrame) -> pd.DataFrame:
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


def overlay_units(source_df: pd.DataFrame, pilot_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([source_df, pilot_df], ignore_index=True, sort=False)
    combined["_overlay_rank"] = np.where(combined["overlay_source"].eq("pilot"), 1, 0)
    combined = combined.sort_values(
        ["split_id", "rule", "ref_id", "radius_key", "_overlay_rank"],
        ascending=[True, True, True, True, True],
    )
    out = combined.drop_duplicates(["split_id", "rule", "ref_id", "radius_key"], keep="last").drop(columns=["_overlay_rank"])
    out = add_delta_phi(out)
    return out.sort_values(["rule", "split_id", "ref_id", "radius"]).reset_index(drop=True)


def write_report(
    out_dir: Path,
    selector: str,
    rule: str,
    radius: float,
    qc_df: pd.DataFrame,
    source_units: pd.DataFrame,
    pilot_units: pd.DataFrame,
    overlay_df: pd.DataFrame,
) -> None:
    row_match = qc_df[
        (qc_df["selector"] == selector)
        & (qc_df["rule"] == rule)
        & np.isclose(qc_df["radius"], radius)
    ]
    row = row_match.iloc[0].to_dict() if not row_match.empty else {}
    selected = overlay_df[
        (overlay_df["rule"] == rule)
        & np.isclose(overlay_df["radius"], radius)
    ].copy()
    payload = {
        "selector": selector,
        "rule": rule,
        "radius": radius,
        "qc_row": row,
        "source_unit_rows": int(len(source_units)),
        "pilot_unit_rows": int(len(pilot_units)),
        "overlay_unit_rows": int(len(overlay_df)),
        "selected_radius_overlay_rows": int(len(selected)),
        "split_gate": SPLIT_GATE,
        "ess_gate": ESS_GATE,
    }
    write_json(out_dir / "overlay_decision.json", payload)
    lines = [
        "# Overlay Selector QC",
        "",
        f"Selector: `{selector}`",
        f"Rule: `{rule}`",
        f"Radius: `{radius:.4f}`",
        "",
        "## Decision",
        "",
        f"- QC pass: `{row.get('qc_pass', 'n/a')}`",
        f"- Claim status: `{row.get('claim_status', 'n/a')}`",
        f"- Observed refs: `{row.get('observed_ref_count', 'n/a')}` / `{row.get('selected_ref_count', 'n/a')}`",
        f"- Missing refs: `{row.get('missing_ref_count', 'n/a')}`",
        f"- q05 ESS: `{row.get('q05_ess_fraction', 'n/a')}`",
        f"- max split logZ/P diff: `{row.get('max_split_logZ_per_P_diff', 'n/a')}`",
        f"- bootstrap sd phi: `{row.get('bootstrap_sd_phi', 'n/a')}`",
        "",
        "## Inputs",
        "",
        f"- Source unit rows: `{len(source_units)}`",
        f"- Pilot unit rows: `{len(pilot_units)}`",
        f"- Overlay unit rows: `{len(overlay_df)}`",
        "",
        "This overlay does not modify retained production outputs. Pilot unit summaries override source rows only inside this analysis directory.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay targeted pilot unit summaries onto source selector QC.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selector", default="dense_qc_stable_ref30")
    parser.add_argument("--rule", default="low_tv_spectral_teacher")
    parser.add_argument("--radius", type=float, default=0.45)
    parser.add_argument("--selector-size", type=int, default=30)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    source_units = load_source_units(args.analysis_dir / "unit_summary_long.csv")
    pilot_units = load_pilot_units(args.pilot_root)
    overlay_df = overlay_units(source_units, pilot_units)
    selectors = pd.read_csv(args.analysis_dir / "selector_membership.csv")
    selectors = selectors[
        (selectors["selector"] == args.selector)
        & (selectors["rule"] == args.rule)
    ].copy()
    if selectors.empty:
        raise ValueError(f"No selector rows found for {args.selector}/{args.rule}")
    qc_df, phi_df = selector_qc(overlay_df, selectors, selector_size=args.selector_size)

    write_csv(out_dir / "overlay_unit_summary_long.csv", overlay_df)
    write_csv(out_dir / "overlay_selector_qc_by_rule_radius.csv", qc_df)
    write_csv(out_dir / "overlay_selector_phi_by_rule_radius.csv", phi_df)
    write_report(out_dir, args.selector, args.rule, args.radius, qc_df, source_units, pilot_units, overlay_df)

    row = qc_df[
        (qc_df["selector"] == args.selector)
        & (qc_df["rule"] == args.rule)
        & np.isclose(qc_df["radius"], args.radius)
    ].iloc[0]
    print(
        json.dumps(
            {
                "selector": args.selector,
                "rule": args.rule,
                "radius": float(args.radius),
                "qc_pass": bool(row["qc_pass"]),
                "claim_status": str(row["claim_status"]),
                "observed_ref_count": int(row["observed_ref_count"]),
                "missing_ref_count": int(row["missing_ref_count"]),
                "q05_ess_fraction": float(row["q05_ess_fraction"]),
                "max_split_logZ_per_P_diff": float(row["max_split_logZ_per_P_diff"]),
                "bootstrap_sd_phi": float(row["bootstrap_sd_phi"]),
                "out_dir": str(out_dir),
            },
            indent=2,
            sort_keys=True,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
