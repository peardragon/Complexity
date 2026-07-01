from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = STAGE_ROOT / "raw_outputs"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "reference_quality"
FIGURE_PER_REF_PATH = FIGURE_INPUT_ROOT / "reference_quality_by_ref.csv"
FIGURE_SUMMARY_PATH = FIGURE_INPUT_ROOT / "reference_quality_by_rule.csv"


def _sem(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def _read_reference_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(RAW_ROOT.glob("rule_*/ref_*/reference_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ref_dir = metadata_path.parent
        rule_id = str(metadata.get("rule_id", ref_dir.parent.name))
        rule_name = str(metadata.get("rule_name", metadata.get("rule", rule_id)))
        rows.append(
            {
                "rule_id": rule_id,
                "rule_name": rule_name,
                "rule": str(metadata.get("rule", rule_name)),
                "ref": str(metadata.get("ref_id", ref_dir.name)),
                "source_ref_id": int(metadata.get("source_ref_id", int(ref_dir.name.removeprefix("ref_")) - 1)),
                "theta_present": bool(metadata.get("theta_payload_exists", (ref_dir / "theta.npy").exists())),
                "dataset_present": bool(metadata.get("dataset_payload_exists", False)),
                "train_error": float(metadata.get("train_error", float("nan"))),
                "test_error": float(metadata.get("test_error", float("nan"))),
                "theta_norm": float(metadata.get("theta_norm", float("nan"))),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no reference metadata found under: {RAW_ROOT / 'rule_*' / 'ref_*'}")
    return pd.DataFrame(rows)


def build_summarized_outputs() -> dict[str, object]:
    refs = _read_reference_metadata().copy()
    for column in ("source_ref_id", "train_error", "test_error", "theta_norm"):
        refs[column] = pd.to_numeric(refs[column], errors="coerce")
    refs["theta_present"] = refs["theta_present"].astype(bool)
    refs["dataset_present"] = refs["dataset_present"].astype(bool)
    refs["train_error_zero"] = refs["train_error"].eq(0.0)

    per_ref = refs[
        [
            "rule_id",
            "rule_name",
            "rule",
            "ref",
            "source_ref_id",
            "theta_present",
            "dataset_present",
            "train_error",
            "test_error",
            "theta_norm",
            "train_error_zero",
        ]
    ].sort_values(["rule_id", "source_ref_id"])

    grouped = per_ref.groupby(["rule_id", "rule_name", "rule"], sort=True, dropna=False)
    summary = grouped.agg(
        n_refs=("source_ref_id", "count"),
        theta_present_count=("theta_present", "sum"),
        dataset_present_count=("dataset_present", "sum"),
        train_error_zero_count=("train_error_zero", "sum"),
        train_error_mean=("train_error", "mean"),
        train_error_max=("train_error", "max"),
        test_error_mean=("test_error", "mean"),
        test_error_sd=("test_error", "std"),
        test_error_min=("test_error", "min"),
        test_error_max=("test_error", "max"),
        theta_norm_mean=("theta_norm", "mean"),
        theta_norm_sd=("theta_norm", "std"),
        theta_norm_min=("theta_norm", "min"),
        theta_norm_max=("theta_norm", "max"),
    ).reset_index()
    summary["test_error_sem"] = grouped["test_error"].apply(_sem).to_numpy(dtype=float)
    summary["theta_norm_sem"] = grouped["theta_norm"].apply(_sem).to_numpy(dtype=float)
    summary["all_payloads_present"] = (
        summary["theta_present_count"].eq(summary["n_refs"])
        & summary["dataset_present_count"].eq(summary["n_refs"])
    )
    summary["all_train_error_zero"] = summary["train_error_zero_count"].eq(summary["n_refs"])
    summary = summary.sort_values("rule_id").reset_index(drop=True)

    FIGURE_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_ref.to_csv(FIGURE_PER_REF_PATH, index=False)
    summary.to_csv(FIGURE_SUMMARY_PATH, index=False)

    status = {
        "status": "complete"
        if bool(summary["all_payloads_present"].all()) and bool(summary["all_train_error_zero"].all())
        else "partial",
        "reference_count": int(len(per_ref)),
        "rule_count": int(summary["rule_id"].nunique()),
        "theta_present_count": int(per_ref["theta_present"].sum()),
        "dataset_present_count": int(per_ref["dataset_present"].sum()),
        "train_error_zero_count": int(per_ref["train_error_zero"].sum()),
        "figure_inputs": [
            "summarized_outputs/figure_inputs/reference_quality/reference_quality_by_rule.csv",
            "summarized_outputs/figure_inputs/reference_quality/reference_quality_by_ref.csv",
        ],
    }
    return status


def main() -> None:
    status = build_summarized_outputs()
    print(f"summary_status={status['status']} refs={status['reference_count']}")


if __name__ == "__main__":
    main()
