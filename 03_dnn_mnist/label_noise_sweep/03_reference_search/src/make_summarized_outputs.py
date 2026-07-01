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
FIGURE_SUMMARY_PATH = FIGURE_INPUT_ROOT / "reference_quality_by_eta.csv"


def _sem(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def _read_reference_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(RAW_ROOT.glob("noise_eta_*/ref_*/reference_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ref_dir = metadata_path.parent
        rows.append(
            {
                "noise_eta": str(metadata.get("noise_eta", ref_dir.parent.name)),
                "eta": float(metadata.get("eta", str(ref_dir.parent.name).removeprefix("noise_eta_").replace("p", "."))),
                "ref": str(metadata.get("ref", ref_dir.name)),
                "source_ref_id": int(metadata.get("source_ref_id", ref_dir.name.removeprefix("ref_"))),
                "theta_present": bool(metadata.get("theta_payload_exists", (ref_dir / "theta.npy").exists())),
                "dataset_present": bool(metadata.get("dataset_payload_exists", False)),
                "train_error": float(metadata.get("train_error", float("nan"))),
                "test_error": float(metadata.get("test_error", float("nan"))),
                "theta_norm": float(metadata.get("theta_norm", float("nan"))),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no reference metadata found under: {RAW_ROOT / 'noise_eta_*' / 'ref_*'}")
    return pd.DataFrame(rows)


def build_summarized_outputs() -> dict[str, object]:
    refs = _read_reference_metadata().copy()
    for column in ("eta", "source_ref_id", "train_error", "test_error", "theta_norm"):
        refs[column] = pd.to_numeric(refs[column], errors="coerce")
    refs["theta_present"] = refs["theta_present"].astype(bool)
    refs["dataset_present"] = refs["dataset_present"].astype(bool)
    refs["train_error_zero"] = refs["train_error"].eq(0.0)

    per_ref = refs[
        [
            "noise_eta",
            "eta",
            "ref",
            "source_ref_id",
            "theta_present",
            "dataset_present",
            "train_error",
            "test_error",
            "theta_norm",
        ]
    ].sort_values(["eta", "source_ref_id"])

    summary = (
        refs.groupby(["noise_eta", "eta"], as_index=False)
        .agg(
            reference_count=("ref", "count"),
            theta_present_count=("theta_present", "sum"),
            dataset_present_count=("dataset_present", "sum"),
            train_error_zero_count=("train_error_zero", "sum"),
            train_error_max=("train_error", "max"),
            test_error_mean=("test_error", "mean"),
            test_error_sd=("test_error", "std"),
            test_error_sem=("test_error", _sem),
            theta_norm_mean=("theta_norm", "mean"),
            theta_norm_sd=("theta_norm", "std"),
            theta_norm_sem=("theta_norm", _sem),
        )
        .sort_values("eta")
        .reset_index(drop=True)
    )

    FIGURE_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_ref.to_csv(FIGURE_PER_REF_PATH, index=False)
    summary.to_csv(FIGURE_SUMMARY_PATH, index=False)

    status = {
        "status": "complete"
        if len(refs) > 0
        and bool(refs["theta_present"].all())
        and bool(refs["dataset_present"].all())
        and bool(refs["train_error_zero"].all())
        else "partial",
        "reference_rows": int(len(refs)),
        "eta_count": int(refs["eta"].nunique()),
        "theta_present_count": int(refs["theta_present"].sum()),
        "dataset_present_count": int(refs["dataset_present"].sum()),
        "train_error_zero_count": int(refs["train_error_zero"].sum()),
        "figure_summary_csv": str(FIGURE_SUMMARY_PATH.relative_to(STAGE_ROOT)),
        "figure_per_ref_csv": str(FIGURE_PER_REF_PATH.relative_to(STAGE_ROOT)),
    }
    return status


def main() -> None:
    status = build_summarized_outputs()
    print(
        f"status={status['status']} refs={status['reference_rows']} "
        f"etas={status['eta_count']} root={SUMMARY_ROOT}"
    )


if __name__ == "__main__":
    main()
