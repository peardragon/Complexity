from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHELL_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "raw_outputs" / "shell_pool"
SUMMARY_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "summarized_outputs"
FIGURE_INPUT_CSV = SUMMARY_ROOT / "fig01_sampling_phi_by_distance.csv"
DEFAULT_METHOD = "exact_shell_l2_vmf_adaptive_ce_smc"


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def row_key(row: pd.Series) -> tuple[int, int, int, str]:
    return (
        int(row["N"]),
        int(row["dataset_id"]),
        int(row["ref_id"]),
        f"{float(row['radius']):.12g}",
    )


def frame_keys(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(row_key, axis=1)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_replacements(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [read_csv(path) for path in paths]
    if not frames:
        raise ValueError("at least one replacement CSV is required")
    replacements = pd.concat(frames, ignore_index=True, sort=False)
    keys = frame_keys(replacements)
    if keys.duplicated().any():
        replacements = replacements.assign(_replacement_key=keys)
        replacements = replacements.drop_duplicates("_replacement_key", keep="last").drop(columns=["_replacement_key"])
    return replacements


def canonical_sample_path(row: pd.Series, radius_index: dict[float, int]) -> str:
    split = "far_split" if int(row["n_particles"]) == 32768 else "near_split"
    return (
        "01_theory/02_theory_sampling/raw_outputs/shell_pool/"
        f"{split}/N_{int(row['N'])}/"
        f"dataset_{int(row['dataset_id']) + 1:03d}/"
        f"ref_{int(row['ref_id']) + 1:03d}/"
        f"r_{radius_index[float(row['radius'])]:03d}/samples.npy"
    )


def assign_canonical_sample_paths(units: pd.DataFrame) -> pd.DataFrame:
    units = units.copy()
    radius_index = {float(radius): idx + 1 for idx, radius in enumerate(sorted(units["radius"].unique()))}
    paths = units.apply(lambda row: canonical_sample_path(row, radius_index), axis=1)
    units["sample_payload_path"] = paths
    units["samples_path"] = paths
    if "replacement_source_path" in units.columns:
        units["replacement_source_path"] = ""
    return units


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda value: str(value).strip().lower() == "true").fillna(False)


def preserve_legacy_fallback_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    legacy_columns = [
        "direct_qc_pass",
        "fallback_used",
        "direct_logZ_CE",
        "direct_ess_frac",
        "direct_split_logZ_per_N_diff",
    ]
    for column in legacy_columns:
        if column in frame.columns:
            frame[f"legacy_{column}"] = frame[column]
    if "fallback_used" in frame.columns:
        frame["fallback_used"] = False
    for column in [
        "direct_qc_pass",
        "direct_logZ_CE",
        "direct_ess_frac",
        "direct_split_logZ_per_N_diff",
    ]:
        if column in frame.columns:
            frame[column] = np.nan
    return frame


def merge_replacements(
    *,
    original: pd.DataFrame,
    replacements: pd.DataFrame,
    particle_cap: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    target_mask = (original["sampler_method"] != DEFAULT_METHOD) | (original["n_particles"] > particle_cap)
    target = original.loc[target_mask].copy()
    target_keys = set(frame_keys(target))
    replacement_keys = set(frame_keys(replacements))
    missing = sorted(target_keys - replacement_keys)
    extra = sorted(replacement_keys - target_keys)
    if missing or extra:
        raise ValueError(
            "replacement key mismatch: "
            f"missing={missing[:10]} extra={extra[:10]} "
            f"missing_count={len(missing)} extra_count={len(extra)}"
        )

    original_columns = list(original.columns)
    extra_columns = [column for column in replacements.columns if column not in original_columns]
    columns = original_columns + extra_columns
    merged = original.reindex(columns=columns).astype(object).copy()
    replacements = replacements.reindex(columns=columns).astype(object).copy()

    replacement_by_key = {row_key(row): row for _, row in replacements.iterrows()}
    for index, row in original.loc[target_mask].iterrows():
        merged.loc[index, columns] = replacement_by_key[row_key(row)].values

    merged = preserve_legacy_fallback_columns(merged)

    validation = {
        "original_unit_count": int(len(original)),
        "replacement_unit_count": int(len(replacements)),
        "target_unit_count": int(len(target)),
        "target_nondefault_method_count": int((target["sampler_method"] != DEFAULT_METHOD).sum()),
        "target_over_particle_cap_count": int((target["n_particles"] > particle_cap).sum()),
    }
    return merged, validation


def build_phi_summary(units: pd.DataFrame, *, split_threshold: float, smc_cess_threshold: float) -> pd.DataFrame:
    units = units.copy()
    numeric_columns = [
        "N",
        "radius",
        "logZ_shell_full",
        "weighted_ce",
        "weighted_error",
        "ess_frac",
        "split_logZ_per_N_diff",
        "smc_min_cess_fraction",
        "smc_step_count",
    ]
    for column in numeric_columns:
        units[column] = pd.to_numeric(units[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for n_value, by_n in units.groupby("N", sort=True):
        n_int = int(n_value)
        r0 = float(by_n["radius"].min())
        base_logz = float(by_n.loc[np.isclose(by_n["radius"], r0), "logZ_shell_full"].mean())
        for radius, group in by_n.groupby("radius", sort=True):
            radius = float(radius)
            logz = float(group["logZ_shell_full"].mean())
            phi_emp = ((n_int - 1.0) / n_int) * math.log(radius / r0) + (logz - base_logz) / n_int
            fallback = bool_series(group["fallback_used"]) if "fallback_used" in group.columns else pd.Series(False, index=group.index)
            min_cess = float(group["smc_min_cess_fraction"].min(skipna=True))
            if math.isnan(min_cess):
                min_cess = float("nan")
            max_steps = float(group["smc_step_count"].max(skipna=True))
            if math.isnan(max_steps):
                max_steps = float("nan")
            rows.append(
                {
                    "r": radius,
                    "N": n_int,
                    "phi_emp": float(phi_emp),
                    "weighted_CE": float(group["weighted_ce"].mean()),
                    "weighted_err": float(group["weighted_error"].mean()),
                    "reference_count": int(len(group)),
                    "mean_ess_frac": float(group["ess_frac"].mean()),
                    "q05_ess_frac": float(group["ess_frac"].quantile(0.05)),
                    "max_split_logZ_per_N_diff": float(group["split_logZ_per_N_diff"].max()),
                    "fallback_unit_count": int(fallback.sum()),
                    "fallback_unit_fraction": float(fallback.mean()),
                    "min_smc_cess_fraction": min_cess,
                    "max_smc_step_count": max_steps,
                }
            )
    return pd.DataFrame(rows)


def build_qc_summary(phi: pd.DataFrame, *, split_threshold: float, smc_cess_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in phi.sort_values(["N", "r"]).iterrows():
        min_cess = float(row["min_smc_cess_fraction"])
        split_pass = float(row["max_split_logZ_per_N_diff"]) <= split_threshold
        smc_pass = (not math.isfinite(min_cess)) or min_cess >= smc_cess_threshold
        rows.append(
            {
                "N": int(row["N"]),
                "r": float(row["r"]),
                "reference_count": int(row["reference_count"]),
                "split_pass": bool(split_pass),
                "smc_pass": bool(smc_pass),
                "claim": "pass" if split_pass and smc_pass else "fail",
                "max_split_logZ_per_N_diff": float(row["max_split_logZ_per_N_diff"]),
                "min_smc_cess_fraction": min_cess,
                "max_smc_step_count": row["max_smc_step_count"],
            }
        )
    return pd.DataFrame(rows)


def validate_payload_paths(units: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    for value in units["sample_payload_path"].dropna().astype(str):
        path = PROJECT_ROOT / value
        if not path.exists():
            missing.append(value)
            if len(missing) >= 20:
                break
    return missing


def validation_summary(
    units: pd.DataFrame,
    phi: pd.DataFrame,
    qc: pd.DataFrame,
    *,
    base: dict[str, object],
    particle_cap: int,
    split_threshold: float,
    smc_cess_threshold: float,
) -> dict[str, object]:
    method_counts = {str(key): int(value) for key, value in units["sampler_method"].value_counts(dropna=False).items()}
    particle_counts = {str(int(key)): int(value) for key, value in units["n_particles"].value_counts(dropna=False).sort_index().items()}
    replacement_reason_counts = (
        {str(key): int(value) for key, value in units["replacement_reason"].value_counts(dropna=True).items()}
        if "replacement_reason" in units.columns
        else {}
    )
    missing_payload_paths = validate_payload_paths(units)
    return {
        **base,
        "final_unit_count": int(len(units)),
        "method_counts": method_counts,
        "particle_counts": particle_counts,
        "max_n_particles": int(units["n_particles"].max()),
        "nondefault_method_count": int((units["sampler_method"] != DEFAULT_METHOD).sum()),
        "over_particle_cap_count": int((units["n_particles"] > particle_cap).sum()),
        "fallback_unit_count": int(bool_series(units["fallback_used"]).sum()) if "fallback_used" in units.columns else 0,
        "max_split_logZ_per_N_diff": float(units["split_logZ_per_N_diff"].max()),
        "split_threshold_per_N": float(split_threshold),
        "split_fail_cell_count": int((qc["split_pass"] != True).sum()),
        "smc_cess_threshold": float(smc_cess_threshold),
        "min_smc_cess_fraction": float(units["smc_min_cess_fraction"].min(skipna=True)),
        "smc_fail_cell_count": int((qc["smc_pass"] != True).sum()),
        "qc_cell_count": int(len(qc)),
        "reference_count_min": int(phi["reference_count"].min()),
        "reference_count_max": int(phi["reference_count"].max()),
        "replacement_reason_counts": replacement_reason_counts,
        "missing_payload_path_count": int(len(missing_payload_paths)),
        "missing_payload_path_examples": missing_payload_paths,
        "full_default_smc_pass": bool(
            len(units) == int(base["original_unit_count"])
            and int((units["sampler_method"] != DEFAULT_METHOD).sum()) == 0
            and int((units["n_particles"] > particle_cap).sum()) == 0
            and int((qc["split_pass"] != True).sum()) == 0
            and int((qc["smc_pass"] != True).sum()) == 0
            and int(len(missing_payload_paths)) == 0
        ),
    }


def write_report(path: Path, validation: dict[str, object]) -> None:
    lines = [
        "# Two-pool sampling shell_pool run report",
        "",
        "## Config",
        "",
        "- Final result was rebuilt as default adaptive CE-tempered SMC only.",
        "- Replaced all legacy direct-vMF IS units and all p262144 bad-split units.",
        f"- Particle cap: `{validation['max_n_particles']}` observed, target cap `32768`.",
        f"- Split logZ threshold per N: `{validation['split_threshold_per_N']}`.",
        "- Legacy direct/fallback decision columns are preserved as `legacy_*`; final policy columns set no direct fallback.",
        "",
        "## Output files",
        "",
        "- `near_split/N_*/dataset_*/ref_*/r_*/samples.npy`: canonical 2048-particle near-radius payloads.",
        "- `far_split/N_*/dataset_*/ref_*/r_*/samples.npy`: canonical 32768-particle far-radius payloads.",
        "- `sample_unit_summary.csv`: unit-level final summary.",
        "- `sampling_phi_by_N_alpha0p1.csv`: sampling empirical phi table for figures.",
        "- `sampling_logz_stability_by_N_radius.csv`: radius/N split-logZ and SMC stability summary.",
        "- `default_smc_final_validation.json`: merge and QC validation summary.",
        "",
        "## Validation",
        "",
        f"- Units: `{validation['final_unit_count']}`.",
        f"- Method counts: `{validation['method_counts']}`.",
        f"- Particle counts: `{validation['particle_counts']}`.",
        f"- Replacement reason counts: `{validation['replacement_reason_counts']}`.",
        f"- Non-default method count: `{validation['nondefault_method_count']}`.",
        f"- Over-cap unit count: `{validation['over_particle_cap_count']}`.",
        f"- Final fallback unit count: `{validation['fallback_unit_count']}`.",
        f"- Max split logZ / N: `{validation['max_split_logZ_per_N_diff']}`.",
        f"- Split-fail QC cells: `{validation['split_fail_cell_count']}`.",
        f"- Min SMC CESS fraction: `{validation['min_smc_cess_fraction']}`.",
        f"- SMC-fail QC cells: `{validation['smc_fail_cell_count']}`.",
        f"- Payload paths missing: `{validation['missing_payload_path_count']}`.",
        f"- Full default-SMC pass: `{validation['full_default_smc_pass']}`.",
        "",
        "## Reproduction chain",
        "",
        "Canonical near/far raw sample files are aggregated into `sample_unit_summary.csv`; "
        "the compact phi and QC tables are regenerated from this summary. "
        "Sampling-only and theory-comparison figures read the compact CSVs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final theory shell-pool tables after default-SMC replacements.")
    parser.add_argument("--original-summary", type=Path, default=SUMMARY_ROOT / "shell_pool" / "sample_unit_summary.csv")
    parser.add_argument("--replacement-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=SUMMARY_ROOT / "shell_pool")
    parser.add_argument("--particle-cap", type=int, default=32768)
    parser.add_argument("--split-threshold", type=float, default=0.006)
    parser.add_argument("--smc-cess-threshold", type=float, default=0.8)
    args = parser.parse_args()

    original = read_csv(project_path(args.original_summary))
    replacements = load_replacements(project_path(path) for path in args.replacement_csv)
    units, base_validation = merge_replacements(
        original=original,
        replacements=replacements,
        particle_cap=int(args.particle_cap),
    )
    units = assign_canonical_sample_paths(units)
    phi = build_phi_summary(
        units,
        split_threshold=float(args.split_threshold),
        smc_cess_threshold=float(args.smc_cess_threshold),
    )
    qc = build_qc_summary(
        phi,
        split_threshold=float(args.split_threshold),
        smc_cess_threshold=float(args.smc_cess_threshold),
    )
    validation = validation_summary(
        units,
        phi,
        qc,
        base=base_validation,
        particle_cap=int(args.particle_cap),
        split_threshold=float(args.split_threshold),
        smc_cess_threshold=float(args.smc_cess_threshold),
    )

    output_root = project_path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    units.to_csv(output_root / "sample_unit_summary.csv", index=False)
    phi.to_csv(output_root / "sampling_phi_by_N_alpha0p1.csv", index=False)
    FIGURE_INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    phi.to_csv(FIGURE_INPUT_CSV, index=False)
    qc.to_csv(output_root / "sampling_logz_stability_by_N_radius.csv", index=False)
    (output_root / "default_smc_final_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(output_root / "run_report.md", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
