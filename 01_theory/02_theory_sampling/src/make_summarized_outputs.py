from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
PHI_FIGURE_INPUT_ROOT = FIGURE_INPUT_ROOT / "phi_by_sampling"
PHI_ENERGETIC_FIGURE_INPUT_ROOT = FIGURE_INPUT_ROOT / "phi_energetic_by_sampling"
LOGZ_FIGURE_INPUT_ROOT = FIGURE_INPUT_ROOT / "logZ_split"
DEFAULT_METHOD = "exact_shell_l2_vmf_adaptive_ce_tempered_smc"


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda value: str(value).strip().lower() == "true").fillna(False)


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
            phi_radius_emp = ((n_int - 1.0) / n_int) * math.log(radius / r0)
            phi_energy_emp = (logz - base_logz) / n_int
            phi_emp = phi_radius_emp + phi_energy_emp
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
                    "phi_radius_emp": float(phi_radius_emp),
                    "phi_energy_emp": float(phi_energy_emp),
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


def build_logz_split_frame(
    units: pd.DataFrame,
    *,
    source_path: Path = SUMMARY_ROOT / "sample_unit_summary.csv",
) -> pd.DataFrame:
    """Build the transient logZ split view used for compact figure inputs."""
    units = units.copy()
    numeric_columns = [
        "N",
        "dataset_id",
        "ref_id",
        "radius",
        "logZ_shell_full",
        "logZ_CE",
        "logZ_shell_stripped",
        "split0_logZ_shell",
        "split1_logZ_shell",
        "ess_frac",
        "smc_min_cess_fraction",
        "n_particles",
    ]
    for column in numeric_columns:
        if column in units.columns:
            units[column] = pd.to_numeric(units[column], errors="coerce")

    n_values = units["N"].astype(int)
    split0 = units["split0_logZ_shell"].astype(float)
    split1 = units["split1_logZ_shell"].astype(float)
    signed_split = (split0 - split1) / n_values
    if "sample_payload_path" in units.columns:
        payload_paths = units["sample_payload_path"].fillna("").astype(str)
    elif "samples_path" in units.columns:
        payload_paths = units["samples_path"].fillna("").astype(str)
    else:
        payload_paths = pd.Series("", index=units.index)
    far_from_path = payload_paths.str.contains("far_split", regex=False)
    far_from_particles = units.get("n_particles", pd.Series(np.nan, index=units.index)).astype(float) >= 32768
    payload_split = np.where(far_from_path | far_from_particles, "far_split", "near_split")

    out = pd.DataFrame(
        {
            "stage": "01_theory",
            "block": "theory_sampling_shell_pool",
            "condition_name": "N",
            "condition_value": n_values,
            "dataset_id": units["dataset_id"].astype(int),
            "ref_id": units["ref_id"].astype(int),
            "radius": units["radius"].astype(float),
            "source_type": "sample_unit_summary_csv",
            "logZ_main": units["logZ_shell_full"].astype(float),
            "logZ_CE": units["logZ_CE"].astype(float),
            "logZ_stripped": units["logZ_shell_stripped"].astype(float),
            "logZ_full": units["logZ_shell_full"].astype(float),
            "reference_prior_log_weight": np.nan,
            "log_prefactor": np.nan,
            "dlogZ_dr": np.nan,
            "split0_logZ": split0,
            "split1_logZ": split1,
            "signed_split_logZ_per_scale": signed_split,
            "split_logZ_per_scale_diff": signed_split.abs(),
            "dlogZ_dr_split0": np.nan,
            "dlogZ_dr_split1": np.nan,
            "split_dlogZ_dr_per_scale_diff": np.nan,
            "scale_name": "N",
            "scale_value": n_values,
            "ess_fraction": units["ess_frac"].astype(float),
            "smc_min_cess_fraction": units["smc_min_cess_fraction"].astype(float),
            "smc_completed": units["smc_completed"],
            "sampler_method": units["sampler_method"],
            "source_path": str(source_path.resolve()),
            "n_particles": units["n_particles"].astype(int),
            "sample_payload_path": payload_paths,
            "payload_split": payload_split,
        }
    )
    far_start = out.loc[out["payload_split"].eq("far_split")].groupby("scale_value")["radius"].min()
    out["far_split_start_r"] = out["scale_value"].map(far_start)
    return out.sort_values(["scale_value", "dataset_id", "ref_id", "radius"]).reset_index(drop=True)


def clear_csvs(path: Path, pattern: str = "*.csv") -> None:
    if not path.exists():
        return
    for csv_path in path.glob(pattern):
        csv_path.unlink()


def write_phi_figure_inputs(phi: pd.DataFrame, output_root: Path = PHI_FIGURE_INPUT_ROOT) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    clear_csvs(output_root, "N_*.csv")
    outputs: list[Path] = []
    for n_value, group in phi.sort_values(["N", "r"]).groupby("N", sort=True):
        out = output_root / f"N_{int(n_value)}.csv"
        group.to_csv(out, index=False)
        outputs.append(out)
    return outputs


def write_phi_energetic_figure_inputs(
    phi: pd.DataFrame,
    output_root: Path = PHI_ENERGETIC_FIGURE_INPUT_ROOT,
) -> list[Path]:
    return write_phi_figure_inputs(phi, output_root)


def write_logz_figure_inputs(logz_split: pd.DataFrame, output_root: Path = LOGZ_FIGURE_INPUT_ROOT) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    clear_csvs(output_root, "N_*.csv")
    frame = pd.DataFrame(
        {
            "N": pd.to_numeric(logz_split["scale_value"], errors="coerce"),
            "r": pd.to_numeric(logz_split["radius"], errors="coerce"),
            "split_logZ_per_N_diff": pd.to_numeric(logz_split["split_logZ_per_scale_diff"], errors="coerce"),
        }
    )
    optional_columns = {
        "dataset_id": "dataset_id",
        "ref_id": "ref_id",
        "split0_logZ": "split0_logZ",
        "split1_logZ": "split1_logZ",
        "signed_split_logZ_per_scale": "signed_split_logZ_per_N_diff",
        "ess_fraction": "ess_fraction",
        "smc_min_cess_fraction": "smc_min_cess_fraction",
        "smc_completed": "smc_completed",
        "sampler_method": "sampler_method",
        "n_particles": "n_particles",
        "sample_payload_path": "sample_payload_path",
        "payload_split": "payload_split",
        "far_split_start_r": "far_split_start_r",
        "source_path": "source_path",
    }
    for source, target in optional_columns.items():
        if source in logz_split.columns:
            frame[target] = logz_split[source]
    frame = frame.dropna(subset=["N", "r", "split_logZ_per_N_diff"]).copy()
    frame["N"] = frame["N"].astype(int)
    outputs: list[Path] = []
    sort_columns = [column for column in ["N", "r", "dataset_id", "ref_id"] if column in frame.columns]
    for n_value, group in frame.sort_values(sort_columns).groupby("N", sort=True):
        out = output_root / f"N_{int(n_value)}.csv"
        group.to_csv(out, index=False)
        outputs.append(out)
    return outputs


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
        "missing_payload_path_count": int(len(missing_payload_paths)),
        "missing_payload_path_examples": missing_payload_paths,
        "full_default_smc_pass": bool(
            len(units) == int(base["input_unit_count"])
            and int((units["sampler_method"] != DEFAULT_METHOD).sum()) == 0
            and int((units["n_particles"] > particle_cap).sum()) == 0
            and int((qc["split_pass"] != True).sum()) == 0
            and int((qc["smc_pass"] != True).sum()) == 0
            and int(len(missing_payload_paths)) == 0
        ),
    }

def summary_validation_base(units: pd.DataFrame, *, particle_cap: int) -> dict[str, object]:
    return {
        "input_unit_count": int(len(units)),
        "processed_unit_count": int(len(units)),
        "nondefault_method_count_before_validation": int((units["sampler_method"] != DEFAULT_METHOD).sum()),
        "over_particle_cap_count_before_validation": int(
            (pd.to_numeric(units["n_particles"], errors="coerce") > particle_cap).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build theory sampling summarized outputs and figure-input CSVs.")
    parser.add_argument(
        "--sample-summary",
        dest="sample_summary",
        type=Path,
        default=SUMMARY_ROOT / "sample_unit_summary.csv",
    )
    parser.add_argument("--output-root", type=Path, default=SUMMARY_ROOT)
    parser.add_argument("--particle-cap", type=int, default=32768)
    parser.add_argument("--split-threshold", type=float, default=0.006)
    parser.add_argument("--smc-cess-threshold", type=float, default=0.8)
    args = parser.parse_args()

    units = read_csv(project_path(args.sample_summary))
    base_validation = summary_validation_base(units, particle_cap=int(args.particle_cap))
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
    sample_summary_path = output_root / "sample_unit_summary.csv"
    units.to_csv(sample_summary_path, index=False)
    logz_split = build_logz_split_frame(units, source_path=sample_summary_path)
    write_phi_figure_inputs(phi, output_root / "figure_inputs" / "phi_by_sampling")
    write_phi_energetic_figure_inputs(phi, output_root / "figure_inputs" / "phi_energetic_by_sampling")
    write_logz_figure_inputs(logz_split, output_root / "figure_inputs" / "logZ_split")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
