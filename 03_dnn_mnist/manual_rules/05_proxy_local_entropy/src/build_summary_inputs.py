from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
MANUAL_ROOT = REPO_ROOT / "03_dnn_mnist" / "manual_rules"
STAGE_ROOT = MANUAL_ROOT / "05_proxy_local_entropy"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
QC_ROOT = SUMMARY_ROOT / "qc"
CANONICAL_ROOT = SUMMARY_ROOT / "canonical_refpool1024_dense30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
LEGACY_RUN_ROOT = (
    SUMMARY_ROOT / "active_rules_refpool1024_dense30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
DIRECT_RUN_TAG = "direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
DIRECT_UNIT_TABLE = (
    MANUAL_ROOT
    / "04_sampling"
    / "summarized_outputs"
    / DIRECT_RUN_TAG
    / "01_active_rules_sampling"
    / "05_pool2_pm_sais_sampling"
    / "shell_summary_by_unit_with_phi_derivatives.csv"
)
CANONICAL_UNIT_TABLE = CANONICAL_ROOT / "shell_summary_by_unit_with_phi_derivatives.csv"
LEGACY_UNIT_TABLE = LEGACY_RUN_ROOT / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
UNIT_INDEX = MANUAL_ROOT / "04_sampling" / "raw_outputs" / "shell_pool" / "unit_index.csv"
RULE_MAPPING = MANUAL_ROOT / "config" / "rule_mapping.csv"
COMPLEXITY_TABLE = (
    LEGACY_RUN_ROOT / "06_results_figures" / "tables" / "nmstv_values_for_raw_phi_plot.csv"
)

SPLIT_LOGZ_THRESHOLD = 0.004
TARGET_REF_COUNT = 30
RULE_ORDER = {
    "rule_001": 1,
    "rule_002": 2,
    "rule_003": 3,
    "rule_004": 4,
}
FALLBACK_NMSTV = {
    "very_low_tv_spectral_teacher": 0.3245703473792008,
    "real_even_odd": 0.4932864276461805,
    "teacher_nn": 0.6843772639598127,
    "random_label": 0.985558573825462,
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def _sd(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1))


def _q(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < 3 or window <= 1:
        return values.astype(float, copy=True)
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window <= 1:
        return values.astype(float, copy=True)
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.pad(values.astype(float), (pad, pad), mode="edge"), kernel, mode="valid")


def _load_rules() -> pd.DataFrame:
    rules = pd.read_csv(RULE_MAPPING)
    rules["rule_order"] = rules["rule_id"].map(RULE_ORDER)
    return rules


def _load_complexity(rules: pd.DataFrame) -> pd.DataFrame:
    if COMPLEXITY_TABLE.exists():
        complexity = pd.read_csv(COMPLEXITY_TABLE)
        complexity = complexity.rename(columns={"rule": "rule_name", "label": "complexity_label"})
        keep = [c for c in ["rule_name", "nmstv_mean", "tv_mean", "n_datasets"] if c in complexity.columns]
        complexity = complexity[keep].copy()
    else:
        complexity = pd.DataFrame(
            [{"rule_name": key, "nmstv_mean": value} for key, value in FALLBACK_NMSTV.items()]
        )
    for col in ["tv_mean", "n_datasets"]:
        if col not in complexity.columns:
            complexity[col] = np.nan
    out = rules.merge(complexity, on="rule_name", how="left")
    out["nmstv_mean"] = out.apply(
        lambda row: FALLBACK_NMSTV.get(str(row["rule_name"]), row["nmstv_mean"]),
        axis=1,
    )
    return out


def _unit_source() -> Path:
    if DIRECT_UNIT_TABLE.exists():
        return DIRECT_UNIT_TABLE
    if CANONICAL_UNIT_TABLE.exists():
        return CANONICAL_UNIT_TABLE
    if LEGACY_UNIT_TABLE.exists():
        return LEGACY_UNIT_TABLE
    raise FileNotFoundError(f"missing unit summary table: {CANONICAL_UNIT_TABLE} or {LEGACY_UNIT_TABLE}")


def _load_units() -> pd.DataFrame:
    units = pd.read_csv(_unit_source())
    rules = _load_rules()
    complexity = _load_complexity(rules)

    units["radius_key"] = pd.to_numeric(units["radius"], errors="coerce").round(10)
    units["ref_id"] = pd.to_numeric(units["ref_id"], errors="coerce").astype("Int64")

    if "rule_id" in units.columns and "ref_path_id" in units.columns:
        merged = units.copy()
    elif not UNIT_INDEX.exists():
        merged = units.rename(columns={"rule": "rule_name"}).copy()
        merged = merged.merge(
            complexity[
                [
                    "rule_id",
                    "rule_name",
                    "label",
                    "rule_order",
                    "nmstv_mean",
                    "tv_mean",
                    "n_datasets",
                ]
            ],
            on="rule_name",
            how="left",
            validate="many_to_one",
        )
        if merged["rule_id"].isna().any():
            missing = merged[merged["rule_id"].isna()][["rule_name", "ref_id", "radius"]].head(20)
            raise ValueError(f"rule mapping failed for rows:\n{missing}")
        merged["rule_label"] = merged["label"]
        merged["ref_path_id"] = merged["ref_id"].map(lambda value: f"ref_{int(value):03d}")
        merged["radius_path_id"] = merged["radius"].map(lambda value: f"r_{float(value):0.4f}".replace(".", "p"))
    else:
        index = pd.read_csv(UNIT_INDEX)
        rename = {
            "rule": "rule_name",
            "label": "rule_label",
            "ref_id": "ref_path_id",
            "original_ref_id": "ref_id",
        }
        index = index.rename(columns=rename)
        index["radius_key"] = pd.to_numeric(index["radius"], errors="coerce").round(10)
        merged = units.merge(
            index[
                [
                    "rule_id",
                    "rule_name",
                    "rule_label",
                    "ref_path_id",
                    "ref_id",
                    "radius_key",
                    "radius_path_id",
                    "samples_path",
                    "unit_summary_path",
                ]
            ],
            left_on=["rule", "ref_id", "radius_key"],
            right_on=["rule_name", "ref_id", "radius_key"],
            how="left",
            suffixes=("", "_final"),
            validate="one_to_one",
        )
        if merged["rule_id"].isna().any():
            missing = merged[merged["rule_id"].isna()][["rule", "ref_id", "radius"]].head(20)
            raise ValueError(f"unit index merge failed for rows:\n{missing}")
        for col in ["samples_path", "unit_summary_path"]:
            if f"{col}_final" in merged.columns:
                merged[col] = merged[f"{col}_final"]
                merged = merged.drop(columns=[f"{col}_final"])

    if "rule_order" not in merged.columns or merged["rule_order"].isna().any():
        merged = merged.merge(
            complexity[
                [
                    "rule_id",
                    "rule_name",
                    "label",
                    "rule_order",
                    "nmstv_mean",
                    "tv_mean",
                    "n_datasets",
                ]
            ],
            on=["rule_id", "rule_name"],
            how="left",
            suffixes=("", "_rulemap"),
        )
    if "rule_label" not in merged.columns:
        merged["rule_label"] = merged.get("label")
    merged["rule_label"] = merged["rule_label"].fillna(merged.get("label"))
    merged = merged.drop(columns=[col for col in ["label"] if col in merged.columns])

    merged["radius"] = pd.to_numeric(merged["radius"], errors="coerce")
    merged["ref_id"] = pd.to_numeric(merged["ref_id"], errors="coerce").astype(int)
    merged["theta_path"] = merged.apply(
        lambda row: (
            "Complexity/03_dnn_mnist/manual_rules/03_reference_search/raw_outputs/"
            f"{row['rule_id']}/{row['ref_path_id']}/theta.npy"
        ),
        axis=1,
    )
    merged["dataset_path"] = merged.apply(
        lambda row: (
            "Complexity/03_dnn_mnist/manual_rules/01_dataset/raw_outputs/"
            f"{row['rule_id']}/dataset.npz"
        ),
        axis=1,
    )
    merged["rule"] = merged["rule_name"]
    numeric_cols = [
        "ess",
        "ess_fraction",
        "smc_min_cess_fraction",
        "split_logZ_per_P_diff",
        "logZ_inf_full",
        "phi_energy_raw",
        "delta_phi_energy_unit",
        "delta_phi_full_unit",
        "d_phi_energy_raw_dd",
        "d_delta_phi_energy_unit_dd",
        "d_delta_phi_full_unit_dd",
        "d_phi_energy_direct_dd_unit",
        "d_delta_phi_energy_direct_dd_unit",
        "dlogZ_inf_full_dr",
        "split_dlogZ_dr_per_P_diff",
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged.sort_values(["rule_order", "ref_path_id", "radius"]).reset_index(drop=True)


def _aggregate_rule_radius(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["rule_id", "rule_name", "rule_label", "rule_order", "nmstv_mean", "radius"]
    for key, sub in units.groupby(group_cols, dropna=False, sort=True):
        rule_id, rule_name, rule_label, rule_order, nmstv_mean, radius = key
        row: dict[str, Any] = {
            "rule_id": rule_id,
            "rule_name": rule_name,
            "rule_label": rule_label,
            "rule_order": int(rule_order),
            "nmstv_mean": float(nmstv_mean),
            "radius": float(radius),
            "radius_path_id": str(sub["radius_path_id"].iloc[0]),
            "n_units": int(len(sub)),
            "ref_count": int(sub["ref_path_id"].nunique()),
            "target_ref_count": TARGET_REF_COUNT,
        }
        for col in [
            "phi_energy_raw",
            "delta_phi_energy_unit",
            "delta_phi_full_unit",
            "d_phi_energy_raw_dd",
            "d_delta_phi_energy_unit_dd",
            "d_delta_phi_full_unit_dd",
            "d_phi_energy_direct_dd_unit",
            "d_delta_phi_energy_direct_dd_unit",
            "logZ_inf_full",
            "dlogZ_inf_full_dr",
            "split_logZ_per_P_diff",
            "split_dlogZ_dr_per_P_diff",
            "ess_fraction",
            "smc_min_cess_fraction",
        ]:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_sd"] = _sd(sub[col])
                row[f"{col}_sem"] = _sem(sub[col])
        row["q95_split_logZ_per_P_diff"] = _q(sub["split_logZ_per_P_diff"], 0.95)
        row["max_split_logZ_per_P_diff"] = float(sub["split_logZ_per_P_diff"].max())
        if "split_dlogZ_dr_per_P_diff" in sub.columns:
            row["q95_split_dlogZ_dr_per_P_diff"] = _q(sub["split_dlogZ_dr_per_P_diff"], 0.95)
            row["max_split_dlogZ_dr_per_P_diff"] = float(sub["split_dlogZ_dr_per_P_diff"].max())
        row["q05_ess_fraction"] = _q(sub["ess_fraction"], 0.05)
        row["q05_smc_min_cess_fraction"] = _q(sub["smc_min_cess_fraction"], 0.05)
        completed = sub["smc_completed"].astype(bool) if "smc_completed" in sub.columns else pd.Series([False])
        finite = sub["finite"].astype(bool) if "finite" in sub.columns else pd.Series([False])
        row["smc_completed_count"] = int(completed.sum())
        row["finite_unit_count"] = int(finite.sum())
        row["sampling_status"] = "complete" if row["n_units"] == TARGET_REF_COUNT else "inspect"
        row["qc_claim"] = (
            "pass"
            if row["n_units"] == TARGET_REF_COUNT
            and row["smc_completed_count"] == row["n_units"]
            and row["max_split_logZ_per_P_diff"] <= SPLIT_LOGZ_THRESHOLD
            else "inspect"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rule_order", "radius"]).reset_index(drop=True)


def _build_phase(units: pd.DataFrame, rule_radius: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for (rule_id, ref_path_id), sub in units.groupby(["rule_id", "ref_path_id"], sort=True):
        sub = sub.sort_values("radius")
        radius = sub["radius"].to_numpy(dtype=float)
        if "d_phi_energy_direct_dd_unit" in sub.columns:
            dphi = sub["d_phi_energy_direct_dd_unit"].to_numpy(dtype=float)
            derivative_source = "sampling_time_direct_radial_score_derivative"
        else:
            phi = sub["phi_energy_raw"].to_numpy(dtype=float)
            dphi = np.gradient(phi, radius)
            derivative_source = "unit_curve_finite_difference_fallback_missing_direct_radial_derivative"
        if len(radius) < 3 or not np.isfinite(dphi).all():
            continue
        curvature = np.gradient(dphi, radius)
        a_kappa = _trapz(np.maximum(curvature, 0.0), radius)
        row0 = sub.iloc[0]
        ref_rows.append(
            {
                "rule_id": rule_id,
                "rule_name": row0["rule_name"],
                "rule_label": row0["rule_label"],
                "rule_order": int(row0["rule_order"]),
                "nmstv_mean": float(row0["nmstv_mean"]),
                "ref_path_id": ref_path_id,
                "ref_id": int(row0["ref_id"]),
                "A_kappa": a_kappa,
                "derivative_source": derivative_source,
            }
        )
        for r, dp, curv in zip(radius, dphi, curvature):
            curve_rows.append(
                {
                    "rule_id": rule_id,
                    "rule_name": row0["rule_name"],
                    "rule_label": row0["rule_label"],
                    "rule_order": int(row0["rule_order"]),
                    "nmstv_mean": float(row0["nmstv_mean"]),
                    "ref_path_id": ref_path_id,
                    "radius": float(r),
                    "dphi_dr": float(dp),
                    "dphi_dr_smooth": float(dp),
                    "positive_curvature": float(max(curv, 0.0)),
                    "derivative_source": derivative_source,
                }
            )
    by_ref = pd.DataFrame(ref_rows).sort_values(["rule_order", "ref_path_id"]).reset_index(drop=True)
    ref_curves = pd.DataFrame(curve_rows)
    curve_summary = (
        ref_curves.groupby(
            ["rule_id", "rule_name", "rule_label", "rule_order", "nmstv_mean", "radius"],
            dropna=False,
            sort=True,
        )
        .agg(
            n_refs=("ref_path_id", "nunique"),
            dphi_dr_mean=("dphi_dr", "mean"),
            dphi_dr_sem=("dphi_dr", _sem),
            dphi_dr_smooth_mean=("dphi_dr_smooth", "mean"),
            dphi_dr_smooth_sem=("dphi_dr_smooth", _sem),
            positive_curvature_mean=("positive_curvature", "mean"),
            derivative_source=("derivative_source", "first"),
        )
        .reset_index()
        .sort_values(["rule_order", "radius"])
    )
    case_summary = (
        by_ref.groupby(["rule_id", "rule_name", "rule_label", "rule_order", "nmstv_mean"], sort=True)
        .agg(
            n_refs=("ref_path_id", "nunique"),
            A_kappa_mean=("A_kappa", "mean"),
            A_kappa_sd=("A_kappa", _sd),
            A_kappa_sem=("A_kappa", _sem),
            derivative_source=("derivative_source", "first"),
        )
        .reset_index()
        .sort_values("rule_order")
    )
    return by_ref, curve_summary, case_summary


def _write_figure_inputs(rule_radius: pd.DataFrame, curve_summary: pd.DataFrame, case_summary: pd.DataFrame) -> None:
    phi = rule_radius[
        [
            "rule_id",
            "rule_name",
            "rule_label",
            "rule_order",
            "nmstv_mean",
            "radius",
            "n_units",
            "delta_phi_energy_unit_mean",
            "delta_phi_energy_unit_sem",
            "phi_energy_raw_mean",
            "phi_energy_raw_sem",
        ]
    ].copy()
    direct_present = "d_phi_energy_direct_dd_unit_mean" in rule_radius.columns
    if direct_present:
        dphi = rule_radius[
            [
                "rule_id",
                "rule_name",
                "rule_label",
                "rule_order",
                "nmstv_mean",
                "radius",
                "n_units",
                "d_phi_energy_direct_dd_unit_mean",
                "d_phi_energy_direct_dd_unit_sem",
                "d_delta_phi_energy_direct_dd_unit_mean",
                "d_delta_phi_energy_direct_dd_unit_sem",
            ]
        ].copy()
        dphi["derivative_source"] = "sampling_time_direct_radial_score_derivative"
    else:
        dphi = rule_radius[
            [
                "rule_id",
                "rule_name",
                "rule_label",
                "rule_order",
                "nmstv_mean",
                "radius",
                "n_units",
                "d_delta_phi_energy_unit_dd_mean",
                "d_delta_phi_energy_unit_dd_sem",
                "d_phi_energy_raw_dd_mean",
                "d_phi_energy_raw_dd_sem",
            ]
        ].copy()
        dphi = dphi.rename(
            columns={
                "d_phi_energy_raw_dd_mean": "d_phi_energy_direct_dd_unit_mean",
                "d_phi_energy_raw_dd_sem": "d_phi_energy_direct_dd_unit_sem",
                "d_delta_phi_energy_unit_dd_mean": "d_delta_phi_energy_direct_dd_unit_mean",
                "d_delta_phi_energy_unit_dd_sem": "d_delta_phi_energy_direct_dd_unit_sem",
            }
        )
        dphi["derivative_source"] = "unit_curve_finite_difference_fallback_missing_direct_radial_derivative"
    dphi = dphi[
        [
            "rule_id",
            "rule_name",
            "rule_label",
            "rule_order",
            "nmstv_mean",
            "radius",
            "n_units",
            "d_delta_phi_energy_direct_dd_unit_mean",
            "d_delta_phi_energy_direct_dd_unit_sem",
            "d_phi_energy_direct_dd_unit_mean",
            "d_phi_energy_direct_dd_unit_sem",
            "derivative_source",
        ]
    ].copy()
    curve_summary = curve_summary.drop(
        columns=["dphi_dr_smooth_mean", "dphi_dr_smooth_sem", "derivative_source"],
        errors="ignore",
    ).merge(
        dphi[
            [
                "rule_id",
                "radius",
                "d_phi_energy_direct_dd_unit_mean",
                "d_phi_energy_direct_dd_unit_sem",
                "derivative_source",
            ]
        ].rename(
            columns={
                "d_phi_energy_direct_dd_unit_mean": "dphi_dr_smooth_mean",
                "d_phi_energy_direct_dd_unit_sem": "dphi_dr_smooth_sem",
            }
        ),
        on=["rule_id", "radius"],
        how="left",
    )
    for name in ["phi_d_curve", "phi_energetic_d_curve"]:
        out = _ensure_dir(FIGURE_INPUT_ROOT / name)
        phi.to_csv(out / f"{name}.csv", index=False)
    for name in ["derivative_phi_d_curve", "derivative_phi_energetic_d_curve"]:
        out = _ensure_dir(FIGURE_INPUT_ROOT / name)
        dphi.to_csv(out / f"{name}.csv", index=False)
    for name in ["phase_like_A_by_rule", "phase_like_A_by_complexity"]:
        out = _ensure_dir(FIGURE_INPUT_ROOT / name)
        case_summary.to_csv(out / f"{name}.csv", index=False)
        curve_summary.to_csv(out / "phase_derivative_curves.csv", index=False)


def _write_qc(units: pd.DataFrame, rule_radius: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logz_cols = [
        "rule_id",
        "rule_name",
        "rule_label",
        "rule_order",
        "nmstv_mean",
        "radius",
        "radius_path_id",
        "ref_count",
        "smc_completed_count",
        "q95_split_logZ_per_P_diff",
        "max_split_logZ_per_P_diff",
        "q95_split_dlogZ_dr_per_P_diff",
        "max_split_dlogZ_dr_per_P_diff",
        "q05_ess_fraction",
        "q05_smc_min_cess_fraction",
        "split_logZ_per_P_diff_mean",
        "threshold_max_split_logZ_per_P_diff",
        "qc_claim",
    ]
    logz = rule_radius.copy()
    logz["threshold_max_split_logZ_per_P_diff"] = SPLIT_LOGZ_THRESHOLD
    logz = logz[logz_cols]
    ref = rule_radius[
        [
            "rule_id",
            "rule_name",
            "rule_label",
            "rule_order",
            "nmstv_mean",
            "radius",
            "radius_path_id",
            "ref_count",
            "phi_energy_raw_mean",
            "phi_energy_raw_sd",
            "phi_energy_raw_sem",
            "delta_phi_energy_unit_mean",
            "delta_phi_energy_unit_sd",
            "delta_phi_energy_unit_sem",
            "logZ_inf_full_mean",
            "logZ_inf_full_sd",
            "logZ_inf_full_sem",
        ]
    ].rename(
        columns={
            "phi_energy_raw_sd": "reference_sd_phi_energy_raw",
            "phi_energy_raw_sem": "reference_se_phi_energy_raw",
            "delta_phi_energy_unit_sd": "reference_sd_delta_phi_energy_unit",
            "delta_phi_energy_unit_sem": "reference_se_delta_phi_energy_unit",
            "logZ_inf_full_sd": "reference_sd_logZ_inf_full",
            "logZ_inf_full_sem": "reference_se_logZ_inf_full",
        }
    )
    _ensure_dir(QC_ROOT)
    logz.to_csv(QC_ROOT / "logZ_split_qc_results.csv", index=False)
    ref.to_csv(QC_ROOT / "reference_variability_results.csv", index=False)
    out = _ensure_dir(FIGURE_INPUT_ROOT / "logZ_split_qc_results")
    logz.to_csv(out / "logZ_split_qc_results.csv", index=False)
    out = _ensure_dir(FIGURE_INPUT_ROOT / "reference_variability_results")
    ref.to_csv(out / "reference_variability_results.csv", index=False)

    methods = sorted(str(value) for value in units["sampler_method"].dropna().unique())
    report = [
        "# MNIST manual-rule QC report",
        "",
        "- axis policy: manual rules, not beta.",
        f"- units: `{len(units)}` = 4 rules x 30 references x 100 radii.",
        f"- sampler methods: `{', '.join(methods)}`.",
        f"- tempered-path default: `{'tempered' in ' '.join(methods).lower()}`.",
        "- QC A: reference variability is summarized as SD and SE across references for each rule/radius.",
        "- QC B: split logZ stability is summarized by q95/max split logZ/P difference for each rule/radius.",
        "- Dataset variability is not plotted because this manual-rule MNIST stage uses a single dataset per rule.",
        "",
        "## Figure inputs",
        "",
        "- `figure_inputs/logZ_split_qc_results/logZ_split_qc_results.csv` -> `figures/logZ_split_qc_results.png`",
        "- `figure_inputs/reference_variability_results/reference_variability_results.csv` -> `figures/reference_variability_results.png`",
    ]
    (QC_ROOT / "qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return logz, ref


def build_outputs() -> dict[str, Path]:
    units = _load_units()
    _ensure_dir(CANONICAL_ROOT)
    units.to_csv(CANONICAL_UNIT_TABLE, index=False)

    rule_radius = _aggregate_rule_radius(units)
    by_ref, curve_summary, case_summary = _build_phase(units, rule_radius)

    rule_radius.to_csv(CANONICAL_ROOT / "curve_summary_by_rule_radius.csv", index=False)
    by_ref.to_csv(CANONICAL_ROOT / "A_kappa_by_reference.csv", index=False)
    curve_summary.to_csv(CANONICAL_ROOT / "phase_derivative_curve_summary_by_rule_radius.csv", index=False)
    case_summary.to_csv(CANONICAL_ROOT / "case_summary_A_kappa.csv", index=False)

    _write_figure_inputs(rule_radius, curve_summary, case_summary)
    _write_qc(units, rule_radius)

    config = {
        "stage": "05_proxy_local_entropy",
        "axis_policy": "manual_rule",
        "rule_axis": "rule_001..rule_004",
        "source_unit_table": str(_unit_source()),
        "canonical_unit_table": str(CANONICAL_UNIT_TABLE),
        "sampler_methods": sorted(str(value) for value in units["sampler_method"].dropna().unique()),
        "tempered_path_default": bool(units["sampler_method"].astype(str).str.contains("tempered", case=False).all()),
        "unit_count": int(len(units)),
        "target_ref_count_per_rule": TARGET_REF_COUNT,
        "radii_count_per_reference": int(units["radius"].nunique()),
        "split_logZ_threshold": SPLIT_LOGZ_THRESHOLD,
        "derivative_curve_policy": "Standalone derivative and A-measure panels use sampling-time direct radial score derivative from dlogZ_inf_full_dr / P.",
        "direct_radial_derivative_payload_present": bool("dlogZ_inf_full_dr" in units.columns),
        "radial_derivative_methodology": sorted(
            str(value) for value in units.get("radial_derivative_methodology_id", pd.Series(dtype=str)).dropna().unique()
        ),
        "retained_figures": [
            "phi_d_curve.png",
            "phi_energetic_d_curve.png",
            "derivative_phi_d_curve.png",
            "derivative_phi_energetic_d_curve.png",
            "phase_like_A_by_rule.png",
            "phase_like_A_by_complexity.png",
            "logZ_split_qc_results.png",
            "reference_variability_results.png",
        ],
    }
    _write_json(CANONICAL_ROOT / "run_config_resolved.json", config)
    return {
        "canonical_unit_table": CANONICAL_UNIT_TABLE,
        "curve_summary": CANONICAL_ROOT / "curve_summary_by_rule_radius.csv",
        "case_summary": CANONICAL_ROOT / "case_summary_A_kappa.csv",
        "qc_logz": QC_ROOT / "logZ_split_qc_results.csv",
        "qc_reference": QC_ROOT / "reference_variability_results.csv",
        "run_config": CANONICAL_ROOT / "run_config_resolved.json",
    }


def main() -> None:
    for path in build_outputs().values():
        print(path)


if __name__ == "__main__":
    main()
