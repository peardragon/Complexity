from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROOT = STAGE_ROOT.parent
DNN_ROOT = MANUAL_ROOT.parent
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
DIRECT_UNIT_TABLE = (
    MANUAL_ROOT
    / "04_sampling"
    / "summarized_outputs"
    / "unit_summary"
    / "shell_summary_by_unit_with_phi_derivatives.csv"
)
SAMPLING_UNIT_SUMMARY_ROOT = MANUAL_ROOT / "04_sampling" / "summarized_outputs" / "unit_summary"
UNIT_INDEX = MANUAL_ROOT / "04_sampling" / "raw_outputs" / "shell_pool" / "unit_index.csv"
RULE_MAPPING = MANUAL_ROOT / "config" / "rule_mapping.csv"
COMPLEXITY_TABLE = (
    MANUAL_ROOT / "02_complexity_measure" / "summarized_outputs" / "manual_rule_complexity_summary.csv"
)
FIGURE_INPUT_FILES = (
    FIGURE_INPUT_ROOT / "phi_d_curve" / "phi_d_curve.csv",
    FIGURE_INPUT_ROOT / "phi_energetic_d_curve" / "phi_energetic_d_curve.csv",
    FIGURE_INPUT_ROOT / "derivative_phi_d_curve" / "derivative_phi_d_curve.csv",
    FIGURE_INPUT_ROOT / "derivative_phi_energetic_d_curve" / "derivative_phi_energetic_d_curve.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_rule" / "phase_like_A_by_rule.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_rule" / "phase_derivative_curves.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_like_A_by_complexity.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_derivative_curves.csv",
)

TARGET_REF_COUNT = 30
RULE_ORDER = {
    "rule_001": 1,
    "rule_002": 2,
    "rule_003": 3,
    "rule_004": 4,
}
def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manual_stage_path(*parts: object) -> str:
    return (Path("..") / Path(*(str(part) for part in parts))).as_posix()


def _normalize_metadata_path(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\\", "/")
    if text in {"", "nan"}:
        return ""
    if text.startswith("../"):
        return text

    prefixes = (
        f"{MANUAL_ROOT.resolve().as_posix()}/",
        f"{MANUAL_ROOT.as_posix()}/",
        "Complexity/03_dnn_mnist/manual_rules/",
        "manual_rules/",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return _manual_stage_path(text.removeprefix(prefix))

    manual_roots = (
        "01_dataset/",
        "02_complexity_measure/",
        "03_reference_search/",
        "04_sampling/",
        "config/",
    )
    if text.startswith(manual_roots):
        return _manual_stage_path(text)
    return text


def _theta_ref_path_id(ref_id: object) -> str:
    return f"ref_{int(ref_id) + 1:03d}"


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
    if not COMPLEXITY_TABLE.exists():
        raise FileNotFoundError(
            f"run manual_rules/02_complexity_measure/src/make_summarized_outputs.py first: {COMPLEXITY_TABLE}"
        )

    complexity = pd.read_csv(COMPLEXITY_TABLE)
    complexity = complexity.rename(columns={"rule": "rule_name", "label": "complexity_label"})
    if "nmstv_mean" not in complexity.columns:
        if "complexity_mean" not in complexity.columns:
            raise ValueError(f"{COMPLEXITY_TABLE} is missing nmstv_mean or complexity_mean")
        complexity["nmstv_mean"] = complexity["complexity_mean"]
    if "n_datasets" not in complexity.columns and "dataset_count" in complexity.columns:
        complexity["n_datasets"] = complexity["dataset_count"]
    for col in ["tv_mean", "n_datasets"]:
        if col not in complexity.columns:
            complexity[col] = np.nan
    keep = [c for c in ["rule_name", "nmstv_mean", "tv_mean", "n_datasets"] if c in complexity.columns]
    complexity = complexity[keep].copy()
    out = rules.merge(complexity, on="rule_name", how="left")
    if out["nmstv_mean"].isna().any():
        missing = out[out["nmstv_mean"].isna()][["rule_id", "rule_name"]]
        raise ValueError(f"complexity mapping failed for rules:\n{missing}")
    return out


def _resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = (
        STAGE_ROOT / path,
        MANUAL_ROOT / path,
        DNN_ROOT / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _unit_source(source_unit_table: Path | None = None) -> Path:
    if source_unit_table is not None:
        candidate = _resolve_input_path(source_unit_table)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"missing source unit table: {candidate}")

    if DIRECT_UNIT_TABLE.exists():
        return DIRECT_UNIT_TABLE
    raise FileNotFoundError(
        "missing source unit table. searched:\n"
        f"{DIRECT_UNIT_TABLE}"
    )


def _rule_summary_files() -> list[Path]:
    return sorted(
        SAMPLING_UNIT_SUMMARY_ROOT.glob("rule_*.csv"),
        key=lambda path: RULE_ORDER.get(path.stem, 999),
    )


def _load_units_from_sampling_summaries(complexity: pd.DataFrame) -> pd.DataFrame:
    files = _rule_summary_files()
    if not files:
        raise FileNotFoundError(f"no rule_*.csv files found under {SAMPLING_UNIT_SUMMARY_ROOT}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        if "rule_id" not in frame.columns:
            frame["rule_id"] = str(path.stem)
        frame["source_unit_summary"] = str(path)
        frames.append(frame)
    units = pd.concat(frames, ignore_index=True)

    required = {"rule_id", "ref_id", "radius", "logZ_full", "dlogZ_dr", "scale_value"}
    missing = required.difference(units.columns)
    if missing:
        raise ValueError(f"{SAMPLING_UNIT_SUMMARY_ROOT} summary CSVs are missing columns: {sorted(missing)}")

    units["rule_id"] = units["rule_id"].astype(str)
    units["ref_id"] = pd.to_numeric(units["ref_id"], errors="coerce").astype("Int64")
    units["radius"] = pd.to_numeric(units["radius"], errors="coerce")
    units["scale_value"] = pd.to_numeric(units["scale_value"], errors="coerce")
    units["phi_energy_raw"] = pd.to_numeric(units["logZ_full"], errors="coerce") / units["scale_value"]
    units["d_phi_energy_direct_dd_unit"] = pd.to_numeric(units["dlogZ_dr"], errors="coerce") / units["scale_value"]
    units["d_delta_phi_energy_direct_dd_unit"] = units["d_phi_energy_direct_dd_unit"]

    baseline = units.loc[np.isclose(units["radius"], 0.1), ["rule_id", "ref_id", "phi_energy_raw"]].rename(
        columns={"phi_energy_raw": "phi_r0"}
    )
    units = units.merge(baseline, on=["rule_id", "ref_id"], how="left", validate="many_to_one")
    if units["phi_r0"].isna().any():
        missing_baseline = units.loc[units["phi_r0"].isna(), ["rule_id", "ref_id"]].drop_duplicates().head(20)
        raise ValueError(f"missing r0=0.1 baseline rows in manual-rule sampling summaries:\n{missing_baseline}")
    units["delta_phi_energy_unit"] = units["phi_energy_raw"] - units["phi_r0"]
    units["delta_phi_full_unit"] = units["delta_phi_energy_unit"]
    units["d_phi_energy_raw_dd"] = units["d_phi_energy_direct_dd_unit"]
    units["d_delta_phi_energy_unit_dd"] = units["d_phi_energy_direct_dd_unit"]
    units["d_delta_phi_full_unit_dd"] = units["d_phi_energy_direct_dd_unit"]

    metadata = complexity[
        [
            "rule_id",
            "rule_name",
            "label",
            "rule_order",
            "nmstv_mean",
            "tv_mean",
            "n_datasets",
        ]
    ].copy()
    merged = units.merge(metadata, on="rule_id", how="left", validate="many_to_one")
    if merged["rule_name"].isna().any():
        missing_rules = merged.loc[merged["rule_name"].isna(), ["rule_id"]].drop_duplicates().head(20)
        raise ValueError(f"rule mapping failed for sampling summaries:\n{missing_rules}")
    merged["rule_label"] = merged["label"]
    merged["rule"] = merged["rule_name"]
    merged["ref_id"] = merged["ref_id"].astype(int)
    merged["ref_path_id"] = merged["ref_id"].map(lambda value: f"ref_{int(value):03d}")
    merged["radius_path_id"] = merged["radius"].map(lambda value: f"r_{float(value):0.4f}".replace(".", "p"))
    merged["logZ_inf_full"] = pd.to_numeric(merged["logZ_full"], errors="coerce")
    merged["dlogZ_inf_full_dr"] = pd.to_numeric(merged["dlogZ_dr"], errors="coerce")
    merged.attrs["source_unit_table"] = str(SAMPLING_UNIT_SUMMARY_ROOT)
    return merged.sort_values(["rule_order", "ref_path_id", "radius"]).reset_index(drop=True)


def _retained_figure_inputs_complete() -> bool:
    return all(path.exists() for path in FIGURE_INPUT_FILES)


def _load_units(source_unit_table: Path | None = None) -> pd.DataFrame:
    rules = _load_rules()
    complexity = _load_complexity(rules)
    try:
        unit_source = _unit_source(source_unit_table)
    except FileNotFoundError:
        if source_unit_table is None:
            return _load_units_from_sampling_summaries(complexity)
        raise

    units = pd.read_csv(unit_source)
    units.attrs["source_unit_table"] = str(unit_source)

    units["radius_key"] = pd.to_numeric(units["radius"], errors="coerce").round(10)
    units["ref_id"] = pd.to_numeric(units["ref_id"], errors="coerce").astype("Int64")

    if "rule_id" in units.columns and "ref_path_id" in units.columns:
        merged = units.copy()
        if "rule_name" not in merged.columns and "rule" in merged.columns:
            merged["rule_name"] = merged["rule"].astype(str)
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
            _manual_stage_path(
                "03_reference_search",
                "raw_outputs",
                row["rule_id"],
                _theta_ref_path_id(row["ref_id"]),
                "theta.npy",
            )
        ),
        axis=1,
    )
    merged["dataset_path"] = merged.apply(
        lambda row: (
            _manual_stage_path("01_dataset", "raw_outputs", row["rule_id"], "dataset.npz")
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
    for col in ["dataset_path", "theta_path", "samples_path", "unit_summary_path"]:
        if col in merged.columns:
            merged[col] = merged[col].map(_normalize_metadata_path)
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


def build(source_unit_table: Path | None = None) -> dict[str, Path]:
    try:
        units = _load_units(source_unit_table)
    except FileNotFoundError:
        if source_unit_table is None and _retained_figure_inputs_complete():
            return {
                "figure_inputs": FIGURE_INPUT_ROOT,
            }
        raise

    rule_radius = _aggregate_rule_radius(units)
    _by_ref, curve_summary, case_summary = _build_phase(units, rule_radius)

    _write_figure_inputs(rule_radius, curve_summary, case_summary)
    return {
        "figure_inputs": FIGURE_INPUT_ROOT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build standalone summarized_outputs for MNIST manual-rule proxy local entropy."
    )
    parser.add_argument(
        "--source-unit-table",
        type=Path,
        default=None,
        help=(
            "Optional source shell_summary_by_unit_with_phi_derivatives.csv. "
            "Relative paths are resolved from the stage, manual_rules, DNN root, then cwd."
        ),
    )
    args = parser.parse_args()
    for path in build(args.source_unit_table).values():
        print(path)


if __name__ == "__main__":
    main()
