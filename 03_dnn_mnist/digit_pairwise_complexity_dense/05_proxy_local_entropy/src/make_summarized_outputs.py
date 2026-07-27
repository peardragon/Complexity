from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
PAIRWISE_ROOT = STAGE_ROOT.parent
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
SAMPLING_UNIT_SUMMARY_ROOT = PAIRWISE_ROOT / "04_sampling" / "summarized_outputs" / "unit_summary"
DIRECT_UNIT_TABLE = SAMPLING_UNIT_SUMMARY_ROOT / "shell_summary_by_unit_with_phi_derivatives.csv"
FIGURE_INPUT_FILES = (
    FIGURE_INPUT_ROOT / "phi_d_curve" / "phi_d_curve.csv",
    FIGURE_INPUT_ROOT / "phi_energetic_d_curve" / "phi_energetic_d_curve.csv",
    FIGURE_INPUT_ROOT / "derivative_phi_d_curve" / "derivative_phi_d_curve.csv",
    FIGURE_INPUT_ROOT / "derivative_phi_energetic_d_curve" / "derivative_phi_energetic_d_curve.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_pair" / "phase_like_A_by_pair.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_pair" / "phase_derivative_curves.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_like_A_by_complexity.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_derivative_curves.csv",
)
DERIVATIVE_SOURCE = "sampling_time_direct_radial_score_derivative"
TARGET_REF_COUNT = 30


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _pair_sort_key(pair_id: str) -> tuple[int, int]:
    _prefix, a, b = str(pair_id).split("_")
    return int(a), int(b)


def _pair_summary_files() -> list[Path]:
    return sorted(SAMPLING_UNIT_SUMMARY_ROOT.glob("pair_*.csv"), key=lambda path: _pair_sort_key(path.stem))


def _unit_source(source_unit_table: Path | None = None) -> Path:
    if source_unit_table is not None:
        candidate = source_unit_table if source_unit_table.is_absolute() else (STAGE_ROOT / source_unit_table)
        if candidate.exists():
            return candidate
        candidate = (PAIRWISE_ROOT / source_unit_table).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"missing source unit table: {source_unit_table}")
    if DIRECT_UNIT_TABLE.exists():
        return DIRECT_UNIT_TABLE
    raise FileNotFoundError(f"missing source unit table: {DIRECT_UNIT_TABLE}")


def _retained_figure_inputs_complete() -> bool:
    return all(path.exists() for path in FIGURE_INPUT_FILES)


def _load_units_from_pair_summaries() -> pd.DataFrame:
    files = _pair_summary_files()
    if not files:
        raise FileNotFoundError(f"no pair_*.csv files found under {SAMPLING_UNIT_SUMMARY_ROOT}")
    frames = [pd.read_csv(path).assign(source_unit_summary=str(path)) for path in files]
    return pd.concat(frames, ignore_index=True)


def _load_units(source_unit_table: Path | None = None) -> pd.DataFrame:
    try:
        units = pd.read_csv(_unit_source(source_unit_table))
    except FileNotFoundError:
        if source_unit_table is None:
            units = _load_units_from_pair_summaries()
        else:
            raise

    required = {
        "pair_id",
        "pair_label",
        "pair_order",
        "pair_rank_complexity_desc",
        "complexity_mean",
        "ref_id",
        "ref_path_id",
        "radius",
        "logZ_full",
        "dlogZ_dr",
        "scale_value",
    }
    missing = required.difference(units.columns)
    if missing:
        raise ValueError(f"sampling unit summaries are missing columns: {sorted(missing)}")

    for col in [
        "pair_order",
        "pair_rank_complexity_desc",
        "complexity_mean",
        "ref_id",
        "radius",
        "logZ_full",
        "dlogZ_dr",
        "scale_value",
        "split_logZ_per_scale_diff",
        "split_dlogZ_dr_per_scale_diff",
        "ess_fraction",
        "smc_min_cess_fraction",
    ]:
        if col in units.columns:
            units[col] = pd.to_numeric(units[col], errors="coerce")

    units["phi_energy_raw"] = units["logZ_full"] / units["scale_value"]
    units["d_phi_energy_direct_dd_unit"] = units["dlogZ_dr"] / units["scale_value"]
    units["d_delta_phi_energy_direct_dd_unit"] = units["d_phi_energy_direct_dd_unit"]
    baseline = units.loc[
        np.isclose(units["radius"], 0.1),
        ["pair_id", "ref_path_id", "phi_energy_raw"],
    ].rename(columns={"phi_energy_raw": "phi_r0"})
    units = units.merge(baseline, on=["pair_id", "ref_path_id"], how="left", validate="many_to_one")
    if units["phi_r0"].isna().any():
        missing_baseline = units.loc[units["phi_r0"].isna(), ["pair_id", "ref_path_id"]].drop_duplicates().head(20)
        raise ValueError(f"missing r0=0.1 baseline rows in digit-pair sampling summaries:\n{missing_baseline}")
    units["delta_phi_energy_unit"] = units["phi_energy_raw"] - units["phi_r0"]
    units["delta_phi_full_unit"] = units["delta_phi_energy_unit"]
    units["d_phi_energy_raw_dd"] = units["d_phi_energy_direct_dd_unit"]
    units["d_delta_phi_energy_unit_dd"] = units["d_phi_energy_direct_dd_unit"]
    units["d_delta_phi_full_unit_dd"] = units["d_phi_energy_direct_dd_unit"]
    return units.sort_values(["pair_order", "ref_path_id", "radius"]).reset_index(drop=True)


def _aggregate_pair_radius(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "pair_id",
        "pair_label",
        "pair_order",
        "pair_rank_complexity_desc",
        "complexity_mean",
        "radius",
    ]
    for key, sub in units.groupby(group_cols, dropna=False, sort=True):
        pair_id, pair_label, pair_order, rank, complexity, radius = key
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "pair_label": pair_label,
            "pair_order": int(pair_order),
            "pair_rank_complexity_desc": int(rank),
            "complexity_mean": float(complexity),
            "radius": float(radius),
            "radius_path_id": str(sub["radius_path_id"].iloc[0]) if "radius_path_id" in sub.columns else "",
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
            "logZ_full",
            "dlogZ_dr",
            "split_logZ_per_scale_diff",
            "split_dlogZ_dr_per_scale_diff",
            "ess_fraction",
            "smc_min_cess_fraction",
        ]:
            if col in sub.columns:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_sd"] = _sd(sub[col])
                row[f"{col}_sem"] = _sem(sub[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pair_order", "radius"]).reset_index(drop=True)


def _build_phase(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for (pair_id, ref_path_id), sub in units.groupby(["pair_id", "ref_path_id"], sort=True):
        sub = sub.sort_values("radius")
        radius = sub["radius"].to_numpy(dtype=float)
        dphi = sub["d_phi_energy_direct_dd_unit"].to_numpy(dtype=float)
        if len(radius) < 3 or not np.isfinite(dphi).all():
            continue
        curvature = np.gradient(dphi, radius)
        row0 = sub.iloc[0]
        ref_rows.append(
            {
                "pair_id": pair_id,
                "pair_label": row0["pair_label"],
                "pair_order": int(row0["pair_order"]),
                "pair_rank_complexity_desc": int(row0["pair_rank_complexity_desc"]),
                "complexity_mean": float(row0["complexity_mean"]),
                "ref_path_id": ref_path_id,
                "ref_id": int(row0["ref_id"]),
                "A_kappa": _trapz(np.maximum(curvature, 0.0), radius),
                "derivative_source": DERIVATIVE_SOURCE,
            }
        )
        for r, dp, curv in zip(radius, dphi, curvature):
            curve_rows.append(
                {
                    "pair_id": pair_id,
                    "pair_label": row0["pair_label"],
                    "pair_order": int(row0["pair_order"]),
                    "pair_rank_complexity_desc": int(row0["pair_rank_complexity_desc"]),
                    "complexity_mean": float(row0["complexity_mean"]),
                    "ref_path_id": ref_path_id,
                    "radius": float(r),
                    "dphi_dr": float(dp),
                    "dphi_dr_smooth": float(dp),
                    "positive_curvature": float(max(curv, 0.0)),
                    "derivative_source": DERIVATIVE_SOURCE,
                }
            )
    by_ref = pd.DataFrame(ref_rows).sort_values(["pair_order", "ref_path_id"]).reset_index(drop=True)
    ref_curves = pd.DataFrame(curve_rows)
    if by_ref.empty or ref_curves.empty:
        raise ValueError("could not compute phase-like A from digit-pair sampling summaries")
    curve_summary = (
        ref_curves.groupby(
            ["pair_id", "pair_label", "pair_order", "pair_rank_complexity_desc", "complexity_mean", "radius"],
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
        .sort_values(["pair_order", "radius"])
    )
    case_summary = (
        by_ref.groupby(["pair_id", "pair_label", "pair_order", "pair_rank_complexity_desc", "complexity_mean"], sort=True)
        .agg(
            n_refs=("ref_path_id", "nunique"),
            A_kappa_mean=("A_kappa", "mean"),
            A_kappa_sd=("A_kappa", _sd),
            A_kappa_sem=("A_kappa", _sem),
            derivative_source=("derivative_source", "first"),
        )
        .reset_index()
        .sort_values("pair_order")
    )
    return curve_summary, case_summary


def _write_figure_inputs(pair_radius: pd.DataFrame, curve_summary: pd.DataFrame, case_summary: pd.DataFrame) -> None:
    phi = pair_radius[
        [
            "pair_id",
            "pair_label",
            "pair_order",
            "pair_rank_complexity_desc",
            "complexity_mean",
            "radius",
            "n_units",
            "delta_phi_energy_unit_mean",
            "delta_phi_energy_unit_sem",
            "phi_energy_raw_mean",
            "phi_energy_raw_sem",
        ]
    ].copy()
    dphi = pair_radius[
        [
            "pair_id",
            "pair_label",
            "pair_order",
            "pair_rank_complexity_desc",
            "complexity_mean",
            "radius",
            "n_units",
            "d_phi_energy_direct_dd_unit_mean",
            "d_phi_energy_direct_dd_unit_sem",
            "d_delta_phi_energy_direct_dd_unit_mean",
            "d_delta_phi_energy_direct_dd_unit_sem",
        ]
    ].copy()
    dphi["derivative_source"] = DERIVATIVE_SOURCE
    curve_summary = curve_summary.drop(
        columns=["dphi_dr_smooth_mean", "dphi_dr_smooth_sem", "derivative_source"],
        errors="ignore",
    ).merge(
        dphi[
            [
                "pair_id",
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
        on=["pair_id", "radius"],
        how="left",
    )

    for name in ("phi_d_curve", "phi_energetic_d_curve"):
        out = _ensure_dir(FIGURE_INPUT_ROOT / name)
        phi.to_csv(out / f"{name}.csv", index=False)
    for name in ("derivative_phi_d_curve", "derivative_phi_energetic_d_curve"):
        out = _ensure_dir(FIGURE_INPUT_ROOT / name)
        dphi.to_csv(out / f"{name}.csv", index=False)
    for name in ("phase_like_A_by_pair", "phase_like_A_by_complexity"):
        out = _ensure_dir(FIGURE_INPUT_ROOT / name)
        case_summary.to_csv(out / f"{name}.csv", index=False)
        curve_summary.to_csv(out / "phase_derivative_curves.csv", index=False)


def build(source_unit_table: Path | None = None) -> dict[str, Path]:
    try:
        units = _load_units(source_unit_table)
    except FileNotFoundError:
        if source_unit_table is None and _retained_figure_inputs_complete():
            return {"figure_inputs": FIGURE_INPUT_ROOT}
        raise
    pair_radius = _aggregate_pair_radius(units)
    curve_summary, case_summary = _build_phase(units)
    _write_figure_inputs(pair_radius, curve_summary, case_summary)
    return {"figure_inputs": FIGURE_INPUT_ROOT}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build summarized_outputs for MNIST digit-pair proxy local entropy.")
    parser.add_argument("--source-unit-table", type=Path, default=None)
    args = parser.parse_args()
    for path in build(args.source_unit_table).values():
        print(path)


if __name__ == "__main__":
    main()
