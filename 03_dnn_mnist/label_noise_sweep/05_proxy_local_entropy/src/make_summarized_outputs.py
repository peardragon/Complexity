from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = STAGE_ROOT.parent
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
SOURCE_SUMMARY_REL = Path("direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0")
DERIVATIVE_SOURCE = "sampling_time_direct_radial_score_derivative"
COMPLEXITY_SUMMARY_PATH = SWEEP_ROOT / "02_complexity_measure" / "summarized_outputs" / "eta_complexity_summary.csv"
SAMPLING_UNIT_SUMMARY_ROOT = SWEEP_ROOT / "04_sampling" / "summarized_outputs" / "unit_summary"
FIGURE_INPUT_FILES = (
    FIGURE_INPUT_ROOT / "phi_d_curve" / "phi_d_curve.csv",
    FIGURE_INPUT_ROOT / "phi_energetic_d_curve" / "phi_energetic_d_curve.csv",
    FIGURE_INPUT_ROOT / "derivative_phi_d_curve" / "derivative_phi_d_curve.csv",
    FIGURE_INPUT_ROOT / "derivative_phi_energetic_d_curve" / "derivative_phi_energetic_d_curve.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_eta" / "phase_like_A_by_eta.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_eta" / "phase_derivative_curves.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_like_A_by_complexity.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_derivative_curves.csv",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _eta_from_token(value: object) -> float:
    text = str(value)
    for prefix in ("noise_eta_", "eta_"):
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
            break
    return float(text.replace("p", "."))


def _eta_case_id(eta: float) -> str:
    return f"eta_{eta:.2f}"


def _eta_rule_label(eta: float) -> str:
    return f"eta_{eta:.2f}".replace(".", "p")


def _eta_complexity_lookup() -> dict[float, float]:
    if not COMPLEXITY_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"run label_noise_sweep/02_complexity_measure/src/make_summarized_outputs.py first: {COMPLEXITY_SUMMARY_PATH}"
        )
    lookup: dict[float, float] = {}
    for row in _read_csv(COMPLEXITY_SUMMARY_PATH):
        eta = _float(row.get("eta"))
        complexity = _float(row.get("complexity_mean"))
        if np.isfinite(eta) and np.isfinite(complexity):
            lookup[round(float(eta), 10)] = float(complexity)
    return lookup


def _find_source_summary_root(source_summary_root: Path | None) -> Path:
    if source_summary_root is not None:
        candidates = [source_summary_root]
    else:
        candidates = [SUMMARY_ROOT / SOURCE_SUMMARY_REL]

    required = (
        "eta_reference_phi_by_eta_radius.csv",
        "eta_reference_dphi_dd_by_eta_radius.csv",
        "combined_direct_curvature_metrics_by_group.csv",
        "combined_direct_curvature_curve_by_group_radius.csv",
    )
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find retained source summary tables. searched:\n{searched}")


def _retained_figure_inputs_complete() -> bool:
    return all(path.exists() for path in FIGURE_INPUT_FILES)


def _rewrite_nmstv_from_complexity(path: Path, eta_complexity: dict[float, float]) -> None:
    rows = _read_csv(path)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if "eta" not in fieldnames or "nmstv" not in fieldnames:
        return
    for row in rows:
        eta = _float(row.get("eta"))
        if not np.isfinite(eta):
            continue
        row["nmstv"] = _complexity_for_eta(eta_complexity, eta)
    _write_csv(path, rows, fieldnames)


def _repair_retained_phase_complexity() -> None:
    eta_complexity = _eta_complexity_lookup()
    for name in ("phase_like_A_by_eta", "phase_like_A_by_complexity"):
        _rewrite_nmstv_from_complexity(FIGURE_INPUT_ROOT / name / f"{name}.csv", eta_complexity)
        _rewrite_nmstv_from_complexity(FIGURE_INPUT_ROOT / name / "phase_derivative_curves.csv", eta_complexity)


def _complexity_for_eta(eta_complexity: dict[float, float], eta: float) -> float:
    key = round(float(eta), 10)
    if key not in eta_complexity:
        raise ValueError(f"{COMPLEXITY_SUMMARY_PATH} is missing eta={eta:.10g}")
    return eta_complexity[key]


def _sampling_unit_summary_files() -> list[Path]:
    return sorted(
        SAMPLING_UNIT_SUMMARY_ROOT.glob("eta_*.csv"),
        key=lambda path: _eta_from_token(path.stem),
    )


def _load_sampling_units() -> pd.DataFrame:
    files = _sampling_unit_summary_files()
    if not files:
        raise FileNotFoundError(f"no eta_*.csv files found under {SAMPLING_UNIT_SUMMARY_ROOT}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        if "eta" not in frame.columns:
            frame["eta"] = pd.to_numeric(frame.get("condition_value"), errors="coerce")
        frame["source_unit_summary"] = str(path)
        frames.append(frame)
    units = pd.concat(frames, ignore_index=True)

    required = {"eta", "ref_id", "radius", "logZ_full", "dlogZ_dr", "scale_value"}
    missing = required.difference(units.columns)
    if missing:
        raise ValueError(f"{SAMPLING_UNIT_SUMMARY_ROOT} summary CSVs are missing columns: {sorted(missing)}")

    units["eta"] = pd.to_numeric(units["eta"], errors="coerce")
    units["ref_id"] = pd.to_numeric(units["ref_id"], errors="coerce").astype("Int64")
    units["radius"] = pd.to_numeric(units["radius"], errors="coerce")
    units["scale_value"] = pd.to_numeric(units["scale_value"], errors="coerce")
    units["phi_energy_raw"] = pd.to_numeric(units["logZ_full"], errors="coerce") / units["scale_value"]
    units["d_phi_energy_direct_dd_unit"] = pd.to_numeric(units["dlogZ_dr"], errors="coerce") / units["scale_value"]
    units["d_delta_phi_energy_direct_dd_unit"] = units["d_phi_energy_direct_dd_unit"]

    baseline = units.loc[np.isclose(units["radius"], 0.1), ["eta", "ref_id", "phi_energy_raw"]].rename(
        columns={"phi_energy_raw": "phi_r0"}
    )
    units = units.merge(baseline, on=["eta", "ref_id"], how="left", validate="many_to_one")
    if units["phi_r0"].isna().any():
        missing = units.loc[units["phi_r0"].isna(), ["eta", "ref_id"]].drop_duplicates().head(20)
        raise ValueError(f"missing r0=0.1 baseline rows in label-noise sampling summaries:\n{missing}")
    units["delta_phi_energy_unit"] = units["phi_energy_raw"] - units["phi_r0"]
    units = units.dropna(subset=["eta", "ref_id", "radius", "phi_energy_raw", "d_phi_energy_direct_dd_unit"])
    return units.sort_values(["eta", "ref_id", "radius"]).reset_index(drop=True)


def _build_from_sampling_unit_summaries() -> None:
    units = _load_sampling_units()
    eta_complexity = _eta_complexity_lookup()

    curve_summary = (
        units.groupby(["eta", "radius"], dropna=False, sort=True)
        .agg(
            n_units=("ref_id", "count"),
            delta_phi_energy_mean=("delta_phi_energy_unit", "mean"),
            delta_phi_energy_sem=("delta_phi_energy_unit", _sem),
            phi_energy_raw_mean=("phi_energy_raw", "mean"),
            phi_energy_raw_sem=("phi_energy_raw", _sem),
            d_phi_energy_direct_dd=("d_phi_energy_direct_dd_unit", "mean"),
            d_phi_energy_direct_dd_sem=("d_phi_energy_direct_dd_unit", _sem),
        )
        .reset_index()
        .sort_values(["eta", "radius"])
    )
    curve_summary["rule"] = curve_summary["eta"].map(_eta_rule_label)

    phi_inputs = curve_summary[
        [
            "eta",
            "rule",
            "radius",
            "n_units",
            "delta_phi_energy_mean",
            "delta_phi_energy_sem",
            "phi_energy_raw_mean",
            "phi_energy_raw_sem",
        ]
    ].copy()
    dphi_inputs = curve_summary[
        [
            "eta",
            "rule",
            "radius",
            "n_units",
            "d_phi_energy_direct_dd",
            "d_phi_energy_direct_dd_sem",
        ]
    ].copy()
    dphi_inputs["d_delta_phi_energy_dd"] = dphi_inputs["d_phi_energy_direct_dd"]
    dphi_inputs["d_delta_phi_energy_dd_sem"] = dphi_inputs["d_phi_energy_direct_dd_sem"]
    dphi_inputs["derivative_source"] = DERIVATIVE_SOURCE
    dphi_inputs = dphi_inputs[
        [
            "eta",
            "rule",
            "radius",
            "n_units",
            "d_delta_phi_energy_dd",
            "d_delta_phi_energy_dd_sem",
            "d_phi_energy_direct_dd",
            "d_phi_energy_direct_dd_sem",
            "derivative_source",
        ]
    ]

    ref_rows: list[dict[str, object]] = []
    phase_curve_rows: list[dict[str, object]] = []
    for (eta, ref_id), sub in units.groupby(["eta", "ref_id"], sort=True):
        sub = sub.sort_values("radius")
        radius = sub["radius"].to_numpy(dtype=float)
        dphi = sub["d_phi_energy_direct_dd_unit"].to_numpy(dtype=float)
        if len(radius) < 3 or not np.isfinite(dphi).all():
            continue
        curvature = np.gradient(dphi, radius)
        ref_rows.append(
            {
                "eta": float(eta),
                "ref_id": int(ref_id),
                "A_kappa": _trapz(np.maximum(curvature, 0.0), radius),
            }
        )
        for r, dp, curv in zip(radius, dphi, curvature):
            phase_curve_rows.append(
                {
                    "eta": float(eta),
                    "radius": float(r),
                    "dphi_dr_smooth": float(dp),
                    "positive_curvature": float(max(curv, 0.0)),
                }
            )

    by_ref = pd.DataFrame(ref_rows)
    if by_ref.empty:
        raise ValueError("could not compute phase-like A from sampling unit summaries")
    phase_summary = (
        by_ref.groupby("eta", sort=True)
        .agg(
            n_refs=("ref_id", "nunique"),
            A_kappa_mean=("A_kappa", "mean"),
            A_kappa_sem=("A_kappa", _sem),
        )
        .reset_index()
        .sort_values("eta")
    )
    phase_summary["source"] = "flip"
    phase_summary["case_id"] = phase_summary["eta"].map(_eta_case_id)
    phase_summary["case_label"] = phase_summary["eta"].map(lambda eta: f"flip eta={float(eta):.2f}")
    phase_summary["nmstv"] = phase_summary["eta"].map(lambda eta: _complexity_for_eta(eta_complexity, float(eta)))
    phase_summary["derivative_source"] = DERIVATIVE_SOURCE
    phase_inputs = phase_summary[
        [
            "source",
            "case_id",
            "case_label",
            "eta",
            "nmstv",
            "n_refs",
            "A_kappa_mean",
            "A_kappa_sem",
            "derivative_source",
        ]
    ]

    phase_curve = (
        pd.DataFrame(phase_curve_rows)
        .groupby(["eta", "radius"], sort=True)
        .agg(
            n_refs=("dphi_dr_smooth", "count"),
            dphi_dr_smooth_mean=("dphi_dr_smooth", "mean"),
            dphi_dr_smooth_sem=("dphi_dr_smooth", _sem),
        )
        .reset_index()
        .sort_values(["eta", "radius"])
    )
    phase_curve["source"] = "flip"
    phase_curve["case_id"] = phase_curve["eta"].map(_eta_case_id)
    phase_curve["case_label"] = phase_curve["eta"].map(lambda eta: f"flip eta={float(eta):.2f}")
    phase_curve["nmstv"] = phase_curve["eta"].map(lambda eta: _complexity_for_eta(eta_complexity, float(eta)))
    phase_curve["derivative_source"] = DERIVATIVE_SOURCE
    phase_curve_inputs = phase_curve[
        [
            "source",
            "case_id",
            "case_label",
            "eta",
            "nmstv",
            "radius",
            "n_refs",
            "dphi_dr_smooth_mean",
            "dphi_dr_smooth_sem",
            "derivative_source",
        ]
    ]

    FIGURE_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("phi_d_curve", "phi_energetic_d_curve"):
        out = FIGURE_INPUT_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        phi_inputs.to_csv(out / f"{name}.csv", index=False)
    for name in ("derivative_phi_d_curve", "derivative_phi_energetic_d_curve"):
        out = FIGURE_INPUT_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        dphi_inputs.to_csv(out / f"{name}.csv", index=False)
    for name in ("phase_like_A_by_eta", "phase_like_A_by_complexity"):
        out = FIGURE_INPUT_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        phase_inputs.to_csv(out / f"{name}.csv", index=False)
        phase_curve_inputs.to_csv(out / "phase_derivative_curves.csv", index=False)


def _phi_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "eta": row.get("eta", ""),
                "rule": row.get("rule", ""),
                "radius": row.get("radius", ""),
                "n_units": row.get("n_units", ""),
                "delta_phi_energy_mean": row.get("delta_phi_energy_mean", ""),
                "delta_phi_energy_sem": row.get("delta_phi_energy_sem", ""),
                "phi_energy_raw_mean": row.get("phi_energy_raw_mean", ""),
                "phi_energy_raw_sem": row.get("phi_energy_raw_sem", ""),
            }
        )
    return out


def _dphi_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        dphi = row.get("d_phi_energy_direct_dd_unit_mean", "")
        dphi_sem = row.get("d_phi_energy_direct_dd_unit_sem", "")
        out.append(
            {
                "eta": row.get("eta", ""),
                "rule": row.get("rule", ""),
                "radius": row.get("radius", ""),
                "n_units": row.get("n_units", ""),
                "d_delta_phi_energy_dd": dphi,
                "d_delta_phi_energy_dd_sem": dphi_sem,
                "d_phi_energy_direct_dd": dphi,
                "d_phi_energy_direct_dd_sem": dphi_sem,
                "derivative_source": DERIVATIVE_SOURCE,
            }
        )
    return out


def _phase_rows(
    metric_rows: list[dict[str, str]],
    curve_rows_source: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    phase_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    eta_complexity = _eta_complexity_lookup()

    for row in metric_rows:
        if str(row.get("source", "")) != "flip":
            continue
        eta = _eta_from_token(row.get("group", ""))
        complexity = _complexity_for_eta(eta_complexity, eta)
        phase_rows.append(
            {
                "source": "flip",
                "case_id": _eta_case_id(eta),
                "case_label": row.get("label", f"flip eta={eta:.2f}"),
                "eta": eta,
                "nmstv": complexity,
                "n_refs": row.get("n_refs", ""),
                "A_kappa_mean": row.get("positive_curvature_mass_mean", ""),
                "A_kappa_sem": row.get("positive_curvature_mass_sem", ""),
                "derivative_source": DERIVATIVE_SOURCE,
            }
        )

    for row in curve_rows_source:
        if str(row.get("source", "")) != "flip":
            continue
        eta = _eta_from_token(row.get("group", ""))
        complexity = _complexity_for_eta(eta_complexity, eta)
        curve_rows.append(
            {
                "source": "flip",
                "case_id": _eta_case_id(eta),
                "case_label": row.get("label", f"flip eta={eta:.2f}"),
                "eta": eta,
                "nmstv": complexity,
                "radius": row.get("radius", ""),
                "n_refs": row.get("n_refs", ""),
                "dphi_dr_smooth_mean": row.get("d_phi_energy_direct_dd_mean", ""),
                "dphi_dr_smooth_sem": row.get("d_phi_energy_direct_dd_sem", ""),
                "derivative_source": DERIVATIVE_SOURCE,
            }
        )

    phase_rows.sort(key=lambda row: _float(row.get("eta")))
    curve_rows.sort(key=lambda row: (_float(row.get("eta")), _float(row.get("radius"))))
    return phase_rows, curve_rows


def build(source_summary_root: Path | None = None) -> None:
    try:
        source_root = _find_source_summary_root(source_summary_root)
    except FileNotFoundError:
        if source_summary_root is None:
            try:
                _build_from_sampling_unit_summaries()
                return
            except FileNotFoundError:
                pass
        if source_summary_root is None and _retained_figure_inputs_complete():
            _repair_retained_phase_complexity()
            return
        raise

    phi_source = _read_csv(source_root / "eta_reference_phi_by_eta_radius.csv")
    dphi_source = _read_csv(source_root / "eta_reference_dphi_dd_by_eta_radius.csv")
    phase_source = _read_csv(source_root / "combined_direct_curvature_metrics_by_group.csv")
    curve_source = _read_csv(source_root / "combined_direct_curvature_curve_by_group_radius.csv")

    FIGURE_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    phi_inputs = _phi_rows(phi_source)
    dphi_inputs = _dphi_rows(dphi_source)
    phase_inputs, phase_curve_inputs = _phase_rows(phase_source, curve_source)

    phi_fields = [
        "eta",
        "rule",
        "radius",
        "n_units",
        "delta_phi_energy_mean",
        "delta_phi_energy_sem",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
    ]
    dphi_fields = [
        "eta",
        "rule",
        "radius",
        "n_units",
        "d_delta_phi_energy_dd",
        "d_delta_phi_energy_dd_sem",
        "d_phi_energy_direct_dd",
        "d_phi_energy_direct_dd_sem",
        "derivative_source",
    ]
    phase_fields = [
        "source",
        "case_id",
        "case_label",
        "eta",
        "nmstv",
        "n_refs",
        "A_kappa_mean",
        "A_kappa_sem",
        "derivative_source",
    ]
    phase_curve_fields = [
        "source",
        "case_id",
        "case_label",
        "eta",
        "nmstv",
        "radius",
        "n_refs",
        "dphi_dr_smooth_mean",
        "dphi_dr_smooth_sem",
        "derivative_source",
    ]
    _write_csv(FIGURE_INPUT_ROOT / "phi_d_curve" / "phi_d_curve.csv", phi_inputs, phi_fields)
    _write_csv(FIGURE_INPUT_ROOT / "phi_energetic_d_curve" / "phi_energetic_d_curve.csv", phi_inputs, phi_fields)
    _write_csv(FIGURE_INPUT_ROOT / "derivative_phi_d_curve" / "derivative_phi_d_curve.csv", dphi_inputs, dphi_fields)
    _write_csv(
        FIGURE_INPUT_ROOT / "derivative_phi_energetic_d_curve" / "derivative_phi_energetic_d_curve.csv",
        dphi_inputs,
        dphi_fields,
    )
    for name in ("phase_like_A_by_eta", "phase_like_A_by_complexity"):
        _write_csv(FIGURE_INPUT_ROOT / name / f"{name}.csv", phase_inputs, phase_fields)
        _write_csv(FIGURE_INPUT_ROOT / name / "phase_derivative_curves.csv", phase_curve_inputs, phase_curve_fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build figure-specific summarized_outputs for MNIST label-noise proxy local entropy."
    )
    parser.add_argument(
        "--source-summary-root",
        type=Path,
        default=None,
        help="Retained source summary directory. Defaults to this stage's direct-derivative summary.",
    )
    args = parser.parse_args()
    build(args.source_summary_root)


if __name__ == "__main__":
    main()
