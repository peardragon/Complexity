from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


DNN_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = DNN_ROOT / "05_proxy_local_entropy"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
COMPLEXITY_SUMMARY = (
    DNN_ROOT / "02_complexity_measure" / "summarized_outputs" / "beta_complexity_summary.csv"
)
P_DIM_DEFAULT = 2545.0
DERIVATIVE_SOURCE = "d_sampling_linear_radius_summary"
A_MEASURE_AXIS = "linear_radius"
FIGURE_INPUT_FILES = (
    FIGURE_INPUT_ROOT / "phi_d_curve" / "phi_d_curve.csv",
    FIGURE_INPUT_ROOT / "phi_energetic_d_curve" / "phi_energetic_d_curve.csv",
    FIGURE_INPUT_ROOT / "derivative_phi_d_curve" / "derivative_phi_d_curve.csv",
    FIGURE_INPUT_ROOT / "derivative_phi_energetic_d_curve" / "derivative_phi_energetic_d_curve.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_beta" / "phase_like_A_by_beta.csv",
    FIGURE_INPUT_ROOT / "phase_like_A_by_beta" / "phase_derivative_curves.csv",
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


def _format_optional(value: float) -> str:
    return f"{value:.12g}" if np.isfinite(value) else ""


def _beta_case_id(beta: float) -> str:
    return f"beta_{beta:.8g}".replace(".", "p")


def _find_source_summary_root(source_summary_root: Path | None) -> Path:
    if source_summary_root is None:
        raise FileNotFoundError(
            "source summary tables are not retained under an automatic default path; pass --source-summary-root "
            "or keep the retained figure_inputs already present in this stage."
        )

    candidates = [source_summary_root]

    for candidate in candidates:
        if (
            (candidate / "absolute_phi_by_beta_radius.csv").exists()
            and (candidate / "dphi_dr_by_beta_radius.csv").exists()
        ):
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find source summary tables. searched:\n{searched}")


def _retained_figure_inputs_complete() -> bool:
    return all(path.exists() for path in FIGURE_INPUT_FILES)


def _complexity_by_beta() -> dict[float, float]:
    rows = _read_csv(COMPLEXITY_SUMMARY)
    return {round(float(row["beta"]), 8): float(row["complexity_mean"]) for row in rows}


def _load_phi_se(source_summary_root: Path) -> dict[tuple[float, float], dict[str, str]]:
    path = source_summary_root / "phi_standard_error_by_beta_radius.csv"
    if not path.exists():
        return {}
    out: dict[tuple[float, float], dict[str, str]] = {}
    for row in _read_csv(path):
        beta = round(_float(row.get("beta")), 8)
        radius = round(_float(row.get("radius")), 8)
        if np.isfinite(beta) and np.isfinite(radius):
            out[(beta, radius)] = row
    return out


def _infer_derivative_se(row: dict[str, str], key: str) -> str:
    existing = row.get(key)
    if existing not in (None, ""):
        return str(existing)

    sd_dlogz = _float(row.get("sd_dlogZ_inf_full_dr"))
    n_ref = _float(row.get("derivative_ref_count", row.get("ref_count")))
    if np.isfinite(sd_dlogz) and np.isfinite(n_ref) and n_ref > 0:
        return f"{sd_dlogz / P_DIM_DEFAULT / math.sqrt(n_ref):.12g}"
    return ""


def _with_derivative_se(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        clean = dict(row)
        for key in ("dphi_full_dr_se", "dphi_energy_dr_se", "dphi_entropic_dr_se"):
            clean[key] = _infer_derivative_se(clean, key)
        out.append(clean)
    return out


def _phi_rows(rows: list[dict[str, str]], phi_se: dict[tuple[float, float], dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        beta = _float(row.get("beta"))
        radius = _float(row.get("radius"))
        key = (round(beta, 8), round(radius, 8))
        se_row = phi_se.get(key, {})
        out.append(
            {
                "source": "synthetic",
                "case_id": _beta_case_id(beta),
                "case_label": f"beta={beta:.2f}",
                "beta": beta,
                "radius": radius,
                "n_units": row.get("ref_count", ""),
                "phi_full_mean": row.get("phi_full", ""),
                "phi_full_sem": se_row.get("phi_full_se", row.get("phi_full_se", "")),
                "phi_energy_mean": row.get("phi_energy", ""),
                "phi_energy_sem": se_row.get("phi_energy_se", row.get("phi_energy_se", "")),
                "standard_error_source": "logZ_inf_full" if se_row else "",
                "x_axis_scale": "linear",
            }
        )
    return out


def _dphi_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        beta = _float(row.get("beta"))
        out.append(
            {
                "source": "synthetic",
                "case_id": _beta_case_id(beta),
                "case_label": f"beta={beta:.2f}",
                "beta": beta,
                "radius": row.get("radius", ""),
                "n_units": row.get("derivative_ref_count", row.get("ref_count", "")),
                "dphi_full_dr_mean": row.get("dphi_full_dr", ""),
                "dphi_full_dr_sem": row.get("dphi_full_dr_se", ""),
                "dphi_energy_dr_mean": row.get("dphi_energy_dr", ""),
                "dphi_energy_dr_sem": row.get("dphi_energy_dr_se", ""),
                "dphi_entropic_dr_mean": row.get("dphi_entropic_dr", ""),
                "dphi_entropic_dr_sem": row.get("dphi_entropic_dr_se", ""),
                "mean_dlogZ_inf_full_dr": row.get("mean_dlogZ_inf_full_dr", ""),
                "sd_dlogZ_inf_full_dr": row.get("sd_dlogZ_inf_full_dr", ""),
                "standard_error_source": "sd_dlogZ_inf_full_dr / P / sqrt(n_units)",
                "derivative_source": DERIVATIVE_SOURCE,
                "x_axis_scale": "linear",
            }
        )
    return out


def _group_energy_derivative(
    rows: Iterable[dict[str, str]],
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    grouped: dict[float, list[tuple[float, float, float, str]]] = {}
    for row in rows:
        beta = round(_float(row.get("beta")), 8)
        radius = _float(row.get("radius"))
        value = _float(row.get("dphi_energy_dr"))
        sem = _float(row.get("dphi_energy_dr_se"))
        n_units = row.get("derivative_ref_count", row.get("ref_count", ""))
        if np.isfinite(beta) and np.isfinite(radius) and np.isfinite(value):
            grouped.setdefault(beta, []).append((radius, value, sem, n_units))

    out: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for beta, values in grouped.items():
        ordered = sorted(values)
        out[beta] = (
            np.asarray([radius for radius, _value, _sem, _n_units in ordered], dtype=np.float64),
            np.asarray([value for _radius, value, _sem, _n_units in ordered], dtype=np.float64),
            np.asarray([sem for _radius, _value, sem, _n_units in ordered], dtype=np.float64),
            [str(n_units) for _radius, _value, _sem, n_units in ordered],
        )
    return dict(sorted(out.items()))


def _phase_rows(
    dphi_source_rows: list[dict[str, str]],
    complexity_lookup: dict[float, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    phase_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for beta, (radius, value, sem, n_units) in _group_energy_derivative(dphi_source_rows).items():
        mask = np.isfinite(radius) & np.isfinite(value)
        radius = radius[mask]
        value = value[mask]
        sem = sem[mask]
        n_units = [unit for unit, keep in zip(n_units, mask) if keep]
        if radius.size < 3:
            continue

        slope = np.gradient(value, radius)
        total_variation = float(np.trapz(np.abs(slope), radius))
        signed_variation = float(np.trapz(slope, radius))
        complexity = complexity_lookup.get(round(beta, 8), float("nan"))
        case_id = _beta_case_id(beta)
        case_label = f"beta={beta:.2f}"

        phase_rows.append(
            {
                "source": "synthetic",
                "case_id": case_id,
                "case_label": case_label,
                "beta": beta,
                "complexity_mean": _format_optional(complexity),
                "radius_count": int(radius.size),
                "A_transition_total_variation_mean": f"{total_variation:.12g}",
                "A_transition_total_variation_sem": "",
                "A_transition_signed_variation": f"{signed_variation:.12g}",
                "A_measure_axis": A_MEASURE_AXIS,
                "derivative_source": DERIVATIVE_SOURCE,
            }
        )

        for x, y, err, unit_count in zip(radius, value, sem, n_units):
            curve_rows.append(
                {
                    "source": "synthetic",
                    "case_id": case_id,
                    "case_label": case_label,
                    "beta": beta,
                    "complexity_mean": _format_optional(complexity),
                    "radius": x,
                    "n_units": unit_count,
                    "dphi_dr_smooth_mean": y,
                    "dphi_dr_smooth_sem": _format_optional(err),
                    "standard_error_source": "sd_dlogZ_inf_full_dr / P / sqrt(n_units)",
                    "derivative_source": DERIVATIVE_SOURCE,
                    "x_axis_scale": "linear",
                }
            )

    return phase_rows, curve_rows


def build(source_summary_root: Path | None = None) -> None:
    try:
        source_root = _find_source_summary_root(source_summary_root)
    except FileNotFoundError:
        if source_summary_root is None and _retained_figure_inputs_complete():
            return
        raise
    phi_source = _read_csv(source_root / "absolute_phi_by_beta_radius.csv")
    dphi_source = _with_derivative_se(_read_csv(source_root / "dphi_dr_by_beta_radius.csv"))
    phi_se = _load_phi_se(source_root)
    complexity_lookup = _complexity_by_beta()

    FIGURE_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    phi_inputs = _phi_rows(phi_source, phi_se)
    dphi_inputs = _dphi_rows(dphi_source)
    phase_inputs, phase_curve_inputs = _phase_rows(dphi_source, complexity_lookup)

    phi_fields = [
        "source",
        "case_id",
        "case_label",
        "beta",
        "radius",
        "n_units",
        "phi_full_mean",
        "phi_full_sem",
        "phi_energy_mean",
        "phi_energy_sem",
        "standard_error_source",
        "x_axis_scale",
    ]
    dphi_fields = [
        "source",
        "case_id",
        "case_label",
        "beta",
        "radius",
        "n_units",
        "dphi_full_dr_mean",
        "dphi_full_dr_sem",
        "dphi_energy_dr_mean",
        "dphi_energy_dr_sem",
        "dphi_entropic_dr_mean",
        "dphi_entropic_dr_sem",
        "mean_dlogZ_inf_full_dr",
        "sd_dlogZ_inf_full_dr",
        "standard_error_source",
        "derivative_source",
        "x_axis_scale",
    ]
    phase_fields = [
        "source",
        "case_id",
        "case_label",
        "beta",
        "complexity_mean",
        "radius_count",
        "A_transition_total_variation_mean",
        "A_transition_total_variation_sem",
        "A_transition_signed_variation",
        "A_measure_axis",
        "derivative_source",
    ]
    phase_curve_fields = [
        "source",
        "case_id",
        "case_label",
        "beta",
        "complexity_mean",
        "radius",
        "n_units",
        "dphi_dr_smooth_mean",
        "dphi_dr_smooth_sem",
        "standard_error_source",
        "derivative_source",
        "x_axis_scale",
    ]

    _write_csv(FIGURE_INPUT_ROOT / "phi_d_curve" / "phi_d_curve.csv", phi_inputs, phi_fields)
    _write_csv(
        FIGURE_INPUT_ROOT / "phi_energetic_d_curve" / "phi_energetic_d_curve.csv",
        phi_inputs,
        phi_fields,
    )
    _write_csv(
        FIGURE_INPUT_ROOT / "derivative_phi_d_curve" / "derivative_phi_d_curve.csv",
        dphi_inputs,
        dphi_fields,
    )
    _write_csv(
        FIGURE_INPUT_ROOT / "derivative_phi_energetic_d_curve" / "derivative_phi_energetic_d_curve.csv",
        dphi_inputs,
        dphi_fields,
    )
    for name in ("phase_like_A_by_beta", "phase_like_A_by_complexity"):
        _write_csv(FIGURE_INPUT_ROOT / name / f"{name}.csv", phase_inputs, phase_fields)
        _write_csv(
            FIGURE_INPUT_ROOT / name / "phase_derivative_curves.csv",
            phase_curve_inputs,
            phase_curve_fields,
        )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build figure-specific summarized_outputs for DNN synthetic proxy local entropy."
    )
    parser.add_argument(
        "--source-summary-root",
        type=Path,
        default=None,
        help="Source summary_tables directory. Defaults to retained in-repo figure inputs when omitted.",
    )
    args = parser.parse_args()
    build(args.source_summary_root)


if __name__ == "__main__":
    main()
