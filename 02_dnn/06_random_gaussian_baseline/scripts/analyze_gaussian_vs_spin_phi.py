from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
BASE_ROOT = SCRIPT_PATH.parents[1]
DNN_ROOT = SCRIPT_PATH.parents[2]

RUN_REFERENCE = "gaussian_random_90_dataset_30_reference"
RANGE_NAME = "d_0.01_to_2.50_dense"

GAUSSIAN_CURVE = (
    BASE_ROOT
    / "raw_outputs"
    / "05_proxy_local_entropy"
    / RUN_REFERENCE
    / RANGE_NAME
    / "summary_tables"
    / "high_beta_curve_comparison.csv"
)
SPIN_DETAIL = (
    DNN_ROOT
    / "05_proxy_local_entropy"
    / "figures"
    / "high_beta_energy_derivatives_ci_30_60_90"
    / "energy_phi_d1_d2_ci_detail.csv"
)
SPIN_ALL_BETA_ABSOLUTE = (
    DNN_ROOT
    / "05_proxy_local_entropy"
    / "raw_outputs"
    / "18_beta_cell_90_dataset_30_reference"
    / RANGE_NAME
    / "summary_tables"
    / "absolute_phi_by_beta_radius.csv"
)
COMPLEXITY_NEAREST = (
    BASE_ROOT
    / "raw_outputs"
    / "02_complexity_measure"
    / RUN_REFERENCE
    / "summary_tables"
    / "nearest_spin_beta_to_gaussian_complexity.csv"
)
ANALYSIS_ROOT = BASE_ROOT / "analysis"


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def group_gaussian_by_radius(rows: list[dict[str, str]], max_radius: float) -> dict[float, dict[str, float]]:
    grouped: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        radius = round(finite_float(row.get("radius")), 8)
        if not math.isfinite(radius) or radius > max_radius:
            continue
        grouped.setdefault(radius, []).append(
            {
                "mean": finite_float(row.get("phi_energy_mean")),
                "ci95_low": finite_float(row.get("phi_energy_ci95_low")),
                "ci95_high": finite_float(row.get("phi_energy_ci95_high")),
            }
        )
    out: dict[float, dict[str, float]] = {}
    for radius, values in grouped.items():
        clean_mean = [row["mean"] for row in values if math.isfinite(row["mean"])]
        clean_low = [row["ci95_low"] for row in values if math.isfinite(row["ci95_low"])]
        clean_high = [row["ci95_high"] for row in values if math.isfinite(row["ci95_high"])]
        if clean_mean:
            out[radius] = {
                "mean": mean(clean_mean),
                "ci95_low": mean(clean_low) if clean_low else float("nan"),
                "ci95_high": mean(clean_high) if clean_high else float("nan"),
            }
    return out


def group_spin_by_beta_radius(rows: list[dict[str, str]], dataset_count: int, max_radius: float) -> dict[float, dict[float, dict[str, float]]]:
    grouped: dict[float, dict[float, dict[str, float]]] = {}
    for row in rows:
        if row.get("metric") != "phi_energy":
            continue
        if int(finite_float(row.get("dataset_count"), -1)) != dataset_count:
            continue
        radius = round(finite_float(row.get("radius")), 8)
        beta = round(finite_float(row.get("beta")), 8)
        if not math.isfinite(radius) or not math.isfinite(beta) or radius > max_radius:
            continue
        grouped.setdefault(beta, {})[radius] = {
            "mean": finite_float(row.get("mean")),
            "ci95_low": finite_float(row.get("ci95_low")),
            "ci95_high": finite_float(row.get("ci95_high")),
        }
    return grouped


def group_spin_all_beta_absolute(rows: list[dict[str, str]], max_radius: float) -> dict[float, dict[float, dict[str, float]]]:
    grouped: dict[float, dict[float, dict[str, float]]] = {}
    for row in rows:
        radius = round(finite_float(row.get("radius")), 8)
        beta = round(finite_float(row.get("beta")), 8)
        if not math.isfinite(radius) or not math.isfinite(beta) or radius > max_radius:
            continue
        value = finite_float(row.get("phi_energy"))
        if not math.isfinite(value):
            continue
        grouped.setdefault(beta, {})[radius] = {
            "mean": value,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    return grouped


def ci_overlaps(a: dict[str, float], b: dict[str, float]) -> bool:
    lo_a, hi_a = a.get("ci95_low", float("nan")), a.get("ci95_high", float("nan"))
    lo_b, hi_b = b.get("ci95_low", float("nan")), b.get("ci95_high", float("nan"))
    if not all(math.isfinite(x) for x in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def compare_curves(
    gaussian: dict[float, dict[str, float]],
    spin: dict[float, dict[float, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gaussian_radii = set(gaussian)
    for beta, by_radius in sorted(spin.items()):
        common = sorted(gaussian_radii.intersection(by_radius))
        diffs: list[float] = []
        abs_diffs: list[float] = []
        overlap_count = 0
        for radius in common:
            g = gaussian[radius]
            s = by_radius[radius]
            if not math.isfinite(g["mean"]) or not math.isfinite(s["mean"]):
                continue
            diff = g["mean"] - s["mean"]
            diffs.append(diff)
            abs_diffs.append(abs(diff))
            overlap_count += int(ci_overlaps(g, s))
        if not diffs:
            continue
        rows.append(
            {
                "spin_beta": beta,
                "common_radius_count": len(diffs),
                "rmse": math.sqrt(mean([diff * diff for diff in diffs])),
                "mae": mean(abs_diffs),
                "max_abs_diff": max(abs_diffs),
                "mean_signed_diff_gaussian_minus_spin": mean(diffs),
                "ci95_overlap_fraction": overlap_count / len(diffs),
            }
        )
    rows.sort(key=lambda row: (row["rmse"], row["mae"]))
    for rank, row in enumerate(rows, start=1):
        row["phi_distance_rank"] = rank
    return rows


def row_for_beta(rows: list[dict[str, Any]], beta: float | None) -> dict[str, Any] | None:
    if beta is None or not math.isfinite(float(beta)):
        return None
    for row in rows:
        if abs(float(row.get("spin_beta", float("nan"))) - float(beta)) < 1e-9:
            return row
    return None


def rmse_ratio(row: dict[str, Any] | None, best_row: dict[str, Any] | None) -> float | None:
    if row is None or best_row is None:
        return None
    rmse = finite_float(row.get("rmse"))
    best = finite_float(best_row.get("rmse"))
    if not math.isfinite(rmse) or not math.isfinite(best) or best <= 0.0:
        return None
    return rmse / best


def read_complexity_nearest() -> dict[str, Any]:
    if not COMPLEXITY_NEAREST.exists():
        return {
            "available": False,
            "nearest_beta": None,
            "nearest_abs_gap": None,
            "rows": [],
        }
    rows = read_csv(COMPLEXITY_NEAREST)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "beta": finite_float(row.get("beta")),
                "knn_edge_disagreement_mean": finite_float(row.get("knn_edge_disagreement_mean")),
                "abs_gap_to_gaussian_knn_disagreement": finite_float(row.get("abs_gap_to_gaussian_knn_disagreement")),
            }
        )
    parsed = [row for row in parsed if math.isfinite(row["beta"])]
    parsed.sort(key=lambda row: row.get("abs_gap_to_gaussian_knn_disagreement", float("inf")))
    first = parsed[0] if parsed else {}
    return {
        "available": bool(parsed),
        "nearest_beta": first.get("beta"),
        "nearest_abs_gap": first.get("abs_gap_to_gaussian_knn_disagreement"),
        "rows": parsed,
    }


def write_markdown(
    path: Path,
    report: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    all_beta_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gaussian Baseline vs Spin phi(d) Analysis",
        "",
        f"- gaussian curve available: `{report['gaussian_curve_available']}`",
        f"- spin detail: `{report['spin_detail']}`",
        f"- dataset_count: `{report['dataset_count']}`",
        f"- max_radius: `{report['max_radius']}`",
        f"- complexity nearest spin beta: `{report['complexity_nearest_beta']}`",
        f"- complexity nearest beta present in overlay spin detail: `{report['complexity_nearest_beta_present_in_spin_detail']}`",
        f"- complexity nearest beta present in all-beta spin phi table: `{report['complexity_nearest_beta_present_in_all_beta_phi']}`",
        f"- closest available spin beta by phi(d) RMSE: `{report['phi_closest_available_spin_beta']}`",
        f"- closest all-beta spin beta by phi(d) RMSE: `{report['phi_closest_all_beta_spin_beta']}`",
        f"- complexity-nearest all-beta phi rank: `{report.get('complexity_nearest_all_beta_phi_rank')}`",
        f"- complexity-nearest all-beta phi RMSE: `{report.get('complexity_nearest_all_beta_phi_rmse')}`",
        f"- best all-beta phi RMSE: `{report.get('best_all_beta_phi_rmse')}`",
        f"- complexity-nearest RMSE / best RMSE: `{report.get('complexity_nearest_all_beta_rmse_ratio_to_best')}`",
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
    ]
    if comparison_rows:
        fields = [
            "phi_distance_rank",
            "spin_beta",
            "common_radius_count",
            "rmse",
            "mae",
            "max_abs_diff",
            "mean_signed_diff_gaussian_minus_spin",
            "ci95_overlap_fraction",
        ]
        lines.extend(
            [
                "## phi(d) Distance Table",
                "",
                "|" + "|".join(fields) + "|",
                "|" + "|".join(["---:" for _ in fields]) + "|",
            ]
        )
        for row in comparison_rows:
            lines.append("|" + "|".join(str(row.get(field, "")) for field in fields) + "|")
        lines.append("")
    if all_beta_rows:
        fields = [
            "phi_distance_rank",
            "spin_beta",
            "common_radius_count",
            "rmse",
            "mae",
            "max_abs_diff",
            "mean_signed_diff_gaussian_minus_spin",
            "ci95_overlap_fraction",
        ]
        lines.extend(
            [
                "## All-Beta phi(d) Distance Table",
                "",
                "This table uses the raw all-beta `absolute_phi_by_beta_radius.csv` spin output. It does not include CI overlap.",
                "",
                "|" + "|".join(fields) + "|",
                "|" + "|".join(["---:" for _ in fields]) + "|",
            ]
        )
        for row in all_beta_rows:
            lines.append("|" + "|".join(str(row.get(field, "")) for field in fields) + "|")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def read_all_beta_spin(max_radius: float) -> dict[float, dict[float, dict[str, float]]]:
    if not SPIN_ALL_BETA_ABSOLUTE.exists():
        return {}
    return group_spin_all_beta_absolute(read_csv(SPIN_ALL_BETA_ABSOLUTE), max_radius)


def build_report(dataset_count: int, max_radius: float, allow_missing: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    complexity = read_complexity_nearest()
    all_beta_spin = read_all_beta_spin(max_radius)
    all_beta_available_betas = sorted(all_beta_spin)
    complexity_beta = complexity.get("nearest_beta")
    complexity_beta_present_all_beta = any(abs(beta - complexity_beta) < 1e-9 for beta in all_beta_available_betas) if complexity_beta is not None else False
    if not GAUSSIAN_CURVE.exists():
        report = {
            "gaussian_curve_available": False,
            "gaussian_curve": str(GAUSSIAN_CURVE),
            "spin_detail": str(SPIN_DETAIL),
            "spin_all_beta_absolute": str(SPIN_ALL_BETA_ABSOLUTE),
            "dataset_count": dataset_count,
            "max_radius": max_radius,
            "complexity_nearest_beta": complexity_beta,
            "complexity_nearest_abs_gap": complexity.get("nearest_abs_gap"),
            "complexity_nearest_beta_present_in_spin_detail": False,
            "complexity_nearest_beta_present_in_all_beta_phi": complexity_beta_present_all_beta,
            "phi_closest_available_spin_beta": None,
            "phi_closest_all_beta_spin_beta": None,
            "complexity_nearest_available_phi_rank": None,
            "complexity_nearest_available_phi_rmse": None,
            "best_available_phi_rmse": None,
            "complexity_nearest_available_rmse_ratio_to_best": None,
            "complexity_nearest_all_beta_phi_rank": None,
            "complexity_nearest_all_beta_phi_rmse": None,
            "best_all_beta_phi_rmse": None,
            "complexity_nearest_all_beta_rmse_ratio_to_best": None,
            "comparison_rows": 0,
            "all_beta_comparison_rows": 0,
            "interpretation": "Gaussian phi(d) curve is not available yet; wait for sampling and postprocess to finish.",
        }
        if allow_missing:
            return report, [], []
        raise FileNotFoundError(GAUSSIAN_CURVE)
    if not SPIN_DETAIL.exists():
        raise FileNotFoundError(SPIN_DETAIL)

    gaussian = group_gaussian_by_radius(read_csv(GAUSSIAN_CURVE), max_radius)
    spin = group_spin_by_beta_radius(read_csv(SPIN_DETAIL), dataset_count, max_radius)
    comparison = compare_curves(gaussian, spin)
    all_beta_comparison = compare_curves(gaussian, all_beta_spin)
    available_betas = sorted(spin)
    complexity_beta_present = any(abs(beta - complexity_beta) < 1e-9 for beta in available_betas) if complexity_beta is not None else False
    phi_closest_beta = comparison[0]["spin_beta"] if comparison else None
    phi_closest_all_beta = all_beta_comparison[0]["spin_beta"] if all_beta_comparison else None
    complexity_row = row_for_beta(comparison, complexity_beta)
    complexity_all_beta_row = row_for_beta(all_beta_comparison, complexity_beta)
    best_row = comparison[0] if comparison else None
    best_all_beta_row = all_beta_comparison[0] if all_beta_comparison else None
    complexity_rank = complexity_row.get("phi_distance_rank") if complexity_row else None
    complexity_all_beta_rank = complexity_all_beta_row.get("phi_distance_rank") if complexity_all_beta_row else None
    complexity_rmse = complexity_row.get("rmse") if complexity_row else None
    complexity_all_beta_rmse = complexity_all_beta_row.get("rmse") if complexity_all_beta_row else None
    best_rmse = best_row.get("rmse") if best_row else None
    best_all_beta_rmse = best_all_beta_row.get("rmse") if best_all_beta_row else None
    complexity_rmse_ratio = rmse_ratio(complexity_row, best_row)
    complexity_all_beta_rmse_ratio = rmse_ratio(complexity_all_beta_row, best_all_beta_row)

    if complexity_beta_present_all_beta and phi_closest_all_beta is not None and abs(phi_closest_all_beta - complexity_beta) < 1e-9:
        interpretation = (
            "The all-beta spin phi(d) table includes the complexity-nearest spin beta, and that beta is also closest to the "
            "Gaussian baseline by phi(d) RMSE. This supports the complexity proxy as a relevant coordinate for the local "
            "energy landscape, while the high-beta overlay remains a visual random-baseline contrast."
        )
    elif complexity_beta_present_all_beta and phi_closest_all_beta is not None:
        rank_text = f"rank {complexity_all_beta_rank}" if complexity_all_beta_rank is not None else "not ranked"
        ratio_text = (
            f"{complexity_all_beta_rmse_ratio:.3f}x the best RMSE"
            if complexity_all_beta_rmse_ratio is not None
            else "an unavailable RMSE ratio"
        )
        interpretation = (
            "The all-beta spin phi(d) table includes the complexity-nearest spin beta, but the closest phi(d) curve is a "
            f"different beta. The complexity-nearest beta is {rank_text} by phi(d) RMSE, with {ratio_text}. That is evidence "
            "that the scalar roughness complexity is not sufficient on its own for the local energy landscape measured by "
            "phi(d), unless the rank/ratio gap is small enough to be practically negligible."
        )
    elif not complexity_beta_present:
        interpretation = (
            "The current overlay spin detail is a high-beta slice, while the Gaussian roughness complexity is closest to "
            f"spin beta={complexity_beta}. This overlay is therefore a useful random-baseline contrast against the high-beta "
            "energy curves, but it is not by itself the strongest same-complexity-implies-same-phi test. A direct sufficiency "
            "test would require phi(d) curves for the complexity-nearest low-beta spin setting as well."
        )
    elif phi_closest_beta is not None and abs(phi_closest_beta - complexity_beta) < 1e-9:
        interpretation = (
            "The spin beta closest by the complexity proxy is also closest by phi(d) distance. This supports the complexity "
            "proxy as a relevant coordinate for the local energy landscape, though it is supporting evidence rather than a proof."
        )
    else:
        interpretation = (
            "The spin beta closest by the complexity proxy is not closest by phi(d) distance. This suggests the scalar "
            "complexity proxy is not sufficient on its own for the local phi(d) energy landscape."
        )

    report = {
        "gaussian_curve_available": True,
        "gaussian_curve": str(GAUSSIAN_CURVE),
        "spin_detail": str(SPIN_DETAIL),
        "spin_all_beta_absolute": str(SPIN_ALL_BETA_ABSOLUTE),
        "dataset_count": dataset_count,
        "max_radius": max_radius,
        "gaussian_radius_count": len(gaussian),
        "spin_available_betas": available_betas,
        "all_beta_spin_available_betas": all_beta_available_betas,
        "complexity_nearest_beta": complexity_beta,
        "complexity_nearest_abs_gap": complexity.get("nearest_abs_gap"),
        "complexity_nearest_beta_present_in_spin_detail": complexity_beta_present,
        "complexity_nearest_beta_present_in_all_beta_phi": complexity_beta_present_all_beta,
        "phi_closest_available_spin_beta": phi_closest_beta,
        "phi_closest_all_beta_spin_beta": phi_closest_all_beta,
        "complexity_nearest_available_phi_rank": complexity_rank,
        "complexity_nearest_available_phi_rmse": complexity_rmse,
        "best_available_phi_rmse": best_rmse,
        "complexity_nearest_available_rmse_ratio_to_best": complexity_rmse_ratio,
        "complexity_nearest_all_beta_phi_rank": complexity_all_beta_rank,
        "complexity_nearest_all_beta_phi_rmse": complexity_all_beta_rmse,
        "best_all_beta_phi_rmse": best_all_beta_rmse,
        "complexity_nearest_all_beta_rmse_ratio_to_best": complexity_all_beta_rmse_ratio,
        "comparison_rows": len(comparison),
        "all_beta_comparison_rows": len(all_beta_comparison),
        "interpretation": interpretation,
    }
    return report, comparison, all_beta_comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-count", type=int, default=90)
    parser.add_argument("--max-radius", type=float, default=0.30)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    report, comparison, all_beta_comparison = build_report(args.dataset_count, args.max_radius, args.allow_missing)
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = ANALYSIS_ROOT / "gaussian_vs_spin_phi_analysis.json"
    csv_path = ANALYSIS_ROOT / "gaussian_vs_spin_phi_distance_table.csv"
    all_beta_csv_path = ANALYSIS_ROOT / "gaussian_vs_spin_all_beta_phi_distance_table.csv"
    md_path = ANALYSIS_ROOT / "gaussian_vs_spin_phi_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if comparison:
        write_csv(
            csv_path,
            comparison,
            [
                "phi_distance_rank",
                "spin_beta",
                "common_radius_count",
                "rmse",
                "mae",
                "max_abs_diff",
                "mean_signed_diff_gaussian_minus_spin",
                "ci95_overlap_fraction",
            ],
        )
    if all_beta_comparison:
        write_csv(
            all_beta_csv_path,
            all_beta_comparison,
            [
                "phi_distance_rank",
                "spin_beta",
                "common_radius_count",
                "rmse",
                "mae",
                "max_abs_diff",
                "mean_signed_diff_gaussian_minus_spin",
                "ci95_overlap_fraction",
            ],
        )
    write_markdown(md_path, report, comparison, all_beta_comparison)
    print(
        json.dumps(
            {
                "report": str(json_path),
                "markdown": str(md_path),
                "comparison_csv": str(csv_path),
                "all_beta_comparison_csv": str(all_beta_csv_path),
                **report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
