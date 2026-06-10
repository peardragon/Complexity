from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEASIBLE_CSV = Path("01_theory/01_theory_analytic/raw_outputs/theory_full_feasible_rs_alpha0p1.csv")
DEFAULT_BASELINE_CSV = Path("01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv")
DEFAULT_SAMPLING_CSV = Path("01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv")
DEFAULT_OUT_DIR = Path("01_theory/03_theory_comparison/raw_outputs/full_feasible_rs_alpha0p1")


def finite_float(x: object, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if np.isfinite(y) else default


def interp_branch(branch_df: pd.DataFrame, r: np.ndarray) -> np.ndarray:
    sub = branch_df.sort_values("r")
    return np.interp(
        r.astype(float),
        sub["r"].to_numpy(dtype=float),
        sub["phi_rel"].to_numpy(dtype=float),
    )


def peak_radius(r: np.ndarray, phi: np.ndarray) -> float:
    mask = np.isfinite(r) & np.isfinite(phi)
    if not np.any(mask):
        return float("nan")
    rr = r[mask]
    pp = phi[mask]
    return float(rr[int(np.nanargmax(pp))])


def baseline_reproduction(theory: pd.DataFrame, baseline_csv: Path) -> dict[str, Any]:
    if not baseline_csv.exists():
        return {
            "available": False,
            "baseline_csv": baseline_csv.as_posix(),
            "pass_2e_minus_3": False,
        }

    baseline = pd.read_csv(baseline_csv)
    boundary = theory[theory["branch"] == "boundary_mixed_eta0"].copy()
    if baseline.empty or boundary.empty:
        return {
            "available": True,
            "baseline_csv": baseline_csv.as_posix(),
            "pass_2e_minus_3": False,
            "reason": "missing baseline or boundary rows",
        }

    merged = baseline[["r", "phi_rel"]].merge(
        boundary[["r", "phi_rel"]],
        on="r",
        how="inner",
        suffixes=("_baseline", "_boundary"),
    )
    if merged.empty:
        return {
            "available": True,
            "baseline_csv": baseline_csv.as_posix(),
            "pass_2e_minus_3": False,
            "reason": "no shared radii",
        }

    diff = merged["phi_rel_boundary"].astype(float) - merged["phi_rel_baseline"].astype(float)
    max_abs = float(np.nanmax(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff * diff)))
    return {
        "available": True,
        "baseline_csv": baseline_csv.as_posix(),
        "shared_radius_count": int(len(merged)),
        "max_abs_phi_rel_diff": max_abs,
        "rmse_phi_rel_diff": rmse,
        "pass_1e_minus_3": bool(max_abs < 1.0e-3),
        "pass_2e_minus_3": bool(max_abs < 2.0e-3),
        "pass_3e_minus_3": bool(max_abs < 3.0e-3),
    }


def branch_diagnostics(theory: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch, bdf in theory.groupby("branch"):
        A = bdf["A"].to_numpy(dtype=float)
        eta = bdf["eta"].to_numpy(dtype=float)
        rows.append(
            {
                "branch": str(branch),
                "radius_count": int(len(bdf)),
                "max_A": float(np.nanmax(A)),
                "mean_A": float(np.nanmean(A)),
                "max_eta": float(np.nanmax(eta)),
                "mean_eta": float(np.nanmean(eta)),
                "interior_radius_count_A_gt_1e_minus_4": int(np.sum(A > 1.0e-4)),
                "interior_radius_count_eta_gt_1e_minus_3": int(np.sum(eta > 1.0e-3)),
            }
        )
    return rows


def interpret_status(err_df: pd.DataFrame, diag_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "case": "insufficient_data",
        "message": "No finite comparison rows were available.",
    }
    if err_df.empty:
        return out

    largest_N = int(err_df["N"].max())
    largest = err_df[err_df["N"] == largest_N].copy().sort_values("rmse")
    if largest.empty:
        return out

    best = largest.iloc[0].to_dict()
    diag_by_branch = {str(row["branch"]): row for row in diag_rows}
    full_diag = diag_by_branch.get("full_mixed_maxQ_min_s_eta", {})
    full_max_A = finite_float(full_diag.get("max_A"))
    full_max_eta = finite_float(full_diag.get("max_eta"))
    full_collapses = bool(
        np.isfinite(full_max_A)
        and np.isfinite(full_max_eta)
        and full_max_A <= 1.0e-4
        and full_max_eta <= 1.0e-3
    )

    if full_collapses:
        case = "full_mixed_collapses_to_boundary"
        message = "The full mixed saddle selected eta approximately zero across the tested radii."
    elif best["branch"] == "full_mixed_maxQ_min_s_eta":
        case = "full_mixed_best"
        message = "The full mixed branch has the smallest largest-N RMSE among tested branches."
    elif best["branch"] == "boundary_mixed_eta0":
        case = "boundary_best"
        message = "The A=0 boundary branch has the smallest largest-N RMSE among tested branches."
    else:
        case = "diagnostic_branch_best"
        message = "The diagnostic max envelope has the smallest largest-N RMSE; do not treat this as physical without further saddle analysis."

    return {
        "case": case,
        "message": message,
        "largest_N": largest_N,
        "best_branch": str(best["branch"]),
        "best_rmse": float(best["rmse"]),
        "best_peak_radius_abs_diff": float(best["peak_radius_abs_diff"]),
        "full_mixed_max_A": full_max_A,
        "full_mixed_max_eta": full_max_eta,
    }


def compare(
    *,
    feasible_csv: Path,
    sampling_csv: Path,
    out_dir: Path,
    baseline_csv: Path = DEFAULT_BASELINE_CSV,
) -> dict[str, Any]:
    theory = pd.read_csv(feasible_csv)
    sampling = pd.read_csv(sampling_csv)

    out_dir.mkdir(parents=True, exist_ok=True)

    branches = sorted(str(x) for x in theory["branch"].dropna().unique())
    comparison_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for N, sdf in sampling.groupby("N"):
        sdf = sdf.copy()
        sdf["r"] = sdf["r"].astype(float)
        sdf["phi_emp"] = sdf["phi_emp"].astype(float)
        r = sdf["r"].to_numpy(dtype=float)
        emp = sdf["phi_emp"].to_numpy(dtype=float)
        emp_peak = peak_radius(r, emp)

        for branch in branches:
            bdf = theory[theory["branch"] == branch]
            pred = interp_branch(bdf, r)
            err = emp - pred
            rmse = float(np.sqrt(np.nanmean(err * err)))
            max_abs = float(np.nanmax(np.abs(err)))
            pred_peak = peak_radius(r, pred)
            peak_diff = abs(emp_peak - pred_peak) if np.isfinite(emp_peak) and np.isfinite(pred_peak) else float("nan")

            error_rows.append(
                {
                    "N": int(N),
                    "branch": branch,
                    "rmse": rmse,
                    "max_abs_error": max_abs,
                    "emp_peak_radius": emp_peak,
                    "theory_peak_radius": pred_peak,
                    "peak_radius_abs_diff": peak_diff,
                }
            )

            for _, row in sdf.iterrows():
                rr = finite_float(row["r"])
                pp = finite_float(row["phi_emp"])
                th = float(np.interp(rr, bdf["r"].to_numpy(dtype=float), bdf["phi_rel"].to_numpy(dtype=float)))
                comparison_rows.append(
                    {
                        "N": int(N),
                        "r": rr,
                        "branch": branch,
                        "phi_emp": pp,
                        "phi_theory": th,
                        "finiteN_error": pp - th,
                    }
                )

    cmp_df = pd.DataFrame(comparison_rows)
    err_df = pd.DataFrame(error_rows)
    diag_rows = branch_diagnostics(theory)
    diag_df = pd.DataFrame(diag_rows)

    cmp_df.to_csv(out_dir / "comparison_phi_full_feasible_by_N_alpha0p1.csv", index=False)
    err_df.to_csv(out_dir / "finiteN_error_full_feasible_summary.csv", index=False)
    diag_df.to_csv(out_dir / "branch_A_eta_diagnostics.csv", index=False)

    status: dict[str, Any] = {
        "feasible_csv": feasible_csv.as_posix(),
        "sampling_csv": sampling_csv.as_posix(),
        "branches": branches,
        "baseline_reproduction": baseline_reproduction(theory, baseline_csv),
        "best_by_largest_N": {},
        "A_eta_diagnostics": diag_rows,
    }
    if not err_df.empty:
        largest_N = int(err_df["N"].max())
        sub = err_df[err_df["N"] == largest_N].sort_values("rmse")
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            status["best_by_largest_N"] = {
                "N": largest_N,
                "branch": row["branch"],
                "rmse": float(row["rmse"]),
                "peak_radius_abs_diff": float(row["peak_radius_abs_diff"]),
            }
    status["interpretation"] = interpret_status(err_df, diag_rows)

    (out_dir / "full_feasible_goal_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full feasible RS branches against PM-SAIS sampling.")
    parser.add_argument("--feasible-csv", type=Path, default=DEFAULT_FEASIBLE_CSV)
    parser.add_argument("--sampling-csv", type=Path, default=DEFAULT_SAMPLING_CSV)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    status = compare(
        feasible_csv=args.feasible_csv,
        sampling_csv=args.sampling_csv,
        baseline_csv=args.baseline_csv,
        out_dir=args.out_dir,
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
