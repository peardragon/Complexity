from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def load_comparison_rows(analytic_csv: Path, sampling_csv: Path) -> list[dict[str, str]]:
    analytic_rows = read_csv(analytic_csv)
    sampling_rows = read_csv(sampling_csv)
    radii = [as_float(row, "r") for row in analytic_rows if math.isfinite(as_float(row, "r"))]
    phi = [as_float(row, "phi_rel") for row in analytic_rows if math.isfinite(as_float(row, "r"))]
    if not radii or not phi:
        raise ValueError(f"analytic CSV lacks finite r/phi_rel rows: {analytic_csv}")
    order = np.argsort(np.asarray(radii, dtype=np.float64))
    r_sorted = np.asarray(radii, dtype=np.float64)[order]
    phi_sorted = np.asarray(phi, dtype=np.float64)[order]
    rows: list[dict[str, str]] = []
    for row in sampling_rows:
        r = as_float(row, "r")
        out = dict(row)
        if math.isfinite(r):
            out["phi_theory"] = f"{float(np.interp(r, r_sorted, phi_sorted)):.17g}"
            out["finiteN_error"] = f"{as_float(row, 'phi_emp') - float(out['phi_theory']):.17g}"
        else:
            out["phi_theory"] = "nan"
            out["finiteN_error"] = "nan"
        rows.append(out)
    return rows


def peak_radius(r: list[float], phi: list[float]) -> float:
    pairs = [(rr, pp) for rr, pp in zip(r, phi) if math.isfinite(rr) and math.isfinite(pp)]
    if not pairs:
        return float("nan")
    return float(max(pairs, key=lambda item: item[1])[0])


def finite_n_error_rows(comparison_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    n_values = sorted({as_int(row, "N") for row in comparison_rows})
    for n in n_values:
        rows = [row for row in comparison_rows if as_int(row, "N") == n]
        errors = [as_float(row, "finiteN_error") for row in rows if math.isfinite(as_float(row, "finiteN_error"))]
        if not errors:
            continue
        radii = [as_float(row, "r") for row in rows]
        emp = [as_float(row, "phi_emp") for row in rows]
        theory = [as_float(row, "phi_theory") for row in rows]
        emp_peak = peak_radius(radii, emp)
        theory_peak = peak_radius(radii, theory)
        peak_diff = abs(emp_peak - theory_peak) if math.isfinite(emp_peak) and math.isfinite(theory_peak) else float("nan")
        out.append(
            {
                "N": str(n),
                "inv_N": f"{1.0 / n:.17g}" if n else "nan",
                "rmse_to_theory": f"{float(np.sqrt(np.mean(np.square(errors)))):.17g}",
                "max_abs_error_to_theory": f"{float(np.max(np.abs(errors))):.17g}",
                "peak_radius_emp": f"{emp_peak:.17g}",
                "peak_radius_theory": f"{theory_peak:.17g}",
                "peak_radius_abs_diff": f"{peak_diff:.17g}",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def goal_status(error_rows: list[dict[str, str]], qc_rows: list[dict[str, str]]) -> dict[str, object]:
    by_n = {as_int(row, "N"): row for row in error_rows}
    n_values = sorted(by_n)
    rmse = {n: as_float(by_n[n], "rmse_to_theory") for n in n_values}
    largest_n = n_values[-1] if n_values else 0
    smallest_n = n_values[0] if n_values else 0
    rmse_improves = bool(n_values and rmse[largest_n] < rmse[smallest_n])
    rmse_monotone = all(rmse[n_values[i + 1]] <= rmse[n_values[i]] for i in range(len(n_values) - 1))
    peak_diff = as_float(by_n.get(largest_n, {}), "peak_radius_abs_diff")
    radii = sorted({as_float(row, "r") for row in qc_rows if math.isfinite(as_float(row, "r"))})
    grid_step = float(np.min(np.diff(radii))) if len(radii) >= 2 else float("nan")
    peak_ok = bool(math.isfinite(peak_diff) and (not math.isfinite(grid_step) or peak_diff <= grid_step + 1.0e-12))
    no_claim = [row for row in qc_rows if str(row.get("claim", "")).lower() != "pass"]
    split_fail = [row for row in qc_rows if str(row.get("split_pass", "")).lower() != "true"]
    smc_fail = [row for row in qc_rows if str(row.get("smc_pass", "")).lower() != "true"]
    return {
        "largest_n": largest_n,
        "smallest_n": smallest_n,
        "rmse_largest": rmse.get(largest_n, float("nan")),
        "rmse_smallest": rmse.get(smallest_n, float("nan")),
        "rmse_improves": rmse_improves,
        "rmse_monotone": rmse_monotone,
        "peak_radius_abs_diff_largest": peak_diff,
        "radius_grid_step": grid_step,
        "peak_ok": peak_ok,
        "qc_cell_count": len(qc_rows),
        "no_claim_cell_count": len(no_claim),
        "split_fail_cell_count": len(split_fail),
        "smc_fail_cell_count": len(smc_fail),
        "full_qc_pass": bool(not no_claim and not smc_fail),
        "goal_supported": bool(rmse_improves and peak_ok and not smc_fail),
    }


def write_goal_figure(summary_root: Path, comparison_rows: list[dict[str, str]], error_rows: list[dict[str, str]], qc_rows: list[dict[str, str]], status: dict[str, object], figure_dir: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = figure_dir if figure_dir is not None else summary_root / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    n_values = sorted({as_int(row, "N") for row in comparison_rows})
    radii = sorted({as_float(row, "r") for row in comparison_rows if math.isfinite(as_float(row, "r"))})
    theory = {}
    for row in comparison_rows:
        r = as_float(row, "r")
        phi = as_float(row, "phi_theory")
        if math.isfinite(r) and math.isfinite(phi):
            theory[r] = phi

    fig = plt.figure(figsize=(15.0, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
    ax_curve = fig.add_subplot(gs[:, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_qc = fig.add_subplot(gs[1, 1])

    if radii:
        ax_curve.plot(radii, [theory.get(r, float("nan")) for r in radii], color="black", linewidth=3.0, label="full-RS theory")
    markers = ["s", "^", "D", "v", "o"]
    for idx, n in enumerate(n_values):
        rows = sorted([row for row in comparison_rows if as_int(row, "N") == n], key=lambda row: as_float(row, "r"))
        ax_curve.plot(
            [as_float(row, "r") for row in rows],
            [as_float(row, "phi_emp") for row in rows],
            "--",
            marker=markers[idx % len(markers)],
            markersize=4,
            linewidth=1.8,
            label=f"N={n}",
        )
    title_status = "supported" if bool(status["goal_supported"]) else "partial"
    ax_curve.set_title(f"Dense QC two-pool validation: N convergence {title_status}")
    ax_curve.set_xlabel("r")
    ax_curve.set_ylabel("Phi(r) - Phi(r0)")
    ax_curve.grid(True, alpha=0.30)
    ax_curve.legend(fontsize=9)

    rows_err = sorted(error_rows, key=lambda row: as_int(row, "N"))
    x = [as_float(row, "inv_N") for row in rows_err]
    rmse = [as_float(row, "rmse_to_theory") for row in rows_err]
    maxerr = [as_float(row, "max_abs_error_to_theory") for row in rows_err]
    ax_rmse.plot(x, rmse, "-o", label="RMSE")
    ax_rmse.plot(x, maxerr, "--s", label="max abs error")
    ax_rmse.invert_xaxis()
    ax_rmse.set_xlabel("1 / N")
    ax_rmse.set_ylabel("error to theory")
    ax_rmse.set_title("Finite-N error")
    ax_rmse.grid(True, alpha=0.30)
    ax_rmse.legend(fontsize=9)

    for idx, n in enumerate(n_values):
        rows = sorted([row for row in qc_rows if as_int(row, "N") == n], key=lambda row: as_float(row, "r"))
        ax_qc.plot(
            [as_float(row, "r") for row in rows],
            [as_float(row, "max_split_logZ_per_N_diff") for row in rows],
            marker=markers[idx % len(markers)],
            linewidth=1.4,
            markersize=3.5,
            label=f"N={n}",
        )
    ax_qc.axhline(0.004, color="black", linestyle=":", linewidth=1.2, label="split threshold")
    ax_qc.set_xlabel("r")
    ax_qc.set_ylabel("max split |Delta logZ| / N")
    ax_qc.set_title("QC by radius")
    ax_qc.grid(True, alpha=0.30)
    ax_qc.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out = figs / "fig00_dense_qc_N_convergence_alpha0p1.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def write_report(summary_root: Path, status: dict[str, object], figure_path: Path) -> Path:
    lines = [
        "# Dense QC Goal Status",
        "",
        f"- Figure: `{figure_path.as_posix()}`",
        f"- Goal supported: `{status['goal_supported']}`",
        f"- Full QC pass: `{status['full_qc_pass']}`",
        f"- RMSE improves largest vs smallest N: `{status['rmse_improves']}`",
        f"- RMSE monotone over all retained N: `{status['rmse_monotone']}`",
        f"- Smallest-N RMSE: `{float(status['rmse_smallest']):.6g}`",
        f"- Largest-N RMSE: `{float(status['rmse_largest']):.6g}`",
        f"- Largest-N peak radius diff: `{float(status['peak_radius_abs_diff_largest']):.6g}`",
        f"- Radius grid step: `{float(status['radius_grid_step']):.6g}`",
        f"- QC cells: `{status['qc_cell_count']}`",
        f"- No-claim QC cells: `{status['no_claim_cell_count']}`",
        f"- Split-fail QC cells: `{status['split_fail_cell_count']}`",
        f"- SMC-fail QC cells: `{status['smc_fail_cell_count']}`",
    ]
    out = summary_root / "goal_status_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write dense QC goal figure and status report.")
    parser.add_argument("summary_root", type=Path)
    parser.add_argument("--analytic-csv", type=Path, default=Path("01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv"))
    parser.add_argument("--sampling-csv", type=Path, default=Path("01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_phi_by_N_alpha0p1.csv"))
    parser.add_argument("--qc-csv", type=Path, default=Path("01_theory/02_theory_sampling/raw_outputs/shell_pool/sampling_qc_by_N_radius.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("01_theory/03_theory_comparison/figures"))
    args = parser.parse_args()
    summary_root = args.summary_root.resolve()
    comparison_rows = load_comparison_rows(args.analytic_csv.resolve(), args.sampling_csv.resolve())
    comparison_fieldnames = [
        "r",
        "N",
        "phi_theory",
        "phi_emp",
        "finiteN_error",
        "weighted_CE",
        "weighted_err",
        "reference_count",
        "mean_ess_frac",
        "q05_ess_frac",
        "max_split_logZ_per_N_diff",
        "fallback_unit_count",
        "fallback_unit_fraction",
        "min_smc_cess_fraction",
        "max_smc_step_count",
    ]
    write_csv(summary_root / "comparison_phi_by_N_alpha0p1.csv", comparison_rows, comparison_fieldnames)
    error_rows = finite_n_error_rows(comparison_rows)
    write_csv(summary_root / "finiteN_error_summary.csv", error_rows)
    qc_rows = read_csv(args.qc_csv.resolve())
    write_csv(summary_root / "comparison_qc_summary.csv", qc_rows)
    status = goal_status(error_rows, qc_rows)
    figure = write_goal_figure(summary_root, comparison_rows, error_rows, qc_rows, status, args.figure_dir.resolve())
    report = write_report(summary_root, status, figure)
    (summary_root / "goal_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(figure)
    print(report)


if __name__ == "__main__":
    main()
