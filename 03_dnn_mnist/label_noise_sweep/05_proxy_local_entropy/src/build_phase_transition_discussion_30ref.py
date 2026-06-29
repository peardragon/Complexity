#!/usr/bin/env python3
"""Build first-pass phase-transition discussion artifacts for the dense 4-eta run.

This script is intentionally analysis-only: it reads existing aggregated CSVs and
does not invoke reference search, sampling, or unit-summary recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
STAGE_ROOT = SCRIPT_PATH.parents[1]
DNN_ROOT = SCRIPT_PATH.parents[3]
MANUAL_ROOT = DNN_ROOT / "manual_rules"

DEFAULT_ETA_RUN = (
    STAGE_ROOT
    / "summarized_outputs"
    / "eta_reference_phi_dense_4eta_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
DEFAULT_RULE_RUN = (
    MANUAL_ROOT
    / "05_proxy_local_entropy"
    / "summarized_outputs"
    / "active_rules_refpool1024_dense30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
DEFAULT_COMPLEXITY_TABLE = (
    STAGE_ROOT.parent
    / "02_complexity_measure"
    / "summarized_outputs"
    / "complexity_axis_spin_mnist_30ref_eta0p02_0p05_0p15_0p25"
    / "mnist_complexity_axis_metrics.csv"
)
DEFAULT_OUT_DIR = (
    STAGE_ROOT
    / "figures"
    / "phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25"
)
DEFAULT_SUMMARY_DIR = (
    STAGE_ROOT
    / "summarized_outputs"
    / "phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25"
)

EVEN_ODD_RULE = "real_even_odd"
EVEN_ODD_CASE_ID = "even_odd"
EVEN_ODD_LABEL = "even odd"
ADV_NMSTV = {
    "very_low_tv_spectral_teacher": 0.3245703473792008,
    "real_even_odd": 0.4932864276461805,
    "teacher_nn": 0.6843772639598127,
    "random_label": 0.985558573825462,
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.astype(np.float64, copy=True)
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window <= 1:
        return values.astype(np.float64, copy=True)
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_etas(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def eta_case_id(eta: float) -> str:
    return f"eta_{eta:.2f}"


def eta_label(eta: float) -> str:
    return f"eta={eta:.2f}"


def eta_nmstv_map(
    graph_run: Path | None,
    complexity_table: Path,
    etas: list[float],
) -> tuple[dict[float, float], dict[float, str]]:
    values: dict[float, float] = {}
    methods: dict[float, str] = {}
    path = graph_run / "summary_by_eta_k.csv" if graph_run is not None else None
    if path is not None and path.exists():
        graph = pd.read_csv(path)
        graph["k"] = pd.to_numeric(graph["k"], errors="coerce")
        graph["eta"] = pd.to_numeric(graph["eta"], errors="coerce")
        graph["knn_nmstv_mean"] = pd.to_numeric(graph["knn_nmstv_mean"], errors="coerce")
        graph = graph[graph["k"].eq(3)].dropna(subset=["eta", "knn_nmstv_mean"]).sort_values("eta")
        x = graph["eta"].to_numpy(dtype=np.float64)
        y = graph["knn_nmstv_mean"].to_numpy(dtype=np.float64)
        if len(x):
            for eta in etas:
                eta = float(eta)
                exact = np.where(np.isclose(x, eta))[0]
                if len(exact):
                    values[eta] = float(y[exact[0]])
                    methods[eta] = f"exact k=3 lookup from {path}"
                elif x.min() <= eta <= x.max():
                    values[eta] = float(np.interp(eta, x, y))
                    methods[eta] = f"linear interpolation in k=3 table from {path}"
    if complexity_table.exists():
        metrics = pd.read_csv(complexity_table)
        metrics = metrics[metrics["source"].astype(str).eq("flip")].copy()
        metrics["nmstv"] = pd.to_numeric(metrics["nmstv"], errors="coerce")
        for eta in etas:
            eta = float(eta)
            group = f"eta_{eta:.2f}"
            match = metrics[metrics["group"].astype(str).eq(group)].dropna(subset=["nmstv"])
            if not match.empty:
                values[eta] = float(match.iloc[0]["nmstv"])
                methods[eta] = f"exact flip lookup from {complexity_table}"
    for eta in etas:
        eta = float(eta)
        values.setdefault(eta, eta)
        methods.setdefault(eta, "fallback: eta value used as ordering coordinate")
    return values, methods


def numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_units(
    eta_run: Path,
    rule_run: Path,
    graph_run: Path | None,
    complexity_table: Path,
    etas: list[float],
    d_min: float,
    d_max: float,
) -> tuple[pd.DataFrame, dict[float, str], dict[str, Path]]:
    eta_unit_path = eta_run / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    eta_summary_path = eta_run / "06_results_figures" / "eta_reference_phi_by_eta_radius.csv"
    eta_summary_fallback = (
        STAGE_ROOT
        / "summarized_outputs"
        / eta_run.name
        / "06_results_figures"
        / "eta_reference_phi_by_eta_radius.csv"
    )
    if not eta_summary_path.exists() and eta_summary_fallback.exists():
        eta_summary_path = eta_summary_fallback
    rule_unit_path = rule_run / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    rule_summary_path = rule_run / "06_results_figures" / "phi_energy_by_rule_radius.csv"
    required = {
        "eta_unit_table": eta_unit_path,
        "eta_summary_table": eta_summary_path,
        "even_odd_unit_table": rule_unit_path,
        "even_odd_summary_table": rule_summary_path,
        "eta_complexity_table": complexity_table,
    }
    if graph_run is not None:
        required["eta_graph_nmstv_table"] = graph_run / "summary_by_eta_k.csv"
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input CSV(s): " + ", ".join(missing))

    nmstv, nmstv_methods = eta_nmstv_map(graph_run, complexity_table, etas)

    eta_units = pd.read_csv(eta_unit_path)
    eta_units = numeric_columns(eta_units, ["eta", "radius", "ref_id", "phi_energy_raw"])
    eta_units = eta_units[eta_units["eta"].map(lambda value: any(np.isclose(value, eta) for eta in etas))].copy()
    eta_units = eta_units[(eta_units["radius"] >= d_min - 1e-12) & (eta_units["radius"] <= d_max + 1e-12)].copy()
    eta_units["source"] = "eta_flip"
    eta_units["case_id"] = eta_units["eta"].map(lambda value: eta_case_id(float(value)))
    eta_units["case_label"] = eta_units["eta"].map(lambda value: eta_label(float(value)))
    eta_units["nmstv"] = eta_units["eta"].map(lambda value: nmstv[float(value)])
    eta_units["ref_key"] = (
        eta_units["source"].astype(str)
        + ":"
        + eta_units["case_id"].astype(str)
        + ":"
        + eta_units["ref_id"].astype(int).astype(str)
    )

    rule_units = pd.read_csv(rule_unit_path)
    rule_units = numeric_columns(rule_units, ["radius", "ref_id", "phi_energy_raw"])
    rule_units = rule_units[rule_units["rule"].astype(str).eq(EVEN_ODD_RULE)].copy()
    rule_units = rule_units[(rule_units["radius"] >= d_min - 1e-12) & (rule_units["radius"] <= d_max + 1e-12)].copy()
    rule_units["eta"] = np.nan
    rule_units["source"] = "mnist_rule"
    rule_units["case_id"] = EVEN_ODD_CASE_ID
    rule_units["case_label"] = EVEN_ODD_LABEL
    rule_units["nmstv"] = ADV_NMSTV[EVEN_ODD_RULE]
    rule_units["ref_key"] = (
        rule_units["source"].astype(str)
        + ":"
        + rule_units["case_id"].astype(str)
        + ":"
        + rule_units["ref_id"].astype(int).astype(str)
    )

    cols = ["source", "case_id", "case_label", "eta", "nmstv", "ref_id", "ref_key", "radius", "phi_energy_raw"]
    units = pd.concat([rule_units[cols], eta_units[cols]], ignore_index=True)
    units = units.dropna(subset=["radius", "phi_energy_raw"]).sort_values(["source", "case_id", "ref_key", "radius"])
    return units.reset_index(drop=True), nmstv_methods, required


def derive_curves(units: pd.DataFrame, smooth_window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    ref_metrics: list[dict[str, Any]] = []
    group_cols = ["source", "case_id", "case_label", "eta", "nmstv", "ref_key"]
    for key, sub in units.groupby(group_cols, dropna=False, sort=False):
        source, case_id, case_label, eta, nmstv, ref_key = key
        sub = sub.sort_values("radius").drop_duplicates("radius", keep="first")
        if len(sub) < 3:
            continue
        radius = sub["radius"].to_numpy(dtype=np.float64)
        phi = sub["phi_energy_raw"].to_numpy(dtype=np.float64)
        dphi_raw = np.gradient(phi, radius)
        dphi_smooth = centered_moving_average(dphi_raw, smooth_window)
        curvature = np.gradient(dphi_smooth, radius)
        positive_curvature = np.maximum(curvature, 0.0)
        a_kappa = trapz(positive_curvature, radius)
        min_dphi_idx = int(np.nanargmin(dphi_smooth))
        max_curv_idx = int(np.nanargmax(curvature))
        ref_metrics.append(
            {
                "source": source,
                "case_id": case_id,
                "case_label": case_label,
                "eta": eta,
                "nmstv": float(nmstv),
                "ref_key": ref_key,
                "n_radii": int(len(radius)),
                "d_min": float(radius.min()),
                "d_max": float(radius.max()),
                "A_kappa": float(a_kappa),
                "min_dphi_dr": float(dphi_smooth[min_dphi_idx]),
                "min_dphi_dr_radius": float(radius[min_dphi_idx]),
                "max_curvature": float(curvature[max_curv_idx]),
                "max_curvature_radius": float(radius[max_curv_idx]),
            }
        )
        for r, p, g_raw, g_smooth, kappa, pos_kappa in zip(
            radius, phi, dphi_raw, dphi_smooth, curvature, positive_curvature
        ):
            rows.append(
                {
                    "source": source,
                    "case_id": case_id,
                    "case_label": case_label,
                    "eta": eta,
                    "nmstv": float(nmstv),
                    "ref_key": ref_key,
                    "radius": float(r),
                    "phi_energy_raw": float(p),
                    "dphi_dr_raw": float(g_raw),
                    "dphi_dr_smooth": float(g_smooth),
                    "curvature_smooth": float(kappa),
                    "positive_curvature_smooth": float(pos_kappa),
                }
            )
    ref_curve = pd.DataFrame(rows)
    ref_metric = pd.DataFrame(ref_metrics)
    curve_summary = (
        ref_curve.groupby(["source", "case_id", "case_label", "eta", "nmstv", "radius"], dropna=False, as_index=False)
        .agg(
            n_refs=("ref_key", "nunique"),
            phi_energy_raw_mean=("phi_energy_raw", "mean"),
            phi_energy_raw_sd=("phi_energy_raw", "std"),
            phi_energy_raw_sem=("phi_energy_raw", sem),
            dphi_dr_smooth_mean=("dphi_dr_smooth", "mean"),
            dphi_dr_smooth_sd=("dphi_dr_smooth", "std"),
            dphi_dr_smooth_sem=("dphi_dr_smooth", sem),
            curvature_smooth_mean=("curvature_smooth", "mean"),
            curvature_smooth_sd=("curvature_smooth", "std"),
            curvature_smooth_sem=("curvature_smooth", sem),
            positive_curvature_smooth_mean=("positive_curvature_smooth", "mean"),
        )
        .sort_values(["source", "nmstv", "case_id", "radius"])
        .reset_index(drop=True)
    )
    return ref_curve, ref_metric, curve_summary


def summarize_cases(ref_metric: pd.DataFrame, curve_summary: pd.DataFrame) -> pd.DataFrame:
    metric_summary = (
        ref_metric.groupby(["source", "case_id", "case_label", "eta", "nmstv"], dropna=False, as_index=False)
        .agg(
            n_refs=("ref_key", "nunique"),
            A_kappa_mean=("A_kappa", "mean"),
            A_kappa_sd=("A_kappa", "std"),
            A_kappa_sem=("A_kappa", sem),
            min_dphi_dr_mean=("min_dphi_dr", "mean"),
            min_dphi_dr_sd=("min_dphi_dr", "std"),
            min_dphi_dr_sem=("min_dphi_dr", sem),
            min_dphi_dr_radius_mean=("min_dphi_dr_radius", "mean"),
            max_curvature_mean=("max_curvature", "mean"),
            max_curvature_sd=("max_curvature", "std"),
            max_curvature_sem=("max_curvature", sem),
            max_curvature_radius_mean=("max_curvature_radius", "mean"),
        )
        .sort_values(["source", "nmstv", "case_id"])
        .reset_index(drop=True)
    )
    mean_curve_rows = []
    for _, sub in curve_summary.groupby(["source", "case_id", "case_label", "eta", "nmstv"], dropna=False):
        sub = sub.sort_values("radius")
        min_dphi_idx = int(sub["dphi_dr_smooth_mean"].idxmin())
        max_curv_idx = int(sub["curvature_smooth_mean"].idxmax())
        end_idx = int(sub["radius"].idxmax())
        mean_curve_rows.append(
            {
                "case_id": sub["case_id"].iloc[0],
                "mean_curve_min_dphi_dr": float(sub.loc[min_dphi_idx, "dphi_dr_smooth_mean"]),
                "mean_curve_min_dphi_dr_radius": float(sub.loc[min_dphi_idx, "radius"]),
                "mean_curve_max_curvature": float(sub.loc[max_curv_idx, "curvature_smooth_mean"]),
                "mean_curve_max_curvature_radius": float(sub.loc[max_curv_idx, "radius"]),
                "phi_energy_raw_at_dmax": float(sub.loc[end_idx, "phi_energy_raw_mean"]),
                "d_max": float(sub.loc[end_idx, "radius"]),
            }
        )
    mean_curve_summary = pd.DataFrame(mean_curve_rows)
    return metric_summary.merge(mean_curve_summary, on="case_id", how="left")


def compute_gap_tables(
    curve_summary: pd.DataFrame,
    case_summary: pd.DataFrame,
    baseline_case_id: str = EVEN_ODD_CASE_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = curve_summary[curve_summary["case_id"].eq(baseline_case_id)].copy()
    if baseline.empty:
        raise RuntimeError(f"Baseline case not found in curve summary: {baseline_case_id}")
    base_cols = [
        "radius",
        "phi_energy_raw_mean",
        "dphi_dr_smooth_mean",
        "curvature_smooth_mean",
        "positive_curvature_smooth_mean",
    ]
    baseline = baseline[base_cols].rename(
        columns={
            "phi_energy_raw_mean": "even_odd_phi_energy_raw_mean",
            "dphi_dr_smooth_mean": "even_odd_dphi_dr_smooth_mean",
            "curvature_smooth_mean": "even_odd_curvature_smooth_mean",
            "positive_curvature_smooth_mean": "even_odd_positive_curvature_smooth_mean",
        }
    )
    gap_rows = []
    for _, sub in curve_summary[~curve_summary["case_id"].eq(baseline_case_id)].groupby("case_id", sort=False):
        joined = sub.merge(baseline, on="radius", how="inner").sort_values("radius")
        for _, row in joined.iterrows():
            phi_gap = row["phi_energy_raw_mean"] - row["even_odd_phi_energy_raw_mean"]
            dphi_gap = row["dphi_dr_smooth_mean"] - row["even_odd_dphi_dr_smooth_mean"]
            curvature_gap = row["curvature_smooth_mean"] - row["even_odd_curvature_smooth_mean"]
            pos_curv_gap = row["positive_curvature_smooth_mean"] - row["even_odd_positive_curvature_smooth_mean"]
            gap_rows.append(
                {
                    "source": row["source"],
                    "case_id": row["case_id"],
                    "case_label": row["case_label"],
                    "eta": row["eta"],
                    "nmstv": row["nmstv"],
                    "radius": row["radius"],
                    "phi_gap_to_even_odd": phi_gap,
                    "abs_phi_gap_to_even_odd": abs(phi_gap),
                    "dphi_dr_gap_to_even_odd": dphi_gap,
                    "abs_dphi_dr_gap_to_even_odd": abs(dphi_gap),
                    "curvature_gap_to_even_odd": curvature_gap,
                    "abs_curvature_gap_to_even_odd": abs(curvature_gap),
                    "positive_curvature_gap_to_even_odd": pos_curv_gap,
                    "abs_positive_curvature_gap_to_even_odd": abs(pos_curv_gap),
                }
            )
    gap_by_radius = pd.DataFrame(gap_rows).sort_values(["nmstv", "case_id", "radius"]).reset_index(drop=True)

    baseline_a = case_summary[case_summary["case_id"].eq(baseline_case_id)].iloc[0]
    metric_rows = []
    for _, sub in gap_by_radius.groupby("case_id", sort=False):
        x = sub["radius"].to_numpy(dtype=np.float64)
        span = float(x.max() - x.min()) if len(x) > 1 else 1.0
        case_id = str(sub["case_id"].iloc[0])
        case_a = case_summary[case_summary["case_id"].eq(case_id)].iloc[0]

        def metrics(prefix: str) -> dict[str, float]:
            values = sub[f"{prefix}_gap_to_even_odd"].to_numpy(dtype=np.float64)
            abs_values = np.abs(values)
            max_idx = int(np.nanargmax(abs_values))
            return {
                f"{prefix}_signed_area_gap": trapz(values, x),
                f"{prefix}_mean_abs_gap": trapz(abs_values, x) / span,
                f"{prefix}_rmse_gap": float(np.sqrt(np.nanmean(values**2))),
                f"{prefix}_max_abs_gap": float(abs_values[max_idx]),
                f"{prefix}_max_abs_gap_radius": float(x[max_idx]),
            }

        row = {
            "source": sub["source"].iloc[0],
            "case_id": case_id,
            "case_label": sub["case_label"].iloc[0],
            "eta": sub["eta"].iloc[0],
            "nmstv": float(sub["nmstv"].iloc[0]),
            "baseline_case_id": baseline_case_id,
            "A_kappa_mean": float(case_a["A_kappa_mean"]),
            "even_odd_A_kappa_mean": float(baseline_a["A_kappa_mean"]),
            "A_kappa_gap_to_even_odd": float(case_a["A_kappa_mean"] - baseline_a["A_kappa_mean"]),
            "A_kappa_ratio_to_even_odd": float(case_a["A_kappa_mean"] / baseline_a["A_kappa_mean"]),
        }
        row.update(metrics("phi"))
        row.update(metrics("dphi_dr"))
        row.update(metrics("curvature"))
        metric_rows.append(row)
    gap_metrics = pd.DataFrame(metric_rows).sort_values(["nmstv", "case_id"]).reset_index(drop=True)
    return gap_by_radius, gap_metrics


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "0.88",
            "grid.linewidth": 0.7,
        }
    )


def case_colors(curve_summary: pd.DataFrame) -> dict[str, Any]:
    eta_cases = curve_summary[curve_summary["source"].eq("eta_flip")][["case_id", "eta"]].drop_duplicates()
    eta_values = eta_cases["eta"].dropna().to_numpy(dtype=np.float64)
    norm = plt.Normalize(vmin=float(eta_values.min()), vmax=float(eta_values.max())) if len(eta_values) else None
    cmap = plt.get_cmap("viridis")
    colors = {EVEN_ODD_CASE_ID: "0.08"}
    for _, row in eta_cases.iterrows():
        colors[str(row["case_id"])] = cmap(norm(float(row["eta"]))) if norm is not None else cmap(0.5)
    return colors


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_raw_phi(curve_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    colors = case_colors(curve_summary)
    fig, ax = plt.subplots(figsize=(7.2, 4.55), constrained_layout=True)
    for case_id, sub in curve_summary.groupby("case_id", sort=False):
        sub = sub.sort_values("radius")
        color = colors.get(str(case_id), "0.35")
        lw = 2.35 if case_id == EVEN_ODD_CASE_ID else 1.95
        ls = "-" if case_id == EVEN_ODD_CASE_ID else "--"
        label = str(sub["case_label"].iloc[0])
        x = sub["radius"].to_numpy(dtype=np.float64)
        y = sub["phi_energy_raw_mean"].to_numpy(dtype=np.float64)
        err = 1.96 * sub["phi_energy_raw_sem"].fillna(0.0).to_numpy(dtype=np.float64)
        ax.plot(x, y, color=color, lw=lw, ls=ls, label=label)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.10 if case_id != EVEN_ODD_CASE_ID else 0.075, linewidth=0)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel("radius r")
    ax.set_ylabel(r"raw $\phi_E(r)$")
    ax.set_title("Raw local free-energy curves")
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, out_dir, "fig01_raw_phi_E_even_odd_eta_comparison")


def plot_dphi(curve_summary: pd.DataFrame, out_dir: Path, smooth_window: int) -> list[Path]:
    colors = case_colors(curve_summary)
    fig, ax = plt.subplots(figsize=(7.2, 4.55), constrained_layout=True)
    for case_id, sub in curve_summary.groupby("case_id", sort=False):
        sub = sub.sort_values("radius")
        color = colors.get(str(case_id), "0.35")
        lw = 2.35 if case_id == EVEN_ODD_CASE_ID else 1.95
        ls = "-" if case_id == EVEN_ODD_CASE_ID else "--"
        label = str(sub["case_label"].iloc[0])
        x = sub["radius"].to_numpy(dtype=np.float64)
        y = sub["dphi_dr_smooth_mean"].to_numpy(dtype=np.float64)
        err = 1.96 * sub["dphi_dr_smooth_sem"].fillna(0.0).to_numpy(dtype=np.float64)
        ax.plot(x, y, color=color, lw=lw, ls=ls, label=label)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.10 if case_id != EVEN_ODD_CASE_ID else 0.075, linewidth=0)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel("radius r")
    ax.set_ylabel(r"smoothed $d\phi_E/dr$")
    ax.set_title(f"Smoothed derivative curves (window={smooth_window} radii)")
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, out_dir, "fig02_smoothed_dphi_dr_even_odd_eta_comparison")


def plot_curvature_a(curve_summary: pd.DataFrame, case_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    colors = case_colors(curve_summary)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), gridspec_kw={"width_ratios": [1.45, 1.0]}, constrained_layout=True)
    ax_curve, ax_a = axes
    for case_id, sub in curve_summary.groupby("case_id", sort=False):
        sub = sub.sort_values("radius")
        color = colors.get(str(case_id), "0.35")
        lw = 2.25 if case_id == EVEN_ODD_CASE_ID else 1.75
        ls = "-" if case_id == EVEN_ODD_CASE_ID else "--"
        ax_curve.plot(
            sub["radius"],
            sub["curvature_smooth_mean"],
            color=color,
            lw=lw,
            ls=ls,
            label=str(sub["case_label"].iloc[0]),
        )
    ax_curve.axhline(0.0, color="0.35", lw=0.8)
    ax_curve.set_xlabel("radius r")
    ax_curve.set_ylabel(r"curvature $d^2\phi_E/dr^2$")
    ax_curve.set_title("Curvature from smoothed derivative")
    ax_curve.legend(frameon=False, fontsize=8.0)

    ordered = case_summary.sort_values(["source", "nmstv", "case_id"]).copy()
    ordered["x"] = np.arange(len(ordered))
    for _, row in ordered.iterrows():
        color = colors.get(str(row["case_id"]), "0.35")
        err = 1.96 * (0.0 if pd.isna(row["A_kappa_sem"]) else float(row["A_kappa_sem"]))
        ax_a.errorbar(
            [row["x"]],
            [row["A_kappa_mean"]],
            yerr=[err],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=3.5,
            markersize=5.5,
        )
    ax_a.plot(ordered["x"], ordered["A_kappa_mean"], color="0.72", lw=0.8, zorder=0)
    ax_a.set_xticks(ordered["x"])
    ax_a.set_xticklabels(ordered["case_label"], rotation=35, ha="right")
    ax_a.set_ylabel(r"$A_\kappa = \int \max(\kappa,0)\,dr$")
    ax_a.set_title("Positive curvature mass")
    return save_figure(fig, out_dir, "fig03_curvature_A_kappa_even_odd_eta_comparison")


def plot_gaps(gap_by_radius: pd.DataFrame, gap_metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    colors = {}
    eta_values = gap_by_radius[["case_id", "eta"]].drop_duplicates()["eta"].to_numpy(dtype=np.float64)
    norm = plt.Normalize(vmin=float(np.nanmin(eta_values)), vmax=float(np.nanmax(eta_values)))
    cmap = plt.get_cmap("viridis")
    for _, row in gap_by_radius[["case_id", "eta"]].drop_duplicates().iterrows():
        colors[str(row["case_id"])] = cmap(norm(float(row["eta"])))

    fig = plt.figure(figsize=(10.2, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])
    ax_phi = fig.add_subplot(gs[0, 0])
    ax_dphi = fig.add_subplot(gs[0, 1])
    ax_curv = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    for case_id, sub in gap_by_radius.groupby("case_id", sort=False):
        sub = sub.sort_values("radius")
        color = colors[str(case_id)]
        label = str(sub["case_label"].iloc[0])
        ax_phi.plot(sub["radius"], sub["phi_gap_to_even_odd"], color=color, lw=1.8, label=label)
        ax_dphi.plot(sub["radius"], sub["dphi_dr_gap_to_even_odd"], color=color, lw=1.8, label=label)
        ax_curv.plot(sub["radius"], sub["curvature_gap_to_even_odd"], color=color, lw=1.8, label=label)
    for ax, title, ylabel in [
        (ax_phi, "Raw phi_E gap", r"$\Delta\phi_E$"),
        (ax_dphi, "Smoothed derivative gap", r"$\Delta d\phi_E/dr$"),
        (ax_curv, "Curvature gap", r"$\Delta \kappa$"),
    ]:
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_xlabel("radius r")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} vs even odd")
    ax_phi.legend(frameon=False, ncol=2)

    heat_cols = ["phi_rmse_gap", "dphi_dr_rmse_gap", "curvature_rmse_gap", "A_kappa_gap_to_even_odd"]
    heat_labels = ["phi RMSE", "dphi RMSE", "curv RMSE", "A_kappa gap"]
    heat = gap_metrics.set_index("case_label")[heat_cols].copy()
    scaled = heat.abs()
    for col in scaled.columns:
        denom = float(scaled[col].max())
        if denom > 0:
            scaled[col] = scaled[col] / denom
    image = ax_heat.imshow(scaled.to_numpy(dtype=np.float64), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax_heat.set_xticks(np.arange(len(heat_cols)))
    ax_heat.set_xticklabels(heat_labels, rotation=25, ha="right")
    ax_heat.set_yticks(np.arange(len(heat.index)))
    ax_heat.set_yticklabels(list(heat.index))
    ax_heat.set_title("Gap metric magnitude, column-scaled")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            text_color = "white" if scaled.iloc[i, j] > 0.55 else "black"
            ax_heat.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=7.5, color=text_color)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.025)
    cbar.set_label("column-scaled |metric|")
    return save_figure(fig, out_dir, "fig04_gap_metrics_to_even_odd")


def markdown_table(df: pd.DataFrame, columns: list[str], floatfmt: str = ".4g") -> str:
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else format(float(value), floatfmt))
        else:
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in view.columns]
    rows = view.astype(str).values.tolist()
    widths = [
        max([len(headers[idx]), *[len(row[idx]) for row in rows]]) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), sep, *[fmt_row(row) for row in rows]])


def build_report(
    out_dir: Path,
    config: dict[str, Any],
    case_summary: pd.DataFrame,
    gap_metrics: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    report_path = out_dir / "REPORT.md"
    case_cols = [
        "case_label",
        "n_refs",
        "nmstv",
        "A_kappa_mean",
        "A_kappa_sem",
        "mean_curve_min_dphi_dr",
        "mean_curve_min_dphi_dr_radius",
        "phi_energy_raw_at_dmax",
    ]
    gap_cols = [
        "case_label",
        "phi_rmse_gap",
        "dphi_dr_rmse_gap",
        "curvature_rmse_gap",
        "A_kappa_gap_to_even_odd",
        "A_kappa_ratio_to_even_odd",
    ]
    provenance = config["input_paths"]
    lines = [
        "# Phase Transition Discussion: 30ref Four-Eta MNIST",
        "",
        "First-pass reproducible analysis for the current dense four-eta run. This report reads existing CSVs only; it does not rerun reference search, PM-SAIS sampling, or any expensive unit generation.",
        "",
        "## Provenance",
        "",
        f"- Script: `{config['script_path']}`",
        f"- Eta run root: `{provenance['eta_run_root']}`",
        f"- Eta unit CSV: `{provenance['eta_unit_table']}`",
        f"- Eta summary CSV: `{provenance['eta_summary_table']}`",
        f"- Even/odd rule run root: `{provenance['rule_run_root']}`",
        f"- Even/odd unit CSV: `{provenance['even_odd_unit_table']}`",
        f"- Even/odd summary CSV: `{provenance['even_odd_summary_table']}`",
        f"- Eta NMSTV helper CSV: `{provenance['eta_graph_nmstv_table']}`",
        "",
        "## Method",
        "",
        f"- Radius window: `{config['d_min']}` to `{config['d_max']}` on the shared 0.01-spaced grid.",
        f"- Baseline: `{EVEN_ODD_RULE}` from the dense 30-reference active-rule run, labelled `even odd` here.",
        f"- Eta cases: `{', '.join(eta_label(float(x)) for x in config['etas'])}` from the final eta-specific-reference run.",
        f"- Derivative: per-reference finite difference of raw phi_E, then centered edge-padded moving-average smoothing with `{config['smooth_window']}` radii.",
        "- Curvature: finite difference of the smoothed derivative.",
        "- A_kappa: radius integral of positive smoothed curvature, computed per reference and summarized by case.",
        "- Gap metrics: group-mean eta curves minus the group-mean even/odd curve on the same radius grid. These are not reference-paired gaps.",
        "",
        "## Case Summary",
        "",
        markdown_table(case_summary, case_cols),
        "",
        "## Gap Metrics To Even/Odd",
        "",
        markdown_table(gap_metrics, gap_cols),
        "",
        "## Outputs",
        "",
    ]
    for path in output_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            f"python {config['script_path']}",
            "```",
            "",
            "The resolved inputs, parameters, and SHA-256 hashes are in `run_config_resolved.json` and `provenance_paths.json`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta-run-root", type=Path, default=DEFAULT_ETA_RUN)
    parser.add_argument("--rule-run-root", type=Path, default=DEFAULT_RULE_RUN)
    parser.add_argument("--graph-run-root", type=Path, default=None)
    parser.add_argument("--complexity-table", type=Path, default=DEFAULT_COMPLEXITY_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--etas", type=str, default="0.02,0.05,0.15,0.25")
    parser.add_argument("--d-min", type=float, default=0.01)
    parser.add_argument("--d-max", type=float, default=1.0)
    parser.add_argument("--smooth-window", type=int, default=7)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    summary_dir = ensure_dir(args.summary_dir)
    etas = parse_etas(args.etas)
    units, nmstv_methods, input_paths = load_units(
        args.eta_run_root,
        args.rule_run_root,
        args.graph_run_root,
        args.complexity_table,
        etas,
        args.d_min,
        args.d_max,
    )
    ref_curve, ref_metric, curve_summary = derive_curves(units, args.smooth_window)
    case_summary = summarize_cases(ref_metric, curve_summary)
    gap_by_radius, gap_metrics = compute_gap_tables(curve_summary, case_summary)

    table_paths = [
        summary_dir / "curve_summary_by_case_radius.csv",
        summary_dir / "reflevel_derivative_curvature_by_case_radius.csv",
        summary_dir / "A_kappa_by_reference.csv",
        summary_dir / "case_summary_A_kappa.csv",
        summary_dir / "gap_to_even_odd_by_radius.csv",
        summary_dir / "gap_metrics_to_even_odd.csv",
    ]
    curve_summary.to_csv(table_paths[0], index=False)
    ref_curve.to_csv(table_paths[1], index=False)
    ref_metric.to_csv(table_paths[2], index=False)
    case_summary.to_csv(table_paths[3], index=False)
    gap_by_radius.to_csv(table_paths[4], index=False)
    gap_metrics.to_csv(table_paths[5], index=False)

    setup_plot_style()
    figure_paths: list[Path] = []
    figure_paths.extend(plot_raw_phi(curve_summary, out_dir))
    figure_paths.extend(plot_dphi(curve_summary, out_dir, args.smooth_window))
    figure_paths.extend(plot_curvature_a(curve_summary, case_summary, out_dir))
    figure_paths.extend(plot_gaps(gap_by_radius, gap_metrics, out_dir))

    input_paths_json = {
        "eta_run_root": str(args.eta_run_root.resolve()),
        "rule_run_root": str(args.rule_run_root.resolve()),
        "graph_run_root": str(args.graph_run_root.resolve()) if args.graph_run_root is not None else None,
        "complexity_table": str(args.complexity_table.resolve()),
        **{key: str(path.resolve()) for key, path in input_paths.items()},
    }
    input_hashes = {
        key: file_sha256(path)
        for key, path in input_paths.items()
        if path.exists() and path.is_file()
    }
    config = {
        "script_path": str(SCRIPT_PATH),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "out_dir": str(out_dir.resolve()),
        "summary_dir": str(summary_dir.resolve()),
        "d_min": float(args.d_min),
        "d_max": float(args.d_max),
        "smooth_window": int(args.smooth_window),
        "etas": etas,
        "even_odd_rule": EVEN_ODD_RULE,
        "input_paths": input_paths_json,
        "input_sha256": input_hashes,
        "eta_nmstv_methods": {f"{eta:.2f}": method for eta, method in nmstv_methods.items()},
        "n_unit_rows_loaded": int(len(units)),
        "n_ref_curve_rows": int(len(ref_curve)),
        "n_ref_metric_rows": int(len(ref_metric)),
        "outputs": [str(path.resolve()) for path in [*table_paths, *figure_paths, summary_dir / "REPORT.md"]],
    }
    write_json(summary_dir / "run_config_resolved.json", config)
    write_json(
        summary_dir / "provenance_paths.json",
        {
            "input_paths": input_paths_json,
            "input_sha256": input_hashes,
            "output_paths": config["outputs"],
        },
    )
    build_report(
        summary_dir,
        config,
        case_summary,
        gap_metrics,
        [*table_paths, *figure_paths, summary_dir / "run_config_resolved.json", summary_dir / "provenance_paths.json", summary_dir / "REPORT.md"],
    )
    print(summary_dir / "REPORT.md")
    print(gap_metrics[["case_label", "phi_rmse_gap", "dphi_dr_rmse_gap", "curvature_rmse_gap", "A_kappa_gap_to_even_odd"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
