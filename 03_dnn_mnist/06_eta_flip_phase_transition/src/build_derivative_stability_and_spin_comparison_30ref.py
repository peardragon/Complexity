#!/usr/bin/env python3
"""Derivative stability and spin-vs-MNIST comparison for the 30ref eta run.

This script is analysis-only. It reads completed MNIST and 3NN/spin CSVs,
compares several posthoc derivative estimators, and writes discussion figures
without invoking reference search or sampling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


SCRIPT_PATH = Path(__file__).resolve()
STAGE_ROOT = SCRIPT_PATH.parents[1]
LOCAL_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = LOCAL_ROOT.parent

ETA_RUN = (
    STAGE_ROOT
    / "raw_outputs"
    / "eta_reference_phi_dense_4eta_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
RULE_RUN = (
    LOCAL_ROOT
    / "04_sampling"
    / "raw_outputs"
    / "active_rules_refpool1024_dense30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
DISCUSSION_DIR = (
    STAGE_ROOT
    / "figures"
    / "phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25"
)
SPIN_ROOT = PROJECT_ROOT / "02_dnn"
SPIN_DPHI = (
    SPIN_ROOT
    / "05_proxy_local_entropy"
    / "raw_outputs"
    / "18_beta_cell_90_dataset_30_reference"
    / "d_0.01_to_2.50_dense"
    / "summary_tables"
    / "dphi_dr_by_beta_radius.csv"
)
SPIN_PHI = (
    SPIN_ROOT
    / "05_proxy_local_entropy"
    / "raw_outputs"
    / "18_beta_cell_90_dataset_30_reference"
    / "d_0.01_to_2.50_dense"
    / "summary_tables"
    / "absolute_phi_by_beta_radius.csv"
)
SPIN_AKAPPA = (
    SPIN_ROOT
    / "06_random_gaussian_baseline"
    / "figures"
    / "gaussian_overlay_final_derivative"
    / "measure_search"
    / "positive_curvature_mass_composite_spin_only.csv"
)

EVEN_ODD_RULE = "real_even_odd"
ADV_NMSTV = {
    "real_even_odd": 0.4932864276461805,
}
ETA_NMSTV = {
    0.02: 0.356969,
    0.05: 0.430574,
    0.15: 0.655129,
    0.25: 0.811137,
}

CASE_ORDER = ["even_odd", "eta_0.02", "eta_0.05", "eta_0.15", "eta_0.25"]
CASE_LABEL = {
    "even_odd": "even odd",
    "eta_0.02": "eta=0.02",
    "eta_0.05": "eta=0.05",
    "eta_0.15": "eta=0.15",
    "eta_0.25": "eta=0.25",
}
CASE_COLOR = {
    "even_odd": "#111111",
    "eta_0.02": "#4b006e",
    "eta_0.05": "#4b3590",
    "eta_0.15": "#1f9e89",
    "eta_0.25": "#f4d21b",
}
CASE_STYLE = {
    "even_odd": "-",
    "eta_0.02": "--",
    "eta_0.05": "--",
    "eta_0.15": "--",
    "eta_0.25": "--",
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


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
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


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_mnist_units() -> pd.DataFrame:
    eta_path = ETA_RUN / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    rule_path = RULE_RUN / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv"
    if not eta_path.exists():
        raise FileNotFoundError(eta_path)
    if not rule_path.exists():
        raise FileNotFoundError(rule_path)

    eta = pd.read_csv(eta_path)
    eta = numeric(eta, ["eta", "radius", "ref_id", "phi_energy_raw", "split_logZ_per_P_diff", "ess_fraction"])
    eta = eta[eta["eta"].isin(sorted(ETA_NMSTV))].copy()
    eta["case_id"] = eta["eta"].map(lambda value: f"eta_{float(value):.2f}")
    eta["case_label"] = eta["case_id"].map(CASE_LABEL)
    eta["nmstv"] = eta["eta"].map(lambda value: ETA_NMSTV[float(value)])
    eta["source"] = "eta_flip"

    rule = pd.read_csv(rule_path)
    rule = numeric(rule, ["radius", "ref_id", "phi_energy_raw", "split_logZ_per_P_diff", "ess_fraction"])
    rule = rule[rule["rule"].astype(str).eq(EVEN_ODD_RULE)].copy()
    rule["eta"] = np.nan
    rule["case_id"] = "even_odd"
    rule["case_label"] = CASE_LABEL["even_odd"]
    rule["nmstv"] = ADV_NMSTV[EVEN_ODD_RULE]
    rule["source"] = "mnist_rule"

    cols = [
        "source",
        "case_id",
        "case_label",
        "eta",
        "nmstv",
        "ref_id",
        "radius",
        "phi_energy_raw",
        "split_logZ_per_P_diff",
        "ess_fraction",
    ]
    units = pd.concat([rule[cols], eta[cols]], ignore_index=True)
    units = units.dropna(subset=["radius", "phi_energy_raw"])
    units = units[(units["radius"] >= 0.01 - 1e-12) & (units["radius"] <= 1.0 + 1e-12)].copy()
    units["radius"] = units["radius"].round(10)
    units["ref_key"] = units["source"] + ":" + units["case_id"] + ":" + units["ref_id"].astype(int).astype(str)
    return units.sort_values(["case_id", "ref_id", "radius"]).reset_index(drop=True)


def estimator_curves(radius: np.ndarray, phi: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "raw_gradient":
        d1 = np.gradient(phi, radius)
        d2 = np.gradient(d1, radius)
        return radius, d1, d2
    if method == "moving_avg_7":
        d1 = centered_moving_average(np.gradient(phi, radius), 7)
        d2 = np.gradient(d1, radius)
        return radius, d1, d2
    if method == "savgol_11_3":
        step = float(np.median(np.diff(radius)))
        d1 = savgol_filter(phi, window_length=11, polyorder=3, deriv=1, delta=step, mode="interp")
        d2 = savgol_filter(phi, window_length=11, polyorder=3, deriv=2, delta=step, mode="interp")
        return radius, d1, d2
    if method == "savgol_21_3":
        step = float(np.median(np.diff(radius)))
        d1 = savgol_filter(phi, window_length=21, polyorder=3, deriv=1, delta=step, mode="interp")
        d2 = savgol_filter(phi, window_length=21, polyorder=3, deriv=2, delta=step, mode="interp")
        return radius, d1, d2
    if method == "coarsen_0p05":
        keep = np.isclose(((radius * 100).round().astype(int) % 5), 0) | np.isclose(radius, radius.min())
        x = radius[keep]
        y = phi[keep]
        d1 = np.gradient(y, x)
        d2 = np.gradient(d1, x)
        return x, d1, d2
    if method == "coarsen_0p10":
        keep = np.isclose(((radius * 100).round().astype(int) % 10), 0) | np.isclose(radius, radius.min())
        x = radius[keep]
        y = phi[keep]
        d1 = np.gradient(y, x)
        d2 = np.gradient(d1, x)
        return x, d1, d2
    raise ValueError(f"unknown method: {method}")


def sign_changes(values: np.ndarray) -> int:
    clean = values[np.isfinite(values)]
    clean = clean[np.abs(clean) > 1e-12]
    if len(clean) <= 1:
        return 0
    return int(np.count_nonzero(np.diff(np.sign(clean)) != 0))


def build_derivative_tables(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = [
        "raw_gradient",
        "moving_avg_7",
        "savgol_11_3",
        "savgol_21_3",
        "coarsen_0p05",
        "coarsen_0p10",
    ]
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for (case_id, ref_id), sub in units.groupby(["case_id", "ref_id"], sort=False):
        sub = sub.sort_values("radius").drop_duplicates("radius", keep="first")
        radius = sub["radius"].to_numpy(dtype=np.float64)
        phi = sub["phi_energy_raw"].to_numpy(dtype=np.float64)
        if len(radius) < 11:
            continue
        meta = {
            "source": sub["source"].iloc[0],
            "case_id": case_id,
            "case_label": sub["case_label"].iloc[0],
            "eta": sub["eta"].iloc[0],
            "nmstv": float(sub["nmstv"].iloc[0]),
            "ref_id": int(ref_id),
        }
        for method in methods:
            x, d1, d2 = estimator_curves(radius, phi, method)
            pos = np.maximum(d2, 0.0)
            a_kappa = trapz(pos, x)
            min_idx = int(np.nanargmin(d1))
            max_idx = int(np.nanargmax(d2))
            metric_rows.append(
                {
                    **meta,
                    "method": method,
                    "n_radius": int(len(x)),
                    "dphi_min": float(d1[min_idx]),
                    "dphi_min_radius": float(x[min_idx]),
                    "curvature_max": float(d2[max_idx]),
                    "curvature_max_radius": float(x[max_idx]),
                    "curvature_sign_changes": sign_changes(d2),
                    "A_kappa": a_kappa,
                }
            )
            for rr, dd, cc in zip(x, d1, d2):
                curve_rows.append(
                    {
                        **meta,
                        "method": method,
                        "radius": float(rr),
                        "dphi_dr": float(dd),
                        "curvature": float(cc),
                    }
                )

    ref_curves = pd.DataFrame(curve_rows)
    ref_metrics = pd.DataFrame(metric_rows)
    group_curves = (
        ref_curves.groupby(["source", "case_id", "case_label", "eta", "nmstv", "method", "radius"], dropna=False)
        .agg(
            ref_count=("ref_id", "nunique"),
            dphi_dr_mean=("dphi_dr", "mean"),
            dphi_dr_sd=("dphi_dr", "std"),
            dphi_dr_sem=("dphi_dr", sem),
            curvature_mean=("curvature", "mean"),
            curvature_sd=("curvature", "std"),
            curvature_sem=("curvature", sem),
        )
        .reset_index()
    )
    metric_summary = (
        ref_metrics.groupby(["source", "case_id", "case_label", "eta", "nmstv", "method"], dropna=False)
        .agg(
            ref_count=("ref_id", "nunique"),
            A_kappa_mean=("A_kappa", "mean"),
            A_kappa_sd=("A_kappa", "std"),
            A_kappa_sem=("A_kappa", sem),
            dphi_min_mean=("dphi_min", "mean"),
            dphi_min_radius_mean=("dphi_min_radius", "mean"),
            curvature_max_mean=("curvature_max", "mean"),
            curvature_max_radius_mean=("curvature_max_radius", "mean"),
            curvature_sign_changes_mean=("curvature_sign_changes", "mean"),
            curvature_sign_changes_sd=("curvature_sign_changes", "std"),
        )
        .reset_index()
    )
    return ref_curves, ref_metrics, group_curves.merge(metric_summary, how="left")


def group_phi_summary(units: pd.DataFrame) -> pd.DataFrame:
    return (
        units.groupby(["source", "case_id", "case_label", "eta", "nmstv", "radius"], dropna=False)
        .agg(
            ref_count=("ref_id", "nunique"),
            phi_mean=("phi_energy_raw", "mean"),
            phi_sd=("phi_energy_raw", "std"),
            phi_sem=("phi_energy_raw", sem),
            split_mean=("split_logZ_per_P_diff", "mean"),
            split_max=("split_logZ_per_P_diff", "max"),
            ess_fraction_mean=("ess_fraction", "mean"),
        )
        .reset_index()
    )


def write_csvs(
    out_dir: Path,
    units: pd.DataFrame,
    phi_summary: pd.DataFrame,
    ref_curves: pd.DataFrame,
    ref_metrics: pd.DataFrame,
    group_curves: pd.DataFrame,
) -> None:
    units.to_csv(out_dir / "analysis_units_even_odd_eta30ref.csv", index=False)
    phi_summary.to_csv(out_dir / "phi_summary_even_odd_eta30ref.csv", index=False)
    ref_curves.to_csv(out_dir / "derivative_estimator_ref_curves.csv", index=False)
    ref_metrics.to_csv(out_dir / "derivative_estimator_ref_metrics.csv", index=False)
    group_curves.to_csv(out_dir / "derivative_estimator_group_curves_and_metrics.csv", index=False)

    stability = (
        group_curves.drop_duplicates(["case_id", "method"])
        .loc[
            :,
            [
                "source",
                "case_id",
                "case_label",
                "eta",
                "nmstv",
                "method",
                "ref_count",
                "A_kappa_mean",
                "A_kappa_sem",
                "dphi_min_mean",
                "dphi_min_radius_mean",
                "curvature_max_mean",
                "curvature_max_radius_mean",
                "curvature_sign_changes_mean",
            ],
        ]
        .sort_values(["case_id", "method"])
    )
    stability.to_csv(out_dir / "derivative_stability_summary_by_case_method.csv", index=False)


def plot_estimator_sensitivity(out_dir: Path, phi_summary: pd.DataFrame, group_curves: pd.DataFrame) -> None:
    method_labels = {
        "raw_gradient": "raw grad",
        "moving_avg_7": "MA7",
        "savgol_11_3": "SG11",
        "savgol_21_3": "SG21",
        "coarsen_0p05": "0.05 step",
        "coarsen_0p10": "0.10 step",
    }
    method_styles = {
        "raw_gradient": (":", 1.4, 0.55),
        "moving_avg_7": ("-", 2.2, 0.88),
        "savgol_11_3": ("--", 1.8, 0.78),
        "savgol_21_3": ("-.", 1.8, 0.78),
        "coarsen_0p05": ("-", 1.7, 0.70),
        "coarsen_0p10": ("-", 1.4, 0.55),
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    ax_phi, ax_even, ax_eta25, ax_ak = axes.ravel()

    for case_id in CASE_ORDER:
        sub = phi_summary[phi_summary["case_id"].eq(case_id)].sort_values("radius")
        if sub.empty:
            continue
        color = CASE_COLOR[case_id]
        ax_phi.plot(
            sub["radius"],
            sub["phi_mean"],
            CASE_STYLE[case_id],
            color=color,
            lw=2.6 if case_id != "even_odd" else 3.0,
            label=CASE_LABEL[case_id],
        )
        ax_phi.fill_between(
            sub["radius"].to_numpy(dtype=float),
            (sub["phi_mean"] - sub["phi_sem"]).to_numpy(dtype=float),
            (sub["phi_mean"] + sub["phi_sem"]).to_numpy(dtype=float),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    ax_phi.axhline(0.0, color="0.25", lw=1.0)
    ax_phi.set_title("Primary observable: raw phi_E(d)")
    ax_phi.set_xlabel("radius d")
    ax_phi.set_ylabel("phi_E")
    ax_phi.legend(ncol=2, fontsize=9)
    ax_phi.grid(alpha=0.3)

    for ax, case_id, title in [
        (ax_even, "even_odd", "Derivative estimator sweep: even odd"),
        (ax_eta25, "eta_0.25", "Derivative estimator sweep: eta=0.25"),
    ]:
        for method, label in method_labels.items():
            sub = group_curves[
                group_curves["case_id"].eq(case_id) & group_curves["method"].eq(method)
            ].sort_values("radius")
            if sub.empty:
                continue
            ls, lw, alpha = method_styles[method]
            ax.plot(sub["radius"], sub["dphi_dr_mean"], ls=ls, lw=lw, alpha=alpha, label=label)
        ax.axhline(0.0, color="0.25", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("radius d")
        ax.set_ylabel("d phi_E / dd")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    ak = group_curves.drop_duplicates(["case_id", "method"])
    x_base = np.arange(len(CASE_ORDER), dtype=float)
    offsets = {
        "raw_gradient": -0.25,
        "moving_avg_7": -0.15,
        "savgol_11_3": -0.05,
        "savgol_21_3": 0.05,
        "coarsen_0p05": 0.15,
        "coarsen_0p10": 0.25,
    }
    for method, label in method_labels.items():
        vals = []
        errs = []
        for case_id in CASE_ORDER:
            row = ak[ak["case_id"].eq(case_id) & ak["method"].eq(method)]
            vals.append(float(row["A_kappa_mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["A_kappa_sem"].iloc[0]) if not row.empty else np.nan)
        ax_ak.errorbar(
            x_base + offsets[method],
            vals,
            yerr=errs,
            fmt="o",
            ms=4,
            lw=1.0,
            capsize=2,
            alpha=0.85,
            label=label,
        )
    ax_ak.set_xticks(x_base)
    ax_ak.set_xticklabels([CASE_LABEL[item] for item in CASE_ORDER], rotation=35, ha="right")
    ax_ak.set_title("A_kappa is method-sensitive")
    ax_ak.set_ylabel("positive curvature mass")
    ax_ak.grid(axis="y", alpha=0.3)
    ax_ak.legend(fontsize=8, ncol=2)

    fig.suptitle("MNIST 30ref derivative stability: phi_E is stable, derivatives are estimator dependent")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig05_derivative_estimator_stability.{ext}", dpi=220)
    plt.close(fig)


def plot_qc_and_noise(out_dir: Path, units: pd.DataFrame, group_curves: pd.DataFrame) -> pd.DataFrame:
    qc = (
        units.groupby(["case_id", "case_label", "radius"], dropna=False)
        .agg(
            split_mean=("split_logZ_per_P_diff", "mean"),
            split_max=("split_logZ_per_P_diff", "max"),
            split_fail_frac=("split_logZ_per_P_diff", lambda x: float((x > 0.004).mean())),
            ess_fraction_mean=("ess_fraction", "mean"),
            phi_sd=("phi_energy_raw", "std"),
            phi_sem=("phi_energy_raw", sem),
        )
        .reset_index()
    )
    raw = group_curves[group_curves["method"].eq("raw_gradient")][
        ["case_id", "radius", "curvature_mean", "curvature_sem", "dphi_dr_mean"]
    ]
    qc = qc.merge(raw, on=["case_id", "radius"], how="left")
    qc["curvature_abs_mean"] = qc["curvature_mean"].abs()
    qc.to_csv(out_dir / "qc_noise_derivative_spike_by_case_radius.csv", index=False)

    corr_rows = []
    for case_id, sub in qc.groupby("case_id"):
        for a, b in [
            ("split_mean", "curvature_abs_mean"),
            ("split_fail_frac", "curvature_abs_mean"),
            ("phi_sem", "curvature_abs_mean"),
            ("ess_fraction_mean", "curvature_abs_mean"),
        ]:
            clean = sub[[a, b]].dropna()
            corr = float(clean[a].corr(clean[b])) if len(clean) > 2 else np.nan
            corr_rows.append({"case_id": case_id, "x": a, "y": b, "pearson_corr": corr})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out_dir / "qc_noise_derivative_spike_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for case_id in CASE_ORDER:
        sub = qc[qc["case_id"].eq(case_id)].sort_values("radius")
        if sub.empty:
            continue
        color = CASE_COLOR[case_id]
        axes[0].plot(sub["radius"], sub["split_fail_frac"], CASE_STYLE[case_id], color=color, label=CASE_LABEL[case_id])
        axes[1].plot(sub["radius"], sub["phi_sem"], CASE_STYLE[case_id], color=color)
        axes[2].plot(sub["radius"], sub["curvature_abs_mean"], CASE_STYLE[case_id], color=color)
    axes[0].set_title("Unit split-fail fraction")
    axes[0].set_ylabel("fraction with split diff > 0.004")
    axes[1].set_title("phi_E SEM across refs")
    axes[1].set_ylabel("SEM")
    axes[2].set_title("raw-gradient curvature magnitude")
    axes[2].set_ylabel("|mean curvature|")
    for ax in axes:
        ax.set_xlabel("radius d")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Noise diagnostics for derivative instability")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig06_qc_noise_derivative_diagnostics.{ext}", dpi=220)
    plt.close(fig)
    return corr


def load_spin_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (SPIN_DPHI, SPIN_PHI, SPIN_AKAPPA):
        if not path.exists():
            raise FileNotFoundError(path)
    spin_dphi = pd.read_csv(SPIN_DPHI)
    spin_phi = pd.read_csv(SPIN_PHI)
    spin_ak = pd.read_csv(SPIN_AKAPPA)
    spin_dphi = numeric(spin_dphi, ["beta", "radius", "dphi_energy_dr", "ref_count"])
    spin_phi = numeric(spin_phi, ["beta", "radius", "phi_energy", "ref_count"])
    spin_ak = numeric(spin_ak, ["beta", "A_kappa"])
    return spin_dphi, spin_phi, spin_ak


def plot_spin_mnist_comparison(
    out_dir: Path,
    phi_summary: pd.DataFrame,
    group_curves: pd.DataFrame,
    spin_dphi: pd.DataFrame,
    spin_ak: pd.DataFrame,
) -> pd.DataFrame:
    method = "savgol_21_3"
    mnist_ak = group_curves[group_curves["method"].eq(method)].drop_duplicates(["case_id", "method"]).copy()
    mnist_ak = mnist_ak[mnist_ak["case_id"].isin(CASE_ORDER)].copy()
    spin_selected = [0.05, 0.15, 0.19, 0.21, 0.25, 0.33]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    ax_spin_d, ax_mnist_d, ax_spin_a, ax_mnist_a = axes.ravel()

    cmap = plt.get_cmap("viridis")
    beta_values = sorted(spin_dphi["beta"].dropna().unique())
    beta_min, beta_max = min(beta_values), max(beta_values)
    for beta in spin_selected:
        sub = spin_dphi[np.isclose(spin_dphi["beta"], beta) & (spin_dphi["radius"] <= 1.0)].sort_values("radius")
        if sub.empty:
            continue
        color = cmap((beta - beta_min) / (beta_max - beta_min))
        ax_spin_d.plot(sub["radius"], sub["dphi_energy_dr"], color=color, lw=2.0, label=f"beta={beta:.2f}")
    ax_spin_d.axhline(0.0, color="0.25", lw=1.0)
    ax_spin_d.set_title("3NN spin: direct sampled dphi_E/dd")
    ax_spin_d.set_xlabel("radius d")
    ax_spin_d.set_ylabel("dphi_E/dd")
    ax_spin_d.grid(alpha=0.3)
    ax_spin_d.legend(fontsize=8, ncol=2)

    for case_id in CASE_ORDER:
        sub = group_curves[
            group_curves["case_id"].eq(case_id) & group_curves["method"].eq(method)
        ].sort_values("radius")
        if sub.empty:
            continue
        ax_mnist_d.plot(
            sub["radius"],
            sub["dphi_dr_mean"],
            CASE_STYLE[case_id],
            color=CASE_COLOR[case_id],
            lw=2.5 if case_id != "even_odd" else 3.0,
            label=CASE_LABEL[case_id],
        )
        ax_mnist_d.fill_between(
            sub["radius"].to_numpy(float),
            (sub["dphi_dr_mean"] - sub["dphi_dr_sem"]).to_numpy(float),
            (sub["dphi_dr_mean"] + sub["dphi_dr_sem"]).to_numpy(float),
            color=CASE_COLOR[case_id],
            alpha=0.10,
            linewidth=0,
        )
    ax_mnist_d.axhline(0.0, color="0.25", lw=1.0)
    ax_mnist_d.set_title("MNIST: posthoc smoothed finite-difference dphi_E/dd")
    ax_mnist_d.set_xlabel("radius d")
    ax_mnist_d.set_ylabel("dphi_E/dd")
    ax_mnist_d.grid(alpha=0.3)
    ax_mnist_d.legend(fontsize=8, ncol=2)

    spin_ak = spin_ak.sort_values("beta")
    ax_spin_a.plot(spin_ak["beta"], spin_ak["A_kappa"], "o-", color="#375a9e", lw=2.0)
    ax_spin_a.set_title("3NN spin A_kappa: sharp collapse")
    ax_spin_a.set_xlabel("spin beta")
    ax_spin_a.set_ylabel("A_kappa")
    ax_spin_a.grid(alpha=0.3)

    x_labels = [CASE_LABEL[item] for item in CASE_ORDER]
    x = np.arange(len(CASE_ORDER))
    y = []
    yerr = []
    for case_id in CASE_ORDER:
        row = mnist_ak[mnist_ak["case_id"].eq(case_id)]
        y.append(float(row["A_kappa_mean"].iloc[0]) if not row.empty else np.nan)
        yerr.append(float(row["A_kappa_sem"].iloc[0]) if not row.empty else np.nan)
    ax_mnist_a.errorbar(x, y, yerr=yerr, fmt="o-", color="#222222", lw=1.6, capsize=3)
    for idx, case_id in enumerate(CASE_ORDER):
        ax_mnist_a.scatter([idx], [y[idx]], color=CASE_COLOR[case_id], s=55, zorder=3)
    ax_mnist_a.set_xticks(x)
    ax_mnist_a.set_xticklabels(x_labels, rotation=35, ha="right")
    ax_mnist_a.set_title("MNIST A_kappa: smooth crossover, not collapse")
    ax_mnist_a.set_ylabel(f"A_kappa ({method})")
    ax_mnist_a.grid(axis="y", alpha=0.3)

    fig.suptitle("3NN synthetic spin versus MNIST real-label/flip geometry")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig07_spin_synthetic_vs_mnist_real_phase_comparison.{ext}", dpi=220)
    plt.close(fig)

    comparison_rows = []
    for _, row in spin_ak.iterrows():
        comparison_rows.append(
            {
                "system": "3nn_spin",
                "control": "beta",
                "control_value": float(row["beta"]),
                "case_label": f"beta={float(row['beta']):.2f}",
                "A_kappa": float(row["A_kappa"]),
                "A_kappa_sem": np.nan,
                "derivative_source": "direct sampler derivative dlogZ/dd aggregated by beta/radius",
            }
        )
    for case_id in CASE_ORDER:
        row = mnist_ak[mnist_ak["case_id"].eq(case_id)].iloc[0]
        comparison_rows.append(
            {
                "system": "mnist_real_or_flip",
                "control": "eta_or_rule",
                "control_value": float(row["eta"]) if not pd.isna(row["eta"]) else np.nan,
                "case_label": CASE_LABEL[case_id],
                "A_kappa": float(row["A_kappa_mean"]),
                "A_kappa_sem": float(row["A_kappa_sem"]),
                "derivative_source": f"posthoc {method} finite difference from sampled phi_E(d)",
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "phase_metric_comparison_spin_vs_mnist.csv", index=False)
    return comparison


def write_report(out_dir: Path, corr: pd.DataFrame, comparison: pd.DataFrame, group_curves: pd.DataFrame) -> None:
    stable = group_curves.drop_duplicates(["case_id", "method"])
    raw = stable[stable["method"].eq("raw_gradient")].set_index("case_id")
    sg21 = stable[stable["method"].eq("savgol_21_3")].set_index("case_id")
    spin = comparison[comparison["system"].eq("3nn_spin")].copy()
    mnist = comparison[comparison["system"].eq("mnist_real_or_flip")].copy()

    def value(df: pd.DataFrame, case_id: str, col: str) -> float:
        return float(df.loc[case_id, col])

    rows = [
        "# Derivative Stability And Spin Comparison",
        "",
        "This analysis reads existing completed CSVs only. It does not rerun reference search or sampling.",
        "",
        "## Main Findings",
        "",
        "1. `phi_E(d)` is the primary MNIST observable and is stable at the mean-curve level.",
        "2. Current MNIST `dphi_E/dd` is not a direct sampled derivative. It is reconstructed from `phi_E(d)` by finite differences, so it is sensitive to radius step, smoothing, and edge handling.",
        "3. The positive-curvature mass `A_kappa` is useful as a descriptive diagnostic, but its absolute value is method dependent in MNIST.",
        "4. The 3NN spin result is methodologically stronger for phase-transition claims because the first derivative was stored directly by the sampler and aggregated over a much larger beta/radius/reference table.",
        "5. The MNIST eta/rule curves support a smooth crossover interpretation rather than a sharp order-parameter collapse.",
        "",
        "## Estimator Sensitivity Snapshot",
        "",
        "| case | raw A_kappa | SG21 A_kappa | raw sign changes | SG21 sign changes |",
        "| ---- | ----------- | ------------ | ---------------- | ----------------- |",
    ]
    for case_id in CASE_ORDER:
        rows.append(
            f"| {CASE_LABEL[case_id]} | "
            f"{value(raw, case_id, 'A_kappa_mean'):.4g} | "
            f"{value(sg21, case_id, 'A_kappa_mean'):.4g} | "
            f"{value(raw, case_id, 'curvature_sign_changes_mean'):.2f} | "
            f"{value(sg21, case_id, 'curvature_sign_changes_mean'):.2f} |"
        )
    rows.extend(
        [
            "",
            "The raw-gradient curvature contains many sign changes because a 0.01-spaced finite difference amplifies small SMC/logZ fluctuations. Smoothing or coarsening preserves the broad ordering by eta but changes `A_kappa` scale and peak locations.",
            "",
            "## Spin Versus MNIST",
            "",
            f"- Spin `A_kappa` ranges from {spin['A_kappa'].max():.6g} to {spin['A_kappa'].min():.6g} and collapses to zero across the beta sweep.",
            f"- MNIST `A_kappa` in the SG21 diagnostic ranges from {mnist['A_kappa'].min():.4g} to {mnist['A_kappa'].max():.4g}, but it is already positive at the lowest eta/even-odd cases and changes gradually.",
            "- Therefore the MNIST evidence is better described as a smooth family-dependent crossover in local free energy, not a sharp phase transition.",
            "",
            "## Noise/QC Correlations",
            "",
        ]
    )
    for _, row in corr.iterrows():
        rows.append(
            f"- {row['case_id']}: corr({row['x']}, {row['y']}) = "
            f"{row['pearson_corr']:.3f}"
        )
    rows.extend(
        [
            "",
            "## Outputs",
            "",
            "- `fig05_derivative_estimator_stability.png`",
            "- `fig06_qc_noise_derivative_diagnostics.png`",
            "- `fig07_spin_synthetic_vs_mnist_real_phase_comparison.png`",
            "- `derivative_stability_summary_by_case_method.csv`",
            "- `qc_noise_derivative_spike_by_case_radius.csv`",
            "- `qc_noise_derivative_spike_correlations.csv`",
            "- `phase_metric_comparison_spin_vs_mnist.csv`",
            "",
            "## Interpretation For Paper Discussion",
            "",
            "For MNIST, we should claim robust ordering and smooth crossover in `phi_E(d)` as eta/rule complexity increases. We should not claim a spin-like phase transition from the current derivative/curvature diagnostics alone. To make a stronger phase-transition claim on MNIST, the next necessary experiment is either a direct derivative estimator analogous to the 3NN spin stack, or independent replicate sampling at selected radii combined with a pre-registered smoothing/coarsening rule.",
            "",
        ]
    )
    (out_dir / "DERIVATIVE_STABILITY_AND_SPIN_COMPARISON_REPORT.md").write_text(
        "\n".join(rows),
        encoding="utf-8",
    )


def main() -> None:
    out_dir = ensure_dir(DISCUSSION_DIR)
    units = load_mnist_units()
    phi_summary = group_phi_summary(units)
    ref_curves, ref_metrics, group_curves = build_derivative_tables(units)
    write_csvs(out_dir, units, phi_summary, ref_curves, ref_metrics, group_curves)
    plot_estimator_sensitivity(out_dir, phi_summary, group_curves)
    corr = plot_qc_and_noise(out_dir, units, group_curves)
    spin_dphi, _spin_phi, spin_ak = load_spin_tables()
    comparison = plot_spin_mnist_comparison(out_dir, phi_summary, group_curves, spin_dphi, spin_ak)
    write_report(out_dir, corr, comparison, group_curves)

    config = {
        "inputs": {
            "eta_run": ETA_RUN,
            "rule_run": RULE_RUN,
            "spin_dphi": SPIN_DPHI,
            "spin_phi": SPIN_PHI,
            "spin_A_kappa": SPIN_AKAPPA,
        },
        "outputs": {
            "out_dir": out_dir,
            "figures": [
                "fig05_derivative_estimator_stability.png",
                "fig06_qc_noise_derivative_diagnostics.png",
                "fig07_spin_synthetic_vs_mnist_real_phase_comparison.png",
            ],
        },
        "methods": [
            "raw_gradient",
            "moving_avg_7",
            "savgol_11_3",
            "savgol_21_3",
            "coarsen_0p05",
            "coarsen_0p10",
        ],
    }
    (out_dir / "derivative_stability_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote derivative stability and spin comparison to {out_dir}")


if __name__ == "__main__":
    main()
