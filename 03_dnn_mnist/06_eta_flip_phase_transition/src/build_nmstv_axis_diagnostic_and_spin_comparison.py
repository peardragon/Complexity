#!/usr/bin/env python3
"""Diagnose the MNIST NMSTV axis and compare the direct MNIST run to 3NN spin."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
PROJECT_ROOT = LOCAL_ROOT.parent
STAGE_ROOT = LOCAL_ROOT / "06_eta_flip_phase_transition"

DIRECT_FIG_DIR = (
    STAGE_ROOT
    / "figures"
    / "direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
DIRECT_RUN_ROOT = (
    STAGE_ROOT
    / "raw_outputs"
    / "direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
OLD_AXIS_DIR = STAGE_ROOT / "figures" / "complexity_axis_spin_mnist_30ref_eta0p02_0p05_0p15_0p25"
OUT_DIR = STAGE_ROOT / "figures" / "nmstv_axis_diagnostic_and_spin_comparison_direct30ref"

MNIST_COMPLEXITY = DIRECT_FIG_DIR / "combined_direct_dataset_complexity_metrics.csv"
MNIST_CURVATURE = DIRECT_FIG_DIR / "combined_direct_curvature_metrics_by_group.csv"
MNIST_PHI = DIRECT_FIG_DIR / "combined_direct_phi_by_group_radius.csv"
OLD_MNIST_AXIS = OLD_AXIS_DIR / "mnist_complexity_axis_metrics.csv"
ACTIVE_REF_INDEX = DIRECT_RUN_ROOT / "01_active_rules_sampling" / "04_reference_pool" / "reference_pool_index.csv"
ETA_REF_INDEX = DIRECT_RUN_ROOT / "02_eta_flip_sampling" / "04_reference_pool" / "reference_index.csv"

SPIN_DPHI = (
    PROJECT_ROOT
    / "02_dnn"
    / "05_proxy_local_entropy"
    / "raw_outputs"
    / "18_beta_cell_90_dataset_30_reference"
    / "d_0.01_to_2.50_dense"
    / "summary_tables"
    / "dphi_dr_by_beta_radius.csv"
)
SPIN_AKAPPA = (
    PROJECT_ROOT
    / "02_dnn"
    / "06_random_gaussian_baseline"
    / "figures"
    / "gaussian_overlay_final_derivative"
    / "measure_search"
    / "positive_curvature_mass_composite_spin_only.csv"
)
SPIN_CANDIDATE = (
    PROJECT_ROOT
    / "02_dnn"
    / "06_random_gaussian_baseline"
    / "figures"
    / "gaussian_overlay_final_derivative"
    / "measure_search"
    / "candidate_measures_by_beta_spin.csv"
)

RULE_LABELS = {
    "very_low_tv_spectral_teacher": "rule: very low tv",
    "real_even_odd": "rule: even odd",
    "teacher_nn": "rule: teacher nn",
    "random_label": "rule: random label",
}
RULE_SHORT = {
    "rule: very low tv": "very low tv",
    "rule: even odd": "even odd",
    "rule: teacher nn": "teacher nn",
    "rule: random label": "random label",
    "flip eta=0.05": "eta=0.05",
    "flip eta=0.15": "eta=0.15",
    "flip eta=0.25": "eta=0.25",
}
CASE_ORDER = {
    "rule: very low tv": 0,
    "rule: even odd": 1,
    "flip eta=0.05": 2,
    "rule: teacher nn": 3,
    "flip eta=0.15": 4,
    "flip eta=0.25": 5,
    "rule: random label": 6,
}
TREND_LABELS = [
    "rule: very low tv",
    "rule: even odd",
    "flip eta=0.05",
    "flip eta=0.15",
    "flip eta=0.25",
    "rule: random label",
]


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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def norm01(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if not np.any(mask):
        return out
    lo = float(np.min(arr[mask]))
    hi = float(np.max(arr[mask]))
    if hi <= lo:
        return out
    out[mask] = (arr[mask] - lo) / (hi - lo)
    return out


def eta_token_to_float(value: object) -> float | None:
    text = str(value)
    if text.startswith("eta_"):
        text = text.replace("eta_", "").replace("p", ".")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def label_short(label: object) -> str:
    return RULE_SHORT.get(str(label), str(label).replace("flip ", "").replace("rule: ", ""))


def case_color_map(metrics: pd.DataFrame) -> dict[str, Any]:
    cmap = plt.colormaps["viridis"]
    normed = norm01(metrics["nmstv"].to_numpy(dtype=float))
    return {str(label): cmap(float(value)) for label, value in zip(metrics["label"], normed)}


def load_ref_error_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if ACTIVE_REF_INDEX.exists():
        active = pd.read_csv(ACTIVE_REF_INDEX)
        active["source"] = "advanced"
        active["group"] = active["rule"].astype(str)
        active["label"] = active["rule"].map(RULE_LABELS).fillna(active["rule"].astype(str))
        rows.append(active)
    if ETA_REF_INDEX.exists():
        eta = pd.read_csv(ETA_REF_INDEX)
        eta["source"] = "flip"
        eta_value = pd.to_numeric(eta.get("eta"), errors="coerce")
        eta["group"] = eta_value.map(lambda value: f"eta_{value:.2f}" if pd.notna(value) else "")
        eta["label"] = eta_value.map(lambda value: f"flip eta={value:.2f}" if pd.notna(value) else "")
        rows.append(eta)
    if not rows:
        return pd.DataFrame(columns=["source", "group", "label", "ref_count", "ref_test_error_mean", "ref_test_error_sem"])

    refs = pd.concat(rows, ignore_index=True)
    return (
        refs.groupby(["source", "group", "label"], as_index=False)
        .agg(
            ref_count=("ref_id", "nunique"),
            ref_test_error_mean=("test_error", "mean"),
            ref_test_error_sem=("test_error", sem),
            ref_train_error_mean=("train_error", "mean"),
            ref_CE_test_mean=("CE_mean_test", "mean"),
        )
        .sort_values(["source", "group"])
        .reset_index(drop=True)
    )


def load_mnist_metrics() -> pd.DataFrame:
    complexity = pd.read_csv(MNIST_COMPLEXITY)
    curvature = pd.read_csv(MNIST_CURVATURE)
    refs = load_ref_error_metrics()

    keep_complexity = [
        "source",
        "group",
        "label",
        "pos_fraction",
        "same_X_as_even_odd",
        "label_diff_vs_even_odd",
        "stored_flip_rate",
        "tv_mean",
        "baseline_mean",
        "nmstv",
    ]
    metrics = complexity[[c for c in keep_complexity if c in complexity.columns]].merge(
        curvature.drop(columns=["nmstv"], errors="ignore"),
        on=["source", "group", "label"],
        how="left",
    )
    metrics = metrics.merge(refs, on=["source", "group", "label"], how="left")
    metrics["case_order"] = metrics["label"].map(CASE_ORDER).fillna(99).astype(int)
    metrics["short_label"] = metrics["label"].map(label_short)
    metrics["trend_family"] = np.where(metrics["label"].isin(TREND_LABELS), "rule_eta_random_trend", "separate_teacher_rule")
    metrics["complexity_norm"] = norm01(metrics["nmstv"].to_numpy(dtype=float))
    metrics["A_kappa_norm"] = norm01(metrics["positive_curvature_mass_mean"].to_numpy(dtype=float))
    metrics["ref_test_error_norm"] = norm01(metrics["ref_test_error_mean"].to_numpy(dtype=float))
    return metrics.sort_values(["case_order", "nmstv"]).reset_index(drop=True)


def load_stale_vs_corrected(metrics: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_MNIST_AXIS)
    old = old[["source", "group", "label", "nmstv", "complexity_proxy"]].copy()
    old = old.rename(columns={"nmstv": "stale_nmstv", "complexity_proxy": "stale_complexity_proxy"})
    merged = metrics[
        [
            "source",
            "group",
            "label",
            "short_label",
            "case_order",
            "nmstv",
            "tv_mean",
            "baseline_mean",
            "label_diff_vs_even_odd",
            "stored_flip_rate",
        ]
    ].merge(old[["source", "group", "stale_nmstv", "stale_complexity_proxy"]], on=["source", "group"], how="left")
    merged["nmstv_delta_corrected_minus_stale"] = merged["nmstv"] - merged["stale_nmstv"]
    merged["was_stale_left_of_even_odd"] = False
    even_old = merged.loc[merged["label"].eq("rule: even odd"), "stale_nmstv"]
    if not even_old.empty:
        merged["was_stale_left_of_even_odd"] = merged["stale_nmstv"] < float(even_old.iloc[0])
    return merged.sort_values(["case_order", "nmstv"]).reset_index(drop=True)


def expected_eta_nmstv(metrics: pd.DataFrame) -> pd.DataFrame:
    even = metrics[metrics["label"].eq("rule: even odd")]
    if even.empty:
        return pd.DataFrame()
    q0 = float(even["tv_mean"].iloc[0])
    rows = []
    for _, row in metrics[metrics["source"].eq("flip")].sort_values("case_order").iterrows():
        eta_policy = eta_token_to_float(row["group"])
        eta_actual = row.get("stored_flip_rate")
        eta_for_formula = float(eta_actual) if pd.notna(eta_actual) else float(eta_policy)
        a = 2.0 * eta_for_formula * (1.0 - eta_for_formula)
        pred_tv = q0 * (1.0 - 2.0 * a) + a
        baseline = float(row["baseline_mean"]) if pd.notna(row.get("baseline_mean")) else 0.5
        pred_nmstv = pred_tv / baseline if baseline > 0 else float("nan")
        rows.append(
            {
                "label": row["label"],
                "group": row["group"],
                "eta_policy": eta_policy,
                "stored_flip_rate": eta_actual,
                "eta_for_formula": eta_for_formula,
                "base_even_odd_tv_mean": q0,
                "flip_edge_noise_a": a,
                "expected_tv_mean": pred_tv,
                "baseline_mean": baseline,
                "expected_nmstv": pred_nmstv,
                "actual_nmstv": float(row["nmstv"]),
                "actual_minus_expected_nmstv": float(row["nmstv"]) - pred_nmstv,
            }
        )
    return pd.DataFrame(rows)


def load_spin_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    dphi = pd.read_csv(SPIN_DPHI)
    akappa = pd.read_csv(SPIN_AKAPPA)
    candidate = pd.read_csv(SPIN_CANDIDATE)
    spin = candidate[candidate["family"].eq("spin")].copy()
    spin = spin.merge(akappa, on="beta", how="inner")
    spin = spin.sort_values("spin_complexity_proxy" if "spin_complexity_proxy" in spin else "knn_edge_disagreement_mean")
    if "spin_complexity_proxy" not in spin.columns:
        spin["spin_complexity_proxy"] = spin["knn_edge_disagreement_mean"]
    spin["complexity_norm"] = norm01(spin["spin_complexity_proxy"].to_numpy(dtype=float))
    spin["A_kappa_norm"] = norm01(spin["A_kappa"].to_numpy(dtype=float))
    spin["beta_label"] = spin["beta"].map(lambda value: f"beta={value:.2f}")
    return dphi, spin


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_stale_vs_corrected(ax: plt.Axes, compare: pd.DataFrame, colors: dict[str, Any]) -> None:
    sub = compare.dropna(subset=["stale_nmstv"]).sort_values("case_order")
    y = np.arange(len(sub))
    for yi, (_, row) in zip(y, sub.iterrows()):
        color = colors.get(str(row["label"]), "black")
        ax.plot([row["stale_nmstv"], row["nmstv"]], [yi, yi], color=color, alpha=0.55, lw=2.0)
        ax.scatter(row["stale_nmstv"], yi, s=36, facecolor="white", edgecolor=color, lw=1.6, zorder=3)
        ax.scatter(row["nmstv"], yi, s=42, color=color, edgecolor="black", linewidth=0.35, zorder=4)
        if str(row["label"]) == "flip eta=0.05":
            ax.annotate(
                "eta=0.05 moved right\nwhen recomputed",
                xy=(row["nmstv"], yi),
                xytext=(0.66, yi + 0.55),
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
                fontsize=8.5,
                ha="left",
            )
    even = compare[compare["label"].eq("rule: even odd")]
    if not even.empty:
        ax.axvline(float(even["nmstv"].iloc[0]), color="#333333", ls=":", lw=1.2, alpha=0.7)
        ax.text(float(even["nmstv"].iloc[0]) + 0.01, -0.65, "even odd", fontsize=8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["short_label"])
    ax.invert_yaxis()
    ax.set_xlabel("NMSTV")
    ax.set_title("A. stale axis vs actual dataset NMSTV")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#444444", label="stale table"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#444444", markeredgecolor="black", label="corrected"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    style_axes(ax)


def plot_corrected_trend(ax: plt.Axes, metrics: pd.DataFrame, expected: pd.DataFrame, colors: dict[str, Any]) -> None:
    trend = metrics[metrics["label"].isin(TREND_LABELS)].copy().sort_values("case_order")
    x = np.arange(len(trend))
    ax.plot(x, trend["nmstv"], color="#222222", lw=1.2, alpha=0.65)
    for xi, (_, row) in zip(x, trend.iterrows()):
        color = colors.get(str(row["label"]), "black")
        marker = "s" if row["source"] == "flip" else "o"
        ax.scatter(xi, row["nmstv"], s=62, marker=marker, color=color, edgecolor="black", linewidth=0.4, zorder=3)
        ax.text(xi, row["nmstv"] + 0.025, f"{row['nmstv']:.3f}", ha="center", va="bottom", fontsize=8)
    if not expected.empty:
        exp = expected.merge(metrics[["label", "case_order"]], on="label", how="left").sort_values("case_order")
        exp_x = [int(metrics.loc[metrics["label"].eq(label), "case_order"].iloc[0]) for label in exp["label"]]
        ax.plot(exp_x, exp["expected_nmstv"], color="#111111", ls="--", lw=1.1, alpha=0.75, label="eta expectation from even odd")
        ax.scatter(exp_x, exp["expected_nmstv"], color="white", edgecolor="#111111", s=38, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(trend["short_label"], rotation=25, ha="right")
    ax.set_ylabel("NMSTV")
    ax.set_title("B. corrected trend recovers rule -> eta -> random")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    style_axes(ax)


def plot_mnist_phase_with_error(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    colors: dict[str, Any],
    *,
    title: str,
    normalized: bool = False,
    annotate: bool = True,
) -> plt.Axes:
    xcol = "complexity_norm" if normalized else "nmstv"
    ycol = "A_kappa_norm" if normalized else "positive_curvature_mass_mean"
    yerr_col = None if normalized else "positive_curvature_mass_sem"
    sub = metrics.sort_values("case_order")
    for _, row in sub.iterrows():
        color = colors.get(str(row["label"]), "black")
        marker = "s" if row["source"] == "flip" else "o"
        yerr = float(row[yerr_col]) if yerr_col and pd.notna(row.get(yerr_col)) else None
        ax.errorbar(
            row[xcol],
            row[ycol],
            yerr=yerr,
            fmt=marker,
            ms=6,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.35,
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
        if annotate:
            ax.text(row[xcol] + 0.012, row[ycol], label_short(row["label"]), fontsize=8.5, va="center")

    trend = sub[sub["label"].isin(TREND_LABELS)].sort_values("case_order")
    ax.plot(trend[xcol], trend[ycol], color="#333333", lw=1.1, alpha=0.55)
    ax.set_xlabel("normalized NMSTV" if normalized else "NMSTV")
    ax.set_ylabel(r"normalized $A_\kappa$" if normalized else r"$A_\kappa$")
    ax.set_title(title)
    style_axes(ax)

    ax2 = ax.twinx()
    err_y = "ref_test_error_norm" if normalized else "ref_test_error_mean"
    ax2.plot(
        trend[xcol],
        trend[err_y],
        color="#111111",
        ls=":",
        lw=1.7,
        marker="x",
        ms=5,
        label="ref test error",
        zorder=2,
    )
    ax2.set_ylabel("normalized ref test error" if normalized else "ref test error")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.legend(loc="lower right", frameon=False, fontsize=8)
    return ax2


def plot_diagnostic(metrics: pd.DataFrame, compare: pd.DataFrame, expected: pd.DataFrame, out_dir: Path) -> Path:
    colors = case_color_map(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0), constrained_layout=True)
    plot_stale_vs_corrected(axes[0], compare, colors)
    plot_corrected_trend(axes[1], metrics, expected, colors)
    plot_mnist_phase_with_error(axes[2], metrics, colors, title="C. phase metric with error metric line")
    fig.suptitle("MNIST NMSTV-axis diagnostic: stale metadata caused the even odd / eta=0.05 inversion", fontsize=13)
    out = out_dir / "fig01_nmstv_axis_diagnostic.png"
    savefig(fig, out)
    return out


def plot_mnist_derivative_panel(ax: plt.Axes, phi: pd.DataFrame, metrics: pd.DataFrame, colors: dict[str, Any]) -> None:
    for _, row in metrics.sort_values("case_order").iterrows():
        sub = phi[(phi["source"].eq(row["source"])) & (phi["group"].eq(row["group"]))].sort_values("radius")
        if sub.empty:
            continue
        color = colors.get(str(row["label"]), "black")
        ls = "--" if row["source"] == "flip" else "-"
        x = sub["radius"].to_numpy(dtype=float)
        y = sub["d_phi_energy_direct_dd_mean"].to_numpy(dtype=float)
        err = sub["d_phi_energy_direct_dd_sem"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, ls=ls, lw=2.0, label=f"{label_short(row['label'])} ({row['nmstv']:.3f})")
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.08, linewidth=0)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$d\phi_E/dd$")
    ax.set_title("A. MNIST direct derivative")
    ax.legend(loc="lower right", frameon=False, fontsize=7.3, ncol=1)
    style_axes(ax)


def plot_spin_derivative_panel(ax: plt.Axes, dphi: pd.DataFrame, spin: pd.DataFrame) -> None:
    sub_dphi = dphi[(dphi["radius"] >= 0.01 - 1e-9) & (dphi["radius"] <= 1.0 + 1e-9)].copy()
    spin_lookup = spin.set_index("beta")
    cmap = plt.colormaps["magma_r"]
    for beta, group in sub_dphi.groupby("beta", sort=True):
        if beta not in spin_lookup.index:
            continue
        cval = float(spin_lookup.loc[beta, "complexity_norm"])
        color = cmap(cval)
        group = group.sort_values("radius")
        ax.plot(group["radius"], group["dphi_energy_dr"], color=color, lw=1.55, alpha=0.88)
    for beta in [float(spin["beta"].min()), float(spin.loc[spin["spin_complexity_proxy"].idxmax(), "beta"]), 0.15, 0.25]:
        close = spin.iloc[(spin["beta"] - beta).abs().argsort()[:1]]
        if close.empty:
            continue
        actual_beta = float(close["beta"].iloc[0])
        curve = sub_dphi[sub_dphi["beta"].eq(actual_beta)].sort_values("radius")
        if curve.empty:
            continue
        last = curve.iloc[-1]
        ax.text(float(last["radius"]) + 0.015, float(last["dphi_energy_dr"]), f"b={actual_beta:.2f}", fontsize=7.2, va="center")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$d\phi_E/dr$")
    ax.set_title("C. 3NN spin direct derivative")
    style_axes(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_label("spin complexity norm", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def plot_spin_phase_panel(ax: plt.Axes, spin: pd.DataFrame) -> None:
    spin = spin.sort_values("spin_complexity_proxy")
    cmap = plt.colormaps["magma_r"]
    colors = cmap(spin["complexity_norm"].to_numpy(dtype=float))
    ax.plot(spin["spin_complexity_proxy"], spin["A_kappa"], color="#333333", lw=1.1, alpha=0.55)
    ax.scatter(spin["spin_complexity_proxy"], spin["A_kappa"], c=colors, s=44, edgecolor="black", linewidth=0.35, zorder=3)
    for _, row in spin.iterrows():
        beta = float(row["beta"])
        if beta in {0.05, 0.15, 0.25}:
            ax.text(row["spin_complexity_proxy"] + 0.003, row["A_kappa"], f"b={beta:.2f}", fontsize=7.2, va="center")
    ax.set_xlabel("3NN spin complexity\nkNN edge disagreement")
    ax.set_ylabel(r"$A_\kappa$")
    ax.set_title("D. 3NN spin phase metric")
    style_axes(ax)


def plot_spin_mnist_composite(metrics: pd.DataFrame, out_dir: Path) -> Path:
    phi = pd.read_csv(MNIST_PHI)
    spin_dphi, spin = load_spin_metrics()
    colors = case_color_map(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), constrained_layout=True)
    plot_mnist_derivative_panel(axes[0, 0], phi, metrics, colors)
    plot_mnist_phase_with_error(axes[0, 1], metrics, colors, title="B. MNIST phase metric + ref test error")
    plot_spin_derivative_panel(axes[1, 0], spin_dphi, spin)
    plot_spin_phase_panel(axes[1, 1], spin)
    fig.suptitle("Direct MNIST eta/rule run and previous 3NN spin comparison", fontsize=14)
    out = out_dir / "fig02_mnist_direct_vs_3nn_spin_composite.png"
    savefig(fig, out)
    return out


def plot_normalized_overlay(metrics: pd.DataFrame, out_dir: Path) -> Path:
    _spin_dphi, spin = load_spin_metrics()
    colors = case_color_map(metrics)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    plot_mnist_phase_with_error(
        ax,
        metrics,
        colors,
        title="Normalized phase metric overlay",
        normalized=True,
        annotate=False,
    )
    spin = spin.sort_values("complexity_norm")
    ax.plot(spin["complexity_norm"], spin["A_kappa_norm"], color="#555555", lw=1.2, alpha=0.65, label="3NN spin")
    ax.scatter(
        spin["complexity_norm"],
        spin["A_kappa_norm"],
        s=38,
        c=plt.colormaps["magma_r"](spin["complexity_norm"].to_numpy(dtype=float)),
        edgecolor="black",
        linewidth=0.3,
        zorder=4,
    )
    for beta, dx, dy in [(0.05, -0.13, -0.055), (0.15, 0.012, -0.035)]:
        close = spin.iloc[(spin["beta"] - beta).abs().argsort()[:1]]
        if close.empty:
            continue
        row = close.iloc[0]
        ax.text(
            row["complexity_norm"] + dx,
            row["A_kappa_norm"] + dy,
            f"spin b={row['beta']:.2f}",
            fontsize=7.2,
            va="center",
        )
    mnist_offsets = {
        "rule: very low tv": (0.018, 0.035),
        "rule: even odd": (0.014, -0.040),
        "flip eta=0.05": (0.014, 0.020),
        "rule: teacher nn": (0.014, 0.000),
        "flip eta=0.15": (0.014, 0.020),
        "flip eta=0.25": (0.014, -0.030),
        "rule: random label": (-0.115, 0.035),
    }
    for _, row in metrics.sort_values("case_order").iterrows():
        dx, dy = mnist_offsets.get(str(row["label"]), (0.012, 0.0))
        ax.text(
            float(row["complexity_norm"]) + dx,
            float(row["A_kappa_norm"]) + dy,
            label_short(row["label"]),
            fontsize=8.3,
            va="center",
        )
    handles = [
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", label="MNIST rule/eta"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#999999", label="3NN spin"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    out = out_dir / "fig03_normalized_phase_overlay_spin_mnist.png"
    savefig(fig, out)
    return out


def write_report(
    out_dir: Path,
    metrics: pd.DataFrame,
    compare: pd.DataFrame,
    expected: pd.DataFrame,
    figures: list[Path],
) -> None:
    eta05 = compare[compare["label"].eq("flip eta=0.05")]
    even = compare[compare["label"].eq("rule: even odd")]
    if not eta05.empty and not even.empty:
        eta05_old = float(eta05["stale_nmstv"].iloc[0])
        eta05_new = float(eta05["nmstv"].iloc[0])
        even_new = float(even["nmstv"].iloc[0])
    else:
        eta05_old = eta05_new = even_new = float("nan")

    trend = metrics[metrics["label"].isin(TREND_LABELS)].sort_values("case_order")
    trend_text = " -> ".join(f"{label_short(row.label)} ({row.nmstv:.3f})" for row in trend.itertuples())
    lines = [
        "# NMSTV Axis Diagnostic And Spin Comparison",
        "",
        "## Main finding",
        "",
        "The even odd / eta=0.05 inversion was caused by stale/intermediate NMSTV metadata in the older complexity-axis table.",
        "When NMSTV is recomputed from the actual direct-run dataset labels used by the sampling/reference rows, eta=0.05 moves to the right of even odd.",
        "",
        f"- stale eta=0.05 NMSTV: `{eta05_old:.6f}`",
        f"- corrected eta=0.05 NMSTV: `{eta05_new:.6f}`",
        f"- corrected even odd NMSTV: `{even_new:.6f}`",
        f"- corrected rule/eta/random trend: `{trend_text}`",
        "",
        "The teacher NN rule is kept as a separate rule family and is not part of the monotone label-flip interpolation.",
        "",
        "## Method notes",
        "",
        "- Corrected MNIST complexity is graph TV NMSTV recomputed on the actual dataset paths in the direct run.",
        "- The eta expectation uses the even odd graph-TV mean and independent label-flip formula `a=2 eta (1-eta)`, `TV_eta ~= TV_even*(1-2a)+a`.",
        "- MNIST phase metric uses direct first derivatives from sampling and finite differences only for the second derivative.",
        "- The error metric line is reference-pool test error, drawn on the MNIST phase panels.",
        "- The 3NN comparison uses the previous 18 beta, 90 dataset, 30 reference dense spin tables.",
        "",
        "## Figures",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{fig}`")
    lines.extend(
        [
            "",
            "## Key tables",
            "",
            "- `corrected_mnist_case_metrics.csv`",
            "- `stale_vs_corrected_nmstv.csv`",
            "- `eta_expected_nmstv_from_even_odd.csv`",
            "- `spin_case_metrics.csv`",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = ensure_dir(OUT_DIR)
    metrics = load_mnist_metrics()
    compare = load_stale_vs_corrected(metrics)
    expected = expected_eta_nmstv(metrics)
    _spin_dphi, spin = load_spin_metrics()

    metrics.to_csv(out_dir / "corrected_mnist_case_metrics.csv", index=False)
    compare.to_csv(out_dir / "stale_vs_corrected_nmstv.csv", index=False)
    expected.to_csv(out_dir / "eta_expected_nmstv_from_even_odd.csv", index=False)
    spin.to_csv(out_dir / "spin_case_metrics.csv", index=False)

    figures = [
        plot_diagnostic(metrics, compare, expected, out_dir),
        plot_spin_mnist_composite(metrics, out_dir),
        plot_normalized_overlay(metrics, out_dir),
    ]
    write_json(
        out_dir / "run_config_resolved.json",
        {
            "inputs": {
                "mnist_complexity": MNIST_COMPLEXITY,
                "mnist_curvature": MNIST_CURVATURE,
                "mnist_phi": MNIST_PHI,
                "old_mnist_axis": OLD_MNIST_AXIS,
                "active_ref_index": ACTIVE_REF_INDEX,
                "eta_ref_index": ETA_REF_INDEX,
                "spin_dphi": SPIN_DPHI,
                "spin_Akappa": SPIN_AKAPPA,
                "spin_candidate": SPIN_CANDIDATE,
            },
            "outputs": [str(fig) for fig in figures],
            "mnist_cases": int(len(metrics)),
            "spin_betas": int(spin["beta"].nunique()),
            "finding": "Older MNIST complexity-axis NMSTV metadata placed eta=0.05 left of even odd; actual direct-run dataset NMSTV places eta=0.05 right of even odd.",
        },
    )
    write_report(out_dir, metrics, compare, expected, figures)
    for fig in figures:
        print(fig)
    print(out_dir / "REPORT.md")


if __name__ == "__main__":
    main()
