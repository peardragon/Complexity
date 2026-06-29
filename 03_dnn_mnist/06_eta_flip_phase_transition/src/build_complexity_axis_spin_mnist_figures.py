#!/usr/bin/env python3
"""Freeze current figures and rebuild spin/MNIST plots on a common complexity axis.

The raw controls point in different semantic directions: spin beta is an
inverse-temperature, while MNIST eta is a label-flip/noise rate. This script
keeps the original artifacts and creates plots using empirical complexity
proxies with a shared increasing-complexity convention.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
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

SOURCE_DISCUSSION_DIR = (
    STAGE_ROOT
    / "figures"
    / "phase_transition_discussion_30ref_eta0p02_0p05_0p15_0p25"
)
OUT_DIR = (
    STAGE_ROOT
    / "figures"
    / "complexity_axis_spin_mnist_30ref_eta0p02_0p05_0p15_0p25"
)
FROZEN_DIR = OUT_DIR / "00_frozen_current_figures"

MNIST_COMBINED = (
    STAGE_ROOT
    / "figures"
    / "combined_dense30ref_rules_eta_phi_energy_r1p0_cpu60_gpu0_eta0p02_0p05_0p15_0p25"
    / "combined_phi_energy_by_group_radius.csv"
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
SPIN_AKAPPA = (
    PROJECT_ROOT
    / "02_dnn"
    / "06_random_gaussian_baseline"
    / "figures"
    / "gaussian_overlay_final_derivative"
    / "measure_search"
    / "positive_curvature_mass_composite_spin_only.csv"
)
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

CASE_COLOR = {
    "adv very low tv": "#1f77b4",
    "adv even odd": "#ff7f0e",
    "adv teacher nn": "#2ca02c",
    "adv random": "#d62728",
    "flip eta 0.02": "#4b006e",
    "flip eta 0.05": "#4b3590",
    "flip eta 0.15": "#1f9e89",
    "flip eta 0.25": "#f4d21b",
}
CASE_STYLE = {
    "adv very low tv": "-",
    "adv even odd": "-",
    "adv teacher nn": "-",
    "adv random": "-",
    "flip eta 0.02": "--",
    "flip eta 0.05": "--",
    "flip eta 0.15": "--",
    "flip eta 0.25": "--",
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


def norm01(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1.0e-15:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def freeze_current_figures() -> pd.DataFrame:
    ensure_dir(FROZEN_DIR)
    candidates = [
        "fig01_raw_phi_E_even_odd_eta_comparison",
        "fig02_smoothed_dphi_dr_even_odd_eta_comparison",
        "fig03_curvature_A_kappa_even_odd_eta_comparison",
        "fig04_gap_metrics_to_even_odd",
        "fig05_derivative_estimator_stability",
        "fig06_qc_noise_derivative_diagnostics",
        "fig07_spin_synthetic_vs_mnist_real_phase_comparison",
    ]
    copied: list[dict[str, Any]] = []
    for stem in candidates:
        for suffix in (".png", ".pdf"):
            src = SOURCE_DISCUSSION_DIR / f"{stem}{suffix}"
            if not src.exists():
                continue
            dst = FROZEN_DIR / src.name
            shutil.copy2(src, dst)
            copied.append({"source": str(src), "frozen_copy": str(dst), "bytes": int(dst.stat().st_size)})
    for name in [
        "REPORT.md",
        "DERIVATIVE_STABILITY_AND_SPIN_COMPARISON_REPORT.md",
        "phase_metric_comparison_spin_vs_mnist.csv",
        "derivative_stability_summary_by_case_method.csv",
    ]:
        src = SOURCE_DISCUSSION_DIR / name
        if src.exists():
            dst = FROZEN_DIR / name
            shutil.copy2(src, dst)
            copied.append({"source": str(src), "frozen_copy": str(dst), "bytes": int(dst.stat().st_size)})
    manifest = pd.DataFrame(copied)
    manifest.to_csv(FROZEN_DIR / "frozen_current_figures_manifest.csv", index=False)
    return manifest


def load_spin_complexity() -> pd.DataFrame:
    candidate = pd.read_csv(SPIN_CANDIDATE)
    ak = pd.read_csv(SPIN_AKAPPA)
    dphi = pd.read_csv(SPIN_DPHI)
    candidate["beta"] = pd.to_numeric(candidate["beta"], errors="coerce")
    candidate["spin_complexity_proxy"] = pd.to_numeric(
        candidate["knn_edge_disagreement_mean"], errors="coerce"
    )
    out = candidate[["beta", "spin_complexity_proxy", "knn_label_autocorrelation_mean"]].merge(
        ak, on="beta", how="inner"
    )
    out["complexity_norm"] = norm01(out["spin_complexity_proxy"])
    out["beta_reversed_norm"] = norm01(-out["beta"])
    out["system"] = "3NN spin"
    # Use a direct-derivative landmark from the published spin derivative table.
    dsmall = dphi[dphi["radius"].between(0.01, 1.0)].copy()
    landmark_rows = []
    for beta, sub in dsmall.groupby("beta"):
        sub = sub.sort_values("radius")
        idx = int(np.nanargmin(sub["dphi_energy_dr"].to_numpy(dtype=float)))
        landmark_rows.append(
            {
                "beta": float(beta),
                "spin_min_dphi_dr": float(sub["dphi_energy_dr"].iloc[idx]),
                "spin_min_dphi_dr_radius": float(sub["radius"].iloc[idx]),
            }
        )
    out = out.merge(pd.DataFrame(landmark_rows), on="beta", how="left")
    return out.sort_values("complexity_norm").reset_index(drop=True)


def load_mnist_complexity() -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(MNIST_COMBINED)
    for col in ["nmstv", "radius", "phi_energy_raw_mean", "phi_energy_raw_sem"]:
        curves[col] = pd.to_numeric(curves[col], errors="coerce")
    curves["complexity_proxy"] = curves["nmstv"]
    meta = (
        curves[["source", "group", "label", "nmstv", "complexity_proxy"]]
        .drop_duplicates()
        .sort_values("complexity_proxy")
        .reset_index(drop=True)
    )
    meta["complexity_norm"] = norm01(meta["complexity_proxy"])
    curves = curves.merge(meta, on=["source", "group", "label", "nmstv", "complexity_proxy"], how="left")
    return curves, meta


def mnist_group_metrics(curves: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, sub in curves.groupby("label", sort=False):
        sub = sub.sort_values("radius")
        x = sub["radius"].to_numpy(dtype=np.float64)
        y = sub["phi_energy_raw_mean"].to_numpy(dtype=np.float64)
        if len(x) < 11:
            continue
        step = float(np.median(np.diff(x)))
        window = min(21, len(x) if len(x) % 2 == 1 else len(x) - 1)
        if window < 5:
            continue
        d1 = savgol_filter(y, window_length=window, polyorder=3, deriv=1, delta=step, mode="interp")
        d2 = savgol_filter(y, window_length=window, polyorder=3, deriv=2, delta=step, mode="interp")
        pos = np.maximum(d2, 0.0)
        a_kappa = trapz(pos, x)
        min_idx = int(np.nanargmin(d1))
        row_meta = meta[meta["label"].eq(label)].iloc[0].to_dict()
        rows.append(
            {
                **row_meta,
                "A_kappa_savgol21_group_mean": a_kappa,
                "min_dphi_dr_savgol21": float(d1[min_idx]),
                "min_dphi_dr_radius_savgol21": float(x[min_idx]),
                "phi_energy_at_d1": float(sub.loc[np.isclose(sub["radius"], 1.0), "phi_energy_raw_mean"].iloc[0])
                if np.isclose(sub["radius"], 1.0).any()
                else float(y[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values("complexity_norm").reset_index(drop=True)


def save_complexity_tables(spin: pd.DataFrame, mnist_metrics: pd.DataFrame) -> None:
    spin.to_csv(OUT_DIR / "spin_complexity_axis_metrics.csv", index=False)
    mnist_metrics.to_csv(OUT_DIR / "mnist_complexity_axis_metrics.csv", index=False)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complexity_convention": "larger complexity_norm means more locally disordered/complex labels within each system",
        "spin_complexity_proxy": "knn_edge_disagreement_mean from candidate_measures_by_beta_spin.csv; beta is also shown reversed for provenance",
        "mnist_complexity_proxy": "NMSTV from combined_phi_energy_by_group_radius.csv",
        "inputs": {
            "source_discussion_dir": SOURCE_DISCUSSION_DIR,
            "mnist_combined": MNIST_COMBINED,
            "spin_candidate": SPIN_CANDIDATE,
            "spin_A_kappa": SPIN_AKAPPA,
            "spin_dphi": SPIN_DPHI,
        },
    }
    (OUT_DIR / "complexity_axis_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def plot_mnist_phi_by_complexity(curves: pd.DataFrame, meta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=180)
    for _, row in meta.sort_values("complexity_norm").iterrows():
        label = str(row["label"])
        sub = curves[curves["label"].eq(label)].sort_values("radius")
        color = CASE_COLOR.get(label, plt.cm.viridis(float(row["complexity_norm"])))
        style = CASE_STYLE.get(label, "-")
        lw = 3.0 if str(row["source"]) == "advanced" else 2.6
        ax.plot(
            sub["radius"],
            sub["phi_energy_raw_mean"],
            linestyle=style,
            color=color,
            lw=lw,
            alpha=0.95,
            label=f"{label}  C={float(row['complexity_norm']):.2f}",
        )
        ax.fill_between(
            sub["radius"].to_numpy(float),
            (sub["phi_energy_raw_mean"] - sub["phi_energy_raw_sem"]).to_numpy(float),
            (sub["phi_energy_raw_mean"] + sub["phi_energy_raw_sem"]).to_numpy(float),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
    ax.axhline(0.0, color="0.25", lw=1.0)
    ax.set_xlabel("radius d")
    ax.set_ylabel("phi_E(d)")
    ax.set_title("MNIST phi_E curves ordered by empirical complexity C")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig01_mnist_phi_energy_by_complexity_axis.{ext}", dpi=220)
    plt.close(fig)


def plot_phase_metrics(spin: pd.DataFrame, mnist_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), dpi=180, constrained_layout=True)
    ax0, ax1, ax2 = axes

    spin_a_norm = norm01(spin["A_kappa"])
    mnist_a_norm = norm01(mnist_metrics["A_kappa_savgol21_group_mean"])
    ax0.plot(
        spin["complexity_norm"],
        spin_a_norm,
        "o-",
        color="#375a9e",
        lw=2.0,
        label="3NN spin normalized A_kappa",
    )
    for idx, row in mnist_metrics.reset_index(drop=True).iterrows():
        color = CASE_COLOR.get(str(row["label"]), "#222222")
        marker = "o" if row["source"] == "advanced" else "s"
        ax0.scatter(
            [row["complexity_norm"]],
            [mnist_a_norm[idx]],
            color=color,
            marker=marker,
            s=60,
            zorder=3,
            label=str(row["label"]),
        )
    ax0.set_xlabel("normalized complexity C")
    ax0.set_ylabel("within-system normalized A_kappa")
    ax0.set_title("Shape comparison after aligning complexity direction")
    ax0.grid(alpha=0.3)

    ax1.plot(spin["beta"], spin["spin_complexity_proxy"], "o-", color="#375a9e", lw=1.8)
    ax1.invert_xaxis()
    ax1.set_xlabel("spin beta (inverted axis)")
    ax1.set_ylabel("spin C proxy")
    ax1.set_title("Spin beta is inverse-temperature")
    ax1.grid(alpha=0.3)

    for _, row in mnist_metrics.iterrows():
        color = CASE_COLOR.get(str(row["label"]), "#222222")
        ax2.scatter(
            [row["complexity_norm"]],
            [row["phi_energy_at_d1"]],
            color=color,
            s=65,
            label=str(row["label"]),
        )
        ax2.text(
            float(row["complexity_norm"]) + 0.012,
            float(row["phi_energy_at_d1"]),
            str(row["label"]).replace("adv ", "").replace("flip ", ""),
            fontsize=8,
            va="center",
        )
    ax2.set_xlabel("MNIST normalized complexity C (NMSTV)")
    ax2.set_ylabel("phi_E(d=1)")
    ax2.set_title("MNIST endpoint energy versus complexity")
    ax2.grid(alpha=0.3)

    handles, labels = ax0.get_legend_handles_labels()
    # Keep the main legend readable by deduplicating labels.
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax0.legend(dedup.values(), dedup.keys(), fontsize=7, ncol=1, frameon=False)
    fig.suptitle("Complexity-axis view: beta/eta/rule controls aligned by empirical complexity")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig02_complexity_axis_phase_metrics.{ext}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=180, constrained_layout=True)
    ax_spin, ax_mnist = axes
    ax_spin.plot(spin["complexity_norm"], spin["A_kappa"], "o-", color="#375a9e", lw=2.0)
    ax_spin.set_xlabel("spin normalized complexity C")
    ax_spin.set_ylabel("spin A_kappa")
    ax_spin.set_title("3NN spin raw A_kappa scale")
    ax_spin.grid(alpha=0.3)

    for _, row in mnist_metrics.iterrows():
        color = CASE_COLOR.get(str(row["label"]), "#222222")
        marker = "o" if row["source"] == "advanced" else "s"
        ax_mnist.scatter(
            [row["complexity_norm"]],
            [row["A_kappa_savgol21_group_mean"]],
            color=color,
            marker=marker,
            s=60,
        )
        ax_mnist.text(
            float(row["complexity_norm"]) + 0.012,
            float(row["A_kappa_savgol21_group_mean"]),
            str(row["label"]).replace("adv ", "").replace("flip ", ""),
            fontsize=8,
            va="center",
        )
    ax_mnist.set_xlabel("MNIST normalized complexity C")
    ax_mnist.set_ylabel("MNIST A_kappa (SG21 mean curve)")
    ax_mnist.set_title("MNIST raw A_kappa scale")
    ax_mnist.grid(alpha=0.3)
    fig.suptitle("Raw A_kappa scales are not directly comparable across systems")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig03_complexity_axis_Akappa_separate_scales.{ext}", dpi=220)
    plt.close(fig)


def write_report(frozen_manifest: pd.DataFrame, spin: pd.DataFrame, mnist_metrics: pd.DataFrame) -> None:
    lines = [
        "# Complexity-Axis Spin/MNIST Figures",
        "",
        "This bundle freezes the current discussion figures and rebuilds the comparison on an empirical complexity axis.",
        "",
        "## Complexity Convention",
        "",
        "- Spin: `C_spin = knn_edge_disagreement_mean`, normalized within the spin beta sweep.",
        "- MNIST: `C_mnist = NMSTV`, normalized within the current four-rule + four-eta panel.",
        "- Larger normalized `C` means more locally disordered/complex labels. This avoids comparing raw `beta` and `eta` directions directly.",
        "",
        "## Frozen Inputs",
        "",
        f"- Frozen file count: `{len(frozen_manifest)}`",
        f"- Frozen directory: `{FROZEN_DIR}`",
        "",
        "## Spin Complexity Range",
        "",
        f"- beta range: `{spin['beta'].min():.3g}` to `{spin['beta'].max():.3g}`",
        f"- spin C proxy range: `{spin['spin_complexity_proxy'].min():.6g}` to `{spin['spin_complexity_proxy'].max():.6g}`",
        "",
        "## MNIST Complexity Table",
        "",
        "| label | source | NMSTV | C_norm | A_kappa(SG21 mean curve) | phi_E(d=1) |",
        "| ----- | ------ | ----- | ------ | ------------------------ | ---------- |",
    ]
    for _, row in mnist_metrics.iterrows():
        lines.append(
            f"| {row['label']} | {row['source']} | {row['nmstv']:.6g} | "
            f"{row['complexity_norm']:.3f} | {row['A_kappa_savgol21_group_mean']:.6g} | "
            f"{row['phi_energy_at_d1']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `fig01_mnist_phi_energy_by_complexity_axis.png`",
            "- `fig02_complexity_axis_phase_metrics.png`",
            "- `fig03_complexity_axis_Akappa_separate_scales.png`",
            "- `spin_complexity_axis_metrics.csv`",
            "- `mnist_complexity_axis_metrics.csv`",
            "- `00_frozen_current_figures/frozen_current_figures_manifest.csv`",
            "",
        ]
    )
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUT_DIR)
    frozen = freeze_current_figures()
    spin = load_spin_complexity()
    curves, meta = load_mnist_complexity()
    mnist_metrics = mnist_group_metrics(curves, meta)
    save_complexity_tables(spin, mnist_metrics)
    plot_mnist_phi_by_complexity(curves, meta)
    plot_phase_metrics(spin, mnist_metrics)
    write_report(frozen, spin, mnist_metrics)
    print(f"[done] wrote complexity-axis figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
