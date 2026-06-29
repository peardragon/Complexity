#!/usr/bin/env python3
"""Compare MNIST advanced rules, eta flips, and previous spin 3NN outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
SPIN_ROOT = Path("/home/bjyong/Complexity/local_project/02_dnn")

ADV_RUN = LOCAL_ROOT / "04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref"
FLIP_RUN = LOCAL_ROOT / "06_eta_flip_phase_transition/raw_outputs/eta_reference_phi_advanced_4eta_90ref_r0p1_to_1p0_step0p05_n1024_cpu60_gpu0"
FLIP_CURV_DIR = LOCAL_ROOT / "06_eta_flip_phase_transition/figures/eta_positive_curvature_mass_advanced_90ref_r1p0_n1024_cpu60_gpu0"
FLIP_GRAPH_RUN = LOCAL_ROOT / "06_eta_flip_phase_transition/raw_outputs/eta_sweep_pilot_cpu35_gpu0"

SPIN_ABS = SPIN_ROOT / "05_proxy_local_entropy/raw_outputs/18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense/summary_tables/absolute_phi_by_beta_radius.csv"
SPIN_DPHI = SPIN_ROOT / "05_proxy_local_entropy/raw_outputs/18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense/summary_tables/dphi_dr_by_beta_radius.csv"
SPIN_A = SPIN_ROOT / "06_random_gaussian_baseline/figures/gaussian_overlay_final_derivative/measure_search/positive_curvature_mass_composite_spin_only.csv"

OUT_DIR = LOCAL_ROOT / "06_eta_flip_phase_transition/figures/all_rules_all_flips_spin_comparison_90ref_r1p0"

ADV_LABELS = {
    "very_low_tv_spectral_teacher": "rule: very low tv",
    "real_even_odd": "rule: even odd",
    "teacher_nn": "rule: teacher nn",
    "random_label": "rule: random label",
}
ADV_NMSTV = {
    "very_low_tv_spectral_teacher": 0.3245703473792008,
    "real_even_odd": 0.4932864276461805,
    "teacher_nn": 0.6843772639598127,
    "random_label": 0.985558573825462,
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def radius_grid_summary(values: pd.Series) -> dict[str, Any]:
    radii = np.array(sorted(pd.to_numeric(values, errors="coerce").dropna().unique()), dtype=float)
    if radii.size <= 1:
        steps = np.array([], dtype=float)
    else:
        steps = np.round(np.diff(radii), 12)
    unique_steps = sorted(set(float(x) for x in steps))
    return {
        "radius_count": int(radii.size),
        "radius_min": float(radii.min()) if radii.size else np.nan,
        "radius_max": float(radii.max()) if radii.size else np.nan,
        "radius_step_min": float(steps.min()) if steps.size else np.nan,
        "radius_step_max": float(steps.max()) if steps.size else np.nan,
        "radius_unique_steps": ",".join(f"{x:g}" for x in unique_steps[:8]),
    }


def eta_nmstv_map(etas: list[float]) -> dict[float, float]:
    out: dict[float, float] = {}
    path = FLIP_GRAPH_RUN / "summary_by_eta_k.csv"
    if path.exists():
        graph = pd.read_csv(path)
        graph["eta"] = pd.to_numeric(graph["eta"], errors="coerce")
        graph["k"] = pd.to_numeric(graph["k"], errors="coerce")
        graph["knn_nmstv_mean"] = pd.to_numeric(graph["knn_nmstv_mean"], errors="coerce")
        graph = graph[graph["k"].eq(3)].copy()
        for eta in etas:
            idx = (graph["eta"] - float(eta)).abs().idxmin()
            out[float(eta)] = float(graph.loc[idx, "knn_nmstv_mean"])
    for eta in etas:
        out.setdefault(float(eta), float(eta))
    return out


def load_mnist_groups(d_min: float, d_max: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    adv_summary = pd.read_csv(ADV_RUN / "06_results_figures/phi_energy_by_rule_radius.csv")
    adv_units = pd.read_csv(ADV_RUN / "05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv")
    flip_summary = pd.read_csv(FLIP_RUN / "06_results_figures/eta_reference_phi_by_eta_radius.csv")
    flip_units = pd.read_csv(FLIP_RUN / "05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv")

    adv_summary["radius"] = pd.to_numeric(adv_summary["radius"], errors="coerce")
    adv_summary = adv_summary[(adv_summary["radius"] >= d_min - 1e-12) & (adv_summary["radius"] <= d_max + 1e-12)].copy()
    adv_summary = adv_summary.rename(columns={"phi_energy_raw": "phi_energy_raw_mean"})
    adv_summary["source"] = "mnist_rule"
    adv_summary["case_id"] = adv_summary["rule"].astype(str)
    adv_summary["case_label"] = adv_summary["rule"].map(lambda x: ADV_LABELS.get(str(x), str(x)))
    adv_summary["complexity_axis"] = adv_summary["rule"].map(lambda x: ADV_NMSTV.get(str(x), np.nan))
    adv_summary["line_style"] = "solid"
    adv_summary["basis"] = "90ref_1024"

    for col in ["radius", "ref_id", "phi_energy_raw", "d_phi_energy_raw_dd_unit"]:
        if col in adv_units:
            adv_units[col] = pd.to_numeric(adv_units[col], errors="coerce")
    adv_units = adv_units[(adv_units["radius"] >= d_min - 1e-12) & (adv_units["radius"] <= d_max + 1e-12)].copy()
    adv_units["source"] = "mnist_rule"
    adv_units["case_id"] = adv_units["rule"].astype(str)
    adv_units["case_label"] = adv_units["rule"].map(lambda x: ADV_LABELS.get(str(x), str(x)))
    adv_units["complexity_axis"] = adv_units["rule"].map(lambda x: ADV_NMSTV.get(str(x), np.nan))
    adv_units["ref_key"] = adv_units["source"].astype(str) + ":" + adv_units["case_id"].astype(str) + ":" + adv_units["ref_id"].astype(str)

    flip_summary["eta"] = pd.to_numeric(flip_summary["eta"], errors="coerce")
    etas = sorted(flip_summary["eta"].dropna().unique().tolist())
    nmstv = eta_nmstv_map(etas)
    flip_summary["radius"] = pd.to_numeric(flip_summary["radius"], errors="coerce")
    flip_summary = flip_summary[(flip_summary["radius"] >= d_min - 1e-12) & (flip_summary["radius"] <= d_max + 1e-12)].copy()
    flip_summary["source"] = "mnist_flip"
    flip_summary["case_id"] = flip_summary["eta"].map(lambda x: f"eta_{float(x):.2f}")
    flip_summary["case_label"] = flip_summary["eta"].map(lambda x: f"flip eta={float(x):.2f}")
    flip_summary["complexity_axis"] = flip_summary["eta"].map(lambda x: nmstv[float(x)])
    flip_summary["line_style"] = "dashed"
    flip_summary["basis"] = "90ref_1024"

    for col in ["eta", "radius", "ref_id", "phi_energy_raw", "d_phi_energy_raw_dd_unit"]:
        if col in flip_units:
            flip_units[col] = pd.to_numeric(flip_units[col], errors="coerce")
    flip_units = flip_units[(flip_units["radius"] >= d_min - 1e-12) & (flip_units["radius"] <= d_max + 1e-12)].copy()
    flip_units["source"] = "mnist_flip"
    flip_units["case_id"] = flip_units["eta"].map(lambda x: f"eta_{float(x):.2f}")
    flip_units["case_label"] = flip_units["eta"].map(lambda x: f"flip eta={float(x):.2f}")
    flip_units["complexity_axis"] = flip_units["eta"].map(lambda x: nmstv[float(x)])
    flip_units["ref_key"] = flip_units["source"].astype(str) + ":" + flip_units["case_id"].astype(str) + ":" + flip_units["ref_id"].astype(str)

    group_cols = ["source", "case_id", "case_label", "complexity_axis", "line_style", "basis", "radius", "n_units"]
    summary = pd.concat(
        [
            adv_summary[group_cols + ["phi_energy_raw_mean", "phi_energy_raw_sem"]],
            flip_summary[group_cols + ["phi_energy_raw_mean", "phi_energy_raw_sem"]],
        ],
        ignore_index=True,
    )

    unit_cols = ["source", "case_id", "case_label", "complexity_axis", "ref_key", "ref_id", "radius", "phi_energy_raw"]
    units = pd.concat(
        [
            adv_units[unit_cols],
            flip_units[unit_cols],
        ],
        ignore_index=True,
    )
    return summary.sort_values(["source", "complexity_axis", "case_id", "radius"]), units


def group_derivatives(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source, case_id, case_label, complexity_axis, ref_key), sub in units.groupby(
        ["source", "case_id", "case_label", "complexity_axis", "ref_key"], sort=False
    ):
        sub = sub.sort_values("radius").drop_duplicates("radius")
        if len(sub) < 3:
            continue
        x = sub["radius"].to_numpy(dtype=float)
        y = sub["phi_energy_raw"].to_numpy(dtype=float)
        d1 = np.gradient(y, x)
        d2 = np.gradient(d1, x)
        for r, yy, g, k in zip(x, y, d1, d2):
            rows.append(
                {
                    "source": source,
                    "case_id": case_id,
                    "case_label": case_label,
                    "complexity_axis": float(complexity_axis),
                    "ref_key": ref_key,
                    "radius": float(r),
                    "phi_energy_raw": float(yy),
                    "d_phi_energy_raw_dd": float(g),
                    "d2_phi_energy_raw_dd2": float(k),
                    "positive_curvature": float(max(k, 0.0)),
                }
            )
    ref_curve = pd.DataFrame(rows)
    summary = (
        ref_curve.groupby(["source", "case_id", "case_label", "complexity_axis", "radius"], as_index=False)
        .agg(
            n_refs=("ref_key", "nunique"),
            phi_energy_raw_mean=("phi_energy_raw", "mean"),
            phi_energy_raw_sem=("phi_energy_raw", sem),
            d_phi_energy_raw_dd_mean=("d_phi_energy_raw_dd", "mean"),
            d_phi_energy_raw_dd_sem=("d_phi_energy_raw_dd", sem),
            d2_phi_energy_raw_dd2_mean=("d2_phi_energy_raw_dd2", "mean"),
            d2_phi_energy_raw_dd2_sem=("d2_phi_energy_raw_dd2", sem),
            positive_curvature_mean=("positive_curvature", "mean"),
        )
        .sort_values(["source", "complexity_axis", "case_id", "radius"])
    )
    return ref_curve, summary


def curvature_mass_by_case(ref_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, case_id, case_label, complexity_axis, ref_key), sub in ref_curve.groupby(
        ["source", "case_id", "case_label", "complexity_axis", "ref_key"], sort=False
    ):
        sub = sub.sort_values("radius")
        x = sub["radius"].to_numpy(dtype=float)
        y = sub["positive_curvature"].to_numpy(dtype=float)
        if hasattr(np, "trapezoid"):
            mass = float(np.trapezoid(y, x))
        else:
            mass = float(np.trapz(y, x))
        rows.append(
            {
                "source": source,
                "case_id": case_id,
                "case_label": case_label,
                "complexity_axis": float(complexity_axis),
                "ref_key": ref_key,
                "positive_curvature_mass": mass,
            }
        )
    ref_metrics = pd.DataFrame(rows)
    return (
        ref_metrics.groupby(["source", "case_id", "case_label", "complexity_axis"], as_index=False)
        .agg(
            n_refs=("ref_key", "nunique"),
            positive_curvature_mass_mean=("positive_curvature_mass", "mean"),
            positive_curvature_mass_sd=("positive_curvature_mass", "std"),
            positive_curvature_mass_sem=("positive_curvature_mass", sem),
        )
        .sort_values(["source", "complexity_axis", "case_id"])
    )


def spin_summaries(d_min: float, d_max: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spin_abs = pd.read_csv(SPIN_ABS)
    spin_dphi = pd.read_csv(SPIN_DPHI)
    spin_a = pd.read_csv(SPIN_A)
    for df in (spin_abs, spin_dphi):
        df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
        df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    spin_abs_window = spin_abs[(spin_abs["radius"] >= d_min - 1e-12) & (spin_abs["radius"] <= d_max + 1e-12)].copy()
    spin_dphi_window = spin_dphi[(spin_dphi["radius"] >= d_min - 1e-12) & (spin_dphi["radius"] <= d_max + 1e-12)].copy()
    spin_a["beta"] = pd.to_numeric(spin_a["beta"], errors="coerce")
    spin_a["A_kappa"] = pd.to_numeric(spin_a["A_kappa"], errors="coerce")
    return spin_abs_window, spin_dphi_window, spin_a


def normalize_series(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    lo = float(v.min())
    hi = float(v.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-15:
        return pd.Series(np.zeros(len(v)), index=v.index)
    return (v - lo) / (hi - lo)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    tmp = df.copy()
    for col in tmp.columns:
        tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else str(x))
    widths = {
        col: max(len(str(col)), int(tmp[col].map(len).max()) if len(tmp) else 0)
        for col in tmp.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in tmp.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in tmp.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in tmp.columns) + " |"
        for _, row in tmp.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def write_data_state(summary: pd.DataFrame, spin_abs: pd.DataFrame, spin_dphi: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    adv_status = json_load(ADV_RUN / "SAMPLING_STATUS.json")
    flip_status = json_load(FLIP_RUN / "SAMPLING_STATUS.json")
    adv_cfg = json_load(ADV_RUN / "run_config_resolved.json")
    flip_cfg = json_load(FLIP_RUN / "run_config_resolved.json")
    spin_abs_full = pd.read_csv(SPIN_ABS)
    spin_abs_full["beta"] = pd.to_numeric(spin_abs_full["beta"], errors="coerce")
    spin_abs_full["radius"] = pd.to_numeric(spin_abs_full["radius"], errors="coerce")

    dense_status_path = LOCAL_ROOT / "06_eta_flip_phase_transition/raw_outputs/eta_reference_phi_dense_eta_ref1_d1_n1024_cpu35_gpu0/SAMPLING_STATUS.json"
    dense_status = json_load(dense_status_path) if dense_status_path.exists() else {}

    rows = []
    rows.append(
        {
            "system": "MNIST 4-rule advanced",
            "cases": "4 rules",
            "case_values": ",".join(map(str, adv_status.get("rules", []))),
            "unit_count": adv_status.get("completed_units"),
            "expected_unit_count": adv_status.get("expected_units"),
            "reference_count_basis": "90 refs per rule/radius",
            "samples_per_ref_radius": adv_status.get("samples_per_ref_radius"),
            "save_unit_samples_npz": adv_status.get("save_unit_samples_npz"),
            "derivative_status": "posthoc finite difference from saved phi(d), not sampler-stored dlogZ/dd",
            "analysis_basis": "diagnostic; QC not used to skip sampling",
            **radius_grid_summary(pd.Series(adv_status.get("radii", []))),
            "source_path": str(ADV_RUN),
        }
    )
    rows.append(
        {
            "system": "MNIST label-flip advanced",
            "cases": "4 eta flips at 90ref",
            "case_values": ",".join(map(str, flip_status.get("etas", []))),
            "unit_count": flip_status.get("completed_units"),
            "expected_unit_count": flip_status.get("expected_units"),
            "reference_count_basis": "90 refs per eta/radius",
            "samples_per_ref_radius": flip_status.get("samples_per_ref_radius"),
            "save_unit_samples_npz": bool((flip_cfg.get("outputs") or {}).get("save_unit_samples_npz", False)),
            "derivative_status": "posthoc finite difference from saved phi(d), not sampler-stored dlogZ/dd",
            "analysis_basis": "complete for d<=1.0; split diagnostic max exceeds old strict 0.004 gate",
            **radius_grid_summary(pd.Series(flip_status.get("radii", []))),
            "source_path": str(FLIP_RUN),
        }
    )
    if dense_status:
        rows.append(
            {
                "system": "MNIST label-flip dense eta smoke",
                "cases": "6 eta flips at ref1",
                "case_values": ",".join(map(str, dense_status.get("etas", []))),
                "unit_count": dense_status.get("completed_units"),
                "expected_unit_count": dense_status.get("expected_units"),
                "reference_count_basis": "1 ref per eta/radius",
                "samples_per_ref_radius": dense_status.get("samples_per_ref_radius"),
                "save_unit_samples_npz": False,
                "derivative_status": "insufficient/ref1 exploratory; not comparable to 90ref curves",
                "analysis_basis": "orientation only",
                **radius_grid_summary(pd.Series(dense_status.get("radii", []))),
                "source_path": str(dense_status_path.parent),
            }
        )
    rows.append(
        {
            "system": "previous 3NN spin",
            "cases": "18 beta cells",
            "case_values": ",".join(f"{b:g}" for b in sorted(spin_abs_full["beta"].dropna().unique())),
            "unit_count": int(len(spin_abs_full)),
            "expected_unit_count": int(len(spin_abs_full)),
            "reference_count_basis": "2700 aggregate per beta/radius in summary table",
            "samples_per_ref_radius": "stored in upstream shell pool, not explicit in proxy summary",
            "save_unit_samples_npz": "not inspected in retained proxy summary",
            "derivative_status": "first derivative stored as mean_dlogZ_inf_full_dr / P; second derivative finite difference downstream",
            "analysis_basis": "all proxy summary rows claim=pass",
            **radius_grid_summary(spin_abs_full["radius"]),
            "source_path": str(SPIN_ABS.parent),
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "data_state_comparison.csv", index=False)
    return out


def collect_available_flip_runs(out_dir: Path) -> pd.DataFrame:
    rows = []
    base = LOCAL_ROOT / "06_eta_flip_phase_transition/raw_outputs"
    for root in sorted(base.glob("eta_reference_phi*")):
        phi_path = root / "06_results_figures/eta_reference_phi_by_eta_radius.csv"
        if not phi_path.exists():
            continue
        status_path = root / "SAMPLING_STATUS.json"
        status = json_load(status_path) if status_path.exists() else {}
        phi = pd.read_csv(phi_path)
        eta_values = sorted(pd.to_numeric(phi.get("eta"), errors="coerce").dropna().unique()) if "eta" in phi else []
        radius_values = sorted(pd.to_numeric(phi.get("radius"), errors="coerce").dropna().unique()) if "radius" in phi else []
        n_units_values = sorted(pd.to_numeric(phi.get("n_units"), errors="coerce").dropna().unique()) if "n_units" in phi else []
        if "advanced_4eta_90ref" in root.name:
            role = "main_90ref"
        elif "promoted_4eta_10ref" in root.name:
            role = "support_10ref"
        elif "dense_eta_ref1" in root.name:
            role = "support_dense_eta_ref1"
        elif "4eta_3ref" in root.name:
            role = "support_3ref_n128"
        elif "4eta_1ref" in root.name:
            role = "support_1ref"
        elif "unit_timing" in root.name:
            role = "timing_only"
        else:
            role = "other"
        rows.append(
            {
                "run_name": root.name,
                "role": role,
                "status": status.get("status", ""),
                "completed_units": status.get("completed_units", ""),
                "expected_units": status.get("expected_units", ""),
                "status_ref_count": status.get("ref_count", ""),
                "status_samples_per_ref_radius": status.get("samples_per_ref_radius", ""),
                "eta_count": len(eta_values),
                "eta_values": ",".join(f"{float(v):g}" for v in eta_values),
                "radius_count": len(radius_values),
                "radius_min": min(radius_values) if radius_values else np.nan,
                "radius_max": max(radius_values) if radius_values else np.nan,
                "n_units_values": ",".join(str(int(v)) for v in n_units_values),
                "phi_table": str(phi_path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "available_flip_phi_runs.csv", index=False)
    return out


def load_flip_summary_for_plot(root: Path, d_min: float, d_max: float) -> pd.DataFrame:
    df = pd.read_csv(root / "06_results_figures/eta_reference_phi_by_eta_radius.csv")
    df["eta"] = pd.to_numeric(df["eta"], errors="coerce")
    df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
    value_col = "phi_energy_raw_mean" if "phi_energy_raw_mean" in df.columns else "phi_energy_raw"
    if value_col not in df.columns:
        raise ValueError(f"no phi energy column in {root}")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[(df["radius"] >= d_min - 1e-12) & (df["radius"] <= d_max + 1e-12)].copy()
    df = df.rename(columns={value_col: "phi_energy_raw_mean"})
    return df


def plot_all_available_flips(summary: pd.DataFrame, inventory: pd.DataFrame, out_dir: Path, d_min: float, d_max: float) -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 6.8), dpi=220)
    rule_summary = summary[summary["source"].eq("mnist_rule")].copy()
    rule_colors = dict(zip(sorted(rule_summary["case_id"].unique()), plt.get_cmap("tab10")(np.linspace(0.0, 0.36, 4))))
    for case_id, sub in rule_summary.groupby("case_id", sort=False):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["phi_energy_raw_mean"], lw=2.9, color=rule_colors[case_id], label=str(sub["case_label"].iloc[0]))

    eta_norm = plt.Normalize(0.0, 0.5)
    eta_cmap = plt.get_cmap("viridis")
    role_style = {
        "main_90ref": {"lw": 2.5, "ls": "--", "alpha": 0.98, "marker": None, "zorder": 5},
        "support_10ref": {"lw": 1.6, "ls": "-.", "alpha": 0.68, "marker": None, "zorder": 4},
        "support_dense_eta_ref1": {"lw": 1.25, "ls": ":", "alpha": 0.72, "marker": "o", "zorder": 3},
        "support_3ref_n128": {"lw": 1.1, "ls": ":", "alpha": 0.45, "marker": "x", "zorder": 2},
        "support_1ref": {"lw": 1.1, "ls": ":", "alpha": 0.42, "marker": "+", "zorder": 2},
        "timing_only": {"lw": 0.0, "ls": "None", "alpha": 0.0, "marker": None, "zorder": 0},
        "other": {"lw": 0.9, "ls": ":", "alpha": 0.30, "marker": None, "zorder": 1},
    }
    seen_labels: set[str] = set()
    for row in inventory.sort_values(["role", "run_name"]).to_dict("records"):
        role = str(row["role"])
        if role == "timing_only":
            continue
        root = Path(str(row["phi_table"])).parents[2]
        try:
            df = load_flip_summary_for_plot(root, d_min, d_max)
        except Exception:
            continue
        if df.empty:
            continue
        style = role_style.get(role, role_style["other"])
        for eta, sub in df.groupby("eta", sort=True):
            sub = sub.sort_values("radius")
            color = eta_cmap(eta_norm(float(eta)))
            label = f"{role}: eta={float(eta):.2f}"
            use_label = label if label not in seen_labels and role in {"main_90ref", "support_10ref", "support_dense_eta_ref1"} else None
            if use_label:
                seen_labels.add(label)
            ax.plot(
                sub["radius"],
                sub["phi_energy_raw_mean"],
                color=color,
                lw=float(style["lw"]),
                ls=str(style["ls"]),
                alpha=float(style["alpha"]),
                marker=style["marker"],
                ms=3.2 if style["marker"] else 0.0,
                label=use_label,
                zorder=int(style["zorder"]),
            )
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xlim(d_min - 0.02, d_max + 0.04)
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$\phi_E(d)=\log Z_{\infty,\mathrm{full}}/P$")
    ax.set_title("All phi-bearing label-flip runs overlaid with 4 MNIST rules")
    ax.grid(True, color="0.90", linewidth=0.65)
    ax.text(
        0.012,
        0.018,
        "Flip support runs are not precision-matched: 90ref dashed, 10ref dash-dot, ref1/dense dotted markers.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="0.22",
    )
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.005))
    out = out_dir / "fig04_all_available_flip_phi_runs_with_rules.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def write_report(
    out_dir: Path,
    data_state: pd.DataFrame,
    flip_inventory: pd.DataFrame,
    summary: pd.DataFrame,
    deriv_summary: pd.DataFrame,
    mass: pd.DataFrame,
    spin_a: pd.DataFrame,
) -> None:
    rule_range = (
        summary.groupby("source")["phi_energy_raw_mean"]
        .agg(["min", "max"])
        .reset_index()
        .to_string(index=False)
    )
    mass_preview = mass.sort_values(["source", "complexity_axis"])[
        ["source", "case_label", "n_refs", "positive_curvature_mass_mean", "positive_curvature_mass_sem"]
    ].to_string(index=False)
    spin_preview = spin_a.to_string(index=False)
    lines = [
        "# All Rules + All 90ref Flips vs Previous Spin 3NN: Data-State Audit",
        "",
        "## Scope",
        "",
        "- MNIST 4-rule advanced: 4 rules, 90 references, `n=1024`, radius `0.10..2.50` step `0.05`; figures here use the common `0.10..1.00` window.",
        "- MNIST flip advanced: all flip cases with 90-reference phi sampling currently available: eta `0.25,0.30,0.35,0.40`, radius `0.10..1.00` step `0.05`, `n=1024`.",
        "- Dense eta/ref1 flip outputs exist, but are not mixed into the main 90ref figure because their reference count and radius support are not comparable.",
        "- Previous spin 3NN comparison uses `02_dnn/05_proxy_local_entropy/.../18_beta_cell_90_dataset_30_reference/d_0.01_to_2.50_dense` and the spin-only positive-curvature-mass figure/table.",
        "",
        "## Key Findings",
        "",
        "- In one common plot, the flip eta curves sit far below the MNIST rule curves in raw `phi_E(d)` over `d<=1.0`; the separation is much larger than the rule-to-rule spread.",
        "- MNIST rule/flip first derivatives in these retained outputs are posthoc finite differences along radius; they are not sampler-stored `dlogZ/dd` values.",
        "- Previous spin 3NN stores a direct first derivative column, `mean_dlogZ_inf_full_dr`, and `dphi_energy_dr`; its second-derivative/curvature-style analyses are downstream finite differences from that stored first derivative.",
        "- Spin has much denser radius support (`0.01` spacing, 250 radii, `0.01..2.50`) and many more aggregate references per beta/radius (`2700`) than current MNIST flip (`19` radii, 90 refs).",
        "- Therefore, phase-like curvature mass is qualitatively comparable but not precision-matched: MNIST flip `A_kappa` is based on finite-difference first derivatives and a coarser `0.05` radius grid.",
        "",
        "## Data State Table",
        "",
        markdown_table(data_state),
        "",
        "## Available Flip Phi Runs",
        "",
        markdown_table(flip_inventory[["run_name", "role", "status", "completed_units", "expected_units", "eta_values", "radius_count", "radius_min", "radius_max", "n_units_values"]]),
        "",
        "## Phi Range by MNIST Source",
        "",
        "```text",
        rule_range,
        "```",
        "",
        "## MNIST Positive Curvature Mass by Case",
        "",
        "```text",
        mass_preview,
        "```",
        "",
        "## Previous Spin A_kappa",
        "",
        "```text",
        spin_preview,
        "```",
        "",
        "## Generated Artifacts",
        "",
        "- `fig01_all_rules_all_flips_phi_energy.png`: all 4 MNIST rules plus all 90ref flip eta curves on common `d<=1.0` axes.",
        "- `fig02_all_rules_all_flips_derivative_curvature.png`: finite-difference first and second derivative summaries for MNIST groups.",
        "- `fig03_mnist_flip_vs_spin_phase_metrics.png`: normalized curvature-mass/order-parameter comparison against previous spin 3NN.",
        "- `fig04_all_available_flip_phi_runs_with_rules.png`: all phi-bearing flip runs overlaid with 4 rules; precision classes are visually separated.",
        "- `data_state_comparison.csv`: grid/ref/sample/derivative-state audit table.",
        "- `available_flip_phi_runs.csv`: inventory of every flip phi run currently found under `06_eta_flip_phase_transition/raw_outputs`.",
        "- `all_mnist_phi_groups.csv`, `all_mnist_derivative_groups.csv`, `all_mnist_positive_curvature_mass.csv`: numeric tables backing the figures.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_phi(summary: pd.DataFrame, units: pd.DataFrame, out_dir: Path) -> Path:
    cmap_rule = plt.get_cmap("tab10")
    cases = summary[["source", "case_id", "case_label", "complexity_axis"]].drop_duplicates().reset_index(drop=True)
    rule_cases = cases[cases["source"].eq("mnist_rule")].sort_values("complexity_axis")
    flip_cases = cases[cases["source"].eq("mnist_flip")].sort_values("complexity_axis")
    color_map: dict[str, Any] = {}
    for i, row in enumerate(rule_cases.to_dict("records")):
        color_map[str(row["case_id"])] = cmap_rule(i)
    flip_norm = plt.Normalize(float(flip_cases["complexity_axis"].min()), float(flip_cases["complexity_axis"].max()))
    flip_cmap = plt.get_cmap("viridis")
    for row in flip_cases.to_dict("records"):
        color_map[str(row["case_id"])] = flip_cmap(flip_norm(float(row["complexity_axis"])))

    fig, ax = plt.subplots(figsize=(11.0, 6.4), dpi=220)
    for case_id, sub in units.groupby("case_id", sort=False):
        color = color_map[str(case_id)]
        ls = "-" if sub["source"].iloc[0] == "mnist_rule" else "--"
        alpha = 0.065 if sub["source"].iloc[0] == "mnist_rule" else 0.045
        for _, sref in sub.groupby("ref_key", sort=False):
            sref = sref.sort_values("radius")
            ax.plot(sref["radius"], sref["phi_energy_raw"], color=color, lw=0.45, alpha=alpha, ls=ls)
    label_rows = []
    for case_id, sub in summary.groupby("case_id", sort=False):
        sub = sub.sort_values("radius")
        color = color_map[str(case_id)]
        ls = "-" if sub["source"].iloc[0] == "mnist_rule" else "--"
        ax.plot(sub["radius"], sub["phi_energy_raw_mean"], color=color, lw=2.6, ls=ls)
        ax.fill_between(
            sub["radius"].to_numpy(dtype=float),
            (sub["phi_energy_raw_mean"] - 1.96 * sub["phi_energy_raw_sem"]).to_numpy(dtype=float),
            (sub["phi_energy_raw_mean"] + 1.96 * sub["phi_energy_raw_sem"]).to_numpy(dtype=float),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        last = sub.iloc[-1]
        label_rows.append({"label": str(last["case_label"]), "x": float(last["radius"]) + 0.025, "y": float(last["phi_energy_raw_mean"]), "color": color})
    label_rows = sorted(label_rows, key=lambda r: r["y"])
    min_gap = 0.005
    prev = None
    for row in label_rows:
        if prev is not None and row["y"] - prev < min_gap:
            row["y"] = prev + min_gap
        prev = row["y"]
        ax.text(row["x"], row["y"], row["label"], va="center", color=row["color"], fontsize=8.6)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlim(0.08, 1.18)
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$\phi_E(d)=\log Z_{\infty,\mathrm{full}}/P$")
    ax.set_title("MNIST advanced 4 rules + all 90ref label flips")
    ax.grid(True, color="0.90", linewidth=0.65)
    out = out_dir / "fig01_all_rules_all_flips_phi_energy.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_derivatives(deriv_summary: pd.DataFrame, out_dir: Path) -> Path:
    cases = deriv_summary[["source", "case_id", "case_label", "complexity_axis"]].drop_duplicates().reset_index(drop=True)
    colors = dict(zip(cases["case_id"], plt.get_cmap("tab10")(np.linspace(0, 1, len(cases)))))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=220, sharex=True)
    for case_id, sub in deriv_summary.groupby("case_id", sort=False):
        sub = sub.sort_values("radius")
        color = colors[case_id]
        ls = "-" if sub["source"].iloc[0] == "mnist_rule" else "--"
        label = str(sub["case_label"].iloc[0])
        axes[0].plot(sub["radius"], sub["d_phi_energy_raw_dd_mean"], color=color, ls=ls, lw=2.0, label=label)
        axes[0].fill_between(
            sub["radius"].to_numpy(dtype=float),
            (sub["d_phi_energy_raw_dd_mean"] - 1.96 * sub["d_phi_energy_raw_dd_sem"]).to_numpy(dtype=float),
            (sub["d_phi_energy_raw_dd_mean"] + 1.96 * sub["d_phi_energy_raw_dd_sem"]).to_numpy(dtype=float),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        axes[1].plot(sub["radius"], sub["d2_phi_energy_raw_dd2_mean"], color=color, ls=ls, lw=2.0, label=label)
    axes[0].set_ylabel(r"finite-diff $d\phi_E/dd$")
    axes[1].set_ylabel(r"finite-diff $d^2\phi_E/dd^2$")
    for ax in axes:
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.set_xlabel("radius d")
        ax.grid(True, color="0.90", linewidth=0.65)
    axes[0].set_title("First derivative")
    axes[1].set_title("Second derivative from first derivative")
    axes[1].legend(frameon=False, fontsize=7.2, loc="best")
    out = out_dir / "fig02_all_rules_all_flips_derivative_curvature.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_phase_metrics(mass: pd.DataFrame, spin_a: pd.DataFrame, out_dir: Path) -> Path:
    flip = mass[mass["source"].eq("mnist_flip")].copy().sort_values("complexity_axis")
    rule = mass[mass["source"].eq("mnist_rule")].copy().sort_values("complexity_axis")
    spin = spin_a.copy().sort_values("beta")
    flip["eta"] = flip["case_id"].str.replace("eta_", "", regex=False).astype(float)
    spin["A_norm"] = normalize_series(spin["A_kappa"])
    flip["A_norm"] = normalize_series(flip["positive_curvature_mass_mean"])
    rule["A_norm"] = normalize_series(rule["positive_curvature_mass_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=220)
    axes[0].plot(spin["beta"], spin["A_norm"], color="0.18", lw=1.9, marker="o", ms=3.2, label="spin beta normalized")
    axes[0].plot(flip["eta"], flip["A_norm"], color="#0072B2", lw=1.9, marker="s", ms=4.0, label="MNIST flip eta normalized")
    axes[0].set_xlabel("control parameter: beta or eta")
    axes[0].set_ylabel("normalized positive curvature mass")
    axes[0].set_title("Phase-like metric shape only")
    axes[0].grid(True, color="0.90", linewidth=0.65)
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(rule))
    axes[1].bar(x - 0.18, rule["positive_curvature_mass_mean"], width=0.36, color="#D55E00", alpha=0.78, label="rules")
    xf = np.arange(len(flip)) + len(rule) + 0.7
    axes[1].bar(xf + 0.18, flip["positive_curvature_mass_mean"], width=0.36, color="#0072B2", alpha=0.78, label="flips")
    labels = list(rule["case_label"].str.replace("rule: ", "", regex=False)) + list(flip["case_label"].str.replace("flip ", "", regex=False))
    axes[1].set_xticks(list(x) + list(xf))
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=7.4)
    axes[1].set_ylabel("MNIST finite-diff positive curvature mass")
    axes[1].set_title("Raw MNIST scale, d=0.10..1.00")
    axes[1].grid(True, axis="y", color="0.90", linewidth=0.65)
    axes[1].legend(frameon=False, fontsize=8)
    out = out_dir / "fig03_mnist_flip_vs_spin_phase_metrics.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--d-min", type=float, default=0.10)
    parser.add_argument("--d-max", type=float, default=1.00)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    summary, units = load_mnist_groups(float(args.d_min), float(args.d_max))
    ref_curve, deriv_summary = group_derivatives(units)
    mass = curvature_mass_by_case(ref_curve)
    spin_abs, spin_dphi, spin_a = spin_summaries(float(args.d_min), float(args.d_max))
    data_state = write_data_state(summary, spin_abs, spin_dphi, out_dir)
    flip_inventory = collect_available_flip_runs(out_dir)

    summary.to_csv(out_dir / "all_mnist_phi_groups.csv", index=False)
    units.to_csv(out_dir / "all_mnist_phi_by_ref.csv", index=False)
    ref_curve.to_csv(out_dir / "all_mnist_ref_derivative_curvature.csv", index=False)
    deriv_summary.to_csv(out_dir / "all_mnist_derivative_groups.csv", index=False)
    mass.to_csv(out_dir / "all_mnist_positive_curvature_mass.csv", index=False)
    spin_abs.to_csv(out_dir / "spin_phi_window_d0p1_to_1p0.csv", index=False)
    spin_dphi.to_csv(out_dir / "spin_dphi_window_d0p1_to_1p0.csv", index=False)
    spin_a.to_csv(out_dir / "spin_positive_curvature_mass.csv", index=False)

    fig1 = plot_phi(summary, units, out_dir)
    fig2 = plot_derivatives(deriv_summary, out_dir)
    fig3 = plot_phase_metrics(mass, spin_a, out_dir)
    fig4 = plot_all_available_flips(summary, flip_inventory, out_dir, float(args.d_min), float(args.d_max))
    write_report(out_dir, data_state, flip_inventory, summary, deriv_summary, mass, spin_a)

    run_config = {
        "d_min": float(args.d_min),
        "d_max": float(args.d_max),
        "advanced_run": str(ADV_RUN),
        "flip_run": str(FLIP_RUN),
        "spin_abs": str(SPIN_ABS),
        "spin_dphi": str(SPIN_DPHI),
        "spin_positive_curvature_mass": str(SPIN_A),
        "figures": [str(fig1), str(fig2), str(fig3), str(fig4)],
    }
    (out_dir / "run_config_resolved.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(fig1)
    print(fig2)
    print(fig3)
    print(fig4)
    print(out_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
