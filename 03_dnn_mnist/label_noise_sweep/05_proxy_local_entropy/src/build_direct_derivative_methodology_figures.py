#!/usr/bin/env python3
"""Build MNIST rule/eta figures from direct radial derivative sampling outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


SCRIPT_PATH = Path(__file__).resolve()
STAGE_ROOT = SCRIPT_PATH.parents[1]
WINDOWS_ROOT = Path("/home/bjyong/Complexity/windows_project")
DEFAULT_RUN_ROOT = (
    STAGE_ROOT
    / "summarized_outputs"
    / "direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
DEFAULT_OUT_DIR = (
    STAGE_ROOT
    / "figures"
    / "direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
DEFAULT_SUMMARY_DIR = (
    STAGE_ROOT
    / "summarized_outputs"
    / "direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
)
DEFAULT_COMPLEXITY_TABLE = (
    STAGE_ROOT.parent
    / "02_complexity_measure"
    / "summarized_outputs"
    / "complexity_axis_spin_mnist_30ref_eta0p02_0p05_0p15_0p25"
    / "mnist_complexity_axis_metrics.csv"
)
P_DEFAULT = 2461.0
GRAPH_KS = (8, 16, 32)

RULE_LABELS = {
    "very_low_tv_spectral_teacher": "rule: very low tv",
    "real_even_odd": "rule: even odd",
    "teacher_nn": "rule: teacher nn",
    "random_label": "rule: random label",
}
RULE_NMSTV = {
    "very_low_tv_spectral_teacher": 0.3245703473792008,
    "real_even_odd": 0.4932864276461805,
    "teacher_nn": 0.6843772639598127,
    "random_label": 0.985558573825462,
}
ETA_NMSTV = {
    0.02: 0.356969,
    0.05: 0.573493,
    0.15: 0.776292,
    0.25: 0.897528,
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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def eta_token_to_float(value: str) -> float | None:
    text = str(value)
    if not text.startswith("eta_"):
        return None
    try:
        return float(text.replace("eta_", "").replace("p", "."))
    except ValueError:
        return None


def load_complexity_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        rows = []
        for rule, label in RULE_LABELS.items():
            rows.append(
                {
                    "source": "advanced",
                    "group": rule,
                    "label": label.replace("rule: ", "adv "),
                    "nmstv": RULE_NMSTV[rule],
                    "complexity_norm": np.nan,
                }
            )
        for eta, nmstv in ETA_NMSTV.items():
            rows.append(
                {
                    "source": "flip",
                    "group": f"eta_{eta:.2f}",
                    "label": f"flip eta {eta:.2f}",
                    "nmstv": nmstv,
                    "complexity_norm": np.nan,
                }
            )
        return pd.DataFrame(rows)
    df = pd.read_csv(path)
    for col in ["nmstv", "complexity_norm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def complexity_lookup(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for _, row in table.iterrows():
        source = str(row.get("source", ""))
        group = str(row.get("group", ""))
        nmstv = float(row.get("nmstv", np.nan))
        cnorm = float(row.get("complexity_norm", np.nan))
        if source == "advanced":
            out[group] = {"nmstv": nmstv, "complexity_norm": cnorm}
        elif source == "flip":
            eta = eta_token_to_float(group)
            if eta is not None:
                out[f"eta_{eta:.2f}"] = {"nmstv": nmstv, "complexity_norm": cnorm}
    return out


def resolve_data_path(path_value: str | Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return WINDOWS_ROOT / path


def graph_tv_nmstv(x: np.ndarray, y: np.ndarray, k: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    nn = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean")
    nn.fit(x)
    distances, indices = nn.kneighbors(x, return_distance=True)
    d = distances[:, 1:]
    j = indices[:, 1:]
    nonzero = d[d > 0.0]
    sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    edge_weight: dict[tuple[int, int], float] = {}
    for i in range(x.shape[0]):
        for dist, jj in zip(d[i], j[i]):
            a, b = sorted((int(i), int(jj)))
            weight = float(math.exp(-(float(dist) ** 2) / (2.0 * sigma * sigma)))
            edge_weight[(a, b)] = max(edge_weight.get((a, b), 0.0), weight)
    total_w = float(sum(edge_weight.values()))
    cut_w = float(sum(w for (a, b), w in edge_weight.items() if y[a] != y[b]))
    tv = cut_w / max(total_w, 1.0e-300)
    p = float(np.mean(y == 1))
    baseline = 2.0 * p * (1.0 - p)
    return {
        "k": int(k),
        "edge_count": int(len(edge_weight)),
        "sigma_k": sigma,
        "tv": tv,
        "baseline": baseline,
        "nmstv": tv / max(baseline, 1.0e-12),
    }


def recompute_case_dataset_metrics(units: pd.DataFrame) -> pd.DataFrame:
    if "dataset_path" not in units.columns:
        return pd.DataFrame()
    case_paths = (
        units[["source", "group", "label", "dataset_path"]]
        .dropna(subset=["dataset_path"])
        .drop_duplicates(["source", "group"])
        .sort_values(["source", "group"])
    )
    base_row = case_paths[case_paths["group"].eq("real_even_odd")]
    base_x = None
    base_y = None
    if not base_row.empty:
        base_ds = np.load(resolve_data_path(str(base_row.iloc[0]["dataset_path"])))
        base_x = np.asarray(base_ds["X_train"])
        base_y = np.asarray(base_ds["y_train"])

    rows: list[dict[str, Any]] = []
    for _, row in case_paths.iterrows():
        path = resolve_data_path(str(row["dataset_path"]))
        if not path.exists():
            continue
        payload = np.load(path)
        x = np.asarray(payload["X_train"])
        y = np.asarray(payload["y_train"])
        out: dict[str, Any] = {
            "source": str(row["source"]),
            "group": str(row["group"]),
            "label": str(row["label"]),
            "dataset_path_resolved": str(path),
            "pos_fraction": float(np.mean(y == 1)),
        }
        if base_x is not None and base_y is not None and x.shape == base_x.shape and np.array_equal(x, base_x):
            out["same_X_as_even_odd"] = True
            out["label_diff_vs_even_odd"] = float(np.mean(y != base_y))
        else:
            out["same_X_as_even_odd"] = False
            out["label_diff_vs_even_odd"] = np.nan
        if "eta_flip_mask_train" in payload.files:
            out["stored_flip_rate"] = float(np.mean(payload["eta_flip_mask_train"]))
        else:
            out["stored_flip_rate"] = np.nan
        for k in GRAPH_KS:
            metrics = graph_tv_nmstv(x, y, k)
            for key, value in metrics.items():
                if key != "k":
                    out[f"{key}_k{k}"] = value
        out["tv_mean"] = float(np.mean([out[f"tv_k{k}"] for k in GRAPH_KS]))
        out["baseline_mean"] = float(np.mean([out[f"baseline_k{k}"] for k in GRAPH_KS]))
        out["nmstv"] = float(np.mean([out[f"nmstv_k{k}"] for k in GRAPH_KS]))
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["nmstv", "source", "group"]).reset_index(drop=True)


def apply_case_nmstv(units: pd.DataFrame, case_metrics: pd.DataFrame) -> pd.DataFrame:
    if case_metrics.empty:
        return units
    mapping = {
        (str(row["source"]), str(row["group"])): float(row["nmstv"])
        for _, row in case_metrics.iterrows()
    }
    out = units.copy()
    out["nmstv"] = out.apply(
        lambda row: mapping.get((str(row["source"]), str(row["group"])), float(row["nmstv"])),
        axis=1,
    )
    return out


def read_unit_jsons(run_root: Path) -> pd.DataFrame:
    paths = sorted((run_root / "05_pool2_pm_sais_sampling" / "unit_summaries").rglob("unit_summary.json"))
    if not paths:
        raise FileNotFoundError(f"no unit summaries under {run_root}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unit_summary_path"] = str(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def load_run_units(run_root: Path) -> pd.DataFrame:
    candidates = [
        run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv",
        run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv",
        run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return read_unit_jsons(run_root)


def normalize_units(
    df: pd.DataFrame,
    *,
    source: str,
    p_dim: float,
    complexity: dict[str, dict[str, float]],
) -> pd.DataFrame:
    out = df.copy()
    for col in ["radius", "ref_id", "logZ_inf_full", "dlogZ_inf_full_dr", "eta"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "rule" not in out.columns:
        raise ValueError("unit table is missing rule column")
    out["source"] = source
    if "eta" not in out.columns:
        out["eta"] = out["rule"].map(eta_token_to_float)
    out["group"] = np.where(
        source == "flip",
        out["eta"].map(lambda x: f"eta_{float(x):.2f}" if pd.notna(x) else ""),
        out["rule"].astype(str),
    )
    out["label"] = out.apply(
        lambda row: f"flip eta={float(row['eta']):.2f}"
        if row["source"] == "flip"
        else RULE_LABELS.get(str(row["group"]), f"rule: {row['group']}"),
        axis=1,
    )
    if "phi_energy_raw" not in out.columns:
        out["phi_energy_raw"] = pd.to_numeric(out["logZ_inf_full"], errors="coerce") / float(p_dim)
    else:
        out["phi_energy_raw"] = pd.to_numeric(out["phi_energy_raw"], errors="coerce")
    if "d_phi_energy_direct_dd_unit" not in out.columns and "dlogZ_inf_full_dr" in out.columns:
        out["d_phi_energy_direct_dd_unit"] = pd.to_numeric(out["dlogZ_inf_full_dr"], errors="coerce") / float(p_dim)
    elif "d_phi_energy_direct_dd_unit" in out.columns:
        out["d_phi_energy_direct_dd_unit"] = pd.to_numeric(out["d_phi_energy_direct_dd_unit"], errors="coerce")
    else:
        out["d_phi_energy_direct_dd_unit"] = np.nan
    out["ref_id"] = pd.to_numeric(out["ref_id"], errors="coerce").astype("Int64")
    out["ref_key"] = (
        out["source"].astype(str)
        + ":"
        + out["group"].astype(str)
        + ":"
        + out["ref_id"].astype(str)
    )

    def nmstv_for(row: pd.Series) -> float:
        group = str(row["group"])
        if group in complexity and np.isfinite(complexity[group]["nmstv"]):
            return float(complexity[group]["nmstv"])
        if row["source"] == "flip" and pd.notna(row.get("eta", np.nan)):
            return float(ETA_NMSTV.get(round(float(row["eta"]), 2), float(row["eta"])))
        return float(RULE_NMSTV.get(group, np.nan))

    out["nmstv"] = out.apply(nmstv_for, axis=1)
    return out.dropna(subset=["source", "group", "ref_id", "radius", "phi_energy_raw"]).copy()


def add_finite_difference_derivatives(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, sub in units.groupby("ref_key", sort=False):
        work = sub.sort_values("radius").drop_duplicates("radius", keep="first").copy()
        radius = work["radius"].to_numpy(dtype=np.float64)
        phi = work["phi_energy_raw"].to_numpy(dtype=np.float64)
        if len(radius) >= 2:
            work["d_phi_energy_fd_dd_unit"] = np.gradient(phi, radius)
        else:
            work["d_phi_energy_fd_dd_unit"] = np.nan
        rows.append(work)
    return pd.concat(rows, ignore_index=True)


def summarize_units(units: pd.DataFrame) -> pd.DataFrame:
    return (
        units.groupby(["source", "group", "label", "nmstv", "radius"], as_index=False)
        .agg(
            n_refs=("ref_id", "nunique"),
            phi_energy_raw_mean=("phi_energy_raw", "mean"),
            phi_energy_raw_sd=("phi_energy_raw", "std"),
            phi_energy_raw_sem=("phi_energy_raw", sem),
            d_phi_energy_direct_dd_mean=("d_phi_energy_direct_dd_unit", "mean"),
            d_phi_energy_direct_dd_sd=("d_phi_energy_direct_dd_unit", "std"),
            d_phi_energy_direct_dd_sem=("d_phi_energy_direct_dd_unit", sem),
            d_phi_energy_fd_dd_mean=("d_phi_energy_fd_dd_unit", "mean"),
            d_phi_energy_fd_dd_sd=("d_phi_energy_fd_dd_unit", "std"),
            d_phi_energy_fd_dd_sem=("d_phi_energy_fd_dd_unit", sem),
            ess_fraction_min=("ess_fraction", "min") if "ess_fraction" in units.columns else ("phi_energy_raw", "count"),
            split_logZ_per_P_diff_max=("split_logZ_per_P_diff", "max")
            if "split_logZ_per_P_diff" in units.columns
            else ("phi_energy_raw", "count"),
            split_dlogZ_dr_per_P_diff_max=("split_dlogZ_dr_per_P_diff", "max")
            if "split_dlogZ_dr_per_P_diff" in units.columns
            else ("phi_energy_raw", "count"),
            ce_replay_max_abs_diff_max=("ce_replay_max_abs_diff", "max")
            if "ce_replay_max_abs_diff" in units.columns
            else ("phi_energy_raw", "count"),
        )
        .sort_values(["nmstv", "source", "group", "radius"])
        .reset_index(drop=True)
    )


def compute_curvature(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for (source, group, label, nmstv, ref_id), sub in units.groupby(
        ["source", "group", "label", "nmstv", "ref_id"], sort=False
    ):
        work = sub.sort_values("radius").drop_duplicates("radius", keep="first")
        if len(work) < 3:
            continue
        radius = work["radius"].to_numpy(dtype=np.float64)
        d1 = work["d_phi_energy_direct_dd_unit"].to_numpy(dtype=np.float64)
        if not np.isfinite(d1).all():
            d1 = work["d_phi_energy_fd_dd_unit"].to_numpy(dtype=np.float64)
        if not np.isfinite(d1).all():
            continue
        d2 = np.gradient(d1, radius)
        pos = np.maximum(d2, 0.0)
        neg = np.maximum(-d2, 0.0)
        min_idx = int(np.nanargmin(d2))
        max_idx = int(np.nanargmax(d2))
        metric_rows.append(
            {
                "source": source,
                "group": group,
                "label": label,
                "nmstv": float(nmstv),
                "ref_id": int(ref_id),
                "n_radii": int(len(radius)),
                "positive_curvature_mass": trapz(pos, radius),
                "negative_curvature_mass": trapz(neg, radius),
                "min_second_derivative": float(d2[min_idx]),
                "min_second_derivative_radius": float(radius[min_idx]),
                "max_second_derivative": float(d2[max_idx]),
                "max_second_derivative_radius": float(radius[max_idx]),
            }
        )
        for r, g1, g2 in zip(radius, d1, d2):
            curve_rows.append(
                {
                    "source": source,
                    "group": group,
                    "label": label,
                    "nmstv": float(nmstv),
                    "ref_id": int(ref_id),
                    "radius": float(r),
                    "d_phi_energy_direct_dd": float(g1),
                    "d2_phi_energy_direct_dd2": float(g2),
                    "positive_curvature": float(max(g2, 0.0)),
                    "negative_curvature": float(max(-g2, 0.0)),
                }
            )
    curves = pd.DataFrame(curve_rows).sort_values(["nmstv", "group", "ref_id", "radius"])
    metrics = pd.DataFrame(metric_rows).sort_values(["nmstv", "group", "ref_id"])
    if metrics.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            metrics.groupby(["source", "group", "label", "nmstv"], as_index=False)
            .agg(
                n_refs=("ref_id", "nunique"),
                positive_curvature_mass_mean=("positive_curvature_mass", "mean"),
                positive_curvature_mass_sd=("positive_curvature_mass", "std"),
                positive_curvature_mass_sem=("positive_curvature_mass", sem),
                negative_curvature_mass_mean=("negative_curvature_mass", "mean"),
                negative_curvature_mass_sem=("negative_curvature_mass", sem),
                min_second_derivative_mean=("min_second_derivative", "mean"),
                min_second_derivative_sem=("min_second_derivative", sem),
                min_second_derivative_radius_mean=("min_second_derivative_radius", "mean"),
                max_second_derivative_mean=("max_second_derivative", "mean"),
                max_second_derivative_radius_mean=("max_second_derivative_radius", "mean"),
            )
            .sort_values(["nmstv", "source", "group"])
            .reset_index(drop=True)
        )
    return curves, metrics, summary


def color_setup(summary: pd.DataFrame) -> tuple[plt.Normalize, Any]:
    vals = summary["nmstv"].to_numpy(dtype=np.float64)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if abs(hi - lo) < 1.0e-12:
        hi = lo + 1.0
    return plt.Normalize(lo, hi), plt.get_cmap("viridis")


def style_for_source(source: str) -> str:
    return "-" if source == "advanced" else "--"


def label_rows_nonoverlap(rows: list[dict[str, Any]], min_gap: float) -> list[dict[str, Any]]:
    if not rows:
        return rows
    rows = sorted(rows, key=lambda row: row["y"])
    prev = rows[0]["y"]
    for row in rows[1:]:
        if row["y"] - prev < min_gap:
            row["y"] = prev + min_gap
        prev = row["y"]
    return rows


def add_nmstv_colorbar(fig: plt.Figure, cax: plt.Axes, lax: plt.Axes, summary: pd.DataFrame, norm: plt.Normalize, cmap: Any) -> None:
    groups = (
        summary[["source", "group", "label", "nmstv"]]
        .drop_duplicates()
        .sort_values(["nmstv", "source", "group"])
        .reset_index(drop=True)
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("NMSTV")
    cbar.set_ticks(groups["nmstv"].to_numpy(dtype=float))
    cbar.set_ticklabels([f"{v:.3f}" for v in groups["nmstv"].to_numpy(dtype=float)])
    cbar.ax.tick_params(labelsize=7.2, pad=2)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("right")

    vals = groups["nmstv"].to_numpy(dtype=float)
    lo = float(vals.min())
    hi = float(vals.max())
    pad = 0.04 * max(hi - lo, 1.0e-9)
    lax.set_xlim(0.0, 1.0)
    lax.set_ylim(lo - pad, hi + pad)
    lax.axis("off")
    for _, row in groups.iterrows():
        y = float(row["nmstv"])
        color = cmap(norm(y))
        ls = style_for_source(str(row["source"]))
        cbar.ax.hlines(y, 0.34, 0.92, color="white", lw=1.1, alpha=0.9)
        lax.hlines(y, 0.0, 0.18, color="0.42", lw=0.72)
        lax.plot([0.24, 0.40], [y, y], color=color, lw=2.2, ls=ls)
        lax.text(0.44, y, f"{row['label']}  {y:.3f}", va="center", ha="left", fontsize=8.1, color="0.12")


def plot_phi_spaghetti(units: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> Path:
    norm, cmap = color_setup(summary)
    fig = plt.figure(figsize=(15.8, 7.4), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[18.0, 1.15, 3.25], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lax = fig.add_subplot(gs[0, 2])

    for (_, group), sub in units.groupby(["source", "group"], sort=False):
        color = cmap(norm(float(sub["nmstv"].iloc[0])))
        ls = style_for_source(str(sub["source"].iloc[0]))
        for _, ref in sub.groupby("ref_key", sort=False):
            ref = ref.sort_values("radius")
            ax.plot(ref["radius"], ref["phi_energy_raw"], color=color, lw=0.42, alpha=0.07, ls=ls)

    label_rows = []
    for (_, group), sub in summary.groupby(["source", "group"], sort=False):
        sub = sub.sort_values("radius")
        color = cmap(norm(float(sub["nmstv"].iloc[0])))
        ls = style_for_source(str(sub["source"].iloc[0]))
        x = sub["radius"].to_numpy(dtype=float)
        y = sub["phi_energy_raw_mean"].to_numpy(dtype=float)
        err = sub["phi_energy_raw_sem"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, color=color, lw=2.8, ls=ls)
        ax.fill_between(x, y - 1.96 * err, y + 1.96 * err, color=color, alpha=0.08, linewidth=0)
        last = sub.iloc[-1]
        label_rows.append({"x": float(last["radius"]) + 0.022, "y": float(last["phi_energy_raw_mean"]), "label": str(last["label"]), "color": color})

    ymin = float(units["phi_energy_raw"].quantile(0.002))
    ymax = max(0.006, float(units["phi_energy_raw"].quantile(0.998)))
    span = max(ymax - ymin, 1.0e-6)
    for row in label_rows_nonoverlap(label_rows, 0.035 * span):
        ax.text(row["x"], row["y"], row["label"], color=row["color"], va="center", fontsize=9.5)
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xlim(0.0, 1.18)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$\phi_E(d) = \log Z_{\infty,\mathrm{full}}/P$")
    ax.set_title(r"MNIST active rules + eta flips: direct-run raw $\phi_E(d)$")
    ax.grid(True, color="0.91", linewidth=0.65)
    add_nmstv_colorbar(fig, cax, lax, summary, norm, cmap)
    out = out_dir / "fig01_rules_eta_phi_energy_spaghetti.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_direct_derivative(units: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> Path:
    norm, cmap = color_setup(summary)
    fig = plt.figure(figsize=(15.8, 7.4), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[18.0, 1.15, 3.25], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    lax = fig.add_subplot(gs[0, 2])

    valid_units = units[np.isfinite(units["d_phi_energy_direct_dd_unit"])].copy()
    for (_, group), sub in valid_units.groupby(["source", "group"], sort=False):
        color = cmap(norm(float(sub["nmstv"].iloc[0])))
        ls = style_for_source(str(sub["source"].iloc[0]))
        for _, ref in sub.groupby("ref_key", sort=False):
            ref = ref.sort_values("radius")
            ax.plot(ref["radius"], ref["d_phi_energy_direct_dd_unit"], color=color, lw=0.38, alpha=0.045, ls=ls)

    for (_, group), sub in summary.groupby(["source", "group"], sort=False):
        sub = sub.sort_values("radius")
        color = cmap(norm(float(sub["nmstv"].iloc[0])))
        ls = style_for_source(str(sub["source"].iloc[0]))
        x = sub["radius"].to_numpy(dtype=float)
        y = sub["d_phi_energy_direct_dd_mean"].to_numpy(dtype=float)
        err = sub["d_phi_energy_direct_dd_sem"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, color=color, lw=2.6, ls=ls)
        ax.fill_between(x, y - 1.96 * err, y + 1.96 * err, color=color, alpha=0.08, linewidth=0)

    yvals = valid_units["d_phi_energy_direct_dd_unit"].replace([np.inf, -np.inf], np.nan).dropna()
    ymin = float(yvals.quantile(0.002))
    ymax = float(yvals.quantile(0.998))
    span = max(ymax - ymin, 1.0e-6)
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.10 * span)
    ax.set_xlabel("radius d")
    ax.set_ylabel(r"$d\phi_E/dd$ from radial score")
    ax.set_title(r"MNIST active rules + eta flips: direct first derivative")
    ax.grid(True, color="0.91", linewidth=0.65)
    add_nmstv_colorbar(fig, cax, lax, summary, norm, cmap)
    out = out_dir / "fig02_rules_eta_direct_dphi_dd_spaghetti.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_curvature_phase_like(summary: pd.DataFrame, metrics_summary: pd.DataFrame, curve_summary: pd.DataFrame, out_dir: Path) -> Path:
    norm, cmap = color_setup(summary)
    fig = plt.figure(figsize=(12.0, 4.8), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.30)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    for (_, group), sub in curve_summary.groupby(["source", "group"], sort=False):
        sub = sub.sort_values("radius")
        color = cmap(norm(float(sub["nmstv"].iloc[0])))
        ls = style_for_source(str(sub["source"].iloc[0]))
        x = sub["radius"].to_numpy(dtype=float)
        y = sub["d_phi_energy_direct_dd_mean"].to_numpy(dtype=float)
        err = sub["d_phi_energy_direct_dd_sem"].fillna(0.0).to_numpy(dtype=float)
        ax_left.plot(x, y, color=color, lw=2.0, ls=ls)
        ax_left.fill_between(x, y - 1.96 * err, y + 1.96 * err, color=color, alpha=0.08, linewidth=0)
    ax_left.axhline(0.0, color="0.25", lw=0.8)
    ax_left.set_xlabel("radius d")
    ax_left.set_ylabel(r"$d\phi_E/dd$")
    ax_left.set_title("direct derivative curves")
    ax_left.grid(True, color="0.90", linewidth=0.65)

    metric = metrics_summary.sort_values("nmstv")
    y_lows = []
    y_highs = []
    for _, row in metric.iterrows():
        color = cmap(norm(float(row["nmstv"])))
        ls = style_for_source(str(row["source"]))
        yerr = 1.96 * float(row.get("positive_curvature_mass_sem", 0.0) or 0.0)
        y_lows.append(float(row["positive_curvature_mass_mean"]) - yerr)
        y_highs.append(float(row["positive_curvature_mass_mean"]) + yerr)
        ax_right.errorbar(
            float(row["nmstv"]),
            float(row["positive_curvature_mass_mean"]),
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            ms=6.0,
        )
        ax_right.plot(
            [float(row["nmstv"]) - 0.015, float(row["nmstv"]) + 0.015],
            [float(row["positive_curvature_mass_mean"]), float(row["positive_curvature_mass_mean"])],
            color=color,
            lw=1.5,
            ls=ls,
        )
        ax_right.text(
            float(row["nmstv"]) + 0.012,
            float(row["positive_curvature_mass_mean"]),
            str(row["label"]).replace("rule: ", "").replace("flip ", ""),
            va="center",
            fontsize=7.6,
            color="0.12",
            clip_on=False,
        )
    if not metric.empty:
        x_vals = metric["nmstv"].to_numpy(dtype=float)
        x_span = max(float(np.nanmax(x_vals) - np.nanmin(x_vals)), 1.0e-6)
        y_min = float(np.nanmin(y_lows)) if y_lows else 0.0
        y_max = float(np.nanmax(y_highs)) if y_highs else 1.0
        y_span = max(y_max - y_min, 1.0e-6)
        ax_right.set_xlim(float(np.nanmin(x_vals)) - 0.07 * x_span, float(np.nanmax(x_vals)) + 0.26 * x_span)
        ax_right.set_ylim(y_min - 0.12 * y_span, y_max + 0.18 * y_span)
    ax_right.set_xlabel("NMSTV")
    ax_right.set_ylabel(r"$A_\kappa=\int_0^1 \max(d^2\phi_E/dd^2,0)\,dd$")
    ax_right.set_title("phase-like curvature mass")
    ax_right.grid(True, color="0.90", linewidth=0.65)

    if not metric.empty:
        chosen = metric.sort_values("positive_curvature_mass_mean", ascending=False).iloc[0]
        inset = ax_right.inset_axes([0.10, 0.63, 0.38, 0.31])
        sub = curve_summary[
            curve_summary["source"].eq(chosen["source"]) & curve_summary["group"].eq(chosen["group"])
        ].sort_values("radius")
        x = sub["radius"].to_numpy(dtype=float)
        y = sub["d2_phi_energy_direct_dd2_mean"].to_numpy(dtype=float)
        color = cmap(norm(float(chosen["nmstv"])))
        inset.plot(x, y, color=color, lw=1.2)
        inset.fill_between(x, 0.0, y, where=y > 0.0, color=color, alpha=0.24, interpolate=True)
        inset.axhline(0.0, color="0.25", lw=0.65)
        inset.set_xlabel("d", fontsize=7)
        inset.set_ylabel(r"$d^2\phi_E/dd^2$", fontsize=7)
        inset.tick_params(labelsize=7)
        inset.grid(True, color="0.92", linewidth=0.45)

    out = out_dir / "fig03_rules_eta_direct_curvature_phase_like.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def summarize_curvature_curves(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    return (
        curves.groupby(["source", "group", "label", "nmstv", "radius"], as_index=False)
        .agg(
            n_refs=("ref_id", "nunique"),
            d_phi_energy_direct_dd_mean=("d_phi_energy_direct_dd", "mean"),
            d_phi_energy_direct_dd_sem=("d_phi_energy_direct_dd", sem),
            d2_phi_energy_direct_dd2_mean=("d2_phi_energy_direct_dd2", "mean"),
            d2_phi_energy_direct_dd2_sem=("d2_phi_energy_direct_dd2", sem),
            positive_curvature_mean=("positive_curvature", "mean"),
            negative_curvature_mean=("negative_curvature", "mean"),
        )
        .sort_values(["nmstv", "source", "group", "radius"])
    )


def write_report(out_dir: Path, paths: list[Path], status: dict[str, Any]) -> None:
    lines = [
        "# Direct Derivative Methodology Figures",
        "",
        "This bundle combines the MNIST active-rule and eta-flip direct radial derivative runs.",
        "",
        "Method contract:",
        "- phi_E(d) is `logZ_inf_full / P`.",
        "- The first derivative uses the stored radial-score estimator `dlogZ_inf_full_dr / P`.",
        "- The second derivative is a finite difference of the stored first derivative along the radius grid.",
        "- Rules and eta flips use the same shell SMC/resampling estimator after their own dataset/reference rows are loaded.",
        "- The NMSTV color/axis metric is recomputed from the actual direct-run `dataset_path` labels on kNN graphs with k=8,16,32.",
        "",
        "Run status:",
    ]
    for key, value in status.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Figures:"])
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--complexity-table", type=Path, default=DEFAULT_COMPLEXITY_TABLE)
    parser.add_argument("--p-dim", type=float, default=P_DEFAULT)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    summary_dir = ensure_dir(args.summary_dir)
    complexity = complexity_lookup(load_complexity_table(args.complexity_table))
    rules_root = args.run_root / "01_active_rules_sampling"
    eta_root = args.run_root / "02_eta_flip_sampling"
    rule_units = normalize_units(load_run_units(rules_root), source="advanced", p_dim=args.p_dim, complexity=complexity)
    eta_units = normalize_units(load_run_units(eta_root), source="flip", p_dim=args.p_dim, complexity=complexity)
    rule_units = rule_units.dropna(axis=1, how="all")
    eta_units = eta_units.dropna(axis=1, how="all")
    units = pd.concat([rule_units, eta_units], ignore_index=True)
    case_metrics = recompute_case_dataset_metrics(units)
    units = apply_case_nmstv(units, case_metrics)
    units = units[(units["radius"] >= 0.01 - 1.0e-9) & (units["radius"] <= 1.0 + 1.0e-9)].copy()
    units = add_finite_difference_derivatives(units)
    units = units.sort_values(["nmstv", "source", "group", "ref_id", "radius"]).reset_index(drop=True)
    summary = summarize_units(units)
    curves, ref_metrics, metric_summary = compute_curvature(units)
    curve_summary = summarize_curvature_curves(curves)

    units.to_csv(summary_dir / "combined_direct_derivative_units.csv", index=False)
    summary.to_csv(summary_dir / "combined_direct_phi_by_group_radius.csv", index=False)
    curves.to_csv(summary_dir / "combined_direct_curvature_by_ref_radius.csv", index=False)
    ref_metrics.to_csv(summary_dir / "combined_direct_curvature_metrics_by_ref.csv", index=False)
    metric_summary.to_csv(summary_dir / "combined_direct_curvature_metrics_by_group.csv", index=False)
    curve_summary.to_csv(summary_dir / "combined_direct_curvature_curve_by_group_radius.csv", index=False)
    if not case_metrics.empty:
        case_metrics.to_csv(summary_dir / "combined_direct_dataset_complexity_metrics.csv", index=False)

    fig_paths = [
        plot_phi_spaghetti(units, summary, out_dir),
        plot_direct_derivative(units, summary, out_dir),
        plot_curvature_phase_like(summary, metric_summary, curve_summary, out_dir),
    ]
    status = {
        "rule_units": int(len(rule_units)),
        "eta_units": int(len(eta_units)),
        "total_units": int(len(units)),
        "groups": int(summary[["source", "group"]].drop_duplicates().shape[0]),
        "radius_min": float(units["radius"].min()),
        "radius_max": float(units["radius"].max()),
        "p_dim": float(args.p_dim),
        "dataset_metric": "recomputed graph TV/NMSTV from each direct-run dataset_path, averaged over k=8,16,32",
    }
    write_json(
        summary_dir / "run_config_resolved.json",
        {
            "run_root": args.run_root,
            "out_dir": out_dir,
            "summary_dir": summary_dir,
            "complexity_table": args.complexity_table,
            "status": status,
            "figures": fig_paths,
        },
    )
    write_report(summary_dir, fig_paths, status)
    for path in fig_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
