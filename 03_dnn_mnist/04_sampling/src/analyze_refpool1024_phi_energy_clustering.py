from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "raw_outputs" / "refpool1024_all_radii_90ref"
OUT_ROOT = RUN_ROOT / "06_results_figures" / "stability_clustering"
P = 2461.0
SPLIT_GATE = 0.004
ESS_GATE = 0.04
RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
RULE_LABELS = {
    "low_tv_spectral_teacher": "low tv spectral teacher",
    "real_even_odd": "real even/odd",
    "teacher_nn": "teacher nn",
    "random_label": "random label",
}
COLORS = {
    "low_tv_spectral_teacher": "#0072B2",
    "real_even_odd": "#009E73",
    "teacher_nn": "#D55E00",
    "random_label": "#CC79A7",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_units() -> pd.DataFrame:
    sampling_dir = RUN_ROOT / "05_pool2_pm_sais_sampling"
    unit_path = sampling_dir / "shell_summary_by_unit_with_phi.csv"
    if not unit_path.exists():
        unit_path = sampling_dir / "shell_summary_by_unit.csv"
    if not unit_path.exists():
        raise FileNotFoundError(unit_path)
    df = pd.read_csv(unit_path)
    df["rule"] = df["rule"].astype(str)
    df["ref_id"] = df["ref_id"].astype(int)
    df["radius"] = df["radius"].astype(float)
    df["phi_energy"] = pd.to_numeric(df["logZ_inf_full"], errors="coerce") / P
    df["split_gate_pass"] = pd.to_numeric(df["split_logZ_per_P_diff"], errors="coerce") <= SPLIT_GATE
    df["ess_gate_pass"] = pd.to_numeric(df["ess_fraction"], errors="coerce") >= ESS_GATE
    df["finite_gate_pass"] = df["finite"].astype(bool) & np.isfinite(pd.to_numeric(df["logZ_inf_full"], errors="coerce"))
    df["unit_sampling_fit_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & df["finite_gate_pass"]
    return df.sort_values(["rule", "ref_id", "radius"]).reset_index(drop=True)


def sorted_rules(df: pd.DataFrame) -> list[str]:
    available = set(df["rule"].astype(str).unique())
    out = [rule for rule in RULES if rule in available]
    out.extend(sorted(available - set(out)))
    return out


def build_suitability_tables(unit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "rule",
        "ref_id",
        "radius",
        "n_samples",
        "phi_energy",
        "logZ_inf_full",
        "split_logZ_per_P_diff",
        "ess_fraction",
        "finite_gate_pass",
        "ess_gate_pass",
        "split_gate_pass",
        "unit_sampling_fit_pass",
        "weighted_ce",
        "weighted_error",
        "weighted_h",
        "theta_ref_norm",
        "samples_path",
    ]
    available = [col for col in cols if col in unit.columns]
    unit_out = unit[available].copy()
    summary = (
        unit.groupby(["rule", "radius"])
        .agg(
            ref_count=("ref_id", "nunique"),
            unit_count=("ref_id", "size"),
            unit_sampling_fit_pass_count=("unit_sampling_fit_pass", "sum"),
            split_fail_count=("split_gate_pass", lambda s: int((~s).sum())),
            split_fail_rate=("split_gate_pass", lambda s: float((~s).mean())),
            finite_fail_count=("finite_gate_pass", lambda s: int((~s).sum())),
            ess_fail_count=("ess_gate_pass", lambda s: int((~s).sum())),
            split_q50=("split_logZ_per_P_diff", "median"),
            split_q90=("split_logZ_per_P_diff", lambda s: float(np.quantile(s, 0.90))),
            split_q95=("split_logZ_per_P_diff", lambda s: float(np.quantile(s, 0.95))),
            split_q99=("split_logZ_per_P_diff", lambda s: float(np.quantile(s, 0.99))),
            split_max=("split_logZ_per_P_diff", "max"),
            ess_q05=("ess_fraction", lambda s: float(np.quantile(s, 0.05))),
            ess_min=("ess_fraction", "min"),
            phi_energy_mean=("phi_energy", "mean"),
            phi_energy_sd=("phi_energy", "std"),
            phi_energy_q25=("phi_energy", lambda s: float(np.quantile(s, 0.25))),
            phi_energy_q50=("phi_energy", "median"),
            phi_energy_q75=("phi_energy", lambda s: float(np.quantile(s, 0.75))),
        )
        .reset_index()
        .sort_values(["rule", "radius"])
    )
    summary["pool_q95_split_pass"] = summary["split_q95"] <= SPLIT_GATE
    summary["pool_fail_rate_le_0p05"] = summary["split_fail_rate"] <= 0.05
    return unit_out, summary


def add_derivatives(curves: pd.DataFrame, group_cols: list[str], value_col: str = "phi_energy") -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, sub in curves.sort_values(group_cols + ["radius"]).groupby(group_cols, sort=False):
        out = sub.copy()
        x = out["radius"].to_numpy(dtype=float)
        y = out[value_col].to_numpy(dtype=float)
        if len(out) >= 2:
            d1 = np.gradient(y, x)
            d2 = np.gradient(d1, x)
        else:
            d1 = np.full(len(out), np.nan)
            d2 = np.full(len(out), np.nan)
        out["d_phi_energy_dd"] = d1
        out["d2_phi_energy_dd2"] = d2
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_phi_tables(unit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_curve = unit[
        [
            "rule",
            "ref_id",
            "radius",
            "phi_energy",
            "split_logZ_per_P_diff",
            "ess_fraction",
            "unit_sampling_fit_pass",
        ]
    ].copy()
    ref_curve = add_derivatives(ref_curve, ["rule", "ref_id"], "phi_energy")
    rule_curve = (
        ref_curve.groupby(["rule", "radius"])
        .agg(
            ref_count=("ref_id", "nunique"),
            phi_energy_mean=("phi_energy", "mean"),
            phi_energy_sd=("phi_energy", "std"),
            phi_energy_sem=("phi_energy", lambda s: float(np.std(s, ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0),
            phi_energy_q25=("phi_energy", lambda s: float(np.quantile(s, 0.25))),
            phi_energy_q50=("phi_energy", "median"),
            phi_energy_q75=("phi_energy", lambda s: float(np.quantile(s, 0.75))),
            d_phi_energy_dd_mean=("d_phi_energy_dd", "mean"),
            d_phi_energy_dd_sd=("d_phi_energy_dd", "std"),
            d2_phi_energy_dd2_mean=("d2_phi_energy_dd2", "mean"),
            d2_phi_energy_dd2_sd=("d2_phi_energy_dd2", "std"),
        )
        .reset_index()
        .rename(columns={"phi_energy_mean": "phi_energy"})
    )
    # Also compute derivatives of the mean curve directly; this is the smoothest rule-level trace.
    rule_curve = add_derivatives(rule_curve, ["rule"], "phi_energy")
    rule_curve = rule_curve.rename(
        columns={
            "d_phi_energy_dd": "d_phi_energy_mean_curve_dd",
            "d2_phi_energy_dd2": "d2_phi_energy_mean_curve_dd2",
        }
    )
    return ref_curve, rule_curve


def pivot_features(ref_curve: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    radii = sorted(ref_curve["radius"].unique())
    keys = ref_curve[["rule", "ref_id"]].drop_duplicates().sort_values(["rule", "ref_id"]).reset_index(drop=True)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for col in feature_cols:
        pivot = (
            ref_curve.pivot_table(index=["rule", "ref_id"], columns="radius", values=col, aggfunc="mean")
            .reindex(pd.MultiIndex.from_frame(keys[["rule", "ref_id"]]))
            .sort_index(axis=1)
        )
        arr = pivot.to_numpy(dtype=float)
        col_mean = np.nanmean(arr, axis=0)
        inds = np.where(~np.isfinite(arr))
        if len(inds[0]):
            arr[inds] = np.take(col_mean, inds[1])
        blocks.append(arr)
        names.extend([f"{col}@{float(radius):.1f}" for radius in radii])
    return keys, np.hstack(blocks), names


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    return (x - mean) / sd, mean, sd


def pca_scores(z: np.ndarray, n_components: int = 3) -> np.ndarray:
    centered = z - np.mean(z, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


def kmeans_pp_init(z: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = z.shape[0]
    centers = np.empty((k, z.shape[1]), dtype=float)
    first = int(rng.integers(0, n))
    centers[0] = z[first]
    dist2 = np.sum((z - centers[0]) ** 2, axis=1)
    for idx in range(1, k):
        total = float(np.sum(dist2))
        if total <= 0 or not np.isfinite(total):
            centers[idx] = z[int(rng.integers(0, n))]
        else:
            centers[idx] = z[int(rng.choice(n, p=dist2 / total))]
        dist2 = np.minimum(dist2, np.sum((z - centers[idx]) ** 2, axis=1))
    return centers


def run_kmeans(z: np.ndarray, k: int, seed: int, n_init: int = 16, max_iter: int = 120) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    for _ in range(n_init):
        centers = kmeans_pp_init(z, k, rng)
        labels = np.zeros(z.shape[0], dtype=int)
        for _iter in range(max_iter):
            d2 = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(labels, new_labels) and _iter > 0:
                break
            labels = new_labels
            for cluster in range(k):
                mask = labels == cluster
                if np.any(mask):
                    centers[cluster] = np.mean(z[mask], axis=0)
                else:
                    centers[cluster] = z[int(rng.integers(0, z.shape[0]))]
        inertia = float(np.sum(np.min(np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    assert best_labels is not None
    return best_labels, best_inertia


def silhouette_score(z: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    diff = z[:, None, :] - z[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    scores = np.zeros(len(labels), dtype=float)
    for idx in range(len(labels)):
        same = labels == labels[idx]
        same[idx] = False
        a = float(np.mean(dist[idx, same])) if np.any(same) else 0.0
        b_vals = []
        for cluster in unique:
            if cluster == labels[idx]:
                continue
            other = labels == cluster
            if np.any(other):
                b_vals.append(float(np.mean(dist[idx, other])))
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores[idx] = (b - a) / denom if denom > 0 else 0.0
    return float(np.mean(scores))


@dataclass
class ClusterResult:
    labels: np.ndarray
    k: int
    inertia: float
    silhouette: float
    scores: pd.DataFrame


def choose_kmeans(z: np.ndarray, k_values: Iterable[int], seed: int) -> ClusterResult:
    rows = []
    best: ClusterResult | None = None
    for k in k_values:
        labels, inertia = run_kmeans(z, int(k), seed + int(k) * 101)
        sil = silhouette_score(z, labels)
        rows.append({"k": int(k), "inertia": inertia, "silhouette": sil})
        if best is None or (np.isfinite(sil) and sil > best.silhouette):
            best = ClusterResult(labels=labels, k=int(k), inertia=inertia, silhouette=sil, scores=pd.DataFrame())
    assert best is not None
    best.scores = pd.DataFrame(rows)
    return best


def build_clusters(ref_curve: pd.DataFrame, unit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys, x, feature_names = pivot_features(ref_curve, ["phi_energy", "d_phi_energy_dd", "d2_phi_energy_dd2"])
    z, _, _ = standardize(x)
    pcs = pca_scores(z, 3)
    result = choose_kmeans(z, range(2, 9), seed=20260618)
    assignments = keys.copy()
    assignments["global_cluster"] = result.labels.astype(int)
    assignments["pc1"] = pcs[:, 0]
    assignments["pc2"] = pcs[:, 1]
    assignments["pc3"] = pcs[:, 2] if pcs.shape[1] > 2 else 0.0
    ref_stats = (
        unit.groupby(["rule", "ref_id"])
        .agg(
            split_fail_count=("split_gate_pass", lambda s: int((~s).sum())),
            split_fail_rate=("split_gate_pass", lambda s: float((~s).mean())),
            max_split=("split_logZ_per_P_diff", "max"),
            mean_split=("split_logZ_per_P_diff", "mean"),
            mean_phi_energy=("phi_energy", "mean"),
            theta_ref_norm=("theta_ref_norm", "first"),
        )
        .reset_index()
    )
    assignments = assignments.merge(ref_stats, on=["rule", "ref_id"], how="left")
    scores = result.scores.copy()
    scores["chosen"] = scores["k"].eq(result.k)
    scores["feature_count"] = len(feature_names)

    cluster_curve = (
        ref_curve.merge(assignments[["rule", "ref_id", "global_cluster"]], on=["rule", "ref_id"], how="left")
        .groupby(["global_cluster", "radius"])
        .agg(
            ref_count=("ref_id", "nunique"),
            phi_energy=("phi_energy", "mean"),
            d_phi_energy_dd=("d_phi_energy_dd", "mean"),
            d2_phi_energy_dd2=("d2_phi_energy_dd2", "mean"),
            phi_energy_sd=("phi_energy", "std"),
        )
        .reset_index()
        .sort_values(["global_cluster", "radius"])
    )
    cluster_summary = (
        assignments.groupby(["global_cluster", "rule"])
        .agg(
            ref_count=("ref_id", "nunique"),
            split_fail_rate_mean=("split_fail_rate", "mean"),
            max_split_max=("max_split", "max"),
            mean_phi_energy=("mean_phi_energy", "mean"),
        )
        .reset_index()
        .sort_values(["global_cluster", "rule"])
    )
    return assignments, scores, cluster_curve, cluster_summary


def build_within_rule_clusters(ref_curve: pd.DataFrame, unit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_assignments: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    for rule in sorted_rules(ref_curve):
        sub_curve = ref_curve[ref_curve["rule"] == rule].copy()
        keys, x, _ = pivot_features(sub_curve, ["phi_energy", "d_phi_energy_dd", "d2_phi_energy_dd2"])
        z, _, _ = standardize(x)
        pcs = pca_scores(z, 3)
        result = choose_kmeans(z, range(2, 7), seed=20260618 + RULES.index(rule) * 1000 if rule in RULES else 20260618)
        assn = keys.copy()
        assn["within_rule_cluster"] = result.labels.astype(int)
        assn["within_pc1"] = pcs[:, 0]
        assn["within_pc2"] = pcs[:, 1]
        assn["within_pc3"] = pcs[:, 2] if pcs.shape[1] > 2 else 0.0
        stats = (
            unit[unit["rule"] == rule]
            .groupby(["rule", "ref_id"])
            .agg(
                split_fail_count=("split_gate_pass", lambda s: int((~s).sum())),
                split_fail_rate=("split_gate_pass", lambda s: float((~s).mean())),
                max_split=("split_logZ_per_P_diff", "max"),
                mean_phi_energy=("phi_energy", "mean"),
                theta_ref_norm=("theta_ref_norm", "first"),
            )
            .reset_index()
        )
        assn = assn.merge(stats, on=["rule", "ref_id"], how="left")
        scores = result.scores.copy()
        scores["rule"] = rule
        scores["chosen"] = scores["k"].eq(result.k)
        all_assignments.append(assn)
        all_scores.append(scores)
    return pd.concat(all_assignments, ignore_index=True), pd.concat(all_scores, ignore_index=True)


def plot_sampling_suitability(summary: pd.DataFrame, fig_dir: Path) -> None:
    rules = sorted_rules(summary)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.2), sharex=True)
    metrics = [("split_q50", "q50"), ("split_q90", "q90"), ("split_q95", "q95"), ("split_max", "max")]
    for ax, rule in zip(axes.ravel(), rules):
        sub = summary[summary["rule"] == rule].sort_values("radius")
        for col, label in metrics:
            ax.plot(sub["radius"], sub[col], linewidth=1.55, label=label)
        ax.axhline(SPLIT_GATE, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(RULE_LABELS.get(rule, rule))
        ax.set_xlabel("radius d")
        ax.set_ylabel("split logZ/P diff")
        ax.grid(True, linewidth=0.4, alpha=0.25)
    axes.ravel()[0].legend(fontsize=8, ncol=4)
    fig.suptitle("Sampling pool split stability quantiles, 90ref n=1024", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_sampling_split_quantiles.png", dpi=185)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    for rule in rules:
        sub = summary[summary["rule"] == rule].sort_values("radius")
        ax.plot(sub["radius"], sub["split_fail_rate"], color=COLORS.get(rule), linewidth=2.0, label=RULE_LABELS.get(rule, rule))
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("radius d")
    ax.set_ylabel("unit split fail rate")
    ax.set_title("Fraction of references failing split gate per rule/radius")
    ax.grid(True, linewidth=0.45, alpha=0.28)
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_sampling_split_fail_rate.png", dpi=190)
    plt.close(fig)


def plot_phi_derivatives(rule_curve: pd.DataFrame, fig_dir: Path) -> None:
    rules = sorted_rules(rule_curve)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 10.2), sharex=True)
    ycols = [
        ("phi_energy", "phi(d)_energy = mean logZ_inf_full / P", "Raw phi(d)_energy"),
        ("d_phi_energy_mean_curve_dd", "d phi_energy / dd", "First derivative"),
        ("d2_phi_energy_mean_curve_dd2", "d2 phi_energy / dd2", "Second derivative"),
    ]
    for ax, (col, ylabel, title) in zip(axes, ycols):
        for rule in rules:
            sub = rule_curve[rule_curve["rule"] == rule].sort_values("radius")
            ax.plot(sub["radius"], sub[col], color=COLORS.get(rule), linewidth=2.0, label=RULE_LABELS.get(rule, rule))
            if col == "phi_energy":
                ax.fill_between(
                    sub["radius"],
                    sub["phi_energy_q25"],
                    sub["phi_energy_q75"],
                    color=COLORS.get(rule),
                    alpha=0.13,
                    linewidth=0,
                )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.45, alpha=0.28)
    axes[-1].set_xlabel("radius d")
    axes[0].legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_raw_phi_energy_derivatives_by_rule.png", dpi=190)
    plt.close(fig)


def plot_clusters(assignments: pd.DataFrame, cluster_curve: pd.DataFrame, within_assignments: pd.DataFrame, ref_curve: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    for rule in sorted_rules(assignments):
        sub = assignments[assignments["rule"] == rule]
        ax.scatter(sub["pc1"], sub["pc2"], s=32, alpha=0.82, color=COLORS.get(rule), label=RULE_LABELS.get(rule, rule), c=None)
    for cluster, sub in assignments.groupby("global_cluster"):
        ax.scatter(sub["pc1"].mean(), sub["pc2"].mean(), marker="X", s=170, color="black")
        ax.text(sub["pc1"].mean(), sub["pc2"].mean(), f"C{int(cluster)}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Reference clustering features: raw phi, dphi/dd, d2phi/dd2")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_global_reference_cluster_pca.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    cmap = plt.get_cmap("tab10")
    for cluster, sub in cluster_curve.groupby("global_cluster"):
        ax.plot(sub["radius"], sub["phi_energy"], linewidth=2.0, color=cmap(int(cluster) % 10), label=f"cluster {int(cluster)}")
    ax.set_xlabel("radius d")
    ax.set_ylabel("cluster mean raw phi_energy")
    ax.set_title("Global cluster mean phi(d)_energy curves")
    ax.grid(True, linewidth=0.45, alpha=0.28)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_global_cluster_phi_energy_curves.png", dpi=190)
    plt.close(fig)

    merged = ref_curve.merge(within_assignments[["rule", "ref_id", "within_rule_cluster"]], on=["rule", "ref_id"], how="left")
    rules = sorted_rules(merged)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), sharex=True)
    for ax, rule in zip(axes.ravel(), rules):
        sub = (
            merged[merged["rule"] == rule]
            .groupby(["within_rule_cluster", "radius"])
            .agg(phi_energy=("phi_energy", "mean"), ref_count=("ref_id", "nunique"))
            .reset_index()
        )
        for cluster, csub in sub.groupby("within_rule_cluster"):
            ax.plot(csub["radius"], csub["phi_energy"], linewidth=1.9, label=f"C{int(cluster)} n={int(csub['ref_count'].max())}")
        ax.set_title(RULE_LABELS.get(rule, rule))
        ax.set_xlabel("radius d")
        ax.set_ylabel("raw phi_energy")
        ax.grid(True, linewidth=0.4, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Within-rule reference clusters: mean raw phi(d)_energy", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_within_rule_cluster_phi_energy_curves.png", dpi=190)
    plt.close(fig)


def write_report(
    out_root: Path,
    summary: pd.DataFrame,
    rule_curve: pd.DataFrame,
    assignments: pd.DataFrame,
    cluster_scores: pd.DataFrame,
    within_scores: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Sampling Suitability And Raw Phi Clustering",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- Run: `refpool1024_all_radii_90ref`.",
        "- Uses existing `n=1024` unit summaries and saved sample metadata.",
        "- `phi(d)_energy` is the raw value `logZ_inf_full / P`, not delta from `d0`.",
        "- The saved `samples.npz` contains normalized target weights, not per-particle unnormalized logZ contributions. Therefore arbitrary random re-split logZ estimates cannot be reconstructed from the saved pool alone; independent SMC replicates or richer saved SMC increments are required for true random multi-split logZ experiments.",
        "",
        "## Sampling Suitability",
        "",
        "| rule | q95 split pass radii | fail-rate<=5% radii | max split | max split fail rate | min ESS q05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rule in sorted_rules(summary):
        sub = summary[summary["rule"] == rule]
        lines.append(
            f"| {rule} | {int(sub['pool_q95_split_pass'].sum())}/25 | {int(sub['pool_fail_rate_le_0p05'].sum())}/25 | "
            f"{float(sub['split_max'].max()):.6f} | {float(sub['split_fail_rate'].max()):.3f} | {float(sub['ess_q05'].min()):.6f} |"
        )
    chosen = cluster_scores[cluster_scores["chosen"]]
    chosen_text = f"k={int(chosen['k'].iloc[0])}, silhouette={float(chosen['silhouette'].iloc[0]):.4f}" if len(chosen) else "n/a"
    lines.extend(
        [
            "",
            "## Raw Phi Energy",
            "",
            f"- Rule/radius rows: `{len(rule_curve)}`.",
            "- Derivatives are numerical derivatives on the radius grid using `numpy.gradient`.",
            "",
            "## Clustering",
            "",
            f"- Global reference clustering selected `{chosen_text}` over k=2..8.",
            "- Features: raw `phi_energy`, `d_phi_energy/dd`, and `d2_phi_energy/dd2` over all 25 radii.",
            "",
            "Global cluster composition:",
            "",
            "| cluster | refs | dominant rules | mean split fail rate |",
            "| ---: | ---: | --- | ---: |",
        ]
    )
    for cluster, sub in assignments.groupby("global_cluster"):
        rule_counts = sub["rule"].value_counts()
        dom = ", ".join(f"{rule}:{count}" for rule, count in rule_counts.items())
        lines.append(f"| {int(cluster)} | {len(sub)} | {dom} | {float(sub['split_fail_rate'].mean()):.3f} |")
    lines.extend(
        [
            "",
            "Within-rule selected k:",
            "",
            "| rule | k | silhouette |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in within_scores[within_scores["chosen"]].sort_values("rule").iterrows():
        lines.append(f"| {row['rule']} | {int(row['k'])} | {float(row['silhouette']):.4f} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `tables/unit_sampling_suitability.csv`",
            "- `tables/sampling_suitability_by_rule_radius.csv`",
            "- `tables/raw_phi_energy_by_ref_radius.csv`",
            "- `tables/raw_phi_energy_by_rule_radius.csv`",
            "- `tables/reference_cluster_assignments_global.csv`",
            "- `tables/reference_cluster_assignments_within_rule.csv`",
            "- `figures/fig01_sampling_split_quantiles.png`",
            "- `figures/fig02_sampling_split_fail_rate.png`",
            "- `figures/fig03_raw_phi_energy_derivatives_by_rule.png`",
            "- `figures/fig04_global_reference_cluster_pca.png`",
            "- `figures/fig05_global_cluster_phi_energy_curves.png`",
            "- `figures/fig06_within_rule_cluster_phi_energy_curves.png`",
            "",
        ]
    )
    (out_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir(OUT_ROOT)
    table_dir = ensure_dir(OUT_ROOT / "tables")
    fig_dir = ensure_dir(OUT_ROOT / "figures")
    unit = load_units()
    unit_out, suitability = build_suitability_tables(unit)
    ref_curve, rule_curve = build_phi_tables(unit)
    assignments, cluster_scores, cluster_curve, cluster_summary = build_clusters(ref_curve, unit)
    within_assignments, within_scores = build_within_rule_clusters(ref_curve, unit)

    write_csv(table_dir / "unit_sampling_suitability.csv", unit_out)
    write_csv(table_dir / "sampling_suitability_by_rule_radius.csv", suitability)
    write_csv(table_dir / "raw_phi_energy_by_ref_radius.csv", ref_curve)
    write_csv(table_dir / "raw_phi_energy_by_rule_radius.csv", rule_curve)
    write_csv(table_dir / "reference_cluster_assignments_global.csv", assignments)
    write_csv(table_dir / "reference_cluster_scores_global.csv", cluster_scores)
    write_csv(table_dir / "cluster_phi_energy_by_radius_global.csv", cluster_curve)
    write_csv(table_dir / "cluster_summary_by_rule_global.csv", cluster_summary)
    write_csv(table_dir / "reference_cluster_assignments_within_rule.csv", within_assignments)
    write_csv(table_dir / "reference_cluster_scores_within_rule.csv", within_scores)

    plot_sampling_suitability(suitability, fig_dir)
    plot_phi_derivatives(rule_curve, fig_dir)
    plot_clusters(assignments, cluster_curve, within_assignments, ref_curve, fig_dir)
    write_report(OUT_ROOT, suitability, rule_curve, assignments, cluster_scores, within_scores)

    print(f"wrote analysis to {OUT_ROOT}")
    print(f"tables: {table_dir}")
    print(f"figures: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
