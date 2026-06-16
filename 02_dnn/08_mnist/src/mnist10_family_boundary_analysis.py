from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[2]
SOURCE_RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
PILOT_RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_ref30_selector_targeted_pilot"
SOURCE_ANALYSIS_DIR = SOURCE_RUN_ROOT / "07_reference_family_analysis"
OVERLAY_DIR = PILOT_RUN_ROOT / "06_overlay_selector_qc"
DEFAULT_OUT_DIR = PILOT_RUN_ROOT / "07_family_boundary_analysis"
DEFAULT_SELECTOR = "dense_qc_stable_ref30"
DEFAULT_RULE = "low_tv_spectral_teacher"
SPLIT_GATE = 0.004
ESS_GATE = 0.04
BLOCKED_RADIUS = 0.85

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mnist14_model import MNIST14Arch, logits_np


ARCH = MNIST14Arch(input_dim=100, hidden_width=20)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def repo_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_inputs(source_analysis_dir: Path, overlay_dir: Path, selector: str, rule: str) -> dict[str, pd.DataFrame]:
    membership = pd.read_csv(source_analysis_dir / "selector_membership.csv")
    selectors = membership[(membership["selector"] == selector) & (membership["rule"] == rule)].copy()
    if selectors.empty:
        raise ValueError(f"No selector rows found for {selector}/{rule}")

    units = pd.read_csv(overlay_dir / "overlay_unit_summary_long.csv")
    units["rule"] = units["rule"].astype(str)
    for col in [
        "split_id",
        "ref_id",
        "radius",
        "ess_fraction",
        "split_logZ_per_P_diff",
        "delta_phi_energy_unit",
        "delta_phi_full_unit",
    ]:
        if col in units.columns:
            units[col] = pd.to_numeric(units[col], errors="coerce")
    if "unit_qc_pass" in units.columns:
        units["unit_qc_pass"] = to_bool(units["unit_qc_pass"])
    units = units[(units["rule"] == rule) & (units["ref_id"].astype(int).isin(selectors["ref_id"].astype(int)))].copy()

    qc = pd.read_csv(overlay_dir / "overlay_selector_qc_by_rule_radius.csv")
    qc = qc[(qc["selector"] == selector) & (qc["rule"] == rule)].copy()
    for col in [
        "radius",
        "observed_ref_count",
        "missing_ref_count",
        "q05_ess_fraction",
        "max_split_logZ_per_P_diff",
        "bootstrap_sd_phi",
    ]:
        if col in qc.columns:
            qc[col] = pd.to_numeric(qc[col], errors="coerce")
    qc["qc_pass"] = to_bool(qc["qc_pass"])
    qc["complete"] = to_bool(qc["complete"])

    phi = pd.read_csv(overlay_dir / "overlay_selector_phi_by_rule_radius.csv")
    phi = phi[(phi["selector"] == selector) & (phi["rule"] == rule)].copy()
    for col in ["radius", "delta_phi_energy", "delta_phi_full", "n_units"]:
        if col in phi.columns:
            phi[col] = pd.to_numeric(phi[col], errors="coerce")
    phi["qc_pass"] = to_bool(phi["qc_pass"])

    refs = pd.read_csv(SOURCE_RUN_ROOT / "04_exact_reference_search" / "reference_index.csv")
    refs = refs[(refs["rule"] == rule) & (refs["ref_id"].astype(int).isin(selectors["ref_id"].astype(int)))].copy()
    for col in ["ref_id", "theta_norm", "CE_mean_train", "CE_mean_test", "train_error", "test_error", "min_margin", "q05_margin", "median_margin", "mean_margin"]:
        if col in refs.columns:
            refs[col] = pd.to_numeric(refs[col], errors="coerce")

    return {"selectors": selectors, "units": units, "qc": qc, "phi": phi, "refs": refs}


def compute_reference_geometry(refs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    theta_by_ref: dict[int, np.ndarray] = {}
    for row in refs.sort_values("ref_id").itertuples():
        theta = np.load(repo_path(row.theta_path))
        theta_by_ref[int(row.ref_id)] = np.asarray(theta, dtype=np.float64).reshape(-1)

    ref_ids = sorted(theta_by_ref)
    theta_mat = np.vstack([theta_by_ref[r] for r in ref_ids])
    centroid = theta_mat.mean(axis=0)
    ref27 = theta_by_ref.get(27)

    dataset_path = repo_path(str(refs["dataset_path"].dropna().iloc[0]))
    data = np.load(dataset_path)
    x_eval = np.asarray(data["X_test"], dtype=np.float64)
    pred_mat = []
    for ref_id in ref_ids:
        pred_mat.append(np.where(logits_np(theta_by_ref[ref_id], x_eval, ARCH) >= 0.0, 1, -1))
    pred_mat_arr = np.vstack(pred_mat)
    pairwise_disagree = np.not_equal(pred_mat_arr[:, None, :], pred_mat_arr[None, :, :]).mean(axis=2)
    ref27_idx = ref_ids.index(27) if 27 in ref_ids else None

    for i, ref_id in enumerate(ref_ids):
        theta = theta_by_ref[ref_id]
        rows.append(
            {
                "ref_id": int(ref_id),
                "theta_distance_to_centroid": float(np.linalg.norm(theta - centroid)),
                "theta_distance_to_ref027": float(np.linalg.norm(theta - ref27)) if ref27 is not None else float("nan"),
                "mean_pairwise_test_disagreement": float(pairwise_disagree[i].mean()),
                "test_disagreement_to_ref027": float(pairwise_disagree[i, ref27_idx]) if ref27_idx is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_reference_diagnostics(units: pd.DataFrame, refs: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_refs = sorted(int(r) for r in refs["ref_id"].unique())
    for ref_id in selected_refs:
        sub = units[units["ref_id"].astype(int) == ref_id].copy().sort_values("radius")
        row_ref = refs[refs["ref_id"].astype(int) == ref_id].iloc[0].to_dict()
        observed_radii = sorted(float(r) for r in sub["radius"].dropna().unique())
        fail_sub = sub[(sub["split_logZ_per_P_diff"] > SPLIT_GATE) | (sub["ess_fraction"] < ESS_GATE)]
        blocked_row = sub[np.isclose(sub["radius"], BLOCKED_RADIUS)]
        status = "unresolved_missing_0p85"
        if not blocked_row.empty and bool(((blocked_row["split_logZ_per_P_diff"] <= SPLIT_GATE) & (blocked_row["ess_fraction"] >= ESS_GATE)).all()):
            status = "observed_pass_0p85"
        if not blocked_row.empty and bool(((blocked_row["split_logZ_per_P_diff"] > SPLIT_GATE) | (blocked_row["ess_fraction"] < ESS_GATE)).any()):
            status = "boundary_fail_0p85"
        rows.append(
            {
                "ref_id": int(ref_id),
                "boundary_status": status,
                "observed_radius_count": int(len(observed_radii)),
                "observed_radii": ";".join(f"{r:.4f}" for r in observed_radii),
                "first_fail_radius": float(fail_sub["radius"].min()) if len(fail_sub) else float("nan"),
                "max_split_logZ_per_P_diff": float(sub["split_logZ_per_P_diff"].max()) if len(sub) else float("nan"),
                "min_ess_fraction": float(sub["ess_fraction"].min()) if len(sub) else float("nan"),
                "split_at_0p85": float(blocked_row["split_logZ_per_P_diff"].iloc[0]) if len(blocked_row) else float("nan"),
                "ess_at_0p85": float(blocked_row["ess_fraction"].iloc[0]) if len(blocked_row) else float("nan"),
                "delta_phi_energy_at_0p65": value_at_radius(sub, 0.65, "delta_phi_energy_unit"),
                "delta_phi_energy_at_0p85": value_at_radius(sub, 0.85, "delta_phi_energy_unit"),
                "theta_norm": float(row_ref.get("theta_norm", np.nan)),
                "CE_mean_train": float(row_ref.get("CE_mean_train", np.nan)),
                "CE_mean_test": float(row_ref.get("CE_mean_test", np.nan)),
                "train_error": float(row_ref.get("train_error", np.nan)),
                "test_error": float(row_ref.get("test_error", np.nan)),
                "min_margin": float(row_ref.get("min_margin", np.nan)),
                "q05_margin": float(row_ref.get("q05_margin", np.nan)),
            }
        )
    out = pd.DataFrame(rows).merge(geometry, on="ref_id", how="left")
    return add_family_clusters(out)


def value_at_radius(sub: pd.DataFrame, radius: float, col: str) -> float:
    row = sub[np.isclose(sub["radius"], radius)]
    if row.empty or col not in row.columns:
        return float("nan")
    return float(row[col].iloc[0])


def add_family_clusters(diag: pd.DataFrame) -> pd.DataFrame:
    out = diag.copy()
    features = [
        "delta_phi_energy_at_0p65",
        "theta_norm",
        "CE_mean_train",
        "CE_mean_test",
        "min_margin",
        "q05_margin",
        "theta_distance_to_centroid",
        "theta_distance_to_ref027",
        "mean_pairwise_test_disagreement",
        "test_disagreement_to_ref027",
        "max_split_logZ_per_P_diff",
    ]
    x = out[features].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    scaled = StandardScaler().fit_transform(x.to_numpy(dtype=np.float64))
    pca = PCA(n_components=2, random_state=20260615)
    pcs = pca.fit_transform(scaled)
    k = min(3, len(out))
    labels = KMeans(n_clusters=k, random_state=20260615, n_init=30).fit_predict(scaled)
    out["family_cluster"] = labels
    out["family_pc1"] = pcs[:, 0]
    out["family_pc2"] = pcs[:, 1]
    return out.sort_values(["boundary_status", "family_cluster", "ref_id"]).reset_index(drop=True)


def build_claimable_curve(qc: pd.DataFrame, phi: pd.DataFrame) -> pd.DataFrame:
    merged = phi.merge(qc[["radius", "observed_ref_count", "missing_ref_count", "max_split_logZ_per_P_diff", "q05_ess_fraction", "bootstrap_sd_phi"]], on="radius", how="left")
    merged["claimable"] = merged["qc_pass"]
    return merged.sort_values("radius").reset_index(drop=True)


def build_candidate_selectors(diag: pd.DataFrame) -> dict[str, list[int]]:
    all_refs = sorted(int(r) for r in diag["ref_id"].unique())
    boundary_fail_refs = sorted(int(r) for r in diag.loc[diag["boundary_status"] == "boundary_fail_0p85", "ref_id"].unique())
    observed_pass_0p85 = sorted(int(r) for r in diag.loc[diag["boundary_status"] == "observed_pass_0p85", "ref_id"].unique())
    candidates = {
        "original_ref30": all_refs,
        "ref29_minus_boundary_fail_ref027": [r for r in all_refs if r != 27],
        "minus_all_observed_boundary_fail_refs": [r for r in all_refs if r not in boundary_fail_refs],
        "observed_pass_0p85_ref12": observed_pass_0p85,
    }
    return candidates


def candidate_qc_status(units: pd.DataFrame, candidates: dict[str, list[int]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_radii = sorted(float(r) for r in units["radius"].dropna().unique())
    for candidate, ref_ids in candidates.items():
        selected = units[units["ref_id"].astype(int).isin(ref_ids)].copy()
        selected_count = len(ref_ids)
        for radius in all_radii:
            sub = selected[np.isclose(selected["radius"], radius)].copy()
            observed = sorted(int(r) for r in sub["ref_id"].dropna().unique())
            missing = sorted(set(ref_ids) - set(observed))
            q05_ess = float(np.quantile(sub["ess_fraction"].dropna(), 0.05)) if len(sub) else float("nan")
            max_split = float(sub["split_logZ_per_P_diff"].max()) if len(sub) else float("nan")
            observed_fail_refs = sorted(
                int(r)
                for r in sub.loc[(sub["split_logZ_per_P_diff"] > SPLIT_GATE) | (sub["ess_fraction"] < ESS_GATE), "ref_id"].dropna().unique()
            )
            complete = len(observed) == selected_count
            observed_qc_fail = len(observed_fail_refs) > 0
            if complete and not observed_qc_fail:
                status = "complete_observed_pass"
            elif observed_qc_fail and len(missing):
                status = "missing_and_observed_fail"
            elif observed_qc_fail:
                status = "observed_fail"
            else:
                status = "missing_only"
            rows.append(
                {
                    "candidate_selector": candidate,
                    "radius": float(radius),
                    "selected_ref_count": int(selected_count),
                    "observed_ref_count": int(len(observed)),
                    "missing_ref_count": int(len(missing)),
                    "missing_refs": ";".join(str(r) for r in missing),
                    "observed_fail_count": int(len(observed_fail_refs)),
                    "observed_fail_refs": ";".join(str(r) for r in observed_fail_refs),
                    "max_split_logZ_per_P_diff": max_split,
                    "q05_ess_fraction": q05_ess,
                    "status": status,
                    "candidate_qc_pass": bool(complete and not observed_qc_fail),
                }
            )
    return pd.DataFrame(rows).sort_values(["candidate_selector", "radius"]).reset_index(drop=True)


def build_recovery_tasks(candidate_status: pd.DataFrame, candidate: str = "ref29_minus_boundary_fail_ref027") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sub = candidate_status[candidate_status["candidate_selector"] == candidate].copy()
    for row in sub.itertuples():
        radius = float(row.radius)
        missing_refs = [int(x) for x in str(row.missing_refs).split(";") if str(x).strip()]
        fail_refs = [int(x) for x in str(row.observed_fail_refs).split(";") if str(x).strip()]
        task_refs = sorted(set(missing_refs + fail_refs))
        if not task_refs:
            continue
        if "observed_fail" in str(row.status):
            priority = "blocked_observed_fail"
        elif radius < BLOCKED_RADIUS:
            priority = "low_dense_gap"
        elif radius == BLOCKED_RADIUS:
            priority = "next_boundary_test"
        else:
            priority = "post_boundary_only_after_0p85_pass"
        rows.append(
            {
                "candidate_selector": candidate,
                "radius": radius,
                "priority": priority,
                "task_ref_count": int(len(task_refs)),
                "task_refs": ",".join(str(r) for r in task_refs),
                "reason": row.status,
            }
        )
    return pd.DataFrame(rows).sort_values(["radius", "priority"]).reset_index(drop=True)


def plot_claimable_curve(curve: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    claim = curve[curve["claimable"]].copy()
    no_claim = curve[~curve["claimable"]].copy()
    ax.plot(claim["radius"], claim["delta_phi_energy"], marker="o", linewidth=1.8, color="#1f6f8b", label="complete QC pass")
    ax.scatter(no_claim["radius"], no_claim["delta_phi_energy"], marker="x", s=42, color="#9c3848", label="not claimable")
    ax.axvline(BLOCKED_RADIUS, color="#9c3848", linestyle="--", linewidth=1.2)
    ax.text(BLOCKED_RADIUS * 1.02, ax.get_ylim()[0] * 0.82, "blocked d=0.85", color="#9c3848", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("d_raw")
    ax.set_ylabel("mean delta phi(d)_energy")
    ax.set_title("Claimable dense-stable ref30 phi(d)_energy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig01_claimable_phi_energy_curve.png", dpi=180)
    plt.close(fig)


def plot_spaghetti(units: pd.DataFrame, diag: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    status_by_ref = dict(zip(diag["ref_id"].astype(int), diag["boundary_status"]))
    colors = {
        "boundary_fail_0p85": "#b1283a",
        "observed_pass_0p85": "#2f7d32",
        "unresolved_missing_0p85": "#6c7a89",
    }
    for ref_id, sub in units.groupby("ref_id"):
        ref_id_int = int(ref_id)
        sub = sub.sort_values("radius")
        status = status_by_ref.get(ref_id_int, "unresolved_missing_0p85")
        line_width = 2.2 if ref_id_int == 27 else 0.9
        alpha = 0.95 if ref_id_int in {22, 27} else 0.58
        ax.plot(sub["radius"], sub["delta_phi_energy_unit"], marker="o", markersize=2.5, linewidth=line_width, alpha=alpha, color=colors.get(status, "#777777"))
        failed = sub[(sub["split_logZ_per_P_diff"] > SPLIT_GATE) | (sub["ess_fraction"] < ESS_GATE)]
        if len(failed):
            ax.scatter(failed["radius"], failed["delta_phi_energy_unit"], marker="x", s=54, color="#b1283a", zorder=5)
    ax.axvline(BLOCKED_RADIUS, color="#b1283a", linestyle="--", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("d_raw")
    ax.set_ylabel("delta phi(d)_energy per reference")
    ax.set_title("Reference-level phi(d)_energy and d=0.85 boundary")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig02_reference_phi_energy_spaghetti_boundary.png", dpi=180)
    plt.close(fig)


def plot_split_heatmap(units: pd.DataFrame, diag: pd.DataFrame, out_dir: Path) -> None:
    radii = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30, 0.45, 0.65, 0.85]
    refs = sorted(int(r) for r in diag["ref_id"].unique())
    matrix = np.full((len(refs), len(radii)), np.nan)
    for i, ref_id in enumerate(refs):
        sub = units[units["ref_id"].astype(int) == ref_id]
        for j, radius in enumerate(radii):
            row = sub[np.isclose(sub["radius"], radius)]
            if len(row):
                matrix[i, j] = float(row["split_logZ_per_P_diff"].max())
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f1f1f1")
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=0.006)
    ax.set_xticks(np.arange(len(radii)))
    ax.set_xticklabels([f"{r:g}" for r in radii], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(refs)))
    ax.set_yticklabels([f"{r:02d}" for r in refs])
    for i in range(len(refs)):
        for j in range(len(radii)):
            val = matrix[i, j]
            if np.isfinite(val) and val > SPLIT_GATE:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#b1283a", linewidth=1.8))
    ax.set_xlabel("d_raw")
    ax.set_ylabel("reference id")
    ax.set_title("Split-logZ/P by reference")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("split logZ/P diff")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig03_split_logz_heatmap_boundary.png", dpi=180)
    plt.close(fig)


def plot_family_scatter(diag: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    colors = {
        "boundary_fail_0p85": "#b1283a",
        "observed_pass_0p85": "#2f7d32",
        "unresolved_missing_0p85": "#6c7a89",
    }
    for status, sub in diag.groupby("boundary_status"):
        ax.scatter(sub["family_pc1"], sub["family_pc2"], s=58, color=colors.get(status, "#777777"), label=status, alpha=0.9)
    for row in diag[diag["ref_id"].isin([22, 27, 52])].itertuples():
        ax.annotate(f"ref{int(row.ref_id):02d}", (row.family_pc1, row.family_pc2), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("family feature PC1")
    ax.set_ylabel("family feature PC2")
    ax.set_title("Reference family diagnostics")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig04_reference_family_pca.png", dpi=180)
    plt.close(fig)


def plot_candidate_qc(candidate_status: pd.DataFrame, out_dir: Path, candidate: str = "ref29_minus_boundary_fail_ref027") -> None:
    sub = candidate_status[candidate_status["candidate_selector"] == candidate].copy().sort_values("radius")
    if sub.empty:
        return
    color_by_status = {
        "complete_observed_pass": "#1f6f8b",
        "missing_only": "#d79b2d",
        "missing_and_observed_fail": "#b1283a",
        "observed_fail": "#b1283a",
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for status, status_sub in sub.groupby("status"):
        ax.scatter(
            status_sub["radius"],
            status_sub["observed_ref_count"],
            s=58,
            color=color_by_status.get(status, "#777777"),
            label=status,
            zorder=4,
        )
    ax.plot(sub["radius"], sub["selected_ref_count"], color="#333333", linewidth=1.2, label="selected refs")
    ax.plot(sub["radius"], sub["observed_ref_count"], color="#6c7a89", linewidth=1.0, alpha=0.8)
    ax.axvline(BLOCKED_RADIUS, color="#b1283a", linestyle="--", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("d_raw")
    ax.set_ylabel("reference count")
    ax.set_title("Candidate ref29-minus-ref027 observed coverage")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig05_candidate_ref29_qc_coverage.png", dpi=180)
    plt.close(fig)


def write_report(
    out_dir: Path,
    selector: str,
    rule: str,
    curve: pd.DataFrame,
    diag: pd.DataFrame,
    candidate_status: pd.DataFrame,
    recovery_tasks: pd.DataFrame,
) -> None:
    claim = curve[curve["claimable"]].copy()
    blocked = curve[np.isclose(curve["radius"], BLOCKED_RADIUS)]
    fail_refs = diag[diag["boundary_status"] == "boundary_fail_0p85"].sort_values("ref_id")
    fail_ref_list = ", ".join(f"ref{int(r):03d}" for r in fail_refs["ref_id"].tolist())
    pass_radii = ", ".join(f"{float(r):g}" for r in claim["radius"])
    row = blocked.iloc[0].to_dict() if len(blocked) else {}
    fail_lines = "\n".join(
        f"| {int(r.ref_id)} | {float(r.split_at_0p85):.12f} | {float(r.ess_at_0p85):.6f} | {r.boundary_status} |"
        for r in fail_refs.itertuples()
    )
    if not fail_lines:
        fail_lines = "| none | n/a | n/a | n/a |"
    ref29 = candidate_status[
        (candidate_status["candidate_selector"] == "ref29_minus_boundary_fail_ref027")
        & np.isclose(candidate_status["radius"], BLOCKED_RADIUS)
    ]
    ref29_row = ref29.iloc[0].to_dict() if len(ref29) else {}
    next_tasks = recovery_tasks[recovery_tasks["priority"] == "next_boundary_test"].copy()
    ref29_has_observed_fail = bool(str(ref29_row.get("observed_fail_refs", "")).strip())
    if ref29_has_observed_fail:
        next_task_section = (
            "No missing-fill task is safe for this candidate as a promotion path, because it already has an observed "
            f"failure at `d_raw=0.85`: `{ref29_row.get('observed_fail_refs', '')}`. The next safe action is to define "
            "a new predeclared family law that excludes the observed boundary-failing references, or to run a separate "
            "hard-reference audit explicitly labeled as diagnostic."
        )
    else:
        next_task_lines = "\n".join(
            f"| {float(r.radius):g} | {int(r.task_ref_count)} | `{r.task_refs}` | {r.reason} |"
            for r in next_tasks.itertuples()
        )
        if not next_task_lines:
            next_task_lines = "| none | 0 | n/a | n/a |"
        next_task_section = f"""The next safe diagnostic task, if this updated family law is accepted before execution, is:

| radius | task refs | refs | reason |
| ---: | ---: | --- | --- |
{next_task_lines}"""
    report = f"""# MNIST10 Family Boundary Analysis

Selector: `{selector}`

Rule: `{rule}`

## Decision

The current evidence supports a single-family averaged `phi(d)_energy` curve only through:

`{pass_radii}`

`d_raw=0.85` is blocked for the current single-family selector. Boundary-failing references observed so far: `{fail_ref_list}`.

## Blocked Radius Summary

- blocked radius: `{BLOCKED_RADIUS}`
- selector QC pass: `{row.get("qc_pass", False)}`
- claim status: `{row.get("claim_status", "n/a")}`
- observed refs at blocked radius: `{row.get("n_units", row.get("observed_ref_count", "n/a"))}`
- max split logZ/P diff at blocked radius: `{row.get("max_split_logZ_per_P_diff", "n/a")}`

## Boundary Reference

| ref | split at 0.85 | ESS at 0.85 | status |
| --- | ---: | ---: | --- |
{fail_lines}

## Interpretation

The `dense_qc_stable_ref30` selector behaves like a coherent single family up to `d_raw=0.65`; both `d_raw=0.45` and `d_raw=0.65` are complete 30-reference QC-pass rows after targeted overlay. At `d_raw=0.85`, ref027 fails split-logZ stability in both the source row and the forced targeted rerun. A diagnostic ref29-minus-ref027 recovery attempt then found another boundary failure at ref033. These failures are not ESS failures.

This means sparse large-domain production should not be promoted as one averaged ref30 curve from the current selector. The next safe step is family decomposition: either define a predeclared updated family law that excludes the observed boundary-failing references and rerun selector QC, or report separate family curves once enough large-radius rows exist for each family.

## Candidate Ref29 Recovery State

Candidate: `ref29_minus_boundary_fail_ref027`

This candidate is diagnostic only. It removes the repeated boundary-failing ref027 from the original 30-reference selector.

At `d_raw=0.85`:

- selected refs: `{ref29_row.get("selected_ref_count", "n/a")}`
- observed refs: `{ref29_row.get("observed_ref_count", "n/a")}`
- missing refs: `{ref29_row.get("missing_ref_count", "n/a")}`
- observed fail refs: `{ref29_row.get("observed_fail_refs", "")}`
- status: `{ref29_row.get("status", "n/a")}`

{next_task_section}

## Artifacts

- `claimable_phi_curve.csv`
- `boundary_reference_diagnostics.csv`
- `selector_qc_status.csv`
- `family_cluster_assignments.csv`
- `candidate_selector_qc_status.csv`
- `candidate_recovery_tasks.csv`
- `large_domain_decision.json`
- `figures/fig01_claimable_phi_energy_curve.png`
- `figures/fig02_reference_phi_energy_spaghetti_boundary.png`
- `figures/fig03_split_logz_heatmap_boundary.png`
- `figures/fig04_reference_family_pca.png`
- `figures/fig05_candidate_ref29_qc_coverage.png`
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-analysis-dir", type=Path, default=SOURCE_ANALYSIS_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=OVERLAY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selector", default=DEFAULT_SELECTOR)
    parser.add_argument("--rule", default=DEFAULT_RULE)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    ensure_dir(out_dir / "figures")
    inputs = load_inputs(args.source_analysis_dir, args.overlay_dir, args.selector, args.rule)
    geometry = compute_reference_geometry(inputs["refs"])
    diag = build_reference_diagnostics(inputs["units"], inputs["refs"], geometry)
    curve = build_claimable_curve(inputs["qc"], inputs["phi"])
    cluster_assignments = diag[["ref_id", "boundary_status", "family_cluster", "family_pc1", "family_pc2"]].copy()
    candidate_status = candidate_qc_status(inputs["units"], build_candidate_selectors(diag))
    recovery_tasks = build_recovery_tasks(candidate_status)
    boundary_fail_refs = sorted(int(r) for r in diag.loc[diag["boundary_status"] == "boundary_fail_0p85", "ref_id"].unique())

    decision = {
        "selector": args.selector,
        "rule": args.rule,
        "single_family_supported_through_radius": float(curve[curve["claimable"]]["radius"].max()),
        "blocked_radius": BLOCKED_RADIUS,
        "blocked_reason": "observed split-logZ/P instability at d_raw=0.85",
        "observed_boundary_fail_refs": boundary_fail_refs,
        "large_domain_sparse_production_supported": False,
        "next_safe_action": "family-boundary decomposition before any sparse large-domain promotion",
        "diagnostic_candidate": "ref29_minus_boundary_fail_ref027",
        "diagnostic_candidate_status_at_0p85": candidate_status[
            (candidate_status["candidate_selector"] == "ref29_minus_boundary_fail_ref027")
            & np.isclose(candidate_status["radius"], BLOCKED_RADIUS)
        ].iloc[0].to_dict(),
        "stage_blocked": rel(PILOT_RUN_ROOT / "05_pool2_pm_sais_sampling" / "STAGE_BLOCKED.md"),
    }

    write_csv(out_dir / "claimable_phi_curve.csv", curve)
    write_csv(out_dir / "boundary_reference_diagnostics.csv", diag)
    write_csv(out_dir / "selector_qc_status.csv", inputs["qc"].sort_values("radius"))
    write_csv(out_dir / "family_cluster_assignments.csv", cluster_assignments)
    write_csv(out_dir / "candidate_selector_qc_status.csv", candidate_status)
    write_csv(out_dir / "candidate_recovery_tasks.csv", recovery_tasks)
    write_json(out_dir / "large_domain_decision.json", decision)

    plot_claimable_curve(curve, out_dir)
    plot_spaghetti(inputs["units"], diag, out_dir)
    plot_split_heatmap(inputs["units"], diag, out_dir)
    plot_family_scatter(diag, out_dir)
    plot_candidate_qc(candidate_status, out_dir)
    write_report(out_dir, args.selector, args.rule, curve, diag, candidate_status, recovery_tasks)

    print(json.dumps({**decision, "out_dir": str(out_dir)}, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
