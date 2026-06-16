from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
DEFAULT_OUTPUT_NAME = "07_reference_family_analysis"
RULES = ["low_tv_spectral_teacher", "real_even_odd", "teacher_nn", "random_label"]
COMMON_DENSE_RADII = [0.010, 0.020, 0.030, 0.050, 0.080]
SPLIT_GATE = 0.004
ESS_GATE = 0.04
BOOTSTRAP_SD_GATE = 0.012

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mnist14_model import MNIST14Arch, logits_np, normalize_labels


ARCH = MNIST14Arch(input_dim=100, hidden_width=20)
P = ARCH.param_count


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


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


def bootstrap_sd(values: np.ndarray, seed: int, n_boot: int = 300) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = np.mean(sample)
    return float(np.std(means, ddof=1))


def load_unit_summaries(run_root: Path) -> pd.DataFrame:
    unit_root = run_root / "05_pool2_pm_sais_sampling" / "unit_summaries"
    rows: list[dict[str, Any]] = []
    for path in sorted(unit_root.rglob("unit_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unit_summary_path"] = rel(path)
        rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"No unit_summary.json files found under {unit_root}")
    df = pd.DataFrame(rows)
    for col in ["split_id", "ref_id", "radius", "ess_fraction", "split_logZ_per_P_diff", "logZ_inf_full", "weighted_ce", "weighted_error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rule"] = df["rule"].astype(str)
    df["radius_key"] = df["radius"].round(4)
    return df


def load_references(run_root: Path) -> pd.DataFrame:
    path = run_root / "04_exact_reference_search" / "reference_index.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    ref_df = pd.read_csv(path)
    for col in [
        "dataset_id",
        "split_id",
        "ref_id",
        "P",
        "train_error",
        "test_error",
        "CE_mean_train",
        "CE_mean_test",
        "theta_norm",
        "min_margin",
        "q05_margin",
        "median_margin",
        "mean_margin",
    ]:
        if col in ref_df.columns:
            ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce")
    ref_df["rule"] = ref_df["rule"].astype(str)
    return ref_df


def add_delta_phi(unit_df: pd.DataFrame) -> pd.DataFrame:
    key = ["split_id", "rule", "ref_id"]
    r0_rows = unit_df[np.isclose(unit_df["radius"], 0.010)][key + ["logZ_inf_full"]].rename(columns={"logZ_inf_full": "logZ_r0"})
    joined = unit_df.merge(r0_rows, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined["logZ_inf_full"] - joined["logZ_r0"]) / float(P)
    joined["delta_phi_full_unit"] = np.where(
        joined["radius"] > 0,
        ((P - 1.0) / P) * np.log(joined["radius"] / 0.010) + joined["delta_phi_energy_unit"],
        np.nan,
    )
    joined["unit_split_pass"] = joined["split_logZ_per_P_diff"] <= SPLIT_GATE
    joined["unit_ess_pass"] = joined["ess_fraction"] >= ESS_GATE
    joined["unit_qc_pass"] = joined["unit_split_pass"] & joined["unit_ess_pass"] & np.isfinite(joined["delta_phi_energy_unit"])
    return joined


def first_fail_radius(sub: pd.DataFrame) -> float:
    failed = sub[~sub["unit_qc_pass"]].sort_values("radius")
    if failed.empty:
        return float("nan")
    return float(failed["radius"].iloc[0])


def max_pass_radius(sub: pd.DataFrame) -> float:
    passed = sub[sub["unit_qc_pass"]].sort_values("radius")
    if passed.empty:
        return float("nan")
    return float(passed["radius"].max())


def build_reference_diagnostics(ref_df: pd.DataFrame, unit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (rule, ref_id), sub in unit_df.groupby(["rule", "ref_id"]):
        ref_match = ref_df[(ref_df["rule"] == rule) & (ref_df["ref_id"] == ref_id)]
        ref = ref_match.iloc[0].to_dict() if not ref_match.empty else {}
        dense_sub = sub[sub["radius_key"].isin([round(r, 4) for r in COMMON_DENSE_RADII])].sort_values("radius")
        dense_complete = int(dense_sub["radius"].nunique()) == len(COMMON_DENSE_RADII)
        phi_values = {
            f"delta_phi_energy_r_{float(row.radius):0.4f}".replace(".", "p"): float(row.delta_phi_energy_unit)
            for row in dense_sub.itertuples()
        }
        rows.append(
            {
                "rule": str(rule),
                "split_id": int(sub["split_id"].iloc[0]),
                "ref_id": int(ref_id),
                "observed_radius_count": int(sub["radius"].nunique()),
                "observed_radius_min": float(sub["radius"].min()),
                "observed_radius_max": float(sub["radius"].max()),
                "unit_qc_fail_count": int((~sub["unit_qc_pass"]).sum()),
                "split_fail_count": int((sub["split_logZ_per_P_diff"] > SPLIT_GATE).sum()),
                "ess_fail_count": int((sub["ess_fraction"] < ESS_GATE).sum()),
                "dense_complete": bool(dense_complete),
                "dense_unit_qc_fail_count": int((~dense_sub["unit_qc_pass"]).sum()) if len(dense_sub) else len(COMMON_DENSE_RADII),
                "dense_max_split_logZ_per_P_diff": float(dense_sub["split_logZ_per_P_diff"].max()) if len(dense_sub) else float("nan"),
                "dense_min_ess_fraction": float(dense_sub["ess_fraction"].min()) if len(dense_sub) else float("nan"),
                "first_fail_radius": first_fail_radius(sub),
                "max_pass_radius": max_pass_radius(sub),
                "max_split_logZ_per_P_diff": float(sub["split_logZ_per_P_diff"].max()),
                "min_ess_fraction": float(sub["ess_fraction"].min()),
                "mean_delta_phi_energy_observed": float(sub["delta_phi_energy_unit"].mean()),
                "small_d_slope_0p01_0p08": slope_for_radii(dense_sub, 0.010, 0.080),
                "theta_norm": float(ref.get("theta_norm", np.nan)),
                "CE_mean_train": float(ref.get("CE_mean_train", np.nan)),
                "CE_mean_test": float(ref.get("CE_mean_test", np.nan)),
                "train_error": float(ref.get("train_error", np.nan)),
                "test_error": float(ref.get("test_error", np.nan)),
                "min_margin": float(ref.get("min_margin", np.nan)),
                "q05_margin": float(ref.get("q05_margin", np.nan)),
                "median_margin": float(ref.get("median_margin", np.nan)),
                "mean_margin": float(ref.get("mean_margin", np.nan)),
                "theta_path": str(ref.get("theta_path", "")),
                "dataset_path": str(ref.get("dataset_path", "")),
                **phi_values,
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["rule", "unit_qc_fail_count", "max_split_logZ_per_P_diff", "ref_id"], ascending=[True, True, True, True])
    out["hardness_rank"] = out.groupby("rule").cumcount() + 1
    return out.sort_values(["rule", "hardness_rank", "ref_id"]).reset_index(drop=True)


def slope_for_radii(sub: pd.DataFrame, r_a: float, r_b: float) -> float:
    a = sub[np.isclose(sub["radius"], r_a)]
    b = sub[np.isclose(sub["radius"], r_b)]
    if a.empty or b.empty:
        return float("nan")
    return float((b["delta_phi_energy_unit"].iloc[0] - a["delta_phi_energy_unit"].iloc[0]) / (r_b - r_a))


def add_phi_qc_clusters(ref_diag: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    out = ref_diag.copy()
    phi_cols = [c for c in out.columns if c.startswith("delta_phi_energy_r_")]
    feature_cols = phi_cols + [
        "unit_qc_fail_count",
        "max_split_logZ_per_P_diff",
        "min_ess_fraction",
        "small_d_slope_0p01_0p08",
        "theta_norm",
        "CE_mean_train",
        "min_margin",
    ]
    out["phi_qc_cluster"] = -1
    out["phi_qc_cluster_quality"] = "unassigned"
    for rule, idx in out.groupby("rule").groups.items():
        sub = out.loc[idx, feature_cols].replace([np.inf, -np.inf], np.nan)
        sub = sub.fillna(sub.median(numeric_only=True)).fillna(0.0)
        k = min(int(n_clusters), len(sub))
        if k <= 1:
            out.loc[idx, "phi_qc_cluster"] = 0
            out.loc[idx, "phi_qc_cluster_quality"] = "single_cluster"
            continue
        x = StandardScaler().fit_transform(sub.to_numpy(dtype=np.float64))
        labels = KMeans(n_clusters=k, random_state=20260615, n_init=30).fit_predict(x)
        out.loc[idx, "phi_qc_cluster"] = labels
        cluster_stats = (
            out.loc[idx]
            .assign(_label=labels)
            .groupby("_label")
            .agg(fails=("unit_qc_fail_count", "mean"), max_split=("max_split_logZ_per_P_diff", "mean"), count=("ref_id", "size"))
            .sort_values(["fails", "max_split"], ascending=[True, True])
        )
        quality_by_cluster = {}
        if len(cluster_stats):
            quality_by_cluster[int(cluster_stats.index[0])] = "stable_candidate"
            if len(cluster_stats) > 1:
                quality_by_cluster[int(cluster_stats.index[-1])] = "hard_candidate"
            for label in cluster_stats.index:
                quality_by_cluster.setdefault(int(label), "intermediate")
        out.loc[idx, "phi_qc_cluster_quality"] = [quality_by_cluster[int(label)] for label in labels]
    return out


def add_function_clusters(ref_diag: pd.DataFrame, run_root: Path, n_clusters: int, *, skip_function: bool = False) -> pd.DataFrame:
    out = ref_diag.copy()
    out["function_cluster"] = -1
    out["function_pc1"] = np.nan
    out["function_pc2"] = np.nan
    if skip_function:
        return out
    for rule, group in out.groupby("rule"):
        dataset_path = group["dataset_path"].dropna().astype(str).iloc[0]
        ds_path = (REPO_ROOT / dataset_path).resolve() if not Path(dataset_path).is_absolute() else Path(dataset_path)
        if not ds_path.exists():
            continue
        ds = np.load(ds_path)
        x = ds["X_train"]
        y = normalize_labels(ds["y_train"])
        signatures: list[np.ndarray] = []
        valid_indices: list[int] = []
        for idx, row in group.iterrows():
            theta_path = str(row["theta_path"])
            path = (REPO_ROOT / theta_path).resolve() if not Path(theta_path).is_absolute() else Path(theta_path)
            if not path.exists():
                continue
            theta = np.load(path)
            logits = logits_np(theta, x, arch=ARCH)
            margins = y * logits
            centered = margins - float(np.mean(margins))
            scale = float(np.std(centered))
            signatures.append(centered / scale if scale > 1.0e-12 else centered)
            valid_indices.append(idx)
        if len(signatures) < 2:
            continue
        x_sig = np.vstack(signatures)
        pca = PCA(n_components=min(5, x_sig.shape[0], x_sig.shape[1]), random_state=20260615)
        pcs = pca.fit_transform(x_sig)
        k = min(int(n_clusters), len(signatures))
        labels = KMeans(n_clusters=k, random_state=20260615, n_init=30).fit_predict(StandardScaler().fit_transform(pcs))
        out.loc[valid_indices, "function_cluster"] = labels
        out.loc[valid_indices, "function_pc1"] = pcs[:, 0]
        if pcs.shape[1] > 1:
            out.loc[valid_indices, "function_pc2"] = pcs[:, 1]
    return out


def build_selector_membership(ref_diag: pd.DataFrame, selector_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selectors = {
        "optimizer_first30_ref30": ("ref_id", True),
        "l2_min_norm_ref30": ("theta_norm", True),
        "high_margin_ref30": ("min_margin", False),
        "qc_hardness_stable_ref30": ("hardness_rank", True),
    }
    for rule, group in ref_diag.groupby("rule"):
        for selector, (sort_col, ascending) in selectors.items():
            selected = group.sort_values([sort_col, "ref_id"], ascending=[ascending, True]).head(selector_size)
            for row in selected.itertuples():
                rows.append(
                    {
                        "selector": selector,
                        "rule": rule,
                        "ref_id": int(row.ref_id),
                        "selector_role": "predeclared" if selector in {"optimizer_first30_ref30", "l2_min_norm_ref30", "high_margin_ref30"} else "diagnostic_posthoc",
                    }
                )
        dense_selected = group.sort_values(
            ["dense_complete", "dense_unit_qc_fail_count", "dense_max_split_logZ_per_P_diff", "ref_id"],
            ascending=[False, True, True, True],
        ).head(selector_size)
        for row in dense_selected.itertuples():
            rows.append(
                {
                    "selector": "dense_qc_stable_ref30",
                    "rule": rule,
                    "ref_id": int(row.ref_id),
                    "selector_role": "predeclared_dense_only",
                }
            )
        stable = group[group["phi_qc_cluster_quality"] == "stable_candidate"].sort_values(["hardness_rank", "ref_id"])
        if len(stable) >= selector_size:
            cluster_selected = stable.head(selector_size)
        else:
            cluster_selected = pd.concat([stable, group[~group.index.isin(stable.index)].sort_values(["hardness_rank", "ref_id"])]).head(selector_size)
        for row in cluster_selected.itertuples():
            rows.append(
                {
                    "selector": "phi_qc_cluster_stable_ref30",
                    "rule": rule,
                    "ref_id": int(row.ref_id),
                    "selector_role": "diagnostic_posthoc",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["selector", "rule", "ref_id"]).sort_values(["selector", "rule", "ref_id"])


def selector_qc(unit_df: pd.DataFrame, selectors: pd.DataFrame, selector_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_rows: list[dict[str, Any]] = []
    phi_rows: list[dict[str, Any]] = []
    all_radii = sorted(float(r) for r in unit_df["radius"].dropna().unique())
    for (selector, rule), refs_sub in selectors.groupby(["selector", "rule"]):
        ref_ids = sorted(int(r) for r in refs_sub["ref_id"].unique())
        selected_units = unit_df[(unit_df["rule"] == rule) & (unit_df["ref_id"].isin(ref_ids))].copy()
        for radius in all_radii:
            sub = selected_units[np.isclose(selected_units["radius"], radius)].copy()
            observed_refs = sorted(int(r) for r in sub["ref_id"].dropna().unique())
            missing_refs = sorted(set(ref_ids) - set(observed_refs))
            complete = len(observed_refs) == selector_size
            finite_fraction = float(np.mean(np.isfinite(sub["delta_phi_energy_unit"]))) if len(sub) else 0.0
            q05_ess = float(np.quantile(sub["ess_fraction"].dropna(), 0.05)) if len(sub) else float("nan")
            max_split = float(sub["split_logZ_per_P_diff"].max()) if len(sub) else float("nan")
            boot_sd = bootstrap_sd(sub["delta_phi_energy_unit"].to_numpy(), seed=771000 + int(round(radius * 10000)) + len(selector) * 97) if len(sub) else float("nan")
            observed_qc_fail = bool(
                len(sub)
                and (
                    finite_fraction < 0.90
                    or (np.isfinite(q05_ess) and q05_ess < ESS_GATE)
                    or (np.isfinite(max_split) and max_split > SPLIT_GATE)
                    or (np.isfinite(boot_sd) and boot_sd > BOOTSTRAP_SD_GATE)
                )
            )
            pass_qc = bool(
                complete
                and finite_fraction >= 0.90
                and np.isfinite(q05_ess)
                and q05_ess >= ESS_GATE
                and np.isfinite(max_split)
                and max_split <= SPLIT_GATE
                and np.isfinite(boot_sd)
                and boot_sd <= BOOTSTRAP_SD_GATE
            )
            if pass_qc:
                claim_status = "claimable_selector_radius"
            elif not complete and observed_qc_fail:
                claim_status = "missing_units_and_qc_fail"
            elif not complete:
                claim_status = "missing_units"
            else:
                claim_status = "no_claim_qc_fail"
            mean_delta = float(sub["delta_phi_energy_unit"].mean()) if len(sub) else float("nan")
            mean_full = float(sub["delta_phi_full_unit"].mean()) if len(sub) else float("nan")
            qc_rows.append(
                {
                    "selector": selector,
                    "rule": rule,
                    "radius": float(radius),
                    "selected_ref_count": int(selector_size),
                    "observed_ref_count": int(len(observed_refs)),
                    "missing_ref_count": int(len(missing_refs)),
                    "missing_refs": ";".join(str(r) for r in missing_refs[:20]),
                    "complete": bool(complete),
                    "finite_unit_fraction": finite_fraction,
                    "q05_ess_fraction": q05_ess,
                    "max_split_logZ_per_P_diff": max_split,
                    "bootstrap_sd_phi": boot_sd,
                    "observed_qc_fail": observed_qc_fail,
                    "qc_pass": pass_qc,
                    "claim_status": claim_status,
                }
            )
            phi_rows.append(
                {
                    "selector": selector,
                    "rule": rule,
                    "radius": float(radius),
                    "delta_phi_energy": mean_delta,
                    "delta_phi_full": mean_full,
                    "n_units": int(len(sub)),
                    "qc_pass": pass_qc,
                    "claim_status": claim_status,
                }
            )
    return pd.DataFrame(qc_rows), pd.DataFrame(phi_rows)


def large_domain_decision(selector_qc_df: pd.DataFrame, ref_diag: pd.DataFrame) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    large_radii = [0.450, 0.650, 0.850, 1.000, 1.250, 1.500, 1.750, 2.000, 2.250, 2.500]
    for (selector, rule), sub in selector_qc_df.groupby(["selector", "rule"]):
        large = sub[sub["radius"].isin(large_radii)]
        complete_large = bool(len(large) == len(large_radii) and large["complete"].all())
        pass_large = bool(complete_large and large["qc_pass"].all())
        passed = sub[sub["qc_pass"]].sort_values("radius")
        max_pass = float(passed["radius"].max()) if not passed.empty else float("nan")
        decisions.append(
            {
                "selector": selector,
                "rule": rule,
                "complete_large_domain": complete_large,
                "all_large_domain_qc_pass": pass_large,
                "max_qc_pass_radius_observed": max_pass,
                "large_domain_missing_or_failed_rows": int(len(large) - int(large["qc_pass"].sum())) if len(large) else len(large_radii),
            }
        )
    decision_df = pd.DataFrame(decisions)
    predeclared = decision_df[
        decision_df["selector"].isin(["optimizer_first30_ref30", "l2_min_norm_ref30", "high_margin_ref30", "dense_qc_stable_ref30"])
    ]
    supported = predeclared[predeclared["all_large_domain_qc_pass"]]
    lowtv_supported = supported[supported["rule"] == "low_tv_spectral_teacher"]
    hard_refs = (
        ref_diag[ref_diag["rule"] == "low_tv_spectral_teacher"]
        .sort_values(["unit_qc_fail_count", "max_split_logZ_per_P_diff"], ascending=[False, False])
        .head(10)[["ref_id", "unit_qc_fail_count", "first_fail_radius", "max_split_logZ_per_P_diff", "theta_norm", "min_margin"]]
        .to_dict("records")
    )
    return {
        "large_domain_supported_for_predeclared_lowtv_ref30": bool(not lowtv_supported.empty),
        "decision_table": decisions,
        "criteria": {
            "selector_size": 30,
            "large_radii": large_radii,
            "required": "predeclared selector has complete selected-reference units and all QC gates pass at every large radius",
            "split_gate": SPLIT_GATE,
            "ess_gate": ESS_GATE,
            "bootstrap_sd_gate": BOOTSTRAP_SD_GATE,
        },
        "next_safe_action": (
            "Proceed to sparse large-domain production only for a predeclared selector whose large-domain rows are complete and QC-pass."
            if not lowtv_supported.empty
            else "Do not launch sparse large-domain production from the current evidence. First define a predeclared reference law such as l2_min_norm_ref30, run a targeted Stage05 pilot on its missing/hard large-radii units, and only promote if complete selector-level QC passes."
        ),
        "hard_lowtv_reference_examples": hard_refs,
    }


def plot_lowtv_spaghetti(unit_df: pd.DataFrame, ref_diag: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    low_units = unit_df[unit_df["rule"] == "low_tv_spectral_teacher"].copy()
    low_diag = ref_diag[ref_diag["rule"] == "low_tv_spectral_teacher"].set_index("ref_id")
    color_map = {"stable_candidate": "#2a7f62", "intermediate": "#5f6f91", "hard_candidate": "#ba3f3f", "unassigned": "#777777"}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for ref_id, sub in low_units.groupby("ref_id"):
        quality = str(low_diag.loc[int(ref_id), "phi_qc_cluster_quality"]) if int(ref_id) in low_diag.index else "unassigned"
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["delta_phi_energy_unit"], color=color_map.get(quality, "#777777"), alpha=0.42, linewidth=0.9)
        failed = sub[~sub["unit_qc_pass"]]
        if not failed.empty:
            ax.scatter(failed["radius"], failed["delta_phi_energy_unit"], marker="x", s=18, color="#ba3f3f", alpha=0.75)
    ax.axvline(0.08, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("reference-level delta phi(d)_energy")
    ax.set_title("low_tv_spectral_teacher reference phi(d)_energy profiles")
    handles = [plt.Line2D([0], [0], color=v, lw=2, label=k) for k, v in color_map.items() if k != "unassigned"]
    ax.legend(handles=handles, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_lowtv_reference_phi_energy_spaghetti.png", dpi=180)
    plt.close(fig)


def plot_lowtv_split_heatmap(unit_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    low_units = unit_df[unit_df["rule"] == "low_tv_spectral_teacher"].copy()
    pivot = low_units.pivot_table(index="ref_id", columns="radius", values="split_logZ_per_P_diff", aggfunc="max")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    im = ax.imshow(np.minimum(pivot.to_numpy(dtype=np.float64), 0.012), aspect="auto", interpolation="nearest", vmin=0.0, vmax=0.012, cmap="magma")
    ax.set_xlabel("d_raw")
    ax.set_ylabel("reference id")
    ax.set_title("low_tv_spectral_teacher split logZ/P by reference and radius")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" if c >= 0.1 else f"{c:.3f}" for c in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, len(pivot.index), 5))
    ax.set_yticklabels([str(int(pivot.index[i])) for i in range(0, len(pivot.index), 5)], fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("split logZ/P diff, capped at 0.012")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_lowtv_split_logz_heatmap.png", dpi=180)
    plt.close(fig)


def plot_selector_phi(selector_phi_df: pd.DataFrame, selector_qc_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    focus = selector_phi_df[selector_phi_df["rule"] == "low_tv_spectral_teacher"].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = {
        "l2_min_norm_ref30": "#2563a6",
        "high_margin_ref30": "#7c5c1d",
        "qc_hardness_stable_ref30": "#2a7f62",
        "phi_qc_cluster_stable_ref30": "#7b3f98",
    }
    for selector, sub in focus.groupby("selector"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["delta_phi_energy"], marker="o", markersize=3, linewidth=1.2, color=colors.get(selector, None), label=selector)
        no_claim = sub[~sub["qc_pass"]]
        if not no_claim.empty:
            ax.scatter(no_claim["radius"], no_claim["delta_phi_energy"], marker="x", color=colors.get(selector, "#555555"), s=28)
    ax.axvline(0.08, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("selector mean delta phi(d)_energy")
    ax.set_title("low_tv_spectral_teacher selector-level phi(d)_energy diagnostics")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_lowtv_selector_phi_energy.png", dpi=180)
    plt.close(fig)

    low_qc = selector_qc_df[selector_qc_df["rule"] == "low_tv_spectral_teacher"].copy()
    if low_qc.empty:
        return
    pivot = low_qc.pivot_table(index="selector", columns="radius", values="qc_pass", aggfunc="max").fillna(False).astype(float)
    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("d_raw")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" if c >= 0.1 else f"{c:.3f}" for c in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_title("low_tv selector-level QC pass map")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_lowtv_selector_qc_pass_map.png", dpi=180)
    plt.close(fig)


def plot_cluster_scatter(ref_diag: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    low = ref_diag[ref_diag["rule"] == "low_tv_spectral_teacher"].copy()
    if low.empty:
        return
    color_map = {"stable_candidate": "#2a7f62", "intermediate": "#5f6f91", "hard_candidate": "#ba3f3f", "unassigned": "#777777"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for quality, sub in low.groupby("phi_qc_cluster_quality"):
        axes[0].scatter(sub["theta_norm"], sub["max_split_logZ_per_P_diff"], s=36, alpha=0.85, color=color_map.get(quality, "#777777"), label=quality)
    axes[0].axhline(SPLIT_GATE, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("theta norm")
    axes[0].set_ylabel("max split logZ/P")
    axes[0].legend(fontsize=7, frameon=False)
    valid = low[np.isfinite(low["function_pc1"]) & np.isfinite(low["function_pc2"])]
    for quality, sub in valid.groupby("phi_qc_cluster_quality"):
        axes[1].scatter(sub["function_pc1"], sub["function_pc2"], s=36, alpha=0.85, color=color_map.get(quality, "#777777"), label=quality)
    axes[1].set_xlabel("function PC1")
    axes[1].set_ylabel("function PC2")
    fig.suptitle("low_tv reference family diagnostics")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_lowtv_reference_family_scatter.png", dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_report(
    out_dir: Path,
    run_root: Path,
    ref_diag: pd.DataFrame,
    selector_qc_df: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    low = ref_diag[ref_diag["rule"] == "low_tv_spectral_teacher"]
    hard = low.sort_values(["unit_qc_fail_count", "max_split_logZ_per_P_diff"], ascending=[False, False]).head(8)
    selector_summary_rows = []
    for (selector, rule), sub in selector_qc_df.groupby(["selector", "rule"]):
        passed = sub[sub["qc_pass"]]
        selector_summary_rows.append(
            [
                selector,
                rule,
                int(len(passed)),
                f"{float(passed['radius'].max()):.4f}" if not passed.empty else "none",
                int(sub["claim_status"].isin(["missing_units", "missing_units_and_qc_fail"]).sum()),
                int(sub["claim_status"].isin(["no_claim_qc_fail", "missing_units_and_qc_fail"]).sum()),
            ]
        )
    hard_rows = [
        [
            int(row.ref_id),
            int(row.unit_qc_fail_count),
            f"{float(row.first_fail_radius):.4f}" if np.isfinite(row.first_fail_radius) else "none",
            f"{float(row.max_split_logZ_per_P_diff):.6f}",
            f"{float(row.theta_norm):.3f}",
            f"{float(row.min_margin):.3f}",
        ]
        for row in hard.itertuples()
    ]
    report = f"""# MNIST10 Reference Family Analysis

Run root: `{rel(run_root)}`

This analysis decomposes the optimizer-induced exact reference ensemble into reference-level phi(d) and QC behavior. It is diagnostic unless a selector is explicitly marked predeclared.

## Main Finding

Large-domain sparse production is **not supported by the current evidence** for a predeclared 30-reference low_tv selector.

Decision: `{decision["large_domain_supported_for_predeclared_lowtv_ref30"]}`.

Next safe action: {decision["next_safe_action"]}

## Selector QC Summary

{markdown_table(["selector", "rule", "qc-pass rows", "max pass radius", "missing rows", "QC-fail rows"], selector_summary_rows)}

## Hard low_tv Reference Examples

{markdown_table(["ref", "fail count", "first fail d", "max split", "theta norm", "min margin"], hard_rows)}

## Outputs

{markdown_table(
    ["artifact", "path"],
    [
        ["reference diagnostics", "reference_diagnostics.csv"],
        ["selector membership", "selector_membership.csv"],
        ["selector QC", "selector_qc_by_rule_radius.csv"],
        ["selector phi", "selector_phi_by_rule_radius.csv"],
        ["large-domain decision", "large_domain_decision.json"],
        ["low_tv phi spaghetti", "figures/fig01_lowtv_reference_phi_energy_spaghetti.png"],
        ["low_tv split heatmap", "figures/fig02_lowtv_split_logz_heatmap.png"],
        ["low_tv selector phi", "figures/fig03_lowtv_selector_phi_energy.png"],
        ["low_tv selector QC map", "figures/fig04_lowtv_selector_qc_pass_map.png"],
        ["low_tv family scatter", "figures/fig05_lowtv_reference_family_scatter.png"],
    ],
)}

## Interpretation

The current 60-reference low_tv ensemble behaves like a mixture of effective reference families. Several references repeatedly fail split-logZ stability despite acceptable ESS, so simply increasing sample count is not enough evidence for a large-domain all-reference claim. The optimizer-first, l2-min-norm, high-margin, and dense-QC-stable 30-reference selectors are predeclared diagnostics here; any production claim must rerun or complete their missing large-radius units and pass selector-level QC before promotion.
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def write_stage_blocked_if_needed(out_dir: Path, selector_qc_df: pd.DataFrame, decision: dict[str, Any]) -> None:
    if bool(decision["large_domain_supported_for_predeclared_lowtv_ref30"]):
        blocked_path = out_dir / "STAGE_BLOCKED.md"
        if blocked_path.exists():
            blocked_path.unlink()
        return
    low_first30 = selector_qc_df[
        (selector_qc_df["selector"] == "optimizer_first30_ref30")
        & (selector_qc_df["rule"] == "low_tv_spectral_teacher")
        & (selector_qc_df["claim_status"].isin(["no_claim_qc_fail", "missing_units_and_qc_fail"]))
    ].sort_values("radius")
    low_l2 = selector_qc_df[
        (selector_qc_df["selector"] == "l2_min_norm_ref30")
        & (selector_qc_df["rule"] == "low_tv_spectral_teacher")
        & (selector_qc_df["radius"] >= 0.45)
    ].sort_values("radius")
    first_fail = low_first30.iloc[0].to_dict() if not low_first30.empty else {}
    l2_first_large = low_l2.iloc[0].to_dict() if not low_l2.empty else {}
    text = f"""# STAGE_BLOCKED

Stage: `07_reference_family_analysis` sparse large-domain continuation decision.

## Exact Failing Condition

Sparse large-domain production was not launched because no predeclared 30-reference low_tv selector has complete and QC-passing evidence over the large-domain radii.

## Observed

- `large_domain_supported_for_predeclared_lowtv_ref30`: `{decision["large_domain_supported_for_predeclared_lowtv_ref30"]}`
- `optimizer_first30_ref30` first low_tv no-claim radius: `{first_fail.get("radius", "n/a")}`
- `optimizer_first30_ref30` first low_tv no-claim status: `{first_fail.get("claim_status", "n/a")}`
- `optimizer_first30_ref30` first low_tv no-claim max split logZ/P: `{first_fail.get("max_split_logZ_per_P_diff", "n/a")}`
- split gate: `{SPLIT_GATE}`
- `l2_min_norm_ref30` first large-radius observed refs: `{l2_first_large.get("observed_ref_count", "n/a")} / 30`
- `l2_min_norm_ref30` first large-radius missing refs: `{l2_first_large.get("missing_ref_count", "n/a")}`
- `l2_min_norm_ref30` first large-radius status: `{l2_first_large.get("claim_status", "n/a")}`

## Expected

- A predeclared 30-reference low_tv selector must have complete selected-reference units at every large radius.
- Every selected selector/radius row must satisfy split logZ/P <= `{SPLIT_GATE}`, q05 ESS >= `{ESS_GATE}`, and bootstrap sd phi <= `{BOOTSTRAP_SD_GATE}`.

## Next Safe Action

{decision["next_safe_action"]}

Do not promote sparse large-domain phi(d)_energy figures from incomplete/no-claim rows.
"""
    (out_dir / "STAGE_BLOCKED.md").write_text(text, encoding="utf-8")


def run_analysis(run_root: Path, out_dir: Path, *, selector_size: int, clusters: int, skip_function: bool) -> dict[str, Any]:
    ensure_dir(out_dir)
    ref_df = load_references(run_root)
    unit_df = add_delta_phi(load_unit_summaries(run_root))
    ref_diag = build_reference_diagnostics(ref_df, unit_df)
    ref_diag = add_phi_qc_clusters(ref_diag, n_clusters=clusters)
    ref_diag = add_function_clusters(ref_diag, run_root, n_clusters=clusters, skip_function=skip_function)
    selectors = build_selector_membership(ref_diag, selector_size=selector_size)
    selector_qc_df, selector_phi_df = selector_qc(unit_df, selectors, selector_size=selector_size)
    decision = large_domain_decision(selector_qc_df, ref_diag)

    write_csv(out_dir / "unit_summary_long.csv", unit_df)
    write_csv(out_dir / "reference_index_input.csv", ref_df)
    write_csv(out_dir / "reference_diagnostics.csv", ref_diag)
    write_csv(out_dir / "selector_membership.csv", selectors)
    write_csv(out_dir / "selector_qc_by_rule_radius.csv", selector_qc_df)
    write_csv(out_dir / "selector_phi_by_rule_radius.csv", selector_phi_df)
    write_json(out_dir / "large_domain_decision.json", decision)
    write_json(
        out_dir / "run_config_resolved.json",
        {
            "run_root": rel(run_root),
            "output_dir": rel(out_dir),
            "selector_size": selector_size,
            "clusters": clusters,
            "skip_function": skip_function,
            "split_gate": SPLIT_GATE,
            "ess_gate": ESS_GATE,
            "bootstrap_sd_gate": BOOTSTRAP_SD_GATE,
            "P": P,
            "common_dense_radii": COMMON_DENSE_RADII,
        },
    )

    plot_lowtv_spaghetti(unit_df, ref_diag, out_dir)
    plot_lowtv_split_heatmap(unit_df, out_dir)
    plot_selector_phi(selector_phi_df, selector_qc_df, out_dir)
    plot_cluster_scatter(ref_diag, out_dir)
    write_report(out_dir, run_root, ref_diag, selector_qc_df, decision)
    write_stage_blocked_if_needed(out_dir, selector_qc_df, decision)
    return {
        "references": int(len(ref_df)),
        "unit_rows": int(len(unit_df)),
        "output_dir": rel(out_dir),
        "large_domain_supported": bool(decision["large_domain_supported_for_predeclared_lowtv_ref30"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--selector-size", type=int, default=30)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--skip-function", action="store_true")
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    out_dir = args.output_dir.resolve() if args.output_dir else run_root / DEFAULT_OUTPUT_NAME
    result = run_analysis(run_root, out_dir, selector_size=args.selector_size, clusters=args.clusters, skip_function=args.skip_function)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
