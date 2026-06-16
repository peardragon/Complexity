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
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mnist10_reference_family_analysis import P, SPLIT_GATE, ensure_dir, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_ref30_selector_targeted_pilot"
SOURCE_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
OUT_DIR = PILOT_ROOT / "07_failure_mode_diagnostics" / "lowtv_d0p85_ref027_033_049"
TARGET_REFS = [27, 33, 49]
RULE = "low_tv_spectral_teacher"
RADIUS = 0.85


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


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(logsumexp(values) - math.log(values.size)) if values.size else float("nan")


def load_payload(path: Path, source: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_artifact_path"] = rel(path)
    payload["_artifact_source"] = source
    return payload


def d085_paths() -> list[tuple[int, Path, str]]:
    out: list[tuple[int, Path, str]] = []
    for ref_dir in sorted((PILOT_ROOT / "05_pool2_pm_sais_sampling" / "unit_summaries" / "split_000" / RULE).glob("ref_*")):
        try:
            ref_id = int(ref_dir.name.split("_")[1])
        except Exception:
            continue
        out.append((ref_id, ref_dir / "r_0p8500" / "unit_summary.json", "targeted_pilot"))
    for ref_dir in sorted((SOURCE_ROOT / "05_pool2_pm_sais_sampling" / "unit_summaries" / "split_000" / RULE).glob("ref_*")):
        try:
            ref_id = int(ref_dir.name.split("_")[1])
        except Exception:
            continue
        out.append((ref_id, ref_dir / "r_0p8500" / "unit_summary.json", "source_sparse"))
    return out


def load_d085_payloads() -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    # Prefer targeted pilot, because it contains the strengthened 16-replicate summaries.
    source_rank = {"source_sparse": 0, "targeted_pilot": 1}
    for ref_id, path, source in d085_paths():
        payload = load_payload(path, source)
        if payload is None:
            continue
        current = payloads.get(ref_id)
        if current is None or source_rank[source] >= source_rank[str(current["_artifact_source"])]:
            payloads[ref_id] = payload
    return payloads


def four_blocks(values: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return [values[idx] for idx in np.array_split(np.arange(values.size), 4)]


def summarize_payload(ref_id: int, payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reps = payload.get("replicate_summaries") or []
    rep_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    if not isinstance(reps, list) or len(reps) < 4:
        return (
            {
                "ref_id": ref_id,
                "artifact_source": payload.get("_artifact_source"),
                "artifact_path": payload.get("_artifact_path"),
                "has_replicate_summaries": False,
                "replicates": int(payload.get("replicates", 0) or 0),
                "overall_split_logZ_per_P_diff": float(payload.get("split_logZ_per_P_diff", float("nan"))),
                "diagnostic_status": "insufficient_retained_summary",
            },
            rep_rows,
            block_rows,
        )

    for rep in reps:
        prior = float(rep.get("reference_prior_log_weight", payload.get("reference_prior_log_weight", 0.0)))
        rep_rows.append(
            {
                "ref_id": ref_id,
                "replicate_id": int(rep["replicate_id"]),
                "seed": int(rep["seed"]),
                "logZ_inf_full": float(rep["logZ_inf_full"]),
                "logZ_per_P": float(rep["logZ_inf_full"]) / float(P),
                "split0_logZ_full_proxy": float(rep["split0_logZ"]) + prior,
                "split1_logZ_full_proxy": float(rep["split1_logZ"]) + prior,
                "replicate_split_logZ_per_P_diff": float(rep["split_logZ_per_P_diff"]),
                "ess_fraction": float(rep["ess_fraction"]),
                "weighted_ce": float(rep["weighted_ce"]),
                "weighted_error": float(rep["weighted_error"]),
                "weighted_h": float(rep["weighted_h"]),
                "smc_mean_mh_acceptance": float(rep["smc_mean_mh_acceptance"]),
                "smc_step_count": int(rep["smc_step_count"]),
            }
        )
    rep_df = pd.DataFrame(rep_rows).sort_values("replicate_id")
    full = rep_df["logZ_inf_full"].to_numpy(dtype=np.float64)
    ce = rep_df["weighted_ce"].to_numpy(dtype=np.float64)
    split = rep_df["replicate_split_logZ_per_P_diff"].to_numpy(dtype=np.float64)
    ess = rep_df["ess_fraction"].to_numpy(dtype=np.float64)
    even = full[0::2]
    odd = full[1::2]
    quarter_logz = [logmeanexp(block) for block in four_blocks(full)]
    quarter_ce = [float(np.mean(block)) for block in four_blocks(ce)]
    half_values = []
    for row in rep_df.sort_values("replicate_id").itertuples():
        half_values.append(float(row.split0_logZ_full_proxy))
        half_values.append(float(row.split1_logZ_full_proxy))
    half_values_np = np.asarray(half_values, dtype=np.float64)
    half_quarter_logz = [logmeanexp(block) for block in four_blocks(half_values_np)]

    for i, value in enumerate(quarter_logz):
        block_rows.append(
            {
                "ref_id": ref_id,
                "block_kind": "replicate_quarter",
                "block_id": i,
                "block_logZ": value,
                "block_logZ_centered_per_P": (value - logmeanexp(full)) / float(P),
                "block_weighted_ce_mean": quarter_ce[i],
            }
        )
    for i, value in enumerate(half_quarter_logz):
        block_rows.append(
            {
                "ref_id": ref_id,
                "block_kind": "half_split_quarter",
                "block_id": i,
                "block_logZ": value,
                "block_logZ_centered_per_P": (value - logmeanexp(half_values_np)) / float(P),
                "block_weighted_ce_mean": float("nan"),
            }
        )

    features = rep_df[["logZ_per_P", "weighted_ce", "weighted_error", "weighted_h", "ess_fraction", "replicate_split_logZ_per_P_diff"]].to_numpy(dtype=np.float64)
    features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=2, random_state=20260616, n_init=20).fit(features)
    labels = kmeans.labels_
    silhouette = float(silhouette_score(features, labels)) if len(np.unique(labels)) == 2 and len(rep_df) > 3 else float("nan")
    cluster_delta_ce = float(abs(np.mean(ce[labels == 0]) - np.mean(ce[labels == 1]))) if len(np.unique(labels)) == 2 else float("nan")
    cluster_delta_logz_per_P = (
        float(abs(logmeanexp(full[labels == 0]) - logmeanexp(full[labels == 1])) / float(P))
        if len(np.unique(labels)) == 2
        else float("nan")
    )
    logz_ce_corr = float(np.corrcoef(full, ce)[0, 1]) if len(full) > 2 and np.std(full) > 0 and np.std(ce) > 0 else float("nan")

    rep_range = float((np.max(full) - np.min(full)) / float(P))
    quarter_range = float((np.max(quarter_logz) - np.min(quarter_logz)) / float(P))
    half_quarter_range = float((np.max(half_quarter_logz) - np.min(half_quarter_logz)) / float(P))
    even_odd = float(abs(logmeanexp(even) - logmeanexp(odd)) / float(P))
    target_failed = ref_id in TARGET_REFS
    multi_sector_score = 0
    if quarter_range > SPLIT_GATE:
        multi_sector_score += 1
    if half_quarter_range > SPLIT_GATE:
        multi_sector_score += 1
    if silhouette >= 0.45 and (cluster_delta_logz_per_P > 0.002 or cluster_delta_ce > 0.01):
        multi_sector_score += 1
    if abs(logz_ce_corr) >= 0.65 and rep_range > SPLIT_GATE:
        multi_sector_score += 1
    if multi_sector_score >= 2:
        status = "multi_sector_suspect"
    elif even_odd > SPLIT_GATE and quarter_range <= SPLIT_GATE and half_quarter_range <= SPLIT_GATE:
        status = "split_noise_or_pairing_artifact_suspect"
    else:
        status = "ambiguous_summary_level"

    summary = {
        "ref_id": ref_id,
        "target_failed_ref": bool(target_failed),
        "artifact_source": payload.get("_artifact_source"),
        "artifact_path": payload.get("_artifact_path"),
        "has_replicate_summaries": True,
        "replicates": int(len(rep_df)),
        "n_samples_each": int(payload.get("n_samples_each", 0) or 0),
        "n_samples_total": int(payload.get("n_samples_total", 0) or 0),
        "overall_split_logZ_per_P_diff": float(payload.get("split_logZ_per_P_diff", float("nan"))),
        "even_odd_replicate_split_per_P_recomputed": even_odd,
        "replicate_logZ_range_per_P": rep_range,
        "replicate_quarter_logZ_range_per_P": quarter_range,
        "half_split_quarter_logZ_range_per_P": half_quarter_range,
        "replicate_split_diff_max": float(np.max(split)),
        "replicate_split_diff_median": float(np.median(split)),
        "ess_fraction_mean": float(np.mean(ess)),
        "ess_fraction_min": float(np.min(ess)),
        "weighted_ce_mean": float(np.mean(ce)),
        "weighted_ce_range": float(np.max(ce) - np.min(ce)),
        "logZ_weightedCE_corr": logz_ce_corr,
        "feature_kmeans2_silhouette": silhouette,
        "feature_kmeans2_delta_ce": cluster_delta_ce,
        "feature_kmeans2_delta_logZ_per_P": cluster_delta_logz_per_P,
        "multi_sector_score": int(multi_sector_score),
        "diagnostic_status": status,
    }
    return summary, rep_rows, block_rows


def plot_four_split(summary_df: pd.DataFrame, block_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    target_blocks = block_df[block_df["ref_id"].isin(TARGET_REFS)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), sharey=True)
    for ax, ref_id in zip(axes, TARGET_REFS):
        sub = target_blocks[target_blocks["ref_id"] == ref_id]
        for kind, marker, label in [
            ("replicate_quarter", "o", "replicate 4-split"),
            ("half_split_quarter", "s", "half 4-split"),
        ]:
            ks = sub[sub["block_kind"] == kind].sort_values("block_id")
            ax.plot(ks["block_id"], ks["block_logZ_centered_per_P"], marker=marker, linewidth=1.2, label=label)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        ax.axhline(SPLIT_GATE, color="#b23a48", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.axhline(-SPLIT_GATE, color="#b23a48", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(f"ref_{ref_id:03d}")
        ax.set_xlabel("block id")
        ax.grid(True, linewidth=0.35, alpha=0.25)
    axes[0].set_ylabel("block logZ centered / P")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_target_refs_4split_logZ_blocks.png", dpi=190)
    plt.close(fig)

    plot_df = summary_df[summary_df["has_replicate_summaries"]].sort_values("replicate_quarter_logZ_range_per_P", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#b23a48" if int(ref) in TARGET_REFS else "#5f6f91" for ref in plot_df["ref_id"]]
    ax.bar([f"{int(r):03d}" for r in plot_df["ref_id"]], plot_df["replicate_quarter_logZ_range_per_P"], color=colors)
    ax.axhline(SPLIT_GATE, color="black", linestyle="--", linewidth=1.0, label="split gate")
    ax.set_xlabel("ref id")
    ax.set_ylabel("replicate 4-split logZ range / P")
    ax.set_title("d=0.85 replicate 4-split range rank")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_all_refs_4split_range_rank.png", dpi=190)
    plt.close(fig)


def plot_ce_projection_hist(rep_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    target = rep_df[rep_df["ref_id"].isin(TARGET_REFS)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, ref_id in zip(axes, TARGET_REFS):
        sub = target[target["ref_id"] == ref_id].sort_values("replicate_id").copy()
        center = logmeanexp(sub["logZ_inf_full"].to_numpy(dtype=np.float64))
        q = sub["replicate_id"].to_numpy() // 4
        sc = ax.scatter((sub["logZ_inf_full"] - center) / float(P), sub["weighted_ce"], c=q, cmap="viridis", s=42)
        ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(f"ref_{ref_id:03d}")
        ax.set_xlabel("replicate logZ centered / P")
        ax.set_ylabel("weighted CE")
        ax.grid(True, linewidth=0.35, alpha=0.25)
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="replicate quarter")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_target_refs_CE_vs_logZ.png", dpi=190)
    plt.close(fig)

    feature_cols = ["logZ_per_P", "weighted_ce", "weighted_error", "weighted_h", "ess_fraction", "replicate_split_logZ_per_P_diff"]
    clean = rep_df.dropna(subset=feature_cols).copy()
    x = StandardScaler().fit_transform(clean[feature_cols].to_numpy(dtype=np.float64))
    pcs = PCA(n_components=2, random_state=20260616).fit_transform(x)
    clean["pc1"] = pcs[:, 0]
    clean["pc2"] = pcs[:, 1]
    write_csv(out_dir / "replicate_feature_projection.csv", clean)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    normal = clean[~clean["ref_id"].isin(TARGET_REFS)]
    ax.scatter(normal["pc1"], normal["pc2"], s=16, color="#9aa0a6", alpha=0.45, label="other d=0.85 refs")
    colors = {27: "#b23a48", 33: "#2f6f9f", 49: "#2a7f62"}
    for ref_id in TARGET_REFS:
        sub = clean[clean["ref_id"] == ref_id]
        ax.scatter(sub["pc1"], sub["pc2"], s=42, color=colors[ref_id], label=f"ref_{ref_id:03d}")
        for row in sub.itertuples():
            ax.text(row.pc1, row.pc2, str(int(row.replicate_id)), fontsize=6, color=colors[ref_id])
    ax.set_xlabel("summary-feature PC1")
    ax.set_ylabel("summary-feature PC2")
    ax.set_title("Replicate summary-feature projection at d=0.85")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.35, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_replicate_feature_projection.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.0))
    fields = [
        ("logZ_per_P", "replicate logZ / P"),
        ("weighted_ce", "weighted CE"),
        ("replicate_split_logZ_per_P_diff", "replicate internal split / P"),
    ]
    for row_idx, ref_id in enumerate(TARGET_REFS):
        sub = target[target["ref_id"] == ref_id].copy()
        for col_idx, (field, label) in enumerate(fields):
            ax = axes[row_idx, col_idx]
            values = sub[field].to_numpy(dtype=np.float64)
            if field == "logZ_per_P":
                values = values - logmeanexp(sub["logZ_inf_full"].to_numpy(dtype=np.float64)) / float(P)
            ax.hist(values, bins=min(8, max(4, len(values) // 2)), color="#5f6f91", alpha=0.8)
            ax.set_title(f"ref_{ref_id:03d}: {label}")
            ax.grid(True, linewidth=0.35, alpha=0.25)
            if field == "replicate_split_logZ_per_P_diff":
                ax.axvline(SPLIT_GATE, color="#b23a48", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_target_refs_logZ_CE_split_histograms.png", dpi=190)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, summary_df: pd.DataFrame, inventory_df: pd.DataFrame) -> None:
    target = summary_df[summary_df["ref_id"].isin(TARGET_REFS)].sort_values("ref_id")
    rows: list[list[Any]] = []
    for row in target.to_dict("records"):
        rows.append(
            [
                f"ref_{int(row['ref_id']):03d}",
                f"{float(row['overall_split_logZ_per_P_diff']):.6f}",
                f"{float(row.get('replicate_quarter_logZ_range_per_P', float('nan'))):.6f}",
                f"{float(row.get('half_split_quarter_logZ_range_per_P', float('nan'))):.6f}",
                f"{float(row.get('weighted_ce_range', float('nan'))):.6f}",
                f"{float(row.get('feature_kmeans2_silhouette', float('nan'))):.3f}",
                str(row.get("diagnostic_status")),
            ]
        )
    statuses = target["diagnostic_status"].astype(str).tolist()
    if all(s == "multi_sector_suspect" for s in statuses):
        conclusion = "summary-level evidence favors multi-sector / heterogeneous-sector behavior over pure Monte Carlo noise."
    elif any(s == "multi_sector_suspect" for s in statuses):
        conclusion = "summary-level evidence is mixed, with at least one failed reference showing multi-sector-like behavior."
    else:
        conclusion = "summary-level evidence does not cleanly prove multi-sector structure; the retained summaries are more consistent with split noise, pairing artifacts, or an unresolved ambiguous case."
    raw_sample_note = (
        "No retained per-particle direction/sample arrays were found for these units. "
        "Therefore the projection and histograms below are replicate-summary projections/histograms, not raw shell-sample projections. "
        "A raw geometric sector test would require rerunning the same units with particle/projection retention enabled."
    )
    report = f"""# d=0.85 Failure Mode Diagnostic

Rule: `{RULE}`

Target references: `ref027`, `ref033`, `ref049`

Radius: `{RADIUS}`

Split gate: `{SPLIT_GATE}`

## Conclusion

{conclusion}

Important limitation: {raw_sample_note}

## Target Summary

{markdown_table(["ref", "overall split/P", "rep 4-split range/P", "half 4-split range/P", "CE range", "feature silhouette", "status"], rows)}

## How This Was Tested From Existing Artifacts

- 4-split test 1: split the 16 retained replicate summaries into four consecutive replicate blocks and compare block `logmeanexp(logZ)` per parameter.
- 4-split test 2: split the 32 retained per-replicate half-logZ summaries into four consecutive half blocks and compare block `logmeanexp(logZ)` per parameter.
- CE test: inspect replicate-level weighted CE range and correlation with replicate logZ.
- Projection test: PCA projection of retained replicate-level summary features: `logZ/P`, weighted CE, weighted error, weighted H, ESS fraction, and internal split diff.
- Histogram test: replicate-summary histograms for centered logZ/P, weighted CE, and replicate internal split/P.

## Outputs

- `ref_diagnostics_summary.csv`
- `target_replicate_diagnostics.csv`
- `target_4split_blocks.csv`
- `artifact_inventory.csv`
- `replicate_feature_projection.csv`
- `figures/fig01_target_refs_4split_logZ_blocks.png`
- `figures/fig02_all_refs_4split_range_rank.png`
- `figures/fig03_target_refs_CE_vs_logZ.png`
- `figures/fig04_replicate_feature_projection.png`
- `figures/fig05_target_refs_logZ_CE_split_histograms.png`

## Artifact Inventory

Loaded d=0.85 unit summaries: `{len(inventory_df)}`

Retained raw particle/sample arrays found: `0`
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    out_dir = ensure_dir(OUT_DIR)
    payloads = load_d085_payloads()
    inventory_rows = []
    summaries = []
    rep_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for ref_id, payload in sorted(payloads.items()):
        inventory_rows.append(
            {
                "ref_id": ref_id,
                "artifact_source": payload.get("_artifact_source"),
                "artifact_path": payload.get("_artifact_path"),
                "has_replicate_summaries": bool(isinstance(payload.get("replicate_summaries"), list) and payload.get("replicate_summaries")),
                "replicates": int(payload.get("replicates", 0) or 0),
                "split_logZ_per_P_diff": float(payload.get("split_logZ_per_P_diff", float("nan"))),
            }
        )
        summary, reps, blocks = summarize_payload(ref_id, payload)
        summaries.append(summary)
        rep_rows.extend(reps)
        block_rows.extend(blocks)
    inventory_df = pd.DataFrame(inventory_rows).sort_values("ref_id")
    summary_df = pd.DataFrame(summaries).sort_values("ref_id")
    rep_df = pd.DataFrame(rep_rows).sort_values(["ref_id", "replicate_id"])
    block_df = pd.DataFrame(block_rows).sort_values(["ref_id", "block_kind", "block_id"])
    target_rep_df = rep_df[rep_df["ref_id"].isin(TARGET_REFS)].copy()
    target_block_df = block_df[block_df["ref_id"].isin(TARGET_REFS)].copy()
    write_csv(out_dir / "artifact_inventory.csv", inventory_df)
    write_csv(out_dir / "ref_diagnostics_summary.csv", summary_df)
    write_csv(out_dir / "all_replicate_diagnostics.csv", rep_df)
    write_csv(out_dir / "target_replicate_diagnostics.csv", target_rep_df)
    write_csv(out_dir / "target_4split_blocks.csv", target_block_df)
    plot_four_split(summary_df, block_df, out_dir)
    plot_ce_projection_hist(rep_df, summary_df, out_dir)
    write_report(out_dir, summary_df, inventory_df)
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "status": "diagnostic_complete",
            "rule": RULE,
            "radius": RADIUS,
            "target_refs": TARGET_REFS,
            "loaded_unit_summaries": int(len(inventory_df)),
            "target_refs_with_replicate_summaries": int(summary_df[summary_df["ref_id"].isin(TARGET_REFS)]["has_replicate_summaries"].sum()),
            "raw_particle_arrays_found": 0,
            "output_dir": rel(out_dir),
        },
    )
    print(json.dumps({"output_dir": rel(out_dir), "target_summary": summary_df[summary_df["ref_id"].isin(TARGET_REFS)].to_dict("records")}, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
