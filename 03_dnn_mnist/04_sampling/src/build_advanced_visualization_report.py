#!/usr/bin/env python3
"""Build advanced refpool visualizations from the completed 90-ref run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
WINDOWS_PROJECT_ROOT = Path("/home/bjyong/Complexity/windows_project")
RUN_ROOT = LOCAL_ROOT / "04_sampling" / "raw_outputs" / "refpool1024_advanced_90ref"
OUT_ROOT = RUN_ROOT / "06_results_figures" / "advanced_visualization"
DATASET_FALLBACK_ROOT = (
    WINDOWS_PROJECT_ROOT
    / "02_dnn"
    / "08_mnist"
    / "runs"
    / "final"
    / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
    / "01_dataset_prepare"
    / "raw_datasets"
    / "split_000"
)

RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
RULE_LABELS = {
    "low_tv_spectral_teacher": "low tv",
    "real_even_odd": "even/odd",
    "teacher_nn": "teacher",
    "random_label": "random",
}
COLORS = {
    "low_tv_spectral_teacher": "#0072B2",
    "real_even_odd": "#009E73",
    "teacher_nn": "#D55E00",
    "random_label": "#CC79A7",
}
LABEL_COLORS = {-1: "#555555", 1: "#D55E00"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def resolve_dataset_path(raw_path: str | Path) -> Path:
    raw = Path(str(raw_path))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([LOCAL_ROOT / raw, WINDOWS_PROJECT_ROOT / raw])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve dataset path {raw_path!s}; tried {candidates}")


def dataset_metadata_note(rule: str, metadata: dict[str, object]) -> str:
    if rule == "real_even_odd":
        return "even digit -> +1, odd digit -> -1"
    if rule == "teacher_nn":
        return (
            f"teacher seed={metadata.get('teacher_seed')}, "
            f"threshold={float(metadata.get('train_median_logit_threshold')):.5f}"
        )
    if rule == "random_label":
        return f"balanced random labels, train seed={metadata.get('train_seed')}"
    return (
        f"kNN spectral teacher, k={metadata.get('graph_k')}, "
        f"spectral_k={metadata.get('spectral_k')}, seed={metadata.get('selected_seed')}"
    )


@dataclass
class RuleDataset:
    rule: str
    dataset_path: Path
    metadata_path: Path
    x_train: np.ndarray
    x_train_raw10: np.ndarray
    y_train: np.ndarray
    digit_train: np.ndarray
    train_indices: np.ndarray
    metadata: dict[str, object]


def load_rule_datasets() -> dict[str, RuleDataset]:
    ref_pool = read_csv_required(RUN_ROOT / "04_reference_pool" / "reference_pool_index.csv")
    datasets: dict[str, RuleDataset] = {}
    for rule in RULES:
        paths = ref_pool.loc[ref_pool["rule"].astype(str).eq(rule), "dataset_path"].dropna().astype(str)
        if len(paths):
            dataset_path = resolve_dataset_path(paths.iloc[0])
        else:
            dataset_path = DATASET_FALLBACK_ROOT / rule / "dataset.npz"
        metadata_path = dataset_path.with_name("dataset_metadata.json")
        payload = np.load(dataset_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        datasets[rule] = RuleDataset(
            rule=rule,
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            x_train=np.asarray(payload["X_train"], dtype=np.float32),
            x_train_raw10=np.asarray(payload["X_train_raw10"], dtype=np.float32),
            y_train=np.asarray(payload["y_train"], dtype=np.int8),
            digit_train=np.asarray(payload["digit_train"], dtype=np.int16),
            train_indices=np.asarray(payload["train_indices"], dtype=np.int64),
            metadata=metadata,
        )
    return datasets


def pca2_scaled(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    x64 = x64 - x64.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x64, full_matrices=False)
    emb = x64 @ vt[:2].T
    scale = np.max(np.abs(emb), axis=0, keepdims=True)
    scale = np.where(scale < 1.0e-12, 1.0, scale)
    return emb / scale


def representative_indices(emb: np.ndarray, labels: np.ndarray, label: int, n: int = 10) -> np.ndarray:
    candidates = np.flatnonzero(labels == int(label))
    if len(candidates) < n:
        raise ValueError(f"Need at least {n} examples for label {label}, got {len(candidates)}")
    centroid = emb[candidates].mean(axis=0)
    order = np.argsort(np.linalg.norm(emb[candidates] - centroid[None, :], axis=1))
    return candidates[order[:n]]


def build_dataset_tables(datasets: dict[str, RuleDataset], table_dir: Path) -> pd.DataFrame:
    base_rule = RULES[0]
    base = datasets[base_rule]
    rows: list[dict[str, object]] = []
    for idx in range(len(base.train_indices)):
        row: dict[str, object] = {
            "train_row": idx,
            "mnist_index": int(base.train_indices[idx]),
            "digit": int(base.digit_train[idx]),
        }
        for rule, data in datasets.items():
            row[f"label_{rule}"] = int(data.y_train[idx])
        rows.append(row)
    sample_df = pd.DataFrame(rows)

    summary_rows: list[dict[str, object]] = []
    for rule, data in datasets.items():
        for label in [-1, 1]:
            mask = data.y_train == label
            summary_rows.append(
                {
                    "rule": rule,
                    "label": int(label),
                    "count": int(mask.sum()),
                    "digit_histogram": json.dumps(
                        {str(int(d)): int((data.digit_train[mask] == d).sum()) for d in sorted(np.unique(data.digit_train))},
                        sort_keys=True,
                    ),
                    "dataset_path": str(data.dataset_path),
                    "metadata_path": str(data.metadata_path),
                    "metadata_note": dataset_metadata_note(rule, data.metadata),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    write_csv(table_dir / "dataset_train_samples_with_rule_labels.csv", sample_df)
    write_csv(table_dir / "dataset_label_summary.csv", summary_df)
    return sample_df


def build_dataset_embedding(datasets: dict[str, RuleDataset], table_dir: Path) -> pd.DataFrame:
    base = datasets[RULES[0]]
    x = np.asarray(base.x_train, dtype=np.float64)
    xz = StandardScaler().fit_transform(x)
    tsne_xy = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        random_state=20260620,
        max_iter=1200,
    ).fit_transform(xz)
    umap_xy = umap.UMAP(
        n_components=2,
        n_neighbors=24,
        min_dist=0.08,
        metric="euclidean",
        random_state=20260620,
    ).fit_transform(xz)
    rows: list[dict[str, object]] = []
    for idx in range(x.shape[0]):
        row: dict[str, object] = {
            "train_row": idx,
            "mnist_index": int(base.train_indices[idx]),
            "digit": int(base.digit_train[idx]),
            "tsne_1": float(tsne_xy[idx, 0]),
            "tsne_2": float(tsne_xy[idx, 1]),
            "umap_1": float(umap_xy[idx, 0]),
            "umap_2": float(umap_xy[idx, 1]),
        }
        for rule, data in datasets.items():
            row[f"label_{rule}"] = int(data.y_train[idx])
        rows.append(row)
    emb = pd.DataFrame(rows)
    write_csv(table_dir / "dataset_tsne_umap_embedding.csv", emb)
    return emb


def plot_dataset_representatives(datasets: dict[str, RuleDataset], fig_dir: Path) -> Path:
    path = fig_dir / "fig01_dataset_label_representatives.png"
    fig, axes = plt.subplots(len(RULES) * 2, 10, figsize=(13.5, 10.0))
    for rule_i, rule in enumerate(RULES):
        data = datasets[rule]
        x_raw = data.x_train_raw10.reshape(-1, 10, 10)
        emb = pca2_scaled(data.x_train)
        for local_row, label in enumerate([1, -1]):
            chosen = representative_indices(emb, data.y_train, label)
            row = rule_i * 2 + local_row
            for col, idx in enumerate(chosen):
                ax = axes[row, col]
                ax.imshow(x_raw[idx], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{int(data.digit_train[idx])}\n#{int(data.train_indices[idx])}", fontsize=6.4, pad=1.5)
                if col == 0:
                    ax.set_ylabel(
                        f"{RULE_LABELS[rule]}\n{label:+d}",
                        rotation=0,
                        labelpad=28,
                        va="center",
                        fontsize=8.2,
                        weight="bold",
                    )
            if local_row == 1:
                axes[row, 9].text(
                    1.08,
                    0.5,
                    dataset_metadata_note(rule, data.metadata),
                    transform=axes[row, 9].transAxes,
                    ha="left",
                    va="center",
                    fontsize=7.2,
                    wrap=True,
                )
    fig.suptitle("Dataset label representatives by rule", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.02, 0.84, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_dataset_embedding(emb: pd.DataFrame, fig_dir: Path) -> Path:
    path = fig_dir / "fig02_dataset_tsne_umap_embeddings.png"
    panels = ["digit", *[f"label_{rule}" for rule in RULES]]
    fig, axes = plt.subplots(2, len(panels), figsize=(18.0, 7.2))
    coords = [("tsne_1", "tsne_2", "t-SNE"), ("umap_1", "umap_2", "UMAP")]
    for row_i, (xcol, ycol, method) in enumerate(coords):
        for col_i, panel in enumerate(panels):
            ax = axes[row_i, col_i]
            if panel == "digit":
                sc = ax.scatter(emb[xcol], emb[ycol], c=emb["digit"], s=18, cmap="tab10", alpha=0.78, linewidth=0)
                if row_i == 0 and col_i == 0:
                    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                    cbar.set_label("digit")
                title = f"{method}: digit"
            else:
                rule = panel.replace("label_", "")
                labels = emb[panel].astype(int)
                colors = [LABEL_COLORS[int(label)] for label in labels]
                ax.scatter(emb[xcol], emb[ycol], c=colors, s=18, alpha=0.78, linewidth=0)
                title = f"{method}: {RULE_LABELS.get(rule, rule)} label"
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, linewidth=0.25, alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def sorted_rules(df: pd.DataFrame) -> list[str]:
    available = set(df["rule"].astype(str).unique())
    ordered = [rule for rule in RULES if rule in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def load_phi_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    phi = read_csv_required(RUN_ROOT / "06_results_figures" / "phi_by_rule_radius.csv")
    dphi = read_csv_required(RUN_ROOT / "06_results_figures" / "dphi_dd_by_rule_radius.csv")
    for df in (phi, dphi):
        df["rule"] = df["rule"].astype(str)
        df["radius"] = df["radius"].astype(float)
    return phi, dphi


def plot_phi_energy(phi: pd.DataFrame, fig_dir: Path) -> Path:
    path = fig_dir / "fig03_phi_energy_by_rule.png"
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.8), sharex=True)
    specs = [
        ("phi_energy_raw", "raw phi(d)_energy", "Raw phi(d)_energy"),
        ("delta_phi_energy", "delta phi(d)_energy from d0", "Delta phi(d)_energy"),
        ("delta_phi_full", "delta phi(d)_full from d0", "Delta phi(d)_full"),
    ]
    for ax, (col, ylabel, title) in zip(axes, specs):
        for rule in sorted_rules(phi):
            sub = phi[phi["rule"].eq(rule)].sort_values("radius")
            ax.plot(sub["radius"], sub[col], color=COLORS.get(rule), linewidth=2.0, label=RULE_LABELS.get(rule, rule))
            pass_sub = sub[sub["qc_diagnostic_pass"].astype(bool)]
            fail_sub = sub[~sub["qc_diagnostic_pass"].astype(bool)]
            ax.scatter(pass_sub["radius"], pass_sub[col], color=COLORS.get(rule), s=20, zorder=3)
            if len(fail_sub):
                ax.scatter(fail_sub["radius"], fail_sub[col], color=COLORS.get(rule), s=30, marker="x", zorder=4)
        ax.axhline(0.0, color="black", linewidth=0.65, alpha=0.32)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.45, alpha=0.26)
    axes[-1].set_xlabel("radius d")
    axes[0].legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=195)
    plt.close(fig)
    return path


def plot_dphi_energy(dphi: pd.DataFrame, fig_dir: Path) -> Path:
    path = fig_dir / "fig04_dphi_energy_by_rule.png"
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.8), sharex=True)
    specs = [
        ("d_phi_energy_raw_dd", "d raw phi/dd", "First derivative of raw phi(d)_energy"),
        ("d_delta_phi_energy_dd", "d delta phi/dd", "First derivative of delta phi(d)_energy"),
        ("d2_phi_energy_raw_dd2", "d2 raw phi/dd2", "Second derivative of raw phi(d)_energy"),
    ]
    for ax, (col, ylabel, title) in zip(axes, specs):
        for rule in sorted_rules(dphi):
            sub = dphi[dphi["rule"].eq(rule)].sort_values("radius")
            ax.plot(sub["radius"], sub[col], color=COLORS.get(rule), linewidth=2.0, label=RULE_LABELS.get(rule, rule))
            ax.scatter(sub["radius"], sub[col], color=COLORS.get(rule), s=18, zorder=3)
        ax.axhline(0.0, color="black", linewidth=0.65, alpha=0.32)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.45, alpha=0.26)
    axes[-1].set_xlabel("radius d")
    axes[0].legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=195)
    plt.close(fig)
    return path


def pivot_reference_features(units: pd.DataFrame, value_cols: Iterable[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    keys = units[["rule", "ref_id"]].drop_duplicates().sort_values(["rule", "ref_id"]).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(keys[["rule", "ref_id"]])
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for col in value_cols:
        pivot = (
            units.pivot_table(index=["rule", "ref_id"], columns="radius", values=col, aggfunc="mean")
            .reindex(index)
            .sort_index(axis=1)
        )
        arr = pivot.to_numpy(dtype=float)
        col_mean = np.nanmean(arr, axis=0)
        inds = np.where(~np.isfinite(arr))
        if len(inds[0]):
            arr[inds] = np.take(col_mean, inds[1])
        blocks.append(arr)
        names.extend([f"{col}@{float(radius):.2f}" for radius in pivot.columns])
    x = np.hstack(blocks)
    return keys, x, names


def build_reference_embedding(table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    units = read_csv_required(RUN_ROOT / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi_derivatives.csv")
    units["rule"] = units["rule"].astype(str)
    units["ref_id"] = units["ref_id"].astype(int)
    units["radius"] = units["radius"].astype(float)
    keys, x, feature_names = pivot_reference_features(
        units,
        ["phi_energy_raw", "d_phi_energy_raw_dd", "d2_phi_energy_raw_dd2"],
    )
    z = StandardScaler().fit_transform(x)
    tsne_xy = TSNE(
        n_components=2,
        perplexity=32,
        learning_rate="auto",
        init="pca",
        random_state=20260620,
        max_iter=1200,
    ).fit_transform(z)
    umap_xy = umap.UMAP(
        n_components=2,
        n_neighbors=22,
        min_dist=0.06,
        metric="euclidean",
        random_state=20260620,
    ).fit_transform(z)
    emb = keys.copy()
    emb["tsne_1"] = tsne_xy[:, 0]
    emb["tsne_2"] = tsne_xy[:, 1]
    emb["umap_1"] = umap_xy[:, 0]
    emb["umap_2"] = umap_xy[:, 1]
    for rule in RULES:
        mask = emb["rule"].eq(rule)
        if mask.any():
            centroid = emb.loc[mask, ["umap_1", "umap_2"]].mean().to_numpy(dtype=float)
            coords = emb.loc[mask, ["umap_1", "umap_2"]].to_numpy(dtype=float)
            emb.loc[mask, "dist_to_rule_umap_centroid"] = np.linalg.norm(coords - centroid[None, :], axis=1)

    nn = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(z)
    distances, indices = nn.kneighbors(z)
    neighbor_rows: list[dict[str, object]] = []
    for i, row in emb.iterrows():
        for rank in range(1, distances.shape[1]):
            n_idx = int(indices[i, rank])
            neighbor_rows.append(
                {
                    "rule": row["rule"],
                    "ref_id": int(row["ref_id"]),
                    "neighbor_rank": rank,
                    "neighbor_rule": emb.iloc[n_idx]["rule"],
                    "neighbor_ref_id": int(emb.iloc[n_idx]["ref_id"]),
                    "feature_distance": float(distances[i, rank]),
                }
            )
    neighbors = pd.DataFrame(neighbor_rows)
    feature_table = pd.DataFrame({"feature": feature_names})
    write_csv(table_dir / "reference_phi_curve_tsne_umap_embedding.csv", emb)
    write_csv(table_dir / "reference_phi_curve_nearest_neighbors.csv", neighbors)
    write_csv(table_dir / "reference_embedding_features.csv", feature_table)
    return emb, neighbors, units


def plot_reference_embedding(emb: pd.DataFrame, fig_dir: Path) -> Path:
    path = fig_dir / "fig05_reference_phi_curve_tsne_umap.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for ax, (xcol, ycol, title) in zip(axes, [("tsne_1", "tsne_2", "Reference t-SNE"), ("umap_1", "umap_2", "Reference UMAP")]):
        for rule in sorted_rules(emb):
            sub = emb[emb["rule"].eq(rule)]
            ax.scatter(sub[xcol], sub[ycol], s=34, alpha=0.84, color=COLORS.get(rule), label=RULE_LABELS.get(rule, rule))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linewidth=0.35, alpha=0.2)
    axes[0].legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=195)
    plt.close(fig)
    return path


def plot_proximal_landscape(emb: pd.DataFrame, units: pd.DataFrame, fig_dir: Path, table_dir: Path) -> tuple[Path, Path]:
    heatmap_path = fig_dir / "fig06_proximal_phi_landscape_heatmap.png"
    curves_path = fig_dir / "fig07_proximal_reference_examples.png"
    examples: list[dict[str, object]] = []

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    for ax, rule in zip(axes.ravel(), RULES):
        ref_order = (
            emb[emb["rule"].eq(rule)]
            .sort_values(["umap_1", "umap_2", "ref_id"])[["ref_id", "umap_1", "umap_2", "dist_to_rule_umap_centroid"]]
            .reset_index(drop=True)
        )
        sub = units[units["rule"].eq(rule)].copy()
        pivot = (
            sub.pivot_table(index="ref_id", columns="radius", values="phi_energy_raw", aggfunc="mean")
            .reindex(ref_order["ref_id"])
            .sort_index(axis=1)
        )
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest", origin="lower")
        xticks = np.linspace(0, len(pivot.columns) - 1, 7, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{float(pivot.columns[i]):.2f}" for i in xticks], rotation=45, ha="right")
        ax.set_title(f"{RULE_LABELS.get(rule, rule)} refs sorted by UMAP proximity")
        ax.set_xlabel("radius d")
        ax.set_ylabel("reference order")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="raw phi energy")

        if len(ref_order):
            picked = pd.concat(
                [
                    ref_order.nsmallest(1, "dist_to_rule_umap_centroid").assign(example_kind="centroid"),
                    ref_order.nsmallest(1, "umap_1").assign(example_kind="left_edge"),
                    ref_order.nlargest(1, "umap_1").assign(example_kind="right_edge"),
                ],
                ignore_index=True,
            ).drop_duplicates("ref_id")
            for _, row in picked.iterrows():
                examples.append(
                    {
                        "rule": rule,
                        "ref_id": int(row["ref_id"]),
                        "example_kind": str(row["example_kind"]),
                        "umap_1": float(row["umap_1"]),
                        "umap_2": float(row["umap_2"]),
                        "dist_to_rule_umap_centroid": float(row["dist_to_rule_umap_centroid"]),
                    }
                )
    fig.suptitle("Proximal phi(d)_energy landscape: references ordered by curve-UMAP coordinates", y=0.995)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=195)
    plt.close(fig)

    examples_df = pd.DataFrame(examples).sort_values(["rule", "example_kind", "ref_id"]).reset_index(drop=True)
    write_csv(table_dir / "proximal_landscape_example_refs.csv", examples_df)

    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8.4), sharex=True)
    for ax, rule in zip(axes.ravel(), RULES):
        sub_examples = examples_df[examples_df["rule"].eq(rule)]
        rule_units = units[units["rule"].eq(rule)]
        for _, example in sub_examples.iterrows():
            curve = rule_units[rule_units["ref_id"].eq(int(example["ref_id"]))].sort_values("radius")
            ax.plot(
                curve["radius"],
                curve["phi_energy_raw"],
                linewidth=2.0,
                label=f"{example['example_kind']} ref {int(example['ref_id'])}",
            )
        mean_curve = rule_units.groupby("radius", as_index=False)["phi_energy_raw"].mean().sort_values("radius")
        ax.plot(mean_curve["radius"], mean_curve["phi_energy_raw"], color="black", linewidth=2.4, alpha=0.78, label="rule mean")
        ax.set_title(RULE_LABELS.get(rule, rule))
        ax.set_xlabel("radius d")
        ax.set_ylabel("raw phi(d)_energy")
        ax.grid(True, linewidth=0.4, alpha=0.25)
        ax.legend(fontsize=7.2)
    fig.suptitle("Proximal landscape example curves", y=0.995)
    fig.tight_layout()
    fig.savefig(curves_path, dpi=195)
    plt.close(fig)
    return heatmap_path, curves_path


def write_report(
    out_root: Path,
    table_paths: list[Path],
    figure_paths: list[Path],
    phi: pd.DataFrame,
    emb_dataset: pd.DataFrame,
    emb_refs: pd.DataFrame,
) -> None:
    lines = [
        "# Advanced Refpool Visualization Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- Source run: `refpool1024_advanced_90ref`.",
        "- Radius grid: `0.10, 0.15, ..., 2.50` (49 radii).",
        "- Rules: `low_tv_spectral_teacher`, `real_even_odd`, `teacher_nn`, `random_label`.",
        "- Sampling status: complete, 90 references per rule/radius.",
        "",
        "## Checks",
        "",
        f"- Rule/radius rows: `{len(phi)}`.",
        f"- Dataset embedding rows: `{len(emb_dataset)}` train samples.",
        f"- Reference embedding rows: `{len(emb_refs)}` references.",
        "",
        "## Tables",
        "",
    ]
    for path in table_paths:
        lines.append(f"- `tables/{path.name}`")
    lines.extend(["", "## Figures", ""])
    for path in figure_paths:
        lines.append(f"- `figures/{path.name}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Dataset t-SNE/UMAP uses the shared 512 MNIST train images and overlays digit plus each binary rule label.",
            "- Reference t-SNE/UMAP uses per-reference curves over raw phi, dphi/dd, and d2phi/dd2 across all 49 radii.",
            "- The proximal landscape heatmap orders references by curve-UMAP coordinates within each rule.",
            "",
        ]
    )
    (out_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    out_root = ensure_dir(OUT_ROOT)
    table_dir = ensure_dir(out_root / "tables")
    fig_dir = ensure_dir(out_root / "figures")

    datasets = load_rule_datasets()
    dataset_samples = build_dataset_tables(datasets, table_dir)
    dataset_embedding = build_dataset_embedding(datasets, table_dir)

    phi, dphi = load_phi_tables()
    write_csv(table_dir / "advanced_phi_by_rule_radius.csv", phi)
    write_csv(table_dir / "advanced_dphi_dd_by_rule_radius.csv", dphi)

    ref_embedding, neighbors, units = build_reference_embedding(table_dir)
    _ = neighbors

    figure_paths = [
        plot_dataset_representatives(datasets, fig_dir),
        plot_dataset_embedding(dataset_embedding, fig_dir),
        plot_phi_energy(phi, fig_dir),
        plot_dphi_energy(dphi, fig_dir),
        plot_reference_embedding(ref_embedding, fig_dir),
    ]
    landscape_paths = plot_proximal_landscape(ref_embedding, units, fig_dir, table_dir)
    figure_paths.extend(list(landscape_paths))

    table_paths = [
        table_dir / "dataset_train_samples_with_rule_labels.csv",
        table_dir / "dataset_label_summary.csv",
        table_dir / "dataset_tsne_umap_embedding.csv",
        table_dir / "advanced_phi_by_rule_radius.csv",
        table_dir / "advanced_dphi_dd_by_rule_radius.csv",
        table_dir / "reference_phi_curve_tsne_umap_embedding.csv",
        table_dir / "reference_phi_curve_nearest_neighbors.csv",
        table_dir / "reference_embedding_features.csv",
        table_dir / "proximal_landscape_example_refs.csv",
    ]
    metadata = {
        "source_run": str(RUN_ROOT),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_rows": int(len(dataset_samples)),
        "dataset_embedding_rows": int(len(dataset_embedding)),
        "reference_embedding_rows": int(len(ref_embedding)),
        "phi_rows": int(len(phi)),
        "dphi_rows": int(len(dphi)),
        "sklearn_version": str(sklearn.__version__),
        "umap_version": str(umap.__version__),
        "figures": [str(path.relative_to(out_root)) for path in figure_paths],
        "tables": [str(path.relative_to(out_root)) for path in table_paths],
    }
    write_json(out_root / "VISUALIZATION_STATUS.json", metadata)
    write_report(out_root, table_paths, figure_paths, phi, dataset_embedding, ref_embedding)

    print(f"wrote advanced visualization to {out_root}")
    print(f"tables: {table_dir}")
    print(f"figures: {fig_dir}")
    for path in figure_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
