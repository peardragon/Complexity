#!/usr/bin/env python3
"""Plot active MNIST10 rule label examples and t-SNE embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
WINDOWS_ROOT = Path("/home/bjyong/Complexity/windows_project")
SOURCE_DATASET_ROOT = (
    WINDOWS_ROOT
    / "02_dnn/08_mnist/runs/final/"
    "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500/"
    "01_dataset_prepare/raw_datasets/split_000"
)
SYNTHETIC_DATASET_ROOT = (
    LOCAL_ROOT
    / "01_dataset_gen/raw_outputs/very_low_tv_spectral_teacher_v1/"
    "01_dataset_prepare/raw_datasets/split_000"
)
RESULT_ROOT = LOCAL_ROOT / "04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_90ref/06_results_figures"
COMPLEXITY_TABLE = RESULT_ROOT / "tables/nmstv_values_for_raw_phi_plot.csv"

ACTIVE_RULES = [
    "very_low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
DEPRECATED_RULES = ["low_tv_spectral_teacher"]
RULE_LABELS = {
    "very_low_tv_spectral_teacher": "very low tv",
    "real_even_odd": "even odd",
    "teacher_nn": "teacher nn",
    "random_label": "random",
    "low_tv_spectral_teacher": "low tv",
}
LABEL_COLORS = {-1: "#355C9A", 1: "#D24B35"}


def dataset_dir(rule: str) -> Path:
    if rule == "very_low_tv_spectral_teacher":
        return SYNTHETIC_DATASET_ROOT / rule
    return SOURCE_DATASET_ROOT / rule


def load_rule(rule: str) -> dict[str, object]:
    root = dataset_dir(rule)
    payload = np.load(root / "dataset.npz")
    meta_path = root / "dataset_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "X_train": np.asarray(payload["X_train"], dtype=np.float64),
        "X_train_raw10": np.asarray(payload["X_train_raw10"], dtype=np.float32),
        "y_train": np.asarray(payload["y_train"], dtype=np.int8),
        "digit_train": np.asarray(payload["digit_train"], dtype=np.int16),
        "train_indices": np.asarray(payload["train_indices"], dtype=np.int64),
        "metadata": metadata,
    }


def pca2_scaled(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    emb = x @ vt[:2].T
    scale = np.max(np.abs(emb), axis=0, keepdims=True)
    return emb / np.where(scale < 1.0e-12, 1.0, scale)


def representative_indices(emb: np.ndarray, y: np.ndarray, label: int, n: int = 10) -> np.ndarray:
    candidates = np.flatnonzero(y == int(label))
    if candidates.size < n:
        raise ValueError(f"Need at least {n} examples for label {label}, got {candidates.size}")
    centroid = emb[candidates].mean(axis=0)
    order = np.argsort(np.linalg.norm(emb[candidates] - centroid[None, :], axis=1))
    return candidates[order[:n]]


def metadata_note(rule: str, metadata: dict[str, object], nmstv: float | None) -> str:
    prefix = f"NMSTV={nmstv:.3f}; " if nmstv is not None and np.isfinite(nmstv) else ""
    if rule == "very_low_tv_spectral_teacher":
        cand = metadata.get("candidate", {})
        if isinstance(cand, dict):
            return (
                f"{prefix}spectral teacher, k={cand.get('spectral_k')}, "
                f"draw={cand.get('draw_idx')}, threshold={float(cand.get('threshold', 0.0)):.5f}"
            )
        return f"{prefix}very-low-TV spectral teacher"
    if rule == "real_even_odd":
        return f"{prefix}even digit -> +1, odd digit -> -1"
    if rule == "teacher_nn":
        return (
            f"{prefix}teacher seed={metadata.get('teacher_seed')}, "
            f"threshold={float(metadata.get('train_median_logit_threshold', 0.0)):.5f}"
        )
    if rule == "random_label":
        return f"{prefix}balanced random labels, train seed={metadata.get('train_seed')}"
    return f"{prefix}{rule}"


def load_nmstv() -> dict[str, float]:
    if not COMPLEXITY_TABLE.exists():
        return {}
    comp = pd.read_csv(COMPLEXITY_TABLE)
    return {str(row.rule): float(row.nmstv_mean) for row in comp.itertuples(index=False)}


def draw_rule_grid(rule: str, data: dict[str, object], nmstv: float | None, path: Path) -> None:
    x_raw = np.asarray(data["X_train_raw10"], dtype=np.float32).reshape(-1, 10, 10)
    x_scaled = np.asarray(data["X_train"], dtype=np.float64)
    y = np.asarray(data["y_train"], dtype=np.int8)
    digits = np.asarray(data["digit_train"])
    indices = np.asarray(data["train_indices"])
    metadata = data["metadata"]
    assert isinstance(metadata, dict)

    emb = pca2_scaled(x_scaled)
    chosen = {1: representative_indices(emb, y, 1), -1: representative_indices(emb, y, -1)}
    fig, axes = plt.subplots(2, 10, figsize=(13.2, 3.45))
    for row, label in enumerate((1, -1)):
        for col, idx in enumerate(chosen[label]):
            ax = axes[row, col]
            ax.imshow(x_raw[idx], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"y={label:+d}\nd={int(digits[idx])}", fontsize=7, pad=2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.35)
                spine.set_edgecolor(LABEL_COLORS[label])
            if col == 0:
                ax.set_ylabel(f"{label:+d}", rotation=0, labelpad=18, va="center", fontsize=11, weight="bold")
    fig.suptitle(f"{RULE_LABELS[rule]} train examples by binary label", fontsize=12.5)
    fig.text(0.015, 0.015, metadata_note(rule, metadata, nmstv), ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.90))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def draw_combined_grids(all_data: dict[str, dict[str, object]], nmstv: dict[str, float], path: Path) -> None:
    fig, axes = plt.subplots(len(ACTIVE_RULES) * 2, 10, figsize=(13.8, 9.8))
    for rule_i, rule in enumerate(ACTIVE_RULES):
        data = all_data[rule]
        x_raw = np.asarray(data["X_train_raw10"], dtype=np.float32).reshape(-1, 10, 10)
        x_scaled = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y_train"], dtype=np.int8)
        digits = np.asarray(data["digit_train"])
        emb = pca2_scaled(x_scaled)
        chosen = {1: representative_indices(emb, y, 1), -1: representative_indices(emb, y, -1)}
        for local_row, label in enumerate((1, -1)):
            row = rule_i * 2 + local_row
            for col, idx in enumerate(chosen[label]):
                ax = axes[row, col]
                ax.imshow(x_raw[idx], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"y={label:+d}\nd={int(digits[idx])}", fontsize=6.6, pad=1.5)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)
                    spine.set_edgecolor(LABEL_COLORS[label])
                if col == 0:
                    ax.set_ylabel(
                        f"{RULE_LABELS[rule]}\n{label:+d}",
                        rotation=0,
                        labelpad=33,
                        va="center",
                        fontsize=8.5,
                        weight="bold",
                    )
            if local_row == 1:
                note = f"NMSTV {nmstv.get(rule, float('nan')):.3f}"
                axes[row, 9].text(1.08, 0.5, note, transform=axes[row, 9].transAxes, va="center", fontsize=8)
    fig.suptitle("Active MNIST10 label-rule examples; low_tv_spectral_teacher is deprecated", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.02, 0.90, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def compute_tsne(x: np.ndarray, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)["embedding"]
    x = np.asarray(x, dtype=np.float64)
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        init="pca",
        learning_rate="auto",
        max_iter=1200,
        random_state=20260618,
        method="barnes_hut",
        angle=0.5,
    )
    emb = tsne.fit_transform(x)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embedding=emb)
    return emb


def draw_tsne_panel(ax: plt.Axes, emb: np.ndarray, data: dict[str, object], rule: str, nmstv: float | None) -> None:
    y = np.asarray(data["y_train"], dtype=np.int8)
    digits = np.asarray(data["digit_train"])
    for label in (-1, 1):
        mask = y == label
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=15,
            c=LABEL_COLORS[label],
            alpha=0.68,
            linewidths=0.0,
            label=f"{label:+d}",
        )
    rng = np.random.default_rng(20260618)
    for idx in rng.choice(np.arange(len(y)), size=42, replace=False):
        ax.text(emb[idx, 0], emb[idx, 1], str(int(digits[idx])), fontsize=5.4, color="black", alpha=0.55)
    title = RULE_LABELS[rule]
    if nmstv is not None and np.isfinite(nmstv):
        title = f"{title}  NMSTV={nmstv:.3f}"
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linewidth=0.25, alpha=0.18)


def draw_tsne(all_data: dict[str, dict[str, object]], nmstv: dict[str, float], out_dir: Path) -> None:
    first = all_data[ACTIVE_RULES[0]]
    emb = compute_tsne(np.asarray(first["X_train"], dtype=np.float64), out_dir / "tables/mnist10_train_tsne_embedding.npz")
    emb_table = pd.DataFrame(
        {
            "train_row": np.arange(emb.shape[0]),
            "tsne_1": emb[:, 0],
            "tsne_2": emb[:, 1],
            "digit": np.asarray(first["digit_train"], dtype=np.int16),
        }
    )
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    emb_table.to_csv(out_dir / "tables/mnist10_train_tsne_embedding.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), sharex=True, sharey=True)
    for ax, rule in zip(axes.ravel(), ACTIVE_RULES):
        draw_tsne_panel(ax, emb, all_data[rule], rule, nmstv.get(rule))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="binary label", loc="center right", frameon=False)
    fig.suptitle("t-SNE view of the shared MNIST10 train images under active label rules", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 0.91, 0.96))
    fig.savefig(out_dir / "figures/fig03_active_rule_tsne_label_embedding.png", dpi=190)
    plt.close(fig)

    for rule in ACTIVE_RULES:
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        draw_tsne_panel(ax, emb, all_data[rule], rule, nmstv.get(rule))
        ax.legend(title="binary label", frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / f"fig_tsne_label_embedding_{rule}.png", dpi=190)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot active rule dataset examples and t-SNE embeddings.")
    parser.add_argument("--out-root", default=str(RESULT_ROOT))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_root)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    nmstv = load_nmstv()
    all_data = {rule: load_rule(rule) for rule in ACTIVE_RULES}
    for rule, data in all_data.items():
        draw_rule_grid(rule, data, nmstv.get(rule), out_dir / "figures" / f"fig_dataset_label_examples_{rule}.png")
    draw_combined_grids(all_data, nmstv, out_dir / "figures/fig02_active_rule_dataset_label_examples.png")
    draw_tsne(all_data, nmstv, out_dir)

    pd.DataFrame({"rule": ACTIVE_RULES, "status": "active"}).to_csv(out_dir / "tables/active_rules_for_result_figures.csv", index=False)
    pd.DataFrame({"rule": DEPRECATED_RULES, "status": "deprecated"}).to_csv(
        out_dir / "tables/deprecated_rules_for_result_figures.csv",
        index=False,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
