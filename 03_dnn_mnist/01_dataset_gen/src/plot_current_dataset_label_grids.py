#!/usr/bin/env python3
"""Plot the current MNIST10 dataset labels as 2x10 image grids."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
DATASET_ROOT = Path(
    "/home/bjyong/Complexity/windows_project/02_dnn/08_mnist/runs/final/"
    "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500/"
    "01_dataset_prepare/raw_datasets/split_000"
)
OUT = LOCAL_ROOT / "01_dataset_gen" / "figures" / "current_dataset_label_grids"

RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]

RULE_LABELS = {
    "low_tv_spectral_teacher": "low_tv",
    "real_even_odd": "even_odd",
    "teacher_nn": "teacher_nn",
    "random_label": "random",
}


def pca2_scaled(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    emb = x @ vt[:2].T
    scale = np.max(np.abs(emb), axis=0, keepdims=True)
    scale = np.where(scale < 1.0e-12, 1.0, scale)
    return emb / scale


def representative_indices(emb: np.ndarray, y: np.ndarray, label: int, n: int = 10) -> np.ndarray:
    candidates = np.flatnonzero(y == int(label))
    if candidates.size < n:
        raise ValueError(f"Need at least {n} examples for label {label}, got {candidates.size}")
    centroid = emb[candidates].mean(axis=0)
    order = np.argsort(np.linalg.norm(emb[candidates] - centroid[None, :], axis=1))
    return candidates[order[:n]]


def metadata_note(rule: str, metadata: dict[str, object]) -> str:
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
        f"kNN spectral teacher, k={metadata.get('graph_k')}, spectral_k={metadata.get('spectral_k')}, "
        f"seed={metadata.get('selected_seed')}, threshold={float(metadata.get('train_threshold')):.5f}"
    )


def load_rule(rule: str) -> dict[str, np.ndarray | dict[str, object]]:
    ds_path = DATASET_ROOT / rule / "dataset.npz"
    meta_path = DATASET_ROOT / rule / "dataset_metadata.json"
    payload = np.load(ds_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "X_train": payload["X_train"],
        "X_train_raw10": payload["X_train_raw10"],
        "y_train": payload["y_train"].astype(np.int8),
        "digit_train": payload["digit_train"],
        "train_indices": payload["train_indices"],
        "metadata": metadata,
    }


def draw_rule_grid(rule: str, data: dict[str, np.ndarray | dict[str, object]], path: Path) -> None:
    x_raw = np.asarray(data["X_train_raw10"], dtype=np.float32).reshape(-1, 10, 10)
    x_scaled = np.asarray(data["X_train"], dtype=np.float64)
    y = np.asarray(data["y_train"], dtype=np.int8)
    digits = np.asarray(data["digit_train"])
    indices = np.asarray(data["train_indices"])
    metadata = data["metadata"]
    assert isinstance(metadata, dict)

    emb = pca2_scaled(x_scaled)
    pos_idx = representative_indices(emb, y, 1)
    neg_idx = representative_indices(emb, y, -1)

    fig, axes = plt.subplots(2, 10, figsize=(13.0, 3.2))
    for row, (label, chosen) in enumerate(((1, pos_idx), (-1, neg_idx))):
        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(x_raw[idx], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"d={int(digits[idx])}\n#{int(indices[idx])}", fontsize=7, pad=2)
            if col == 0:
                ax.set_ylabel(f"{label:+d}", rotation=0, labelpad=18, va="center", fontsize=11, weight="bold")

    fig.suptitle(f"{RULE_LABELS[rule]} label representatives: +1 top, -1 bottom", fontsize=12)
    fig.text(0.015, 0.01, metadata_note(rule, metadata), ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.90))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_combined(all_data: dict[str, dict[str, np.ndarray | dict[str, object]]], path: Path) -> None:
    fig, axes = plt.subplots(len(RULES) * 2, 10, figsize=(13.5, 10.0))
    for rule_i, rule in enumerate(RULES):
        data = all_data[rule]
        x_raw = np.asarray(data["X_train_raw10"], dtype=np.float32).reshape(-1, 10, 10)
        x_scaled = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y_train"], dtype=np.int8)
        digits = np.asarray(data["digit_train"])
        indices = np.asarray(data["train_indices"])
        metadata = data["metadata"]
        assert isinstance(metadata, dict)

        emb = pca2_scaled(x_scaled)
        chosen_by_label = ((1, representative_indices(emb, y, 1)), (-1, representative_indices(emb, y, -1)))
        for local_row, (label, chosen) in enumerate(chosen_by_label):
            row = rule_i * 2 + local_row
            for col, idx in enumerate(chosen):
                ax = axes[row, col]
                ax.imshow(x_raw[idx], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{int(digits[idx])}\n#{int(indices[idx])}", fontsize=6.5, pad=1.5)
                if col == 0:
                    ax.set_ylabel(
                        f"{RULE_LABELS[rule]}\n{label:+d}",
                        rotation=0,
                        labelpad=30,
                        va="center",
                        fontsize=8.5,
                        weight="bold",
                    )
            if local_row == 1:
                axes[row, 9].text(
                    1.08,
                    0.5,
                    metadata_note(rule, metadata),
                    transform=axes[row, 9].transAxes,
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    wrap=True,
                )

    fig.suptitle("Current MNIST10 train-set label representatives by dataset rule", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.02, 0.84, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_data = {rule: load_rule(rule) for rule in RULES}
    for rule, data in all_data.items():
        draw_rule_grid(rule, data, OUT / f"fig_label_grid_{rule}.png")
    draw_combined(all_data, OUT / "fig01_current_dataset_label_grids_all_rules.png")

    notes = [
        "# Current Dataset Label Grids",
        "",
        "Each rule uses the same 512 MNIST10 train images. Only the binary label rule changes.",
        "Every panel is a 2x10 representative grid selected near the label centroid in the train-set PCA view.",
        "Top row is `+1`; bottom row is `-1`. Cell titles show `digit` and original MNIST index.",
        "",
        "Rules:",
    ]
    for rule, data in all_data.items():
        metadata = data["metadata"]
        assert isinstance(metadata, dict)
        notes.append(f"- `{rule}`: {metadata_note(rule, metadata)}")
    notes.extend(
        [
            "",
            "Files:",
            "- `fig01_current_dataset_label_grids_all_rules.png`",
            *[f"- `fig_label_grid_{rule}.png`" for rule in RULES],
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
