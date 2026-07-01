from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from .layout import (
    FIGURE_ROOT,
    REPO_ROOT,
    SAMPLE_FIGURE_PATH,
    SAMPLE_SUMMARY_ROOT,
    SPIN_FIGURE_PATH,
    SPIN_SUMMARY_ROOT,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run make_summarized_outputs.py first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_source(path_value: str) -> Path:
    return REPO_ROOT / path_value


def build_sample_figure() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(SAMPLE_SUMMARY_ROOT / "selected_sample_indices.csv")
    if not rows:
        raise ValueError("sample figure summary is empty")

    ncols = 6
    nrows = int(math.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.45 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, row in enumerate(rows):
        image_path = _resolve_source(row["source_image_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        ax = axes[idx // ncols, idx % ncols]
        ax.imshow(mpimg.imread(image_path))
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.05)
    fig.savefig(SAMPLE_FIGURE_PATH, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_spin_dynamics_figure() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(SPIN_SUMMARY_ROOT / "spin_alignment_by_beta.csv")
    if not rows:
        raise ValueError("spin dynamics summary is empty")

    beta_values = np.asarray([float(row["beta_ising"]) for row in rows], dtype=np.float64)
    mean_alignment = np.asarray([float(row["mean_edge_alignment"]) for row in rows], dtype=np.float64)
    sem_alignment = np.asarray([float(row["sem_edge_alignment"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(beta_values, mean_alignment, color="#252525", linewidth=2.2, marker="o", markersize=4.8)
    ax.fill_between(
        beta_values,
        mean_alignment - sem_alignment,
        mean_alignment + sem_alignment,
        color="#5b8db8",
        alpha=0.22,
        linewidth=0.0,
    )
    ax.set_xlabel("inverse temperature beta (lower T to the right)")
    ax.set_ylabel("mean edge spin alignment <s_i s_j>")
    ax.set_title("Spin-dynamics snapshots show temperature-driven ordering")
    ax.set_ylim(0.0, 0.96)
    ax.set_xlim(float(beta_values.min()) - 0.01, float(beta_values.max()) + 0.01)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax.text(
        0.03,
        0.93,
        "90 final snapshots per beta\n2000 Kawasaki sweeps per snapshot",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ticks = np.asarray([0.05, 0.10, 0.20, 0.30, 0.39], dtype=np.float64)
    top_ax.set_xticks(top_ticks)
    top_ax.set_xticklabels([f"{1.0 / tick:.1f}" for tick in top_ticks])
    top_ax.set_xlabel("temperature T = 1 / beta")

    fig.tight_layout()
    fig.savefig(SPIN_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_figures() -> None:
    build_sample_figure()
    build_spin_dynamics_figure()
