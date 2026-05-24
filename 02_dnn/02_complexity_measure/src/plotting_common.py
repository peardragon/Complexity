from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from io_utils import ensure_dir


def plot_line(x: np.ndarray, y: np.ndarray, out_path: Path, title: str, ylabel: str, *, logy: bool = False, xlim: Optional[Tuple[float, float]] = None, xlabel: str = "distance d") -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(x[m], y[m])
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mean_with_band(x: np.ndarray, mean: np.ndarray, std: np.ndarray, out_path: Path, *, title: str, ylabel: str, xlabel: str = "distance d") -> None:
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(mean) & np.isfinite(std)
    if np.sum(m) < 2:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(6.6, 4.0))
    plt.plot(x[m], mean[m])
    plt.fill_between(x[m], mean[m] - std[m], mean[m] + std[m], alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(x: np.ndarray, y: np.ndarray, out_path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(5.2, 4.0))
    plt.scatter(x[m], y[m], s=26, alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_scatter_with_errorbars(x: np.ndarray, mean: np.ndarray, std: np.ndarray, out_path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(mean) & np.isfinite(std)
    if np.sum(m) < 1:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(5.2, 4.0))
    plt.errorbar(x[m], mean[m], yerr=std[m], fmt="o", capsize=4, markersize=5, alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_overlay_curves(curves: Sequence[Tuple[str, np.ndarray, np.ndarray]], out_path: Path, *, title: str, ylabel: str, xlabel: str = "distance d") -> None:
    if not curves:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(7.2, 4.4))
    for label, x, y in curves:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) < 2:
            continue
        plt.plot(x[m], y[m], label=label, alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_heatmap(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], out_path: Path, *, title: str, colorbar_label: str) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(1.2 * max(3, len(col_labels)) + 1.5, 0.5 * max(4, len(row_labels)) + 1.8))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(list(col_labels))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(list(row_labels))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_multiscale_curves(
    curve_rows: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    xlabel: str = "scale",
) -> None:
    if not curve_rows:
        return
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cmap = plt.get_cmap("viridis", max(2, len(curve_rows)))
    for idx, (label, x, mean, std) in enumerate(curve_rows):
        x_arr = np.asarray(x, dtype=np.float64)
        mean_arr = np.asarray(mean, dtype=np.float64)
        std_arr = np.asarray(std, dtype=np.float64)
        mask = np.isfinite(x_arr) & np.isfinite(mean_arr)
        if np.sum(mask) < 2:
            continue
        color = cmap(idx)
        ax.plot(x_arr[mask], mean_arr[mask], label=label, color=color, linewidth=1.8)
        if std_arr.size == mean_arr.size:
            std_mask = mask & np.isfinite(std_arr)
            if np.any(std_mask):
                ax.fill_between(
                    x_arr[std_mask],
                    mean_arr[std_mask] - std_arr[std_mask],
                    mean_arr[std_mask] + std_arr[std_mask],
                    color=color,
                    alpha=0.18,
                )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


__all__ = [
    "plot_heatmap",
    "plot_line",
    "plot_mean_with_band",
    "plot_multiscale_curves",
    "plot_overlay_curves",
    "plot_scatter",
    "plot_scatter_with_errorbars",
]


