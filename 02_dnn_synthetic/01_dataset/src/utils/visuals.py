from __future__ import annotations

import math
from pathlib import Path
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .io_utils import ensure_dir


BETA_DIR_RE = re.compile(r"^beta_(?P<beta>\d+p\d+)(?:_p_(?P<p>\d+p\d+))?$")


def _parse_cell_id_series_values(cell_id: str) -> tuple[float | None, float | None]:
    match = BETA_DIR_RE.match(str(cell_id))
    if not match:
        return None, None
    p_value = match.group("p") if match.groupdict().get("p") is not None else "0p00"
    return float(match.group("beta").replace("p", ".")), float(p_value.replace("p", "."))


def _cell_in_series(cell_id: str, *, series_name: str) -> bool:
    beta_val, p_val = _parse_cell_id_series_values(cell_id)
    if beta_val is None or p_val is None:
        return False
    if str(series_name) == "beta":
        return abs(p_val) <= 1e-12
    if str(series_name) == "p":
        return abs(beta_val - 0.60) <= 1e-12
    return False


def plot_dataset_2d(X_raw: np.ndarray, y: np.ndarray, out_path: Path, title: str) -> None:
    X_raw = np.asarray(X_raw, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    ensure_dir(out_path.parent)
    plt.figure(figsize=(4.2, 4.2))
    pos = y > 0
    plt.scatter(X_raw[pos, 0], X_raw[pos, 1], s=12, c="black", alpha=0.85)
    plt.scatter(X_raw[~pos, 0], X_raw[~pos, 1], s=12, facecolors="white", edgecolors="black", alpha=0.85)
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_dataset_2d_region(
    X_raw: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    title: str,
    *,
    grid_size: int = 180,
    knn_k: int = 15,
) -> None:
    X_arr = np.asarray(X_raw, dtype=np.float64)
    y_arr = np.asarray(y).reshape(-1)
    ensure_dir(out_path.parent)
    if X_arr.size == 0 or y_arr.size == 0:
        return
    labels = np.where(y_arr > 0, 1.0, -1.0)
    x_min, x_max = -1.05, 1.05
    y_min, y_max = -1.05, 1.05
    x_edges = np.linspace(x_min, x_max, int(grid_size) + 1, dtype=np.float64)
    y_edges = np.linspace(y_min, y_max, int(grid_size) + 1, dtype=np.float64)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    grid_points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    # First use per-cell mean label when samples fall inside the grid cell.
    count_grid = np.zeros((int(grid_size), int(grid_size)), dtype=np.int32)
    sum_grid = np.zeros((int(grid_size), int(grid_size)), dtype=np.float64)
    ix = np.clip(np.searchsorted(x_edges, X_arr[:, 0], side="right") - 1, 0, int(grid_size) - 1)
    iy = np.clip(np.searchsorted(y_edges, X_arr[:, 1], side="right") - 1, 0, int(grid_size) - 1)
    for x_idx, y_idx, label in zip(ix, iy, labels):
        count_grid[int(y_idx), int(x_idx)] += 1
        sum_grid[int(y_idx), int(x_idx)] += float(label)
    score_grid = np.zeros_like(sum_grid)
    occupied = count_grid > 0
    score_grid[occupied] = sum_grid[occupied] / np.maximum(count_grid[occupied], 1)

    # Empty cells get a simple local weighted vote from nearby samples.
    empty_mask = ~occupied.reshape(-1)
    if np.any(empty_mask):
        k_eff = min(int(max(1, knn_k)), X_arr.shape[0])
        diff = grid_points[empty_mask, None, :] - X_arr[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nn_idx = np.argpartition(dist2, kth=max(0, k_eff - 1), axis=1)[:, :k_eff]
        nn_dist2 = np.take_along_axis(dist2, nn_idx, axis=1)
        nn_labels = labels[nn_idx]
        weights = 1.0 / np.maximum(nn_dist2, 1.0e-6)
        fallback_score = np.sum(weights * nn_labels, axis=1) / np.maximum(np.sum(weights, axis=1), 1.0e-12)
        score_grid.reshape(-1)[empty_mask] = fallback_score

    class_grid = (score_grid >= 0.0).astype(np.int32)
    cmap = ListedColormap(["#efe7cf", "#7aa6c2"])
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pcolormesh(x_edges, y_edges, class_grid, shading="flat", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.contour(xx, yy, score_grid, levels=[0.0], colors="black", linewidths=1.0, alpha=0.75)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_dataset_view(X_raw: np.ndarray, X_train: np.ndarray, y: np.ndarray, out_dir: Path, title: str) -> dict[str, object]:
    X_raw = np.asarray(X_raw, dtype=np.float64)
    input_dim = int(X_raw.shape[1])
    if input_dim != 2:
        raise ValueError("Only D=2 dataset views are retained in this cleanup.")
    ensure_dir(out_dir)
    view_path = out_dir / "dataset_view.png"
    region_path = out_dir / "region_fill_d2.png"
    scatter_path = out_dir / "scatter_d2.png"
    plot_dataset_2d_region(X_raw, y, region_path, title=title)
    plot_dataset_2d(X_raw, y, scatter_path, title=title)
    shutil.copyfile(region_path, view_path)
    return {"view_type": "region_fill_d2", "grid_size": 180, "fallback": "inverse_distance_knn"}

def _panel_image_path(dataset_dir: Path, *, dimension: int, panel_mode: str = "filled") -> Path:
    if int(dimension) == 2:
        if str(panel_mode).lower() == "scatter" and (dataset_dir / "scatter_d2.png").exists():
            return dataset_dir / "scatter_d2.png"
        if str(panel_mode).lower() == "filled" and (dataset_dir / "region_fill_d2.png").exists():
            return dataset_dir / "region_fill_d2.png"
        if (dataset_dir / "region_fill_d2.png").exists():
            return dataset_dir / "region_fill_d2.png"
        if (dataset_dir / "scatter_d2.png").exists():
            return dataset_dir / "scatter_d2.png"
    return dataset_dir / "dataset_view.png"


def plot_dataset_series_grid(
    index_rows: list[dict[str, object]],
    part_root: Path,
    out_path: Path,
    *,
    dimension: int,
    series_name: str,
    title: str,
    panel_mode: str = "filled",
) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in index_rows:
        if str(row.get("series", "")).strip().lower() != str(series_name).strip().lower():
            continue
        beta_key = str(row.get("cell_id", row.get("beta_dir", "")))
        if not beta_key:
            continue
        grouped.setdefault(beta_key, []).append(row)
    def row_sort_key(row: dict[str, object]) -> tuple[int, int]:
        dataset_id = row.get("dataset_id")
        if dataset_id is None:
            dataset_label = str(row.get("dataset_label", "dataset_000"))
            dataset_id = dataset_label.removeprefix("dataset_")
        seed = row.get("seed", 0)
        return int(dataset_id), int(seed)

    representative_rows = [sorted(rows, key=row_sort_key)[0] for _, rows in sorted(grouped.items(), key=lambda item: item[0])]
    if not representative_rows:
        return
    ncols = min(3, max(1, len(representative_rows)))
    nrows = int(math.ceil(len(representative_rows) / float(ncols)))
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, row in enumerate(representative_rows):
        ax = axes[idx // ncols, idx % ncols]
        if row.get("dataset_raw_path") is not None:
            dataset_dir = (part_root / Path(str(row["dataset_raw_path"]))).resolve().parent
        else:
            dataset_dir = (part_root / Path(str(row["source_dataset_path"]))).resolve()
        image_path = _panel_image_path(dataset_dir, dimension=int(dimension), panel_mode=str(panel_mode))
        if not image_path.exists():
            ax.text(0.5, 0.5, "missing panel", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.axis("on")
        ax.imshow(mpimg.imread(image_path))
        ax.set_xticks([])
        ax.set_yticks([])
        beta_key = str(row.get("cell_id", row.get("beta_dir", "")))
        rewire_p = float(row.get("rewire_p", 0.0))
        ax.set_title(f"{beta_key}\nbeta={float(row['beta_ising']):.2f}, p={rewire_p:.2f}", fontsize=9)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


__all__ = ["plot_dataset_2d", "plot_dataset_series_grid", "plot_dataset_view"]


