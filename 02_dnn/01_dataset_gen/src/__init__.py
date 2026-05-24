from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from io_utils import ensure_dir


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


def plot_dataset_view(X_raw: np.ndarray, X_train: np.ndarray, y: np.ndarray, out_dir: Path, title: str) -> dict[str, object]:
    X_raw = np.asarray(X_raw, dtype=np.float64)
    input_dim = int(X_raw.shape[1])
    if input_dim != 2:
        raise ValueError("Only D=2 dataset views are retained in this cleanup.")
    ensure_dir(out_dir)
    view_path = out_dir / "dataset_view.png"
    specific_path = out_dir / "scatter_d2.png"
    plot_dataset_2d(X_raw, y, specific_path, title=title)
    shutil.copyfile(specific_path, view_path)
    return {"view_type": "scatter_d2"}


__all__ = ["plot_dataset_2d", "plot_dataset_view"]


