from __future__ import annotations

import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from plotting_common import plot_heatmap, plot_multiscale_curves, plot_scatter, plot_scatter_with_errorbars


CELL_ID_RE = re.compile(r"^cell_beta_(?P<beta>\d+p\d+)_p_(?P<p>\d+p\d+)$")


def _parse_cell_id_series_values(cell_id: str) -> tuple[float | None, float | None]:
    match = CELL_ID_RE.match(str(cell_id))
    if not match:
        return None, None
    return float(match.group("beta").replace("p", ".")), float(match.group("p").replace("p", "."))


def _cell_in_series(cell_id: str, *, series_name: str) -> bool:
    beta_val, p_val = _parse_cell_id_series_values(cell_id)
    if beta_val is None or p_val is None:
        return False
    if str(series_name) == "beta":
        return abs(p_val) <= 1e-12
    if str(series_name) == "p":
        return abs(beta_val - 0.60) <= 1e-12
    return False


def plot_series_grid(
    index_rows: list[dict[str, object]],
    cell_rows: list[dict[str, object]],
    synthetic_part_root: Path,
    out_path: Path,
    *,
    series_name: str,
    title: str,
    panel_mode: str = "filled",
) -> None:
    representative_rows = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in index_rows:
        if str(row.get("series", "")).strip().lower() != str(series_name).strip().lower():
            continue
        grouped.setdefault(str(row["cell_id"]), []).append(row)
    for _, rows in sorted(grouped.items(), key=lambda item: item[0]):
        representative_rows.append(sorted(rows, key=lambda row: (int(row["dataset_id"]), int(row["seed"])))[0])
    if str(series_name) == "beta":
        representative_rows.sort(key=lambda row: float(row["beta_ising"]))
    else:
        representative_rows.sort(key=lambda row: float(row["rewire_p"]))
    if not representative_rows:
        return
    ncols = min(3, max(1, len(representative_rows)))
    nrows = int(math.ceil(len(representative_rows) / float(ncols)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    mean_c_by_cell = {str(row["cell_id"]): float(row["C_mean"]) for row in cell_rows}
    for idx, row in enumerate(representative_rows):
        ax = axes[idx // ncols, idx % ncols]
        dataset_dir = (synthetic_part_root / Path(str(row["dataset_raw_path"]))).resolve().parent
        if str(panel_mode).lower() == "scatter" and (dataset_dir / "scatter_d2.png").exists():
            view_path = dataset_dir / "scatter_d2.png"
        elif str(panel_mode).lower() == "filled" and (dataset_dir / "region_fill_d2.png").exists():
            view_path = dataset_dir / "region_fill_d2.png"
        else:
            view_path = dataset_dir / "dataset_view.png"
        if not view_path.exists():
            ax.text(0.5, 0.5, "missing dataset_view", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.axis("on")
        ax.imshow(mpimg.imread(view_path))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{row['cell_id']}\n"
            f"beta={float(row['beta_ising']):.2f}, p={float(row['rewire_p']):.2f}, "
            f"C={mean_c_by_cell.get(str(row['cell_id']), float('nan')):.3f}",
            fontsize=9,
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


__all__ = ["plot_heatmap", "plot_multiscale_curves", "plot_scatter", "plot_scatter_with_errorbars", "plot_series_grid"]
