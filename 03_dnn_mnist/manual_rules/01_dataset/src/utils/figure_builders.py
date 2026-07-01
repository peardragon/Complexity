from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .io_utils import load_csv_rows
from .layout import FIGURE_ROOT, SAMPLE_FIGURE_INPUT_ROOT, SAMPLE_FIGURE_PATH, repo_path


def _resolve_source(path_value: str) -> Path:
    return repo_path(path_value)


def build_sample_figure() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_csv_rows(SAMPLE_FIGURE_INPUT_ROOT / "selected_sample_indices.csv")
    if not rows:
        raise ValueError("sample figure summary is empty")

    rows_by_rule: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_rule[row["rule_id"]].append(row)

    rule_ids = sorted(rows_by_rule)
    ncols = max(len(rows_by_rule[rule_id]) for rule_id in rule_ids)
    nrows = len(rule_ids)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.25 * ncols, 1.45 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax in axes.ravel():
        ax.axis("off")

    payload_cache: dict[Path, np.lib.npyio.NpzFile] = {}
    try:
        for row_idx, rule_id in enumerate(rule_ids):
            rule_rows = sorted(rows_by_rule[rule_id], key=lambda row: int(row["sample_order"]))
            for col_idx, row in enumerate(rule_rows):
                dataset = _resolve_source(row["source_dataset_path"])
                if dataset not in payload_cache:
                    payload_cache[dataset] = np.load(dataset)
                data = payload_cache[dataset]
                sample_index = int(row["sample_index"])
                image = data[row["source_array"]][sample_index].reshape(10, 10)

                ax = axes[row_idx, col_idx]
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"d={row['digit']} y={row['sample_label']}", fontsize=7)
                if col_idx == 0:
                    ax.text(
                        0.03,
                        0.96,
                        row["rule_label"],
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        color="white",
                        fontsize=7,
                        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
                    )
        fig.savefig(SAMPLE_FIGURE_PATH, dpi=220, bbox_inches="tight", pad_inches=0.03)
    finally:
        for data in payload_cache.values():
            data.close()
        plt.close(fig)


def build_figures() -> None:
    build_sample_figure()


__all__ = ["build_figures", "build_sample_figure"]
