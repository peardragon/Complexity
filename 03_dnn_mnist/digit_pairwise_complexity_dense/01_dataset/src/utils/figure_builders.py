from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .layout import RAW_ROOT, SAMPLE_FIGURE_INPUT_ROOT, SAMPLE_FIGURE_PATH, dataset_path


SAMPLE_INDEX_PATH = SAMPLE_FIGURE_INPUT_ROOT / "selected_sample_indices.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"run src/make_summarized_outputs.py first: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_figures() -> Path:
    rows = _read_rows(SAMPLE_INDEX_PATH)
    by_pair: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    pair_ids = sorted(by_pair, key=lambda value: tuple(int(part) for part in value.removeprefix("pair_").split("_")))
    if not pair_ids:
        raise ValueError(f"no sample rows for {SAMPLE_INDEX_PATH}")

    pairs_per_row = 5
    n_rows = int(np.ceil(len(pair_ids) / pairs_per_row))
    n_cols = pairs_per_row * 2
    SAMPLE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.25), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    for ax in axes_arr.ravel():
        ax.axis("off")

    for pair_index, pair in enumerate(pair_ids):
        row_idx = pair_index // pairs_per_row
        col_start = (pair_index % pairs_per_row) * 2
        payload_path = dataset_path(RAW_ROOT / pair)
        with np.load(payload_path) as data:
            x_raw = np.asarray(data["X_train_raw10"])
            for offset, sample in enumerate(sorted(by_pair[pair], key=lambda item: item["sample_role"])):
                ax = axes_arr[row_idx, col_start + offset]
                local_index = int(sample["local_train_index"])
                ax.imshow(x_raw[local_index].reshape(10, 10), cmap="gray_r", interpolation="nearest")
                ax.set_title(f"{sample['label']} d{sample['digit']}", fontsize=8)
                ax.axis("off")

    fig.savefig(SAMPLE_FIGURE_PATH, dpi=220)
    plt.close(fig)
    return SAMPLE_FIGURE_PATH
