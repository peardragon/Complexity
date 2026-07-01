from __future__ import annotations

import numpy as np

from .dataset_builder import load_config
from .io_utils import save_csv
from .layout import (
    RAW_ROOT,
    SAMPLE_FIGURE_INPUT_ROOT,
    dataset_path,
    eta_dirs,
    eta_from_dir_name,
    source_dataset_path,
)

SAMPLE_FIELDS = [
    "panel_order",
    "noise_eta",
    "eta",
    "sample_order",
    "sample_index",
    "train_index",
    "digit",
    "label",
    "flipped",
    "source_array",
    "source_dataset_path",
]


def build_sample_summary() -> list[dict[str, object]]:
    config = load_config()
    sample_config = config.get("sample_figure", {})
    source_array = str(sample_config.get("source_array", "X_train_raw10"))
    samples_per_eta = int(sample_config.get("samples_per_eta", 8))

    rows: list[dict[str, object]] = []
    for panel_order, noise_dir in enumerate(eta_dirs(), start=1):
        with np.load(dataset_path(noise_dir)) as data:
            eta = eta_from_dir_name(noise_dir.name)
            n_samples = min(samples_per_eta, int(data[source_array].shape[0]))
            for sample_order, sample_index in enumerate(range(n_samples), start=1):
                rows.append(
                    {
                        "panel_order": panel_order,
                        "noise_eta": noise_dir.name,
                        "eta": f"{eta:g}",
                        "sample_order": sample_order,
                        "sample_index": sample_index,
                        "train_index": int(data["train_indices"][sample_index]),
                        "digit": int(data["digit_train"][sample_index]),
                        "label": int(data["y_train"][sample_index]),
                        "flipped": int(bool(data["eta_flip_mask_train"][sample_index])),
                        "source_array": source_array,
                        "source_dataset_path": source_dataset_path(noise_dir),
                    }
                )

    if not rows:
        raise FileNotFoundError(RAW_ROOT)
    save_csv(SAMPLE_FIGURE_INPUT_ROOT / "selected_sample_indices.csv", rows, SAMPLE_FIELDS)
    return rows


def build_summarized_outputs() -> None:
    build_sample_summary()
