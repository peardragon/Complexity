from __future__ import annotations

import numpy as np

from .dataset_builder import load_config, rule_specs
from .io_utils import save_csv
from .layout import RAW_ROOT, SAMPLE_FIGURE_INPUT_ROOT, source_dataset_path

SAMPLE_FIELDS = [
    "panel_order",
    "rule_id",
    "rule_name",
    "rule_label",
    "sample_order",
    "sample_index",
    "train_index",
    "digit",
    "sample_label",
    "source_array",
    "source_dataset_path",
]


def build_sample_summary() -> list[dict[str, object]]:
    config = load_config()
    sample_config = config.get("sample_figure", {})
    source_array = str(sample_config.get("source_array", "X_train_raw10"))
    samples_per_rule = int(sample_config.get("samples_per_rule", 8))

    rows: list[dict[str, object]] = []
    for panel_order, rule in enumerate(rule_specs(), start=1):
        rule_dir = RAW_ROOT / rule["rule_id"]
        with np.load(rule_dir / "dataset.npz") as data:
            n_samples = min(samples_per_rule, int(data[source_array].shape[0]))
            for sample_order, sample_index in enumerate(range(n_samples), start=1):
                rows.append(
                    {
                        "panel_order": panel_order,
                        "rule_id": rule["rule_id"],
                        "rule_name": rule["rule_name"],
                        "rule_label": rule["label"],
                        "sample_order": sample_order,
                        "sample_index": sample_index,
                        "train_index": int(data["train_indices"][sample_index]),
                        "digit": int(data["digit_train"][sample_index]),
                        "sample_label": int(data["y_train"][sample_index]),
                        "source_array": source_array,
                        "source_dataset_path": source_dataset_path(rule_dir),
                    }
                )

    if not rows:
        raise FileNotFoundError(RAW_ROOT)

    save_csv(SAMPLE_FIGURE_INPUT_ROOT / "selected_sample_indices.csv", rows, SAMPLE_FIELDS)
    return rows


def build_summarized_outputs() -> None:
    build_sample_summary()


__all__ = [
    "SAMPLE_FIELDS",
    "build_sample_summary",
    "build_summarized_outputs",
]
