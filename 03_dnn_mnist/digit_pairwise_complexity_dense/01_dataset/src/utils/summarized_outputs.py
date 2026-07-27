from __future__ import annotations

from typing import Any

import numpy as np

from .dataset_builder import dataset_metadata, pair_specs
from .io_utils import save_csv
from .layout import RAW_ROOT, SAMPLE_FIGURE_INPUT_ROOT, SUMMARY_ROOT, dataset_path


DATASET_SUMMARY_PATH = SUMMARY_ROOT / "digit_pairwise_dataset_summary.csv"
SAMPLE_INDEX_PATH = SAMPLE_FIGURE_INPUT_ROOT / "selected_sample_indices.csv"


def _summary_row(spec: dict[str, int | str]) -> dict[str, Any]:
    metadata = dataset_metadata(spec)
    return {
        "pair_id": metadata["pair_id"],
        "digit_a": metadata["digit_a"],
        "digit_b": metadata["digit_b"],
        "label": metadata["label"],
        "dataset_path": metadata["dataset_path"],
        "n_train": metadata["n_train"],
        "n_test": metadata["n_test"],
        "feature_dim": metadata["feature_dim"],
        "train_label_balance": metadata["train_label_balance"],
        "test_label_balance": metadata["test_label_balance"],
        "train_digit_balance": metadata["train_digit_balance"],
        "test_digit_balance": metadata["test_digit_balance"],
    }


def _sample_rows(specs: list[dict[str, int | str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        payload_path = dataset_path(RAW_ROOT / str(spec["pair_id"]))
        with np.load(payload_path) as data:
            digits = np.asarray(data["digit_train"]).reshape(-1)
            for role, digit in (("digit_a", int(spec["digit_a"])), ("digit_b", int(spec["digit_b"]))):
                indices = np.flatnonzero(digits == digit)
                if indices.size == 0:
                    continue
                rows.append(
                    {
                        "pair_id": str(spec["pair_id"]),
                        "digit_a": int(spec["digit_a"]),
                        "digit_b": int(spec["digit_b"]),
                        "label": str(spec["label"]),
                        "sample_role": role,
                        "digit": int(digit),
                        "local_train_index": int(indices[0]),
                    }
                )
    return rows


def build_summarized_outputs() -> dict[str, int]:
    specs = pair_specs()
    summary_rows = [_summary_row(spec) for spec in specs]
    sample_rows = _sample_rows(specs)
    save_csv(DATASET_SUMMARY_PATH, summary_rows)
    save_csv(SAMPLE_INDEX_PATH, sample_rows)
    return {
        "dataset_summary_rows": len(summary_rows),
        "sample_index_rows": len(sample_rows),
    }
