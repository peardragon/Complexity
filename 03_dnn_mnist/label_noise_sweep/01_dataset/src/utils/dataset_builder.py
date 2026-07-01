from __future__ import annotations

from typing import Any

import numpy as np

from .io_utils import load_json
from .layout import CONFIG_PATH, RAW_ROOT, dataset_path, eta_dirs, eta_label


REQUIRED_KEYS = {
    "X_train",
    "y_train",
    "X_test",
    "y_test",
    "X_train_raw10",
    "digit_train",
    "eta",
    "eta_flip_mask_train",
}


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH, default=None)
    if not isinstance(config, dict):
        raise FileNotFoundError(CONFIG_PATH)
    return config


def _validate_payload(path: str) -> None:
    with np.load(path) as data:
        missing = sorted(REQUIRED_KEYS.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")


def build_datasets(*, overwrite: bool = False) -> dict[str, int]:
    del overwrite
    config = load_config()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    expected_dirs = [RAW_ROOT / eta_label(float(eta)) for eta in config["eta_values"]]
    existing_dirs = eta_dirs()
    expected_names = {path.name for path in expected_dirs}
    unexpected = [path for path in existing_dirs if path.name not in expected_names]
    if unexpected:
        raise ValueError(f"Unexpected eta directories: {[path.name for path in unexpected]}")

    missing = 0
    for noise_dir in expected_dirs:
        payload = dataset_path(noise_dir)
        if not payload.exists():
            missing += 1
            continue
        _validate_payload(str(payload))

    if missing:
        missing_names = [path.name for path in expected_dirs if not dataset_path(path).exists()]
        raise FileNotFoundError(f"Missing dataset payloads: {missing_names}")
    return {"validated": len(expected_dirs), "missing": missing}
