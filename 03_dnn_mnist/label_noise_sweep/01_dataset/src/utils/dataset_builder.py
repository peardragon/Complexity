from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .io_utils import load_json
from .layout import CONFIG_PATH, DNN_ROOT, RAW_ROOT, dataset_path, eta_dirs, eta_label


if str(DNN_ROOT) not in sys.path:
    sys.path.insert(0, str(DNN_ROOT))

from _shared.mnist10_standalone import (  # noqa: E402
    DEFAULT_SPLIT_SEED,
    build_mnist10_base_payload,
    ensure_original_mnist,
    eta_noise_payload,
    write_json_atomic,
    write_npz_atomic,
)


REQUIRED_KEYS = {
    "X_train",
    "y_train",
    "X_test",
    "y_test",
    "X_train_raw10",
    "X_test_raw10",
    "X_train_raw",
    "X_test_raw",
    "digit_train",
    "digit_test",
    "train_indices",
    "test_indices",
    "standardization_mean",
    "standardization_std",
    "eta",
    "eta_flip_mask_train",
    "eta_flip_mask_test",
    "eta_seed",
}

GENERATION_METADATA_FILENAME = "generation_meta.json"


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


def _eta_seed(config: dict[str, Any], eta: float) -> int:
    seed_map = config.get("eta_seed_map", {})
    if not isinstance(seed_map, dict):
        raise TypeError("eta_seed_map must be a JSON object")
    candidates = [f"{float(eta):g}", str(float(eta)), eta_label(float(eta))]
    for key in candidates:
        if key in seed_map:
            return int(seed_map[key])
    raise KeyError(f"eta_seed_map is missing a seed for eta={float(eta):g}")


def _generate_noise_payloads(
    targets: list[Path],
    *,
    config: dict[str, Any],
    raw28: np.ndarray,
    digits: np.ndarray,
    source_metadata: dict[str, Any],
) -> int:
    if not targets:
        return 0

    base_payload, split_metadata = build_mnist10_base_payload(
        raw28,
        digits,
        n_train=int(config.get("n_train", 512)),
        n_test=int(config.get("n_test", 2048)),
        split_seed=int(config.get("source", {}).get("split_seed", DEFAULT_SPLIT_SEED)),
    )

    generated = 0
    eta_by_dir = {eta_label(float(eta)): float(eta) for eta in config["eta_values"]}
    for noise_dir in targets:
        eta = eta_by_dir[noise_dir.name]
        seed = _eta_seed(config, eta)
        payload, generation = eta_noise_payload(base_payload, eta=eta, seed=seed)
        write_npz_atomic(dataset_path(noise_dir), payload)
        write_json_atomic(
            noise_dir / GENERATION_METADATA_FILENAME,
            {
                "metadata_schema": "mnist_label_noise_generation_v1",
                "noise_dir": noise_dir.name,
                "source_mnist": source_metadata,
                "split": split_metadata,
                "noise_generation": generation,
            },
        )
        generated += 1
    return generated


def build_datasets(
    *,
    overwrite: bool = False,
    source_cache: str | None = None,
    download: bool = True,
) -> dict[str, int]:
    config = load_config()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    expected_dirs = [RAW_ROOT / eta_label(float(eta)) for eta in config["eta_values"]]
    existing_dirs = eta_dirs()
    expected_names = {path.name for path in expected_dirs}
    unexpected = [path for path in existing_dirs if path.name not in expected_names]
    if unexpected:
        raise ValueError(f"Unexpected eta directories: {[path.name for path in unexpected]}")

    raw28, digits, source_metadata = ensure_original_mnist(RAW_ROOT, source_cache=source_cache, download=download)
    targets = expected_dirs if overwrite else [path for path in expected_dirs if not dataset_path(path).exists()]
    generated = _generate_noise_payloads(
        targets,
        config=config,
        raw28=raw28,
        digits=digits,
        source_metadata=source_metadata,
    )

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
    return {"validated": len(expected_dirs), "missing": missing, "generated": generated}
