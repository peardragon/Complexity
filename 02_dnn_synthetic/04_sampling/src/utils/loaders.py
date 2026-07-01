from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


DNN_ROOT = Path(__file__).resolve().parents[3]
BETA_SLUGS = [
    "0p05",
    "0p07",
    "0p09",
    "0p11",
    "0p13",
    "0p15",
    "0p17",
    "0p19",
    "0p21",
    "0p23",
    "0p25",
    "0p27",
    "0p29",
    "0p31",
    "0p33",
    "0p35",
    "0p37",
    "0p39",
]


@dataclass(frozen=True)
class ReferenceRecord:
    beta: float
    cell_id: str
    dataset_tag: str
    dataset_id: int
    ref_id: int
    dataset_path: str | Path
    theta_path: str | Path


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        candidate = path
    else:
        repo_root = DNN_ROOT.parent
        if path.parts and path.parts[0] == DNN_ROOT.name:
            candidate = repo_root / path
        else:
            candidate = DNN_ROOT / path
    if candidate.exists():
        return candidate
    return _resolve_legacy_path(candidate)


def _resolve_legacy_path(candidate: Path) -> Path:
    normalized = candidate.as_posix()
    nested_ref_match = re.search(
        r"03_reference_search/raw_outputs/(?:cell_)?beta_(\d+p\d+)/(dataset_(\d+))/(ref_\d+)/(theta(?:_init)?\.npy)$",
        normalized,
    )
    if nested_ref_match:
        beta_slug, dataset_tag, dataset_id_text, ref_tag, filename = nested_ref_match.groups()
        dataset_id = int(dataset_id_text)
        cell_index = BETA_SLUGS.index(beta_slug) if beta_slug in BETA_SLUGS else -1
        global_start = cell_index * 90 + 1
        local_dataset_id = dataset_id - global_start if dataset_id >= global_start else dataset_id
        if 0 <= local_dataset_id < 90:
            remapped = (
                DNN_ROOT
                / "03_reference_search"
                / "raw_outputs"
                / f"beta_{beta_slug}"
                / f"dataset_{local_dataset_id:03d}"
                / ref_tag
                / filename
            )
            if remapped.exists():
                return remapped

    ref_match = re.search(
        r"03_reference_search/raw_outputs/(dataset_(\d+))/(ref_\d+)/(theta(?:_init)?\.npy)$",
        normalized,
    )
    if ref_match:
        _dataset_tag, dataset_id_text, ref_tag, filename = ref_match.groups()
        cell_index = (int(dataset_id_text) - 1) // 90
        if 0 <= cell_index < len(BETA_SLUGS):
            local_dataset_id = (int(dataset_id_text) - 1) % 90
            remapped = (
                DNN_ROOT
                / "03_reference_search"
                / "raw_outputs"
                / f"beta_{BETA_SLUGS[cell_index]}"
                / f"dataset_{local_dataset_id:03d}"
                / ref_tag
                / filename
            )
            if remapped.exists():
                return remapped

    dataset_match = re.search(
        r"01_dataset/raw_outputs/18_beta_cell_90_dataset/raw_datasets/"
        r"cell_beta_(\d+p\d+)(?:_p_0p00)?/dataset_(\d+)_seed_\d+/dataset\.npz$",
        normalized,
    )
    if dataset_match:
        beta_slug, local_dataset_id = dataset_match.groups()
        remapped = (
            DNN_ROOT
            / "01_dataset"
            / "raw_outputs"
            / f"beta_{beta_slug}"
            / f"dataset_{int(local_dataset_id):03d}"
            / "dataset.npz"
        )
        if remapped.exists():
            return remapped

    return candidate


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(_resolve(path), allow_pickle=False) as data:
        keys = set(data.files)
        x_key = "X_train" if "X_train" in keys else "X"
        y_key = "y" if "y" in keys else "y_train"
        if x_key not in keys or y_key not in keys:
            raise KeyError(f"Dataset {path} lacks X/y arrays; keys={sorted(keys)}")
        return {
            "X_train": np.asarray(data[x_key], dtype=np.float64),
            "y": np.asarray(data[y_key]),
        }


def load_theta(path: str | Path) -> np.ndarray:
    path = _resolve(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64).reshape(-1)
    with np.load(path, allow_pickle=False) as data:
        for key in ("theta", "theta_ref", "arr_0"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float64).reshape(-1)
    raise KeyError(f"Theta payload {path} does not contain theta/theta_ref/arr_0")
