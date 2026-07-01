from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DNN_ROOT = Path(__file__).resolve().parents[3]


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
