from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLING_ROOT = Path(__file__).resolve().parents[2]
LABEL_ROOT = SAMPLING_ROOT.parent
MNIST_ROOT = LABEL_ROOT.parent
INNER_ROOT = MNIST_ROOT.parent
PROJECT_ROOT = INNER_ROOT.parent


@dataclass(frozen=True)
class ReferenceRecord:
    eta: float
    ref_id: int
    theta_path: Path
    dataset_path: Path
    ce_mean_train: float | None = None


def resolve_existing_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for root in (PROJECT_ROOT, INNER_ROOT, MNIST_ROOT, LABEL_ROOT, SAMPLING_ROOT):
        candidate = root / path
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def project_relative(path: str | Path) -> str:
    resolved = resolve_existing_path(path).resolve()
    for root in (PROJECT_ROOT, INNER_ROOT):
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def load_dataset(path_value: str | Path) -> dict[str, np.ndarray]:
    path = resolve_existing_path(path_value)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        x_key = "X_train" if "X_train" in keys else "X"
        y_key = "y_train" if "y_train" in keys else "y"
        if x_key not in keys or y_key not in keys:
            raise KeyError(f"Dataset {path} lacks X/y arrays; keys={sorted(keys)}")
        out = {key: data[key] for key in data.files}
        out["X_train"] = np.asarray(data[x_key], dtype=np.float64)
        out["y_train"] = np.asarray(data[y_key])
        out["y"] = out["y_train"]
        return out


def load_theta(path_value: str | Path) -> np.ndarray:
    path = resolve_existing_path(path_value)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64).reshape(-1)
    with np.load(path, allow_pickle=False) as data:
        for key in ("theta", "theta_ref", "arr_0"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float64).reshape(-1)
    raise KeyError(f"Theta payload {path} does not contain theta/theta_ref/arr_0")
