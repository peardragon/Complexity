from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(config_path)
    return config


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def stage_dir(stage: str, experiment_id: str) -> Path:
    roots = {
        "dataset": PROJECT_ROOT / "01_dataset_proxy" / "raw_outputs",
        "regions": PROJECT_ROOT / "02_region_protocol" / "raw_outputs",
        "baselines": PROJECT_ROOT / "03_baseline_samplers" / "raw_outputs",
        "vmf": PROJECT_ROOT / "04_vmf_l2_importance" / "raw_outputs",
        "qc": PROJECT_ROOT / "05_qc_and_figures" / "raw_outputs",
        "figures": PROJECT_ROOT / "05_qc_and_figures" / "figures",
    }
    if stage not in roots:
        raise KeyError(f"unknown stage: {stage}")
    return ensure_dir(roots[stage] / experiment_id)
