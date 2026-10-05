#!/usr/bin/env python3
"""Build and validate the compact label-noise complexity table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from sklearn.neighbors import NearestNeighbors


STAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = STAGE_ROOT / "config" / "default.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "mnist.complexity-summary.v1" or config.get("protocol") != "label_noise_sweep":
        raise ValueError("config is not a supported label-noise complexity summary")
    indices = [int(value) for value in config["dataset_indices"]]
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError("dataset_indices must be unique non-negative integers")
    names = [str(row["condition"]) for row in config["conditions"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("conditions must be non-empty and unique")
    selected = int(config["complexity"]["selected_scales"])
    audit = int(config["complexity"]["neighbor_audit_max"])
    n_train = int(config["n_train"])
    if not 1 <= selected <= audit < n_train:
        raise ValueError("complexity scales must satisfy 1 <= selected <= audit < n_train")
    low, high = map(float, config["complexity"]["finite_range"])
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high):
        raise ValueError("complexity finite_range is invalid")
    return config


def resolve_input_root(config: Mapping[str, Any]) -> Path:
    value = Path(str(config["paths"]["dataset_root"]))
    return value if value.is_absolute() else PROTOCOL_ROOT / value


def resolve_output_path(config: Mapping[str, Any], output_dir: Path | None) -> Path:
    configured = Path(str(config["outputs"]["cms_by_dataset"]))
    return output_dir / configured.name if output_dir is not None else (configured if configured.is_absolute() else STAGE_ROOT / configured)


def chance_disagreement(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    n = int(labels.size)
    return float((n * n - int(np.sum(counts.astype(np.int64) ** 2))) / (n * (n - 1)))


def multiscale_complexity(path: Path, config: Mapping[str, Any]) -> float:
    n_train = int(config["n_train"])
    selected = int(config["complexity"]["selected_scales"])
    audit = int(config["complexity"]["neighbor_audit_max"])
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["x_train"], dtype=np.float64)
        labels = np.asarray(payload["y_train"]).reshape(-1)
    if points.shape[0] != n_train or labels.shape != (n_train,) or not np.all(np.isfinite(points)):
        raise ValueError(f"{path}: invalid training data")
    if not np.all(np.isin(labels, (-1, 1))):
        raise ValueError(f"{path}: labels must use the -1/+1 encoding")
    indices = NearestNeighbors(n_neighbors=audit + 1, n_jobs=1).fit(points).kneighbors(points, return_distance=False)
    if not np.array_equal(indices[:, 0], np.arange(n_train, dtype=indices.dtype)):
        raise ValueError(f"{path}: self is not the first neighbor")
    exact = np.mean(labels[:, None] != labels[indices[:, 1:]], axis=0)
    cumulative = np.cumsum(exact) / np.arange(1, audit + 1, dtype=np.float64)
    chance = chance_disagreement(labels)
    if not math.isfinite(chance) or chance <= 0.0:
        raise ValueError(f"{path}: degenerate labels")
    return float(np.mean((cumulative / chance)[:selected]))


def build_rows(config: Mapping[str, Any], dataset_root: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    for order, spec in enumerate(config["conditions"]):
        condition = str(spec["condition"])
        for dataset_index in config["dataset_indices"]:
            index = int(dataset_index)
            rows.append({
                "domain": "mnist_label_noise",
                "condition": condition,
                "condition_order": order,
                "noise_eta": float(spec["noise_eta"]),
                "dataset_index": index,
                "C_MS": multiscale_complexity(dataset_root / f"dataset_{index:03d}" / condition / "dataset.npz", config),
            })
    validate_rows(rows, config)
    return rows, ["domain", "condition", "condition_order", "noise_eta", "dataset_index", "C_MS"]


def validate_rows(rows: list[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    expected = {(str(spec["condition"]), int(index)) for spec in config["conditions"] for index in config["dataset_indices"]}
    actual = [(str(row["condition"]), int(row["dataset_index"])) for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError("complexity output contains duplicate condition/dataset coordinates")
    if set(actual) != expected:
        raise ValueError("complexity output has missing or unexpected coordinates")
    low, high = map(float, config["complexity"]["finite_range"])
    for row in rows:
        spec = next(value for value in config["conditions"] if str(value["condition"]) == str(row["condition"]))
        if str(row["domain"]) != "mnist_label_noise":
            raise ValueError("complexity output domain drift")
        if int(row["condition_order"]) != config["conditions"].index(spec):
            raise ValueError("complexity output condition order drift")
        if float(row["noise_eta"]) != float(spec["noise_eta"]):
            raise ValueError("complexity output noise_eta drift")
        value = float(row["C_MS"])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"C_MS outside configured finite range: {value}")


def atomic_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.check_only and (args.execute or args.force):
        parser.error("--check-only cannot be combined with --execute or --force")
    config = load_config(args.config.resolve())
    output = resolve_output_path(config, args.output_dir)
    if args.force or (args.execute and not output.is_file()):
        rows, fields = build_rows(config, resolve_input_root(config))
        atomic_csv(output, rows, fields)
    if not output.is_file():
        raise FileNotFoundError(f"{output}; run with --execute")
    rows = read_rows(output)
    validate_rows(rows, config)
    print({"status": "validated", "path": str(output), "rows": len(rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
