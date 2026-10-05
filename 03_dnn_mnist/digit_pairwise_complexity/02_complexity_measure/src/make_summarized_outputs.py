#!/usr/bin/env python3
"""Build and validate the compact MNIST multiscale-complexity outputs."""

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
    expected_protocol = (
        "label_noise_sweep"
        if PROTOCOL_ROOT.name == "label_noise_sweep"
        else "digit_pairwise_sweep"
    )
    if config.get("schema_version") != "mnist.complexity-summary.v1":
        raise ValueError("unsupported complexity-summary config schema")
    if config.get("protocol") != expected_protocol:
        raise ValueError("config protocol disagrees with its experiment directory")
    indices = [int(value) for value in config["dataset_indices"]]
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError("dataset_indices must be unique non-negative integers")
    n_train = int(config["n_train"])
    selected = int(config["complexity"]["selected_scales"])
    audit = int(config["complexity"]["neighbor_audit_max"])
    if not 1 <= selected <= audit < n_train:
        raise ValueError("complexity scales must satisfy 1 <= selected <= audit < n_train")
    low, high = map(float, config["complexity"]["finite_range"])
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high):
        raise ValueError("complexity finite_range is invalid")
    return config


def resolve_input_root(config: Mapping[str, Any]) -> Path:
    value = Path(str(config["paths"]["dataset_root"]))
    return value if value.is_absolute() else PROTOCOL_ROOT / value


def resolve_output_path(
    config: Mapping[str, Any], key: str, output_dir: Path | None
) -> Path:
    configured = Path(str(config["outputs"][key]))
    if output_dir is not None:
        return output_dir / configured.name
    return configured if configured.is_absolute() else STAGE_ROOT / configured


def frozen_manifest(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    del config_path
    value = Path(str(config["pair_selection"]["frozen_manifest"]))
    path = value if value.is_absolute() else STAGE_ROOT / value
    return json.loads(path.read_text(encoding="utf-8"))


def condition_specs(
    config: Mapping[str, Any], config_path: Path
) -> list[dict[str, Any]]:
    if config["protocol"] == "label_noise_sweep":
        specs = [dict(row) for row in config["conditions"]]
    else:
        specs = [dict(row) for row in frozen_manifest(config, config_path)["expected_selected_pairs"]]
        for order, row in enumerate(specs):
            row["condition"] = str(row["pair_id"])
            row["condition_order"] = order
            row["pair_rank"] = int(row["rank"])
    names = [str(row["condition"]) for row in specs]
    if not names or len(names) != len(set(names)):
        raise ValueError("condition specifications are empty or duplicated")
    return specs


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
    if points.shape[0] != n_train or labels.shape != (n_train,):
        raise ValueError(f"{path}: unexpected training-data shape")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path}: non-finite features")
    if not np.all(np.isin(labels, (-1, 1))):
        raise ValueError(f"{path}: labels must use the -1/+1 encoding")
    indices = (
        NearestNeighbors(n_neighbors=audit + 1, n_jobs=1)
        .fit(points)
        .kneighbors(points, return_distance=False)
    )
    if not np.array_equal(indices[:, 0], np.arange(n_train, dtype=indices.dtype)):
        raise ValueError(f"{path}: self is not the first neighbor")
    exact = np.mean(labels[:, None] != labels[indices[:, 1:]], axis=0)
    cumulative = np.cumsum(exact) / np.arange(1, audit + 1, dtype=np.float64)
    chance = chance_disagreement(labels)
    if not math.isfinite(chance) or chance <= 0.0:
        raise ValueError(f"{path}: degenerate label-chance normalization")
    return float(np.mean((cumulative / chance)[:selected]))


def build_rows(
    config: Mapping[str, Any], config_path: Path, dataset_root: Path
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    specs = condition_specs(config, config_path)
    dataset_indices = [int(value) for value in config["dataset_indices"]]
    if config["protocol"] == "label_noise_sweep":
        fields = ["domain", "condition", "condition_order", "noise_eta", "dataset_index", "C_MS"]
        for order, spec in enumerate(specs):
            for dataset_index in dataset_indices:
                condition = str(spec["condition"])
                rows.append({
                    "domain": "mnist_label_noise",
                    "condition": condition,
                    "condition_order": order,
                    "noise_eta": float(spec["noise_eta"]),
                    "dataset_index": dataset_index,
                    "C_MS": multiscale_complexity(dataset_root / f"dataset_{dataset_index:03d}" / condition / "dataset.npz", config),
                })
    else:
        fields = ["domain", "condition", "condition_order", "digit_a", "digit_b", "pair_rank", "dataset_index", "C_MS"]
        for order, spec in enumerate(specs):
            for dataset_index in dataset_indices:
                condition = str(spec["condition"])
                rows.append({
                    "domain": "mnist_digit_pair",
                    "condition": condition,
                    "condition_order": order,
                    "digit_a": int(spec["digit_a"]),
                    "digit_b": int(spec["digit_b"]),
                    "pair_rank": int(spec["pair_rank"]),
                    "dataset_index": dataset_index,
                    "C_MS": multiscale_complexity(dataset_root / f"dataset_{dataset_index:03d}" / condition / "dataset.npz", config),
                })
    validate_rows(rows, config, config_path)
    return rows, fields


def validate_rows(
    rows: list[Mapping[str, Any]], config: Mapping[str, Any], config_path: Path
) -> None:
    specs = condition_specs(config, config_path)
    expected = {
        (str(spec["condition"]), int(dataset_index))
        for spec in specs
        for dataset_index in config["dataset_indices"]
    }
    actual = [(str(row["condition"]), int(row["dataset_index"])) for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError("complexity output contains duplicate condition/dataset coordinates")
    if set(actual) != expected:
        raise ValueError("complexity output has missing or unexpected coordinates")
    low, high = map(float, config["complexity"]["finite_range"])
    for row in rows:
        spec = next(value for value in specs if str(value["condition"]) == str(row["condition"]))
        expected_domain = "mnist_label_noise" if config["protocol"] == "label_noise_sweep" else "mnist_digit_pair"
        if str(row["domain"]) != expected_domain:
            raise ValueError("complexity output domain drift")
        expected_order = specs.index(spec)
        if int(row["condition_order"]) != expected_order:
            raise ValueError("complexity output condition order drift")
        for key in ("noise_eta", "digit_a", "digit_b", "pair_rank"):
            if key in spec and float(row[key]) != float(spec[key]):
                raise ValueError(f"complexity output {key} drift")
        value = float(row["C_MS"])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"C_MS outside configured finite range: {value}")


def build_pair_ranking_summary(
    config: Mapping[str, Any], config_path: Path, dataset_root: Path
) -> dict[str, Any]:
    if config["protocol"] != "digit_pairwise_sweep":
        raise ValueError("pair ranking is available only for digit_pairwise_sweep")
    dataset_indices = [int(value) for value in config["dataset_indices"]]
    all_pair_count = int(config["pair_selection"]["all_pair_count"])
    values: dict[str, list[float]] = {}
    coordinates: dict[str, tuple[int, int]] = {}
    for dataset_index in dataset_indices:
        path = dataset_root / f"dataset_{dataset_index:03d}" / "pair_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ranking = list(manifest["ranking"])
        if len(ranking) != all_pair_count:
            raise ValueError(f"{path}: expected {all_pair_count} ranked pairs")
        seen: set[str] = set()
        for row in ranking:
            pair_id = str(row["pair_id"])
            if pair_id in seen:
                raise ValueError(f"{path}: duplicate pair {pair_id}")
            seen.add(pair_id)
            coordinate = (int(row["digit_a"]), int(row["digit_b"]))
            if pair_id in coordinates and coordinates[pair_id] != coordinate:
                raise ValueError(f"{path}: pair coordinate drift for {pair_id}")
            coordinates[pair_id] = coordinate
            value = float(row["complexity"])
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite complexity for {pair_id}")
            values.setdefault(pair_id, []).append(value)
    if len(values) != all_pair_count or any(len(v) != len(dataset_indices) for v in values.values()):
        raise ValueError("pair manifests do not cover the full pair/dataset grid")
    ranking = sorted(
        ({
            "pair_id": pair_id,
            "digit_a": coordinates[pair_id][0],
            "digit_b": coordinates[pair_id][1],
            "C_MS": float(np.mean(pair_values, dtype=np.float64)),
        } for pair_id, pair_values in values.items()),
        key=lambda row: (-float(row["C_MS"]), int(row["digit_a"]), int(row["digit_b"])),
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    frozen = frozen_manifest(config, config_path)
    selected_ranks = [int(value) for value in frozen["selected_ranks_one_based"]]
    selected = [dict(ranking[rank - 1]) for rank in selected_ranks]
    expected = list(frozen["expected_selected_pairs"])
    atol = float(config["pair_selection"]["mean_atol"])
    for actual, target in zip(selected, expected, strict=True):
        if int(actual["rank"]) != int(target["rank"]) or str(actual["pair_id"]) != str(target["pair_id"]):
            raise ValueError("recomputed mean ranking disagrees with frozen pair selection")
        if not math.isclose(float(actual["C_MS"]), float(target["mean_complexity"]), rel_tol=0.0, abs_tol=atol):
            raise ValueError("recomputed pair mean disagrees with frozen C_MS authority")
    summary = {
        "schema_version": "mnist.digit-pair-mean-ranking.v1",
        "metric": "C_MS",
        "dataset_indices": dataset_indices,
        "dataset_count": len(dataset_indices),
        "all_pair_count": all_pair_count,
        "selection_ranks_one_based": selected_ranks,
        "tie_break": ["digit_a_ascending", "digit_b_ascending"],
        "ranking": ranking,
        "selected_pairs": selected,
    }
    validate_pair_ranking_summary(summary, config, config_path)
    return summary


def validate_pair_ranking_summary(
    summary: Mapping[str, Any], config: Mapping[str, Any], config_path: Path
) -> None:
    all_pair_count = int(config["pair_selection"]["all_pair_count"])
    ranking = [dict(row) for row in summary["ranking"]]
    if len(ranking) != all_pair_count:
        raise ValueError("pair-ranking summary has the wrong row count")
    ranks = [int(row["rank"]) for row in ranking]
    pair_ids = [str(row["pair_id"]) for row in ranking]
    if ranks != list(range(1, all_pair_count + 1)) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair-ranking summary ranks or pair ids are invalid")
    for row in ranking:
        value = float(row["C_MS"])
        if not math.isfinite(value):
            raise ValueError("pair-ranking summary contains non-finite C_MS")
        expected_id = f"pair_{int(row['digit_a'])}_{int(row['digit_b'])}"
        if str(row["pair_id"]) != expected_id:
            raise ValueError("pair-ranking summary id/coordinate drift")
    expected_order = sorted(
        ranking,
        key=lambda row: (-float(row["C_MS"]), int(row["digit_a"]), int(row["digit_b"])),
    )
    if [row["pair_id"] for row in ranking] != [row["pair_id"] for row in expected_order]:
        raise ValueError("pair-ranking summary ordering drift")
    frozen = frozen_manifest(config, config_path)
    selected_ranks = [int(value) for value in frozen["selected_ranks_one_based"]]
    selected = [dict(row) for row in summary["selected_pairs"]]
    if [int(row["rank"]) for row in selected] != selected_ranks:
        raise ValueError("pair-ranking selected ranks drift")
    for actual, target in zip(selected, frozen["expected_selected_pairs"], strict=True):
        ranked = ranking[int(target["rank"]) - 1]
        if str(actual["pair_id"]) != str(target["pair_id"]) or actual != ranked:
            raise ValueError("pair-ranking selected pair is inconsistent with ranking/freeze")
        if not math.isclose(float(actual["C_MS"]), float(target["mean_complexity"]), rel_tol=0.0, abs_tol=float(config["pair_selection"]["mean_atol"])):
            raise ValueError("pair-ranking selected mean disagrees with frozen authority")
    if int(summary["dataset_count"]) != len(config["dataset_indices"]) or int(summary["all_pair_count"]) != all_pair_count:
        raise ValueError("pair-ranking summary count metadata drift")


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


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def read_csv(path: Path) -> list[dict[str, str]]:
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
    config_path = args.config.resolve()
    config = load_config(config_path)
    dataset_root = resolve_input_root(config)
    csv_path = resolve_output_path(config, "cms_by_dataset", args.output_dir)
    ranking_path = (
        resolve_output_path(config, "pair_ranking_summary", args.output_dir)
        if config["protocol"] == "digit_pairwise_sweep"
        else None
    )
    missing = [path for path in (csv_path, ranking_path) if path is not None and not path.is_file()]
    if args.force or (args.execute and missing):
        rows: list[dict[str, object]] | None = None
        fields: list[str] | None = None
        if args.force or not csv_path.is_file():
            rows, fields = build_rows(config, config_path, dataset_root)
            atomic_csv(csv_path, rows, fields)
        if ranking_path is not None and (args.force or not ranking_path.is_file()):
            atomic_json(ranking_path, build_pair_ranking_summary(config, config_path, dataset_root))
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path}; run with --execute")
    validate_rows(read_csv(csv_path), config, config_path)
    print({"status": "validated", "path": str(csv_path), "rows": len(read_csv(csv_path))})
    if ranking_path is not None:
        if not ranking_path.is_file():
            raise FileNotFoundError(f"{ranking_path}; run with --execute")
        summary = json.loads(ranking_path.read_text(encoding="utf-8"))
        validate_pair_ranking_summary(summary, config, config_path)
        print({"status": "validated", "path": str(ranking_path), "rows": len(summary["ranking"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
