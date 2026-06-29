from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from defaults import DEFAULT_CONFIG
from io_utils import ensure_dir, load_json, now_iso, save_json, start_verbose_print_capture
from model_types import DNNArch, TrainConfig
from training import train_reference_solutions_simple_batched


def _hash_config(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def merged_config(config_path: Path, upstream_manifest: Path, force: bool) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if Path(config_path).exists():
        config.update(load_json(Path(config_path), {}) or {})
    config["force"] = bool(force or config.get("force", False))
    config["upstream_manifest"] = str(upstream_manifest)
    return config


def _repo_root(part_root: Path) -> Path:
    return Path(part_root).resolve().parents[1]


def _resolve_repo_path(part_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _repo_root(part_root) / path


def _repo_relative(part_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(_repo_root(part_root)).as_posix()
    except ValueError:
        return str(path)


def _read_dataset_index(part_root: Path, upstream_manifest: Path) -> list[dict[str, str]]:
    manifest = load_json(Path(upstream_manifest), None)
    candidates: list[Path] = []
    if isinstance(manifest, dict):
        for value in manifest.get("summary_outputs", []):
            path = _resolve_repo_path(part_root, value)
            if path.name == "dataset_index.csv":
                candidates.append(path)
        raw_root = manifest.get("raw_output_root")
        if raw_root:
            root = _resolve_repo_path(part_root, raw_root)
            candidates.append(root.parent / "dataset_index.csv")
    candidates.append(Path(upstream_manifest))
    for path in candidates:
        if path.exists() and path.name == "dataset_index.csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
    raise FileNotFoundError(f"Could not locate dataset_index.csv from {upstream_manifest}")


def _filter_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, str]]:
    target_series = set(str(x) for x in config.get("target_series", ["beta"]))
    beta_values = [round(float(x), 8) for x in config.get("target_beta_values", [])]
    start = int(config.get("start_dataset_index", 0))
    max_datasets = int(config.get("max_datasets", 0))
    out: list[dict[str, str]] = []
    by_cell_count: dict[str, int] = {}
    for row in rows:
        if str(row.get("series", "")) not in target_series:
            continue
        beta = round(float(row.get("beta_ising", 0.0)), 8)
        if beta_values and beta not in beta_values:
            continue
        dataset_id = int(row.get("dataset_id", 0))
        if dataset_id < start:
            continue
        count = by_cell_count.get(str(row["cell_id"]), 0)
        if max_datasets and count >= max_datasets:
            continue
        by_cell_count[str(row["cell_id"])] = count + 1
        out.append(row)
    return out


def _load_dataset(part_root: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path = _resolve_repo_path(part_root, row["dataset_raw_path"])
    data = np.load(path)
    x = np.asarray(data["X_train"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    if set(float(v) for v in np.unique(y)).issubset({0.0, 1.0}):
        y = 2.0 * y - 1.0
    return x, y


def _attempt_seed(config: dict[str, Any], row: dict[str, str], attempt_id: int) -> int:
    return int(config.get("seed", 0)) + int(row.get("dataset_id", 0)) * 10000 + int(attempt_id)


def _attempt_fields() -> list[str]:
    return [
        "attempt_id",
        "optimizer_chain",
        "final_train_loss",
        "final_cls_err",
        "n_wrong",
        "final_train_accuracy",
        "final_train_accuracy_percent",
        "min_signed_margin",
        "is_exact_solution",
        "sampler_eligible",
        "P_params",
        "adam_epochs_completed",
        "adam_early_stop_epoch",
        "adam_early_stop_reached",
        "lbfgs_iters_completed",
        "lbfgs_early_stop_epoch",
        "lbfgs_early_stop_reached",
        "early_stop_phase",
        "early_stop_epoch",
        "theta_norm",
        "theta_init_norm",
        "theta_final_path",
        "theta_init_path",
        "summary_path",
    ]


def _write_attempt_rows(part_root: Path, width_dir: Path, outputs: list[dict[str, Any]]) -> None:
    with (width_dir / "attempt_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_attempt_fields())
        writer.writeheader()
        for output in outputs:
            attempt_id = int(output["attempt_id"])
            summary = output["summary"]
            attempt_dir = width_dir / f"attempt_{attempt_id:03d}"
            ensure_dir(attempt_dir)
            theta_final = attempt_dir / "theta_final.npy"
            theta_init = attempt_dir / "theta_init.npy"
            summary_path = attempt_dir / "train_summary.json"
            np.save(theta_final, np.asarray(output["theta"], dtype=np.float64))
            np.save(theta_init, np.asarray(output["theta_init"], dtype=np.float64))
            save_json(summary_path, summary)
            writer.writerow(
                {
                    "attempt_id": attempt_id,
                    "optimizer_chain": summary.get("optimizer_chain", ""),
                    "final_train_loss": summary.get("final_train_loss", ""),
                    "final_cls_err": summary.get("final_cls_err", ""),
                    "n_wrong": summary.get("n_wrong", ""),
                    "final_train_accuracy": summary.get("final_train_accuracy", ""),
                    "final_train_accuracy_percent": summary.get("final_train_accuracy_percent", ""),
                    "min_signed_margin": summary.get("min_signed_margin", ""),
                    "is_exact_solution": summary.get("is_exact_solution", ""),
                    "sampler_eligible": summary.get("sampler_eligible", ""),
                    "P_params": summary.get("P_params", ""),
                    "adam_epochs_completed": summary.get("adam_epochs_completed", ""),
                    "adam_early_stop_epoch": summary.get("adam_early_stop_epoch", ""),
                    "adam_early_stop_reached": summary.get("adam_early_stop_reached", ""),
                    "lbfgs_iters_completed": summary.get("lbfgs_iters_completed", ""),
                    "lbfgs_early_stop_epoch": summary.get("lbfgs_early_stop_epoch", ""),
                    "lbfgs_early_stop_reached": summary.get("lbfgs_early_stop_reached", ""),
                    "early_stop_phase": summary.get("early_stop_phase", ""),
                    "early_stop_epoch": summary.get("early_stop_epoch", ""),
                    "theta_norm": summary.get("theta_norm", ""),
                    "theta_init_norm": summary.get("theta_init_norm", ""),
                    "theta_final_path": _repo_relative(part_root, theta_final),
                    "theta_init_path": _repo_relative(part_root, theta_init),
                    "summary_path": _repo_relative(part_root, summary_path),
                }
            )


def run_pipeline(*, part_root: Path, config_path: Path, upstream_manifest: Path, force: bool, verbose: bool = False) -> Path:
    started_at = now_iso()
    config = merged_config(config_path, upstream_manifest, force)
    run_name = str(config.get("run_name", f"reference_search_{_hash_config(config)}"))
    summary_root = Path(part_root) / "raw_outputs" / run_name
    ensure_dir(summary_root)
    log_capture = start_verbose_print_capture(summary_root, enabled=verbose)
    rows = _filter_rows(_read_dataset_index(part_root, upstream_manifest), config)
    widths = [int(x) for x in config.get("widths", [48])]
    attempt_count = int(config.get("attempts_per_dataset", 200))
    arch_by_width = {width: DNNArch(input_dim=2, width1=width, width2=width) for width in widths}
    completed = 0
    for row in rows:
        x, y = _load_dataset(part_root, row)
        dataset_tag = Path(row["dataset_raw_path"]).parent.name
        for width in widths:
            width_dir = summary_root / str(row["cell_id"]) / dataset_tag / f"width_{width:03d}"
            attempt_csv = width_dir / "attempt_results.csv"
            if attempt_csv.exists() and not bool(config.get("force", False)):
                completed += 1
                continue
            ensure_dir(width_dir)
            cfgs = [
                TrainConfig(
                    lr=float(config.get("train_lr", 0.03)),
                    weight_decay=float(config.get("train_weight_decay", 1.0e-4)),
                    momentum=float(config.get("train_momentum", 0.9)),
                    epochs=int(config.get("adam_epochs_ref", 4000)),
                    seed=_attempt_seed(config, row, attempt_id),
                    optimizer_name="adam",
                    lbfgs_max_iter=int(config.get("lbfgs_max_iter", 4000)),
                    init_scale_multiplier=1.0,
                    activation=str(config.get("activation", "tanh")),
                    loss=str(config.get("ref_loss_name", "logistic")),
                    margin=float(config.get("margin", 1.0)),
                )
                for attempt_id in range(attempt_count)
            ]
            outputs = train_reference_solutions_simple_batched(
                x,
                y,
                arch_by_width[width],
                cfgs,
                device=str(config.get("sampling_device", "cuda")),
                verbose=verbose,
                progress_label=f"{row['cell_id']}/{dataset_tag}/width_{width:03d}",
            )
            _write_attempt_rows(part_root, width_dir, outputs)
            completed += 1
    save_json(summary_root / "run_config.json", config)
    save_json(
        summary_root / "manifest.json",
        {
            "pipeline_id": str(config.get("pipeline_id", "simple_reference_search")),
            "methodology_id": str(config.get("methodology_id", "batched_adam_lbfgsb_simple_v1")),
            "status": "success",
            "started_at": started_at,
            "finished_at": now_iso(),
            "config_path": str(config_path),
            "upstream_manifest": str(upstream_manifest),
            "unit_count": completed,
            "summary_outputs": [_repo_relative(part_root, summary_root / "run_config.json")],
        },
    )
    if log_capture is not None:
        log_capture.close()
    return summary_root / "manifest.json"
