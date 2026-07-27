from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ROOT = SCRIPT_DIR.parent
PAIRWISE_ROOT = STAGE_ROOT.parent
DNN_ROOT = PAIRWISE_ROOT.parent
PROJECT_ROOT = DNN_ROOT.parents[1]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.dnn_model import ARCH, P
from utils.io_utils import ensure_dir, read_json, write_json
from utils.pm_sais_core import load_config, run_smc_split


RAW_ROOT = STAGE_ROOT / "raw_outputs" / "shell_pool"
STATUS_PATH = STAGE_ROOT / "raw_outputs" / "SAMPLING_STATUS.json"
UNIT_INDEX_PATH = RAW_ROOT / "unit_index.csv"
DEFAULT_REFERENCE_INDEX = PAIRWISE_ROOT / "03_reference_search" / "raw_outputs" / "reference_index.csv"

UNIT_INDEX_FIELDS = [
    "pair_id",
    "pair_label",
    "digit_a",
    "digit_b",
    "pair_order",
    "pair_rank_complexity_desc",
    "complexity_mean",
    "dataset_id",
    "ref_id",
    "ref_path_id",
    "radius",
    "radius_path_id",
    "samples_path",
    "unit_summary_path",
    "theta_path",
    "dataset_path",
    "seed",
    "smc_completed",
    "ess_fraction",
    "split_logZ_per_P_diff",
    "split_dlogZ_dr_per_P_diff",
]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        (STAGE_ROOT / path).resolve(),
        (PAIRWISE_ROOT / path).resolve(),
        (DNN_ROOT / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _project_relative(path: str | Path) -> str:
    resolved = _resolve_path(path).resolve()
    for root in (PROJECT_ROOT.resolve(), DNN_ROOT.resolve()):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIT_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in UNIT_INDEX_FIELDS} for row in rows])
    tmp.replace(path)


def _radius_path_id(radius: float) -> str:
    return f"r_{float(radius):0.4f}".replace(".", "p")


def _load_reference_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    reference_cfg = config.get("reference_search") or {}
    reference_index = _resolve_path(reference_cfg.get("reference_index", DEFAULT_REFERENCE_INDEX))
    if not reference_index.exists():
        raise FileNotFoundError(reference_index)
    rows = _read_csv(reference_index)
    selected = [str(value) for value in (config.get("ensemble") or {}).get("condition_values", [])]
    selected_set = set(selected)
    out: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row["pair_id"])
        if selected_set and pair_id not in selected_set:
            continue
        out.append(
            {
                **row,
                "pair_id": pair_id,
                "pair_label": str(row.get("pair_label", pair_id)),
                "digit_a": int(float(row.get("digit_a", 0))),
                "digit_b": int(float(row.get("digit_b", 0))),
                "pair_order": int(float(row.get("pair_order", 0))),
                "pair_rank_complexity_desc": int(float(row.get("pair_rank_complexity_desc", 0))),
                "complexity_mean": float(row.get("complexity_mean", "nan")),
                "dataset_id": int(float(row.get("dataset_id", 0))),
                "ref_id": int(float(row.get("ref_id", 0))),
                "CE_mean_train": float(row["CE_mean_train"]),
                "resample_seed_offset": int(float(row.get("resample_seed_offset", config["sampling"].get("seed_offset", 0)))),
            }
        )
    if not out:
        raise ValueError(f"no selected references found in {reference_index}")
    order = {pair_id: idx for idx, pair_id in enumerate(selected)}
    return sorted(out, key=lambda row: (order.get(str(row["pair_id"]), 10**9), int(row["ref_id"])))


def _set_process_threads(threads: int) -> None:
    threads = max(1, int(threads))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def _configure_threads(config: dict[str, Any], workers: int = 1) -> tuple[int, int]:
    count = os.cpu_count() or 1
    fraction = float((config.get("compute") or {}).get("cpu_thread_fraction", 0.8))
    total_threads = max(1, int(math.floor(count * fraction)))
    per_worker_threads = max(1, int(math.floor(total_threads / max(1, int(workers)))))
    _set_process_threads(per_worker_threads if int(workers) > 1 else total_threads)
    return total_threads, per_worker_threads


def _available_cuda_devices() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def _worker_devices(config: dict[str, Any], workers: int) -> list[str]:
    requested = str((config.get("compute") or {}).get("device", "auto"))
    if requested != "auto":
        return [requested for _ in range(max(1, int(workers)))]
    cuda_devices = _available_cuda_devices()
    if not cuda_devices:
        return ["cpu" for _ in range(max(1, int(workers)))]
    return [cuda_devices[idx % len(cuda_devices)] for idx in range(max(1, int(workers)))]


def _unit_output_paths(row: dict[str, Any], radius: float) -> tuple[Path, Path, Path]:
    ref_path_id = str(row.get("ref_path_id") or f"ref_{int(row['ref_id']) + 1:03d}")
    radius_id = _radius_path_id(radius)
    root = RAW_ROOT / str(row["pair_id"]) / ref_path_id / radius_id
    return root, root / "unit_summary.json", root / "samples.npz"


def _summary_payload(
    row: dict[str, Any],
    radius: float,
    seed: int,
    result: dict[str, Any],
    unit_summary_path: Path,
    samples_path: Path,
    lambda_reg: float,
) -> dict[str, Any]:
    summary = {key: value for key, value in result.items() if key != "_samples_npz"}
    summary.update(
        {
            "stage": "03_dnn_mnist",
            "block": "digit_pairwise_complexity_dense",
            "condition_name": "digit_pair",
            "condition_value": str(row["pair_id"]),
            "pair_id": str(row["pair_id"]),
            "pair_label": str(row["pair_label"]),
            "digit_a": int(row["digit_a"]),
            "digit_b": int(row["digit_b"]),
            "pair_order": int(row["pair_order"]),
            "pair_rank_complexity_desc": int(row["pair_rank_complexity_desc"]),
            "complexity_mean": float(row["complexity_mean"]),
            "dataset_id": int(row["dataset_id"]),
            "split_id": int(float(row.get("split_id", 0))),
            "ref_id": int(row["ref_id"]),
            "ref_path_id": str(row.get("ref_path_id") or f"ref_{int(row['ref_id']) + 1:03d}"),
            "radius": float(radius),
            "radius_path_id": _radius_path_id(radius),
            "P": int(P),
            "P_params": int(P),
            "lambda_reg": float(result.get("lambda_reg", lambda_reg)),
            "sampler_method": str(result.get("sampler_method", "exact_shell_l2_vmf_adaptive_ce_tempered_smc")),
            "seed": int(seed),
            "theta_path": _project_relative(row["theta_path"]),
            "dataset_path": _project_relative(row["dataset_path"]),
            "CE_mean_train": float(row["CE_mean_train"]),
            "samples_path": _project_relative(samples_path),
            "unit_summary_path": _project_relative(unit_summary_path),
        }
    )
    return summary


def _unit_index_row(row: dict[str, Any], radius: float, seed: int, summary: dict[str, Any], unit_summary_path: Path, samples_path: Path) -> dict[str, Any]:
    return {
        "pair_id": str(row["pair_id"]),
        "pair_label": str(row["pair_label"]),
        "digit_a": int(row["digit_a"]),
        "digit_b": int(row["digit_b"]),
        "pair_order": int(row["pair_order"]),
        "pair_rank_complexity_desc": int(row["pair_rank_complexity_desc"]),
        "complexity_mean": float(row["complexity_mean"]),
        "dataset_id": int(row["dataset_id"]),
        "ref_id": int(row["ref_id"]),
        "ref_path_id": str(row.get("ref_path_id") or f"ref_{int(row['ref_id']) + 1:03d}"),
        "radius": float(radius),
        "radius_path_id": _radius_path_id(radius),
        "samples_path": _project_relative(samples_path),
        "unit_summary_path": _project_relative(unit_summary_path),
        "theta_path": _project_relative(row["theta_path"]),
        "dataset_path": _project_relative(row["dataset_path"]),
        "seed": int(seed),
        "smc_completed": bool(summary.get("smc_completed", False)),
        "ess_fraction": float(summary.get("ess_fraction", np.nan)),
        "split_logZ_per_P_diff": float(summary.get("split_logZ_per_P_diff", np.nan)),
        "split_dlogZ_dr_per_P_diff": float(summary.get("split_dlogZ_dr_per_P_diff", np.nan)),
    }


def _sort_unit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda u: (int(float(u.get("pair_order", 0))), str(u["ref_path_id"]), float(u["radius"])))


def _replace_unit_row(unit_rows: list[dict[str, Any]], key: tuple[str, str, str], row: dict[str, Any]) -> list[dict[str, Any]]:
    unit_rows = [
        current
        for current in unit_rows
        if not (
            str(current["pair_id"]) == key[0]
            and str(current["ref_path_id"]) == key[1]
            and f"{float(current['radius']):.10f}" == key[2]
        )
    ]
    unit_rows.append(row)
    return unit_rows


def _run_unit_task(task: dict[str, Any]) -> dict[str, Any]:
    config = dict(task["config"])
    config["compute"] = dict(config.get("compute") or {})
    config["compute"]["device"] = str(task["device"])
    _set_process_threads(int(task["threads_per_worker"]))
    row = dict(task["row"])
    radius = float(task["radius"])
    seed = int(task["seed"])
    lambda_reg = float(task["lambda_reg"])
    root, unit_summary_path, samples_path = _unit_output_paths(row, radius)
    result = run_smc_split(
        np.load(_resolve_path(row["theta_path"]), allow_pickle=False).astype(np.float64).reshape(-1),
        _load_dataset_cached(row["dataset_path"]),
        radius,
        int(task["n_samples"]),
        lambda_reg,
        seed,
        config,
        float(row["CE_mean_train"]),
    )
    ensure_dir(root)
    samples = result.get("_samples_npz", {})
    if bool(config["sampling"].get("save_unit_samples_npz", True)):
        np.savez_compressed(samples_path, **samples)
    else:
        samples_path = Path("")
    summary = _summary_payload(row, radius, seed, result, unit_summary_path, samples_path, lambda_reg)
    summary["worker_device"] = str(task["device"])
    write_json(unit_summary_path, summary)
    return _unit_index_row(row, radius, seed, summary, unit_summary_path, samples_path)


def run_pipeline(*, force: bool = False, limit_units: int | None = None, workers: int | None = None) -> Path:
    config = load_config()
    rows = _load_reference_rows(config)
    radii = [float(value) for value in config["sampling"]["radii"]]
    n_samples = int(config["sampling"].get("samples_per_ref_radius", 1024))
    lambda_reg = float(config["sampling"].get("lambda_reg", 1.0))
    seed_offset = int(config["sampling"].get("seed_offset", 2026070100))
    requested_workers = int(workers or (config.get("compute") or {}).get("parallel_workers", 1))
    requested_workers = max(1, requested_workers)
    total_threads, threads_per_worker = _configure_threads(config, requested_workers)
    devices = _worker_devices(config, requested_workers)
    expected_units = len(rows) * len(radii)
    ensure_dir(RAW_ROOT)
    started = time.time()
    unit_rows: list[dict[str, Any]] = []
    if UNIT_INDEX_PATH.exists() and not force:
        unit_rows = [
            {
                **row,
                "ref_id": int(float(row["ref_id"])),
                "radius": float(row["radius"]),
                "seed": int(float(row["seed"])),
            }
            for row in _read_csv(UNIT_INDEX_PATH)
        ]

    completed = {
        (str(row["pair_id"]), str(row["ref_path_id"]), f"{float(row['radius']):.10f}")
        for row in unit_rows
        if _resolve_path(row["unit_summary_path"]).exists()
    }
    done_this_run = 0
    print(
        f"[digit-pair-sampling] references={len(rows)} radii={len(radii)} expected_units={expected_units} "
        f"n_samples={n_samples} workers={requested_workers} total_threads={total_threads} "
        f"threads_per_worker={threads_per_worker} devices={','.join(devices)}",
        flush=True,
    )
    tasks: list[dict[str, Any]] = []
    for row in rows:
        for radius_index, radius in enumerate(radii):
            ref_path_id = str(row.get("ref_path_id") or f"ref_{int(row['ref_id']) + 1:03d}")
            key = (str(row["pair_id"]), ref_path_id, f"{float(radius):.10f}")
            root, unit_summary_path, samples_path = _unit_output_paths(row, radius)
            if key in completed and not force:
                continue
            seed = int(row["resample_seed_offset"]) + seed_offset + int(row["ref_id"]) * 1009 + radius_index
            if requested_workers > 1:
                worker_index = len(tasks) % requested_workers
                tasks.append(
                    {
                        "config": config,
                        "row": row,
                        "radius": radius,
                        "seed": seed,
                        "lambda_reg": lambda_reg,
                        "n_samples": n_samples,
                        "device": devices[worker_index],
                        "threads_per_worker": threads_per_worker,
                    }
                )
                if limit_units is not None and len(tasks) >= int(limit_units):
                    break
                continue
            result = run_smc_split(
                np.load(_resolve_path(row["theta_path"]), allow_pickle=False).astype(np.float64).reshape(-1),
                _load_dataset_cached(row["dataset_path"]),
                radius,
                n_samples,
                lambda_reg,
                seed,
                config,
                float(row["CE_mean_train"]),
            )
            ensure_dir(root)
            samples = result.get("_samples_npz", {})
            if bool(config["sampling"].get("save_unit_samples_npz", True)):
                np.savez_compressed(samples_path, **samples)
            else:
                samples_path = Path("")
            summary = _summary_payload(row, radius, seed, result, unit_summary_path, samples_path, lambda_reg)
            write_json(unit_summary_path, summary)
            unit_rows = _replace_unit_row(unit_rows, key, _unit_index_row(row, radius, seed, summary, unit_summary_path, samples_path))
            _write_csv(UNIT_INDEX_PATH, _sort_unit_rows(unit_rows))
            done_this_run += 1
            if done_this_run % 25 == 0 or done_this_run == 1:
                print(
                    f"[digit-pair-sampling] wrote={done_this_run} total_indexed={len(unit_rows)}/{expected_units} "
                    f"last={row['pair_id']} {ref_path_id} r={radius:.2f}",
                    flush=True,
                )
            if limit_units is not None and done_this_run >= int(limit_units):
                break
        if requested_workers > 1 and limit_units is not None and len(tasks) >= int(limit_units):
            break
        if limit_units is not None and done_this_run >= int(limit_units):
            break

    if requested_workers > 1 and tasks:
        print(f"[digit-pair-sampling] dispatching parallel_tasks={len(tasks)}", flush=True)
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=requested_workers, mp_context=context) as executor:
            futures = {executor.submit(_run_unit_task, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                key = (str(row["pair_id"]), str(row["ref_path_id"]), f"{float(row['radius']):.10f}")
                unit_rows = _replace_unit_row(unit_rows, key, row)
                done_this_run += 1
                if done_this_run % 25 == 0 or done_this_run == 1:
                    _write_csv(UNIT_INDEX_PATH, _sort_unit_rows(unit_rows))
                    print(
                        f"[digit-pair-sampling] wrote={done_this_run} total_indexed={len(unit_rows)}/{expected_units} "
                        f"last={row['pair_id']} {row['ref_path_id']} r={float(row['radius']):.2f}",
                        flush=True,
                    )
        _write_csv(UNIT_INDEX_PATH, _sort_unit_rows(unit_rows))

    status = {
        "status": "complete" if len(unit_rows) >= expected_units else "partial",
        "expected_units": int(expected_units),
        "indexed_units": int(len(unit_rows)),
        "units_written_this_run": int(done_this_run),
        "condition_values": [str(value) for value in (config.get("ensemble") or {}).get("condition_values", [])],
        "references": int(len(rows)),
        "radii": int(len(radii)),
        "samples_per_ref_radius": int(n_samples),
        "cpu_threads_total_budget": int(total_threads),
        "cpu_threads_per_worker": int(threads_per_worker),
        "workers": int(requested_workers),
        "worker_devices": devices,
        "compute_device": str(config["compute"]["device"]),
        "elapsed_s": float(time.time() - started),
    }
    write_json(STATUS_PATH, status)
    print(f"[digit-pair-sampling] status={status['status']} indexed={len(unit_rows)}/{expected_units}", flush=True)
    return UNIT_INDEX_PATH


_DATASET_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _load_dataset_cached(path_value: str | Path) -> dict[str, np.ndarray]:
    path = _resolve_path(path_value)
    key = path.resolve().as_posix()
    cached = _DATASET_CACHE.get(key)
    if cached is not None:
        return cached
    with np.load(path, allow_pickle=False) as data:
        out = {name: data[name] for name in data.files}
        out["X_train"] = np.asarray(out["X_train"], dtype=np.float64)
        out["y_train"] = np.asarray(out["y_train"])
    _DATASET_CACHE[key] = out
    return out


def check_layout() -> dict[str, Any]:
    config = load_config()
    rows: list[dict[str, Any]] = []
    ref_error = ""
    try:
        rows = _load_reference_rows(config)
    except Exception as exc:
        ref_error = str(exc)
    radii = [float(value) for value in config["sampling"]["radii"]]
    unit_json_count = len(list(RAW_ROOT.glob("pair_*/ref_*/r_*/unit_summary.json")))
    sample_count = len(list(RAW_ROOT.glob("pair_*/ref_*/r_*/samples.npz")))
    return {
        "stage_root": str(STAGE_ROOT),
        "entrypoint": "src/sampling.py",
        "run_modes": ["run", "check"],
        "status": "ready" if rows else "missing_references",
        "reference_error": ref_error,
        "condition_values": [str(value) for value in (config.get("ensemble") or {}).get("condition_values", [])],
        "reference_rows": len(rows),
        "radii": len(radii),
        "expected_units": len(rows) * len(radii),
        "raw_unit_summary_json": unit_json_count,
        "raw_samples_npz": sample_count,
        "unit_index_exists": UNIT_INDEX_PATH.exists(),
        "architecture": {
            "input_dim": int(ARCH.input_dim),
            "hidden_width": int(ARCH.hidden_width),
            "hidden_layers": int(ARCH.hidden_layers),
            "activation": ARCH.activation,
            "P": int(P),
        },
        "compute": config.get("compute", {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MNIST digit-pair PM-SAIS sampling.")
    parser.add_argument("--mode", choices=["run", "check"], default="run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-units", type=int, default=None, help="Debug only: stop after this many newly written units.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel unit workers. Defaults to compute.parallel_workers.")
    args = parser.parse_args(argv)
    if args.mode == "check":
        print(json.dumps(check_layout(), indent=2, sort_keys=True))
        return 0
    path = run_pipeline(force=bool(args.force), limit_units=args.limit_units, workers=args.workers)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
