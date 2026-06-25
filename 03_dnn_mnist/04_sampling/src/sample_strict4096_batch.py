#!/usr/bin/env python3
"""Sample a parallel batch of baseline-4096 replacement units for strict fill."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
SCRIPT_DIR = Path(__file__).resolve().parent
PHI_SRC = LOCAL_ROOT / "05_proxy_local_entropy" / "src"
for path in (SCRIPT_DIR, PHI_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fill_qc_passed_phi as fill  # noqa: E402


_CFG: dict[str, Any] | None = None
_RUN_ROOT: Path | None = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def plan_tasks(
    *,
    replacement_run_root: Path,
    existing_policy: str,
    max_target_units: int,
    attempts_per_deficit: float,
) -> pd.DataFrame:
    refs, membership = fill.load_reference_tables(replacement_run_root)
    candidates = fill.candidate_refs(refs, membership)
    source_units = fill.load_source_units()
    replacement_units = fill.load_replacement_units(replacement_run_root)
    all_units = pd.concat([source_units, replacement_units], ignore_index=True, sort=False)
    _selection, summary = fill.build_selection(all_units, existing_policy)
    deficient = summary[summary["deficit_to_30"] > 0].copy().sort_values(["rule", "radius"])
    tasks: list[dict[str, Any]] = []
    planned = set()
    existing_replacement = set(
        (
            str(row.rule),
            int(row.ref_id),
            round(float(row.radius), 4),
        )
        for row in replacement_units[["rule", "ref_id", "radius"]].itertuples(index=False)
    ) if not replacement_units.empty else set()
    baseline_present = set(
        (
            str(row.rule),
            int(row.ref_id),
            round(float(row.radius), 4),
        )
        for row in all_units[all_units["baseline4096"].fillna(False)][["rule", "ref_id", "radius"]].itertuples(index=False)
    ) if not all_units.empty else set()

    def add_task(rule: str, ref_id: int, radius: float, kind: str) -> bool:
        key = (rule, int(ref_id), round(float(radius), 4))
        if key in planned or key in existing_replacement or key in baseline_present:
            return False
        row = fill.row_for_ref(refs, rule, int(ref_id))
        task = dict(row)
        task.update({"kind": kind, "rule": rule, "ref_id": int(ref_id), "radius": float(radius)})
        tasks.append(task)
        planned.add(key)
        return True

    for _, deficit_row in deficient.iterrows():
        if len(tasks) >= max_target_units:
            break
        rule = str(deficit_row["rule"])
        radius = float(deficit_row["radius"])
        need = int(deficit_row["deficit_to_30"])
        target_attempts = max(1, int(np.ceil(need * attempts_per_deficit)))
        added_for_group = 0
        for ref_id in candidates.get(rule, []):
            if len(tasks) >= max_target_units or added_for_group >= target_attempts:
                break
            if not np.isclose(radius, fill.R0):
                add_task(rule, int(ref_id), fill.R0, "anchor")
            if add_task(rule, int(ref_id), radius, "fill"):
                added_for_group += 1
    return pd.DataFrame(tasks)


def _init_worker(replacement_run_root: str, cpu_threads: int, device: str) -> None:
    global _CFG, _RUN_ROOT
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    if device:
        os.environ["MNIST14_DEVICE"] = device
    _RUN_ROOT = Path(replacement_run_root)
    _CFG = fill.configure_replacement_sampler(_RUN_ROOT, cpu_threads, device)


def _sample_worker(task: dict[str, Any]) -> dict[str, Any]:
    if _CFG is None or _RUN_ROOT is None:
        raise RuntimeError("worker was not initialized")
    started = time.time()
    row = dict(task)
    payload = fill.resample_module().sample_unit(row, float(task["radius"]), _CFG, _RUN_ROOT, force=False)
    frame = fill.payload_to_frame(payload)
    return {
        "kind": str(task["kind"]),
        "rule": str(task["rule"]),
        "ref_id": int(task["ref_id"]),
        "radius": float(task["radius"]),
        "elapsed_s": float(time.time() - started),
        "unit_qc_pass": bool(frame.iloc[0]["unit_qc_pass"]),
        "baseline4096": bool(frame.iloc[0]["baseline4096"]),
        "reused": bool(payload.get("reused", False)),
        "split_logZ_per_P_diff": float(frame.iloc[0]["split_logZ_per_P_diff"]),
        "ess_fraction": float(frame.iloc[0]["ess_fraction"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample a parallel strict-4096 replacement batch.")
    parser.add_argument("--replacement-run-root", default=str(fill.DEFAULT_REPLACEMENT_RUN_ROOT))
    parser.add_argument("--out", default=str(fill.LOCAL_ROOT / "05_proxy_local_entropy" / "raw_outputs" / "qc_filled_phi_dmax0p65_strict"))
    parser.add_argument("--existing-policy", choices=["strict4096"], default="strict4096")
    parser.add_argument("--max-target-units", type=int, default=80)
    parser.add_argument("--attempts-per-deficit", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)

    out_dir = ensure_dir(Path(args.out))
    run_root = Path(args.replacement_run_root)
    tasks = plan_tasks(
        replacement_run_root=run_root,
        existing_policy=args.existing_policy,
        max_target_units=int(args.max_target_units),
        attempts_per_deficit=float(args.attempts_per_deficit),
    )
    plan_path = out_dir / f"parallel_batch_plan_{int(time.time())}.csv"
    write_csv(tasks, plan_path)
    if tasks.empty:
        print(json.dumps({"planned_units": 0, "plan_path": str(plan_path)}, indent=2))
        return 2
    started = time.time()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=_init_worker,
        initargs=(str(run_root), int(args.cpu_threads_per_worker), str(args.device)),
    ) as executor:
        futures = [executor.submit(_sample_worker, row) for row in tasks.to_dict("records")]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[parallel] {row['kind']} rule={row['rule']} ref={row['ref_id']:03d} "
                f"r={row['radius']:.4f} pass={row['unit_qc_pass']} elapsed={row['elapsed_s']:.1f}s",
                flush=True,
            )
            write_csv(pd.DataFrame(rows), out_dir / "parallel_sampling_log_latest.csv")
    log_path = out_dir / f"parallel_sampling_log_{int(time.time())}.csv"
    write_csv(pd.DataFrame(rows), log_path)
    status = {
        "planned_units": int(len(tasks)),
        "completed_units": int(len(rows)),
        "qc_pass_units": int(sum(bool(row["unit_qc_pass"] and row["baseline4096"]) for row in rows)),
        "elapsed_s": float(time.time() - started),
        "plan_path": str(plan_path),
        "log_path": str(log_path),
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
