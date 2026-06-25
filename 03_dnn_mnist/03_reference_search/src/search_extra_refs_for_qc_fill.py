#!/usr/bin/env python3
"""Search extra exact references for strict-4096 QC filling.

This is a rule-general version of the one-off random_label helper.  It appends
new exact references to the replacement run's extra reference index so
fill_qc_passed_phi.py can treat them as non-selected candidate refs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
WINDOWS_ROOT = Path("/home/bjyong/Complexity/windows_project")
MNIST_ROOT = WINDOWS_ROOT / "02_dnn" / "08_mnist"
LOCAL_SRC_CACHE = LOCAL_ROOT / ".src_cache_mnist"
SRC_DIR = LOCAL_SRC_CACHE if LOCAL_SRC_CACHE.exists() else MNIST_ROOT / "src"
REFERENCE_RUN_ROOT = (
    MNIST_ROOT
    / "runs"
    / "final"
    / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
)
REFERENCE_INDEX = REFERENCE_RUN_ROOT / "04_exact_reference_search" / "reference_index.csv"
DEFAULT_REPLACEMENT_RUN_ROOT = LOCAL_ROOT / "03_reference_search" / "raw_outputs" / "extra_reference_pool"
EXTRA_REFERENCE_INDEX_NAME = "extra_reference_index.csv"
RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mnist10_single_dataset_10x10_ntrain512_sparse_pipeline as pipe  # noqa: E402

pipe.REPO_ROOT = WINDOWS_ROOT
pipe.smoke.REPO_ROOT = WINDOWS_ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def parse_rules(text: str) -> list[str]:
    if text.strip().lower() in {"", "all"}:
        return list(RULES)
    requested = [item.strip() for item in text.split(",") if item.strip()]
    missing = [rule for rule in requested if rule not in RULES]
    if missing:
        raise ValueError(f"Unknown rules: {missing}")
    return requested


def load_extra_rows(run_root: Path) -> pd.DataFrame:
    extra_path = run_root / "04_extra_reference_search" / EXTRA_REFERENCE_INDEX_NAME
    if extra_path.exists():
        return pd.read_csv(extra_path)
    return pd.DataFrame()


def load_existing_reference_rows(run_root: Path, rule: str) -> pd.DataFrame:
    base = pd.read_csv(REFERENCE_INDEX)
    base = base[base["rule"].eq(rule)].copy()
    extra = load_extra_rows(run_root)
    if not extra.empty:
        base = pd.concat([base, extra[extra["rule"].eq(rule)]], ignore_index=True, sort=False)
    base["ref_id"] = base["ref_id"].astype(int)
    return base.sort_values("ref_id").reset_index(drop=True)


def load_theta(row: pd.Series) -> np.ndarray:
    path = Path(str(row["theta_path"]))
    if not path.is_absolute():
        path = WINDOWS_ROOT / path
    return np.load(path).astype(np.float64).reshape(-1)


def next_ref_id_for(existing: pd.DataFrame) -> int:
    if existing.empty:
        return 1000
    return int(max(1000, int(existing["ref_id"].max()) + 1))


def next_seed_for(rule: str, attempt_rows: list[dict[str, Any]], seed_start: int) -> int:
    rule_offset = RULES.index(rule) * 1_000_000
    seed = int(seed_start) + rule_offset
    seen = [int(row["attempt_seed"]) for row in attempt_rows if str(row.get("rule", "")) == rule]
    if seen:
        seed = max(seed, max(seen) + 1)
    return seed


def reference_summary(
    *,
    out_dir: Path,
    rule: str,
    ref_id: int,
    theta: np.ndarray,
    dataset_row: pd.Series,
    ds: dict[str, np.ndarray],
    attempt_seed: int,
    phase: str,
) -> dict[str, Any]:
    ref_dir = ensure_dir(out_dir / "selected_reference_pool" / "split_000" / rule / f"ref_{ref_id:03d}")
    theta_path = ref_dir / "theta.npy"
    np.save(theta_path, theta)
    ce_train, err_train = pipe.ce_and_error_np(theta, ds["X_train"], ds["y_train"])
    ce_test, err_test = pipe.ce_and_error_np(theta, ds["X_test"], ds["y_test"])
    summary = {
        "dataset_id": int(dataset_row.get("dataset_id", 0)),
        "split_id": int(dataset_row.get("split_id", 0)),
        "rule": rule,
        "ref_id": int(ref_id),
        "theta_path": str(theta_path),
        "dataset_path": str(dataset_row["dataset_path"]),
        "attempt_seed": int(attempt_seed),
        "optimizer_chain": str(phase),
        "P": int(theta.size),
        "train_error": float(err_train),
        "test_error": float(err_test),
        "CE_mean_train": float(ce_train),
        "CE_sum_train": float(ce_train * ds["X_train"].shape[0]),
        "CE_mean_test": float(ce_test),
        "theta_norm": float(np.linalg.norm(theta)),
        **pipe.margin_stats_np(theta, ds["X_train"], ds["y_train"]),
        "extra_reference_search": True,
    }
    write_json(ref_dir / "ref_summary.json", summary)
    return summary


def search_rule(
    *,
    rule: str,
    run_root: Path,
    out_dir: Path,
    target_new: int,
    max_attempts: int,
    batch_size: int,
    seed_start: int,
    max_epochs: int,
    lr: float,
    extra_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    existing = load_existing_reference_rows(run_root, rule)
    if existing.empty:
        raise RuntimeError(f"No existing reference rows found for {rule}")
    dataset_row = existing.iloc[0]
    ds = pipe.load_dataset(dataset_row["dataset_path"])
    selected = [
        {
            "theta": load_theta(row),
            "attempt_seed": int(row.get("attempt_seed", -1)),
            "phase": str(row.get("optimizer_chain", "existing")),
            "train_error": 0.0,
        }
        for _, row in existing.iterrows()
    ]
    next_ref_id = next_ref_id_for(existing)
    seed = next_seed_for(rule, attempt_rows, seed_start)
    started = time.time()
    attempts_used = 0
    new_this_rule = 0
    while attempts_used < max_attempts and new_this_rule < target_new:
        batch_n = min(batch_size, max_attempts - attempts_used)
        seeds = list(range(seed, seed + batch_n))
        print(
            f"[extra refs] rule={rule} seeds={seeds[0]}..{seeds[-1]} "
            f"new={new_this_rule}/{target_new}",
            flush=True,
        )
        batch = pipe.smoke.train_attempt_batch(
            ds["X_train"],
            ds["y_train"],
            seeds,
            max_epochs=max_epochs,
            lr=lr,
        )
        attempts_used += batch_n
        seed += batch_n
        for result in batch:
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ce_train, err_train = pipe.ce_and_error_np(theta, ds["X_train"], ds["y_train"])
            ce_test, err_test = pipe.ce_and_error_np(theta, ds["X_test"], ds["y_test"])
            selected_before = len(selected)
            accepted = False
            if err_train == 0.0 and theta.size == pipe.P:
                accepted = pipe.smoke.select_reference(
                    selected,
                    {
                        "theta": theta,
                        "attempt_seed": int(result["seed"]),
                        "phase": str(result["phase"]),
                        "train_error": 0.0,
                    },
                )
            attempt_rows.append(
                {
                    "rule": rule,
                    "attempt_seed": int(result["seed"]),
                    "phase": str(result["phase"]),
                    "epoch": int(result["epoch"]),
                    "train_error": float(err_train),
                    "test_error": float(err_test),
                    "ce_mean_train": float(ce_train),
                    "ce_mean_test": float(ce_test),
                    "theta_norm": float(np.linalg.norm(theta)),
                    "selected": bool(accepted and len(selected) > selected_before),
                }
            )
            if accepted and len(selected) > selected_before:
                row = reference_summary(
                    out_dir=out_dir,
                    rule=rule,
                    ref_id=next_ref_id,
                    theta=theta,
                    dataset_row=dataset_row,
                    ds=ds,
                    attempt_seed=int(result["seed"]),
                    phase=str(result["phase"]),
                )
                extra_rows.append(row)
                next_ref_id += 1
                new_this_rule += 1
                print(f"[extra refs] accepted rule={rule} ref={int(row['ref_id']):03d} seed={int(result['seed'])}", flush=True)
                if new_this_rule >= target_new:
                    break
    return {
        "rule": rule,
        "new_refs_this_run": int(new_this_rule),
        "attempts_this_rule": int(attempts_used),
        "elapsed_s": float(time.time() - started),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search extra exact references for strict-4096 QC filling.")
    parser.add_argument("--replacement-run-root", default=str(DEFAULT_REPLACEMENT_RUN_ROOT))
    parser.add_argument("--rules", default="all", help="Comma-separated rules, or all.")
    parser.add_argument("--target-new-per-rule", type=int, default=20)
    parser.add_argument("--max-attempts-per-rule", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=2703600)
    parser.add_argument("--max-epochs", type=int, default=4200)
    parser.add_argument("--lr", type=float, default=0.022)
    args = parser.parse_args(argv)

    run_root = Path(args.replacement_run_root)
    out_dir = ensure_dir(run_root / "04_extra_reference_search")
    rules = parse_rules(args.rules)
    extra_path = out_dir / EXTRA_REFERENCE_INDEX_NAME
    extra_rows: list[dict[str, Any]] = []
    if extra_path.exists():
        extra_rows = pd.read_csv(extra_path).to_dict("records")
    attempt_path = out_dir / "attempt_logs" / "attempts.csv"
    attempt_rows: list[dict[str, Any]] = []
    if attempt_path.exists():
        attempt_rows = pd.read_csv(attempt_path).to_dict("records")

    started = time.time()
    rule_status = []
    for rule in rules:
        status = search_rule(
            rule=rule,
            run_root=run_root,
            out_dir=out_dir,
            target_new=int(args.target_new_per_rule),
            max_attempts=int(args.max_attempts_per_rule),
            batch_size=int(args.batch_size),
            seed_start=int(args.seed_start),
            max_epochs=int(args.max_epochs),
            lr=float(args.lr),
            extra_rows=extra_rows,
            attempt_rows=attempt_rows,
        )
        rule_status.append(status)
        write_csv(pd.DataFrame(attempt_rows), attempt_path)
        write_csv(pd.DataFrame(extra_rows), extra_path)

    status = {
        "rules": rules,
        "target_new_per_rule": int(args.target_new_per_rule),
        "new_refs_this_run_total": int(sum(row["new_refs_this_run"] for row in rule_status)),
        "extra_reference_rows_total": int(len(extra_rows)),
        "attempts_this_run_total": int(sum(row["attempts_this_rule"] for row in rule_status)),
        "elapsed_s": float(time.time() - started),
        "rule_status": rule_status,
        "extra_reference_index": str(extra_path),
    }
    write_json(out_dir / "STATUS.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["new_refs_this_run_total"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
