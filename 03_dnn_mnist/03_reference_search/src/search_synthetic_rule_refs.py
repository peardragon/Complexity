#!/usr/bin/env python3
"""Search exact references for a standalone synthetic MNIST10 rule."""

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
SRC_DIR = MNIST_ROOT / "src"
DATASET_RUN_ROOT = LOCAL_ROOT / "01_dataset_gen/raw_outputs/very_low_tv_spectral_teacher_v1"
DEFAULT_OUT = LOCAL_ROOT / "03_reference_search/raw_outputs/very_low_tv_spectral_teacher_v1"
RULE = "very_low_tv_spectral_teacher"


if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mnist10_single_dataset_10x10_ntrain512_sparse_pipeline as pipe  # noqa: E402


pipe.REPO_ROOT = WINDOWS_ROOT
pipe.smoke.REPO_ROOT = WINDOWS_ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def configure_resources(cpu_threads: int, device: str) -> None:
    threads = max(1, int(cpu_threads))
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MNIST14_DEVICE"] = "cpu"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except Exception:
        pass


def load_existing(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ref_path = out_dir / "reference_index.csv"
    attempt_path = out_dir / "attempt_logs/attempts.csv"
    selected: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    if ref_path.exists():
        ref_rows = pd.read_csv(ref_path).to_dict("records")
        for row in ref_rows:
            theta = np.load(Path(str(row["theta_path"]))).astype(np.float64).reshape(-1)
            selected.append(
                {
                    "theta": theta,
                    "attempt_seed": int(row["attempt_seed"]),
                    "phase": str(row["optimizer_chain"]),
                    "train_error": float(row["train_error"]),
                }
            )
    if attempt_path.exists():
        attempt_rows = pd.read_csv(attempt_path).to_dict("records")
    return selected, ref_rows, attempt_rows


def next_seed(attempt_rows: list[dict[str, Any]], seed_start: int) -> int:
    seen = [int(row["attempt_seed"]) for row in attempt_rows if str(row.get("rule", "")) == RULE]
    return max(int(seed_start), max(seen) + 1 if seen else int(seed_start))


def write_reference_row(
    *,
    out_dir: Path,
    ref_id: int,
    theta: np.ndarray,
    ds: dict[str, np.ndarray],
    dataset_path: Path,
    attempt_seed: int,
    phase: str,
) -> dict[str, Any]:
    ref_dir = ensure_dir(out_dir / "selected_reference_pool/split_000" / RULE / f"ref_{int(ref_id):03d}")
    theta_path = ref_dir / "theta.npy"
    np.save(theta_path, theta)
    ce_train, err_train = pipe.ce_and_error_np(theta, ds["X_train"], ds["y_train"])
    ce_test, err_test = pipe.ce_and_error_np(theta, ds["X_test"], ds["y_test"])
    row = {
        "dataset_id": 0,
        "split_id": 0,
        "rule": RULE,
        "ref_id": int(ref_id),
        "theta_path": str(theta_path),
        "dataset_path": str(dataset_path),
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
    }
    write_json(ref_dir / "ref_summary.json", row)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search exact refs for very_low_tv_spectral_teacher.")
    parser.add_argument("--dataset-run-root", default=str(DATASET_RUN_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--target-refs", type=int, default=90)
    parser.add_argument("--max-attempts", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=2840000)
    parser.add_argument("--max-epochs", type=int, default=4200)
    parser.add_argument("--lr", type=float, default=0.022)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)

    configure_resources(int(args.cpu_threads), str(args.device))
    out_root = Path(args.out_root)
    out_dir = ensure_dir(out_root / "04_exact_reference_search")
    dataset_index = pd.read_csv(Path(args.dataset_run_root) / "01_dataset_prepare/dataset_index.csv")
    dataset_row = dataset_index[dataset_index["rule"].eq(RULE)].iloc[0]
    dataset_path = Path(str(dataset_row["dataset_path"]))
    ds = pipe.load_dataset(dataset_path)

    selected, ref_rows, attempt_rows = load_existing(out_dir)
    next_ref_id = max([int(row["ref_id"]) for row in ref_rows], default=-1) + 1
    seed = next_seed(attempt_rows, int(args.seed_start))
    started = time.time()
    attempts_used_this_run = 0

    print(f"[synthetic refs] existing={len(selected)} target={int(args.target_refs)} dataset={dataset_path}", flush=True)
    while len(selected) < int(args.target_refs) and attempts_used_this_run < int(args.max_attempts):
        batch_n = min(int(args.batch_size), int(args.max_attempts) - attempts_used_this_run)
        seeds = list(range(seed, seed + batch_n))
        print(f"[synthetic refs] seeds={seeds[0]}..{seeds[-1]} selected={len(selected)}/{int(args.target_refs)}", flush=True)
        batch = pipe.smoke.train_attempt_batch(ds["X_train"], ds["y_train"], seeds, max_epochs=int(args.max_epochs), lr=float(args.lr))
        seed += batch_n
        attempts_used_this_run += batch_n
        for result in batch:
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ce_train, err_train = pipe.ce_and_error_np(theta, ds["X_train"], ds["y_train"])
            ce_test, err_test = pipe.ce_and_error_np(theta, ds["X_test"], ds["y_test"])
            accepted = False
            if err_train == 0.0 and theta.size == pipe.P:
                before = len(selected)
                accepted = pipe.smoke.select_reference(
                    selected,
                    {
                        "theta": theta,
                        "attempt_seed": int(result["seed"]),
                        "phase": str(result["phase"]),
                        "train_error": 0.0,
                    },
                )
                accepted = bool(accepted and len(selected) > before)
            attempt_rows.append(
                {
                    "dataset_id": 0,
                    "split_id": 0,
                    "rule": RULE,
                    "attempt_seed": int(result["seed"]),
                    "phase": str(result["phase"]),
                    "epoch": int(result["epoch"]),
                    "train_error": float(err_train),
                    "test_error": float(err_test),
                    "ce_mean_train": float(ce_train),
                    "ce_mean_test": float(ce_test),
                    "theta_norm": float(np.linalg.norm(theta)),
                    "selected": bool(accepted),
                }
            )
            if accepted:
                row = write_reference_row(
                    out_dir=out_dir,
                    ref_id=next_ref_id,
                    theta=theta,
                    ds=ds,
                    dataset_path=dataset_path,
                    attempt_seed=int(result["seed"]),
                    phase=str(result["phase"]),
                )
                ref_rows.append(row)
                print(f"[synthetic refs] accepted ref={next_ref_id:03d} seed={int(result['seed'])}", flush=True)
                next_ref_id += 1
                if len(selected) >= int(args.target_refs):
                    break
        write_csv(out_dir / "attempt_logs/attempts.csv", pd.DataFrame(attempt_rows))
        write_csv(out_dir / "reference_index.csv", pd.DataFrame(ref_rows))

    status = {
        "status": "complete" if len(ref_rows) >= int(args.target_refs) else "partial",
        "rule": RULE,
        "target_refs": int(args.target_refs),
        "reference_rows": int(len(ref_rows)),
        "attempts_this_run": int(attempts_used_this_run),
        "attempt_rows_total": int(len(attempt_rows)),
        "elapsed_s": float(time.time() - started),
        "reference_index": str(out_dir / "reference_index.csv"),
    }
    write_json(out_dir / "STATUS.json", status)
    (out_root / "REPORT.md").write_text(
        f"# Synthetic Rule Reference Search\n\n- Rule: `{RULE}`\n- Status: `{status['status']}`\n- References: `{len(ref_rows)}` / `{int(args.target_refs)}`\n- Reference index: `{out_dir / 'reference_index.csv'}`\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
