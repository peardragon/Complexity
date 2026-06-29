#!/usr/bin/env python3
"""Eta-specific exact-reference smoke for MNIST even/odd label flips."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

STAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STAGE_ROOT.parents[2]
PROJECT_ROOT = REPO_ROOT.parent
WINDOWS_ROOT = PROJECT_ROOT / "windows_project"
SOURCE_RUN_ROOT = (
    WINDOWS_ROOT
    / "02_dnn"
    / "08_mnist"
    / "runs"
    / "final"
    / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
)
BASE_DATASET = SOURCE_RUN_ROOT / "01_dataset_prepare" / "raw_datasets" / "split_000" / "real_even_odd" / "dataset.npz"
SRC_DIR = WINDOWS_ROOT / "02_dnn" / "08_mnist" / "src"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MNIST14_DEVICE", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(STAGE_ROOT / ".cache" / "matplotlib"))
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "TORCH_NUM_INTEROP_THREADS",
):
    os.environ.setdefault(_var, "8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mnist10_single_dataset_10x10_ntrain512_sparse_pipeline as p  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def eta_token(eta: float) -> str:
    return f"eta_{float(eta):.2f}".replace(".", "p")


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
    if pd.isna(obj):
        return None
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


def load_base_dataset() -> dict[str, np.ndarray]:
    payload = np.load(BASE_DATASET)
    return {key: payload[key] for key in payload.files}


def materialize_eta_dataset(base: dict[str, np.ndarray], eta: float, seed: int, out_path: Path) -> dict[str, Any]:
    if out_path.exists():
        payload = np.load(out_path)
        y_train = payload["y_train"].astype(np.int8, copy=False)
        y_test = payload["y_test"].astype(np.int8, copy=False)
        return {
            "dataset_path": str(out_path),
            "flip_rate_train": float(np.mean(y_train != base["y_train"])),
            "flip_rate_test": float(np.mean(y_test != base["y_test"])),
            "reused": True,
        }
    rng = np.random.default_rng(int(seed))
    flip_train = rng.random(base["y_train"].shape[0]) < float(eta)
    flip_test = rng.random(base["y_test"].shape[0]) < float(eta)
    arrays = {key: np.asarray(value) for key, value in base.items()}
    arrays["y_train"] = (base["y_train"].astype(np.int8) * np.where(flip_train, -1, 1).astype(np.int8)).astype(np.int8)
    arrays["y_test"] = (base["y_test"].astype(np.int8) * np.where(flip_test, -1, 1).astype(np.int8)).astype(np.int8)
    arrays["eta_flip_mask_train"] = flip_train.astype(np.bool_)
    arrays["eta_flip_mask_test"] = flip_test.astype(np.bool_)
    arrays["eta"] = np.asarray(float(eta), dtype=np.float32)
    arrays["eta_seed"] = np.asarray(int(seed), dtype=np.int64)
    ensure_dir(out_path.parent)
    tmp = out_path.with_name(f"{out_path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out_path)
    return {
        "dataset_path": str(out_path),
        "flip_rate_train": float(np.mean(flip_train)),
        "flip_rate_test": float(np.mean(flip_test)),
        "reused": False,
    }


def configure_torch_threads(cpu_threads: int) -> None:
    threads = max(1, min(8, int(cpu_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except Exception:
        pass


def train_eta_references(
    run_root: Path,
    etas: list[float],
    target_refs: int,
    max_attempts: int,
    batch_size: int,
    max_epochs: int,
    lr: float,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = load_base_dataset()
    dataset_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    out_dir = ensure_dir(run_root / "04_exact_reference_search")

    for eta_idx, eta in enumerate(etas):
        rule = eta_token(eta)
        dataset_path = run_root / "01_dataset_gen" / "split_000" / rule / "rep_000" / "dataset.npz"
        dataset_info = materialize_eta_dataset(base, eta, base_seed + 1000 * eta_idx + int(round(eta * 10000)), dataset_path)
        dataset_rows.append(
            {
                "split_id": 0,
                "rule": rule,
                "eta": float(eta),
                "n_train": int(base["X_train"].shape[0]),
                "n_test": int(base["X_test"].shape[0]),
                "train_pos_fraction": float(np.mean(np.load(dataset_path)["y_train"] == 1)),
                **dataset_info,
            }
        )

        ds = {key: np.load(dataset_path)[key] for key in np.load(dataset_path).files}
        selected: list[dict[str, Any]] = []
        attempts_used = 0
        seed_start = int(base_seed + 100000 * (eta_idx + 1) + int(round(eta * 10000)) * 10)
        print(f"[eta-ref-smoke] rule={rule} selected=0/{target_refs}", flush=True)
        while attempts_used < int(max_attempts) and len(selected) < int(target_refs):
            n_batch = min(int(batch_size), int(max_attempts) - attempts_used)
            seeds = [seed_start + attempts_used + offset for offset in range(n_batch)]
            batch_started = time.time()
            batch = p.smoke.train_attempt_batch(ds["X_train"], ds["y_train"], seeds, max_epochs=int(max_epochs), lr=float(lr))
            batch_elapsed = time.time() - batch_started
            attempts_used += n_batch
            for result in batch:
                theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
                ce_train, err_train = p.ce_and_error_np(theta, ds["X_train"], ds["y_train"])
                ce_test, err_test = p.ce_and_error_np(theta, ds["X_test"], ds["y_test"])
                selected_flag = False
                if err_train == 0.0 and theta.size == p.P:
                    candidate = {
                        "theta": theta,
                        "attempt_seed": int(result["seed"]),
                        "phase": str(result["phase"]),
                        "train_error": 0.0,
                    }
                    selected_flag = p.smoke.select_reference(selected, candidate)
                attempt_rows.append(
                    {
                        "split_id": 0,
                        "rule": rule,
                        "eta": float(eta),
                        "attempt_seed": int(result["seed"]),
                        "phase": str(result["phase"]),
                        "epoch": int(result["epoch"]),
                        "train_error": float(err_train),
                        "test_error": float(err_test),
                        "ce_mean_train": float(ce_train),
                        "ce_mean_test": float(ce_test),
                        "theta_norm": float(np.linalg.norm(theta)),
                        "selected": bool(selected_flag),
                        "batch_elapsed_s": float(batch_elapsed),
                    }
                )
            write_csv(out_dir / "attempt_logs" / "attempts.csv", pd.DataFrame(attempt_rows))
            print(
                f"[eta-ref-smoke] rule={rule} attempts={attempts_used} selected={len(selected)}/{target_refs}",
                flush=True,
            )

        for ref_id, result in enumerate(selected[: int(target_refs)]):
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ref_dir = ensure_dir(out_dir / "selected_reference_pool" / "split_000" / rule / f"ref_{ref_id:03d}")
            theta_path = ref_dir / "theta.npy"
            np.save(theta_path, theta)
            ce_train, err_train = p.ce_and_error_np(theta, ds["X_train"], ds["y_train"])
            ce_test, err_test = p.ce_and_error_np(theta, ds["X_test"], ds["y_test"])
            summary = {
                "dataset_id": eta_idx,
                "split_id": 0,
                "rule": rule,
                "eta": float(eta),
                "ref_id": int(ref_id),
                "theta_path": str(theta_path),
                "dataset_path": str(dataset_path),
                "attempt_seed": int(result["attempt_seed"]),
                "optimizer_chain": str(result["phase"]),
                "P": int(theta.size),
                "train_error": float(err_train),
                "test_error": float(err_test),
                "CE_mean_train": float(ce_train),
                "CE_sum_train": float(ce_train * ds["X_train"].shape[0]),
                "CE_mean_test": float(ce_test),
                "theta_norm": float(np.linalg.norm(theta)),
                **p.margin_stats_np(theta, ds["X_train"], ds["y_train"]),
            }
            write_json(ref_dir / "ref_summary.json", summary)
            reference_rows.append(summary)

    datasets = pd.DataFrame(dataset_rows)
    attempts = pd.DataFrame(attempt_rows)
    references = pd.DataFrame(reference_rows)
    write_csv(run_root / "01_dataset_gen" / "eta_dataset_manifest.csv", datasets)
    write_csv(out_dir / "reference_index.csv", references)
    return datasets, attempts, references


def plot_outputs(run_root: Path, attempts: pd.DataFrame, references: pd.DataFrame) -> None:
    fig_dir = ensure_dir(run_root / "04_exact_reference_search" / "figures")
    if not attempts.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=170)
        for rule, sub in attempts.groupby("rule", sort=True):
            sub = sub.reset_index(drop=True)
            ax.plot(np.arange(1, len(sub) + 1), sub["train_error"], marker="o", ms=3, lw=1.2, label=rule)
        ax.set_xlabel("attempt within eta")
        ax.set_ylabel("train error")
        ax.set_title("Eta reference-search smoke attempts")
        ax.grid(True, color="0.88", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig01_attempt_train_error_by_eta.png")
        plt.close(fig)

        success = attempts.groupby("rule", as_index=False).agg(success_rate=("selected", "mean"), attempts=("selected", "size"))
        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=170)
        ax.bar(success["rule"], success["success_rate"], color="#2451a6")
        ax.set_ylim(0, 1)
        ax.set_ylabel("selected exact-reference fraction")
        ax.tick_params(axis="x", rotation=15)
        ax.set_title("Eta reference selection rate")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig02_reference_success_rate.png")
        plt.close(fig)

    if not references.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=170)
        ax.scatter(references["eta"], references["theta_norm"], s=34, color="#00857a")
        ax.set_xlabel("eta")
        ax.set_ylabel("theta norm")
        ax.set_title("Selected eta-specific references")
        ax.grid(True, color="0.88", linewidth=0.7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig03_selected_reference_norm_by_eta.png")
        plt.close(fig)


def write_report(
    run_root: Path,
    status: dict[str, Any],
    datasets: pd.DataFrame,
    attempts: pd.DataFrame,
    references: pd.DataFrame,
) -> None:
    lines = [
        "# Eta-Specific Reference Search Smoke",
        "",
        f"- Status: `{status['status']}`",
        f"- References: `{status['reference_rows']}` / `{status['expected_reference_rows']}`",
        f"- Attempts: `{status['attempt_rows']}`",
        f"- Elapsed seconds: `{status['elapsed_s']:.3f}`",
        f"- CPU threads: `{status['resource_policy']['cpu_threads_per_process']}`",
        f"- Device: `{status['resource_policy']['device']}`",
        "",
        "This smoke trains eta-specific exact references for label-flipped MNIST even/odd datasets.",
        "It is the formal-reference-search counterpart to the earlier fixed-anchor phi smoke.",
        "",
        "Primary files:",
        "",
        "- `01_dataset_gen/eta_dataset_manifest.csv`",
        "- `04_exact_reference_search/attempt_logs/attempts.csv`",
        "- `04_exact_reference_search/reference_index.csv`",
        "- `04_exact_reference_search/selected_reference_pool/`",
    ]
    if not datasets.empty:
        lines.extend(["", "Datasets:", "", datasets.to_csv(index=False)])
    if not references.empty:
        lines.extend(["", "Selected references:", "", references[["rule", "eta", "ref_id", "CE_mean_train", "train_error", "theta_norm"]].to_csv(index=False)])
    (run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="eta_reference_search_smoke_cpu35_gpu0")
    parser.add_argument("--etas", default="0.35")
    parser.add_argument("--target-refs", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.022)
    parser.add_argument("--base-seed", type=int, default=2026062500)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    configure_torch_threads(int(args.cpu_threads))
    run_root = ensure_dir(STAGE_ROOT / "raw_outputs" / args.run_name)
    ensure_dir(STAGE_ROOT / ".cache" / "matplotlib")
    etas = parse_float_list(args.etas)
    config = {
        "run_name": args.run_name,
        "base_dataset": str(BASE_DATASET),
        "etas": etas,
        "target_refs": int(args.target_refs),
        "max_attempts": int(args.max_attempts),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "lr": float(args.lr),
        "base_seed": int(args.base_seed),
        "resource_policy": {
            "cpu_limit_target": "use <=35% by limiting thread count",
            "gpu_limit_target": "use <=25%; this run uses CPU only",
            "cpu_threads_per_process": max(1, min(8, int(args.cpu_threads))),
            "device": "cpu",
        },
    }
    write_json(run_root / "run_config_resolved.json", config)

    datasets, attempts, references = train_eta_references(
        run_root=run_root,
        etas=etas,
        target_refs=int(args.target_refs),
        max_attempts=int(args.max_attempts),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        lr=float(args.lr),
        base_seed=int(args.base_seed),
    )
    plot_outputs(run_root, attempts, references)
    expected = int(len(etas) * int(args.target_refs))
    elapsed = time.time() - started
    status = {
        "status": "complete" if len(references) >= expected else "partial",
        "etas": etas,
        "reference_rows": int(len(references)),
        "expected_reference_rows": expected,
        "attempt_rows": int(len(attempts)),
        "all_selected_exact": bool(len(references) >= expected and (references["train_error"] == 0.0).all()) if not references.empty else False,
        "theta_length_all_P": bool((references["P"] == int(p.P)).all()) if not references.empty else False,
        "elapsed_s": float(elapsed),
        "resource_policy": config["resource_policy"],
    }
    write_json(run_root / "REFERENCE_SEARCH_STATUS.json", status)
    write_json(run_root / "04_exact_reference_search" / "REFERENCE_SEARCH_STATUS.json", status)
    write_report(run_root, status, datasets, attempts, references)
    print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
