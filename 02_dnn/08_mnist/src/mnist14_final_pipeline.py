from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.datasets import fetch_openml
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[2]
RUN_ROOT = ROOT / "runs" / "final"
CONFIG_PATH = ROOT / "templates" / "config_final.yaml"
RULES = ["real_even_odd", "teacher_nn", "random_label"]
_DATASET_CACHE: dict[str, dict[str, np.ndarray]] = {}
_THETA_CACHE: dict[str, np.ndarray] = {}
STAGES = [
    "01_dataset_prepare",
    "02_complexity_measure",
    "03_pool_design",
    "04_exact_reference_search",
    "05_pool2_pm_sais_sampling",
    "06_results_figures",
]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mnist14_smoke_pipeline as smoke
from mnist14_model import ARCH, P, ce_and_error_np, ce_error_batch_torch, margin_stats_np, normalize_labels
from mnist14_vmf import log_sphere_mgf, sample_vmf, sample_vmf_batch


class FinalBlocked(RuntimeError):
    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        observed: dict[str, Any] | None = None,
        expected: dict[str, Any] | None = None,
        next_action: str = "Inspect the blocked report, repair the cause, and rerun the same final stage.",
    ) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.observed = observed or {}
        self.expected = expected or {}
        self.next_action = next_action


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_dir(stage: str) -> Path:
    return RUN_ROOT / ("final_report" if stage == "06_results_figures" else stage)


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        checkpoint = path.with_name(f"{path.stem}_checkpoint_{int(time.time())}{path.suffix}")
        shutil.copyfile(tmp, checkpoint)
        try:
            if not path.exists():
                shutil.copyfile(tmp, path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def files_under(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [rel(p) for p in sorted(path.rglob("*")) if p.is_file()]


def dense_radii() -> list[float]:
    return [float(f"{x:.2f}") for x in np.round(np.arange(0.01, 2.50 + 0.0001, 0.01), 2)]


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "final"
    cfg["identity"] = "mnist14_3rule_1024_10split_20ref_dense_0p01_to_2p50"
    cfg["python"] = sys.executable
    cfg["repo_root"] = str(REPO_ROOT)
    cfg["mnist_root"] = str(ROOT)
    cfg["resolved_at_unix"] = time.time()
    cfg["sampling"]["r0"] = 0.01
    cfg["sampling"]["radii"] = dense_radii()
    cfg["sampling"]["samples_per_ref_radius"] = {f"{r:.2f}": 1024 for r in cfg["sampling"]["radii"]}
    cfg["sampling"]["proposal"] = "exact_shell_l2_vmf_adaptive_ce_tempered_smc"
    cfg["smc"] = {
        "target_cess_fraction": 0.75,
        "resample_ess_fraction": 0.75,
        "max_steps": 120,
        "min_delta_t": 1.0e-4,
        "bisection_steps": 32,
        "mh_sweeps": 1,
        "move_kappa_factor": 80.0,
    }
    cfg["compute"] = {
        "chunk_size": int(os.environ.get("MNIST14_CHUNK_SIZE", "1024")),
        "device": os.environ.get("MNIST14_DEVICE", "auto"),
        "dtype": os.environ.get("MNIST14_DTYPE", "float32"),
    }
    cfg["outputs"] = {
        "run_root": rel(RUN_ROOT),
        "range_label": "d_0.01_to_2.50_dense",
        "summary_only_pool2": True,
    }
    return cfg


def write_qc(stage: str, status: str, checks: dict[str, Any], *, warnings: list[str] | None = None, hard_failures: list[str] | None = None) -> None:
    out_dir = stage_dir(stage)
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": stage,
            "status": status,
            "checks": checks,
            "warnings": warnings or [],
            "hard_failures": hard_failures or [],
            "files": files_under(out_dir),
        },
    )


def write_blocked(blocked: FinalBlocked) -> None:
    out_dir = stage_dir(blocked.stage)
    observed = "\n".join(f"- {k}: {v}" for k, v in blocked.observed.items()) or "- n/a"
    expected = "\n".join(f"- {k}: {v}" for k, v in blocked.expected.items()) or "- n/a"
    files = "\n".join(f"- {p}" for p in files_under(out_dir)) or "- none"
    write_text(
        out_dir / "STAGE_BLOCKED.md",
        f"""# STAGE_BLOCKED

Stage: `{blocked.stage}`

## Exact Failing Condition

{blocked.reason}

## Observed Metric

{observed}

## Expected Threshold

{expected}

## Files Already Created

{files}

## Next Safe Action

{blocked.next_action}
""",
    )
    write_qc(blocked.stage, "blocked", {"blocked": True, "reason": blocked.reason}, hard_failures=[blocked.reason])


def run_pytest(test_path: Path, *, timeout_s: int = 600) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", str(test_path), "-q"]
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout_s)
    return {
        "cmd": " ".join(cmd),
        "returncode": int(proc.returncode),
        "elapsed_s": float(time.time() - started),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def load_mnist_final() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data_dir = ensure_dir(ROOT / "data" / "mnist")
    cache_path = data_dir / "mnist_openml_uint8.npz"
    if cache_path.exists():
        payload = np.load(cache_path)
        return payload["X"], payload["y"], {"source": "local_npz_cache", "cache_path": rel(cache_path), "download_performed": False}
    arff_cache = data_dir / "openml" / "openml" / "openml.org" / "data" / "v1" / "download" / "52667" / "mnist_784.arff.gz"
    if not arff_cache.exists():
        raise FinalBlocked(
            "01_dataset_prepare",
            "Final dataset stage cannot silently download MNIST.",
            observed={"npz_cache_exists": False, "openml_arff_cache_exists": False},
            expected={"local_mnist_cache": "present"},
            next_action="Run the smoke dataset stage to cache MNIST, or place MNIST locally, then rerun final Stage 01.",
        )
    fetched = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto", data_home=str(data_dir / "openml"))
    x = np.asarray(fetched.data, dtype=np.uint8).reshape(-1, 784)
    y = np.asarray(fetched.target, dtype=np.int16).reshape(-1)
    np.savez_compressed(cache_path, X=x, y=y)
    return x, y, {"source": "materialized_from_existing_openml_cache", "cache_path": rel(cache_path), "download_performed": False}


def stage01_dataset_prepare(*, force: bool = False) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("01_dataset_prepare"))
    index_path = out_dir / "dataset_index.csv"
    expected_rows = int(cfg["dataset"]["n_splits"]) * len(RULES)
    if index_path.exists() and not force:
        existing = pd.read_csv(index_path)
        if len(existing) == expected_rows:
            write_qc("01_dataset_prepare", "pass", {"reused": True, "dataset_index_rows": int(len(existing))})
            return

    raw28, digits, source_meta = load_mnist_final()
    n_train = int(cfg["dataset"]["n_train"])
    n_test = int(cfg["dataset"]["n_test"])
    n_splits = int(cfg["dataset"]["n_splits"])
    even_idx = np.flatnonzero((digits % 2) == 0)
    odd_idx = np.flatnonzero((digits % 2) == 1)
    dataset_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_id in range(n_splits):
        rng = np.random.default_rng(20260610 + split_id)
        even_perm = rng.permutation(even_idx)
        odd_perm = rng.permutation(odd_idx)
        train_idx = np.concatenate([even_perm[: n_train // 2], odd_perm[: n_train // 2]])
        test_idx = np.concatenate([even_perm[n_train // 2 : n_train // 2 + n_test // 2], odd_perm[n_train // 2 : n_train // 2 + n_test // 2]])
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        x_train_raw = smoke.avgpool_14(raw28[train_idx])
        x_test_raw = smoke.avgpool_14(raw28[test_idx])
        mean = x_train_raw.mean(axis=0, keepdims=True)
        std = x_train_raw.std(axis=0, keepdims=True)
        std = np.where(std < 1.0e-6, 1.0, std)
        x_train = ((x_train_raw - mean) / std).astype(np.float32)
        x_test = ((x_test_raw - mean) / std).astype(np.float32)
        digit_train = digits[train_idx].astype(np.int16)
        digit_test = digits[test_idx].astype(np.int16)
        teacher_train = smoke.teacher_logits(x_train, 31001 + split_id)
        teacher_test = smoke.teacher_logits(x_test, 31001 + split_id)
        threshold = float(np.median(teacher_train))
        split_rows.append(
            {
                "split_id": split_id,
                "n_train": n_train,
                "n_test": n_test,
                "train_even_fraction": float(np.mean((digit_train % 2) == 0)),
                "test_even_fraction": float(np.mean((digit_test % 2) == 0)),
                "standardization_mean_mean": float(np.mean(mean)),
                "standardization_std_mean": float(np.mean(std)),
            }
        )
        labels = {
            "real_even_odd": (np.where((digit_train % 2) == 0, 1, -1).astype(np.int8), np.where((digit_test % 2) == 0, 1, -1).astype(np.int8), {"definition": "even digit +1, odd digit -1"}),
            "teacher_nn": (np.where(teacher_train >= threshold, 1, -1).astype(np.int8), np.where(teacher_test >= threshold, 1, -1).astype(np.int8), {"teacher_seed": 31001 + split_id, "train_median_logit_threshold": threshold}),
            "random_label": (smoke.balanced_pm1(n_train, 41001 + split_id), smoke.balanced_pm1(n_test, 42001 + split_id), {"train_seed": 41001 + split_id, "test_seed": 42001 + split_id, "test_accuracy_note": "independent random test labels are not a generalization metric"}),
        }
        for rule, (y_train, y_test, metadata) in labels.items():
            ds_dir = ensure_dir(out_dir / "raw_datasets" / f"split_{split_id:03d}" / rule)
            dataset_path = ds_dir / "dataset.npz"
            np.savez_compressed(
                dataset_path,
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                X_train_raw14=x_train_raw.astype(np.float32),
                X_test_raw14=x_test_raw.astype(np.float32),
                digit_train=digit_train,
                digit_test=digit_test,
                train_indices=train_idx.astype(np.int64),
                test_indices=test_idx.astype(np.int64),
                standardization_mean=mean.astype(np.float32),
                standardization_std=std.astype(np.float32),
            )
            write_json(ds_dir / "dataset_metadata.json", {"rule": rule, "split_id": split_id, **metadata})
            pos_train = float(np.mean(y_train == 1))
            label_rows.append(
                {
                    "split_id": split_id,
                    "rule": rule,
                    "train_pos_fraction": pos_train,
                    "test_pos_fraction": float(np.mean(y_test == 1)),
                    "train_n_pos": int(np.sum(y_train == 1)),
                    "train_n_neg": int(np.sum(y_train == -1)),
                    "test_n_pos": int(np.sum(y_test == 1)),
                    "test_n_neg": int(np.sum(y_test == -1)),
                    **{k: v for k, v in metadata.items() if isinstance(v, (int, float, str, bool))},
                }
            )
            dataset_rows.append(
                {
                    "experiment_id": cfg["experiment_id"],
                    "mode": "final",
                    "split_id": split_id,
                    "rule": rule,
                    "dataset_path": rel(dataset_path),
                    "n_train": n_train,
                    "n_test": n_test,
                    "input_dim": 196,
                    "train_pos_fraction": pos_train,
                }
            )

    dataset_df = pd.DataFrame(dataset_rows)
    label_df = pd.DataFrame(label_rows)
    meta_dir = ensure_dir(out_dir / "metadata")
    write_csv(index_path, dataset_df)
    write_csv(meta_dir / "split_summary.csv", pd.DataFrame(split_rows))
    write_csv(meta_dir / "label_balance_summary.csv", label_df)
    write_json(meta_dir / "mnist_source.json", source_meta)
    smoke.plot_dataset_figures(out_dir, raw28, [{"train_indices": np.load(REPO_ROOT / dataset_rows[0]["dataset_path"])["train_indices"].tolist()}], label_df)
    write_json(out_dir / "run_config_resolved.json", cfg)
    checks = {
        "dataset_index_rows": int(len(dataset_df)),
        "expected_rows": expected_rows,
        "n_train": n_train,
        "n_test": n_test,
        "balance_min": float(label_df["train_pos_fraction"].min()),
        "balance_max": float(label_df["train_pos_fraction"].max()),
        "download_performed": bool(source_meta["download_performed"]),
        "montage_exists": bool((out_dir / "figures" / "fig01_mnist_28_vs_14_montage.png").exists()),
    }
    if checks["dataset_index_rows"] != expected_rows or checks["balance_min"] < 0.45 or checks["balance_max"] > 0.55:
        raise FinalBlocked("01_dataset_prepare", "Final dataset QC failed.", observed=checks, expected={"rows": expected_rows, "balance": "0.45..0.55"})
    write_qc("01_dataset_prepare", "pass", checks)
    write_text(
        out_dir / "REPORT.md",
        f"""# Final Stage 01 Dataset Prepare

Prepared {expected_rows} final datasets: {n_splits} splits x 3 label rules, n_train={n_train}, n_test={n_test}.

MNIST source: {source_meta['source']}; download_performed={source_meta['download_performed']}.
""",
    )


def load_dataset(path_value: str | Path) -> dict[str, np.ndarray]:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    key = str(path.resolve())
    if key not in _DATASET_CACHE:
        payload = np.load(path)
        _DATASET_CACHE[key] = {k: payload[k] for k in payload.files}
    return _DATASET_CACHE[key]


def stage02_complexity_measure(*, force: bool = False) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("02_complexity_measure"))
    if (out_dir / "complexity_by_dataset.csv").exists() and not force:
        write_qc("02_complexity_measure", "pass", {"reused": True})
        return
    index_path = stage_dir("01_dataset_prepare") / "dataset_index.csv"
    if not index_path.exists():
        raise FinalBlocked("02_complexity_measure", "Final dataset index is missing.", observed={"missing": rel(index_path)})
    index_df = pd.read_csv(index_path)
    graph_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for row in index_df.to_dict("records"):
        ds = load_dataset(row["dataset_path"])
        metrics = []
        for k in [8, 16, 32]:
            m = smoke.graph_tv_nmstv(ds["X_train"], ds["y_train"], k)
            metrics.append(m)
            graph_rows.append({**row, **m})
        dataset_rows.append(
            {
                **row,
                "tv_mean": float(np.mean([m["tv"] for m in metrics])),
                "nmstv_mean": float(np.mean([m["nmstv"] for m in metrics])),
                "nmstv_min": float(np.min([m["nmstv"] for m in metrics])),
                "nmstv_max": float(np.max([m["nmstv"] for m in metrics])),
                "edge_count_min": int(np.min([m["edge_count"] for m in metrics])),
            }
        )
    graph_df = pd.DataFrame(graph_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    summary_df = dataset_df.groupby("rule", as_index=False).agg(
        nmstv_mean=("nmstv_mean", "mean"),
        nmstv_sd=("nmstv_mean", "std"),
        tv_mean=("tv_mean", "mean"),
        n_datasets=("rule", "size"),
    )
    write_csv(out_dir / "complexity_by_dataset.csv", dataset_df)
    write_csv(out_dir / "complexity_by_rule_summary.csv", summary_df)
    write_csv(out_dir / "graph_stats_by_dataset_k.csv", graph_df)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(6, 4))
    dataset_df.boxplot(column="nmstv_mean", by="rule", ax=ax)
    ax.set_title("")
    fig.suptitle("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("Mean NMSTV")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_nmstv_by_rule_boxplot.png", dpi=160)
    plt.close(fig)
    write_json(out_dir / "run_config_resolved.json", cfg)
    checks = {
        "dataset_rows": int(len(dataset_df)),
        "graph_rows": int(len(graph_df)),
        "rule_summary_rows": int(len(summary_df)),
        "edge_count_min": int(graph_df["edge_count"].min()),
        "all_finite": bool(np.isfinite(graph_df[["tv", "nmstv", "sigma_k"]].to_numpy()).all()),
    }
    if checks["dataset_rows"] != int(cfg["dataset"]["n_splits"]) * len(RULES) or checks["edge_count_min"] <= 0 or not checks["all_finite"]:
        raise FinalBlocked("02_complexity_measure", "Final complexity QC failed.", observed=checks)
    write_qc("02_complexity_measure", "pass", checks)
    write_text(out_dir / "REPORT.md", "# Final Stage 02 Complexity\n\nComputed finite kNN TV/NMSTV summaries for all final datasets.")


def stage03_pool_design() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("03_pool_design"))
    contract = {
        "experiment_id": cfg["experiment_id"],
        "identity": cfg["identity"],
        "architecture": {"name": "196-16-16-1-tanh", "P": P, "input_dim": ARCH.input_dim, "hidden_width": ARCH.hidden_width, "activation": ARCH.activation},
        "pool1": {
            "law": "optimizer-induced exact reference ensemble",
            "acceptance": "train_error == 0",
            "selected_refs_per_dataset": int(cfg["reference_search"]["selected_refs_per_dataset"]),
            "max_attempts_per_dataset": int(cfg["reference_search"]["max_attempts_per_dataset"]),
            "caveat": "not exact P_ref^0 samples",
        },
        "pool2": {
            "range_label": "d_0.01_to_2.50_dense",
            "radii": cfg["sampling"]["radii"],
            "samples_per_ref_radius": 1024,
            "proposal": cfg["sampling"]["proposal"],
            "summary_only": True,
        },
        "smc": cfg["smc"],
        "qc": cfg["qc"],
    }
    write_json(out_dir / "POOL_CONTRACT.json", contract)
    write_text(out_dir / "POOL_CONTRACT.md", "# Final Pool Contract\n\nFinal uses optimizer-induced exact references and adaptive CE-tempered PM-SAIS over d=0.01..2.50 step 0.01.")
    write_text(out_dir / "MODEL_SPEC.md", f"# Model Spec\n\nArchitecture `196-16-16-1-tanh`, P={P}.")
    write_text(out_dir / "QC_GATES.md", "# QC Gates\n\nFinal gates: q05 ESS >= 0.04, split logZ/P <= 0.004, bootstrap sd phi <= 0.012; failed radii are no_claim.")
    write_json(out_dir / "run_config_resolved.json", cfg)
    pytest_result = run_pytest(ROOT / "tests" / "test_stage03_model_spec.py")
    checks = {"P": P, "P_expected": 3441, "pytest": pytest_result, "dense_radius_count": len(cfg["sampling"]["radii"])}
    if P != 3441 or not pytest_result["passed"]:
        raise FinalBlocked("03_pool_design", "Final model/pool contract QC failed.", observed=checks)
    write_qc("03_pool_design", "pass", checks)
    write_text(out_dir / "REPORT.md", "# Final Stage 03 Pool Design\n\nFinal dense pool contract frozen.")


def existing_reference_index() -> pd.DataFrame:
    path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def stage04_exact_reference_search(*, force: bool = False) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("04_exact_reference_search"))
    index_path = stage_dir("01_dataset_prepare") / "dataset_index.csv"
    if not index_path.exists():
        raise FinalBlocked("04_exact_reference_search", "Final dataset index missing.", observed={"missing": rel(index_path)})
    dataset_index = pd.read_csv(index_path)
    target_refs = int(cfg["reference_search"]["selected_refs_per_dataset"])
    max_attempts = int(cfg["reference_search"]["max_attempts_per_dataset"])
    current = existing_reference_index()
    reference_rows = [] if force or current.empty else current.to_dict("records")
    attempt_log_path = ensure_dir(out_dir / "attempt_logs") / "attempts.csv"
    attempt_rows: list[dict[str, Any]] = []
    if attempt_log_path.exists() and not force:
        attempt_rows = pd.read_csv(attempt_log_path).to_dict("records")
    started = time.time()

    for dataset_id, row in enumerate(dataset_index.to_dict("records")):
        existing = [r for r in reference_rows if int(r["split_id"]) == int(row["split_id"]) and str(r["rule"]) == str(row["rule"]) and Path(REPO_ROOT / str(r["theta_path"])).exists()]
        if len(existing) >= target_refs:
            print(f"[final stage04] reuse split={row['split_id']} rule={row['rule']} refs={len(existing)}", flush=True)
            continue
        ds = load_dataset(row["dataset_path"])
        selected = []
        for old in existing:
            selected.append({"theta": np.load(REPO_ROOT / old["theta_path"]).astype(np.float64), "attempt_seed": int(old["attempt_seed"]), "phase": str(old["optimizer_chain"])})
        seed_base = 1700000 + 10000 * int(row["split_id"]) + 1000 * RULES.index(str(row["rule"]))
        attempts_used = len([r for r in attempt_rows if int(r.get("split_id", -1)) == int(row["split_id"]) and str(r.get("rule", "")) == str(row["rule"])])
        print(f"[final stage04] split={row['split_id']} rule={row['rule']} start selected={len(selected)}/{target_refs}", flush=True)
        while attempts_used < max_attempts and len(selected) < target_refs:
            batch_n = min(12, max_attempts - attempts_used)
            seeds = [seed_base + attempts_used + i for i in range(batch_n)]
            batch = smoke.train_attempt_batch(ds["X_train"], ds["y_train"], seeds, max_epochs=5000, lr=0.02)
            attempts_used += batch_n
            for result in batch:
                theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
                ce_train, err_train = ce_and_error_np(theta, ds["X_train"], ds["y_train"])
                ce_test, err_test = ce_and_error_np(theta, ds["X_test"], ds["y_test"])
                row_attempt = {
                    "dataset_id": dataset_id,
                    "split_id": int(row["split_id"]),
                    "rule": str(row["rule"]),
                    "attempt_seed": int(result["seed"]),
                    "phase": str(result["phase"]),
                    "epoch": int(result["epoch"]),
                    "train_error": float(err_train),
                    "test_error": float(err_test),
                    "ce_mean_train": float(ce_train),
                    "ce_mean_test": float(ce_test),
                    "theta_norm": float(np.linalg.norm(theta)),
                    "selected": False,
                }
                if err_train == 0.0:
                    candidate = {"theta": theta, "attempt_seed": int(result["seed"]), "phase": str(result["phase"])}
                    if smoke.select_reference(selected, {**candidate, "train_error": 0.0}):
                        row_attempt["selected"] = True
                attempt_rows.append(row_attempt)
            write_csv(attempt_log_path, pd.DataFrame(attempt_rows))
            print(f"[final stage04] split={row['split_id']} rule={row['rule']} attempts={attempts_used} selected={len(selected)}/{target_refs}", flush=True)
        if len(selected) < target_refs:
            raise FinalBlocked(
                "04_exact_reference_search",
                "Insufficient exact final references.",
                observed={"split_id": row["split_id"], "rule": row["rule"], "selected": len(selected), "target": target_refs, "attempts": attempts_used},
                expected={"train_error": 0, "refs_per_dataset": target_refs},
                next_action="Increase final max_attempts for the same architecture; if random_label still fails, explicitly evaluate the documented 24-24 backup architecture before continuing.",
            )
        for ref_id, result in enumerate(selected[:target_refs]):
            already = [r for r in reference_rows if int(r["split_id"]) == int(row["split_id"]) and str(r["rule"]) == str(row["rule"]) and int(r["ref_id"]) == ref_id]
            if already and Path(REPO_ROOT / str(already[0]["theta_path"])).exists():
                continue
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ref_dir = ensure_dir(out_dir / "selected_reference_pool" / f"split_{int(row['split_id']):03d}" / str(row["rule"]) / f"ref_{ref_id:03d}")
            theta_path = ref_dir / "theta.npy"
            np.save(theta_path, theta)
            ce_train, err_train = ce_and_error_np(theta, ds["X_train"], ds["y_train"])
            ce_test, err_test = ce_and_error_np(theta, ds["X_test"], ds["y_test"])
            summary = {
                "dataset_id": dataset_id,
                "split_id": int(row["split_id"]),
                "rule": str(row["rule"]),
                "ref_id": ref_id,
                "theta_path": rel(theta_path),
                "dataset_path": str(row["dataset_path"]),
                "attempt_seed": int(result["attempt_seed"]),
                "optimizer_chain": str(result["phase"]),
                "P": int(theta.size),
                "train_error": float(err_train),
                "test_error": float(err_test),
                "CE_mean_train": float(ce_train),
                "CE_sum_train": float(ce_train * ds["X_train"].shape[0]),
                "CE_mean_test": float(ce_test),
                "theta_norm": float(np.linalg.norm(theta)),
                "theta_norm_sq": float(np.dot(theta, theta)),
                **margin_stats_np(theta, ds["X_train"], ds["y_train"]),
                "reference_law_caveat": "optimizer-induced exact reference, not exact P_ref^0 sample",
            }
            write_json(ref_dir / "ref_summary.json", summary)
            reference_rows.append(summary)
        write_csv(out_dir / "reference_index.csv", pd.DataFrame(reference_rows))

    ref_df = pd.DataFrame(reference_rows)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(6, 4))
    ref_df.groupby("rule")["theta_norm"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean theta norm")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_ref_ce_norm_scatter.png", dpi=160)
    plt.close(fig)
    write_json(out_dir / "run_config_resolved.json", {**cfg, "elapsed_s": time.time() - started})
    counts = ref_df.groupby(["split_id", "rule"]).size()
    checks = {
        "reference_rows": int(len(ref_df)),
        "expected_reference_rows": int(cfg["dataset"]["n_splits"]) * len(RULES) * target_refs,
        "min_refs_per_dataset": int(counts.min()) if len(counts) else 0,
        "all_exact": bool((ref_df["train_error"] == 0.0).all()),
        "theta_length_all_P": bool((ref_df["P"] == P).all()),
    }
    if checks["reference_rows"] < checks["expected_reference_rows"] or checks["min_refs_per_dataset"] < target_refs or not checks["all_exact"] or not checks["theta_length_all_P"]:
        raise FinalBlocked("04_exact_reference_search", "Final reference QC failed.", observed=checks)
    write_qc("04_exact_reference_search", "pass", checks)
    write_text(out_dir / "REPORT.md", f"# Final Stage 04 Reference Search\n\nSelected {len(ref_df)} optimizer-induced exact references.")


def normalize_logw(logw: np.ndarray) -> np.ndarray:
    return np.asarray(logw, dtype=np.float64) - logsumexp(logw)


def ess_fraction(logw_norm: np.ndarray) -> float:
    logw_norm = np.asarray(logw_norm, dtype=np.float64)
    return float(np.exp(-logsumexp(2.0 * logw_norm)) / max(1, logw_norm.size))


def cess_fraction(logw_norm: np.ndarray, ce: np.ndarray, delta_t: float, gamma_ce: float) -> float:
    loga = -float(delta_t) * float(gamma_ce) * np.asarray(ce, dtype=np.float64)
    return float(np.exp(2.0 * logsumexp(logw_norm + loga) - logsumexp(logw_norm + 2.0 * loga)))


def choose_temperature(t: float, ce: np.ndarray, logw_norm: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    target = float(cfg["smc"]["target_cess_fraction"])
    full = cess_fraction(logw_norm, ce, 1.0 - t, float(cfg["dataset"]["n_train"]))
    if full >= target:
        return 1.0, full
    low, high = float(t), 1.0
    for _ in range(int(cfg["smc"]["bisection_steps"])):
        mid = 0.5 * (low + high)
        val = cess_fraction(logw_norm, ce, mid - t, float(cfg["dataset"]["n_train"]))
        if val >= target:
            low = mid
        else:
            high = mid
    out = max(low, t + float(cfg["smc"]["min_delta_t"]))
    return min(1.0, out), cess_fraction(logw_norm, ce, min(1.0, out) - t, float(cfg["dataset"]["n_train"]))


def weighted_mean(values: np.ndarray, logw_norm: np.ndarray) -> float:
    w = np.exp(normalize_logw(logw_norm))
    return float(np.sum(w * np.asarray(values, dtype=np.float64)))


def systematic_resample(logw_norm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.exp(normalize_logw(logw_norm))
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cdf, positions, side="left")


def rejuvenate(
    directions: np.ndarray,
    ce: np.ndarray,
    err: np.ndarray,
    theta_ref: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    mu: np.ndarray,
    base_kappa: float,
    t: float,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    move_kappa = float(cfg["smc"]["move_kappa_factor"]) * P
    proposal = sample_vmf_batch(directions, move_kappa, rng)
    theta_prop = theta_ref[None, :] + math.sqrt(P) * float(radius) * proposal
    ce_prop, err_prop = ce_error_batch_torch(theta_prop, x, y, chunk_size=int(cfg["compute"]["chunk_size"]), device=str(cfg["compute"]["device"]), dtype=str(cfg["compute"]["dtype"]))
    current_proj = directions @ mu
    prop_proj = proposal @ mu
    log_accept = -float(t) * float(cfg["dataset"]["n_train"]) * (ce_prop - ce) + float(base_kappa) * (prop_proj - current_proj)
    accept = np.log(rng.random(size=ce.size)) <= np.minimum(0.0, log_accept)
    if np.any(accept):
        directions[accept] = proposal[accept]
        ce[accept] = ce_prop[accept]
        err[accept] = err_prop[accept]
    return directions, ce, err, float(np.mean(accept))


def run_smc_split(theta_ref: np.ndarray, ds: dict[str, np.ndarray], radius: float, n_samples: int, lambda_reg: float, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    theta_ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    ref_norm = float(np.linalg.norm(theta_ref))
    mu = -theta_ref / ref_norm
    base_kappa = float(lambda_reg * float(radius) * ref_norm / math.sqrt(P))
    gamma_ce = float(cfg["dataset"]["n_train"])
    split_outputs: list[dict[str, Any]] = []
    for split_id, n_particles in enumerate([n_samples // 2, n_samples - n_samples // 2]):
        rng = np.random.default_rng(int(seed) + 7919 * (split_id + 1))
        directions = sample_vmf(mu, base_kappa, int(n_particles), rng)
        theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * directions
        ce, err = ce_error_batch_torch(theta_batch, ds["X_train"], ds["y_train"], chunk_size=int(cfg["compute"]["chunk_size"]), device=str(cfg["compute"]["device"]), dtype=str(cfg["compute"]["dtype"]))
        logw_norm = np.full(int(n_particles), -math.log(int(n_particles)), dtype=np.float64)
        t = 0.0
        logz_ce = 0.0
        history: list[dict[str, Any]] = []
        completed = True
        for step in range(int(cfg["smc"]["max_steps"])):
            if t >= 1.0 - 1.0e-12:
                break
            t_new, cess = choose_temperature(t, ce, logw_norm, cfg)
            delta_t = max(0.0, t_new - t)
            loga = -delta_t * gamma_ce * ce
            logz_ce += float(logsumexp(logw_norm + loga))
            logw_norm = normalize_logw(logw_norm + loga)
            ess_after = ess_fraction(logw_norm)
            resampled = ess_after < float(cfg["smc"]["resample_ess_fraction"])
            if resampled:
                idx = systematic_resample(logw_norm, rng)
                directions = directions[idx].copy()
                ce = ce[idx].copy()
                err = err[idx].copy()
                logw_norm = np.full(int(n_particles), -math.log(int(n_particles)), dtype=np.float64)
                ess_after = 1.0
            acc = float("nan")
            for _ in range(int(cfg["smc"]["mh_sweeps"])):
                directions, ce, err, acc = rejuvenate(directions, ce, err, theta_ref, ds["X_train"], ds["y_train"], radius, mu, base_kappa, t_new, cfg, rng)
            history.append({"step": step + 1, "t_start": t, "t_end": t_new, "cess_fraction": cess, "ess_fraction_after_reweight": ess_after, "resampled": resampled, "mh_acceptance": acc})
            t = t_new
        else:
            completed = t >= 1.0 - 1.0e-12
        split_outputs.append({"logZ_CE": logz_ce if completed else float("nan"), "ce": ce, "err": err, "directions": directions, "logw_norm": normalize_logw(logw_norm), "history": history, "completed": completed})

    logz_values = np.asarray([float(s["logZ_CE"]) for s in split_outputs], dtype=np.float64)
    counts = np.asarray([len(split_outputs[0]["ce"]), len(split_outputs[1]["ce"])], dtype=np.float64)
    logz_ce = float(logsumexp(np.log(counts / np.sum(counts)) + logz_values)) if np.all(np.isfinite(logz_values)) else float("nan")
    log_prefactor = -float(lambda_reg) * float(radius) * float(radius) / 2.0 + log_sphere_mgf(P, base_kappa)
    ce = np.concatenate([s["ce"] for s in split_outputs])
    err = np.concatenate([s["err"] for s in split_outputs])
    logw = np.concatenate([math.log(counts[i] / np.sum(counts)) + split_outputs[i]["logw_norm"] for i in range(2)])
    dirs = np.concatenate([s["directions"] for s in split_outputs], axis=0)
    flat_history = [h for s in split_outputs for h in s["history"]]
    return {
        "logZ": float(log_prefactor + logz_ce) if np.isfinite(logz_ce) else float("nan"),
        "logZ_CE": logz_ce,
        "split0_logZ": float(log_prefactor + logz_values[0]) if np.isfinite(logz_values[0]) else float("nan"),
        "split1_logZ": float(log_prefactor + logz_values[1]) if np.isfinite(logz_values[1]) else float("nan"),
        "split_logZ_per_P_diff": float(abs(logz_values[0] - logz_values[1]) / P) if np.all(np.isfinite(logz_values)) else float("inf"),
        "ess_fraction": ess_fraction(logw),
        "weighted_ce": weighted_mean(ce, logw),
        "weighted_error": weighted_mean(err, logw),
        "smc_completed": bool(all(s["completed"] for s in split_outputs)),
        "smc_step_count": int(max(len(s["history"]) for s in split_outputs)),
        "smc_total_step_count": int(sum(len(s["history"]) for s in split_outputs)),
        "smc_min_cess_fraction": float(np.min([h["cess_fraction"] for h in flat_history])) if flat_history else float("nan"),
        "smc_mean_mh_acceptance": float(np.nanmean([h["mh_acceptance"] for h in flat_history])) if flat_history else float("nan"),
        "hard_shell_distance_max_abs_err": float(np.max(np.abs(np.linalg.norm(theta_ref[None, :] + math.sqrt(P) * float(radius) * dirs - theta_ref[None, :], axis=1) / math.sqrt(P) - float(radius)))),
        "direction_unit_norm_max_abs_err": float(np.max(np.abs(np.linalg.norm(dirs, axis=1) - 1.0))),
        "kappa": base_kappa,
        "logM": log_sphere_mgf(P, base_kappa),
        "log_prefactor": log_prefactor,
    }


def unit_summary_path(row: dict[str, Any], radius: float) -> Path:
    return (
        stage_dir("05_pool2_pm_sais_sampling")
        / "unit_summaries"
        / f"split_{int(row['split_id']):03d}"
        / str(row["rule"])
        / f"ref_{int(row['ref_id']):03d}"
        / f"r_{float(radius):.2f}".replace(".", "p")
        / "unit_summary.json"
    )


def reusable_unit_summary(payload: dict[str, Any], radius: float, n_samples: int, lambda_reg: float) -> bool:
    return (
        int(payload.get("n_samples", -1)) >= int(n_samples)
        and str(payload.get("sampler_method", "")) == "exact_shell_l2_vmf_adaptive_ce_tempered_smc"
        and abs(float(payload.get("lambda_reg", float("nan"))) - float(lambda_reg)) <= 1.0e-12
        and abs(float(payload.get("radius", float("nan"))) - float(radius)) <= 1.0e-12
        and math.isfinite(float(payload.get("logZ", float("nan"))))
        and math.isfinite(float(payload.get("split_logZ_per_P_diff", float("nan"))))
    )


def sample_unit(row: dict[str, Any], radius: float, cfg: dict[str, Any], *, n_samples: int, lambda_reg: float, force: bool = False) -> dict[str, Any]:
    path = unit_summary_path(row, radius)
    if path.exists() and not force:
        payload = read_json(path)
        if reusable_unit_summary(payload, radius, n_samples, lambda_reg):
            payload["reused"] = True
            return payload
    ds = load_dataset(row["dataset_path"])
    theta_path = REPO_ROOT / str(row["theta_path"])
    theta_key = str(theta_path.resolve())
    if theta_key not in _THETA_CACHE:
        _THETA_CACHE[theta_key] = np.load(theta_path).astype(np.float64).reshape(-1)
    theta_ref = _THETA_CACHE[theta_key]
    seed = 2900000 + int(row["split_id"]) * 100000 + RULES.index(str(row["rule"])) * 10000 + int(row["ref_id"]) * 251 + int(round(float(radius) * 100))
    started = time.time()
    smc = run_smc_split(theta_ref, ds, float(radius), int(n_samples), float(lambda_reg), seed, cfg)
    payload = {
        "split_id": int(row["split_id"]),
        "rule": str(row["rule"]),
        "ref_id": int(row["ref_id"]),
        "radius": float(radius),
        "n_samples": int(n_samples),
        "lambda_reg": float(lambda_reg),
        "seed": int(seed),
        "theta_path": str(row["theta_path"]),
        "dataset_path": str(row["dataset_path"]),
        "theta_ref_norm": float(np.linalg.norm(theta_ref)),
        "sampler_method": "exact_shell_l2_vmf_adaptive_ce_tempered_smc",
        "elapsed_s": float(time.time() - started),
        "reused": False,
        **smc,
    }
    write_json(path, payload)
    return payload


def sample_unit_worker(args: tuple[dict[str, Any], float, dict[str, Any], int, float, bool]) -> dict[str, Any]:
    row, radius, cfg, n_samples, lambda_reg, force = args
    return sample_unit(row, radius, cfg, n_samples=n_samples, lambda_reg=lambda_reg, force=force)


def logmeanexp(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(logsumexp(arr) - math.log(arr.size)) if arr.size and np.all(np.isfinite(arr)) else float("nan")


def load_theta(row: dict[str, Any]) -> np.ndarray:
    theta_path = REPO_ROOT / str(row["theta_path"])
    theta_key = str(theta_path.resolve())
    if theta_key not in _THETA_CACHE:
        _THETA_CACHE[theta_key] = np.load(theta_path).astype(np.float64).reshape(-1)
    return _THETA_CACHE[theta_key]


def summarize_replicate_rows(
    row: dict[str, Any],
    radius: float,
    theta_ref: np.ndarray,
    replicate_rows: list[dict[str, Any]],
    *,
    n_samples_each: int,
    lambda_reg: float,
    seed: int,
    elapsed_s: float,
) -> dict[str, Any]:
    logz = [float(rep["logZ"]) for rep in replicate_rows]
    even = logz[0::2]
    odd = logz[1::2]
    combined_logz = logmeanexp(logz)
    combined_split = (
        float(abs(logmeanexp(even) - logmeanexp(odd)) / P)
        if even and odd and math.isfinite(logmeanexp(even)) and math.isfinite(logmeanexp(odd))
        else float("inf")
    )
    ess_values = [float(rep["ess_fraction"]) for rep in replicate_rows]
    split_values = [float(rep["split_logZ_per_P_diff"]) for rep in replicate_rows]
    return {
        "split_id": int(row["split_id"]),
        "rule": str(row["rule"]),
        "ref_id": int(row["ref_id"]),
        "radius": float(radius),
        "replicates": int(len(replicate_rows)),
        "n_samples_each": int(n_samples_each),
        "n_samples_total": int(len(replicate_rows)) * int(n_samples_each),
        "lambda_reg": float(lambda_reg),
        "seed": int(seed),
        "theta_path": str(row["theta_path"]),
        "dataset_path": str(row["dataset_path"]),
        "theta_ref_norm": float(np.linalg.norm(theta_ref)),
        "sampler_method": "replicated_exact_shell_l2_vmf_adaptive_ce_tempered_smc",
        "logZ": combined_logz,
        "split_logZ_per_P_diff": combined_split,
        "replicate_logZ_per_P_range": float((np.max(logz) - np.min(logz)) / P) if np.all(np.isfinite(logz)) else float("inf"),
        "replicate_split_logZ_per_P_diff_max": float(np.max(split_values)) if split_values else float("inf"),
        "ess_fraction_min": float(np.min(ess_values)) if ess_values else float("nan"),
        "ess_fraction_mean": float(np.mean(ess_values)) if ess_values else float("nan"),
        "smc_step_count_max": int(max(int(rep["smc_step_count"]) for rep in replicate_rows)),
        "smc_min_cess_fraction": float(min(float(rep["smc_min_cess_fraction"]) for rep in replicate_rows)),
        "smc_mean_mh_acceptance": float(np.mean([float(rep["smc_mean_mh_acceptance"]) for rep in replicate_rows])),
        "hard_shell_distance_max_abs_err": float(max(float(rep["hard_shell_distance_max_abs_err"]) for rep in replicate_rows)),
        "direction_unit_norm_max_abs_err": float(max(float(rep["direction_unit_norm_max_abs_err"]) for rep in replicate_rows)),
        "elapsed_s": float(elapsed_s),
        "replicate_summaries": replicate_rows,
    }


def run_replicated_smc(
    row: dict[str, Any],
    radius: float,
    cfg: dict[str, Any],
    *,
    n_samples_each: int,
    replicates: int,
    lambda_reg: float,
    seed: int,
) -> dict[str, Any]:
    ds = load_dataset(row["dataset_path"])
    theta_ref = load_theta(row)
    started = time.time()
    replicate_rows: list[dict[str, Any]] = []
    for rep_id in range(int(replicates)):
        rep_seed = int(seed) + 1000003 * rep_id
        smc = run_smc_split(theta_ref, ds, float(radius), int(n_samples_each), float(lambda_reg), rep_seed, cfg)
        replicate_rows.append({"replicate_id": rep_id, "seed": rep_seed, **smc})
    return summarize_replicate_rows(
        row,
        radius,
        theta_ref,
        replicate_rows,
        n_samples_each=n_samples_each,
        lambda_reg=lambda_reg,
        seed=seed,
        elapsed_s=time.time() - started,
    )


def stability_case_path(case_name: str, row: dict[str, Any], radius: float) -> Path:
    return (
        stage_dir("05_pool2_pm_sais_sampling")
        / "stability_pilot"
        / case_name
        / f"split_{int(row['split_id']):03d}"
        / str(row["rule"])
        / f"ref_{int(row['ref_id']):03d}"
        / f"r_{float(radius):.2f}".replace(".", "p")
        / "stability_summary.json"
    )


def stage05_stability_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not ref_path.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Reference index missing for stability pilot.", observed={"missing": rel(ref_path)})
    case_name = "rep4_n1024_cess90_mh2_representative"
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling") / "stability_pilot" / case_name)
    pilot_cfg = copy.deepcopy(cfg)
    pilot_cfg["smc"]["target_cess_fraction"] = 0.90
    pilot_cfg["smc"]["resample_ess_fraction"] = 0.90
    pilot_cfg["smc"]["max_steps"] = 220
    pilot_cfg["smc"]["min_delta_t"] = 5.0e-5
    pilot_cfg["smc"]["mh_sweeps"] = 2
    ref_df = pd.read_csv(ref_path)
    tasks: list[tuple[dict[str, Any], float]] = []
    for rule in RULES:
        row = ref_df[ref_df["rule"] == rule].sort_values(["split_id", "ref_id"]).iloc[0].to_dict()
        radii = [2.50] if rule != "real_even_odd" else [0.15, 1.00, 2.13, 2.50]
        for radius in radii:
            tasks.append((row, float(radius)))
    rows: list[dict[str, Any]] = []
    for task_id, (row, radius) in enumerate(tasks):
        path = stability_case_path(case_name, row, radius)
        if path.exists() and not force:
            payload = read_json(path)
            payload["reused"] = True
        else:
            seed = 5100000 + task_id * 10000 + RULES.index(str(row["rule"])) * 1000 + int(round(float(radius) * 100))
            print(f"[stability pilot] {case_name} rule={row['rule']} split={row['split_id']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            payload = run_replicated_smc(row, radius, pilot_cfg, n_samples_each=1024, replicates=4, lambda_reg=1.0, seed=seed)
            payload["case_name"] = case_name
            payload["smc_target_cess_fraction"] = float(pilot_cfg["smc"]["target_cess_fraction"])
            payload["smc_resample_ess_fraction"] = float(pilot_cfg["smc"]["resample_ess_fraction"])
            payload["smc_mh_sweeps"] = int(pilot_cfg["smc"]["mh_sweeps"])
            payload["smc_max_steps"] = int(pilot_cfg["smc"]["max_steps"])
            payload["reused"] = False
            write_json(path, payload)
        rows.append(
            {
                k: payload[k]
                for k in [
                    "case_name",
                    "rule",
                    "split_id",
                    "ref_id",
                    "radius",
                    "replicates",
                    "n_samples_each",
                    "n_samples_total",
                    "split_logZ_per_P_diff",
                    "replicate_logZ_per_P_range",
                    "replicate_split_logZ_per_P_diff_max",
                    "ess_fraction_min",
                    "ess_fraction_mean",
                    "smc_step_count_max",
                    "smc_min_cess_fraction",
                    "smc_mean_mh_acceptance",
                    "elapsed_s",
                    "reused",
                ]
            }
        )
    df = pd.DataFrame(rows)
    df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= float(cfg["qc"]["max_split_logZ_per_P_diff"])
    df["ess_gate_pass"] = df["ess_fraction_min"] >= float(cfg["qc"]["q05_ess_fraction_min"])
    df["hard_shell_gate_pass"] = True
    df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & df["hard_shell_gate_pass"]
    write_csv(out_dir / "stability_case_summary.csv", df)
    failed = df[~df["pilot_pass"]]
    status = "pass" if failed.empty else "needs_stronger_stability"
    next_action = (
        "Resume final sampling with replicated 4x1024, CESS=0.90, MH=2 only if this pilot passes all representative cases."
        if failed.empty
        else "Escalate the stability pilot before final sampling: test more particles/replicates or a better rejuvenation kernel on the failed representative cases."
    )
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_stability_pilot",
            "status": status,
            "case_name": case_name,
            "checks": {
                "cases": int(len(df)),
                "passed_cases": int(df["pilot_pass"].sum()),
                "failed_cases": int((~df["pilot_pass"]).sum()),
                "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()),
                "threshold": float(cfg["qc"]["max_split_logZ_per_P_diff"]),
            },
            "failed_cases": failed.to_dict("records"),
            "next_action": next_action,
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Stability Pilot

Case: `{case_name}`

Representative cases: {len(df)}

Passed cases: {int(df['pilot_pass'].sum())}

Failed cases: {int((~df['pilot_pass']).sum())}

Next action: {next_action}
""",
    )


def stage05_stability_escalated_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    first_case = stage_dir("05_pool2_pm_sais_sampling") / "stability_pilot" / "rep4_n1024_cess90_mh2_representative" / "QC_STATUS.json"
    if not ref_path.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Reference index missing for escalated stability pilot.", observed={"missing": rel(ref_path)})
    if not first_case.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Representative stability pilot missing.", observed={"missing": rel(first_case)})
    first_qc = read_json(first_case)
    failed_cases = list(first_qc.get("failed_cases", []))
    case_name = "rep8_n1024_cess90_mh2_failed_cases"
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling") / "stability_pilot" / case_name)
    pilot_cfg = copy.deepcopy(cfg)
    pilot_cfg["smc"]["target_cess_fraction"] = 0.90
    pilot_cfg["smc"]["resample_ess_fraction"] = 0.90
    pilot_cfg["smc"]["max_steps"] = 220
    pilot_cfg["smc"]["min_delta_t"] = 5.0e-5
    pilot_cfg["smc"]["mh_sweeps"] = 2
    ref_df = pd.read_csv(ref_path)
    rows: list[dict[str, Any]] = []
    for task_id, failed in enumerate(failed_cases):
        sub = ref_df[
            (ref_df["rule"] == str(failed["rule"]))
            & (ref_df["split_id"] == int(failed["split_id"]))
            & (ref_df["ref_id"] == int(failed["ref_id"]))
        ]
        if sub.empty:
            raise FinalBlocked("05_pool2_pm_sais_sampling", "Failed stability pilot reference row missing.", observed=failed)
        row = sub.iloc[0].to_dict()
        radius = float(failed["radius"])
        path = stability_case_path(case_name, row, radius)
        if path.exists() and not force:
            payload = read_json(path)
            payload["reused"] = True
        else:
            seed = 6100000 + task_id * 10000 + RULES.index(str(row["rule"])) * 1000 + int(round(radius * 100))
            print(f"[stability escalated] {case_name} rule={row['rule']} split={row['split_id']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            payload = run_replicated_smc(row, radius, pilot_cfg, n_samples_each=1024, replicates=8, lambda_reg=1.0, seed=seed)
            payload["case_name"] = case_name
            payload["smc_target_cess_fraction"] = float(pilot_cfg["smc"]["target_cess_fraction"])
            payload["smc_resample_ess_fraction"] = float(pilot_cfg["smc"]["resample_ess_fraction"])
            payload["smc_mh_sweeps"] = int(pilot_cfg["smc"]["mh_sweeps"])
            payload["smc_max_steps"] = int(pilot_cfg["smc"]["max_steps"])
            payload["reused"] = False
            write_json(path, payload)
        rows.append(
            {
                k: payload[k]
                for k in [
                    "case_name",
                    "rule",
                    "split_id",
                    "ref_id",
                    "radius",
                    "replicates",
                    "n_samples_each",
                    "n_samples_total",
                    "split_logZ_per_P_diff",
                    "replicate_logZ_per_P_range",
                    "replicate_split_logZ_per_P_diff_max",
                    "ess_fraction_min",
                    "ess_fraction_mean",
                    "smc_step_count_max",
                    "smc_min_cess_fraction",
                    "smc_mean_mh_acceptance",
                    "elapsed_s",
                    "reused",
                ]
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["case_name", "rule", "radius", "pilot_pass"])
        failed = df
        status = "pass"
        next_action = "Representative stability pilot had no failed cases; no escalation needed."
    else:
        df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= float(cfg["qc"]["max_split_logZ_per_P_diff"])
        df["ess_gate_pass"] = df["ess_fraction_min"] >= float(cfg["qc"]["q05_ess_fraction_min"])
        df["hard_shell_gate_pass"] = True
        df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & df["hard_shell_gate_pass"]
        failed = df[~df["pilot_pass"]]
        status = "pass" if failed.empty else "needs_stronger_stability"
        next_action = (
            "Use replicated 8x1024, CESS=0.90, MH=2 as the minimum final sampling stability preset for hard high-radius cases."
            if failed.empty
            else "Escalation still fails; test more particles/replicates or a better rejuvenation kernel before final sampling."
        )
    write_csv(out_dir / "stability_case_summary.csv", df)
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_stability_escalated_pilot",
            "status": status,
            "case_name": case_name,
            "checks": {
                "cases": int(len(df)),
                "passed_cases": int(df["pilot_pass"].sum()) if "pilot_pass" in df else 0,
                "failed_cases": int((~df["pilot_pass"]).sum()) if "pilot_pass" in df else 0,
                "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()) if "split_logZ_per_P_diff" in df and len(df) else None,
                "threshold": float(cfg["qc"]["max_split_logZ_per_P_diff"]),
            },
            "failed_cases": failed.to_dict("records") if len(df) else [],
            "next_action": next_action,
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Escalated Stability Pilot

Case: `{case_name}`

Escalated cases: {len(df)}

Passed cases: {int(df['pilot_pass'].sum()) if 'pilot_pass' in df else 0}

Failed cases: {int((~df['pilot_pass']).sum()) if 'pilot_pass' in df else 0}

Next action: {next_action}
""",
    )


def extend_replicated_smc(
    row: dict[str, Any],
    radius: float,
    cfg: dict[str, Any],
    base_payload: dict[str, Any],
    *,
    target_replicates: int,
    lambda_reg: float,
) -> dict[str, Any]:
    ds = load_dataset(row["dataset_path"])
    theta_ref = load_theta(row)
    started = time.time()
    replicate_rows = list(base_payload["replicate_summaries"])
    existing = {int(rep["replicate_id"]) for rep in replicate_rows}
    seed = int(base_payload["seed"])
    n_samples_each = int(base_payload["n_samples_each"])
    for rep_id in range(int(target_replicates)):
        if rep_id in existing:
            continue
        rep_seed = seed + 1000003 * rep_id
        smc = run_smc_split(theta_ref, ds, float(radius), n_samples_each, float(lambda_reg), rep_seed, cfg)
        replicate_rows.append({"replicate_id": rep_id, "seed": rep_seed, **smc})
    replicate_rows = sorted(replicate_rows, key=lambda rep: int(rep["replicate_id"]))
    elapsed_s = float(base_payload.get("elapsed_s", 0.0)) + float(time.time() - started)
    return summarize_replicate_rows(
        row,
        radius,
        theta_ref,
        replicate_rows,
        n_samples_each=n_samples_each,
        lambda_reg=lambda_reg,
        seed=seed,
        elapsed_s=elapsed_s,
    )


def stage05_stability_rep16_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    rep8_qc_path = stage_dir("05_pool2_pm_sais_sampling") / "stability_pilot" / "rep8_n1024_cess90_mh2_failed_cases" / "QC_STATUS.json"
    if not ref_path.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Reference index missing for rep16 stability pilot.", observed={"missing": rel(ref_path)})
    if not rep8_qc_path.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "rep8 stability pilot missing.", observed={"missing": rel(rep8_qc_path)})
    rep8_qc = read_json(rep8_qc_path)
    failed_cases = list(rep8_qc.get("failed_cases", []))
    case_name = "rep16_n1024_cess90_mh2_failed_cases"
    base_case_name = "rep8_n1024_cess90_mh2_failed_cases"
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling") / "stability_pilot" / case_name)
    pilot_cfg = copy.deepcopy(cfg)
    pilot_cfg["smc"]["target_cess_fraction"] = 0.90
    pilot_cfg["smc"]["resample_ess_fraction"] = 0.90
    pilot_cfg["smc"]["max_steps"] = 220
    pilot_cfg["smc"]["min_delta_t"] = 5.0e-5
    pilot_cfg["smc"]["mh_sweeps"] = 2
    ref_df = pd.read_csv(ref_path)
    rows: list[dict[str, Any]] = []
    for failed in failed_cases:
        sub = ref_df[
            (ref_df["rule"] == str(failed["rule"]))
            & (ref_df["split_id"] == int(failed["split_id"]))
            & (ref_df["ref_id"] == int(failed["ref_id"]))
        ]
        if sub.empty:
            raise FinalBlocked("05_pool2_pm_sais_sampling", "rep16 reference row missing.", observed=failed)
        row = sub.iloc[0].to_dict()
        radius = float(failed["radius"])
        base_path = stability_case_path(base_case_name, row, radius)
        path = stability_case_path(case_name, row, radius)
        if not base_path.exists():
            raise FinalBlocked("05_pool2_pm_sais_sampling", "rep8 case payload missing for rep16 extension.", observed={"missing": rel(base_path)})
        if path.exists() and not force:
            payload = read_json(path)
            payload["reused"] = True
        else:
            print(f"[stability rep16] {case_name} rule={row['rule']} split={row['split_id']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            base_payload = read_json(base_path)
            payload = extend_replicated_smc(row, radius, pilot_cfg, base_payload, target_replicates=16, lambda_reg=1.0)
            payload["case_name"] = case_name
            payload["base_case_name"] = base_case_name
            payload["smc_target_cess_fraction"] = float(pilot_cfg["smc"]["target_cess_fraction"])
            payload["smc_resample_ess_fraction"] = float(pilot_cfg["smc"]["resample_ess_fraction"])
            payload["smc_mh_sweeps"] = int(pilot_cfg["smc"]["mh_sweeps"])
            payload["smc_max_steps"] = int(pilot_cfg["smc"]["max_steps"])
            payload["reused"] = False
            write_json(path, payload)
        rows.append(
            {
                k: payload[k]
                for k in [
                    "case_name",
                    "rule",
                    "split_id",
                    "ref_id",
                    "radius",
                    "replicates",
                    "n_samples_each",
                    "n_samples_total",
                    "split_logZ_per_P_diff",
                    "replicate_logZ_per_P_range",
                    "replicate_split_logZ_per_P_diff_max",
                    "ess_fraction_min",
                    "ess_fraction_mean",
                    "smc_step_count_max",
                    "smc_min_cess_fraction",
                    "smc_mean_mh_acceptance",
                    "elapsed_s",
                    "reused",
                ]
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["case_name", "rule", "radius", "pilot_pass"])
        failed = df
        status = "pass"
        next_action = "rep8 stability pilot had no failed cases; no rep16 extension needed."
    else:
        df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= float(cfg["qc"]["max_split_logZ_per_P_diff"])
        df["ess_gate_pass"] = df["ess_fraction_min"] >= float(cfg["qc"]["q05_ess_fraction_min"])
        df["hard_shell_gate_pass"] = True
        df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & df["hard_shell_gate_pass"]
        failed = df[~df["pilot_pass"]]
        status = "pass" if failed.empty else "needs_stronger_stability"
        next_action = (
            "Use replicated 16x1024, CESS=0.90, MH=2 as the minimum final sampling stability preset for hard high-radius cases."
            if failed.empty
            else "rep16 still fails; use a different rejuvenation kernel or accept that current PM-SAIS final cannot support all high radii under the documented split QC."
        )
    write_csv(out_dir / "stability_case_summary.csv", df)
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_stability_rep16_pilot",
            "status": status,
            "case_name": case_name,
            "checks": {
                "cases": int(len(df)),
                "passed_cases": int(df["pilot_pass"].sum()) if "pilot_pass" in df else 0,
                "failed_cases": int((~df["pilot_pass"]).sum()) if "pilot_pass" in df else 0,
                "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()) if "split_logZ_per_P_diff" in df and len(df) else None,
                "threshold": float(cfg["qc"]["max_split_logZ_per_P_diff"]),
            },
            "failed_cases": failed.to_dict("records") if len(df) else [],
            "next_action": next_action,
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Rep16 Stability Pilot

Case: `{case_name}`

Extended cases: {len(df)}

Passed cases: {int(df['pilot_pass'].sum()) if 'pilot_pass' in df else 0}

Failed cases: {int((~df['pilot_pass']).sum()) if 'pilot_pass' in df else 0}

Next action: {next_action}
""",
    )


def stage05_stability_kernel_scan(*, force: bool = False) -> None:
    cfg = load_config()
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not ref_path.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Reference index missing for stability kernel scan.", observed={"missing": rel(ref_path)})
    ref_df = pd.read_csv(ref_path)
    row = ref_df[ref_df["rule"] == "teacher_nn"].sort_values(["split_id", "ref_id"]).iloc[0].to_dict()
    radius = 2.50
    case_name = "teacher_nn_d2p50_kernel_scan"
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling") / "stability_pilot" / case_name)
    variants = [
        {"name": "rep8_n1024_cess95_mh2_move80", "replicates": 8, "target_cess": 0.95, "resample_ess": 0.90, "mh_sweeps": 2, "move_kappa_factor": 80.0, "max_steps": 320},
        {"name": "rep8_n1024_cess90_mh4_move40", "replicates": 8, "target_cess": 0.90, "resample_ess": 0.90, "mh_sweeps": 4, "move_kappa_factor": 40.0, "max_steps": 260},
    ]
    rows: list[dict[str, Any]] = []
    for variant_id, variant in enumerate(variants):
        variant_cfg = copy.deepcopy(cfg)
        variant_cfg["smc"]["target_cess_fraction"] = float(variant["target_cess"])
        variant_cfg["smc"]["resample_ess_fraction"] = float(variant["resample_ess"])
        variant_cfg["smc"]["mh_sweeps"] = int(variant["mh_sweeps"])
        variant_cfg["smc"]["move_kappa_factor"] = float(variant["move_kappa_factor"])
        variant_cfg["smc"]["max_steps"] = int(variant["max_steps"])
        variant_cfg["smc"]["min_delta_t"] = 5.0e-5
        path = stability_case_path(case_name, row, radius).with_name(f"{variant['name']}_stability_summary.json")
        if path.exists() and not force:
            payload = read_json(path)
            payload["reused"] = True
        else:
            seed = 7100000 + variant_id * 100000 + int(round(radius * 100))
            print(f"[stability kernel scan] {variant['name']} rule={row['rule']} split={row['split_id']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            payload = run_replicated_smc(row, radius, variant_cfg, n_samples_each=1024, replicates=int(variant["replicates"]), lambda_reg=1.0, seed=seed)
            payload["case_name"] = case_name
            payload["variant"] = str(variant["name"])
            payload["smc_target_cess_fraction"] = float(variant_cfg["smc"]["target_cess_fraction"])
            payload["smc_resample_ess_fraction"] = float(variant_cfg["smc"]["resample_ess_fraction"])
            payload["smc_mh_sweeps"] = int(variant_cfg["smc"]["mh_sweeps"])
            payload["smc_move_kappa_factor"] = float(variant_cfg["smc"]["move_kappa_factor"])
            payload["smc_max_steps"] = int(variant_cfg["smc"]["max_steps"])
            payload["reused"] = False
            write_json(path, payload)
        row_out = {
            "case_name": case_name,
            "variant": str(variant["name"]),
            "rule": payload["rule"],
            "split_id": payload["split_id"],
            "ref_id": payload["ref_id"],
            "radius": payload["radius"],
            "replicates": payload["replicates"],
            "n_samples_each": payload["n_samples_each"],
            "n_samples_total": payload["n_samples_total"],
            "split_logZ_per_P_diff": payload["split_logZ_per_P_diff"],
            "replicate_logZ_per_P_range": payload["replicate_logZ_per_P_range"],
            "replicate_split_logZ_per_P_diff_max": payload["replicate_split_logZ_per_P_diff_max"],
            "ess_fraction_min": payload["ess_fraction_min"],
            "ess_fraction_mean": payload["ess_fraction_mean"],
            "smc_step_count_max": payload["smc_step_count_max"],
            "smc_min_cess_fraction": payload["smc_min_cess_fraction"],
            "smc_mean_mh_acceptance": payload["smc_mean_mh_acceptance"],
            "smc_target_cess_fraction": payload["smc_target_cess_fraction"],
            "smc_mh_sweeps": payload["smc_mh_sweeps"],
            "smc_move_kappa_factor": payload["smc_move_kappa_factor"],
            "elapsed_s": payload["elapsed_s"],
            "reused": payload["reused"],
        }
        rows.append(row_out)
    df = pd.DataFrame(rows)
    df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= float(cfg["qc"]["max_split_logZ_per_P_diff"])
    df["ess_gate_pass"] = df["ess_fraction_min"] >= float(cfg["qc"]["q05_ess_fraction_min"])
    df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"]
    failed = df[~df["pilot_pass"]]
    status = "pass" if not df.empty and bool(df["pilot_pass"].any()) else "needs_stronger_stability"
    passing = df[df["pilot_pass"]].sort_values(["split_logZ_per_P_diff", "elapsed_s"]).to_dict("records")
    next_action = (
        f"Use the best passing kernel variant `{passing[0]['variant']}` for a broader all-rule stability pilot."
        if passing
        else "No scanned kernel variant passed teacher_nn d=2.50; current PM-SAIS kernel is not adequate for final high-radius support under documented split QC."
    )
    write_csv(out_dir / "kernel_scan_summary.csv", df)
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_stability_kernel_scan",
            "status": status,
            "case_name": case_name,
            "checks": {
                "variants": int(len(df)),
                "passing_variants": int(df["pilot_pass"].sum()),
                "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()) if len(df) else None,
                "min_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].min()) if len(df) else None,
                "threshold": float(cfg["qc"]["max_split_logZ_per_P_diff"]),
            },
            "passing_variants": passing,
            "failed_variants": failed.to_dict("records"),
            "next_action": next_action,
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Stability Kernel Scan

Case: `teacher_nn d=2.50`

Variants: {len(df)}

Passing variants: {int(df['pilot_pass'].sum())}

Next action: {next_action}
""",
    )


def stage05_stability_best_broad_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not ref_path.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Reference index missing for broad stability pilot.", observed={"missing": rel(ref_path)})
    case_name = "best_rep8_n1024_cess95_mh2_move80_broad"
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling") / "stability_pilot" / case_name)
    pilot_cfg = copy.deepcopy(cfg)
    pilot_cfg["smc"]["target_cess_fraction"] = 0.95
    pilot_cfg["smc"]["resample_ess_fraction"] = 0.90
    pilot_cfg["smc"]["mh_sweeps"] = 2
    pilot_cfg["smc"]["move_kappa_factor"] = 80.0
    pilot_cfg["smc"]["max_steps"] = 320
    pilot_cfg["smc"]["min_delta_t"] = 5.0e-5
    ref_df = pd.read_csv(ref_path)
    tasks: list[tuple[dict[str, Any], float]] = []
    for rule, radii in {
        "real_even_odd": [0.15, 2.13, 2.50],
        "teacher_nn": [2.50],
        "random_label": [2.50],
    }.items():
        row = ref_df[ref_df["rule"] == rule].sort_values(["split_id", "ref_id"]).iloc[0].to_dict()
        for radius in radii:
            tasks.append((row, float(radius)))
    rows: list[dict[str, Any]] = []
    for task_id, (row, radius) in enumerate(tasks):
        path = stability_case_path(case_name, row, radius)
        source_payload_path = (
            stability_case_path("teacher_nn_d2p50_kernel_scan", row, radius).with_name("rep8_n1024_cess95_mh2_move80_stability_summary.json")
            if str(row["rule"]) == "teacher_nn" and abs(float(radius) - 2.50) <= 1.0e-12
            else None
        )
        if path.exists() and not force:
            payload = read_json(path)
            payload["reused"] = True
        elif source_payload_path is not None and source_payload_path.exists() and not force:
            payload = read_json(source_payload_path)
            payload["case_name"] = case_name
            payload["reused"] = True
            payload["reused_from"] = rel(source_payload_path)
            write_json(path, payload)
        else:
            seed = 8100000 + task_id * 10000 + RULES.index(str(row["rule"])) * 1000 + int(round(float(radius) * 100))
            print(f"[stability broad] {case_name} rule={row['rule']} split={row['split_id']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            payload = run_replicated_smc(row, radius, pilot_cfg, n_samples_each=1024, replicates=8, lambda_reg=1.0, seed=seed)
            payload["case_name"] = case_name
            payload["variant"] = "rep8_n1024_cess95_mh2_move80"
            payload["smc_target_cess_fraction"] = float(pilot_cfg["smc"]["target_cess_fraction"])
            payload["smc_resample_ess_fraction"] = float(pilot_cfg["smc"]["resample_ess_fraction"])
            payload["smc_mh_sweeps"] = int(pilot_cfg["smc"]["mh_sweeps"])
            payload["smc_move_kappa_factor"] = float(pilot_cfg["smc"]["move_kappa_factor"])
            payload["smc_max_steps"] = int(pilot_cfg["smc"]["max_steps"])
            payload["reused"] = False
            write_json(path, payload)
        rows.append(
            {
                "case_name": case_name,
                "variant": "rep8_n1024_cess95_mh2_move80",
                "rule": payload["rule"],
                "split_id": payload["split_id"],
                "ref_id": payload["ref_id"],
                "radius": payload["radius"],
                "replicates": payload["replicates"],
                "n_samples_each": payload["n_samples_each"],
                "n_samples_total": payload["n_samples_total"],
                "split_logZ_per_P_diff": payload["split_logZ_per_P_diff"],
                "replicate_logZ_per_P_range": payload["replicate_logZ_per_P_range"],
                "replicate_split_logZ_per_P_diff_max": payload["replicate_split_logZ_per_P_diff_max"],
                "ess_fraction_min": payload["ess_fraction_min"],
                "ess_fraction_mean": payload["ess_fraction_mean"],
                "smc_step_count_max": payload["smc_step_count_max"],
                "smc_min_cess_fraction": payload["smc_min_cess_fraction"],
                "smc_mean_mh_acceptance": payload["smc_mean_mh_acceptance"],
                "elapsed_s": payload["elapsed_s"],
                "reused": payload["reused"],
            }
        )
    df = pd.DataFrame(rows)
    df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= float(cfg["qc"]["max_split_logZ_per_P_diff"])
    df["ess_gate_pass"] = df["ess_fraction_min"] >= float(cfg["qc"]["q05_ess_fraction_min"])
    df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"]
    failed = df[~df["pilot_pass"]]
    status = "pass" if failed.empty else "needs_stronger_stability"
    next_action = (
        "Estimate runtime for an adaptive final sampler that uses the best rep8/CESS95 preset only on hard radii."
        if failed.empty
        else "Best scanned preset does not pass the broad representative pilot; do not resume final sampling."
    )
    write_csv(out_dir / "broad_stability_summary.csv", df)
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_stability_best_broad_pilot",
            "status": status,
            "case_name": case_name,
            "checks": {
                "cases": int(len(df)),
                "passed_cases": int(df["pilot_pass"].sum()),
                "failed_cases": int((~df["pilot_pass"]).sum()),
                "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()),
                "threshold": float(cfg["qc"]["max_split_logZ_per_P_diff"]),
            },
            "failed_cases": failed.to_dict("records"),
            "next_action": next_action,
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Best Broad Stability Pilot

Case: `{case_name}`

Representative cases: {len(df)}

Passed cases: {int(df['pilot_pass'].sum())}

Failed cases: {int((~df['pilot_pass']).sum())}

Next action: {next_action}
""",
    )


def summarize_units(unit_df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    r0 = float(cfg["sampling"]["r0"])
    key = ["split_id", "rule", "ref_id"]
    r0_df = unit_df[unit_df["radius"] == r0][key + ["logZ"]].rename(columns={"logZ": "logZ_r0"})
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined["logZ"] - joined["logZ_r0"]) / P
    summary_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for (rule, radius), sub in joined.groupby(["rule", "radius"]):
        finite_fraction = float(np.mean(np.isfinite(sub["logZ"])))
        q05_ess = float(np.quantile(sub["ess_fraction"], 0.05))
        max_split = float(np.max(sub["split_logZ_per_P_diff"]))
        boot_sd = smoke.bootstrap_sd(sub["delta_phi_energy_unit"].to_numpy(), 39000 + RULES.index(str(rule)) * 1000 + int(round(float(radius) * 100)))
        pass_qc = (
            finite_fraction >= 0.95
            and q05_ess >= float(cfg["qc"]["q05_ess_fraction_min"])
            and max_split <= float(cfg["qc"]["max_split_logZ_per_P_diff"])
            and boot_sd <= float(cfg["qc"]["bootstrap_sd_phi_max"])
        )
        row = {
            "rule": str(rule),
            "radius": float(radius),
            "n_units": int(len(sub)),
            "finite_unit_fraction": finite_fraction,
            "q05_ess_fraction": q05_ess,
            "max_split_logZ_per_P_diff": max_split,
            "bootstrap_sd_phi": boot_sd,
            "mean_logZ": float(np.mean(sub["logZ"])),
            "mean_delta_phi_energy": float(np.mean(sub["delta_phi_energy_unit"])),
            "weighted_ce_mean": float(np.mean(sub["weighted_ce"])),
            "mean_smc_step_count": float(np.mean(sub["smc_step_count"])),
            "min_smc_cess_fraction": float(np.min(sub["smc_min_cess_fraction"])),
            "mean_smc_mh_acceptance": float(np.mean(sub["smc_mean_mh_acceptance"])),
            "qc_pass": bool(pass_qc),
            "claim_status": "claimable_rule_radius" if pass_qc else "no_claim",
        }
        summary_rows.append(row)
        qc_rows.append({k: row[k] for k in ["rule", "radius", "finite_unit_fraction", "q05_ess_fraction", "max_split_logZ_per_P_diff", "bootstrap_sd_phi", "qc_pass", "claim_status"]})
    return pd.DataFrame(summary_rows), pd.DataFrame(qc_rows)


def stage05_sampling_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling") / "pilot_runtime")
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not ref_path.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Reference index missing for final sampling pilot.", observed={"missing": rel(ref_path)})
    ref_df = pd.read_csv(ref_path)
    subset = ref_df.groupby("rule").head(1).reset_index(drop=True)
    pilot_radii = [0.01, 0.05, 0.10, 0.50, 1.00, 1.50, 2.00, 2.50]
    rows = []
    started = time.time()
    for row in subset.to_dict("records"):
        for radius in pilot_radii:
            print(f"[final pilot] rule={row['rule']} split={row['split_id']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            rows.append(sample_unit(row, radius, cfg, n_samples=512, lambda_reg=1.0, force=force))
    pilot_df = pd.DataFrame(rows)
    write_csv(out_dir / "pilot_unit_summary.csv", pilot_df)
    mean_unit = float(pilot_df["elapsed_s"].mean())
    full_units = int(cfg["dataset"]["n_splits"]) * len(RULES) * int(cfg["reference_search"]["selected_refs_per_dataset"]) * len(cfg["sampling"]["radii"])
    estimate_s = mean_unit * 2.0 * full_units
    payload = {
        "pilot_units": int(len(pilot_df)),
        "pilot_elapsed_s": float(time.time() - started),
        "mean_unit_elapsed_s_at_512": mean_unit,
        "full_units": full_units,
        "full_n_samples": 1024,
        "estimated_full_sampling_s": estimate_s,
        "estimated_full_sampling_hours": estimate_s / 3600.0,
        "max_split_logZ_per_P_diff": float(pilot_df["split_logZ_per_P_diff"].max()),
        "q05_ess_fraction": float(np.quantile(pilot_df["ess_fraction"], 0.05)),
        "min_smc_cess_fraction": float(pilot_df["smc_min_cess_fraction"].min()),
    }
    write_json(out_dir / "runtime_estimate.json", payload)
    write_text(
        out_dir / "RUNTIME_ESTIMATE.md",
        f"""# Final Sampling Runtime Estimate

Pilot units: {payload['pilot_units']}

Mean unit elapsed at 512 particles: {mean_unit:.3f} s

Full units: {full_units}

Estimated full 1024-particle sampling: {payload['estimated_full_sampling_hours']:.2f} hours

This estimate is based on adaptive CE-tempered SMC over representative rules and radii.
""",
    )


def stage05_pool2_pm_sais_sampling(*, force: bool = False, max_units: int | None = None, workers: int = 1) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling"))
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not ref_path.exists():
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Final reference index missing.", observed={"missing": rel(ref_path)})
    ref_df = pd.read_csv(ref_path)
    unit_tasks: list[tuple[dict[str, Any], float, dict[str, Any], int, float, bool]] = []
    total = len(ref_df) * len(cfg["sampling"]["radii"])
    count = 0
    started = time.time()
    for row in ref_df.to_dict("records"):
        for radius in cfg["sampling"]["radii"]:
            count += 1
            if max_units is not None and count > max_units:
                break
            unit_tasks.append((row, float(radius), cfg, 1024, 1.0, force))
        if max_units is not None and count > max_units:
            break
    target_units = len(unit_tasks)
    worker_count = max(1, int(workers))
    print(f"[final stage05] prepared {target_units}/{total} units with workers={worker_count}", flush=True)
    unit_rows: list[dict[str, Any]] = []
    if worker_count == 1:
        for idx, (row, radius, task_cfg, n_samples, lambda_reg, task_force) in enumerate(unit_tasks, start=1):
            if idx == 1 or idx % 100 == 0:
                print(f"[final stage05] unit {idx}/{target_units} split={row['split_id']} rule={row['rule']} ref={row['ref_id']} r={float(radius):.2f}", flush=True)
            unit_rows.append(sample_unit(row, radius, task_cfg, n_samples=n_samples, lambda_reg=lambda_reg, force=task_force))
    else:
        chunksize = max(1, min(16, target_units // (worker_count * 8) if target_units else 1))
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for idx, payload in enumerate(executor.map(sample_unit_worker, unit_tasks, chunksize=chunksize), start=1):
                    unit_rows.append(payload)
                    if idx == 1 or idx % 100 == 0 or idx == target_units:
                        elapsed = time.time() - started
                        rate = idx / elapsed if elapsed > 0 else 0.0
                        remaining = (target_units - idx) / rate if rate > 0 else float("nan")
                        print(f"[final stage05] completed {idx}/{target_units} elapsed_h={elapsed / 3600.0:.2f} eta_h={remaining / 3600.0:.2f}", flush=True)
        except Exception as exc:
            raise FinalBlocked(
                "05_pool2_pm_sais_sampling",
                "Parallel PM-SAIS worker failed.",
                observed={
                    "workers": worker_count,
                    "completed_units_before_failure": len(unit_rows),
                    "exception_type": type(exc).__name__,
                    "exception": str(exc)[:2000],
                },
                next_action="Rerun Stage 05 with fewer workers, or set MNIST14_DEVICE=cpu and rerun the same stage.",
            ) from exc
    unit_df = pd.DataFrame(unit_rows)
    if max_units is not None:
        write_csv(out_dir / f"shell_summary_by_unit_partial_{max_units}.csv", unit_df)
        write_json(out_dir / "partial_run_status.json", {"max_units": max_units, "elapsed_s": time.time() - started})
        return
    write_csv(out_dir / "shell_summary_by_unit.csv", unit_df)
    summary_df, qc_df = summarize_units(unit_df, cfg)
    write_csv(out_dir / "shell_summary_by_rule_radius.csv", summary_df)
    write_csv(out_dir / "qc_by_rule_radius.csv", qc_df)
    write_json(out_dir / "selected_lambda.json", {"lambda_reg": 1.0, "selection_rule": "inherits production DNN lambda=1.0 and final pilot stability"})
    write_json(out_dir / "run_config_resolved.json", {**cfg, "elapsed_s": time.time() - started})
    fig_dir = ensure_dir(out_dir / "figures")
    for field, fname in [
        ("q05_ess_fraction", "fig02_sampling_qc_ess_heatmap.png"),
        ("max_split_logZ_per_P_diff", "fig03_sampling_qc_split_logz_heatmap.png"),
        ("mean_smc_step_count", "fig04_smc_step_count_heatmap.png"),
        ("weighted_ce_mean", "fig05_weighted_ce_by_rule_radius.png"),
    ]:
        pivot = summary_df.pivot(index="rule", columns="radius", values=field)
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(pivot.to_numpy(), aspect="auto")
        ticks = list(range(0, len(pivot.columns), 25))
        ax.set_xticks(ticks, [f"{pivot.columns[i]:.2f}" for i in ticks], rotation=45)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(field)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)
    common_pass = sorted(set.intersection(*[set(qc_df[(qc_df["rule"] == rule) & (qc_df["qc_pass"])]["radius"]) for rule in RULES]))
    checks = {
        "unit_rows": int(len(unit_df)),
        "expected_unit_rows": total,
        "rule_radius_rows": int(len(summary_df)),
        "all_logZ_finite": bool(np.isfinite(unit_df["logZ"]).all()),
        "hard_shell_max_abs_err": float(unit_df["hard_shell_distance_max_abs_err"].max()),
        "direction_unit_norm_max_abs_err": float(unit_df["direction_unit_norm_max_abs_err"].max()),
        "common_pass_radii_count": int(len(common_pass)),
        "supported_min": float(min(common_pass)) if common_pass else None,
        "supported_max": float(max(common_pass)) if common_pass else None,
    }
    if checks["unit_rows"] != total or not checks["all_logZ_finite"] or checks["hard_shell_max_abs_err"] > 1.0e-8 or not common_pass:
        raise FinalBlocked("05_pool2_pm_sais_sampling", "Final PM-SAIS QC failed.", observed=checks)
    write_qc("05_pool2_pm_sais_sampling", "pass", checks, warnings=[f"{int((~qc_df['qc_pass']).sum())} rule/radius rows are no_claim."])
    write_text(out_dir / "REPORT.md", f"# Final Stage 05 PM-SAIS\n\nCompleted {len(unit_df)} dense adaptive CE-SMC shell units.")


def stage06_results_figures() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("06_results_figures"))
    stage05 = stage_dir("05_pool2_pm_sais_sampling")
    unit_path = stage05 / "shell_summary_by_unit.csv"
    qc_path = stage05 / "qc_by_rule_radius.csv"
    if not unit_path.exists() or not qc_path.exists():
        raise FinalBlocked("06_results_figures", "Final Stage 05 summaries are missing.", observed={"unit": unit_path.exists(), "qc": qc_path.exists()})
    unit_df = pd.read_csv(unit_path)
    qc_df = pd.read_csv(qc_path)
    r0 = float(cfg["sampling"]["r0"])
    key = ["split_id", "rule", "ref_id"]
    r0_df = unit_df[unit_df["radius"] == r0][key + ["logZ"]].rename(columns={"logZ": "logZ_r0"})
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined["logZ"] - joined["logZ_r0"]) / P
    common_pass = sorted(set.intersection(*[set(qc_df[(qc_df["rule"] == rule) & (qc_df["qc_pass"])]["radius"]) for rule in RULES]))
    phi_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for (rule, radius), sub in joined.groupby(["rule", "radius"]):
        pass_rule_radius = bool(qc_df[(qc_df["rule"] == rule) & (qc_df["radius"] == radius)]["qc_pass"].iloc[0])
        if not pass_rule_radius:
            continue
        values = sub["delta_phi_energy_unit"].to_numpy()
        mean = float(np.mean(values))
        sd = smoke.bootstrap_sd(values, 47000 + RULES.index(str(rule)) * 1000 + int(round(float(radius) * 100)))
        phi_rows.append(
            {
                "rule": str(rule),
                "radius": float(radius),
                "d0": r0,
                "delta_phi_energy": mean,
                "delta_phi_full": float(((P - 1) / P) * math.log(float(radius) / r0) + mean),
                "n_units": int(len(sub)),
                "qc_pass": True,
            }
        )
        boot_rows.append({"rule": str(rule), "radius": float(radius), "delta_phi_energy_mean": mean, "bootstrap_sd": sd, "ci95_low": mean - 1.96 * sd, "ci95_high": mean + 1.96 * sd})
    phi_df = pd.DataFrame(phi_rows)
    boot_df = pd.DataFrame(boot_rows)
    claim_df = pd.DataFrame(
        [
            {
                "radius": float(radius),
                "claim_status": "supported" if float(radius) in common_pass else "no_claim",
                "rules_passed": ";".join(sorted(qc_df[(qc_df["radius"] == radius) & (qc_df["qc_pass"])]["rule"].tolist())),
                "rules_required": ";".join(RULES),
            }
            for radius in sorted(qc_df["radius"].unique())
        ]
    )
    complexity = pd.read_csv(stage_dir("02_complexity_measure") / "complexity_by_rule_summary.csv")
    ref = pd.read_csv(stage_dir("04_exact_reference_search") / "reference_index.csv")
    sampling_summary = pd.read_csv(stage05 / "shell_summary_by_rule_radius.csv")
    joined_summary = sampling_summary.merge(complexity, on="rule", how="left").merge(
        ref.groupby("rule").agg(reference_count=("ref_id", "count"), theta_norm_mean=("theta_norm", "mean"), min_margin_min=("min_margin", "min")).reset_index(),
        on="rule",
        how="left",
    )
    write_csv(out_dir / "phi_by_rule_radius.csv", phi_df)
    write_csv(out_dir / "phi_bootstrap_by_rule_radius.csv", boot_df)
    write_csv(out_dir / "qc_pass_by_rule_radius.csv", qc_df)
    write_csv(out_dir / "complexity_reference_sampling_joined.csv", joined_summary)
    write_csv(out_dir / "final_claim_table.csv", claim_df)
    fig_dir = ensure_dir(out_dir / "figures")
    for field, fname, ylabel in [
        ("delta_phi_energy", "fig04_phi_energy_three_rules_main.png", "Delta phi energy"),
        ("delta_phi_full", "fig05_phi_full_three_rules.png", "Delta phi full"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rule, sub in phi_df.groupby("rule"):
            sub = sub.sort_values("radius")
            ax.plot(sub["radius"], sub[field], linewidth=1.5, label=rule)
        ax.set_xlabel("d_raw")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=180)
        plt.close(fig)
    pivot = qc_df.pivot(index="rule", columns="radius", values="qc_pass").astype(float)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ticks = list(range(0, len(pivot.columns), 25))
    ax.set_xticks(ticks, [f"{pivot.columns[i]:.2f}" for i in ticks], rotation=45)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig07_sampling_qc_pass_heatmap.png", dpi=160)
    plt.close(fig)
    no_claim = [float(r) for r in sorted(set(qc_df["radius"].unique()) - set(common_pass))]
    report = f"""# MNIST14 PM-SAIS Final Report

## Objective

Production-level MNIST14 PM-SAIS run with 10 splits, 20 optimizer-induced exact references per split/rule, and dense d_raw radii 0.01..2.50.

## Supported Radii

Supported d_raw radii: {[float(r) for r in common_pass]}

No-claim d_raw radii: {no_claim}

## Claim Policy

References are optimizer-induced exact references, not exact P_ref^0 samples. Failed radii remain no_claim. Random-label test accuracy is not interpreted as generalization.
"""
    write_text(out_dir / "REPORT.md", report)
    write_json(out_dir / "run_config_resolved.json", {**cfg, "supported_radii": [float(r) for r in common_pass], "no_claim_radii": no_claim})
    checks = {
        "supported_radii_count": int(len(common_pass)),
        "supported_min": float(min(common_pass)) if common_pass else None,
        "supported_max": float(max(common_pass)) if common_pass else None,
        "phi_rows": int(len(phi_df)),
        "main_figure_exists": bool((fig_dir / "fig04_phi_energy_three_rules_main.png").exists()),
    }
    if not common_pass or not checks["main_figure_exists"]:
        raise FinalBlocked("06_results_figures", "Final aggregation QC failed.", observed=checks)
    write_qc("06_results_figures", "pass", checks, warnings=["Final reports only QC-passed radii; no-claim radii are excluded from phi_by_rule_radius.csv."])


def write_run_plan() -> None:
    cfg = load_config()
    write_text(
        RUN_ROOT / "PRODUCTION_RUN_PLAN.md",
        f"""# MNIST14 Final Production Run Plan

## Options and Selected Path

- Direct PM-SAIS was rejected for final because smoke random-label ESS collapsed beyond baseline.
- Final uses the retained 02_dnn/04_sampling production pattern: exact-shell L2 vMF proposal plus adaptive CE-tempered SMC.

## Final Scale

- splits: {cfg['dataset']['n_splits']}
- train size: {cfg['dataset']['n_train']}
- refs per dataset/rule: {cfg['reference_search']['selected_refs_per_dataset']}
- dense radii: 0.01..2.50 step 0.01 ({len(cfg['sampling']['radii'])} radii)
- shell units: {int(cfg['dataset']['n_splits']) * len(RULES) * int(cfg['reference_search']['selected_refs_per_dataset']) * len(cfg['sampling']['radii'])}
- particles per unit: 1024

## Risks

- Random-label exact references may fail under 196-16-16-1; if so, increase attempts first, then evaluate the documented 24-24 backup architecture explicitly.
- Full sampling may be day-scale; per-unit summaries make the run resumable.
- Failed radii stay no_claim and are excluded from final phi curves.

## Rollback

All final artifacts are isolated under `02_dnn/08_mnist/runs/final`; old retained 02_dnn/05 outputs are read-only references and are not modified.
""",
    )


def run_stage(stage: str, *, force: bool = False, max_units: int | None = None, workers: int = 1) -> None:
    write_run_plan()
    if stage == "01_dataset_prepare":
        stage01_dataset_prepare(force=force)
    elif stage == "02_complexity_measure":
        stage02_complexity_measure(force=force)
    elif stage == "03_pool_design":
        stage03_pool_design()
    elif stage == "04_exact_reference_search":
        stage04_exact_reference_search(force=force)
    elif stage == "05_sampling_pilot":
        stage05_sampling_pilot(force=force)
    elif stage == "05_stability_pilot":
        stage05_stability_pilot(force=force)
    elif stage == "05_stability_escalated_pilot":
        stage05_stability_escalated_pilot(force=force)
    elif stage == "05_stability_rep16_pilot":
        stage05_stability_rep16_pilot(force=force)
    elif stage == "05_stability_kernel_scan":
        stage05_stability_kernel_scan(force=force)
    elif stage == "05_stability_best_broad_pilot":
        stage05_stability_best_broad_pilot(force=force)
    elif stage == "05_pool2_pm_sais_sampling":
        stage05_pool2_pm_sais_sampling(force=force, max_units=max_units, workers=workers)
    elif stage == "06_results_figures":
        stage06_results_figures()
    elif stage == "all":
        for item in STAGES:
            run_stage(item, force=force, max_units=max_units if item == "05_pool2_pm_sais_sampling" else None, workers=workers)
    else:
        raise ValueError(f"unknown stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES + ["05_sampling_pilot", "05_stability_pilot", "05_stability_escalated_pilot", "05_stability_rep16_pilot", "05_stability_kernel_scan", "05_stability_best_broad_pilot", "all"], required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        run_stage(args.stage, force=bool(args.force), max_units=args.max_units, workers=args.workers)
    except FinalBlocked as blocked:
        write_blocked(blocked)
        print(f"BLOCKED {blocked.stage}: {blocked.reason}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
