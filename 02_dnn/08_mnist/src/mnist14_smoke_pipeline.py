from __future__ import annotations

import argparse
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
from sklearn.neighbors import NearestNeighbors
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[2]
RUN_ROOT = ROOT / "runs" / "smoke"
CONFIG_PATH = ROOT / "templates" / "config_smoke.yaml"
RULES = ["real_even_odd", "teacher_nn", "random_label"]
STAGE_NAMES = [
    "00_repo_audit",
    "01_dataset_prepare",
    "02_complexity_measure",
    "03_pool_design",
    "04_exact_reference_search",
    "05_pool2_pm_sais_sampling",
    "06_results_figures",
]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mnist14_model import ARCH, P, ce_and_error_np, ce_error_batch_torch, init_theta, logits_np, margin_stats_np, normalize_labels
from mnist14_vmf import log_sphere_mgf, sample_vmf


class StageBlocked(RuntimeError):
    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        observed: dict[str, Any] | None = None,
        expected: dict[str, Any] | None = None,
        next_action: str = "Inspect the blocked report and rerun the same stage after correcting the cause.",
    ) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.observed = observed or {}
        self.expected = expected or {}
        self.next_action = next_action


def stage_dir(stage: str) -> Path:
    if stage == "final_report":
        return RUN_ROOT / "final_report"
    return RUN_ROOT / stage


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
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["resolved_at_unix"] = time.time()
    cfg["python"] = sys.executable
    cfg["repo_root"] = str(REPO_ROOT)
    cfg["mnist_root"] = str(ROOT)
    cfg["allow_smoke_download"] = True
    return cfg


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def files_under(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [rel(p) for p in sorted(path.rglob("*")) if p.is_file()]


def write_qc(stage: str, status: str, checks: dict[str, Any], *, warnings: list[str] | None = None, hard_failures: list[str] | None = None) -> None:
    out_dir = stage_dir("final_report" if stage == "06_results_figures" else stage)
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


def write_blocked_report(blocked: StageBlocked) -> None:
    out_dir = stage_dir("final_report" if blocked.stage == "06_results_figures" else blocked.stage)
    ensure_dir(out_dir)
    observed_lines = "\n".join(f"- {k}: {v}" for k, v in blocked.observed.items()) or "- n/a"
    expected_lines = "\n".join(f"- {k}: {v}" for k, v in blocked.expected.items()) or "- n/a"
    files = files_under(out_dir)
    file_lines = "\n".join(f"- {p}" for p in files) or "- none"
    write_text(
        out_dir / "STAGE_BLOCKED.md",
        f"""# STAGE_BLOCKED

Stage: `{blocked.stage}`

## Exact Failing Condition

{blocked.reason}

## Observed Metric

{observed_lines}

## Expected Threshold

{expected_lines}

## Files Already Created

{file_lines}

## Next Safe Action

{blocked.next_action}
""",
    )
    write_qc(blocked.stage, "blocked", {"blocked": True, "reason": blocked.reason}, hard_failures=[blocked.reason])


def run_pytest(test_path: Path, *, timeout_s: int = 300) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", str(test_path), "-q"]
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout_s)
    return {
        "cmd": " ".join(cmd),
        "returncode": int(proc.returncode),
        "elapsed_s": float(time.time() - started),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "passed": proc.returncode == 0,
    }


def write_stage_report(stage: str, title: str, body: str, *, final: bool = False) -> None:
    out_dir = stage_dir("final_report" if final else stage)
    write_text(out_dir / "REPORT.md", f"# {title}\n\n{body}")


def stage00_repo_audit() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("00_repo_audit"))
    for path in [
        ROOT / "src",
        ROOT / "config",
        ROOT / "tests",
        ROOT / "runs" / "smoke",
        ROOT / "runs" / "candidate",
        ROOT / "runs" / "final",
    ]:
        ensure_dir(path)

    dnn_dirs = [p.name for p in sorted((REPO_ROOT / "02_dnn").glob("*")) if p.is_dir()]
    src_files = []
    for stage in ["03_reference_search", "04_sampling", "05_proxy_local_entropy", "06_reference_atlas"]:
        src_files.extend(rel(p) for p in sorted((REPO_ROOT / "02_dnn" / stage / "src").glob("*.py")))
    reuse_map = {
        "dataset": "New MNIST loader needed; prior 01_dataset_gen is synthetic and not modified.",
        "model_flattening": "Copied/adapted small 3-layer flatten/unflatten convention into 02_dnn/08_mnist/src/mnist14_model.py.",
        "pm_sais_vmf": "Copied/adapted vMF sampling and log sphere MGF utility into 02_dnn/08_mnist/src/mnist14_vmf.py.",
        "aggregation_plotting": "Use local pandas/matplotlib aggregation under 02_dnn/08_mnist; do not alter 05_proxy_local_entropy outputs.",
        "inspected_files": src_files,
    }
    tree_lines = []
    for base, dirs, files in os.walk(ROOT):
        base_path = Path(base)
        if any(part in {".pytest_cache", "__pycache__"} for part in base_path.parts):
            continue
        depth = len(base_path.relative_to(ROOT).parts)
        if depth > 4:
            dirs[:] = []
            continue
        indent = "  " * depth
        tree_lines.append(f"{indent}{base_path.name}/")
        for file in sorted(files):
            tree_lines.append(f"{indent}  {file}")

    write_text(
        out_dir / "AUDIT_REPORT.md",
        "\n".join(
            [
                "# Audit Report",
                "",
                "Inspected the active DNN stage layout without modifying retained outputs.",
                "",
                "## Existing 02_dnn Directories",
                *[f"- {name}" for name in dnn_dirs],
                "",
                "## Scope Decision",
                "All implementation and smoke outputs remain under `02_dnn/08_mnist`.",
            ]
        ),
    )
    write_text(out_dir / "REUSE_MAP.md", "# Reuse Map\n\n```json\n" + json.dumps(reuse_map, indent=2) + "\n```")
    write_text(out_dir / "DIRECTORY_TREE.md", "# Directory Tree\n\n```text\n" + "\n".join(tree_lines) + "\n```")
    write_json(out_dir / "run_config_resolved.json", cfg)
    checks = {
        "local_dirs_exist": all((ROOT / name).exists() for name in ["src", "config", "tests", "runs"]),
        "retained_outputs_modified": False,
        "audit_files_exist": True,
        "next_prompt": "02_dnn/08_mnist/stages/01_dataset_prepare/START_PROMPT.md",
    }
    write_qc("00_repo_audit", "pass", checks)
    write_stage_report(
        "00_repo_audit",
        "Stage 00 Repo Audit",
        "Files created: audit report, reuse map, directory tree, run config, QC status.\n\n"
        "Files modified: local 08_mnist skeleton directories only.\n\n"
        "QC summary: pass; retained production outputs were not modified.\n\n"
        "Blocking issues: none.\n\n"
        "Next command: `.venv/Scripts/python.exe 02_dnn/08_mnist/src/mnist14_smoke_pipeline.py --stage 01_dataset_prepare`",
    )


def load_or_fetch_mnist() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data_dir = ensure_dir(ROOT / "data" / "mnist")
    cache_path = data_dir / "mnist_openml_uint8.npz"
    metadata: dict[str, Any] = {
        "cache_path": rel(cache_path),
        "source": "openml_mnist_784",
        "download_allowed_for_smoke": True,
        "download_performed": False,
    }
    if cache_path.exists():
        payload = np.load(cache_path)
        metadata["source_status"] = "local_cache"
        return payload["X"], payload["y"], metadata
    try:
        fetched = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto", data_home=str(data_dir / "openml"))
    except Exception as exc:
        raise StageBlocked(
            "01_dataset_prepare",
            "MNIST data are not available locally and OpenML fetch failed.",
            observed={"exception": repr(exc), "cache_path": rel(cache_path)},
            expected={"mnist_source": "local data/mnist or successful smoke download"},
            next_action="Place MNIST 784 data in the local cache or restore network/OpenML access, then rerun Stage 01.",
        ) from exc
    x = np.asarray(fetched.data, dtype=np.uint8).reshape(-1, 784)
    y = np.asarray(fetched.target, dtype=np.int16).reshape(-1)
    np.savez_compressed(cache_path, X=x, y=y)
    metadata["download_performed"] = True
    metadata["source_status"] = "downloaded_for_smoke"
    return x, y, metadata


def avgpool_14(x784: np.ndarray) -> np.ndarray:
    x = np.asarray(x784, dtype=np.float32).reshape(-1, 28, 28) / 255.0
    pooled = x.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    return pooled.reshape(-1, 196).astype(np.float32)


def teacher_logits(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    w1 = rng.normal(0.0, 1.0 / math.sqrt(196), size=(32, 196))
    b1 = rng.normal(0.0, 0.05, size=32)
    w2 = rng.normal(0.0, 1.0 / math.sqrt(32), size=(32, 32))
    b2 = rng.normal(0.0, 0.05, size=32)
    w3 = rng.normal(0.0, 1.0 / math.sqrt(32), size=(32,))
    h1 = np.tanh(np.asarray(x, dtype=np.float64) @ w1.T + b1)
    h2 = np.tanh(h1 @ w2.T + b2)
    return h2 @ w3


def balanced_pm1(n: int, seed: int) -> np.ndarray:
    if n % 2 != 0:
        raise ValueError("balanced_pm1 requires even n")
    y = np.concatenate([np.ones(n // 2, dtype=np.int8), -np.ones(n // 2, dtype=np.int8)])
    rng = np.random.default_rng(int(seed))
    rng.shuffle(y)
    return y


def plot_dataset_figures(out_dir: Path, raw28: np.ndarray, split_rows: list[dict[str, Any]], label_df: pd.DataFrame) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    first = split_rows[0]
    train_idx = np.asarray(first["train_indices"], dtype=np.int64)
    sample_idx = train_idx[:10]
    raw14 = avgpool_14(raw28[sample_idx]).reshape(-1, 14, 14)
    raw28_img = raw28[sample_idx].reshape(-1, 28, 28)
    fig, axes = plt.subplots(2, len(sample_idx), figsize=(12, 2.6))
    for j in range(len(sample_idx)):
        axes[0, j].imshow(raw28_img[j], cmap="gray")
        axes[0, j].axis("off")
        axes[1, j].imshow(raw14[j], cmap="gray")
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("28x28")
    axes[1, 0].set_ylabel("14x14")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_mnist_28_vs_14_montage.png", dpi=160)
    plt.close(fig)

    pivot = label_df.pivot_table(index="rule", columns="split_id", values="train_pos_fraction")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    pivot.T.plot(kind="bar", ax=ax)
    ax.axhline(0.45, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.55, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Train +1 fraction")
    ax.set_xlabel("Split")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_label_balance_by_rule.png", dpi=160)
    plt.close(fig)


def stage01_dataset_prepare() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("01_dataset_prepare"))
    raw28, digits, source_meta = load_or_fetch_mnist()
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
        x_train_raw = avgpool_14(raw28[train_idx])
        x_test_raw = avgpool_14(raw28[test_idx])
        mean = x_train_raw.mean(axis=0, keepdims=True)
        std = x_train_raw.std(axis=0, keepdims=True)
        std = np.where(std < 1.0e-6, 1.0, std)
        x_train = ((x_train_raw - mean) / std).astype(np.float32)
        x_test = ((x_test_raw - mean) / std).astype(np.float32)
        digit_train = digits[train_idx].astype(np.int16)
        digit_test = digits[test_idx].astype(np.int16)
        split_rows.append(
            {
                "split_id": split_id,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "n_train": n_train,
                "n_test": n_test,
                "train_even_fraction": float(np.mean((digit_train % 2) == 0)),
                "test_even_fraction": float(np.mean((digit_test % 2) == 0)),
                "standardization_mean_mean": float(np.mean(mean)),
                "standardization_std_mean": float(np.mean(std)),
            }
        )
        teacher_train = teacher_logits(x_train, 31001 + split_id)
        teacher_test = teacher_logits(x_test, 31001 + split_id)
        threshold = float(np.median(teacher_train))
        labels = {
            "real_even_odd": (np.where((digit_train % 2) == 0, 1, -1).astype(np.int8), np.where((digit_test % 2) == 0, 1, -1).astype(np.int8), {"definition": "even digit +1, odd digit -1"}),
            "teacher_nn": (np.where(teacher_train >= threshold, 1, -1).astype(np.int8), np.where(teacher_test >= threshold, 1, -1).astype(np.int8), {"teacher_seed": 31001 + split_id, "train_median_logit_threshold": threshold}),
            "random_label": (balanced_pm1(n_train, 41001 + split_id), balanced_pm1(n_test, 42001 + split_id), {"train_seed": 41001 + split_id, "test_seed": 42001 + split_id, "test_accuracy_note": "independent random test labels are not a generalization metric"}),
        }
        for rule, (y_train, y_test, metadata) in labels.items():
            ds_dir = ensure_dir(out_dir / "raw_datasets" / f"split_{split_id:03d}" / rule)
            dataset_path = ds_dir / "dataset.npz"
            np.savez_compressed(
                dataset_path,
                X_train=x_train.astype(np.float32),
                y_train=y_train.astype(np.int8),
                X_test=x_test.astype(np.float32),
                y_test=y_test.astype(np.int8),
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
            pos = float(np.mean(y_train == 1))
            label_rows.append(
                {
                    "split_id": split_id,
                    "rule": rule,
                    "train_pos_fraction": pos,
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
                    "mode": "smoke",
                    "split_id": split_id,
                    "rule": rule,
                    "dataset_path": rel(dataset_path),
                    "n_train": n_train,
                    "n_test": n_test,
                    "input_dim": 196,
                    "train_pos_fraction": pos,
                }
            )
    dataset_df = pd.DataFrame(dataset_rows)
    label_df = pd.DataFrame(label_rows)
    meta_dir = ensure_dir(out_dir / "metadata")
    dataset_df.to_csv(out_dir / "dataset_index.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k not in {"train_indices", "test_indices"}} for row in split_rows]).to_csv(meta_dir / "split_summary.csv", index=False)
    label_df.to_csv(meta_dir / "label_balance_summary.csv", index=False)
    write_json(meta_dir / "mnist_source.json", source_meta)
    plot_dataset_figures(out_dir, raw28, split_rows, label_df)
    write_json(out_dir / "run_config_resolved.json", cfg)

    checks = {
        "dataset_index_rows": int(len(dataset_df)),
        "all_npz_exist": bool(all((REPO_ROOT / row["dataset_path"]).exists() for row in dataset_rows)),
        "balance_min": float(label_df["train_pos_fraction"].min()),
        "balance_max": float(label_df["train_pos_fraction"].max()),
        "montage_exists": bool((out_dir / "figures" / "fig01_mnist_28_vs_14_montage.png").exists()),
        "download_performed": bool(source_meta["download_performed"]),
    }
    if len(dataset_df) != 9 or checks["balance_min"] < 0.45 or checks["balance_max"] > 0.55:
        raise StageBlocked(
            "01_dataset_prepare",
            "Dataset QC failed.",
            observed=checks,
            expected={"dataset_index_rows": 9, "train_balance": "0.45 <= p(+1) <= 0.55"},
            next_action="Inspect label construction and rerun Stage 01 before downstream stages.",
        )
    pytest_result = run_pytest(ROOT / "tests" / "test_stage01_dataset_prepare.py")
    checks["pytest"] = pytest_result
    if not pytest_result["passed"]:
        raise StageBlocked(
            "01_dataset_prepare",
            "Stage 01 pytest failed.",
            observed=pytest_result,
            expected={"pytest_returncode": 0},
            next_action="Fix the dataset payload or test failure, then rerun Stage 01.",
        )
    write_qc("01_dataset_prepare", "pass", checks, warnings=["MNIST was downloaded for smoke and cached locally."] if source_meta["download_performed"] else [])
    write_stage_report(
        "01_dataset_prepare",
        "Stage 01 Dataset Prepare",
        f"Files created: dataset index, 9 NPZ datasets, metadata summaries, and two figures under `{rel(out_dir)}`.\n\n"
        "Files modified: local 08_mnist smoke outputs only.\n\n"
        f"QC summary: pass; train label balance range {checks['balance_min']:.3f} to {checks['balance_max']:.3f}; pytest passed.\n\n"
        "Blocking issues: none.\n\n"
        "Next command: `.venv/Scripts/python.exe 02_dnn/08_mnist/src/mnist14_smoke_pipeline.py --stage 02_complexity_measure`",
    )


def load_dataset_npz(path_value: str | Path) -> dict[str, np.ndarray]:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    payload = np.load(path)
    return {k: payload[k] for k in payload.files}


def graph_tv_nmstv(x: np.ndarray, y: np.ndarray, k: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = normalize_labels(y)
    nn = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean")
    nn.fit(x)
    distances, indices = nn.kneighbors(x, return_distance=True)
    d = distances[:, 1:]
    j = indices[:, 1:]
    nonzero = d[d > 0.0]
    sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    edge_weight: dict[tuple[int, int], float] = {}
    for i in range(x.shape[0]):
        for dist, jj in zip(d[i], j[i]):
            a, b = sorted((int(i), int(jj)))
            w = float(math.exp(-(float(dist) ** 2) / (2.0 * sigma * sigma)))
            edge_weight[(a, b)] = max(edge_weight.get((a, b), 0.0), w)
    total_w = float(sum(edge_weight.values()))
    cut_w = float(sum(w for (a, b), w in edge_weight.items() if y[a] != y[b]))
    tv = cut_w / max(total_w, 1.0e-300)
    p = float(np.mean(y == 1.0))
    baseline = 2.0 * p * (1.0 - p)
    return {
        "k": int(k),
        "edge_count": int(len(edge_weight)),
        "sigma_k": sigma,
        "tv": float(tv),
        "baseline": float(baseline),
        "nmstv": float(tv / max(baseline, 1.0e-12)),
    }


def stage02_complexity_measure() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("02_complexity_measure"))
    index_path = stage_dir("01_dataset_prepare") / "dataset_index.csv"
    if not index_path.exists():
        raise StageBlocked("02_complexity_measure", "Stage 01 dataset index is missing.", observed={"missing": rel(index_path)}, expected={"file_exists": True})
    index_df = pd.read_csv(index_path)
    graph_rows = []
    dataset_rows = []
    for row in index_df.to_dict("records"):
        ds = load_dataset_npz(row["dataset_path"])
        per_k = []
        for k in [8, 16, 32]:
            metrics = graph_tv_nmstv(ds["X_train"], ds["y_train"], k)
            graph_rows.append({**row, **metrics})
            per_k.append(metrics)
        dataset_rows.append(
            {
                **row,
                "tv_mean": float(np.mean([m["tv"] for m in per_k])),
                "nmstv_mean": float(np.mean([m["nmstv"] for m in per_k])),
                "nmstv_min": float(np.min([m["nmstv"] for m in per_k])),
                "nmstv_max": float(np.max([m["nmstv"] for m in per_k])),
                "edge_count_min": int(np.min([m["edge_count"] for m in per_k])),
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
    dataset_df.to_csv(out_dir / "complexity_by_dataset.csv", index=False)
    summary_df.to_csv(out_dir / "complexity_by_rule_summary.csv", index=False)
    graph_df.to_csv(out_dir / "graph_stats_by_dataset_k.csv", index=False)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(6, 4))
    dataset_df.boxplot(column="nmstv_mean", by="rule", ax=ax)
    ax.set_title("")
    fig.suptitle("")
    ax.set_ylabel("Mean NMSTV over k")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_nmstv_by_rule_boxplot.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule, sub in graph_df.groupby("rule"):
        grouped = sub.groupby("k")["tv"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", label=rule)
    ax.set_xlabel("k")
    ax.set_ylabel("TV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_tv_by_k_rule.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule, sub in dataset_df.groupby("rule"):
        ax.scatter(sub["train_pos_fraction"], sub["nmstv_mean"], label=rule)
    ax.set_xlabel("Train +1 fraction")
    ax.set_ylabel("Mean NMSTV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_complexity_vs_label_balance.png", dpi=160)
    plt.close(fig)
    write_json(out_dir / "run_config_resolved.json", cfg)
    checks = {
        "dataset_rows": int(len(dataset_df)),
        "rule_summary_rows": int(len(summary_df)),
        "graph_rows": int(len(graph_df)),
        "edge_count_min": int(graph_df["edge_count"].min()),
        "all_finite": bool(np.isfinite(graph_df[["tv", "nmstv", "sigma_k"]].to_numpy()).all()),
    }
    if checks["dataset_rows"] != 9 or checks["rule_summary_rows"] != 3 or checks["edge_count_min"] <= 0 or not checks["all_finite"]:
        raise StageBlocked(
            "02_complexity_measure",
            "Complexity QC failed.",
            observed=checks,
            expected={"dataset_rows": 9, "rule_summary_rows": 3, "edge_count_min": "> 0", "finite_metrics": True},
            next_action="Inspect kNN graph construction and Stage 01 datasets, then rerun Stage 02.",
        )
    pytest_result = run_pytest(ROOT / "tests" / "test_stage02_complexity_measure.py")
    checks["pytest"] = pytest_result
    if not pytest_result["passed"]:
        raise StageBlocked("02_complexity_measure", "Stage 02 pytest failed.", observed=pytest_result, expected={"pytest_returncode": 0})
    write_qc("02_complexity_measure", "pass", checks)
    write_stage_report(
        "02_complexity_measure",
        "Stage 02 Complexity Measure",
        f"Files created: complexity tables and three figures under `{rel(out_dir)}`.\n\n"
        "Files modified: local 08_mnist smoke outputs only.\n\n"
        f"QC summary: pass; {checks['graph_rows']} graph rows finite and min edge count {checks['edge_count_min']}.\n\n"
        "Blocking issues: none.\n\n"
        "Next command: `.venv/Scripts/python.exe 02_dnn/08_mnist/src/mnist14_smoke_pipeline.py --stage 03_pool_design`",
    )


def stage03_pool_design() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("03_pool_design"))
    contract = {
        "experiment_id": cfg["experiment_id"],
        "architecture": {
            "name": "196-16-16-1-tanh",
            "input_dim": ARCH.input_dim,
            "hidden_width": ARCH.hidden_width,
            "hidden_layers": ARCH.hidden_layers,
            "activation": ARCH.activation,
            "P": P,
            "flatten_order": ["W1", "b1", "W2", "b2", "W3", "b3"],
        },
        "loss": {
            "ce_mean": "mean_i log(1 + exp(-y_i f_theta(x_i)))",
            "ce_sum": "n_train * ce_mean",
            "gamma_ce": "beta * n_train",
            "beta": float(cfg["loss"]["beta"]),
        },
        "pool1": {
            "law": "optimizer-induced exact reference ensemble",
            "acceptance": "train_error == 0",
            "caveat": "Do not claim exact sampling from P_ref^0.",
            "selected_refs_per_dataset": int(cfg["reference_search"]["selected_refs_per_dataset"]),
            "max_attempts_per_dataset": int(cfg["reference_search"]["max_attempts_per_dataset"]),
        },
        "pool2": {
            "main_estimator": "PM-SAIS H=infinity",
            "hard_shell": "theta = theta_ref + sqrt(P) * d * u",
            "radii": cfg["sampling"]["radii"],
            "r0": float(cfg["sampling"]["r0"]),
            "lambda_reg_candidates": cfg["loss"]["lambda_reg_candidates"],
            "optional_H_ladder": cfg["sampling"]["optional_H_ladder"],
        },
        "qc": cfg["qc"],
        "failed_radius_policy": "no_claim",
    }
    write_json(out_dir / "POOL_CONTRACT.json", contract)
    write_text(
        out_dir / "POOL_CONTRACT.md",
        "# Pool Contract\n\n"
        "Pool 1 is an optimizer-induced exact reference ensemble with `train_error == 0`.\n\n"
        "Pool 2 uses PM-SAIS on hard raw-distance shells with one common lambda across rules.\n\n"
        "Failed radii are marked `no_claim`; references are not claimed to be exact `P_ref^0` samples.",
    )
    write_text(
        out_dir / "MODEL_SPEC.md",
        f"# Model Spec\n\nArchitecture: `196-16-16-1-tanh`\n\nParameter count: `{P}`\n\nFlatten order: `W1, b1, W2, b2, W3, b3`.\n",
    )
    write_text(
        out_dir / "QC_GATES.md",
        "# QC Gates\n\n"
        "- Dataset finite and balanced.\n"
        "- References have train error 0, P=3441, and no duplicate theta.\n"
        "- PM-SAIS finite fraction >= 0.90, q05 ESS fraction >= 0.02, split logZ/P <= 0.004, bootstrap sd phi <= 0.012 for claimed radii.\n",
    )
    write_json(out_dir / "run_config_resolved.json", cfg)
    checks = {"P": P, "P_expected": 3441, "contract_exists": True}
    pytest_result = run_pytest(ROOT / "tests" / "test_stage03_model_spec.py")
    checks["pytest"] = pytest_result
    if P != 3441 or not pytest_result["passed"]:
        raise StageBlocked(
            "03_pool_design",
            "Pool design/model spec QC failed.",
            observed=checks,
            expected={"P": 3441, "pytest_returncode": 0},
            next_action="Fix model spec utilities, then rerun Stage 03.",
        )
    write_qc("03_pool_design", "pass", checks)
    write_stage_report(
        "03_pool_design",
        "Stage 03 Pool Design",
        f"Files created: pool contract, model spec, QC gates, run config, and QC status under `{rel(out_dir)}`.\n\n"
        "Files modified: local 08_mnist smoke outputs and src model utilities only.\n\n"
        "QC summary: pass; P=3441 and model tests passed.\n\n"
        "Blocking issues: none.\n\n"
        "Next command: `.venv/Scripts/python.exe 02_dnn/08_mnist/src/mnist14_smoke_pipeline.py --stage 04_exact_reference_search`",
    )


def torch_logits_batch(theta: Any, x_t: Any) -> Any:
    import torch

    h = ARCH.hidden_width
    d = ARCH.input_dim
    idx = 0
    w1 = theta[:, idx : idx + d * h].reshape(theta.shape[0], h, d)
    idx += d * h
    b1 = theta[:, idx : idx + h]
    idx += h
    w2 = theta[:, idx : idx + h * h].reshape(theta.shape[0], h, h)
    idx += h * h
    b2 = theta[:, idx : idx + h]
    idx += h
    w3 = theta[:, idx : idx + h].reshape(theta.shape[0], 1, h)
    idx += h
    b3 = theta[:, idx : idx + 1].reshape(theta.shape[0])
    h1 = torch.tanh(torch.einsum("nd,bhd->bnh", x_t, w1) + b1[:, None, :])
    h2 = torch.tanh(torch.einsum("bnh,bkh->bnk", h1, w2) + b2[:, None, :])
    return torch.einsum("bnh,bh->bn", h2, w3[:, 0, :]) + b3[:, None]


def train_attempt_batch(x: np.ndarray, y: np.ndarray, seeds: list[int], *, max_epochs: int = 3200, lr: float = 0.025) -> list[dict[str, Any]]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    theta0 = np.vstack([init_theta(seed, scale_multiplier=1.0 + 0.5 * ((seed % 5) / 4.0)) for seed in seeds]).astype(np.float32)
    theta = torch.tensor(theta0, device=device, dtype=dtype, requires_grad=True)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device, dtype=dtype)
    y_t = torch.as_tensor(normalize_labels(y).astype(np.float32), device=device, dtype=dtype)
    opt = torch.optim.Adam([theta], lr=float(lr))
    solved: dict[int, dict[str, Any]] = {}
    best: dict[int, tuple[float, np.ndarray, float]] = {}
    milestones = {int(max_epochs * 0.55), int(max_epochs * 0.80)}
    for epoch in range(1, int(max_epochs) + 1):
        opt.zero_grad(set_to_none=True)
        logits = torch_logits_batch(theta, x_t)
        yz = logits * y_t[None, :]
        loss_rows = torch.nn.functional.softplus(-yz).mean(dim=1)
        loss = loss_rows.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 1000.0)
        opt.step()
        if epoch in milestones:
            for group in opt.param_groups:
                group["lr"] *= 0.35
        if epoch == 1 or epoch % 25 == 0 or epoch == max_epochs:
            with torch.no_grad():
                logits_eval = torch_logits_batch(theta, x_t)
                yz_eval = logits_eval * y_t[None, :]
                err = torch.mean((yz_eval <= 0.0).to(dtype), dim=1).detach().cpu().numpy()
                ce = torch.nn.functional.softplus(-yz_eval).mean(dim=1).detach().cpu().numpy()
                theta_np = theta.detach().cpu().numpy().astype(np.float64)
                for idx, seed in enumerate(seeds):
                    if seed not in best or float(err[idx]) < best[seed][0]:
                        best[seed] = (float(err[idx]), theta_np[idx].copy(), float(ce[idx]))
                    if float(err[idx]) == 0.0 and seed not in solved:
                        solved[seed] = {"seed": seed, "theta": theta_np[idx].copy(), "train_error": 0.0, "ce_mean_train": float(ce[idx]), "epoch": int(epoch), "phase": "adam"}
            if len(solved) == len(seeds):
                break
    out: list[dict[str, Any]] = []
    for seed in seeds:
        if seed in solved:
            out.append(solved[seed])
        else:
            err, theta_best, ce = best[seed]
            out.append({"seed": seed, "theta": theta_best, "train_error": float(err), "ce_mean_train": float(ce), "epoch": int(max_epochs), "phase": "adam_best"})
    return out


def polish_theta_lbfgs(x: np.ndarray, y: np.ndarray, theta_start: np.ndarray, *, max_iter: int = 220) -> dict[str, Any]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float64), device=device, dtype=dtype)
    y_t = torch.as_tensor(normalize_labels(y).astype(np.float64), device=device, dtype=dtype)
    theta = torch.tensor(np.asarray(theta_start, dtype=np.float64).reshape(1, -1), device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=int(max_iter), line_search_fn="strong_wolfe", tolerance_grad=1.0e-8, tolerance_change=1.0e-10)

    def closure() -> Any:
        opt.zero_grad(set_to_none=True)
        logits = torch_logits_batch(theta, x_t)
        loss = torch.nn.functional.softplus(-(logits[0] * y_t)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits = torch_logits_batch(theta, x_t)
        yz = logits[0] * y_t
        ce = torch.nn.functional.softplus(-yz).mean().item()
        err = torch.mean((yz <= 0.0).to(dtype)).item()
    return {"theta": theta.detach().cpu().numpy().reshape(-1).astype(np.float64), "train_error": float(err), "ce_mean_train": float(ce)}


def select_reference(selected: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
    theta = np.asarray(candidate["theta"], dtype=np.float64).reshape(-1)
    if theta.size != P or float(candidate["train_error"]) != 0.0:
        return False
    for row in selected:
        if float(np.linalg.norm(theta - np.asarray(row["theta"], dtype=np.float64).reshape(-1))) <= 1.0e-6:
            return False
    selected.append(candidate)
    return True


def stage04_exact_reference_search() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("04_exact_reference_search"))
    index_path = stage_dir("01_dataset_prepare") / "dataset_index.csv"
    contract_path = stage_dir("03_pool_design") / "POOL_CONTRACT.json"
    if not index_path.exists() or not contract_path.exists():
        raise StageBlocked("04_exact_reference_search", "Required Stage 01 or Stage 03 input is missing.", observed={"dataset_index": index_path.exists(), "pool_contract": contract_path.exists()})
    index_df = pd.read_csv(index_path)
    target_refs = int(cfg["reference_search"]["selected_refs_per_dataset"])
    max_attempts = int(cfg["reference_search"]["max_attempts_per_dataset"])
    reference_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    started = time.time()
    for dataset_id, row in enumerate(index_df.to_dict("records")):
        print(f"[stage04] dataset {dataset_id + 1}/{len(index_df)} split={row['split_id']} rule={row['rule']}", flush=True)
        ds = load_dataset_npz(row["dataset_path"])
        x_train = ds["X_train"]
        y_train = ds["y_train"]
        x_test = ds["X_test"]
        y_test = ds["y_test"]
        selected: list[dict[str, Any]] = []
        seed_base = 700000 + 10000 * int(row["split_id"]) + 1000 * RULES.index(row["rule"])
        attempts_used = 0
        best_unsolved: list[dict[str, Any]] = []
        while attempts_used < max_attempts and len(selected) < target_refs:
            batch_n = min(12, max_attempts - attempts_used)
            seeds = [seed_base + attempts_used + i for i in range(batch_n)]
            batch = train_attempt_batch(x_train, y_train, seeds)
            attempts_used += batch_n
            for result in batch:
                theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
                ce_train, err_train = ce_and_error_np(theta, x_train, y_train)
                ce_test, err_test = ce_and_error_np(theta, x_test, y_test)
                attempt_row = {
                    "dataset_id": dataset_id,
                    "split_id": int(row["split_id"]),
                    "rule": row["rule"],
                    "attempt_seed": int(result["seed"]),
                    "attempt_id": int(attempts_used - batch_n + seeds.index(result["seed"])),
                    "phase": result["phase"],
                    "epoch": int(result["epoch"]),
                    "train_error": float(err_train),
                    "test_error": float(err_test),
                    "ce_mean_train": float(ce_train),
                    "ce_mean_test": float(ce_test),
                    "theta_norm": float(np.linalg.norm(theta)),
                    "selected": False,
                }
                if err_train == 0.0:
                    candidate = {**result, "theta": theta, "ce_mean_train": ce_train, "ce_mean_test": ce_test, "test_error": err_test, "attempt_seed": int(result["seed"]), "phase": result["phase"]}
                    if select_reference(selected, candidate):
                        attempt_row["selected"] = True
                else:
                    best_unsolved.append({**result, "theta": theta, "train_error": err_train, "ce_mean_train": ce_train, "test_error": err_test, "ce_mean_test": ce_test})
                attempt_rows.append(attempt_row)
            print(f"[stage04] split={row['split_id']} rule={row['rule']} attempts={attempts_used} selected={len(selected)}/{target_refs}", flush=True)
            if len(selected) < target_refs and attempts_used >= max_attempts and best_unsolved:
                best_unsolved = sorted(best_unsolved, key=lambda r: (float(r["train_error"]), float(r["ce_mean_train"])))[: min(10, len(best_unsolved))]
                for result in best_unsolved:
                    if len(selected) >= target_refs:
                        break
                    polished = polish_theta_lbfgs(x_train, y_train, result["theta"])
                    theta = polished["theta"]
                    ce_train, err_train = ce_and_error_np(theta, x_train, y_train)
                    ce_test, err_test = ce_and_error_np(theta, x_test, y_test)
                    attempt_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "split_id": int(row["split_id"]),
                            "rule": row["rule"],
                            "attempt_seed": int(result["seed"]),
                            "attempt_id": int(result.get("seed", 0)),
                            "phase": "lbfgs_polish",
                            "epoch": int(result["epoch"]),
                            "train_error": float(err_train),
                            "test_error": float(err_test),
                            "ce_mean_train": float(ce_train),
                            "ce_mean_test": float(ce_test),
                            "theta_norm": float(np.linalg.norm(theta)),
                            "selected": bool(err_train == 0.0),
                        }
                    )
                    if err_train == 0.0:
                        candidate = {**result, "theta": theta, "train_error": err_train, "ce_mean_train": ce_train, "ce_mean_test": ce_test, "test_error": err_test, "attempt_seed": int(result["seed"]), "phase": "lbfgs_polish"}
                        select_reference(selected, candidate)
        if len(selected) < target_refs:
            pd.DataFrame(attempt_rows).to_csv(ensure_dir(out_dir / "attempt_logs") / "attempts.csv", index=False)
            raise StageBlocked(
                "04_exact_reference_search",
                "Insufficient exact references for a dataset/rule.",
                observed={"split_id": row["split_id"], "rule": row["rule"], "selected_refs": len(selected), "target_refs": target_refs, "attempts_used": attempts_used},
                expected={"selected_refs_per_dataset": target_refs, "train_error": 0},
                next_action="Increase max_attempts_per_dataset for the same 196-16-16-1 architecture; if random_label still fails, run the documented backup-architecture decision explicitly before retrying.",
            )
        for ref_id, result in enumerate(selected[:target_refs]):
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ref_dir = ensure_dir(out_dir / "selected_reference_pool" / f"split_{int(row['split_id']):03d}" / row["rule"] / f"ref_{ref_id:03d}")
            theta_path = ref_dir / "theta.npy"
            np.save(theta_path, theta)
            ce_train, err_train = ce_and_error_np(theta, x_train, y_train)
            ce_test, err_test = ce_and_error_np(theta, x_test, y_test)
            margins = margin_stats_np(theta, x_train, y_train)
            summary = {
                "dataset_id": dataset_id,
                "split_id": int(row["split_id"]),
                "rule": row["rule"],
                "ref_id": ref_id,
                "theta_path": rel(theta_path),
                "dataset_path": row["dataset_path"],
                "attempt_seed": int(result["attempt_seed"]),
                "optimizer_chain": result["phase"],
                "P": int(theta.size),
                "train_error": float(err_train),
                "test_error": float(err_test),
                "CE_mean_train": float(ce_train),
                "CE_sum_train": float(ce_train * x_train.shape[0]),
                "CE_mean_test": float(ce_test),
                "theta_norm": float(np.linalg.norm(theta)),
                "theta_norm_sq": float(np.dot(theta, theta)),
                **margins,
                "reference_law_caveat": "optimizer-induced exact reference, not exact P_ref^0 sample",
            }
            write_json(ref_dir / "ref_summary.json", summary)
            reference_rows.append(summary)
            margin_rows.append({k: summary[k] for k in ["split_id", "rule", "ref_id", "min_margin", "q05_margin", "median_margin", "mean_margin"]})
    ref_df = pd.DataFrame(reference_rows)
    attempts_df = pd.DataFrame(attempt_rows)
    ref_df.to_csv(out_dir / "reference_index.csv", index=False)
    ensure_dir(out_dir / "attempt_logs")
    attempts_df.to_csv(out_dir / "attempt_logs" / "attempts.csv", index=False)
    fig_dir = ensure_dir(out_dir / "figures")
    success = attempts_df.groupby("rule")["selected"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(success["rule"], success["selected"])
    ax.set_ylabel("Selected fraction of attempts")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_reference_success_rate_by_rule.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule, sub in ref_df.groupby("rule"):
        ax.scatter(sub["CE_mean_train"], sub["theta_norm"], label=rule)
    ax.set_xlabel("CE mean train")
    ax.set_ylabel("Theta norm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_ref_ce_norm_scatter.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule, sub in pd.DataFrame(margin_rows).groupby("rule"):
        ax.hist(sub["min_margin"], bins=10, alpha=0.5, label=rule)
    ax.set_xlabel("Reference min signed margin")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_margin_distribution_by_rule.png", dpi=160)
    plt.close(fig)
    cfg["stage04_elapsed_s"] = time.time() - started
    write_json(out_dir / "run_config_resolved.json", cfg)
    counts = ref_df.groupby(["split_id", "rule"]).size()
    checks = {
        "reference_rows": int(len(ref_df)),
        "min_refs_per_dataset": int(counts.min()),
        "target_refs_per_dataset": target_refs,
        "all_exact": bool((ref_df["train_error"] == 0.0).all()),
        "theta_length_all_P": bool((ref_df["P"] == P).all()),
        "elapsed_s": float(cfg["stage04_elapsed_s"]),
    }
    pytest_result = run_pytest(ROOT / "tests" / "test_stage04_reference_payload.py", timeout_s=600)
    checks["pytest"] = pytest_result
    if checks["min_refs_per_dataset"] < target_refs or not checks["all_exact"] or not checks["theta_length_all_P"] or not pytest_result["passed"]:
        raise StageBlocked("04_exact_reference_search", "Reference QC failed after writing references.", observed=checks, expected={"refs_per_dataset": target_refs, "all_exact": True, "theta_length": P})
    write_qc("04_exact_reference_search", "pass", checks)
    write_stage_report(
        "04_exact_reference_search",
        "Stage 04 Exact Reference Search",
        f"Files created: {len(ref_df)} theta references, reference index, attempts log, and three figures under `{rel(out_dir)}`.\n\n"
        "Files modified: local 08_mnist smoke outputs only.\n\n"
        f"QC summary: pass; {target_refs} exact references per split/rule, P=3441, pytest passed.\n\n"
        "Blocking issues: none.\n\n"
        "Next command: `.venv/Scripts/python.exe 02_dnn/08_mnist/src/mnist14_smoke_pipeline.py --stage 05_pool2_pm_sais_sampling`",
    )


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(logsumexp(values) - math.log(values.size)) if values.size else float("-inf")


def ess_from_logw(logw: np.ndarray) -> float:
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return 0.0
    return float(np.exp(2.0 * logsumexp(logw) - logsumexp(2.0 * logw)))


def weighted_mean(values: np.ndarray, logw: np.ndarray) -> float:
    weights = np.exp(logw - logsumexp(logw))
    return float(np.sum(weights * values))


def sample_pm_sais_unit(ref_row: dict[str, Any], radius: float, n_samples: int, lambda_reg: float, seed: int, *, chunk_size: int = 128) -> dict[str, Any]:
    ds = load_dataset_npz(ref_row["dataset_path"])
    theta_ref = np.load(REPO_ROOT / ref_row["theta_path"]).astype(np.float64).reshape(-1)
    ref_norm = float(np.linalg.norm(theta_ref))
    if not np.isfinite(ref_norm) or ref_norm <= 0.0:
        raise ValueError(f"bad reference norm for {ref_row['theta_path']}")
    mu = -theta_ref / ref_norm
    kappa = float(lambda_reg * float(radius) * ref_norm / math.sqrt(P))
    rng = np.random.default_rng(int(seed))
    directions = sample_vmf(mu, kappa, int(n_samples), rng)
    theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * directions
    ce, err = ce_error_batch_torch(theta_batch, ds["X_train"], ds["y_train"], chunk_size=chunk_size, device="auto", dtype="float32")
    logw = -float(ds["X_train"].shape[0]) * ce
    split = np.arange(int(n_samples), dtype=np.int32) % 2
    log_prefactor = -float(lambda_reg) * float(radius) * float(radius) / 2.0 + log_sphere_mgf(P, kappa)
    split0 = log_prefactor + logmeanexp(logw[split == 0])
    split1 = log_prefactor + logmeanexp(logw[split == 1])
    logz = log_prefactor + logmeanexp(logw)
    h = np.sqrt(2.0 * np.maximum(ce - float(ref_row["CE_mean_train"]), 0.0))
    return {
        "split_id": int(ref_row["split_id"]),
        "rule": ref_row["rule"],
        "ref_id": int(ref_row["ref_id"]),
        "radius": float(radius),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "lambda_reg": float(lambda_reg),
        "theta_path": ref_row["theta_path"],
        "dataset_path": ref_row["dataset_path"],
        "theta_ref_norm": ref_norm,
        "kappa": kappa,
        "logM": log_sphere_mgf(P, kappa),
        "log_prefactor": log_prefactor,
        "logZ": float(logz),
        "split0_logZ": float(split0),
        "split1_logZ": float(split1),
        "split_logZ_per_P_diff": float(abs(split0 - split1) / P),
        "ess": ess_from_logw(logw),
        "ess_fraction": float(ess_from_logw(logw) / max(1, int(n_samples))),
        "weighted_ce": weighted_mean(ce, logw),
        "weighted_error": weighted_mean(err, logw),
        "weighted_h": weighted_mean(h, logw),
        "finite": bool(np.isfinite(logz) and np.isfinite(ce).all()),
        "hard_shell_distance_mean": float(np.mean(np.linalg.norm(theta_batch - theta_ref[None, :], axis=1) / math.sqrt(P))),
        "hard_shell_distance_max_abs_err": float(np.max(np.abs(np.linalg.norm(theta_batch - theta_ref[None, :], axis=1) / math.sqrt(P) - float(radius)))),
        "direction_unit_norm_max_abs_err": float(np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0))),
    }


def bootstrap_sd(values: np.ndarray, seed: int, n_boot: int = 300) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, values.size, size=values.size)
        means.append(float(np.mean(values[idx])))
    return float(np.std(means, ddof=1))


def summarize_shell_qc(unit_df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    r0 = float(cfg["sampling"]["r0"])
    key = ["split_id", "rule", "ref_id"]
    r0_df = unit_df[unit_df["radius"] == r0][key + ["logZ"]].rename(columns={"logZ": "logZ_r0"})
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined["logZ"] - joined["logZ_r0"]) / P
    summary_rows = []
    qc_rows = []
    for (rule, radius), sub in joined.groupby(["rule", "radius"]):
        finite_fraction = float(np.mean(np.isfinite(sub["logZ"])))
        q05_ess = float(np.quantile(sub["ess_fraction"], 0.05))
        max_split = float(np.max(sub["split_logZ_per_P_diff"]))
        boot_sd = bootstrap_sd(sub["delta_phi_energy_unit"].to_numpy(), 9000 + RULES.index(rule) * 100 + int(round(radius * 100)))
        pass_qc = (
            finite_fraction >= 0.90
            and q05_ess >= float(cfg["qc"]["q05_ess_fraction_min"])
            and max_split <= float(cfg["qc"]["max_split_logZ_per_P_diff"])
            and boot_sd <= float(cfg["qc"]["bootstrap_sd_phi_max"])
        )
        summary_rows.append(
            {
                "rule": rule,
                "radius": float(radius),
                "n_units": int(len(sub)),
                "finite_unit_fraction": finite_fraction,
                "q05_ess_fraction": q05_ess,
                "max_split_logZ_per_P_diff": max_split,
                "bootstrap_sd_phi": boot_sd,
                "mean_logZ": float(np.mean(sub["logZ"])),
                "mean_delta_phi_energy": float(np.mean(sub["delta_phi_energy_unit"])),
                "weighted_ce_mean": float(np.mean(sub["weighted_ce"])),
                "qc_pass": bool(pass_qc),
                "claim_status": "claimable_rule_radius" if pass_qc else "no_claim",
            }
        )
        qc_rows.append(
            {
                "rule": rule,
                "radius": float(radius),
                "pass": bool(pass_qc),
                "finite_unit_fraction": finite_fraction,
                "q05_ess_fraction": q05_ess,
                "max_split_logZ_per_P_diff": max_split,
                "bootstrap_sd_phi": boot_sd,
                "failed_radius_policy": "no_claim" if not pass_qc else "claimable_rule_radius",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(qc_rows)


def run_lambda_pilot(ref_df: pd.DataFrame, cfg: dict[str, Any], out_dir: Path) -> tuple[float, pd.DataFrame]:
    pilot_rows = []
    radii = [0.05, 0.10, 0.20, 0.45, 0.80]
    subset = ref_df[ref_df["split_id"] == 0].groupby("rule").head(3).reset_index(drop=True)
    for lam in cfg["loss"]["lambda_reg_candidates"]:
        unit_rows = []
        for row_id, row in enumerate(subset.to_dict("records")):
            for radius in radii:
                unit_rows.append(sample_pm_sais_unit(row, radius, 128, float(lam), 800000 + int(lam) * 1000 + row_id * 31 + int(radius * 100), chunk_size=64))
        unit_df = pd.DataFrame(unit_rows)
        summary_df, _qc_df = summarize_shell_qc(unit_df, cfg)
        pilot_rows.append(
            {
                "lambda_reg": float(lam),
                "q05_ess_fraction_min": float(summary_df["q05_ess_fraction"].min()),
                "max_split_logZ_per_P_diff": float(summary_df["max_split_logZ_per_P_diff"].max()),
                "pass_basic": bool((summary_df["q05_ess_fraction"].min() >= float(cfg["qc"]["q05_ess_fraction_min"])) and (summary_df["max_split_logZ_per_P_diff"].max() <= float(cfg["qc"]["max_split_logZ_per_P_diff"]))),
                "phi_energy_rule_spread_mean": float(summary_df.groupby("radius")["mean_delta_phi_energy"].std(ddof=0).mean()),
            }
        )
    pilot_df = pd.DataFrame(pilot_rows)
    pilot_df.to_csv(out_dir / "lambda_selection_report.csv", index=False)
    passed = pilot_df[pilot_df["pass_basic"]]
    if len(passed):
        selected = float(passed.sort_values(["phi_energy_rule_spread_mean", "q05_ess_fraction_min"], ascending=[False, False]).iloc[0]["lambda_reg"])
    else:
        selected = float(pilot_df.sort_values(["q05_ess_fraction_min", "max_split_logZ_per_P_diff"], ascending=[False, True]).iloc[0]["lambda_reg"])
    write_json(out_dir / "selected_lambda.json", {"lambda_reg": selected, "selection_rule": "single common lambda across rules", "pilot_any_pass_basic": bool(len(passed))})
    return selected, pilot_df


def stage05_pool2_pm_sais_sampling() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling"))
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not ref_path.exists():
        raise StageBlocked("05_pool2_pm_sais_sampling", "Stage 04 reference index is missing.", observed={"missing": rel(ref_path)}, expected={"file_exists": True})
    ref_df = pd.read_csv(ref_path)
    started = time.time()
    selected_lambda, pilot_df = run_lambda_pilot(ref_df, cfg, out_dir)
    unit_rows = []
    samples_by_radius = {float(k): int(v) for k, v in cfg["sampling"]["samples_per_ref_radius"].items()}
    for row_id, row in enumerate(ref_df.to_dict("records")):
        print(f"[stage05] reference {row_id + 1}/{len(ref_df)} split={row['split_id']} rule={row['rule']} ref={row['ref_id']}", flush=True)
        for radius in [float(r) for r in cfg["sampling"]["radii"]]:
            seed = 910000 + row_id * 1000 + int(round(radius * 100))
            unit_rows.append(sample_pm_sais_unit(row, radius, samples_by_radius[float(radius)], selected_lambda, seed))
    unit_df = pd.DataFrame(unit_rows)
    unit_df.to_csv(out_dir / "shell_summary_by_unit.csv", index=False)
    summary_df, qc_df = summarize_shell_qc(unit_df, cfg)
    summary_df.to_csv(out_dir / "shell_summary_by_rule_radius.csv", index=False)
    qc_df.to_csv(out_dir / "qc_by_rule_radius.csv", index=False)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pilot_df["lambda_reg"], pilot_df["q05_ess_fraction_min"], marker="o")
    ax.axhline(float(cfg["qc"]["q05_ess_fraction_min"]), color="black", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Pilot min q05 ESS fraction")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_lambda_pilot_phi_energy.png", dpi=160)
    plt.close(fig)
    for field, fname, title in [
        ("q05_ess_fraction", "fig02_sampling_qc_ess_heatmap.png", "q05 ESS fraction"),
        ("max_split_logZ_per_P_diff", "fig03_sampling_qc_split_logz_heatmap.png", "max split logZ/P diff"),
        ("weighted_ce_mean", "fig05_weighted_ce_by_rule_radius.png", "Weighted CE"),
    ]:
        pivot = summary_df.pivot(index="rule", columns="radius", values=field)
        fig, ax = plt.subplots(figsize=(8, 2.8))
        im = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [f"{c:.2f}" for c in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)
    cfg["selected_lambda"] = selected_lambda
    cfg["stage05_elapsed_s"] = time.time() - started
    write_json(out_dir / "run_config_resolved.json", cfg)
    common_pass = sorted(set.intersection(*[set(qc_df[(qc_df["rule"] == rule) & (qc_df["pass"])]["radius"]) for rule in RULES]))
    checks = {
        "selected_lambda": float(selected_lambda),
        "unit_rows": int(len(unit_df)),
        "rule_radius_rows": int(len(summary_df)),
        "all_logZ_finite": bool(np.isfinite(unit_df["logZ"]).all()),
        "hard_shell_max_abs_err": float(unit_df["hard_shell_distance_max_abs_err"].max()),
        "direction_unit_norm_max_abs_err": float(unit_df["direction_unit_norm_max_abs_err"].max()),
        "common_pass_radii": [float(x) for x in common_pass],
        "elapsed_s": float(cfg["stage05_elapsed_s"]),
    }
    pytest_result = run_pytest(ROOT / "tests" / "test_stage05_pm_sais_math.py", timeout_s=600)
    checks["pytest"] = pytest_result
    if not checks["all_logZ_finite"] or checks["hard_shell_max_abs_err"] > 1.0e-8 or not common_pass or not pytest_result["passed"]:
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "PM-SAIS hard QC failed.",
            observed=checks,
            expected={"finite_logZ": True, "hard_shell_error": "<= 1e-8", "at_least_one_common_pass_radius": True, "pytest_returncode": 0},
            next_action="Inspect PM-SAIS unit summaries; if only variance gates failed, increase samples per ref/radius and rerun Stage 05.",
        )
    warnings = []
    failed = qc_df[~qc_df["pass"]]
    if len(failed):
        warnings.append(f"{len(failed)} rule/radius QC rows are no_claim by policy.")
    if not bool((pilot_df["pass_basic"]).any()):
        warnings.append("No lambda candidate passed the reduced pilot basic thresholds; selected best available common lambda and applied radius no-claim policy.")
    write_qc("05_pool2_pm_sais_sampling", "pass", checks, warnings=warnings)
    write_stage_report(
        "05_pool2_pm_sais_sampling",
        "Stage 05 Pool2 PM-SAIS Sampling",
        f"Files created: lambda selection, {len(unit_df)} unit summaries, QC tables, and figures under `{rel(out_dir)}`.\n\n"
        "Files modified: local 08_mnist smoke outputs only.\n\n"
        f"QC summary: pass; selected common lambda {selected_lambda}; common pass radii {common_pass}; failed rule/radius rows are marked no_claim.\n\n"
        "Blocking issues: none.\n\n"
        "Next command: `.venv/Scripts/python.exe 02_dnn/08_mnist/src/mnist14_smoke_pipeline.py --stage 06_results_figures`",
    )


def final_bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values)) if values.size else float("nan")
    sd = bootstrap_sd(values, seed)
    return mean, float(mean - 1.96 * sd), float(mean + 1.96 * sd)


def mirror_final_report(source_dir: Path) -> Path:
    mirror_dir = ensure_dir(RUN_ROOT / "mnist14_3rule_1024_5_reference" / "final_report")
    for path in source_dir.rglob("*"):
        if path.is_file():
            target = mirror_dir / path.relative_to(source_dir)
            ensure_dir(target.parent)
            shutil.copyfile(path, target)
    return mirror_dir


def stage06_results_figures() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("final_report"))
    stage05_dir = stage_dir("05_pool2_pm_sais_sampling")
    required = [stage05_dir / "shell_summary_by_unit.csv", stage05_dir / "qc_by_rule_radius.csv"]
    if not all(p.exists() for p in required):
        raise StageBlocked("06_results_figures", "Required Stage 05 summaries are missing.", observed={rel(p): p.exists() for p in required}, expected={"all_required": True})
    unit_df = pd.read_csv(stage05_dir / "shell_summary_by_unit.csv")
    qc_df = pd.read_csv(stage05_dir / "qc_by_rule_radius.csv")
    common_pass = sorted(set.intersection(*[set(qc_df[(qc_df["rule"] == rule) & (qc_df["pass"])]["radius"]) for rule in RULES]))
    if not common_pass:
        raise StageBlocked(
            "06_results_figures",
            "No common QC-passed radius exists across all three rules.",
            observed={"common_pass_radii": []},
            expected={"common_pass_radii": "non-empty"},
            next_action="Increase Stage 05 samples or adjust the common lambda through the documented smoke retry path.",
        )
    d0 = float(cfg["sampling"]["r0"]) if float(cfg["sampling"]["r0"]) in common_pass else float(common_pass[0])
    key = ["split_id", "rule", "ref_id"]
    r0_df = unit_df[unit_df["radius"] == d0][key + ["logZ"]].rename(columns={"logZ": "logZ_r0"})
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined["logZ"] - joined["logZ_r0"]) / P
    phi_rows = []
    boot_rows = []
    for (rule, radius), sub in joined.groupby(["rule", "radius"]):
        pass_rule_radius = bool(qc_df[(qc_df["rule"] == rule) & (qc_df["radius"] == radius)]["pass"].iloc[0])
        if not pass_rule_radius:
            continue
        mean, lo, hi = final_bootstrap(sub["delta_phi_energy_unit"].to_numpy(), 11000 + RULES.index(rule) * 100 + int(round(radius * 100)))
        full = float(((P - 1) / P) * math.log(float(radius) / d0) + mean) if radius > 0 else float("nan")
        phi_rows.append(
            {
                "rule": rule,
                "radius": float(radius),
                "d0": d0,
                "delta_phi_energy": mean,
                "delta_phi_full": full,
                "n_units": int(len(sub)),
                "qc_pass": True,
            }
        )
        boot_rows.append(
            {
                "rule": rule,
                "radius": float(radius),
                "delta_phi_energy_mean": mean,
                "delta_phi_energy_ci95_low": lo,
                "delta_phi_energy_ci95_high": hi,
                "bootstrap_sd": float((hi - lo) / (2 * 1.96)),
            }
        )
    phi_df = pd.DataFrame(phi_rows)
    boot_df = pd.DataFrame(boot_rows)
    final_qc = qc_df.copy()
    final_qc["claimable_all_rules"] = final_qc["radius"].isin(common_pass)
    claim_rows = []
    for radius in sorted(qc_df["radius"].unique()):
        rules_pass = sorted(qc_df[(qc_df["radius"] == radius) & (qc_df["pass"])]["rule"].tolist())
        claim_rows.append(
            {
                "radius": float(radius),
                "claim_status": "supported" if float(radius) in common_pass else "no_claim",
                "rules_passed": ";".join(rules_pass),
                "rules_required": ";".join(RULES),
            }
        )
    claim_df = pd.DataFrame(claim_rows)
    complexity = pd.read_csv(stage_dir("02_complexity_measure") / "complexity_by_rule_summary.csv")
    ref = pd.read_csv(stage_dir("04_exact_reference_search") / "reference_index.csv")
    sampling_summary = pd.read_csv(stage05_dir / "shell_summary_by_rule_radius.csv")
    joined_summary = (
        sampling_summary.merge(complexity, on="rule", how="left")
        .merge(ref.groupby("rule").agg(reference_count=("ref_id", "count"), theta_norm_mean=("theta_norm", "mean"), min_margin_min=("min_margin", "min")).reset_index(), on="rule", how="left")
    )
    phi_df.to_csv(out_dir / "phi_by_rule_radius.csv", index=False)
    boot_df.to_csv(out_dir / "phi_bootstrap_by_rule_radius.csv", index=False)
    final_qc.to_csv(out_dir / "qc_pass_by_rule_radius.csv", index=False)
    joined_summary.to_csv(out_dir / "complexity_reference_sampling_joined.csv", index=False)
    claim_df.to_csv(out_dir / "final_claim_table.csv", index=False)
    fig_dir = ensure_dir(out_dir / "figures")
    stage01_fig = stage_dir("01_dataset_prepare") / "figures" / "fig01_mnist_28_vs_14_montage.png"
    stage02_fig = stage_dir("02_complexity_measure") / "figures" / "fig01_nmstv_by_rule_boxplot.png"
    if stage01_fig.exists():
        shutil.copyfile(stage01_fig, fig_dir / "fig01_dataset_montage_28_vs_14.png")
    if stage02_fig.exists():
        shutil.copyfile(stage02_fig, fig_dir / "fig02_complexity_nmstv_by_rule.png")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ref.groupby("rule")["theta_norm"].mean().plot(kind="bar", ax=axes[0], title="Norm")
    ref.groupby("rule")["min_margin"].min().plot(kind="bar", ax=axes[1], title="Min margin")
    ref.groupby("rule")["train_error"].mean().plot(kind="bar", ax=axes[2], title="Train error")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_reference_summary_success_norm_margin.png", dpi=160)
    plt.close(fig)
    for value, fname, ylabel in [
        ("delta_phi_energy", "fig04_phi_energy_three_rules_main.png", "Delta phi energy"),
        ("delta_phi_full", "fig05_phi_full_three_rules.png", "Delta phi full"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for rule, sub in phi_df.groupby("rule"):
            ax.plot(sub["radius"], sub[value], marker="o", label=rule)
        ax.axvline(d0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("d_raw")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    area = [((P - 1) / P) * math.log(r / d0) for r in sorted(common_pass)]
    ax.plot(sorted(common_pass), area, marker="o", label="area term")
    for rule, sub in phi_df.groupby("rule"):
        ax.plot(sub["radius"], sub["delta_phi_energy"], marker=".", label=f"{rule} energy")
    ax.set_xlabel("d_raw")
    ax.set_ylabel("Contribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_area_energy_decomposition.png", dpi=160)
    plt.close(fig)
    pivot = final_qc.pivot(index="rule", columns="radius", values="pass").astype(float)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), [f"{c:.2f}" for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig07_sampling_qc_pass_heatmap.png", dpi=160)
    plt.close(fig)
    for fields, fname in [
        (["q05_ess_fraction", "max_split_logZ_per_P_diff", "bootstrap_sd_phi"], "fig08_sampling_qc_ess_split_bootstrap.png"),
        (["weighted_ce_mean"], "fig09_weighted_ce_error_by_radius.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for field in fields:
            for rule, sub in joined_summary.groupby("rule"):
                ax.plot(sub["radius"], sub[field], marker="o", label=f"{field} {rule}")
        ax.set_xlabel("d_raw")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for rule, sub in phi_df.groupby("rule"):
        axes[0, 0].plot(sub["radius"], sub["delta_phi_energy"], marker="o", label=rule)
        axes[0, 1].plot(sub["radius"], sub["delta_phi_full"], marker="o", label=rule)
    complexity.plot(kind="bar", x="rule", y="nmstv_mean", ax=axes[1, 0], legend=False)
    pivot.plot(kind="bar", ax=axes[1, 1])
    axes[0, 0].set_title("Energy phi")
    axes[0, 1].set_title("Full phi")
    axes[1, 0].set_title("Complexity")
    axes[1, 1].set_title("QC pass")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_final_storyboard.png", dpi=160)
    plt.close(fig)
    supported = [float(x) for x in common_pass]
    no_claim = [float(x) for x in sorted(set(qc_df["radius"].unique()) - set(common_pass))]
    profile_claim = (
        "Using the same MNIST-14x14 input marginal and the same PM-SAIS estimator, changing only the label rule produces different smoke-scale reference-conditioned shell free-entropy profiles on QC-passed support."
        if len(supported) >= 2
        else "Smoke QC only supports the common baseline radius, so no across-radius phi-profile difference claim is made from this smoke run. All non-baseline radii remain no_claim unless resampled or otherwise repaired through the documented Stage 05 path."
    )
    selected_lambda_payload = read_json(stage05_dir / "selected_lambda.json")
    report = f"""## Objective

Use one MNIST-14x14 input marginal, vary only the label rule, and compare reference-conditioned shell free entropy at smoke scale.

## Dataset Preparation

Stage 01 prepared 3 splits x 3 rules with 256 train and 2048 test examples per split. MNIST images were average pooled from 28x28 to 14x14 and standardized per split.

## Complexity Summary

Graph TV/NMSTV summaries were computed on standardized train vectors for k=8,16,32. Unexpected ordering is not a failure gate.

## Pool1 Reference Summary

Stage 04 selected optimizer-induced exact references only. These are not claimed to be exact samples from `P_ref^0`.

## Pool2 PM-SAIS Summary

Stage 05 used PM-SAIS H=infinity with one common lambda: `{float(selected_lambda_payload['lambda_reg'])}`.

## QC Gates

Supported d_raw radii: {supported}

No-claim d_raw radii: {no_claim}

Failed rule/radius rows are excluded from the final claim table by the `no_claim` policy.

## Main Phi Energy Result

{profile_claim}

## Full Phi Area Warning

`Delta phi_full` includes the high-dimensional shell-area term and should not be interpreted as the primary landscape-quality comparison. The energy-only curve is the main comparison.

## Limitations

This is smoke scale, not candidate or final production. Reference samples are optimizer-induced exact solutions. Random-label test accuracy is not a generalization metric. No claim is made outside QC-passed radii.

## Next Candidate Scale

Use `02_dnn/08_mnist/stages/07_candidate_final_promotion/START_PROMPT.md` only after accepting the smoke report.
"""
    write_stage_report("06_results_figures", "MNIST14 PM-SAIS Smoke Final Report", report, final=True)
    mirror_dir = mirror_final_report(out_dir)
    write_json(out_dir / "run_config_resolved.json", {**cfg, "d0_used": d0, "supported_radii": supported, "no_claim_radii": no_claim, "mirrored_final_report": rel(mirror_dir)})
    mirror_dir = mirror_final_report(out_dir)
    checks = {
        "supported_radii": supported,
        "no_claim_radii": no_claim,
        "d0_used": d0,
        "profile_comparison_supported": bool(len(supported) >= 2),
        "phi_rows": int(len(phi_df)),
        "main_figure_exists": bool((fig_dir / "fig04_phi_energy_three_rules_main.png").exists()),
        "final_claim_table_exists": bool((out_dir / "final_claim_table.csv").exists()),
        "mirrored_final_report": rel(mirror_dir),
        "mirrored_main_figure_exists": bool((mirror_dir / "figures" / "fig04_phi_energy_three_rules_main.png").exists()),
    }
    if not checks["main_figure_exists"] or not checks["final_claim_table_exists"] or not supported:
        raise StageBlocked("06_results_figures", "Final report QC failed.", observed=checks, expected={"main_figure": True, "claim_table": True, "supported_radii": "non-empty"})
    warnings = ["Smoke report only; not production final."]
    if len(supported) < 2:
        warnings.append("Only the common baseline radius passed QC across all three rules; no profile-difference claim is supported.")
    write_qc("06_results_figures", "pass", checks, warnings=warnings)
    mirror_final_report(out_dir)


def run_stage(stage: str) -> None:
    if stage == "00_repo_audit":
        stage00_repo_audit()
    elif stage == "01_dataset_prepare":
        stage01_dataset_prepare()
    elif stage == "02_complexity_measure":
        stage02_complexity_measure()
    elif stage == "03_pool_design":
        stage03_pool_design()
    elif stage == "04_exact_reference_search":
        stage04_exact_reference_search()
    elif stage == "05_pool2_pm_sais_sampling":
        stage05_pool2_pm_sais_sampling()
    elif stage == "06_results_figures":
        stage06_results_figures()
    else:
        raise ValueError(f"unknown stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGE_NAMES + ["all"], required=True)
    args = parser.parse_args(argv)
    stages = STAGE_NAMES if args.stage == "all" else [args.stage]
    for stage in stages:
        try:
            run_stage(stage)
        except StageBlocked as blocked:
            write_blocked_report(blocked)
            print(f"BLOCKED {blocked.stage}: {blocked.reason}", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
