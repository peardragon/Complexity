from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
BASE_ROOT = SCRIPT_PATH.parents[1]
DNN_ROOT = SCRIPT_PATH.parents[2]
LOCAL_ROOT = SCRIPT_PATH.parents[3]
BIN_ROOT = LOCAL_ROOT / "99_codex_bin" / "parallel_sampling_18_90_30"
SAMPLER = BIN_ROOT / "parallel_sampling_18_90_30.py"
POST_90 = BIN_ROOT / "post_completion_90_local.py"
PHI_ANALYSIS = BASE_ROOT / "scripts" / "analyze_gaussian_vs_spin_phi.py"

DATASET_SRC = DNN_ROOT / "01_dataset_gen" / "src"
REFERENCE_SRC = DNN_ROOT / "03_reference_search" / "src"
PROXY_SRC = DNN_ROOT / "05_proxy_local_entropy" / "src"
for _path in (DATASET_SRC, REFERENCE_SRC, PROXY_SRC, BIN_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


RUN_DATASET = "gaussian_random_90_dataset"
RUN_REFERENCE = "gaussian_random_90_dataset_30_reference"
RANGE_NAME = "d_0.01_to_2.50_dense"
SELECTED_BETAS = [0.05]
SELECTED_CELLS = [
    {
        "beta": beta,
        "cell_id": f"cell_beta_{beta:.2f}".replace(".", "p") + "_p_0p00",
        "source_sorted_index": 2 * idx,
    }
    for idx, beta in enumerate(SELECTED_BETAS)
]
DATASETS_PER_BETA = 90
REFERENCES_PER_DATASET = 30
WIDTH = 48
ATTEMPTS_PER_DATASET = 200
REFERENCE_ATTEMPT_BATCH_SIZE = 12
REFERENCE_ADAM_EPOCHS = 4000
REFERENCE_LBFGS_MAX_ITER = 1500
REFERENCE_HPARAM_CYCLE = (
    (0.03, 1.0),
    (0.03, 2.0),
    (0.01, 4.0),
    (0.003, 8.0),
)
SAFE_GPU_IDS = ["2", "3"]
CPU_AFFINITY_CPUS = "0-15"
GAUSSIAN_SEED_OFFSET = 606_000_000
PARAM_COUNT = 2545
SAMPLING_RADII_COUNT = 250

DATASET_ROOT = BASE_ROOT / "raw_outputs" / "01_dataset_gen" / RUN_DATASET
DATASET_RAW = DATASET_ROOT / "raw_datasets"
COMPLEXITY_ROOT = BASE_ROOT / "raw_outputs" / "02_complexity_measure" / RUN_REFERENCE
REFERENCE_ROOT = BASE_ROOT / "raw_outputs" / "03_reference_search" / RUN_REFERENCE
REFERENCE_SELECTED = REFERENCE_ROOT / "selected_references"
REFERENCE_RAW_ATTEMPTS = REFERENCE_ROOT / "raw_attempts"
REFERENCE_POOL_ROOT = BASE_ROOT / "raw_outputs" / "04_sampling" / "reference_pool" / RUN_REFERENCE
REFERENCE_POOL = REFERENCE_POOL_ROOT / "selected_reference_pool"
SHELL_ROOT = BASE_ROOT / "raw_outputs" / "04_sampling" / "shell_pool" / RUN_REFERENCE / RANGE_NAME
PROXY_RAW_ROOT = BASE_ROOT / "raw_outputs" / "05_proxy_local_entropy" / RUN_REFERENCE / RANGE_NAME
PROXY_FIG_ROOT = BASE_ROOT / "figures" / "05_proxy_local_entropy" / RUN_REFERENCE / RANGE_NAME
OVERLAY_ROOT = BASE_ROOT / "figures" / "gaussian_overlay"
MANIFEST = BASE_ROOT / "manifests" / "reference_manifest_gaussian_90_30.csv"
CONFIG = BASE_ROOT / "config" / "gaussian_90_dense.yaml"
PROGRESS_ROOT = BASE_ROOT / "progress"
FULL_PIPELINE_ROOT = PROGRESS_ROOT / "full_pipeline"


@dataclass
class DatasetAccum:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += float(value)
        self.count += 1

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def repo_relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(LOCAL_ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else LOCAL_ROOT / path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--:--"
    try:
        total = max(0, int(float(seconds)))
    except Exception:
        return "--:--:--"
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def env_for_compute(*, cuda: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "COMPLEXITY_CPU_AFFINITY_CPUS": CPU_AFFINITY_CPUS,
        }
    )
    if cuda:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(SAFE_GPU_IDS)
    return env


def pin_current_process_to_cpu_half() -> dict[str, Any]:
    cpus = set(range(16))
    try:
        os.sched_setaffinity(0, cpus)
        return {"ok": True, "cpus": sorted(cpus)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "cpus": sorted(cpus)}


def hardlink_or_copy(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def dataset_seed(source_sorted_index: int, dataset_id: int) -> int:
    return int(GAUSSIAN_SEED_OFFSET + 1000 * int(source_sorted_index) + 100000 * int(dataset_id) + 1234)


def dataset_tag(source_sorted_index: int, dataset_id: int) -> str:
    return f"dataset_{int(dataset_id):03d}_seed_{dataset_seed(source_sorted_index, dataset_id):09d}"


def selected_cell_from_cell_id(cell_id: str) -> dict[str, Any]:
    for cell in SELECTED_CELLS:
        if str(cell["cell_id"]) == str(cell_id):
            return cell
    raise KeyError(cell_id)


def dataset_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in SELECTED_CELLS:
        for dataset_id in range(DATASETS_PER_BETA):
            seed = dataset_seed(int(cell["source_sorted_index"]), dataset_id)
            tag = dataset_tag(int(cell["source_sorted_index"]), dataset_id)
            raw_path = DATASET_RAW / str(cell["cell_id"]) / tag / "dataset.npz"
            meta_path = DATASET_RAW / str(cell["cell_id"]) / tag / "dataset_meta.json"
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "series": "beta",
                    "dataset_id": int(dataset_id),
                    "seed": int(seed),
                    "beta_ising": float(cell["beta"]),
                    "rewire_p": 0.0,
                    "dataset_tag": tag,
                    "dataset_raw_path": repo_relative(raw_path),
                    "dataset_meta_path": repo_relative(meta_path),
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _knn_edges(x: np.ndarray, *, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(x, dtype=np.float64))
    _dist, idx = tree.query(np.asarray(x, dtype=np.float64), k=int(k) + 1)
    edges: set[tuple[int, int]] = set()
    for i in range(idx.shape[0]):
        for j in idx[i, 1:]:
            a, b = sorted((int(i), int(j)))
            if a != b:
                edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64)


def _dataset_complexity_metrics(npz_path: Path, *, k: int = 10) -> dict[str, float]:
    with np.load(npz_path) as data:
        x = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    if set(float(v) for v in np.unique(y)).issubset({0.0, 1.0}):
        y = 2.0 * y - 1.0
    y = np.where(y >= 0.0, 1.0, -1.0)
    edges = _knn_edges(x, k=k)
    if edges.size:
        yi = y[edges[:, 0]]
        yj = y[edges[:, 1]]
        same = yi == yj
        edge_agreement = float(np.mean(same))
        edge_disagreement = float(1.0 - edge_agreement)
        label_autocorrelation = float(np.mean(yi * yj))
    else:
        edge_agreement = float("nan")
        edge_disagreement = float("nan")
        label_autocorrelation = float("nan")
    corr = []
    for dim in range(x.shape[1]):
        x_col = x[:, dim]
        denom = float(np.std(x_col) * np.std(y))
        corr.append(float(np.mean((x_col - np.mean(x_col)) * (y - np.mean(y))) / denom) if denom > 0.0 else 0.0)
    linear_signal = float(np.sqrt(np.sum(np.square(corr))))
    return {
        "n_points": float(x.shape[0]),
        "input_dim": float(x.shape[1]),
        "label_mean": float(np.mean(y)),
        "label_abs_mean": float(abs(np.mean(y))),
        "knn_k": float(k),
        "knn_edge_count": float(edges.shape[0]),
        "knn_edge_agreement": edge_agreement,
        "knn_edge_disagreement": edge_disagreement,
        "knn_label_autocorrelation": label_autocorrelation,
        "feature_label_linear_signal": linear_signal,
    }


def compute_complexity_diagnostics(_: argparse.Namespace | None = None) -> int:
    import pandas as pd

    prepare_directories()
    spin_index = DNN_ROOT / "01_dataset_gen" / "raw_outputs" / "18_beta_cell_90_dataset" / "dataset_index.csv"
    gaussian_index = DATASET_ROOT / "dataset_index.csv"
    if not spin_index.exists():
        raise FileNotFoundError(spin_index)
    if not gaussian_index.exists():
        raise FileNotFoundError(gaussian_index)

    rows: list[dict[str, Any]] = []
    for run_label, index_path in (("spin_dynamics_90_dataset", spin_index), ("gaussian_random_90_dataset", gaussian_index)):
        df = pd.read_csv(index_path)
        for row in df.to_dict("records"):
            npz_path = resolve_repo_path(row["dataset_raw_path"])
            metrics = _dataset_complexity_metrics(npz_path, k=10)
            rows.append(
                {
                    "run": run_label,
                    "cell_id": row["cell_id"],
                    "dataset_id": int(row["dataset_id"]),
                    "seed": int(row["seed"]),
                    "beta": float(row["beta_ising"]),
                    "dataset_path": repo_relative(npz_path),
                    **metrics,
                }
            )
    out_root = COMPLEXITY_ROOT / "summary_tables"
    per_dataset = out_root / "dataset_complexity_per_dataset.csv"
    fields = [
        "run",
        "cell_id",
        "dataset_id",
        "seed",
        "beta",
        "dataset_path",
        "n_points",
        "input_dim",
        "label_mean",
        "label_abs_mean",
        "knn_k",
        "knn_edge_count",
        "knn_edge_agreement",
        "knn_edge_disagreement",
        "knn_label_autocorrelation",
        "feature_label_linear_signal",
    ]
    write_csv(per_dataset, rows, fields)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["run", "beta"], as_index=False)
        .agg(
            dataset_count=("dataset_id", "count"),
            knn_edge_disagreement_mean=("knn_edge_disagreement", "mean"),
            knn_edge_disagreement_std=("knn_edge_disagreement", "std"),
            knn_label_autocorrelation_mean=("knn_label_autocorrelation", "mean"),
            feature_label_linear_signal_mean=("feature_label_linear_signal", "mean"),
            label_abs_mean_mean=("label_abs_mean", "mean"),
        )
        .sort_values(["run", "beta"])
    )
    summary["knn_edge_disagreement_sem"] = summary["knn_edge_disagreement_std"] / np.sqrt(summary["dataset_count"].clip(lower=1))
    summary_path = out_root / "dataset_complexity_by_run_beta.csv"
    summary.to_csv(summary_path, index=False)

    spin = summary[summary["run"] == "spin_dynamics_90_dataset"].copy()
    gauss = summary[summary["run"] == "gaussian_random_90_dataset"].copy()
    gaussian_mean = float(gauss["knn_edge_disagreement_mean"].mean())
    spin["abs_gap_to_gaussian_knn_disagreement"] = (spin["knn_edge_disagreement_mean"] - gaussian_mean).abs()
    nearest = spin.sort_values("abs_gap_to_gaussian_knn_disagreement").head(5)
    nearest_path = out_root / "nearest_spin_beta_to_gaussian_complexity.csv"
    nearest.to_csv(nearest_path, index=False)

    report = COMPLEXITY_ROOT / "complexity_diagnostics_report.md"
    table_rows = [
        "| beta | knn_edge_disagreement_mean | knn_label_autocorrelation_mean | abs_gap_to_gaussian_knn_disagreement |",
        "|---:|---:|---:|---:|",
    ]
    for row in nearest[
        [
            "beta",
            "knn_edge_disagreement_mean",
            "knn_label_autocorrelation_mean",
            "abs_gap_to_gaussian_knn_disagreement",
        ]
    ].to_dict("records"):
        table_rows.append(
            "| {beta:.2f} | {knn_edge_disagreement_mean:.6f} | {knn_label_autocorrelation_mean:.6f} | {abs_gap_to_gaussian_knn_disagreement:.6f} |".format(
                **{key: float(value) for key, value in row.items()}
            )
        )
    report.write_text(
        "# Gaussian Random Baseline Complexity Diagnostics\n\n"
        "Complexity proxy used here: kNN graph label roughness on normalized input features. "
        "For PM1 labels, random independent labels should have edge disagreement near 0.5 and "
        "label autocorrelation near 0.\n\n"
        f"- per-dataset table: `{repo_relative(per_dataset)}`\n"
        f"- beta summary table: `{repo_relative(summary_path)}`\n"
        f"- nearest spin beta table: `{repo_relative(nearest_path)}`\n"
        f"- Gaussian mean kNN edge disagreement: `{gaussian_mean:.6f}`\n\n"
        "Nearest spin beta tags by this proxy:\n\n"
        + "\n".join(table_rows)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": repo_relative(report), "summary": repo_relative(summary_path)}, indent=2, sort_keys=True))
    return 0


def _balanced_random_labels(rng: np.random.Generator, n_points: int) -> np.ndarray:
    y = np.ones(int(n_points), dtype=np.int8)
    y[: int(n_points) // 2] = -1
    rng.shuffle(y)
    return y


def make_gaussian_dataset(seed: int, *, n_points: int = 512, input_dim: int = 2) -> dict[str, Any]:
    from dataset_builder import normalize_features

    rng = np.random.default_rng(int(seed))
    x_raw = rng.normal(loc=0.0, scale=1.0, size=(int(n_points), int(input_dim))).astype(np.float64)
    x_train, norm_stats = normalize_features(x_raw)
    y = _balanced_random_labels(rng, int(n_points))
    return {
        "X_raw": x_raw,
        "X_train": x_train,
        "y": y,
        "meta": {
            "generator": "iid_gaussian_features_balanced_random_labels_v1",
            "seed": int(seed),
            "n_points": int(n_points),
            "input_dim": int(input_dim),
            "label_mode": "balanced_random_independent",
            "created_at": now_iso(),
            **norm_stats,
        },
    }


def prepare_directories(_: argparse.Namespace | None = None) -> int:
    roots = {
        "dataset": DATASET_ROOT,
        "complexity_measure": COMPLEXITY_ROOT,
        "reference_search": REFERENCE_ROOT,
        "reference_pool": REFERENCE_POOL_ROOT,
        "shell_pool": SHELL_ROOT,
        "proxy_raw": PROXY_RAW_ROOT,
        "proxy_fig": PROXY_FIG_ROOT,
        "overlay": OVERLAY_ROOT,
        "manifests": MANIFEST.parent,
        "progress": PROGRESS_ROOT,
    }
    for path in roots.values():
        ensure_dir(path)
    (COMPLEXITY_ROOT / "run_report.md").write_text(
        "# Gaussian random baseline complexity anchor\n\n"
        "This stage-02 root is a provenance anchor for the random Gaussian baseline.\n"
        "The actual phi(d) chain consumes the reference search and sampling outputs.\n\n"
        f"- dataset root: `{repo_relative(DATASET_ROOT)}`\n"
        f"- reference search root: `{repo_relative(REFERENCE_ROOT)}`\n"
        f"- sampling shell root: `{repo_relative(SHELL_ROOT)}`\n",
        encoding="utf-8",
    )
    save_json(PROGRESS_ROOT / "roots.json", {key: repo_relative(value) for key, value in roots.items()})
    return 0


def run_prepare_datasets(args: argparse.Namespace) -> int:
    prepare_directories()
    rows = dataset_rows()
    total = len(rows)
    started = time.time()
    generated = 0
    reused = 0
    for idx, row in enumerate(rows, start=1):
        cell_id = str(row["cell_id"])
        tag = str(row["dataset_tag"])
        target_dir = DATASET_RAW / cell_id / tag
        npz_path = target_dir / "dataset.npz"
        meta_path = target_dir / "dataset_meta.json"
        if npz_path.exists() and meta_path.exists() and not bool(args.force):
            reused += 1
        else:
            ensure_dir(target_dir)
            data = make_gaussian_dataset(int(row["seed"]))
            np.savez_compressed(npz_path, X_raw=data["X_raw"], X_train=data["X_train"], y=data["y"])
            save_json(
                meta_path,
                {
                    "cell_id": cell_id,
                    "series": "beta",
                    "dataset_id": int(row["dataset_id"]),
                    "seed": int(row["seed"]),
                    "beta_ising": float(row["beta_ising"]),
                    "rewire_p": 0.0,
                    "random_baseline": True,
                    "meta": data["meta"],
                },
            )
            generated += 1
        if idx % max(1, int(args.status_every)) == 0 or idx == total:
            save_json(
                PROGRESS_ROOT / "dataset_prepare.json",
                {
                    "timestamp": now_iso(),
                    "completed": idx,
                    "total": total,
                    "generated": generated,
                    "reused": reused,
                    "elapsed": format_duration(time.time() - started),
                },
            )
            print(f"[dataset] {idx}/{total} generated={generated} reused={reused}", flush=True)
    fields = ["cell_id", "series", "dataset_id", "seed", "beta_ising", "rewire_p", "dataset_raw_path", "dataset_meta_path"]
    write_csv(DATASET_ROOT / "dataset_index.csv", rows, fields)
    save_json(
        DATASET_ROOT / "run_config.json",
        {
            "run_name": RUN_DATASET,
            "generator": "iid_gaussian_features_balanced_random_labels_v1",
            "selected_betas": SELECTED_BETAS,
            "datasets_per_beta": DATASETS_PER_BETA,
            "seed_formula": f"{GAUSSIAN_SEED_OFFSET} + 1000 * source_sorted_index + 100000 * dataset_id + 1234",
        },
    )
    (DATASET_ROOT / "run_report.md").write_text(
        f"# Gaussian dataset generation report: {RUN_DATASET}\n\n"
        f"- selected beta tags: `{len(SELECTED_CELLS)}`\n"
        f"- datasets per beta tag: `{DATASETS_PER_BETA}`\n"
        f"- generator: `iid_gaussian_features_balanced_random_labels_v1`\n"
        f"- generated: `{generated}`\n"
        f"- reused: `{reused}`\n"
        f"- elapsed: `{format_duration(time.time() - started)}`\n"
        f"- dataset index: `{repo_relative(DATASET_ROOT / 'dataset_index.csv')}`\n",
        encoding="utf-8",
    )
    return 0


def reference_attempt_seed(dataset_id: int, attempt_id: int) -> int:
    return int(900_000_000 + int(attempt_id) * 100_000 + WIDTH * 1_000 + int(dataset_id))


def reference_attempt_hparams(attempt_id: int) -> tuple[float, float]:
    return REFERENCE_HPARAM_CYCLE[int(attempt_id) % len(REFERENCE_HPARAM_CYCLE)]


def load_dataset_for_training(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    x = np.asarray(data["X_train"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    if set(float(v) for v in np.unique(y)).issubset({0.0, 1.0}):
        y = 2.0 * y - 1.0
    return x, y


def attempt_fieldnames() -> list[str]:
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


def write_attempt_results_from_disk(width_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for attempt_dir in sorted(width_dir.glob("attempt_*")):
        if not attempt_dir.is_dir():
            continue
        try:
            attempt_id = int(attempt_dir.name.rsplit("_", 1)[1])
        except Exception:
            continue
        summary_path = attempt_dir / "train_summary.json"
        theta_path = attempt_dir / "theta_final.npy"
        theta_init_path = attempt_dir / "theta_init.npy"
        if not summary_path.exists() or not theta_path.exists() or not theta_init_path.exists():
            continue
        summary = load_json(summary_path, {})
        rows.append(
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
                "theta_final_path": repo_relative(theta_path),
                "theta_init_path": repo_relative(theta_init_path),
                "summary_path": repo_relative(summary_path),
            }
        )
    write_csv(width_dir / "attempt_results.csv", sorted(rows, key=lambda r: int(r["attempt_id"])), attempt_fieldnames())


def load_attempt_records(width_dir: Path) -> list[dict[str, Any]]:
    attempt_csv = width_dir / "attempt_results.csv"
    if not attempt_csv.exists():
        return []
    records: list[dict[str, Any]] = []
    with attempt_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            theta_path = resolve_repo_path(row["theta_final_path"])
            theta_init_path = resolve_repo_path(row["theta_init_path"])
            summary_path = resolve_repo_path(row["summary_path"])
            if not theta_path.exists() or not theta_init_path.exists() or not summary_path.exists():
                continue
            records.append(
                {
                    "attempt_id": int(row["attempt_id"]),
                    "theta": np.load(theta_path).astype(np.float64),
                    "theta_init": np.load(theta_init_path).astype(np.float64),
                    "theta_path": repo_relative(theta_path),
                    "theta_init_path": repo_relative(theta_init_path),
                    "summary_path": repo_relative(summary_path),
                    "summary": load_json(summary_path, {}),
                    "is_rescue": False,
                }
            )
    return records


def select_reference_unit(cell_id: str, dataset_tag_value: str, width_dir: Path, selected_width_dir: Path) -> dict[str, Any]:
    from rescue import summarize_and_select_reference_candidates

    attempt_records = load_attempt_records(width_dir)
    selection = summarize_and_select_reference_candidates(
        attempt_records,
        cell_id=cell_id,
        dataset_tag=dataset_tag_value,
        width=WIDTH,
        min_train_accuracy=0.95,
        target_valid_count=REFERENCES_PER_DATASET,
        max_selected_count=REFERENCES_PER_DATASET,
        topk=REFERENCES_PER_DATASET,
        dedup_scale=0.25,
        require_exact=True,
        rescue_enabled=False,
        rescue_policy_name="none",
    )
    ensure_dir(selected_width_dir)
    payload_root = selected_width_dir / "selected_ref_payloads"
    retained_rows: list[dict[str, Any]] = []
    for row in selection["selected_rows"]:
        retained = dict(row)
        ref_id = int(retained["ref_id"])
        ref_dir = payload_root / f"ref_{ref_id:03d}"
        ensure_dir(ref_dir)
        theta_path = ref_dir / "theta.npy"
        theta_init_path = ref_dir / "theta_init.npy"
        summary_path = ref_dir / "train_summary.json"
        hardlink_or_copy(resolve_repo_path(retained["theta_path"]), theta_path)
        hardlink_or_copy(resolve_repo_path(retained["theta_init_path"]), theta_init_path)
        hardlink_or_copy(resolve_repo_path(retained["summary_path"]), summary_path)
        summary = load_json(summary_path, {})
        theta = np.load(theta_path).astype(np.float64).reshape(-1)
        retained.update(
            {
                "theta_path": repo_relative(theta_path),
                "theta_init_path": repo_relative(theta_init_path),
                "summary_path": repo_relative(summary_path),
                "final_cls_err": float(summary.get("final_cls_err", float("nan"))),
                "final_train_accuracy": float(summary.get("final_train_accuracy", float("nan"))),
                "final_train_loss": float(summary.get("final_train_loss", float("nan"))),
                "min_signed_margin": float(summary.get("min_signed_margin", float("nan"))),
                "theta_norm": float(np.linalg.norm(theta)),
                "norm_sq": float(np.dot(theta, theta)),
                "lambda_ref": 1.0,
                "selector": "l2_top30_min_norm_exact",
            }
        )
        retained_rows.append(retained)
    save_json(selected_width_dir / "selected_refs.json", {"cell_id": cell_id, "dataset_tag": dataset_tag_value, "width": WIDTH, "selected_refs": retained_rows})
    manifest = selection["manifest_payload"]
    manifest["selected_ref_count"] = len(retained_rows)
    return manifest


def train_reference_unit(
    cell: dict[str, Any],
    dataset_id: int,
    *,
    device: str,
    force: bool,
    attempts: int,
    attempt_batch_size: int,
    adam_epochs: int,
    lbfgs_max_iter: int,
    verbose_training: bool,
) -> dict[str, Any]:
    from model_types import DNNArch, TrainConfig
    from training import train_reference_solutions_simple_batched

    cell_id = str(cell["cell_id"])
    tag = dataset_tag(int(cell["source_sorted_index"]), dataset_id)
    dataset_path = DATASET_RAW / cell_id / tag / "dataset.npz"
    width_dir = REFERENCE_RAW_ATTEMPTS / cell_id / tag / f"width_{WIDTH:03d}"
    selected_width_dir = REFERENCE_SELECTED / cell_id / tag / f"width_{WIDTH:03d}"
    selected_json = selected_width_dir / "selected_refs.json"
    if force:
        if width_dir.exists():
            shutil.rmtree(width_dir)
        if selected_width_dir.exists():
            shutil.rmtree(selected_width_dir)
    if selected_json.exists() and not force:
        payload = load_json(selected_json, {})
        if len(payload.get("selected_refs", [])) >= REFERENCES_PER_DATASET:
            return {"cell_id": cell_id, "dataset_tag": tag, "status": "reuse_selected", "selected_ref_count": len(payload.get("selected_refs", []))}
    ensure_dir(width_dir)

    x, y = load_dataset_for_training(dataset_path)
    arch = DNNArch(input_dim=2, width1=WIDTH, width2=WIDTH)
    existing = load_attempt_records(width_dir)
    existing_ids = sorted(int(record["attempt_id"]) for record in existing)
    start_id = max(existing_ids) + 1 if existing_ids else 0
    remaining = max(0, int(attempts) - len(existing_ids))
    trained_attempts = 0
    while remaining > 0:
        sub_count = min(int(attempt_batch_size), remaining)
        attempt_ids = list(range(start_id, start_id + sub_count))
        cfgs = [
            TrainConfig(
                lr=reference_attempt_hparams(attempt_id)[0],
                weight_decay=1.0e-4,
                momentum=0.9,
                epochs=int(adam_epochs),
                seed=reference_attempt_seed(dataset_id, attempt_id),
                optimizer_name="adam",
                lbfgs_max_iter=int(lbfgs_max_iter),
                init_scale_multiplier=reference_attempt_hparams(attempt_id)[1],
                activation="tanh",
                loss="logistic",
                margin=1.0,
            )
            for attempt_id in attempt_ids
        ]
        outputs = train_reference_solutions_simple_batched(
            x,
            y,
            arch,
            cfgs,
            device=device,
            verbose=bool(verbose_training),
            progress_label=f"{cell_id}/{tag}/width_{WIDTH:03d}/attempts_{attempt_ids[0]:03d}_{attempt_ids[-1]:03d}",
        )
        for output, attempt_id in zip(outputs, attempt_ids):
            output["attempt_id"] = int(attempt_id)
            output["summary"]["attempt_id"] = int(attempt_id)
            attempt_dir = width_dir / f"attempt_{attempt_id:03d}"
            ensure_dir(attempt_dir)
            np.save(attempt_dir / "theta_final.npy", np.asarray(output["theta"], dtype=np.float64))
            np.save(attempt_dir / "theta_init.npy", np.asarray(output["theta_init"], dtype=np.float64))
            save_json(attempt_dir / "train_summary.json", output["summary"])
        write_attempt_results_from_disk(width_dir)
        manifest = select_reference_unit(cell_id, tag, width_dir, selected_width_dir)
        trained_attempts += len(outputs)
        start_id += sub_count
        remaining -= sub_count
        save_json(
            width_dir / "training_status.json",
            {
                "timestamp": now_iso(),
                "attempt_count": len(load_attempt_records(width_dir)),
                "trained_attempts_this_invocation": trained_attempts,
                "selected_ref_count": int(manifest.get("selected_ref_count", 0)),
                "exact_count": int(manifest.get("exact_count", 0)),
            },
        )
        if int(manifest.get("selected_ref_count", 0)) >= REFERENCES_PER_DATASET:
            break
    manifest = select_reference_unit(cell_id, tag, width_dir, selected_width_dir)
    return {
        "cell_id": cell_id,
        "dataset_tag": tag,
        "status": "trained_or_selected",
        "selected_ref_count": int(manifest.get("selected_ref_count", 0)),
        "exact_count": int(manifest.get("exact_count", 0)),
        "attempt_count": int(manifest.get("attempt_count", 0)),
        "trained_attempts": int(trained_attempts),
        "invalid": int(manifest.get("selected_ref_count", 0)) < REFERENCES_PER_DATASET,
    }


def reference_work_items(start_dataset: int = 0, stop_dataset: int = DATASETS_PER_BETA) -> list[tuple[dict[str, Any], int]]:
    return [(cell, dataset_id) for cell in SELECTED_CELLS for dataset_id in range(int(start_dataset), int(stop_dataset))]


def run_reference_worker(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    affinity = pin_current_process_to_cpu_half()
    items = reference_work_items(int(args.start_dataset), int(args.stop_dataset))
    shard_items = [item for idx, item in enumerate(items) if idx % int(args.shard_count) == int(args.shard_id)]
    shard_root = REFERENCE_ROOT / "logs" / "reference_shards" / f"shard_{int(args.shard_id):03d}"
    ensure_dir(shard_root)
    failures: list[dict[str, Any]] = []
    completed = 0
    for cell, dataset_id in shard_items:
        try:
            result = train_reference_unit(
                cell,
                dataset_id,
                device=str(args.device),
                force=bool(args.force),
                attempts=int(args.attempts),
                attempt_batch_size=int(args.attempt_batch_size),
                adam_epochs=int(args.adam_epochs),
                lbfgs_max_iter=int(args.lbfgs_max_iter),
                verbose_training=bool(args.verbose_training),
            )
            if result.get("invalid"):
                failures.append(result)
        except Exception as exc:
            failures.append({"cell_id": cell["cell_id"], "dataset_id": dataset_id, "error": repr(exc), "traceback": traceback.format_exc()})
        completed += 1
        save_json(
            shard_root / "status.json",
            {
                "timestamp": now_iso(),
                "shard_id": int(args.shard_id),
                "shard_count": int(args.shard_count),
                "gpu_id": str(args.gpu_id),
                "completed": completed,
                "total": len(shard_items),
                "failed": len(failures),
                "affinity": affinity,
            },
        )
    save_json(shard_root / "failures.json", failures)
    return 0 if (not failures or bool(args.allow_insufficient)) else 2


def run_reference_supervise(args: argparse.Namespace) -> int:
    prepare_directories()
    items = reference_work_items(int(args.start_dataset), int(args.stop_dataset))
    total = len(items)
    ensure_dir(REFERENCE_ROOT / "logs")
    shard_status_root = REFERENCE_ROOT / "logs" / "reference_shards"
    if shard_status_root.exists():
        shutil.rmtree(shard_status_root)
    ensure_dir(shard_status_root)
    env = env_for_compute(cuda=True)
    procs: list[subprocess.Popen[Any]] = []
    for shard_id in range(int(args.shards)):
        gpu_id = SAFE_GPU_IDS[shard_id % len(SAFE_GPU_IDS)]
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "reference-worker",
            "--shard-id",
            str(shard_id),
            "--shard-count",
            str(args.shards),
            "--gpu-id",
            gpu_id,
            "--device",
            str(args.device),
            "--attempts",
            str(args.attempts),
            "--attempt-batch-size",
            str(args.attempt_batch_size),
            "--adam-epochs",
            str(args.adam_epochs),
            "--lbfgs-max-iter",
            str(args.lbfgs_max_iter),
            "--start-dataset",
            str(args.start_dataset),
            "--stop-dataset",
            str(args.stop_dataset),
        ]
        if args.verbose_training:
            cmd.append("--verbose-training")
        if args.force:
            cmd.append("--force")
        if args.allow_insufficient:
            cmd.append("--allow-insufficient")
        stdout = (REFERENCE_ROOT / "logs" / f"reference_shard_{shard_id:03d}.stdout.log").open("w", encoding="utf-8")
        stderr = (REFERENCE_ROOT / "logs" / f"reference_shard_{shard_id:03d}.stderr.log").open("w", encoding="utf-8")
        worker_env = env.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        procs.append(subprocess.Popen(cmd, cwd=str(LOCAL_ROOT), env=worker_env, stdout=stdout, stderr=stderr))
    started = time.time()
    while True:
        completed = 0
        failed = 0
        for path in sorted(shard_status_root.glob("shard_*/status.json")):
            status = load_json(path, {})
            completed += int(status.get("completed", 0) or 0)
            failed += int(status.get("failed", 0) or 0)
        living = [proc for proc in procs if proc.poll() is None]
        payload = {
            "timestamp": now_iso(),
            "completed": completed,
            "total": total,
            "failed": failed,
            "workers_alive": len(living),
            "elapsed": format_duration(time.time() - started),
            "allowed_physical_gpus": SAFE_GPU_IDS,
            "cpu_affinity_cpus": CPU_AFFINITY_CPUS,
        }
        save_json(REFERENCE_ROOT / "logs" / "reference_supervisor_status.json", payload)
        print(f"[reference] {completed}/{total} failed={failed} alive={len(living)} elapsed={payload['elapsed']}", flush=True)
        if not living:
            break
        time.sleep(max(10.0, float(args.poll_seconds)))
    codes = [int(proc.returncode or 0) for proc in procs]
    write_reference_summaries()
    return 0 if all(code == 0 for code in codes) else 2


def write_reference_summaries() -> None:
    coverage_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for selected_json in sorted(REFERENCE_SELECTED.glob("cell_beta_*/dataset_*/width_048/selected_refs.json")):
        payload = load_json(selected_json, {})
        refs = payload.get("selected_refs", [])
        cell_id = selected_json.parents[2].name
        tag = selected_json.parents[1].name
        attempt_records = load_attempt_records(REFERENCE_RAW_ATTEMPTS / cell_id / tag / f"width_{WIDTH:03d}")
        exact_count = sum(1 for r in attempt_records if bool(r.get("summary", {}).get("is_exact_solution", False)) or finite_float(r.get("summary", {}).get("final_cls_err")) == 0.0)
        coverage_rows.append(
            {
                "cell_id": cell_id,
                "dataset_tag": tag,
                "width": WIDTH,
                "selected_ref_count": len(refs),
                "required_selected_refs": REFERENCES_PER_DATASET,
                "require_exact_selected_refs": True,
                "exact_count": exact_count if attempt_records else "",
                "sampling_eligible_count": len(refs),
            }
        )
        if refs:
            best = refs[0]
            best_rows.append(
                {
                    "cell_id": cell_id,
                    "dataset_tag": tag,
                    "width": WIDTH,
                    "ref_id": best.get("ref_id", 0),
                    "theta_norm": best.get("theta_norm", ""),
                    "final_train_loss": best.get("final_train_loss", ""),
                    "theta_path": best.get("theta_path", ""),
                    "selected_refs_path": repo_relative(selected_json),
                }
            )
    write_csv(REFERENCE_ROOT / "summary_tables" / "selected_ref_coverage.csv", coverage_rows, ["cell_id", "dataset_tag", "width", "selected_ref_count", "required_selected_refs", "require_exact_selected_refs", "exact_count", "sampling_eligible_count"])
    write_csv(REFERENCE_ROOT / "summary_tables" / "best_per_dataset.csv", best_rows, ["cell_id", "dataset_tag", "width", "ref_id", "theta_norm", "final_train_loss", "theta_path", "selected_refs_path"])
    (REFERENCE_ROOT / "run_report.md").write_text(
        f"# Gaussian reference search report: {RUN_REFERENCE}\n\n"
        f"- selected refs per dataset: `{REFERENCES_PER_DATASET}`\n"
        f"- selected references root: `{repo_relative(REFERENCE_SELECTED)}`\n"
        f"- coverage: `{repo_relative(REFERENCE_ROOT / 'summary_tables' / 'selected_ref_coverage.csv')}`\n",
        encoding="utf-8",
    )


def materialize_reference_pool(args: argparse.Namespace) -> int:
    prepare_directories()
    selected_files = sorted(REFERENCE_SELECTED.glob("cell_beta_*/dataset_*/width_048/selected_refs.json"))
    dataset_pools: list[dict[str, Any]] = []
    counts = {"existing": 0, "hardlink": 0, "copy": 0}
    for idx, selected_json in enumerate(selected_files, start=1):
        payload = load_json(selected_json, {})
        cell_id = selected_json.parents[2].name
        tag = selected_json.parents[1].name
        target_width = REFERENCE_POOL / cell_id / tag / f"width_{WIDTH:03d}"
        ensure_dir(target_width)
        retained_refs: list[dict[str, Any]] = []
        for ref in payload.get("selected_refs", [])[:REFERENCES_PER_DATASET]:
            ref_id = int(ref.get("ref_id", len(retained_refs)))
            ref_dir = target_width / "selected_ref_payloads" / f"ref_{ref_id:03d}"
            ensure_dir(ref_dir)
            theta_path = ref_dir / "theta.npy"
            theta_init_path = ref_dir / "theta_init.npy"
            summary_path = ref_dir / "train_summary.json"
            counts[hardlink_or_copy(resolve_repo_path(ref["theta_path"]), theta_path)] += 1
            counts[hardlink_or_copy(resolve_repo_path(ref["theta_init_path"]), theta_init_path)] += 1
            counts[hardlink_or_copy(resolve_repo_path(ref["summary_path"]), summary_path)] += 1
            out_ref = dict(ref)
            out_ref.update({"theta_path": repo_relative(theta_path), "theta_init_path": repo_relative(theta_init_path), "summary_path": repo_relative(summary_path)})
            retained_refs.append(out_ref)
        out_payload = {"cell_id": cell_id, "dataset_tag": tag, "width": WIDTH, "selected_refs": retained_refs}
        save_json(target_width / "selected_refs.json", out_payload)
        dataset_pools.append(out_payload)
        if idx % max(1, int(args.status_every)) == 0 or idx == len(selected_files):
            print(f"[pool] {idx}/{len(selected_files)} payload_counts={counts}", flush=True)
    save_json(
        REFERENCE_POOL / "final_pool1_l2_top30_refs.json",
        {
            "claim": "Gaussian baseline pool1 exact references selected by L2 norm.",
            "lambda_ref": 1.0,
            "reference_count_per_dataset": REFERENCES_PER_DATASET,
            "dataset_pools": dataset_pools,
        },
    )
    save_json(
        REFERENCE_POOL / "selection_manifest.json",
        {
            "run_name": RUN_REFERENCE,
            "target_reference_pool_root": repo_relative(REFERENCE_POOL),
            "dataset_count": len(dataset_pools),
            "selected_reference_count": sum(len(pool.get("selected_refs", [])) for pool in dataset_pools),
            "payload_materialization_counts": counts,
            "finished_at": now_iso(),
        },
    )
    coverage_src = REFERENCE_ROOT / "summary_tables" / "selected_ref_coverage.csv"
    if coverage_src.exists():
        hardlink_or_copy(coverage_src, REFERENCE_POOL / "summary_tables" / "selected_ref_coverage.csv")
    save_json(
        REFERENCE_POOL / "validation_report.json",
        {
            "ok": len(dataset_pools) == len(SELECTED_CELLS) * DATASETS_PER_BETA
            and sum(len(pool.get("selected_refs", [])) for pool in dataset_pools) == len(SELECTED_CELLS) * DATASETS_PER_BETA * REFERENCES_PER_DATASET,
            "expected_dataset_pool_count": len(SELECTED_CELLS) * DATASETS_PER_BETA,
            "dataset_pool_count": len(dataset_pools),
            "expected_selected_reference_count": len(SELECTED_CELLS) * DATASETS_PER_BETA * REFERENCES_PER_DATASET,
            "selected_reference_count": sum(len(pool.get("selected_refs", [])) for pool in dataset_pools),
        },
    )
    return 0


def write_sampling_manifest(_: argparse.Namespace) -> int:
    selected_files = sorted(REFERENCE_POOL.glob("cell_beta_*/dataset_*/width_048/selected_refs.json"))
    fields = ["beta", "cell_id", "dataset_tag", "dataset_id", "ref_id", "theta_path", "theta_init_path", "summary_path", "dataset_path"]
    count = 0
    ensure_dir(MANIFEST.parent)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for selected_json in selected_files:
            payload = load_json(selected_json, {})
            cell_id = str(payload["cell_id"])
            tag = str(payload["dataset_tag"])
            beta = float(cell_id.replace("cell_beta_", "").split("_p_")[0].replace("p", "."))
            dataset_id = int(tag.split("_")[1])
            dataset_path = DATASET_RAW / cell_id / tag / "dataset.npz"
            for ref in payload.get("selected_refs", [])[:REFERENCES_PER_DATASET]:
                writer.writerow(
                    {
                        "beta": beta,
                        "cell_id": cell_id,
                        "dataset_tag": tag,
                        "dataset_id": dataset_id,
                        "ref_id": int(ref["ref_id"]),
                        "theta_path": ref["theta_path"],
                        "theta_init_path": ref["theta_init_path"],
                        "summary_path": ref["summary_path"],
                        "dataset_path": repo_relative(dataset_path),
                    }
                )
                count += 1
    save_json(
        MANIFEST.with_suffix(".metadata.json"),
        {
            "output_manifest": repo_relative(MANIFEST),
            "selected_record_count": count,
            "expected_record_count": len(SELECTED_CELLS) * DATASETS_PER_BETA * REFERENCES_PER_DATASET,
            "projected_unit_count": count * 250,
        },
    )
    print(json.dumps({"manifest": repo_relative(MANIFEST), "rows": count}, indent=2, sort_keys=True))
    return 0 if count == len(SELECTED_CELLS) * DATASETS_PER_BETA * REFERENCES_PER_DATASET else 2


def run_checked(cmd: list[str], *, cwd: Path = LOCAL_ROOT, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd), env=env)


def run_sampling_preflight(_: argparse.Namespace) -> int:
    ensure_dir(SHELL_ROOT / "logs")
    report_path = SHELL_ROOT / "logs" / "gaussian_preflight_report.json"
    if report_path.exists():
        report_path.unlink()
    cmd = [
        sys.executable,
        str(SAMPLER),
        "preflight",
        "--config",
        str(CONFIG),
        "--manifest",
        str(MANIFEST),
        "--out",
        str(report_path),
        "--full-validate",
    ]
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(LOCAL_ROOT), env=env_for_compute(cuda=True), text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)
    payload = load_json(report_path, {})
    expected_records = len(SELECTED_CELLS) * DATASETS_PER_BETA * REFERENCES_PER_DATASET
    expected_units = expected_records * SAMPLING_RADII_COUNT
    def payload_int(key: str, default: int = -1) -> int:
        value = payload.get(key, default)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    ok = (
        payload_int("record_count") == expected_records
        and payload_int("theta_count") == expected_records
        and payload_int("dataset_count") == len(SELECTED_CELLS) * DATASETS_PER_BETA
        and payload_int("radii_count") == SAMPLING_RADII_COUNT
        and payload_int("unit_count") == expected_units
        and payload_int("missing_dataset_count") == 0
        and payload_int("missing_theta_count") == 0
    )
    save_json(
        SHELL_ROOT / "logs" / "gaussian_preflight_acceptance.json",
        {
            "timestamp": now_iso(),
            "accepted": ok,
            "sampler_returncode": int(proc.returncode),
            "expected_record_count": expected_records,
            "expected_dataset_count": len(SELECTED_CELLS) * DATASETS_PER_BETA,
            "expected_radii_count": SAMPLING_RADII_COUNT,
            "expected_unit_count": expected_units,
            "sampler_dataset_count": payload_int("dataset_count"),
            "sampler_record_count": payload_int("record_count"),
            "sampler_theta_count": payload_int("theta_count"),
            "sampler_radii_count": payload_int("radii_count"),
            "sampler_unit_count": payload_int("unit_count"),
            "sampler_missing_dataset_count": payload_int("missing_dataset_count"),
            "sampler_missing_theta_count": payload_int("missing_theta_count"),
            "sampler_report": repo_relative(report_path),
            "note": "The shared sampler reports its original 18x90x30x250 expected_unit_count; this wrapper validates the 90-dataset Gaussian baseline contract instead.",
        },
    )
    return 0 if ok else (int(proc.returncode) or 2)


def run_sampling_smoke(args: argparse.Namespace) -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = BASE_ROOT / "smoke_runs" / f"sampling_smoke_{stamp}"
    run_checked(
        [
            sys.executable,
            str(SAMPLER),
            "supervise",
            "--backend",
            "python",
            "--config",
            str(CONFIG),
            "--manifest",
            str(MANIFEST),
            "--run-root",
            str(run_root),
            "--shards",
            str(args.shards),
            "--radii",
            "0.01,0.02,0.05,0.10",
            "--max-records",
            str(args.max_records),
            "--task-limit",
            str(args.task_limit),
            "--chunk-size",
            str(args.chunk_size),
            "--device",
            str(args.device),
            "--status-every",
            "1",
            "--poll-seconds",
            "5",
            "--force",
        ],
        env=env_for_compute(cuda=True),
    )
    save_json(PROGRESS_ROOT / "latest_sampling_smoke.json", {"timestamp": now_iso(), "run_root": repo_relative(run_root)})
    return 0


def start_sampling_full(args: argparse.Namespace) -> int:
    prepare_directories()
    ensure_dir(SHELL_ROOT / "logs")
    cmd = [
        sys.executable,
        str(SAMPLER),
        "supervise",
        "--backend",
        "python",
        "--config",
        str(CONFIG),
        "--manifest",
        str(MANIFEST),
        "--run-root",
        str(SHELL_ROOT),
        "--shards",
        str(args.shards),
        "--chunk-size",
        str(args.chunk_size),
        "--device",
        str(args.device),
        "--status-every",
        str(args.status_every),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    stdout_path = SHELL_ROOT / "logs" / "supervisor.stdout.log"
    stderr_path = SHELL_ROOT / "logs" / "supervisor.stderr.log"
    env = env_for_compute(cuda=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, cwd=str(LOCAL_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    save_json(
        SHELL_ROOT / "logs" / "full_sampling_launch.json",
        {
            "timestamp": now_iso(),
            "pid": proc.pid,
            "command": cmd,
            "stdout": repo_relative(stdout_path),
            "stderr": repo_relative(stderr_path),
            "run_root": repo_relative(SHELL_ROOT),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            "cpu_affinity_cpus": env.get("COMPLEXITY_CPU_AFFINITY_CPUS", ""),
        },
    )
    print(json.dumps({"pid": proc.pid, "run_root": repo_relative(SHELL_ROOT)}, indent=2, sort_keys=True))
    return 0


def merge_sampling(_: argparse.Namespace | None = None) -> int:
    run_checked([sys.executable, str(SAMPLER), "merge-shards", "--run-root", str(SHELL_ROOT)], env=env_for_compute(cuda=False))
    return 0


def write_beta_radius_summary(run_root: Path) -> Path:
    post_ns: dict[str, Any] = {"__name__": "_gaussian_post_completion_import", "__file__": str(POST_90)}
    exec(POST_90.read_text(encoding="utf-8"), post_ns)
    return post_ns["write_beta_radius_summary"](run_root, run_root / "logs" / "postprocess_gaussian.jsonl")


def make_proxy_tables(_: argparse.Namespace | None = None) -> int:
    run_checked(
        [
            sys.executable,
            str(DNN_ROOT / "05_proxy_local_entropy" / "src" / "make_proxy_tables.py"),
            "--range-root",
            str(SHELL_ROOT),
            "--output-root",
            str(PROXY_RAW_ROOT),
        ],
        env=env_for_compute(cuda=False),
    )
    return 0


def proxy_beta_count() -> int:
    path = PROXY_RAW_ROOT / "summary_tables" / "absolute_phi_by_beta_radius.csv"
    if not path.exists():
        return 0
    betas: set[float] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            beta = finite_float(row.get("beta"))
            if math.isfinite(beta):
                betas.add(round(beta, 8))
    return len(betas)


def render_proxy_figures() -> int:
    post_ns: dict[str, Any] = {"__name__": "_gaussian_post_completion_import", "__file__": str(POST_90)}
    exec(POST_90.read_text(encoding="utf-8"), post_ns)
    post_ns["render_proxy_figures_no_accuracy"](PROXY_RAW_ROOT, PROXY_FIG_ROOT, SHELL_ROOT / "logs" / "postprocess_gaussian.jsonl")
    if proxy_beta_count() < 2:
        save_json(
            PROXY_FIG_ROOT / "energy_phase_maps" / "single_beta_skip_report.json",
            {
                "timestamp": now_iso(),
                "event": "skipped",
                "reason": "Energy phase maps need at least two beta values; the Gaussian baseline has a single beta tag.",
                "beta_count": proxy_beta_count(),
            },
        )
        return 0
    run_checked(
        [
            sys.executable,
            str(DNN_ROOT / "05_proxy_local_entropy" / "src" / "plot_energy_phase_maps.py"),
            "--summary-root",
            str(PROXY_RAW_ROOT / "summary_tables"),
            "--figures-dir",
            str(PROXY_FIG_ROOT / "energy_phase_maps"),
        ],
        env=env_for_compute(cuda=False),
    )
    return 0


def logz_full(row: dict[str, str]) -> float:
    full = finite_float(row.get("logZ_inf_full"))
    if math.isfinite(full):
        return full
    stripped = finite_float(row.get("logZ_inf_stripped", row.get("logZ_inf")))
    correction = finite_float(row.get("reference_prior_log_weight"))
    if math.isfinite(stripped) and math.isfinite(correction):
        return stripped + correction
    return stripped


def scan_dataset_energy(path: Path, *, high_beta_min: float) -> dict[tuple[float, float, int], DatasetAccum]:
    accum: dict[tuple[float, float, int], DatasetAccum] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            beta = round(finite_float(row.get("beta")), 8)
            radius = round(finite_float(row.get("radius")), 8)
            dataset_id = int(float(row.get("dataset_id") or -1))
            if not math.isfinite(beta) or beta < float(high_beta_min) or not math.isfinite(radius) or dataset_id < 0:
                continue
            value = logz_full(row)
            if math.isfinite(value):
                accum.setdefault((beta, radius, dataset_id), DatasetAccum()).add(value / PARAM_COUNT)
    return accum


def ci95(values: list[float]) -> tuple[float, float, float]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return float("nan"), float("nan"), float("nan")
    mu = sum(clean) / len(clean)
    if len(clean) < 2:
        return mu, mu, mu
    sd = math.sqrt(sum((value - mu) ** 2 for value in clean) / (len(clean) - 1))
    half = 1.96 * sd / math.sqrt(len(clean))
    return mu, mu - half, mu + half


def summarize_dataset_accum(label: str, accum: dict[tuple[float, float, int], DatasetAccum]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[float]] = {}
    for (beta, radius, _dataset_id), stats in accum.items():
        grouped.setdefault((beta, radius), []).append(stats.mean())
    rows: list[dict[str, Any]] = []
    for (beta, radius), values in sorted(grouped.items()):
        mu, lo, hi = ci95(values)
        rows.append(
            {
                "run": label,
                "beta": beta,
                "radius": radius,
                "dataset_count": len(values),
                "phi_energy_mean": mu,
                "phi_energy_ci95_low": lo,
                "phi_energy_ci95_high": hi,
                "phi_energy_ci95_half_width": (hi - lo) / 2.0 if math.isfinite(hi) and math.isfinite(lo) else float("nan"),
            }
        )
    return rows


def write_gaussian_curve_comparison(args: argparse.Namespace | None = None) -> int:
    high_beta_min = float(getattr(args, "high_beta_min", -1.0e9) if args is not None else -1.0e9)
    summary_csv = SHELL_ROOT / "summary_tables" / "sample_unit_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    curves = summarize_dataset_accum("gaussian_random_90_dataset", scan_dataset_energy(summary_csv, high_beta_min=high_beta_min))
    fields = ["run", "beta", "radius", "dataset_count", "phi_energy_mean", "phi_energy_ci95_low", "phi_energy_ci95_high", "phi_energy_ci95_half_width"]
    write_csv(PROXY_RAW_ROOT / "summary_tables" / "high_beta_curve_comparison.csv", curves, fields)
    return 0


def overlay_gaussian_energy(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    existing_detail = DNN_ROOT / "05_proxy_local_entropy" / "figures" / "high_beta_energy_derivatives_ci_30_60_90" / "energy_phi_d1_d2_ci_detail.csv"
    gaussian_curve = PROXY_RAW_ROOT / "summary_tables" / "high_beta_curve_comparison.csv"
    if not existing_detail.exists():
        raise FileNotFoundError(existing_detail)
    if not gaussian_curve.exists():
        raise FileNotFoundError(gaussian_curve)
    detail = pd.read_csv(existing_detail)
    gauss = pd.read_csv(gaussian_curve)
    metric_rows = detail[(detail["metric"] == "phi_energy") & (detail["dataset_count"] == int(args.dataset_count))]
    if float(args.max_radius) > 0:
        metric_rows = metric_rows[metric_rows["radius"] <= float(args.max_radius)]
        gauss = gauss[gauss["radius"] <= float(args.max_radius)]
    gaussian_by_radius = gauss.groupby("radius", as_index=False).agg(
        mean=("phi_energy_mean", "mean"),
        ci95_low=("phi_energy_ci95_low", "mean"),
        ci95_high=("phi_energy_ci95_high", "mean"),
    )
    ensure_dir(OVERLAY_ROOT)
    fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    betas = sorted(metric_rows["beta"].unique())
    for idx, beta in enumerate(betas):
        sub = metric_rows[metric_rows["beta"] == beta].sort_values("radius")
        color = cmap(idx / max(1, len(betas) - 1))
        ax.plot(sub["radius"], sub["mean"], color=color, linewidth=1.45, alpha=0.88, label=fr"spin $\beta={beta:g}$")
        ax.fill_between(sub["radius"], sub["ci95_low"], sub["ci95_high"], color=color, alpha=0.08, linewidth=0)
    gaussian_by_radius = gaussian_by_radius.sort_values("radius")
    ax.plot(gaussian_by_radius["radius"], gaussian_by_radius["mean"], color="black", linestyle="--", linewidth=2.1, label="Gaussian random baseline")
    ax.fill_between(gaussian_by_radius["radius"], gaussian_by_radius["ci95_low"], gaussian_by_radius["ci95_high"], color="black", alpha=0.14, linewidth=0)
    ax.set_xlabel(r"distance threshold $d$")
    ax.set_ylabel(r"$E(d)=\langle \log Z_{\mathrm{full}}\rangle/P$")
    ax.set_title("High-beta spin energy phi(d) with Gaussian random baseline")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    path = OVERLAY_ROOT / f"phi_energy_high_beta_spin_90_with_gaussian_baseline_dmax_{str(args.max_radius).replace('.', 'p')}.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    gaussian_by_radius.to_csv(OVERLAY_ROOT / "gaussian_baseline_overlay_curve.csv", index=False)
    save_json(
        OVERLAY_ROOT / "overlay_report.json",
        {
            "timestamp": now_iso(),
            "figure": repo_relative(path),
            "existing_detail": repo_relative(existing_detail),
            "gaussian_curve": repo_relative(gaussian_curve),
            "dataset_count": int(args.dataset_count),
            "max_radius": float(args.max_radius),
            "note": "Gaussian baseline is averaged over its synthetic baseline tag(s); spin comparison rows still come from the requested dataset_count slice.",
        },
    )
    print(repo_relative(path))
    return 0


def analyze_gaussian_vs_spin_phi(args: argparse.Namespace) -> int:
    run_checked(
        [
            sys.executable,
            str(PHI_ANALYSIS),
            "--dataset-count",
            str(args.dataset_count),
            "--max-radius",
            str(args.max_radius),
        ],
        env=env_for_compute(cuda=False),
    )
    return 0


def postprocess(args: argparse.Namespace) -> int:
    merge_sampling()
    write_beta_radius_summary(SHELL_ROOT)
    run_checked(
        [
            sys.executable,
            str(DNN_ROOT / "04_sampling" / "src" / "make_sampling_figures.py"),
            "--range-root",
            str(SHELL_ROOT),
            "--figure-dir",
            str(BASE_ROOT / "figures" / "04_sampling" / RUN_REFERENCE / RANGE_NAME),
        ],
        env=env_for_compute(cuda=False),
    )
    make_proxy_tables()
    render_proxy_figures()
    write_gaussian_curve_comparison(args)
    overlay_gaussian_energy(args)
    analyze_gaussian_vs_spin_phi(args)
    save_json(SHELL_ROOT / "logs" / "postprocess_done.json", {"timestamp": now_iso(), "event": "completed"})
    return 0


def status(_: argparse.Namespace | None = None) -> int:
    dataset_count = len(list(DATASET_RAW.glob("cell_beta_*/dataset_*/dataset.npz")))
    selected_files = list(REFERENCE_SELECTED.glob("cell_beta_*/dataset_*/width_048/selected_refs.json"))
    selected_count = len(selected_files)
    valid_selected_count = 0
    selected_ref_counts: list[int] = []
    for path in selected_files:
        payload = load_json(path, {})
        count = len(payload.get("selected_refs", [])) if isinstance(payload, dict) else 0
        selected_ref_counts.append(count)
        if count >= REFERENCES_PER_DATASET:
            valid_selected_count += 1
    pool_files = list(REFERENCE_POOL.glob("cell_beta_*/dataset_*/width_048/selected_refs.json"))
    pool_count = len(pool_files)
    valid_pool_count = 0
    for path in pool_files:
        payload = load_json(path, {})
        count = len(payload.get("selected_refs", [])) if isinstance(payload, dict) else 0
        if count >= REFERENCES_PER_DATASET:
            valid_pool_count += 1
    manifest_rows = 0
    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
            manifest_rows = max(0, sum(1 for _ in handle) - 1)
    sampling_status = load_json(SHELL_ROOT / "logs" / "aggregate_status.json", {})
    payload = {
        "timestamp": now_iso(),
        "dataset_count": dataset_count,
        "dataset_expected": len(SELECTED_CELLS) * DATASETS_PER_BETA,
        "selected_reference_dataset_units": selected_count,
        "valid_selected_reference_dataset_units": valid_selected_count,
        "max_selected_refs_in_dataset_unit": max(selected_ref_counts) if selected_ref_counts else 0,
        "pool_dataset_units": pool_count,
        "valid_pool_dataset_units": valid_pool_count,
        "manifest_rows": manifest_rows,
        "manifest_expected_rows": len(SELECTED_CELLS) * DATASETS_PER_BETA * REFERENCES_PER_DATASET,
        "sampling": sampling_status,
        "roots": {
            "dataset": repo_relative(DATASET_ROOT),
            "reference": repo_relative(REFERENCE_ROOT),
            "reference_pool": repo_relative(REFERENCE_POOL_ROOT),
            "shell": repo_relative(SHELL_ROOT),
            "proxy": repo_relative(PROXY_RAW_ROOT),
            "overlay": repo_relative(OVERLAY_ROOT),
        },
        "cpu_affinity_cpus": CPU_AFFINITY_CPUS,
        "safe_gpu_ids": SAFE_GPU_IDS,
    }
    save_json(PROGRESS_ROOT / "status_snapshot.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def reference_progress(_: argparse.Namespace | None = None) -> int:
    started_payload = load_json(FULL_PIPELINE_ROOT / "latest.json", {})
    supervisor = load_json(REFERENCE_ROOT / "logs" / "reference_supervisor_status.json", {})
    shard_rows: list[dict[str, Any]] = []
    for path in sorted((REFERENCE_ROOT / "logs" / "reference_shards").glob("shard_*/status.json")):
        row = load_json(path, {})
        if isinstance(row, dict):
            shard_rows.append(row)

    selected_files = list(REFERENCE_SELECTED.glob("cell_beta_*/dataset_*/width_048/selected_refs.json"))
    selected_counts: list[int] = []
    valid_units = 0
    for path in selected_files:
        payload = load_json(path, {})
        count = len(payload.get("selected_refs", [])) if isinstance(payload, dict) else 0
        selected_counts.append(count)
        if count >= REFERENCES_PER_DATASET:
            valid_units += 1

    training_rows: list[dict[str, Any]] = []
    for path in REFERENCE_RAW_ATTEMPTS.glob("cell_beta_*/dataset_*/width_048/training_status.json"):
        payload = load_json(path, {})
        if isinstance(payload, dict):
            training_rows.append(payload)
    attempt_counts = [int(row.get("attempt_count", 0) or 0) for row in training_rows]
    exact_counts = [int(row.get("exact_count", 0) or 0) for row in training_rows]
    selected_training_counts = [int(row.get("selected_ref_count", 0) or 0) for row in training_rows]

    total_expected = len(SELECTED_CELLS) * DATASETS_PER_BETA
    completed_from_shards = sum(int(row.get("completed", 0) or 0) for row in shard_rows)
    failed_from_shards = sum(int(row.get("failed", 0) or 0) for row in shard_rows)
    elapsed_seconds = 0.0
    try:
        reference_start = datetime.fromisoformat(str(started_payload.get("timestamp")))
        elapsed_seconds = max(0.0, (datetime.now().astimezone() - reference_start).total_seconds())
    except Exception:
        elapsed_seconds = 0.0
    rate_units_per_hour = 3600.0 * completed_from_shards / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    remaining_units = max(0, total_expected - completed_from_shards)
    eta_seconds = remaining_units / (rate_units_per_hour / 3600.0) if rate_units_per_hour > 0.0 else None

    payload = {
        "timestamp": now_iso(),
        "stage": "reference_supervise",
        "driver_stage_state": started_payload.get("state", ""),
        "supervisor": supervisor,
        "completed_from_shards": completed_from_shards,
        "failed_from_shards": failed_from_shards,
        "total_expected_dataset_units": total_expected,
        "valid_selected_reference_dataset_units": valid_units,
        "selected_reference_json_files": len(selected_files),
        "max_selected_refs_in_dataset_unit": max(selected_counts) if selected_counts else 0,
        "training_status_files": len(training_rows),
        "max_attempt_count_observed": max(attempt_counts) if attempt_counts else 0,
        "max_exact_count_observed": max(exact_counts) if exact_counts else 0,
        "max_selected_count_observed": max(selected_training_counts) if selected_training_counts else 0,
        "elapsed_seconds": elapsed_seconds,
        "elapsed": format_duration(elapsed_seconds),
        "rate_completed_units_per_hour": rate_units_per_hour,
        "eta_seconds_at_current_reference_rate": eta_seconds,
        "eta_at_current_reference_rate": format_duration(eta_seconds),
        "safe_gpu_ids": SAFE_GPU_IDS,
        "cpu_affinity_cpus": CPU_AFFINITY_CPUS,
    }
    ensure_dir(PROGRESS_ROOT)
    save_json(PROGRESS_ROOT / "reference_progress.json", payload)
    lines = [
        "# Gaussian Reference Progress",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- completed: `{completed_from_shards}/{total_expected}`",
        f"- failed: `{failed_from_shards}`",
        f"- valid selected reference units: `{valid_units}`",
        f"- selected refs max per unit: `{payload['max_selected_refs_in_dataset_unit']}`",
        f"- observed max attempts: `{payload['max_attempt_count_observed']}`",
        f"- observed max exact refs: `{payload['max_exact_count_observed']}`",
        f"- elapsed: `{payload['elapsed']}`",
        f"- rate: `{rate_units_per_hour:.3f}` completed units/hour",
        f"- ETA at current reference rate: `{payload['eta_at_current_reference_rate']}`",
        f"- CPU affinity: `{CPU_AFFINITY_CPUS}`",
        f"- GPUs: `{','.join(SAFE_GPU_IDS)}`",
        "",
        "This ETA only covers the reference-search stage and is noisy early in the run.",
    ]
    (PROGRESS_ROOT / "reference_progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def append_pipeline_event(payload: dict[str, Any]) -> None:
    ensure_dir(FULL_PIPELINE_ROOT)
    row = {"timestamp": now_iso(), **payload}
    with (FULL_PIPELINE_ROOT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    save_json(FULL_PIPELINE_ROOT / "latest.json", row)
    latest_md = [
        "# Gaussian Baseline Full Pipeline",
        "",
        f"- timestamp: `{row['timestamp']}`",
        f"- state: `{row.get('state', '')}`",
        f"- stage: `{row.get('stage', '')}`",
        f"- elapsed: `{row.get('elapsed', '')}`",
        f"- command: `{row.get('command_display', '')}`",
        f"- returncode: `{row.get('returncode', '')}`",
        f"- stdout: `{row.get('stdout', '')}`",
        f"- stderr: `{row.get('stderr', '')}`",
        f"- sampling status: `{repo_relative(SHELL_ROOT / 'logs' / 'aggregate_status.json')}`",
    ]
    (FULL_PIPELINE_ROOT / "latest.md").write_text("\n".join(latest_md) + "\n", encoding="utf-8")


def pipeline_stage(name: str, cmd: list[str], *, cuda: bool, started_pipeline: float) -> None:
    stdout_path = FULL_PIPELINE_ROOT / f"{name}.stdout.log"
    stderr_path = FULL_PIPELINE_ROOT / f"{name}.stderr.log"
    ensure_dir(FULL_PIPELINE_ROOT)
    env = env_for_compute(cuda=cuda)
    display = " ".join(str(part) for part in cmd)
    append_pipeline_event(
        {
            "state": "running",
            "stage": name,
            "command": cmd,
            "command_display": display,
            "stdout": repo_relative(stdout_path),
            "stderr": repo_relative(stderr_path),
            "elapsed": format_duration(time.time() - started_pipeline),
        }
    )
    started = time.time()
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"[{now_iso()}] START {name}\n$ {display}\n")
        stdout.flush()
        proc = subprocess.Popen(cmd, cwd=str(LOCAL_ROOT), env=env, stdout=stdout, stderr=stderr)
        returncode = proc.wait()
        stdout.write(f"\n[{now_iso()}] END {name} returncode={returncode} elapsed={format_duration(time.time() - started)}\n")
    append_pipeline_event(
        {
            "state": "completed" if returncode == 0 else "failed",
            "stage": name,
            "command": cmd,
            "command_display": display,
            "stdout": repo_relative(stdout_path),
            "stderr": repo_relative(stderr_path),
            "returncode": int(returncode),
            "stage_elapsed": format_duration(time.time() - started),
            "elapsed": format_duration(time.time() - started_pipeline),
        }
    )
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def wait_for_sampling_completion(poll_seconds: float, started_pipeline: float) -> None:
    status_path = SHELL_ROOT / "logs" / "aggregate_status.json"
    while True:
        payload = load_json(status_path, {})
        completed = int(payload.get("completed") or 0)
        total = int(payload.get("total") or 0)
        failed = int(payload.get("failed") or 0)
        event = str(payload.get("event", "")).lower()
        append_pipeline_event(
            {
                "state": "waiting",
                "stage": "sampling_full",
                "elapsed": format_duration(time.time() - started_pipeline),
                "sampling_completed": completed,
                "sampling_total": total,
                "sampling_failed": failed,
                "sampling_event": event,
                "sampling_eta": payload.get("eta", ""),
            }
        )
        if failed:
            raise RuntimeError(f"Sampling failed units: {failed}")
        if total > 0 and completed >= total and event == "completed":
            return
        time.sleep(max(30.0, float(poll_seconds)))


def run_full_pipeline(args: argparse.Namespace) -> int:
    started = time.time()
    try:
        pipeline_stage("prepare_dirs", [sys.executable, str(SCRIPT_PATH), "prepare-dirs"], cuda=False, started_pipeline=started)
        pipeline_stage("prepare_datasets", [sys.executable, str(SCRIPT_PATH), "prepare-datasets", "--status-every", "100"], cuda=False, started_pipeline=started)
        pipeline_stage(
            "reference_supervise",
            [
                sys.executable,
                str(SCRIPT_PATH),
                "reference-supervise",
                "--shards",
                str(args.reference_shards),
                "--attempts",
                str(args.attempts),
                "--attempt-batch-size",
                str(args.attempt_batch_size),
                "--adam-epochs",
                str(args.adam_epochs),
                "--lbfgs-max-iter",
                str(args.lbfgs_max_iter),
                "--poll-seconds",
                str(args.poll_seconds),
            ],
            cuda=True,
            started_pipeline=started,
        )
        pipeline_stage("promote_pool1", [sys.executable, str(SCRIPT_PATH), "promote-pool1"], cuda=False, started_pipeline=started)
        pipeline_stage("write_manifest", [sys.executable, str(SCRIPT_PATH), "write-manifest"], cuda=False, started_pipeline=started)
        pipeline_stage("sampling_preflight", [sys.executable, str(SCRIPT_PATH), "sampling-preflight"], cuda=True, started_pipeline=started)
        pipeline_stage("sampling_smoke", [sys.executable, str(SCRIPT_PATH), "sampling-smoke"], cuda=True, started_pipeline=started)
        pipeline_stage(
            "start_sampling_full",
            [
                sys.executable,
                str(SCRIPT_PATH),
                "start-sampling-full",
                "--shards",
                str(args.sampling_shards),
                "--chunk-size",
                str(args.chunk_size),
                "--poll-seconds",
                str(args.poll_seconds),
            ],
            cuda=True,
            started_pipeline=started,
        )
        wait_for_sampling_completion(float(args.poll_seconds), started)
        pipeline_stage("postprocess", [sys.executable, str(SCRIPT_PATH), "postprocess"], cuda=False, started_pipeline=started)
        append_pipeline_event({"state": "completed", "stage": "pipeline_completed", "elapsed": format_duration(time.time() - started), "returncode": 0})
        return 0
    except Exception as exc:
        append_pipeline_event({"state": "failed", "stage": "pipeline_failed", "elapsed": format_duration(time.time() - started), "error": repr(exc), "returncode": getattr(exc, "returncode", 1)})
        return int(getattr(exc, "returncode", 1) or 1)


def wait_postprocess(args: argparse.Namespace) -> int:
    started = time.time()
    try:
        append_pipeline_event(
            {
                "state": "running",
                "stage": "wait_postprocess",
                "elapsed": format_duration(time.time() - started),
                "sampling_status": repo_relative(SHELL_ROOT / "logs" / "aggregate_status.json"),
            }
        )
        wait_for_sampling_completion(float(args.poll_seconds), started)
        pipeline_stage(
            "postprocess",
            [
                sys.executable,
                str(SCRIPT_PATH),
                "postprocess",
                "--high-beta-min",
                str(args.high_beta_min),
                "--dataset-count",
                str(args.dataset_count),
                "--max-radius",
                str(args.max_radius),
            ],
            cuda=False,
            started_pipeline=started,
        )
        append_pipeline_event({"state": "completed", "stage": "pipeline_completed", "elapsed": format_duration(time.time() - started), "returncode": 0})
        pipeline_stage("verify_outputs", [sys.executable, str(BASE_ROOT / "scripts" / "verify_gaussian_baseline_outputs.py")], cuda=False, started_pipeline=started)
        append_pipeline_event({"state": "completed", "stage": "pipeline_verified", "elapsed": format_duration(time.time() - started), "returncode": 0})
        return 0
    except Exception as exc:
        append_pipeline_event(
            {
                "state": "failed",
                "stage": "wait_postprocess_failed",
                "elapsed": format_duration(time.time() - started),
                "error": repr(exc),
                "returncode": getattr(exc, "returncode", 1),
            }
        )
        return int(getattr(exc, "returncode", 1) or 1)


def start_full_pipeline(args: argparse.Namespace) -> int:
    ensure_dir(FULL_PIPELINE_ROOT)
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "run-full-pipeline",
        "--reference-shards",
        str(args.reference_shards),
        "--sampling-shards",
        str(args.sampling_shards),
        "--attempts",
        str(args.attempts),
        "--attempt-batch-size",
        str(args.attempt_batch_size),
        "--adam-epochs",
        str(args.adam_epochs),
        "--lbfgs-max-iter",
        str(args.lbfgs_max_iter),
        "--chunk-size",
        str(args.chunk_size),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    stdout_path = FULL_PIPELINE_ROOT / "driver.stdout.log"
    stderr_path = FULL_PIPELINE_ROOT / "driver.stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, cwd=str(LOCAL_ROOT), env=env_for_compute(cuda=True), stdout=stdout, stderr=stderr, start_new_session=True)
    payload = {
        "state": "launched",
        "stage": "full_pipeline_driver",
        "pid": proc.pid,
        "command": cmd,
        "command_display": " ".join(str(part) for part in cmd),
        "stdout": repo_relative(stdout_path),
        "stderr": repo_relative(stderr_path),
        "cuda_visible_devices": ",".join(SAFE_GPU_IDS),
        "cpu_affinity_cpus": CPU_AFFINITY_CPUS,
    }
    append_pipeline_event(payload)
    print(json.dumps({"pid": proc.pid, "progress": repo_relative(FULL_PIPELINE_ROOT / "latest.md")}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Gaussian random baseline pipeline in 06_random_gaussian_baseline.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare-dirs")
    p.set_defaults(func=prepare_directories)

    p = sub.add_parser("prepare-datasets")
    p.add_argument("--force", action="store_true")
    p.add_argument("--status-every", type=int, default=100)
    p.set_defaults(func=run_prepare_datasets)

    p = sub.add_parser("compute-complexity")
    p.set_defaults(func=compute_complexity_diagnostics)

    p = sub.add_parser("reference-worker")
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--shard-count", type=int, required=True)
    p.add_argument("--gpu-id", choices=SAFE_GPU_IDS, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attempts", type=int, default=ATTEMPTS_PER_DATASET)
    p.add_argument("--attempt-batch-size", type=int, default=REFERENCE_ATTEMPT_BATCH_SIZE)
    p.add_argument("--adam-epochs", type=int, default=REFERENCE_ADAM_EPOCHS)
    p.add_argument("--lbfgs-max-iter", type=int, default=REFERENCE_LBFGS_MAX_ITER)
    p.add_argument("--verbose-training", action="store_true")
    p.add_argument("--start-dataset", type=int, default=0)
    p.add_argument("--stop-dataset", type=int, default=DATASETS_PER_BETA)
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-insufficient", action="store_true")
    p.set_defaults(func=run_reference_worker)

    p = sub.add_parser("reference-supervise")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attempts", type=int, default=ATTEMPTS_PER_DATASET)
    p.add_argument("--attempt-batch-size", type=int, default=REFERENCE_ATTEMPT_BATCH_SIZE)
    p.add_argument("--adam-epochs", type=int, default=REFERENCE_ADAM_EPOCHS)
    p.add_argument("--lbfgs-max-iter", type=int, default=REFERENCE_LBFGS_MAX_ITER)
    p.add_argument("--verbose-training", action="store_true")
    p.add_argument("--start-dataset", type=int, default=0)
    p.add_argument("--stop-dataset", type=int, default=DATASETS_PER_BETA)
    p.add_argument("--poll-seconds", type=float, default=120.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-insufficient", action="store_true")
    p.set_defaults(func=run_reference_supervise)

    p = sub.add_parser("promote-pool1")
    p.add_argument("--status-every", type=int, default=25)
    p.set_defaults(func=materialize_reference_pool)

    p = sub.add_parser("write-manifest")
    p.set_defaults(func=write_sampling_manifest)

    p = sub.add_parser("sampling-preflight")
    p.set_defaults(func=run_sampling_preflight)

    p = sub.add_parser("sampling-smoke")
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--max-records", type=int, default=8)
    p.add_argument("--task-limit", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=run_sampling_smoke)

    p = sub.add_parser("start-sampling-full")
    p.add_argument("--shards", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--status-every", type=int, default=25)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.set_defaults(func=start_sampling_full)

    p = sub.add_parser("merge-sampling")
    p.set_defaults(func=merge_sampling)

    p = sub.add_parser("make-proxy-tables")
    p.set_defaults(func=make_proxy_tables)

    p = sub.add_parser("render-proxy-figures")
    p.set_defaults(func=lambda _args: render_proxy_figures())

    p = sub.add_parser("write-gaussian-curves")
    p.add_argument("--high-beta-min", type=float, default=-1.0e9)
    p.set_defaults(func=write_gaussian_curve_comparison)

    p = sub.add_parser("overlay")
    p.add_argument("--dataset-count", type=int, default=90)
    p.add_argument("--max-radius", type=float, default=0.30)
    p.set_defaults(func=overlay_gaussian_energy)

    p = sub.add_parser("postprocess")
    p.add_argument("--high-beta-min", type=float, default=-1.0e9)
    p.add_argument("--dataset-count", type=int, default=90)
    p.add_argument("--max-radius", type=float, default=0.30)
    p.set_defaults(func=postprocess)

    p = sub.add_parser("run-full-pipeline")
    p.add_argument("--reference-shards", type=int, default=4)
    p.add_argument("--sampling-shards", type=int, default=32)
    p.add_argument("--attempts", type=int, default=ATTEMPTS_PER_DATASET)
    p.add_argument("--attempt-batch-size", type=int, default=REFERENCE_ATTEMPT_BATCH_SIZE)
    p.add_argument("--adam-epochs", type=int, default=REFERENCE_ADAM_EPOCHS)
    p.add_argument("--lbfgs-max-iter", type=int, default=REFERENCE_LBFGS_MAX_ITER)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--poll-seconds", type=float, default=120.0)
    p.set_defaults(func=run_full_pipeline)

    p = sub.add_parser("wait-postprocess")
    p.add_argument("--poll-seconds", type=float, default=300.0)
    p.add_argument("--high-beta-min", type=float, default=-1.0e9)
    p.add_argument("--dataset-count", type=int, default=90)
    p.add_argument("--max-radius", type=float, default=0.30)
    p.set_defaults(func=wait_postprocess)

    p = sub.add_parser("start-full-pipeline")
    p.add_argument("--reference-shards", type=int, default=4)
    p.add_argument("--sampling-shards", type=int, default=32)
    p.add_argument("--attempts", type=int, default=ATTEMPTS_PER_DATASET)
    p.add_argument("--attempt-batch-size", type=int, default=REFERENCE_ATTEMPT_BATCH_SIZE)
    p.add_argument("--adam-epochs", type=int, default=REFERENCE_ADAM_EPOCHS)
    p.add_argument("--lbfgs-max-iter", type=int, default=REFERENCE_LBFGS_MAX_ITER)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--poll-seconds", type=float, default=120.0)
    p.set_defaults(func=start_full_pipeline)

    p = sub.add_parser("status")
    p.set_defaults(func=status)

    p = sub.add_parser("reference-progress")
    p.set_defaults(func=reference_progress)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command is None:
        return status(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
