from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[2]
RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_12x12_30ref_dense_0p01_to_2p50_4rule_lowtv"
RULES = ["low_tv_spectral_teacher", "real_even_odd", "teacher_nn", "random_label"]
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
from mnist14_model import MNIST14Arch, ce_and_error_np as model_ce_and_error_np
from mnist14_model import ce_error_batch_torch as model_ce_error_batch_torch
from mnist14_model import init_theta as model_init_theta
from mnist14_model import logits_np as model_logits_np
from mnist14_model import margin_stats_np as model_margin_stats_np
from mnist14_model import normalize_labels
from mnist14_vmf import log_sphere_mgf, sample_vmf, sample_vmf_batch

ARCH = MNIST14Arch(hidden_width=12)
P = ARCH.param_count


def _patch_smoke_arch() -> None:
    smoke.ARCH = ARCH
    smoke.P = P

    def init_theta_arch(seed: int, *, scale_multiplier: float = 1.0, arch: MNIST14Arch = ARCH) -> np.ndarray:
        return model_init_theta(seed, scale_multiplier=scale_multiplier, arch=ARCH)

    smoke.init_theta = init_theta_arch


_patch_smoke_arch()


class StageBlocked(RuntimeError):
    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        observed: dict[str, Any] | None = None,
        expected: dict[str, Any] | None = None,
        next_action: str = "Inspect STAGE_BLOCKED.md, fix the cause, and rerun the same stage.",
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def files_under(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [rel(p) for p in sorted(path.rglob("*")) if p.is_file()]


def dense_radii() -> list[float]:
    return [float(f"{x:.2f}") for x in np.round(np.arange(0.01, 2.50 + 0.0001, 0.01), 2)]


def load_config() -> dict[str, Any]:
    radii = dense_radii()
    return {
        "experiment_id": "mnist14_single_dataset_12x12_30ref_dense",
        "mode": "final_single_dataset",
        "identity": RUN_ROOT.name,
        "dataset": {"n_splits": 1, "n_train": 1024, "n_test": 2048, "input_dim": 196, "label_rules": RULES},
        "model": {
            "architecture": "196-12-12-1-tanh",
            "input_dim": 196,
            "hidden_width": 12,
            "hidden_layers": 2,
            "activation": "tanh",
            "P": P,
        },
        "reference_search": {"selected_refs_per_dataset": 30, "max_attempts_per_dataset": 300, "optimizer": "adam_then_lbfgs_if_needed"},
        "sampling": {
            "r0": 0.01,
            "radii": radii,
            "samples_per_ref_radius": 256,
            "lambda_reg": 1.0,
            "proposal": "exact_shell_l2_vmf_direct_is_recovery_for_remaining_units",
            "fallback_policies_enabled": False,
            "fallback_policy_note": "Disabled for 30-reference dense production run after replicated fallback units became the runtime bottleneck; QC-failed radii remain no_claim and raw dense curves are diagnostic only.",
            "recovery_note": "Existing 1024-sample adaptive SMC unit summaries are reused when present; missing or non-reusable units are filled with 256-sample direct vMF importance sampling to complete the dense grid.",
        },
        "smc": {
            "target_cess_fraction": 0.0,
            "resample_ess_fraction": 0.0,
            "max_steps": 4,
            "min_delta_t": 0.0001,
            "bisection_steps": 32,
            "mh_sweeps": 0,
            "move_kappa_factor": 80.0,
        },
        "compute": {"chunk_size": 1024, "device": os.environ.get("MNIST14_DEVICE", "auto"), "dtype": "float32"},
        "qc": {"q05_ess_fraction_min": 0.04, "max_split_logZ_per_P_diff": 0.004, "bootstrap_sd_phi_max": 0.012, "finite_unit_fraction_min": 0.90},
        "outputs": {"run_root": rel(RUN_ROOT), "summary_only_pool2": True},
        "python": sys.executable,
        "resolved_at_unix": time.time(),
    }


def write_qc(stage: str, status: str, checks: dict[str, Any], *, warnings: list[str] | None = None, hard_failures: list[str] | None = None) -> None:
    out_dir = stage_dir(stage)
    write_json(
        out_dir / "QC_STATUS.json",
        {"stage": stage, "status": status, "checks": checks, "warnings": warnings or [], "hard_failures": hard_failures or [], "files": files_under(out_dir)},
    )


def write_blocked(blocked: StageBlocked) -> None:
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

## Observed

{observed}

## Expected

{expected}

## Files Already Created

{files}

## Next Safe Action

{blocked.next_action}
""",
    )
    write_qc(blocked.stage, "blocked", {"blocked": True, "reason": blocked.reason}, hard_failures=[blocked.reason])


def run_pytest(test_path: Path, *, timeout_s: int = 600) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run([sys.executable, "-m", "pytest", str(test_path), "-q"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout_s)
    return {"returncode": proc.returncode, "elapsed_s": time.time() - started, "passed": proc.returncode == 0, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


def ce_and_error_np(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return model_ce_and_error_np(theta, x, y, arch=ARCH)


def ce_error_batch_torch(theta_batch: np.ndarray, x: np.ndarray, y: np.ndarray, *, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    return model_ce_error_batch_torch(theta_batch, x, y, chunk_size=chunk_size, device=os.environ.get("MNIST14_DEVICE", "auto"), dtype="float32", arch=ARCH)


def margin_stats_np(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return model_margin_stats_np(theta, x, y, arch=ARCH)


def teacher_logits_arch(x: np.ndarray, seed: int) -> np.ndarray:
    theta = model_init_theta(seed, scale_multiplier=1.0, arch=ARCH)
    return model_logits_np(theta, x, arch=ARCH)


def _knn_weight_graph(x: np.ndarray, k: int) -> tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
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
            weight = float(math.exp(-(float(dist) ** 2) / (2.0 * sigma * sigma)))
            edge_weight[(a, b)] = max(edge_weight.get((a, b), 0.0), weight)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    edge_i: list[int] = []
    edge_j: list[int] = []
    edge_w: list[float] = []
    for (a, b), weight in edge_weight.items():
        rows.extend([a, b])
        cols.extend([b, a])
        vals.extend([weight, weight])
        edge_i.append(a)
        edge_j.append(b)
        edge_w.append(weight)
    w_mat = coo_matrix((vals, (rows, cols)), shape=(x.shape[0], x.shape[0]), dtype=np.float64)
    return w_mat, np.asarray(edge_i, dtype=np.int64), np.asarray(edge_j, dtype=np.int64), np.asarray(edge_w, dtype=np.float64), sigma


def _edge_nmstv(y: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, edge_w: np.ndarray) -> tuple[float, float, float]:
    y = normalize_labels(y)
    total_w = float(np.sum(edge_w))
    cut_w = float(np.sum(edge_w[y[edge_i] != y[edge_j]]))
    tv = cut_w / max(total_w, 1.0e-300)
    p = float(np.mean(y == 1.0))
    baseline = 2.0 * p * (1.0 - p)
    return tv, baseline, float(tv / max(baseline, 1.0e-12))


def _max_digit_label_purity(y: np.ndarray, digits: np.ndarray) -> float:
    y = normalize_labels(y)
    digits = np.asarray(digits)
    purities = []
    for digit in sorted(np.unique(digits)):
        mask = digits == digit
        if np.any(mask):
            pos = float(np.mean(y[mask] == 1.0))
            purities.append(max(pos, 1.0 - pos))
    return float(max(purities)) if purities else float("nan")


def make_low_tv_spectral_teacher(
    x_train: np.ndarray,
    x_test: np.ndarray,
    digit_train: np.ndarray,
    y_even_train: np.ndarray,
    *,
    graph_k: int = 16,
    spectral_k: int = 8,
    max_seed: int = 1000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    w_mat, edge_i, edge_j, edge_w, sigma = _knn_weight_graph(x_train, graph_k)
    degree = np.asarray(w_mat.sum(axis=1)).ravel()
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 1.0e-300
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
    lap = eye(w_mat.shape[0], format="csr", dtype=np.float64) - diags(inv_sqrt_degree) @ w_mat.tocsr() @ diags(inv_sqrt_degree)
    eigvals, eigvecs = eigsh(lap, k=int(spectral_k) + 1, which="SM", tol=1.0e-6)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    basis = eigvecs[:, order[1 : int(spectral_k) + 1]]
    real_tv, real_baseline, real_nmstv = _edge_nmstv(y_even_train, edge_i, edge_j, edge_w)
    candidates: list[dict[str, Any]] = []
    best_any: dict[str, Any] | None = None
    for seed in range(int(max_seed)):
        rng = np.random.default_rng(seed)
        coeff = rng.normal(size=basis.shape[1])
        score = np.asarray(basis @ coeff, dtype=np.float64)
        threshold = float(np.median(score))
        y_train = np.where(score >= threshold, 1, -1).astype(np.int8)
        pos = float(np.mean(y_train == 1))
        tv, baseline, nmstv = _edge_nmstv(y_train, edge_i, edge_j, edge_w)
        corr = float(np.corrcoef(y_train.astype(np.float64), normalize_labels(y_even_train))[0, 1])
        purity = _max_digit_label_purity(y_train, digit_train)
        row = {
            "seed": int(seed),
            "score": float(nmstv + 0.5 * abs(corr) + 0.5 * purity),
            "nmstv": float(nmstv),
            "tv": float(tv),
            "baseline": float(baseline),
            "pos_fraction": pos,
            "corr_even_odd": corr,
            "max_digit_label_purity": purity,
            "threshold": threshold,
            "coefficients": coeff.tolist(),
        }
        if best_any is None or row["score"] < best_any["score"]:
            best_any = row
        if 0.48 <= pos <= 0.52 and nmstv < 0.8 * real_nmstv and abs(corr) < 0.25 and purity < 0.80:
            candidates.append(row)
    if not candidates:
        raise StageBlocked(
            "01_dataset_prepare",
            "No low_tv_spectral_teacher seed passed the requested constraints.",
            observed={"best_any": best_any, "real_even_odd_nmstv": real_nmstv},
            expected={"candidate_count": "> 0", "seed_range": f"0..{int(max_seed) - 1}"},
            next_action="Relax the purity/correlation/NMSTV constraints or increase the spectral seed search range.",
        )
    selected = min(candidates, key=lambda item: item["nmstv"])
    coeff = np.asarray(selected["coefficients"], dtype=np.float64)
    train_score = np.asarray(basis @ coeff, dtype=np.float64)
    threshold = float(selected["threshold"])
    y_train = np.where(train_score >= threshold, 1, -1).astype(np.int8)
    nn = NearestNeighbors(n_neighbors=int(graph_k), metric="euclidean")
    nn.fit(np.asarray(x_train, dtype=np.float64))
    test_dist, test_idx = nn.kneighbors(np.asarray(x_test, dtype=np.float64), return_distance=True)
    test_weight = np.exp(-(test_dist ** 2) / (2.0 * sigma * sigma))
    test_score = np.sum(test_weight * train_score[test_idx], axis=1) / np.maximum(np.sum(test_weight, axis=1), 1.0e-300)
    y_test = np.where(test_score >= threshold, 1, -1).astype(np.int8)
    metadata = {
        "definition": "low-frequency spectral graph teacher on MNIST14 train kNN graph with kNN interpolation for test labels",
        "graph_k": int(graph_k),
        "spectral_k": int(spectral_k),
        "seed_search_range": [0, int(max_seed) - 1],
        "selected_seed": int(selected["seed"]),
        "candidate_count": int(len(candidates)),
        "train_threshold": threshold,
        "train_pos_fraction": float(np.mean(y_train == 1)),
        "test_pos_fraction": float(np.mean(y_test == 1)),
        "train_tv_on_selection_graph": float(selected["tv"]),
        "train_nmstv_on_selection_graph": float(selected["nmstv"]),
        "real_even_odd_tv_on_selection_graph": float(real_tv),
        "real_even_odd_baseline_on_selection_graph": float(real_baseline),
        "real_even_odd_nmstv_on_selection_graph": float(real_nmstv),
        "nmstv_ratio_to_real_even_odd": float(selected["nmstv"] / max(real_nmstv, 1.0e-12)),
        "corr_even_odd": float(selected["corr_even_odd"]),
        "max_digit_label_purity": float(selected["max_digit_label_purity"]),
        "laplacian_eigenvalues": eigvals.tolist(),
        "sigma": float(sigma),
    }
    return y_train, y_test, metadata


def _pca2_scaled(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    emb = centered @ vt[:2].T
    scale = np.max(np.abs(emb), axis=0, keepdims=True)
    scale = np.where(scale < 1.0e-12, 1.0, scale)
    return emb / scale


def _representative_indices(emb: np.ndarray, y: np.ndarray, label: int, n: int) -> np.ndarray:
    y = normalize_labels(y)
    candidates = np.flatnonzero(y == int(label))
    if candidates.size == 0:
        return candidates
    centroid = np.mean(emb[candidates], axis=0)
    order = np.argsort(np.linalg.norm(emb[candidates] - centroid[None, :], axis=1))
    return candidates[order[: min(int(n), candidates.size)]]


def plot_label_representative_figures(
    out_dir: Path,
    x_train_raw14: np.ndarray,
    x_train_scaled: np.ndarray,
    labels: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> None:
    fig_dir = ensure_dir(out_dir / "figures" / "label_representatives")
    x_img = np.asarray(x_train_raw14, dtype=np.float32)
    if x_img.ndim == 2 and x_img.shape[1] == 196:
        x_img = x_img.reshape(-1, 14, 14)
    if x_img.ndim != 3 or x_img.shape[1:] != (14, 14):
        raise ValueError(f"expected MNIST14 image array with shape (n, 14, 14) or (n, 196), got {x_img.shape}")
    emb = _pca2_scaled(x_train_scaled)
    for rule, (y_train, _y_test, metadata) in labels.items():
        y_train = normalize_labels(y_train).astype(np.int8)
        pos_idx = _representative_indices(emb, y_train, 1, 10)
        neg_idx = _representative_indices(emb, y_train, -1, 10)
        fig, axes = plt.subplots(3, 10, figsize=(12, 4.9), gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]})
        for col in range(10):
            ax = axes[0, col]
            ax.imshow(x_img[pos_idx[col]], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("+1", rotation=0, labelpad=13, va="center")
            ax = axes[1, col]
            ax.imshow(x_img[neg_idx[col]], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("-1", rotation=0, labelpad=13, va="center")
        ax = axes[2, 0]
        for extra_ax in axes[2, 1:]:
            extra_ax.remove()
        ax.set_position([0.08, 0.08, 0.86, 0.25])
        ax.scatter(emb[y_train == -1, 0], emb[y_train == -1, 1], s=9, alpha=0.28, color="#5b6f8f", label="-1")
        ax.scatter(emb[y_train == 1, 0], emb[y_train == 1, 1], s=9, alpha=0.28, color="#b24a3f", label="+1")
        ax.scatter(emb[pos_idx, 0], emb[pos_idx, 1], s=36, facecolors="none", edgecolors="#7a1f16", linewidths=1.1)
        ax.scatter(emb[neg_idx, 0], emb[neg_idx, 1], s=36, facecolors="none", edgecolors="#1d3f6e", linewidths=1.1)
        ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.25)
        ax.axvline(0.0, color="black", linewidth=0.4, alpha=0.25)
        ax.set_xlabel("PCA1 scaled")
        ax.set_ylabel("PCA2 scaled")
        ax.legend(loc="upper right", ncols=2, fontsize=8, frameon=False)
        fig.suptitle(f"{rule}: representative labels and low-dimensional scaled input view", fontsize=11)
        note = ""
        if rule == "low_tv_spectral_teacher":
            note = (
                f"seed={metadata.get('selected_seed')}, "
                f"NMSTV ratio={float(metadata.get('nmstv_ratio_to_real_even_odd', float('nan'))):.3f}, "
                f"corr_even={float(metadata.get('corr_even_odd', float('nan'))):.3f}"
            )
        else:
            note = str(metadata.get("definition", ""))[:95]
        fig.text(0.08, 0.005, note, ha="left", va="bottom", fontsize=8)
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.94])
        fig.savefig(fig_dir / f"fig_label_representatives_{rule}.png", dpi=170)
        plt.close(fig)


def load_dataset(path_value: str | Path) -> dict[str, np.ndarray]:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    payload = np.load(path)
    return {k: payload[k] for k in payload.files}


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(logsumexp(values) - math.log(values.size)) if values.size else float("-inf")


def ess_from_logw(logw: np.ndarray) -> float:
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return 0.0
    return float(np.exp(2.0 * logsumexp(logw) - logsumexp(2.0 * logw)))


def normalize_logw(logw: np.ndarray) -> np.ndarray:
    return np.asarray(logw, dtype=np.float64) - logsumexp(logw)


def ess_fraction_from_norm(logw_norm: np.ndarray) -> float:
    logw_norm = np.asarray(logw_norm, dtype=np.float64)
    return float(np.exp(-logsumexp(2.0 * logw_norm)) / max(1, logw_norm.size))


def cess_fraction(logw_norm: np.ndarray, ce: np.ndarray, delta_t: float, gamma_ce: float) -> float:
    loga = -float(delta_t) * float(gamma_ce) * np.asarray(ce, dtype=np.float64)
    return float(np.exp(2.0 * logsumexp(logw_norm + loga) - logsumexp(logw_norm + 2.0 * loga)))


def choose_temperature(t: float, ce: np.ndarray, logw_norm: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    target = float(cfg["smc"]["target_cess_fraction"])
    gamma_ce = float(cfg["dataset"]["n_train"])
    full = cess_fraction(logw_norm, ce, 1.0 - t, gamma_ce)
    if full >= target:
        return 1.0, full
    low, high = float(t), 1.0
    for _ in range(int(cfg["smc"]["bisection_steps"])):
        mid = 0.5 * (low + high)
        val = cess_fraction(logw_norm, ce, mid - t, gamma_ce)
        if val >= target:
            low = mid
        else:
            high = mid
    out = max(low, t + float(cfg["smc"]["min_delta_t"]))
    out = min(1.0, out)
    return out, cess_fraction(logw_norm, ce, out - t, gamma_ce)


def weighted_mean(values: np.ndarray, logw: np.ndarray) -> float:
    weights = np.exp(logw - logsumexp(logw))
    return float(np.sum(weights * values))


def systematic_resample(logw_norm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.exp(normalize_logw(logw_norm))
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cdf, positions, side="left")


def bootstrap_sd(values: np.ndarray, seed: int, n_boot: int = 300) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, values.size, size=values.size)
        out.append(float(np.mean(values[idx])))
    return float(np.std(out, ddof=1))


def stage01_dataset_prepare(*, force: bool = False) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("01_dataset_prepare"))
    if (out_dir / "dataset_index.csv").exists() and not force:
        rep_fig_count = len(list((out_dir / "figures" / "label_representatives").glob("*.png")))
        write_qc(
            "01_dataset_prepare",
            "pass",
            {
                "reused": True,
                "figures_exist": len(list((out_dir / "figures").glob("*.png"))) >= 2,
                "label_representative_figure_count": rep_fig_count,
                "label_representative_figures_exist": rep_fig_count >= len(RULES),
            },
        )
        return
    raw28, digits, source_meta = smoke.load_or_fetch_mnist()
    n_train = int(cfg["dataset"]["n_train"])
    n_test = int(cfg["dataset"]["n_test"])
    even_idx = np.flatnonzero((digits % 2) == 0)
    odd_idx = np.flatnonzero((digits % 2) == 1)
    rng = np.random.default_rng(20260610)
    even_perm = rng.permutation(even_idx)
    odd_perm = rng.permutation(odd_idx)
    train_idx = np.concatenate([even_perm[: n_train // 2], odd_perm[: n_train // 2]])
    test_idx = np.concatenate([even_perm[n_train // 2 : n_train // 2 + n_test // 2], odd_perm[n_train // 2 : n_train // 2 + n_test // 2]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    x_train_raw = smoke.avgpool_14(raw28[train_idx])
    x_test_raw = smoke.avgpool_14(raw28[test_idx])
    mean = x_train_raw.mean(axis=0, keepdims=True)
    std = np.where(x_train_raw.std(axis=0, keepdims=True) < 1.0e-6, 1.0, x_train_raw.std(axis=0, keepdims=True))
    x_train = ((x_train_raw - mean) / std).astype(np.float32)
    x_test = ((x_test_raw - mean) / std).astype(np.float32)
    digit_train = digits[train_idx].astype(np.int16)
    digit_test = digits[test_idx].astype(np.int16)
    teacher_train = teacher_logits_arch(x_train, 31001)
    teacher_test = teacher_logits_arch(x_test, 31001)
    threshold = float(np.median(teacher_train))
    y_even_train = np.where((digit_train % 2) == 0, 1, -1).astype(np.int8)
    y_even_test = np.where((digit_test % 2) == 0, 1, -1).astype(np.int8)
    y_lowtv_train, y_lowtv_test, lowtv_metadata = make_low_tv_spectral_teacher(x_train, x_test, digit_train, y_even_train)
    labels = {
        "low_tv_spectral_teacher": (y_lowtv_train, y_lowtv_test, lowtv_metadata),
        "real_even_odd": (y_even_train, y_even_test, {"definition": "even digit +1, odd digit -1"}),
        "teacher_nn": (np.where(teacher_train >= threshold, 1, -1).astype(np.int8), np.where(teacher_test >= threshold, 1, -1).astype(np.int8), {"teacher_seed": 31001, "teacher_architecture": cfg["model"]["architecture"], "train_median_logit_threshold": threshold}),
        "random_label": (smoke.balanced_pm1(n_train, 41001), smoke.balanced_pm1(n_test, 42001), {"train_seed": 41001, "test_seed": 42001}),
    }
    dataset_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    split_row = {
        "split_id": 0,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "n_train": n_train,
        "n_test": n_test,
        "train_even_fraction": float(np.mean((digit_train % 2) == 0)),
        "test_even_fraction": float(np.mean((digit_test % 2) == 0)),
    }
    for rule, (y_train, y_test, metadata) in labels.items():
        ds_dir = ensure_dir(out_dir / "raw_datasets" / "split_000" / rule)
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
        write_json(ds_dir / "dataset_metadata.json", {"rule": rule, "split_id": 0, **metadata})
        pos = float(np.mean(y_train == 1))
        dataset_rows.append({"experiment_id": cfg["experiment_id"], "mode": cfg["mode"], "split_id": 0, "rule": rule, "dataset_path": rel(dataset_path), "n_train": n_train, "n_test": n_test, "input_dim": 196, "train_pos_fraction": pos})
        label_rows.append({"split_id": 0, "rule": rule, "train_pos_fraction": pos, "test_pos_fraction": float(np.mean(y_test == 1)), "train_n_pos": int(np.sum(y_train == 1)), "train_n_neg": int(np.sum(y_train == -1)), **{k: v for k, v in metadata.items() if isinstance(v, (int, float, str, bool))}})
    dataset_df = pd.DataFrame(dataset_rows)
    label_df = pd.DataFrame(label_rows)
    meta_dir = ensure_dir(out_dir / "metadata")
    write_csv(out_dir / "dataset_index.csv", dataset_df)
    write_csv(meta_dir / "split_summary.csv", pd.DataFrame([{k: v for k, v in split_row.items() if k not in {"train_indices", "test_indices"}}]))
    write_csv(meta_dir / "label_balance_summary.csv", label_df)
    write_json(meta_dir / "mnist_source.json", source_meta)
    smoke.plot_dataset_figures(out_dir, raw28, [split_row], label_df)
    plot_label_representative_figures(out_dir, x_train_raw.astype(np.float32), x_train, labels)
    write_json(out_dir / "run_config_resolved.json", cfg)
    rep_fig_count = len(list((out_dir / "figures" / "label_representatives").glob("*.png")))
    checks = {
        "dataset_rows": int(len(dataset_df)),
        "balance_min": float(label_df["train_pos_fraction"].min()),
        "balance_max": float(label_df["train_pos_fraction"].max()),
        "fig01_exists": bool((out_dir / "figures" / "fig01_mnist_28_vs_14_montage.png").exists()),
        "fig02_exists": bool((out_dir / "figures" / "fig02_label_balance_by_rule.png").exists()),
        "label_representative_figure_count": rep_fig_count,
    }
    if checks["dataset_rows"] != len(RULES) or checks["balance_min"] < 0.45 or checks["balance_max"] > 0.55 or not checks["fig01_exists"] or not checks["fig02_exists"] or checks["label_representative_figure_count"] < len(RULES):
        raise StageBlocked("01_dataset_prepare", "Dataset or figure QC failed.", observed=checks)
    write_qc("01_dataset_prepare", "pass", checks)
    write_text(out_dir / "REPORT.md", f"# Stage 01 Dataset Prepare\n\nPrepared one MNIST14 split with {len(RULES)} label rules, dataset figures, and label representative figures.\n")


def stage02_complexity_measure(*, force: bool = False) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("02_complexity_measure"))
    if (out_dir / "complexity_by_dataset.csv").exists() and not force:
        write_qc("02_complexity_measure", "pass", {"reused": True, "figure_count": len(list((out_dir / "figures").glob("*.png")))})
        return
    index_path = stage_dir("01_dataset_prepare") / "dataset_index.csv"
    if not index_path.exists():
        raise StageBlocked("02_complexity_measure", "Stage 01 dataset index is missing.", observed={"missing": rel(index_path)})
    rows = pd.read_csv(index_path).to_dict("records")
    graph_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for row in rows:
        ds = load_dataset(row["dataset_path"])
        per_k = []
        for k in [8, 16, 32]:
            metrics = smoke.graph_tv_nmstv(ds["X_train"], ds["y_train"], k)
            graph_rows.append({**row, **metrics})
            per_k.append(metrics)
        dataset_rows.append({**row, "tv_mean": float(np.mean([m["tv"] for m in per_k])), "nmstv_mean": float(np.mean([m["nmstv"] for m in per_k])), "edge_count_min": int(np.min([m["edge_count"] for m in per_k]))})
    graph_df = pd.DataFrame(graph_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    summary_df = dataset_df.groupby("rule", as_index=False).agg(nmstv_mean=("nmstv_mean", "mean"), tv_mean=("tv_mean", "mean"), n_datasets=("rule", "size"))
    write_csv(out_dir / "complexity_by_dataset.csv", dataset_df)
    write_csv(out_dir / "complexity_by_rule_summary.csv", summary_df)
    write_csv(out_dir / "graph_stats_by_dataset_k.csv", graph_df)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dataset_df["rule"], dataset_df["nmstv_mean"])
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
    ax.scatter(dataset_df["train_pos_fraction"], dataset_df["nmstv_mean"])
    for _, row in dataset_df.iterrows():
        ax.annotate(str(row["rule"]), (row["train_pos_fraction"], row["nmstv_mean"]), fontsize=8)
    ax.set_xlabel("Train +1 fraction")
    ax.set_ylabel("Mean NMSTV")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_complexity_vs_label_balance.png", dpi=160)
    plt.close(fig)
    write_json(out_dir / "run_config_resolved.json", cfg)
    checks = {"dataset_rows": len(dataset_df), "graph_rows": len(graph_df), "all_finite": bool(np.isfinite(graph_df[["tv", "nmstv", "sigma_k"]].to_numpy()).all()), "figure_count": len(list(fig_dir.glob("*.png")))}
    if checks["dataset_rows"] != len(RULES) or checks["graph_rows"] != 3 * len(RULES) or not checks["all_finite"] or checks["figure_count"] < 3:
        raise StageBlocked("02_complexity_measure", "Complexity or figure QC failed.", observed=checks)
    write_qc("02_complexity_measure", "pass", checks)
    write_text(out_dir / "REPORT.md", "# Stage 02 Complexity Measure\n\nComputed NMSTV/TV diagnostics and required figures.\n")


def stage03_pool_design() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("03_pool_design"))
    write_json(out_dir / "model_spec.json", cfg["model"])
    write_json(out_dir / "POOL_CONTRACT.json", {"pool1": "optimizer-induced exact references", "pool2": "hard L2 shells", "radii": cfg["sampling"]["radii"], "d_raw": "||theta-theta_ref||/sqrt(P)"})
    write_text(out_dir / "QC_GATES.md", "# QC Gates\n\nq05 ESS >= 0.04, split logZ/P <= 0.004, bootstrap sd phi <= 0.012; failed radii are no_claim.\n")
    write_json(out_dir / "run_config_resolved.json", cfg)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["old 16x16", "scaled 12x12", "prior 2D prod"], [3441, P, 2545], color=["#9b9b9b", "#2f6f8f", "#7a9f35"])
    ax.set_ylabel("Parameter count")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_model_parameter_scale.png", dpi=160)
    plt.close(fig)
    checks = {"P": P, "expected_P": 2533, "dense_radius_count": len(cfg["sampling"]["radii"]), "figure_exists": bool((fig_dir / "fig01_model_parameter_scale.png").exists())}
    if checks["P"] != 2533 or not checks["figure_exists"]:
        raise StageBlocked("03_pool_design", "Pool design QC failed.", observed=checks)
    write_qc("03_pool_design", "pass", checks)
    write_text(out_dir / "REPORT.md", "# Stage 03 Pool Design\n\nFixed scaled architecture, dense radii, and QC gates.\n")


def stage04_exact_reference_search(*, force: bool = False) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("04_exact_reference_search"))
    index_path = stage_dir("01_dataset_prepare") / "dataset_index.csv"
    if not index_path.exists():
        raise StageBlocked("04_exact_reference_search", "Stage 01 dataset index missing.", observed={"missing": rel(index_path)})
    target_refs = int(cfg["reference_search"]["selected_refs_per_dataset"])
    max_attempts = int(cfg["reference_search"]["max_attempts_per_dataset"])
    current_path = out_dir / "reference_index.csv"
    if current_path.exists() and not force:
        ref_df = pd.read_csv(current_path)
        counts = ref_df.groupby(["split_id", "rule"]).size()
        if len(ref_df) >= len(RULES) * target_refs and int(counts.min()) >= target_refs:
            write_qc("04_exact_reference_search", "pass", {"reused": True, "reference_rows": int(len(ref_df)), "figure_count": len(list((out_dir / "figures").glob("*.png")))})
            return
    dataset_df = pd.read_csv(index_path)
    reference_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    started = time.time()
    for dataset_id, row in enumerate(dataset_df.to_dict("records")):
        ds = load_dataset(row["dataset_path"])
        selected: list[dict[str, Any]] = []
        seed_base = 2700000 + 1000 * RULES.index(str(row["rule"]))
        attempts_used = 0
        print(f"[scaled stage04] rule={row['rule']} selected=0/{target_refs}", flush=True)
        while attempts_used < max_attempts and len(selected) < target_refs:
            batch_n = min(10, max_attempts - attempts_used)
            seeds = [seed_base + attempts_used + i for i in range(batch_n)]
            batch = smoke.train_attempt_batch(ds["X_train"], ds["y_train"], seeds, max_epochs=4200, lr=0.022)
            attempts_used += batch_n
            for result in batch:
                theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
                ce_train, err_train = ce_and_error_np(theta, ds["X_train"], ds["y_train"])
                ce_test, err_test = ce_and_error_np(theta, ds["X_test"], ds["y_test"])
                selected_flag = False
                if err_train == 0.0 and theta.size == P:
                    candidate = {"theta": theta, "attempt_seed": int(result["seed"]), "phase": str(result["phase"]), "train_error": 0.0}
                    selected_flag = smoke.select_reference(selected, candidate)
                attempt_rows.append({"dataset_id": dataset_id, "split_id": 0, "rule": str(row["rule"]), "attempt_seed": int(result["seed"]), "phase": str(result["phase"]), "epoch": int(result["epoch"]), "train_error": err_train, "test_error": err_test, "ce_mean_train": ce_train, "ce_mean_test": ce_test, "theta_norm": float(np.linalg.norm(theta)), "selected": selected_flag})
            write_csv(ensure_dir(out_dir / "attempt_logs") / "attempts.csv", pd.DataFrame(attempt_rows))
            print(f"[scaled stage04] rule={row['rule']} attempts={attempts_used} selected={len(selected)}/{target_refs}", flush=True)
        if len(selected) < target_refs:
            raise StageBlocked("04_exact_reference_search", "Insufficient exact references for scaled run.", observed={"rule": row["rule"], "selected": len(selected), "target": target_refs, "attempts": attempts_used}, next_action="Increase max_attempts_per_dataset or reduce refs_per_rule before sampling.")
        for ref_id, result in enumerate(selected[:target_refs]):
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ref_dir = ensure_dir(out_dir / "selected_reference_pool" / "split_000" / str(row["rule"]) / f"ref_{ref_id:03d}")
            theta_path = ref_dir / "theta.npy"
            np.save(theta_path, theta)
            ce_train, err_train = ce_and_error_np(theta, ds["X_train"], ds["y_train"])
            ce_test, err_test = ce_and_error_np(theta, ds["X_test"], ds["y_test"])
            summary = {"dataset_id": dataset_id, "split_id": 0, "rule": str(row["rule"]), "ref_id": ref_id, "theta_path": rel(theta_path), "dataset_path": str(row["dataset_path"]), "attempt_seed": int(result["attempt_seed"]), "optimizer_chain": str(result["phase"]), "P": int(theta.size), "train_error": err_train, "test_error": err_test, "CE_mean_train": ce_train, "CE_sum_train": ce_train * ds["X_train"].shape[0], "CE_mean_test": ce_test, "theta_norm": float(np.linalg.norm(theta)), **margin_stats_np(theta, ds["X_train"], ds["y_train"])}
            write_json(ref_dir / "ref_summary.json", summary)
            reference_rows.append(summary)
    ref_df = pd.DataFrame(reference_rows)
    write_csv(out_dir / "reference_index.csv", ref_df)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    attempts_df = pd.DataFrame(attempt_rows)
    attempts_df.groupby("rule")["selected"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Exact ref selection rate")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_reference_success_rate_by_rule.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule, sub in ref_df.groupby("rule"):
        ax.scatter(sub["theta_norm"], sub["CE_mean_train"], label=rule)
    ax.set_xlabel("theta norm")
    ax.set_ylabel("CE train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_ref_ce_norm_scatter.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ref_df.boxplot(column="min_margin", by="rule", ax=ax)
    ax.set_title("")
    fig.suptitle("")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_margin_distribution_by_rule.png", dpi=160)
    plt.close(fig)
    write_json(out_dir / "run_config_resolved.json", {**cfg, "elapsed_s": time.time() - started})
    counts = ref_df.groupby(["split_id", "rule"]).size()
    checks = {"reference_rows": int(len(ref_df)), "expected_reference_rows": len(RULES) * target_refs, "min_refs_per_rule": int(counts.min()), "all_exact": bool((ref_df["train_error"] == 0.0).all()), "theta_length_all_P": bool((ref_df["P"] == P).all()), "figure_count": len(list(fig_dir.glob("*.png")))}
    if checks["reference_rows"] != checks["expected_reference_rows"] or checks["min_refs_per_rule"] < target_refs or not checks["all_exact"] or not checks["theta_length_all_P"] or checks["figure_count"] < 3:
        raise StageBlocked("04_exact_reference_search", "Reference or figure QC failed.", observed=checks)
    write_qc("04_exact_reference_search", "pass", checks)
    write_text(out_dir / "REPORT.md", f"# Stage 04 Reference Search\n\nSelected {len(ref_df)} exact references for the scaled single-split run.\n")


def unit_summary_path(row: dict[str, Any], radius: float) -> Path:
    return stage_dir("05_pool2_pm_sais_sampling") / "unit_summaries" / "split_000" / str(row["rule"]) / f"ref_{int(row['ref_id']):03d}" / f"r_{float(radius):.2f}".replace(".", "p") / "unit_summary.json"


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
    ce_prop, err_prop = ce_error_batch_torch(theta_prop, x, y, chunk_size=int(cfg["compute"]["chunk_size"]))
    current_proj = directions @ mu
    prop_proj = proposal @ mu
    log_accept = -float(t) * float(cfg["dataset"]["n_train"]) * (ce_prop - ce) + float(base_kappa) * (prop_proj - current_proj)
    accept = np.log(rng.random(size=ce.size)) <= np.minimum(0.0, log_accept)
    if np.any(accept):
        directions[accept] = proposal[accept]
        ce[accept] = ce_prop[accept]
        err[accept] = err_prop[accept]
    return directions, ce, err, float(np.mean(accept))


def run_smc_split(
    theta_ref: np.ndarray,
    ds: dict[str, np.ndarray],
    radius: float,
    n_samples: int,
    lambda_reg: float,
    seed: int,
    cfg: dict[str, Any],
    reference_ce: float,
) -> dict[str, Any]:
    theta_ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    ref_norm = float(np.linalg.norm(theta_ref))
    if not np.isfinite(ref_norm) or ref_norm <= 0.0:
        raise ValueError("reference theta has zero or non-finite norm")
    mu = -theta_ref / ref_norm
    base_kappa = float(lambda_reg * float(radius) * ref_norm / math.sqrt(P))
    gamma_ce = float(cfg["dataset"]["n_train"])
    split_outputs: list[dict[str, Any]] = []
    for split_idx, n_particles in enumerate([n_samples // 2, n_samples - n_samples // 2]):
        rng = np.random.default_rng(int(seed) + 7919 * (split_idx + 1))
        directions = sample_vmf(mu, base_kappa, int(n_particles), rng)
        theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * directions
        ce, err = ce_error_batch_torch(theta_batch, ds["X_train"], ds["y_train"], chunk_size=int(cfg["compute"]["chunk_size"]))
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
            ess_after = ess_fraction_from_norm(logw_norm)
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
    logz_stripped = float(log_prefactor + logz_ce) if np.isfinite(logz_ce) else float("nan")
    reference_prior_log_weight = -float(lambda_reg) * ref_norm * ref_norm / (2.0 * P)
    ce = np.concatenate([s["ce"] for s in split_outputs])
    err = np.concatenate([s["err"] for s in split_outputs])
    logw = np.concatenate([math.log(counts[i] / np.sum(counts)) + split_outputs[i]["logw_norm"] for i in range(2)])
    dirs = np.concatenate([s["directions"] for s in split_outputs], axis=0)
    flat_history = [h for s in split_outputs for h in s["history"]]
    mh_acceptances = np.asarray([hrow["mh_acceptance"] for hrow in flat_history], dtype=np.float64)
    mh_acceptances = mh_acceptances[np.isfinite(mh_acceptances)]
    mean_mh_acceptance = float(np.mean(mh_acceptances)) if mh_acceptances.size else 0.0
    h = np.sqrt(2.0 * np.maximum(ce - float(reference_ce), 0.0))
    return {
        "logZ": logz_stripped,
        "logZ_CE": logz_ce,
        "logZ_inf_stripped": logz_stripped,
        "reference_prior_log_weight": reference_prior_log_weight,
        "logZ_inf_full": float(logz_stripped + reference_prior_log_weight) if np.isfinite(logz_stripped) else float("nan"),
        "split0_logZ": float(log_prefactor + logz_values[0]) if np.isfinite(logz_values[0]) else float("nan"),
        "split1_logZ": float(log_prefactor + logz_values[1]) if np.isfinite(logz_values[1]) else float("nan"),
        "split_logZ_per_P_diff": float(abs(logz_values[0] - logz_values[1]) / P) if np.all(np.isfinite(logz_values)) else float("inf"),
        "ess": float(ess_fraction_from_norm(logw) * max(1, logw.size)),
        "ess_fraction": ess_fraction_from_norm(logw),
        "weighted_ce": weighted_mean(ce, logw),
        "weighted_error": weighted_mean(err, logw),
        "weighted_h": weighted_mean(h, logw),
        "smc_completed": bool(all(s["completed"] for s in split_outputs)),
        "smc_step_count": int(max(len(s["history"]) for s in split_outputs)),
        "smc_total_step_count": int(sum(len(s["history"]) for s in split_outputs)),
        "smc_min_cess_fraction": float(np.min([hrow["cess_fraction"] for hrow in flat_history])) if flat_history else float("nan"),
        "smc_mean_mh_acceptance": mean_mh_acceptance,
        "hard_shell_distance_max_abs_err": float(np.max(np.abs(np.linalg.norm(theta_ref[None, :] + math.sqrt(P) * float(radius) * dirs - theta_ref[None, :], axis=1) / math.sqrt(P) - float(radius)))),
        "direction_unit_norm_max_abs_err": float(np.max(np.abs(np.linalg.norm(dirs, axis=1) - 1.0))),
        "kappa": base_kappa,
        "logM": log_sphere_mgf(P, base_kappa),
        "log_prefactor": log_prefactor,
    }


def reusable_unit_summary(payload: dict[str, Any], radius: float, n_samples: int, lambda_reg: float) -> bool:
    return (
        int(payload.get("n_samples", -1)) >= int(n_samples)
        and str(payload.get("sampler_method", "")) == "exact_shell_l2_vmf_adaptive_ce_tempered_smc"
        and abs(float(payload.get("lambda_reg", float("nan"))) - float(lambda_reg)) <= 1.0e-12
        and abs(float(payload.get("radius", float("nan"))) - float(radius)) <= 1.0e-12
        and math.isfinite(float(payload.get("logZ", float("nan"))))
        and math.isfinite(float(payload.get("logZ_inf_full", float("nan"))))
        and math.isfinite(float(payload.get("split_logZ_per_P_diff", float("nan"))))
        and math.isfinite(float(payload.get("smc_mean_mh_acceptance", float("nan"))))
    )


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
    stripped = [float(rep["logZ"]) for rep in replicate_rows]
    full = [float(rep["logZ_inf_full"]) for rep in replicate_rows]
    even = full[0::2]
    odd = full[1::2]
    combined_stripped = logmeanexp(np.asarray(stripped, dtype=np.float64))
    combined_full = logmeanexp(np.asarray(full, dtype=np.float64))
    combined_split = (
        float(abs(logmeanexp(np.asarray(even, dtype=np.float64)) - logmeanexp(np.asarray(odd, dtype=np.float64))) / P)
        if even and odd and math.isfinite(logmeanexp(np.asarray(even, dtype=np.float64))) and math.isfinite(logmeanexp(np.asarray(odd, dtype=np.float64)))
        else float("inf")
    )
    ess_values = [float(rep["ess_fraction"]) for rep in replicate_rows]
    split_values = [float(rep["split_logZ_per_P_diff"]) for rep in replicate_rows]
    full_weights = normalize_logw(np.asarray(full, dtype=np.float64))

    def combine_weighted_metric(name: str) -> float:
        values = np.asarray([float(rep.get(name, float("nan"))) for rep in replicate_rows], dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(np.isfinite(full_weights)):
            return float("nan")
        return float(np.sum(np.exp(full_weights) * values))

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
        "logZ": combined_stripped,
        "logZ_inf_stripped": combined_stripped,
        "reference_prior_log_weight": float(replicate_rows[0]["reference_prior_log_weight"]) if replicate_rows else float("nan"),
        "logZ_inf_full": combined_full,
        "split_logZ_per_P_diff": combined_split,
        "replicate_logZ_per_P_range": float((np.max(full) - np.min(full)) / P) if np.all(np.isfinite(full)) else float("inf"),
        "replicate_split_logZ_per_P_diff_max": float(np.max(split_values)) if split_values else float("inf"),
        "ess_fraction": float(np.mean(ess_values)) if ess_values else float("nan"),
        "ess_fraction_min": float(np.min(ess_values)) if ess_values else float("nan"),
        "weighted_ce": combine_weighted_metric("weighted_ce"),
        "weighted_error": combine_weighted_metric("weighted_error"),
        "weighted_h": combine_weighted_metric("weighted_h"),
        "smc_completed": bool(all(bool(rep["smc_completed"]) for rep in replicate_rows)),
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
    theta_ref = np.load(REPO_ROOT / str(row["theta_path"])).astype(np.float64).reshape(-1)
    started = time.time()
    replicate_rows: list[dict[str, Any]] = []
    for rep_id in range(int(replicates)):
        rep_seed = int(seed) + 1000003 * rep_id
        smc = run_smc_split(theta_ref, ds, float(radius), int(n_samples_each), float(lambda_reg), rep_seed, cfg, float(row["CE_mean_train"]))
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


def sample_pm_sais_unit(row: dict[str, Any], radius: float, cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    path = unit_summary_path(row, radius)
    n_samples = int(cfg["sampling"]["samples_per_ref_radius"])
    lambda_reg = float(cfg["sampling"]["lambda_reg"])
    if path.exists() and not force:
        payload = read_json(path)
        if reusable_unit_summary(payload, radius, n_samples, lambda_reg):
            payload["reused"] = True
            return payload
    ds = load_dataset(row["dataset_path"])
    theta_ref = np.load(REPO_ROOT / str(row["theta_path"])).astype(np.float64).reshape(-1)
    seed = 3900000 + RULES.index(str(row["rule"])) * 100000 + int(row["ref_id"]) * 1000 + int(round(float(radius) * 100))
    started = time.time()
    smc = run_smc_split(theta_ref, ds, float(radius), n_samples, lambda_reg, seed, cfg, float(row["CE_mean_train"]))
    payload = {
        "split_id": 0,
        "rule": str(row["rule"]),
        "ref_id": int(row["ref_id"]),
        "radius": float(radius),
        "n_samples": n_samples,
        "seed": seed,
        "lambda_reg": lambda_reg,
        "theta_path": row["theta_path"],
        "dataset_path": row["dataset_path"],
        "theta_ref_norm": float(np.linalg.norm(theta_ref)),
        "sampler_method": "exact_shell_l2_vmf_adaptive_ce_tempered_smc",
        "finite": bool(np.isfinite(smc["logZ"]) and np.isfinite(smc["logZ_inf_full"])),
        "elapsed_s": float(time.time() - started),
        "reused": False,
        **smc,
    }
    write_json(path, payload)
    return payload


def fallback_policy_for(rule: str, radius: float) -> dict[str, Any] | None:
    key = (str(rule), int(round(float(radius) * 100)))
    rep4 = {
        "name": "rep4_n1024_cess90_mh2",
        "replicates": 4,
        "n_samples_each": 1024,
        "target_cess_fraction": 0.90,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 160,
    }
    rep8 = {
        "name": "rep8_n1024_cess95_mh2",
        "replicates": 8,
        "n_samples_each": 1024,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 220,
    }
    rep8_2048 = {
        "name": "rep8_n2048_cess95_mh2",
        "replicates": 8,
        "n_samples_each": 2048,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 240,
    }
    kernel = {
        "name": "rep4_n1024_cess95_mh4_move20",
        "replicates": 4,
        "n_samples_each": 1024,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 4,
        "move_kappa_factor": 20.0,
        "max_steps": 240,
    }
    policy: dict[tuple[str, int], dict[str, Any]] = {
        ("real_even_odd", 15): rep4,
        ("real_even_odd", 100): rep4,
        ("teacher_nn", 15): rep4,
        ("teacher_nn", 200): rep4,
        ("random_label", 1): rep4,
        ("random_label", 100): rep4,
        ("real_even_odd", 200): rep8,
        ("teacher_nn", 100): rep8,
        ("random_label", 15): rep8_2048,
        ("teacher_nn", 250): kernel,
    }
    return policy.get(key)


def cfg_for_fallback_policy(base_cfg: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    cfg = {**base_cfg, "smc": dict(base_cfg["smc"])}
    cfg["smc"]["target_cess_fraction"] = float(policy["target_cess_fraction"])
    cfg["smc"]["mh_sweeps"] = int(policy["mh_sweeps"])
    cfg["smc"]["move_kappa_factor"] = float(policy["move_kappa_factor"])
    cfg["smc"]["max_steps"] = int(policy["max_steps"])
    return cfg


def reusable_fallback_summary(payload: dict[str, Any], radius: float, lambda_reg: float, policy: dict[str, Any]) -> bool:
    return (
        str(payload.get("sampler_method", "")) == "replicated_exact_shell_l2_vmf_adaptive_ce_tempered_smc"
        and str(payload.get("fallback_policy_name", "")) == str(policy["name"])
        and int(payload.get("replicates", -1)) == int(policy["replicates"])
        and int(payload.get("n_samples_each", -1)) == int(policy["n_samples_each"])
        and abs(float(payload.get("lambda_reg", float("nan"))) - float(lambda_reg)) <= 1.0e-12
        and abs(float(payload.get("radius", float("nan"))) - float(radius)) <= 1.0e-12
        and math.isfinite(float(payload.get("logZ_inf_full", float("nan"))))
        and math.isfinite(float(payload.get("split_logZ_per_P_diff", float("nan"))))
    )


def sample_stage05_unit(row: dict[str, Any], radius: float, cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    policy = fallback_policy_for(str(row["rule"]), float(radius)) if bool(cfg["sampling"].get("fallback_policies_enabled", True)) else None
    if policy is None:
        payload = sample_pm_sais_unit(row, radius, cfg, force=force)
        payload["fallback_policy_name"] = "baseline"
        return payload
    path = unit_summary_path(row, radius)
    lambda_reg = float(cfg["sampling"]["lambda_reg"])
    if path.exists() and not force:
        payload = read_json(path)
        if reusable_fallback_summary(payload, radius, lambda_reg, policy):
            payload["reused"] = True
            return payload
    fallback_cfg = cfg_for_fallback_policy(cfg, policy)
    seed = 9900000 + RULES.index(str(row["rule"])) * 100000 + int(row["ref_id"]) * 1000 + int(round(float(radius) * 100))
    started = time.time()
    payload = run_replicated_smc(
        row,
        float(radius),
        fallback_cfg,
        n_samples_each=int(policy["n_samples_each"]),
        replicates=int(policy["replicates"]),
        lambda_reg=lambda_reg,
        seed=seed,
    )
    payload["fallback_policy_name"] = str(policy["name"])
    payload["fallback_target_cess_fraction"] = float(policy["target_cess_fraction"])
    payload["fallback_mh_sweeps"] = int(policy["mh_sweeps"])
    payload["fallback_move_kappa_factor"] = float(policy["move_kappa_factor"])
    payload["elapsed_s"] = float(time.time() - started)
    payload["finite"] = bool(np.isfinite(payload["logZ_inf_full"]))
    payload["reused"] = False
    write_json(path, payload)
    return payload


def fallback_policy_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(cfg["sampling"].get("fallback_policies_enabled", True)):
        return []
    rows: list[dict[str, Any]] = []
    for rule in RULES:
        for radius in dense_radii():
            policy = fallback_policy_for(rule, radius)
            if policy is not None:
                rows.append({"rule": rule, "radius": float(radius), **policy})
    return rows


def sample_stage05_unit_worker(args: tuple[dict[str, Any], float, dict[str, Any], bool]) -> dict[str, Any]:
    row, radius, cfg, force = args
    return sample_stage05_unit(row, radius, cfg, force=force)


def summarize_units(unit_df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["split_id", "rule", "ref_id"]
    r0 = float(cfg["sampling"]["r0"])
    energy_col = "logZ_inf_full" if "logZ_inf_full" in unit_df.columns else "logZ"
    r0_df = unit_df[unit_df["radius"] == r0][key + [energy_col]].rename(columns={energy_col: "logZ_r0"})
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined[energy_col] - joined["logZ_r0"]) / P
    summary_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for (rule, radius), sub in joined.groupby(["rule", "radius"]):
        finite_fraction = float(np.mean(np.isfinite(sub[energy_col])))
        q05_ess = float(np.quantile(sub["ess_fraction"], 0.05))
        max_split = float(np.max(sub["split_logZ_per_P_diff"]))
        boot_sd = bootstrap_sd(sub["delta_phi_energy_unit"].to_numpy(), 49000 + RULES.index(str(rule)) * 1000 + int(round(float(radius) * 100)))
        pass_qc = finite_fraction >= float(cfg["qc"]["finite_unit_fraction_min"]) and q05_ess >= float(cfg["qc"]["q05_ess_fraction_min"]) and max_split <= float(cfg["qc"]["max_split_logZ_per_P_diff"]) and boot_sd <= float(cfg["qc"]["bootstrap_sd_phi_max"])
        row = {"rule": str(rule), "radius": float(radius), "n_units": int(len(sub)), "finite_unit_fraction": finite_fraction, "q05_ess_fraction": q05_ess, "max_split_logZ_per_P_diff": max_split, "bootstrap_sd_phi": boot_sd, "mean_logZ": float(np.mean(sub[energy_col])), "mean_delta_phi_energy": float(np.mean(sub["delta_phi_energy_unit"])), "weighted_ce_mean": float(np.mean(sub["weighted_ce"])), "qc_pass": bool(pass_qc), "claim_status": "claimable_rule_radius" if pass_qc else "no_claim"}
        summary_rows.append(row)
        qc_rows.append({k: row[k] for k in ["rule", "radius", "finite_unit_fraction", "q05_ess_fraction", "max_split_logZ_per_P_diff", "bootstrap_sd_phi", "qc_pass", "claim_status"]})
    return pd.DataFrame(summary_rows), pd.DataFrame(qc_rows)


def stage05_sampling_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling") / "pilot_runtime")
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not ref_path.exists():
        raise StageBlocked("05_pool2_pm_sais_sampling", "Reference index missing for sampling pilot.", observed={"missing": rel(ref_path)})
    ref_df = pd.read_csv(ref_path).groupby("rule").head(1).reset_index(drop=True)
    radii = [0.01, 0.15, 1.00, 2.00, 2.50]
    rows = []
    started = time.time()
    for row in ref_df.to_dict("records"):
        for radius in radii:
            print(f"[scaled pilot] rule={row['rule']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            rows.append(sample_pm_sais_unit(row, radius, cfg, force=force))
    df = pd.DataFrame(rows)
    write_csv(out_dir / "pilot_unit_summary.csv", df)
    mean_s = float(df["elapsed_s"].mean())
    full_units = len(RULES) * int(cfg["reference_search"]["selected_refs_per_dataset"]) * len(cfg["sampling"]["radii"])
    estimate = mean_s * full_units
    write_json(out_dir / "runtime_estimate.json", {"pilot_units": len(df), "pilot_elapsed_s": time.time() - started, "mean_unit_elapsed_s": mean_s, "full_units": full_units, "estimated_full_sampling_hours": estimate / 3600.0, "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()), "q05_ess_fraction": float(np.quantile(df["ess_fraction"], 0.05))})
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule, sub in df.groupby("rule"):
        ax.plot(sub["radius"], sub["split_logZ_per_P_diff"], marker="o", label=rule)
    ax.axhline(float(cfg["qc"]["max_split_logZ_per_P_diff"]), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("split logZ/P diff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_pilot_split_qc_by_rule.png", dpi=160)
    plt.close(fig)
    write_text(out_dir / "RUNTIME_ESTIMATE.md", f"# Runtime Estimate\n\nEstimated full adaptive CE-tempered PM-SAIS sampling: {estimate / 3600.0:.2f} hours.\n")


def stage05_stability_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    cfg["smc"] = dict(cfg["smc"])
    cfg["smc"]["target_cess_fraction"] = 0.90
    cfg["smc"]["mh_sweeps"] = 2
    cfg["smc"]["max_steps"] = 160
    n_samples_each = 1024
    replicates = 4
    preset_name = "rep4_n1024_cess90_mh2"
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling") / "stability_pilot" / preset_name)
    pilot_path = stage_dir("05_pool2_pm_sais_sampling") / "pilot_runtime" / "pilot_unit_summary.csv"
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not pilot_path.exists() or not ref_path.exists():
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "Cannot run stability pilot because Stage 05 pilot or Stage 04 reference index is missing.",
            observed={"pilot_exists": pilot_path.exists(), "reference_index_exists": ref_path.exists()},
            next_action="Run Stage 04 and Stage 05 pilot before stability recovery.",
        )
    pilot_df = pd.read_csv(pilot_path)
    threshold = float(cfg["qc"]["max_split_logZ_per_P_diff"])
    fail_df = pilot_df[pilot_df["split_logZ_per_P_diff"] > threshold].copy()
    if fail_df.empty:
        write_qc("05_pool2_pm_sais_sampling", "pass", {"stability_pilot": "not_needed", "pilot_failures": 0})
        return
    ref_df = pd.read_csv(ref_path)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for case_idx, failed in enumerate(fail_df.to_dict("records"), start=1):
        match = ref_df[(ref_df["rule"] == str(failed["rule"])) & (ref_df["ref_id"] == int(failed["ref_id"]))]
        if match.empty:
            raise StageBlocked(
                "05_pool2_pm_sais_sampling",
                "Stability pilot could not map a failed pilot row to a selected reference.",
                observed={"rule": failed["rule"], "ref_id": failed["ref_id"], "radius": failed["radius"]},
                next_action="Regenerate the Stage 04 reference index and Stage 05 pilot summaries.",
            )
        row = match.iloc[0].to_dict()
        radius = float(failed["radius"])
        case_name = f"{str(row['rule'])}_ref{int(row['ref_id']):03d}_r{radius:.2f}".replace(".", "p")
        case_path = out_dir / "unit_summaries" / f"{case_name}.json"
        if case_path.exists() and not force:
            payload = read_json(case_path)
            payload["reused"] = True
        else:
            seed = 6100000 + RULES.index(str(row["rule"])) * 100000 + int(row["ref_id"]) * 1000 + int(round(radius * 100))
            print(f"[stability {preset_name}] {case_idx}/{len(fail_df)} rule={row['rule']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            payload = run_replicated_smc(row, radius, cfg, n_samples_each=n_samples_each, replicates=replicates, lambda_reg=float(cfg["sampling"]["lambda_reg"]), seed=seed)
            payload["baseline_split_logZ_per_P_diff"] = float(failed["split_logZ_per_P_diff"])
            payload["baseline_ess_fraction"] = float(failed["ess_fraction"])
            payload["reused"] = False
            write_json(case_path, payload)
        rows.append(payload)
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "replicate_summaries"} for row in rows])
    df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= threshold
    df["ess_gate_pass"] = df["ess_fraction_min"] >= float(cfg["qc"]["q05_ess_fraction_min"])
    df["hard_shell_gate_pass"] = df["hard_shell_distance_max_abs_err"] <= 1.0e-8
    df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & df["hard_shell_gate_pass"] & df["smc_completed"]
    write_csv(out_dir / "stability_unit_summary.csv", df)
    write_json(
        out_dir / "run_config_resolved.json",
        {
            **cfg,
            "preset_name": preset_name,
            "baseline_failed_cases": int(len(fail_df)),
            "replicates": replicates,
            "n_samples_each": n_samples_each,
            "elapsed_s": time.time() - started,
        },
    )
    fig_dir = ensure_dir(out_dir / "figures")
    labels = [f"{r['rule']}\nr={float(r['radius']):.2f}" for r in df.to_dict("records")]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4))
    ax.bar(x, df["split_logZ_per_P_diff"].astype(float).to_numpy(), color=["#2f6f9f" if ok else "#b23a48" for ok in df["split_gate_pass"]])
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("split logZ/P diff")
    ax.set_title(preset_name)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_stability_split_qc_by_case.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4))
    ax.bar(x, df["baseline_split_logZ_per_P_diff"].astype(float).to_numpy(), label="baseline", alpha=0.65)
    ax.bar(x, df["split_logZ_per_P_diff"].astype(float).to_numpy(), label="replicated", alpha=0.75)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("split logZ/P diff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_stability_baseline_vs_replicated.png", dpi=170)
    plt.close(fig)
    checks = {
        "preset_name": preset_name,
        "cases": int(len(df)),
        "passed_cases": int(df["pilot_pass"].sum()),
        "failed_cases": int((~df["pilot_pass"]).sum()),
        "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()),
        "threshold": threshold,
        "min_ess_fraction": float(df["ess_fraction_min"].min()),
        "all_smc_completed": bool(df["smc_completed"].all()),
        "figure_count": len(list(fig_dir.glob("*.png"))),
    }
    status = "pass" if checks["failed_cases"] == 0 and checks["figure_count"] >= 2 else "blocked"
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_pool2_pm_sais_sampling/stability_pilot",
            "status": status,
            "checks": checks,
            "warnings": [],
            "hard_failures": [] if status == "pass" else ["Stability pilot failed for at least one failed baseline case."],
            "files": files_under(out_dir),
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Stability Pilot

Preset: `{preset_name}`

Cases tested: `{checks['cases']}`

Passed cases: `{checks['passed_cases']}`

Failed cases: `{checks['failed_cases']}`

Max split logZ/P diff: `{checks['max_split_logZ_per_P_diff']}` with threshold `{threshold}`.

Figures:

- `figures/fig01_stability_split_qc_by_case.png`
- `figures/fig02_stability_baseline_vs_replicated.png`
""",
    )
    if status != "pass":
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "Targeted stability pilot failed; full dense Stage 05 remains unsafe.",
            observed=checks,
            next_action="Escalate to more replicates/particles or stronger rejuvenation before any full dense Stage 05 run.",
        )


def stage05_stability_escalated_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    cfg["smc"] = dict(cfg["smc"])
    cfg["smc"]["target_cess_fraction"] = 0.95
    cfg["smc"]["mh_sweeps"] = 2
    cfg["smc"]["max_steps"] = 220
    n_samples_each = 1024
    replicates = 8
    source_preset = "rep4_n1024_cess90_mh2"
    preset_name = "rep8_n1024_cess95_mh2"
    stage05 = stage_dir("05_pool2_pm_sais_sampling")
    source_path = stage05 / "stability_pilot" / source_preset / "stability_unit_summary.csv"
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not source_path.exists() or not ref_path.exists():
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "Cannot run escalated stability pilot because the prior stability pilot or reference index is missing.",
            observed={"source_exists": source_path.exists(), "reference_index_exists": ref_path.exists()},
            next_action="Run 05_stability_pilot before 05_stability_escalated_pilot.",
        )
    source_df = pd.read_csv(source_path)
    fail_df = source_df[~source_df["pilot_pass"].astype(bool)].copy()
    out_dir = ensure_dir(stage05 / "stability_pilot" / preset_name)
    if fail_df.empty:
        write_json(out_dir / "QC_STATUS.json", {"stage": "05_pool2_pm_sais_sampling/stability_pilot", "status": "pass", "checks": {"stability_escalation": "not_needed", "source_failed_cases": 0}, "warnings": [], "hard_failures": [], "files": files_under(out_dir)})
        return
    ref_df = pd.read_csv(ref_path)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for case_idx, failed in enumerate(fail_df.to_dict("records"), start=1):
        match = ref_df[(ref_df["rule"] == str(failed["rule"])) & (ref_df["ref_id"] == int(failed["ref_id"]))]
        if match.empty:
            raise StageBlocked(
                "05_pool2_pm_sais_sampling",
                "Escalated stability pilot could not map a failed row to a selected reference.",
                observed={"rule": failed["rule"], "ref_id": failed["ref_id"], "radius": failed["radius"]},
                next_action="Regenerate the Stage 04 reference index and stability pilot summaries.",
            )
        row = match.iloc[0].to_dict()
        radius = float(failed["radius"])
        case_name = f"{str(row['rule'])}_ref{int(row['ref_id']):03d}_r{radius:.2f}".replace(".", "p")
        case_path = out_dir / "unit_summaries" / f"{case_name}.json"
        if case_path.exists() and not force:
            payload = read_json(case_path)
            payload["reused"] = True
        else:
            seed = 7200000 + RULES.index(str(row["rule"])) * 100000 + int(row["ref_id"]) * 1000 + int(round(radius * 100))
            print(f"[stability {preset_name}] {case_idx}/{len(fail_df)} rule={row['rule']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            payload = run_replicated_smc(row, radius, cfg, n_samples_each=n_samples_each, replicates=replicates, lambda_reg=float(cfg["sampling"]["lambda_reg"]), seed=seed)
            payload["source_preset"] = source_preset
            payload["source_split_logZ_per_P_diff"] = float(failed["split_logZ_per_P_diff"])
            payload["source_ess_fraction_min"] = float(failed["ess_fraction_min"])
            payload["reused"] = False
            write_json(case_path, payload)
        rows.append(payload)
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "replicate_summaries"} for row in rows])
    threshold = float(cfg["qc"]["max_split_logZ_per_P_diff"])
    df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= threshold
    df["ess_gate_pass"] = df["ess_fraction_min"] >= float(cfg["qc"]["q05_ess_fraction_min"])
    df["hard_shell_gate_pass"] = df["hard_shell_distance_max_abs_err"] <= 1.0e-8
    df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & df["hard_shell_gate_pass"] & df["smc_completed"]
    write_csv(out_dir / "stability_unit_summary.csv", df)
    write_json(
        out_dir / "run_config_resolved.json",
        {
            **cfg,
            "preset_name": preset_name,
            "source_preset": source_preset,
            "source_failed_cases": int(len(fail_df)),
            "replicates": replicates,
            "n_samples_each": n_samples_each,
            "elapsed_s": time.time() - started,
        },
    )
    fig_dir = ensure_dir(out_dir / "figures")
    labels = [f"{r['rule']}\nr={float(r['radius']):.2f}" for r in df.to_dict("records")]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4))
    ax.bar(x, df["source_split_logZ_per_P_diff"].astype(float).to_numpy(), label=source_preset, alpha=0.6)
    ax.bar(x, df["split_logZ_per_P_diff"].astype(float).to_numpy(), label=preset_name, alpha=0.8)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("split logZ/P diff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_escalated_stability_split_qc.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4))
    ax.bar(x, df["replicate_logZ_per_P_range"].astype(float).to_numpy(), color="#4d7c59")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("replicate logZ/P range")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_escalated_replicate_logz_range.png", dpi=170)
    plt.close(fig)
    checks = {
        "preset_name": preset_name,
        "source_preset": source_preset,
        "cases": int(len(df)),
        "passed_cases": int(df["pilot_pass"].sum()),
        "failed_cases": int((~df["pilot_pass"]).sum()),
        "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()),
        "threshold": threshold,
        "min_ess_fraction": float(df["ess_fraction_min"].min()),
        "all_smc_completed": bool(df["smc_completed"].all()),
        "figure_count": len(list(fig_dir.glob("*.png"))),
    }
    status = "pass" if checks["failed_cases"] == 0 and checks["figure_count"] >= 2 else "blocked"
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_pool2_pm_sais_sampling/stability_pilot",
            "status": status,
            "checks": checks,
            "warnings": [],
            "hard_failures": [] if status == "pass" else ["Escalated stability pilot failed for at least one case."],
            "files": files_under(out_dir),
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Escalated Stability Pilot

Preset: `{preset_name}`

Source preset: `{source_preset}`

Cases tested: `{checks['cases']}`

Passed cases: `{checks['passed_cases']}`

Failed cases: `{checks['failed_cases']}`

Max split logZ/P diff: `{checks['max_split_logZ_per_P_diff']}` with threshold `{threshold}`.

Figures:

- `figures/fig01_escalated_stability_split_qc.png`
- `figures/fig02_escalated_replicate_logz_range.png`
""",
    )
    if status != "pass":
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "Escalated targeted stability pilot failed; full dense Stage 05 remains unsafe.",
            observed=checks,
            next_action="Use a different rejuvenation kernel, larger per-replicate particle count, or narrow the supported radius set before any full dense Stage 05 run.",
        )


def stage05_stability_particle_pilot(*, force: bool = False) -> None:
    cfg = load_config()
    cfg["smc"] = dict(cfg["smc"])
    cfg["smc"]["target_cess_fraction"] = 0.95
    cfg["smc"]["mh_sweeps"] = 2
    cfg["smc"]["max_steps"] = 240
    n_samples_each = 2048
    replicates = 8
    source_preset = "rep8_n1024_cess95_mh2"
    preset_name = "rep8_n2048_cess95_mh2"
    stage05 = stage_dir("05_pool2_pm_sais_sampling")
    source_path = stage05 / "stability_pilot" / source_preset / "stability_unit_summary.csv"
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not source_path.exists() or not ref_path.exists():
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "Cannot run particle escalation because the prior escalated pilot or reference index is missing.",
            observed={"source_exists": source_path.exists(), "reference_index_exists": ref_path.exists()},
            next_action="Run 05_stability_escalated_pilot before 05_stability_particle_pilot.",
        )
    source_df = pd.read_csv(source_path)
    fail_df = source_df[~source_df["pilot_pass"].astype(bool)].copy()
    out_dir = ensure_dir(stage05 / "stability_pilot" / preset_name)
    if fail_df.empty:
        write_json(out_dir / "QC_STATUS.json", {"stage": "05_pool2_pm_sais_sampling/stability_pilot", "status": "pass", "checks": {"particle_escalation": "not_needed", "source_failed_cases": 0}, "warnings": [], "hard_failures": [], "files": files_under(out_dir)})
        return
    ref_df = pd.read_csv(ref_path)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for case_idx, failed in enumerate(fail_df.to_dict("records"), start=1):
        match = ref_df[(ref_df["rule"] == str(failed["rule"])) & (ref_df["ref_id"] == int(failed["ref_id"]))]
        if match.empty:
            raise StageBlocked(
                "05_pool2_pm_sais_sampling",
                "Particle stability pilot could not map a failed row to a selected reference.",
                observed={"rule": failed["rule"], "ref_id": failed["ref_id"], "radius": failed["radius"]},
                next_action="Regenerate reference and stability summaries.",
            )
        row = match.iloc[0].to_dict()
        radius = float(failed["radius"])
        case_name = f"{str(row['rule'])}_ref{int(row['ref_id']):03d}_r{radius:.2f}".replace(".", "p")
        case_path = out_dir / "unit_summaries" / f"{case_name}.json"
        if case_path.exists() and not force:
            payload = read_json(case_path)
            payload["reused"] = True
        else:
            seed = 8300000 + RULES.index(str(row["rule"])) * 100000 + int(row["ref_id"]) * 1000 + int(round(radius * 100))
            print(f"[stability {preset_name}] {case_idx}/{len(fail_df)} rule={row['rule']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            payload = run_replicated_smc(row, radius, cfg, n_samples_each=n_samples_each, replicates=replicates, lambda_reg=float(cfg["sampling"]["lambda_reg"]), seed=seed)
            payload["source_preset"] = source_preset
            payload["source_split_logZ_per_P_diff"] = float(failed["split_logZ_per_P_diff"])
            payload["source_ess_fraction_min"] = float(failed["ess_fraction_min"])
            payload["reused"] = False
            write_json(case_path, payload)
        rows.append(payload)
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "replicate_summaries"} for row in rows])
    threshold = float(cfg["qc"]["max_split_logZ_per_P_diff"])
    df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= threshold
    df["ess_gate_pass"] = df["ess_fraction_min"] >= float(cfg["qc"]["q05_ess_fraction_min"])
    df["hard_shell_gate_pass"] = df["hard_shell_distance_max_abs_err"] <= 1.0e-8
    df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & df["hard_shell_gate_pass"] & df["smc_completed"]
    write_csv(out_dir / "stability_unit_summary.csv", df)
    write_json(
        out_dir / "run_config_resolved.json",
        {
            **cfg,
            "preset_name": preset_name,
            "source_preset": source_preset,
            "source_failed_cases": int(len(fail_df)),
            "replicates": replicates,
            "n_samples_each": n_samples_each,
            "elapsed_s": time.time() - started,
        },
    )
    fig_dir = ensure_dir(out_dir / "figures")
    labels = [f"{r['rule']}\nr={float(r['radius']):.2f}" for r in df.to_dict("records")]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, df["source_split_logZ_per_P_diff"].astype(float).to_numpy(), label=source_preset, alpha=0.6)
    ax.bar(x, df["split_logZ_per_P_diff"].astype(float).to_numpy(), label=preset_name, alpha=0.8)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("split logZ/P diff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_particle_stability_split_qc.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, df["replicate_logZ_per_P_range"].astype(float).to_numpy(), color="#4d7c59")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("replicate logZ/P range")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_particle_replicate_logz_range.png", dpi=170)
    plt.close(fig)
    checks = {
        "preset_name": preset_name,
        "source_preset": source_preset,
        "cases": int(len(df)),
        "passed_cases": int(df["pilot_pass"].sum()),
        "failed_cases": int((~df["pilot_pass"]).sum()),
        "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()),
        "threshold": threshold,
        "min_ess_fraction": float(df["ess_fraction_min"].min()),
        "all_smc_completed": bool(df["smc_completed"].all()),
        "figure_count": len(list(fig_dir.glob("*.png"))),
    }
    status = "pass" if checks["failed_cases"] == 0 and checks["figure_count"] >= 2 else "blocked"
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_pool2_pm_sais_sampling/stability_pilot",
            "status": status,
            "checks": checks,
            "warnings": [],
            "hard_failures": [] if status == "pass" else ["Particle stability pilot failed for at least one case."],
            "files": files_under(out_dir),
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Particle Stability Pilot

Preset: `{preset_name}`

Source preset: `{source_preset}`

Cases tested: `{checks['cases']}`

Passed cases: `{checks['passed_cases']}`

Failed cases: `{checks['failed_cases']}`

Max split logZ/P diff: `{checks['max_split_logZ_per_P_diff']}` with threshold `{threshold}`.

Figures:

- `figures/fig01_particle_stability_split_qc.png`
- `figures/fig02_particle_replicate_logz_range.png`
""",
    )
    if status != "pass":
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "Particle-count stability pilot failed; full dense Stage 05 remains unsafe.",
            observed=checks,
            next_action="Use a different rejuvenation kernel or explicitly narrow supported radii before any full dense Stage 05 run.",
        )


def stage05_stability_kernel_scan(*, force: bool = False) -> None:
    source_preset = "rep8_n2048_cess95_mh2"
    preset_name = "kernel_scan_remaining_teacher_d2p50"
    stage05 = stage_dir("05_pool2_pm_sais_sampling")
    source_path = stage05 / "stability_pilot" / source_preset / "stability_unit_summary.csv"
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not source_path.exists() or not ref_path.exists():
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "Cannot run kernel scan because the particle stability pilot or reference index is missing.",
            observed={"source_exists": source_path.exists(), "reference_index_exists": ref_path.exists()},
            next_action="Run 05_stability_particle_pilot before 05_stability_kernel_scan.",
        )
    source_df = pd.read_csv(source_path)
    fail_df = source_df[~source_df["pilot_pass"].astype(bool)].copy()
    out_dir = ensure_dir(stage05 / "stability_pilot" / preset_name)
    if fail_df.empty:
        write_json(out_dir / "QC_STATUS.json", {"stage": "05_pool2_pm_sais_sampling/stability_pilot", "status": "pass", "checks": {"kernel_scan": "not_needed", "source_failed_cases": 0}, "warnings": [], "hard_failures": [], "files": files_under(out_dir)})
        return
    ref_df = pd.read_csv(ref_path)
    variants = [
        {"name": "rep4_n1024_cess95_mh4_move20", "replicates": 4, "n_samples_each": 1024, "target_cess_fraction": 0.95, "mh_sweeps": 4, "move_kappa_factor": 20.0, "max_steps": 240},
        {"name": "rep4_n1024_cess95_mh4_move40", "replicates": 4, "n_samples_each": 1024, "target_cess_fraction": 0.95, "mh_sweeps": 4, "move_kappa_factor": 40.0, "max_steps": 240},
        {"name": "rep4_n1024_cess95_mh4_move160", "replicates": 4, "n_samples_each": 1024, "target_cess_fraction": 0.95, "mh_sweeps": 4, "move_kappa_factor": 160.0, "max_steps": 240},
        {"name": "rep4_n1024_cess98_mh4_move40", "replicates": 4, "n_samples_each": 1024, "target_cess_fraction": 0.98, "mh_sweeps": 4, "move_kappa_factor": 40.0, "max_steps": 320},
    ]
    rows: list[dict[str, Any]] = []
    started = time.time()
    for failed in fail_df.to_dict("records"):
        match = ref_df[(ref_df["rule"] == str(failed["rule"])) & (ref_df["ref_id"] == int(failed["ref_id"]))]
        if match.empty:
            raise StageBlocked(
                "05_pool2_pm_sais_sampling",
                "Kernel scan could not map a failed row to a selected reference.",
                observed={"rule": failed["rule"], "ref_id": failed["ref_id"], "radius": failed["radius"]},
                next_action="Regenerate reference and stability summaries.",
            )
        row = match.iloc[0].to_dict()
        radius = float(failed["radius"])
        for variant_idx, variant in enumerate(variants, start=1):
            cfg = load_config()
            cfg["smc"] = dict(cfg["smc"])
            cfg["smc"]["target_cess_fraction"] = float(variant["target_cess_fraction"])
            cfg["smc"]["mh_sweeps"] = int(variant["mh_sweeps"])
            cfg["smc"]["move_kappa_factor"] = float(variant["move_kappa_factor"])
            cfg["smc"]["max_steps"] = int(variant["max_steps"])
            case_name = f"{variant['name']}_{str(row['rule'])}_ref{int(row['ref_id']):03d}_r{radius:.2f}".replace(".", "p")
            case_path = out_dir / "unit_summaries" / f"{case_name}.json"
            if case_path.exists() and not force:
                payload = read_json(case_path)
                payload["reused"] = True
            else:
                seed = 9400000 + 10000 * variant_idx + RULES.index(str(row["rule"])) * 100000 + int(row["ref_id"]) * 1000 + int(round(radius * 100))
                print(f"[kernel scan] {variant['name']} rule={row['rule']} ref={row['ref_id']} r={radius:.2f}", flush=True)
                payload = run_replicated_smc(
                    row,
                    radius,
                    cfg,
                    n_samples_each=int(variant["n_samples_each"]),
                    replicates=int(variant["replicates"]),
                    lambda_reg=float(cfg["sampling"]["lambda_reg"]),
                    seed=seed,
                )
                payload["source_preset"] = source_preset
                payload["source_split_logZ_per_P_diff"] = float(failed["split_logZ_per_P_diff"])
                payload["variant_name"] = str(variant["name"])
                payload["variant_target_cess_fraction"] = float(variant["target_cess_fraction"])
                payload["variant_mh_sweeps"] = int(variant["mh_sweeps"])
                payload["variant_move_kappa_factor"] = float(variant["move_kappa_factor"])
                payload["reused"] = False
                write_json(case_path, payload)
            rows.append(payload)
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "replicate_summaries"} for row in rows])
    threshold = float(load_config()["qc"]["max_split_logZ_per_P_diff"])
    df["split_gate_pass"] = df["split_logZ_per_P_diff"] <= threshold
    df["ess_gate_pass"] = df["ess_fraction_min"] >= float(load_config()["qc"]["q05_ess_fraction_min"])
    df["hard_shell_gate_pass"] = df["hard_shell_distance_max_abs_err"] <= 1.0e-8
    df["pilot_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & df["hard_shell_gate_pass"] & df["smc_completed"]
    write_csv(out_dir / "kernel_scan_summary.csv", df)
    write_json(
        out_dir / "run_config_resolved.json",
        {
            "source_preset": source_preset,
            "preset_name": preset_name,
            "source_failed_cases": int(len(fail_df)),
            "variants": variants,
            "elapsed_s": time.time() - started,
            "qc_threshold": threshold,
        },
    )
    fig_dir = ensure_dir(out_dir / "figures")
    plot_df = df.sort_values("split_logZ_per_P_diff")
    labels = [str(v) for v in plot_df["variant_name"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.7), 4))
    ax.bar(x, plot_df["split_logZ_per_P_diff"].astype(float).to_numpy(), color=["#2f6f9f" if ok else "#b23a48" for ok in plot_df["split_gate_pass"]])
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("split logZ/P diff")
    ax.set_title("kernel scan remaining Stage 05 case")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_kernel_scan_split_qc.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.7), 4))
    ax.bar(x, plot_df["replicate_logZ_per_P_range"].astype(float).to_numpy(), color="#4d7c59")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("replicate logZ/P range")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_kernel_scan_replicate_range.png", dpi=170)
    plt.close(fig)
    passing = df[df["pilot_pass"]].sort_values("split_logZ_per_P_diff").to_dict("records")
    checks = {
        "preset_name": preset_name,
        "source_preset": source_preset,
        "cases": int(len(fail_df)),
        "variants": int(len(variants)),
        "passing_variants": int(len(passing)),
        "best_variant": str(passing[0]["variant_name"]) if passing else None,
        "best_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].min()) if len(df) else float("nan"),
        "threshold": threshold,
        "figure_count": len(list(fig_dir.glob("*.png"))),
    }
    status = "pass" if checks["passing_variants"] > 0 and checks["figure_count"] >= 2 else "blocked"
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "05_pool2_pm_sais_sampling/stability_pilot",
            "status": status,
            "checks": checks,
            "warnings": [],
            "hard_failures": [] if status == "pass" else ["Kernel scan found no passing variant for the remaining case."],
            "files": files_under(out_dir),
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 Kernel Scan

Source preset: `{source_preset}`

Remaining failed cases: `{checks['cases']}`

Variants tested: `{checks['variants']}`

Passing variants: `{checks['passing_variants']}`

Best variant: `{checks['best_variant']}`

Best split logZ/P diff: `{checks['best_split_logZ_per_P_diff']}` with threshold `{threshold}`.

Figures:

- `figures/fig01_kernel_scan_split_qc.png`
- `figures/fig02_kernel_scan_replicate_range.png`
""",
    )
    if status != "pass":
        raise StageBlocked(
            "05_pool2_pm_sais_sampling",
            "Kernel scan found no passing variant for the remaining Stage 05 case.",
            observed=checks,
            next_action="Explicitly narrow supported radii or implement a stronger non-local rejuvenation kernel before full dense Stage 05.",
        )


def stage05_pool2_pm_sais_sampling(*, force: bool = False, max_units: int | None = None, workers: int = 1) -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("05_pool2_pm_sais_sampling"))
    ref_path = stage_dir("04_exact_reference_search") / "reference_index.csv"
    if not ref_path.exists():
        raise StageBlocked("05_pool2_pm_sais_sampling", "Reference index missing.", observed={"missing": rel(ref_path)})
    ref_df = pd.read_csv(ref_path)
    tasks = [(row, float(radius)) for row in ref_df.to_dict("records") for radius in cfg["sampling"]["radii"]]
    if max_units is not None:
        tasks = tasks[: int(max_units)]
    rows = []
    started = time.time()
    worker_count = max(1, int(workers))
    if worker_count == 1:
        for idx, (row, radius) in enumerate(tasks, start=1):
            if idx == 1 or idx % 100 == 0 or idx == len(tasks):
                print(f"[scaled stage05] unit {idx}/{len(tasks)} rule={row['rule']} ref={row['ref_id']} r={radius:.2f}", flush=True)
            rows.append(sample_stage05_unit(row, radius, cfg, force=force))
    else:
        print(f"[scaled stage05] prepared {len(tasks)} units with workers={worker_count}", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {
                executor.submit(sample_stage05_unit_worker, (row, radius, cfg, force)): (row, radius)
                for row, radius in tasks
            }
            for idx, future in enumerate(as_completed(future_to_task), start=1):
                row, radius = future_to_task[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    raise StageBlocked(
                        "05_pool2_pm_sais_sampling",
                        "Parallel PM-SAIS worker failed.",
                        observed={"rule": row["rule"], "ref_id": row["ref_id"], "radius": radius, "error": repr(exc)},
                        next_action="Rerun Stage 05 with workers=1 for a precise traceback or fix the failing worker payload.",
                    ) from exc
                if idx == 1 or idx % 100 == 0 or idx == len(tasks):
                    elapsed = time.time() - started
                    rate = idx / max(elapsed, 1.0e-9)
                    remaining = (len(tasks) - idx) / max(rate, 1.0e-9)
                    print(f"[scaled stage05] completed {idx}/{len(tasks)} elapsed_h={elapsed / 3600.0:.2f} eta_h={remaining / 3600.0:.2f}", flush=True)
    unit_df = pd.DataFrame(rows)
    if max_units is not None:
        write_csv(out_dir / f"shell_summary_by_unit_partial_{max_units}.csv", unit_df)
        write_json(out_dir / "partial_run_status.json", {"max_units": max_units, "elapsed_s": time.time() - started})
        return
    write_csv(out_dir / "shell_summary_by_unit.csv", unit_df)
    summary_df, qc_df = summarize_units(unit_df, cfg)
    write_csv(out_dir / "shell_summary_by_rule_radius.csv", summary_df)
    write_csv(out_dir / "qc_by_rule_radius.csv", qc_df)
    policy_rows = fallback_policy_rows(cfg)
    if policy_rows:
        write_csv(out_dir / "fallback_sampling_policy.csv", pd.DataFrame(policy_rows))
    write_json(out_dir / "selected_lambda.json", {"lambda_reg": cfg["sampling"]["lambda_reg"], "selection_rule": "single scaled run fixed lambda=1.0"})
    write_json(out_dir / "run_config_resolved.json", {**cfg, "fallback_policy_rows": policy_rows, "elapsed_s": time.time() - started})
    fig_dir = ensure_dir(out_dir / "figures")
    for field, fname in [("q05_ess_fraction", "fig02_sampling_qc_ess_heatmap.png"), ("max_split_logZ_per_P_diff", "fig03_sampling_qc_split_logz_heatmap.png"), ("weighted_ce_mean", "fig05_weighted_ce_by_rule_radius.png")]:
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
    fig, ax = plt.subplots(figsize=(8, 4))
    for rule, sub in summary_df.groupby("rule"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["mean_delta_phi_energy"], label=rule, linewidth=1)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("mean delta phi energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_phi_energy_preview_by_rule.png", dpi=160)
    plt.close(fig)
    common_pass = sorted(set.intersection(*[set(qc_df[(qc_df["rule"] == rule) & (qc_df["qc_pass"])]["radius"]) for rule in RULES]))
    checks = {"unit_rows": int(len(unit_df)), "expected_unit_rows": len(RULES) * int(cfg["reference_search"]["selected_refs_per_dataset"]) * len(cfg["sampling"]["radii"]), "rule_radius_rows": int(len(summary_df)), "all_logZ_finite": bool(np.isfinite(unit_df["logZ"]).all()), "all_logZ_inf_full_finite": bool(np.isfinite(unit_df["logZ_inf_full"]).all()), "all_smc_completed": bool(unit_df["smc_completed"].all()), "hard_shell_max_abs_err": float(unit_df["hard_shell_distance_max_abs_err"].max()), "common_pass_radii_count": int(len(common_pass)), "fallback_policy_groups": len(policy_rows), "fallback_unit_rows": int(np.sum(unit_df.get("fallback_policy_name", "baseline") != "baseline")) if "fallback_policy_name" in unit_df else 0, "figure_count": len(list(fig_dir.glob("*.png")))}
    if checks["unit_rows"] != checks["expected_unit_rows"] or not checks["all_logZ_finite"] or not checks["all_logZ_inf_full_finite"] or not checks["all_smc_completed"] or checks["hard_shell_max_abs_err"] > 1.0e-8 or checks["figure_count"] < 4:
        raise StageBlocked("05_pool2_pm_sais_sampling", "Sampling or figure QC failed.", observed=checks)
    sample_counts = sorted(int(v) for v in unit_df["n_samples"].dropna().unique()) if "n_samples" in unit_df else []
    write_qc("05_pool2_pm_sais_sampling", "pass", checks, warnings=[f"{int((~qc_df['qc_pass']).sum())} rule/radius rows are no_claim."])
    write_text(
        out_dir / "REPORT.md",
        f"""# Stage 05 PM-SAIS Sampling

Completed {len(unit_df)} shell units with {checks['fallback_unit_rows']} fallback-policy units.

Sample counts present: {sample_counts}

Sampling note: {cfg['sampling'].get('recovery_note', 'n/a')}

QC-failed rule/radius rows remain `no_claim`; dense raw curves are diagnostic only.
""",
    )


def stage06_results_figures() -> None:
    cfg = load_config()
    out_dir = ensure_dir(stage_dir("06_results_figures"))
    stage05 = stage_dir("05_pool2_pm_sais_sampling")
    unit_path = stage05 / "shell_summary_by_unit.csv"
    qc_path = stage05 / "qc_by_rule_radius.csv"
    if not unit_path.exists() or not qc_path.exists():
        raise StageBlocked("06_results_figures", "Stage 05 summaries missing.", observed={"unit": unit_path.exists(), "qc": qc_path.exists()})
    unit_df = pd.read_csv(unit_path)
    qc_df = pd.read_csv(qc_path)
    r0 = float(cfg["sampling"]["r0"])
    key = ["split_id", "rule", "ref_id"]
    energy_col = "logZ_inf_full" if "logZ_inf_full" in unit_df.columns else "logZ"
    r0_df = unit_df[unit_df["radius"] == r0][key + [energy_col]].rename(columns={energy_col: "logZ_r0"})
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined[energy_col] - joined["logZ_r0"]) / P
    common_pass = sorted(set.intersection(*[set(qc_df[(qc_df["rule"] == rule) & (qc_df["qc_pass"])]["radius"]) for rule in RULES]))
    raw_phi_rows: list[dict[str, Any]] = []
    phi_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for (rule, radius), sub in joined.groupby(["rule", "radius"]):
        qc_match = qc_df[(qc_df["rule"] == rule) & (qc_df["radius"] == radius)]
        qc_pass = bool(qc_match["qc_pass"].iloc[0]) if not qc_match.empty else False
        values = sub["delta_phi_energy_unit"].to_numpy()
        raw_mean = float(np.mean(values))
        raw_sd = bootstrap_sd(values, 57000 + RULES.index(str(rule)) * 1000 + int(round(float(radius) * 100)))
        raw_phi_rows.append(
            {
                "rule": str(rule),
                "radius": float(radius),
                "d0": r0,
                "delta_phi_energy": raw_mean,
                "delta_phi_full": float(((P - 1) / P) * math.log(float(radius) / r0) + raw_mean),
                "n_units": int(len(sub)),
                "qc_pass": qc_pass,
                "diagnostic_status": "claimable_rule_radius" if qc_pass else "raw_no_claim",
                "bootstrap_sd": raw_sd,
            }
        )
        qc_match = qc_df[(qc_df["rule"] == rule) & (qc_df["radius"] == radius)]
        if qc_match.empty or not bool(qc_match["qc_pass"].iloc[0]):
            continue
        mean = float(np.mean(values))
        sd = bootstrap_sd(values, 58000 + RULES.index(str(rule)) * 1000 + int(round(float(radius) * 100)))
        phi_rows.append({"rule": str(rule), "radius": float(radius), "d0": r0, "delta_phi_energy": mean, "delta_phi_full": float(((P - 1) / P) * math.log(float(radius) / r0) + mean), "n_units": int(len(sub)), "qc_pass": True})
        boot_rows.append({"rule": str(rule), "radius": float(radius), "delta_phi_energy_mean": mean, "bootstrap_sd": sd, "ci95_low": mean - 1.96 * sd, "ci95_high": mean + 1.96 * sd})
    raw_phi_df = pd.DataFrame(raw_phi_rows)
    phi_df = pd.DataFrame(
        phi_rows,
        columns=["rule", "radius", "d0", "delta_phi_energy", "delta_phi_full", "n_units", "qc_pass"],
    )
    boot_df = pd.DataFrame(
        boot_rows,
        columns=["rule", "radius", "delta_phi_energy_mean", "bootstrap_sd", "ci95_low", "ci95_high"],
    )
    claim_df = pd.DataFrame([{"radius": float(radius), "claim_status": "supported" if float(radius) in common_pass else "no_claim", "rules_passed": ";".join(sorted(qc_df[(qc_df["radius"] == radius) & (qc_df["qc_pass"])]["rule"].tolist())), "rules_required": ";".join(RULES)} for radius in sorted(qc_df["radius"].unique())])
    write_csv(out_dir / "phi_by_rule_radius_raw_diagnostic.csv", raw_phi_df)
    write_csv(out_dir / "phi_by_rule_radius.csv", phi_df)
    write_csv(out_dir / "phi_bootstrap_by_rule_radius.csv", boot_df)
    write_csv(out_dir / "qc_pass_by_rule_radius.csv", qc_df)
    write_csv(out_dir / "final_claim_table.csv", claim_df)
    fig_dir = ensure_dir(out_dir / "figures")
    for field, fname, ylabel in [("delta_phi_energy", "fig04_phi_energy_qc_pass_main.png", "Delta phi energy"), ("delta_phi_full", "fig05_phi_full_qc_pass.png", "Delta phi full")]:
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
    for field, fname, ylabel in [("delta_phi_energy", "fig08_phi_energy_raw_dense_diagnostic.png", "Raw diagnostic delta phi energy"), ("delta_phi_full", "fig09_phi_full_raw_dense_diagnostic.png", "Raw diagnostic delta phi full")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rule, sub in raw_phi_df.groupby("rule"):
            sub = sub.sort_values("radius")
            ax.plot(sub["radius"], sub[field], linewidth=1.0, alpha=0.8, label=rule)
            pass_sub = sub[sub["qc_pass"]].sort_values("radius")
            if not pass_sub.empty:
                ax.scatter(pass_sub["radius"], pass_sub[field], s=12)
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
    write_json(out_dir / "run_config_resolved.json", {**cfg, "supported_radii": [float(r) for r in common_pass], "no_claim_radii": no_claim})
    refs_per_rule = int(unit_df.groupby("rule")["ref_id"].nunique().min()) if not unit_df.empty else 0
    sample_counts = {int(k): int(v) for k, v in unit_df["n_samples"].value_counts().sort_index().items()} if "n_samples" in unit_df.columns else {}
    total_radii = int(qc_df["radius"].nunique())

    def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
        table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            table.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return "\n".join(table)

    complexity_rows: list[list[Any]] = []
    complexity_path = stage_dir("02_complexity_measure") / "complexity_by_rule_summary.csv"
    if complexity_path.exists():
        complexity_df = pd.read_csv(complexity_path)
        for rule in RULES:
            match = complexity_df[complexity_df["rule"] == rule]
            if not match.empty:
                complexity_rows.append([rule, f"{float(match['tv_mean'].iloc[0]):.6f}", f"{float(match['nmstv_mean'].iloc[0]):.6f}"])
    qc_rows: list[list[Any]] = []
    for rule in RULES:
        sub = qc_df[qc_df["rule"] == rule].sort_values("radius")
        passed = sub[sub["qc_pass"]]
        pass_count = int(len(passed))
        if pass_count:
            pass_range = f"{float(passed['radius'].min()):.2f}..{float(passed['radius'].max()):.2f}"
        else:
            pass_range = "none"
        qc_rows.append([rule, pass_count, total_radii, pass_range])

    output_rows = [
        ["raw dense diagnostic", "phi_by_rule_radius_raw_diagnostic.csv", len(raw_phi_df), "no-claim where QC failed"],
        ["QC-pass rule/radius", "phi_by_rule_radius.csv", len(phi_df), "claimable per passing rule/radius"],
        ["common-radius claim table", "final_claim_table.csv", len(claim_df), f"{len(common_pass)} supported, {len(no_claim)} no_claim"],
    ]
    figures_rows = [
        ["label representatives", "../01_dataset_prepare/figures/label_representatives/", f"{len(RULES)} rule figures"],
        ["raw dense energy", "figures/fig08_phi_energy_raw_dense_diagnostic.png", "all rules, 250 radii each"],
        ["raw dense full", "figures/fig09_phi_full_raw_dense_diagnostic.png", "all rules, 250 radii each"],
        ["QC-pass heatmap", "figures/fig07_sampling_qc_pass_heatmap.png", "pass/no-claim by rule and radius"],
        ["QC-pass energy", "figures/fig04_phi_energy_qc_pass_main.png", "only passing rule/radius rows"],
    ]
    comparison_rows = [
        ["model", "2-48-48-1 tanh, P=2545", f"196-12-12-1 tanh, P={P}"],
        ["input/data", "synthetic 2D beta-cell datasets", "one fixed MNIST 1/4 marginal, 12x12 inputs"],
        ["label axis", "many beta-indexed synthetic datasets", "same input marginal, four label rules"],
        ["complexity measure", "dataset/rule complexity used for ordering", "graph TV/NMSTV on the fixed MNIST marginal"],
        ["reference ensemble", "30 references per retained beta/dataset setting", f"{refs_per_rule} exact optimizer-induced references per rule"],
        ["shell sampler", "hard L2 shell PM-SAIS with adaptive CE-tempered SMC", "same hard L2 shell PM-SAIS estimator"],
        ["phi aggregation", "averaged over retained sample/reference units by beta/radius", "averaged over 30 references for each rule/radius"],
        ["QC claim policy", "retained dense production claim where sampler QC passed", "raw dense diagnostic exists, but common-radius claim is empty because teacher_nn fails split-QC"],
    ]
    report = f"""# MNIST14 Single-Dataset 12x12 PM-SAIS Final Report

Architecture: `196-12-12-1-tanh`, P={P}

Scale: one split, {len(RULES)} label rules, {refs_per_rule} exact references per rule, dense d_raw 0.01..2.50.

Main estimator: PM-SAIS averaged over references. `phi_by_rule_radius.csv` contains only QC-passed claim rows. `phi_by_rule_radius_raw_diagnostic.csv` contains dense raw diagnostic rows and must not be used as a claim table where QC failed.

Sampling rows: {len(unit_df)} unit summaries. Sample-count mix: {sample_counts}.

## Complexity

{markdown_table(["rule", "TV", "NMSTV"], complexity_rows)}

## Sampling QC Coverage

{markdown_table(["rule", "QC-pass radii", "total radii", "pass radius range"], qc_rows)}

Common supported radii across all rules: {len(common_pass)} / {total_radii}.

## Phi Outputs

{markdown_table(["scope", "file", "rows", "claim status"], output_rows)}

## Figures

{markdown_table(["figure set", "path", "content"], figures_rows)}

## Pipeline Comparison To Retained 3NN Synthetic

{markdown_table(["item", "retained 3NN synthetic", "MNIST14 run"], comparison_rows)}

## Interpretation

The dense raw phi(d) curves were obtained for all four rules over 250 radii. The formal claim table has no common supported radius because `teacher_nn` has zero QC-passed radii under the current 30-reference, mixed 1024/256-sample recovery run. The raw dense figures are therefore diagnostic curves, while the QC-pass files are the claimable subset.
"""
    write_text(out_dir / "REPORT.md", report)
    checks = {
        "raw_phi_rows": int(len(raw_phi_df)),
        "phi_rows": int(len(phi_df)),
        "supported_radii_count": int(len(common_pass)),
        "main_figure_exists": bool((fig_dir / "fig04_phi_energy_qc_pass_main.png").exists()),
        "raw_dense_figure_exists": bool((fig_dir / "fig08_phi_energy_raw_dense_diagnostic.png").exists()),
        "figure_count": len(list(fig_dir.glob("*.png"))),
    }
    if not checks["main_figure_exists"] or checks["figure_count"] < 3:
        raise StageBlocked("06_results_figures", "Final figure QC failed.", observed=checks)
    write_qc("06_results_figures", "pass", checks, warnings=["Only QC-passed rule/radius rows are plotted in phi_by_rule_radius.csv; raw dense diagnostics are no-claim where QC failed."])


def run_stage(stage: str, *, force: bool = False, max_units: int | None = None, workers: int = 1) -> None:
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
    elif stage == "05_stability_particle_pilot":
        stage05_stability_particle_pilot(force=force)
    elif stage == "05_stability_kernel_scan":
        stage05_stability_kernel_scan(force=force)
    elif stage == "05_pool2_pm_sais_sampling":
        stage05_pool2_pm_sais_sampling(force=force, max_units=max_units, workers=workers)
    elif stage == "06_results_figures":
        stage06_results_figures()
    elif stage == "all":
        for item in STAGES:
            run_stage(item, force=force, max_units=max_units if item == "05_pool2_pm_sais_sampling" else None, workers=workers if item == "05_pool2_pm_sais_sampling" else 1)
    else:
        raise ValueError(f"unknown stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES + ["05_sampling_pilot", "05_stability_pilot", "05_stability_escalated_pilot", "05_stability_particle_pilot", "05_stability_kernel_scan", "all"], required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        run_stage(args.stage, force=args.force, max_units=args.max_units, workers=args.workers)
    except StageBlocked as blocked:
        write_blocked(blocked)
        print(f"BLOCKED {blocked.stage}: {blocked.reason}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
