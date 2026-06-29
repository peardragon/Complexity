#!/usr/bin/env python3
"""CPU-limited eta label-flip pilot for MNIST even/odd complexity checks."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", str(STAGE_ROOT / ".cache" / "matplotlib"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional runtime guard
    threadpool_limits = None


DEFAULT_CONFIG = STAGE_ROOT / "config" / "eta_flip_pilot.json"
SPIN_CURVATURE_REFERENCE = Path(
    "/home/bjyong/Complexity/local_project/02_dnn/06_random_gaussian_baseline/"
    "figures/gaussian_overlay_final_derivative/measure_search/"
    "positive_curvature_mass_composite_spin_only.csv"
)


@dataclass(frozen=True)
class RunPaths:
    raw: Path
    figures: Path
    qc: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("use_gpu", False):
        raise ValueError("This pilot is intentionally CPU-only; set use_gpu=false.")
    return cfg


def make_paths(run_name: str) -> RunPaths:
    paths = RunPaths(
        raw=STAGE_ROOT / "raw_outputs" / run_name,
        figures=STAGE_ROOT / "figures" / run_name,
        qc=STAGE_ROOT / "QC" / run_name,
    )
    for path in (paths.raw, paths.figures, paths.qc):
        path.mkdir(parents=True, exist_ok=True)
    (STAGE_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
    return paths


def limited_threads(thread_limit: int):
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=thread_limit)


def build_knn_edges(x: np.ndarray, k_values: list[int]) -> dict[int, np.ndarray]:
    max_k = max(k_values)
    model = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto", n_jobs=1)
    model.fit(x)
    neighbors = model.kneighbors(x, return_distance=False)[:, 1:]
    n = x.shape[0]
    edges_by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        dst = neighbors[:, :k].reshape(-1)
        src = np.repeat(np.arange(n), k)
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        edges = np.unique(np.column_stack([lo, hi]), axis=0)
        edges_by_k[k] = edges.astype(np.int32, copy=False)
    return edges_by_k


def edge_metrics(y: np.ndarray, edges_by_k: dict[int, np.ndarray]) -> list[dict]:
    y = y.astype(np.int8, copy=False)
    p_pos = float(np.mean(y > 0))
    random_cut = 2.0 * p_pos * (1.0 - p_pos)
    rows = []
    for k, edges in edges_by_k.items():
        yi = y[edges[:, 0]]
        yj = y[edges[:, 1]]
        cut_fraction = float(np.mean(yi != yj))
        signed_alignment = float(np.mean(yi.astype(np.float64) * yj.astype(np.float64)))
        rows.append(
            {
                "k": int(k),
                "edge_count": int(edges.shape[0]),
                "p_pos": p_pos,
                "random_cut_fraction": random_cut,
                "cut_fraction": cut_fraction,
                "knn_nmstv": cut_fraction / random_cut if random_cut > 0 else np.nan,
                "signed_alignment": signed_alignment,
            }
        )
    return rows


def generate_nested_labels(
    y_train: np.ndarray,
    y_test: np.ndarray,
    eta_values: np.ndarray,
    replicates: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(base_seed)
    u_train = rng.random((replicates, y_train.shape[0]), dtype=np.float32)
    u_test = rng.random((replicates, y_test.shape[0]), dtype=np.float32)
    y_train_eta = np.empty((len(eta_values), replicates, y_train.shape[0]), dtype=np.int8)
    y_test_eta = np.empty((len(eta_values), replicates, y_test.shape[0]), dtype=np.int8)
    for eta_idx, eta in enumerate(eta_values):
        train_sign = np.where(u_train < eta, -1, 1).astype(np.int8)
        test_sign = np.where(u_test < eta, -1, 1).astype(np.int8)
        y_train_eta[eta_idx] = y_train[np.newaxis, :] * train_sign
        y_test_eta[eta_idx] = y_test[np.newaxis, :] * test_sign
    return u_train, u_test, y_train_eta, y_test_eta


def summarize_metrics(rep_metrics: pd.DataFrame, eta_values: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["eta", "k"]
    summary = (
        rep_metrics.groupby(group_cols, as_index=False)
        .agg(
            replicate_count=("replicate", "nunique"),
            edge_count=("edge_count", "first"),
            p_pos_mean=("p_pos", "mean"),
            p_pos_sd=("p_pos", "std"),
            cut_fraction_mean=("cut_fraction", "mean"),
            cut_fraction_sd=("cut_fraction", "std"),
            cut_fraction_sem=("cut_fraction", lambda x: float(x.std(ddof=1) / math.sqrt(len(x)))),
            knn_nmstv_mean=("knn_nmstv", "mean"),
            knn_nmstv_sd=("knn_nmstv", "std"),
            knn_nmstv_sem=("knn_nmstv", lambda x: float(x.std(ddof=1) / math.sqrt(len(x)))),
            signed_alignment_mean=("signed_alignment", "mean"),
            signed_alignment_sd=("signed_alignment", "std"),
        )
        .sort_values(["k", "eta"])
        .reset_index(drop=True)
    )

    derivative_rows = []
    for k, g in summary.groupby("k", sort=True):
        g = g.sort_values("eta")
        d_nmstv = np.gradient(g["knn_nmstv_mean"].to_numpy(), g["eta"].to_numpy())
        d_cut = np.gradient(g["cut_fraction_mean"].to_numpy(), g["eta"].to_numpy())
        for eta, dn, dc in zip(g["eta"].to_numpy(), d_nmstv, d_cut):
            derivative_rows.append(
                {
                    "eta": float(eta),
                    "k": int(k),
                    "d_knn_nmstv_d_eta_mean_curve": float(dn),
                    "d_cut_fraction_d_eta_mean_curve": float(dc),
                }
            )

    rep_derivative_rows = []
    for (replicate, k), g in rep_metrics.groupby(["replicate", "k"], sort=True):
        g = g.sort_values("eta")
        if len(g) != len(eta_values):
            continue
        d_nmstv = np.gradient(g["knn_nmstv"].to_numpy(), g["eta"].to_numpy())
        d_cut = np.gradient(g["cut_fraction"].to_numpy(), g["eta"].to_numpy())
        for eta, dn, dc in zip(g["eta"].to_numpy(), d_nmstv, d_cut):
            rep_derivative_rows.append(
                {
                    "replicate": int(replicate),
                    "eta": float(eta),
                    "k": int(k),
                    "d_knn_nmstv_d_eta": float(dn),
                    "d_cut_fraction_d_eta": float(dc),
                }
            )
    rep_derivatives = pd.DataFrame(rep_derivative_rows)
    derivative_summary = (
        rep_derivatives.groupby(["eta", "k"], as_index=False)
        .agg(
            d_knn_nmstv_d_eta_mean=("d_knn_nmstv_d_eta", "mean"),
            d_knn_nmstv_d_eta_sd=("d_knn_nmstv_d_eta", "std"),
            d_knn_nmstv_d_eta_sem=("d_knn_nmstv_d_eta", lambda x: float(x.std(ddof=1) / math.sqrt(len(x)))),
            d_cut_fraction_d_eta_mean=("d_cut_fraction_d_eta", "mean"),
            d_cut_fraction_d_eta_sd=("d_cut_fraction_d_eta", "std"),
            d_cut_fraction_d_eta_sem=("d_cut_fraction_d_eta", lambda x: float(x.std(ddof=1) / math.sqrt(len(x)))),
        )
        .merge(pd.DataFrame(derivative_rows), on=["eta", "k"], how="left")
        .sort_values(["k", "eta"])
        .reset_index(drop=True)
    )
    return summary, derivative_summary


def expected_flip_curve(base_cut_by_k: pd.DataFrame, eta_values: np.ndarray, base_p_pos: float) -> pd.DataFrame:
    rows = []
    for _, row in base_cut_by_k.iterrows():
        q0 = float(row["cut_fraction"])
        for eta in eta_values:
            p_eta = eta + base_p_pos * (1.0 - 2.0 * eta)
            random_cut = 2.0 * p_eta * (1.0 - p_eta)
            q_eta = q0 + 2.0 * eta * (1.0 - eta) * (1.0 - 2.0 * q0)
            rows.append(
                {
                    "eta": float(eta),
                    "k": int(row["k"]),
                    "expected_cut_fraction_nested_independent": q_eta,
                    "expected_knn_nmstv": q_eta / random_cut if random_cut > 0 else np.nan,
                    "expected_d_cut_d_eta": 2.0 * (1.0 - 2.0 * eta) * (1.0 - 2.0 * q0),
                    "expected_d2_cut_d_eta2": -4.0 * (1.0 - 2.0 * q0),
                }
            )
    return pd.DataFrame(rows)


def transition_candidates(summary: pd.DataFrame, derivative_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = [0.75, 0.90, 0.95]
    for k, g in summary.groupby("k", sort=True):
        g = g.sort_values("eta")
        d = derivative_summary[derivative_summary["k"] == k].sort_values("eta")
        nmstv = g["knn_nmstv_mean"].to_numpy()
        etas = g["eta"].to_numpy()
        row = {
            "k": int(k),
            "eta_min": float(etas.min()),
            "eta_max": float(etas.max()),
            "knn_nmstv_eta0": float(nmstv[0]),
            "knn_nmstv_eta0p5": float(nmstv[-1]),
            "eta_peak_d_knn_nmstv_d_eta": float(d.iloc[d["d_knn_nmstv_d_eta_mean"].argmax()]["eta"]),
            "peak_d_knn_nmstv_d_eta": float(d["d_knn_nmstv_d_eta_mean"].max()),
            "positive_slope_mass": float(np.trapz(np.maximum(d["d_knn_nmstv_d_eta_mean"].to_numpy(), 0.0), d["eta"].to_numpy())),
        }
        for threshold in thresholds:
            passed = g[g["knn_nmstv_mean"] >= threshold]
            row[f"eta_cross_nmstv_{str(threshold).replace('.', 'p')}"] = (
                float(passed.iloc[0]["eta"]) if not passed.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_label_store(
    paths: RunPaths,
    cfg: dict,
    data: np.lib.npyio.NpzFile,
    eta_values: np.ndarray,
    u_train: np.ndarray,
    u_test: np.ndarray,
    y_train_eta: np.ndarray,
    y_test_eta: np.ndarray,
) -> Path:
    path = paths.raw / "eta_nested_label_store.npz"
    np.savez_compressed(
        path,
        base_dataset_path=str(cfg["base_dataset_path"]),
        eta_values=eta_values.astype(np.float32),
        flip_uniform_train=u_train,
        flip_uniform_test=u_test,
        y_train_eta=y_train_eta,
        y_test_eta=y_test_eta,
        y_train_base=data["y_train"],
        y_test_base=data["y_test"],
        digit_train=data["digit_train"],
        digit_test=data["digit_test"],
        train_indices=data["train_indices"],
        test_indices=data["test_indices"],
    )
    return path


def plot_nmstv(summary: pd.DataFrame, expected: pd.DataFrame, fig_path: Path) -> None:
    palette = {3: "#2451a6", 5: "#00857a", 8: "#d27a00", 16: "#7a5195", 32: "#b33b34"}
    fig, ax = plt.subplots(figsize=(8.2, 5.1), dpi=180)
    for k, g in summary.groupby("k", sort=True):
        g = g.sort_values("eta")
        color = palette.get(int(k), None)
        ax.plot(g["eta"], g["knn_nmstv_mean"], marker="o", ms=3.4, lw=1.8, color=color, label=f"k={int(k)}")
        ax.fill_between(
            g["eta"].to_numpy(),
            (g["knn_nmstv_mean"] - 1.96 * g["knn_nmstv_sem"]).to_numpy(),
            (g["knn_nmstv_mean"] + 1.96 * g["knn_nmstv_sem"]).to_numpy(),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    for k, g in expected.groupby("k", sort=True):
        g = g.sort_values("eta")
        ax.plot(g["eta"], g["expected_knn_nmstv"], ls="--", lw=0.9, alpha=0.5, color=palette.get(int(k), "0.4"))
    ax.axhline(1.0, color="0.2", lw=1.0, ls=":", label="random-label level")
    ax.set_xlabel("label flip probability eta")
    ax.set_ylabel("kNN normalized MSTV")
    ax.set_title("MNIST even/odd eta sweep: graph-complexity proxy")
    ax.set_xlim(-0.005, 0.505)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def plot_derivative(derivative_summary: pd.DataFrame, expected: pd.DataFrame, fig_path: Path) -> None:
    palette = {3: "#2451a6", 5: "#00857a", 8: "#d27a00", 16: "#7a5195", 32: "#b33b34"}
    fig, ax = plt.subplots(figsize=(8.2, 5.1), dpi=180)
    for k, g in derivative_summary.groupby("k", sort=True):
        g = g.sort_values("eta")
        color = palette.get(int(k), None)
        ax.plot(g["eta"], g["d_knn_nmstv_d_eta_mean"], marker="o", ms=3.0, lw=1.8, color=color, label=f"k={int(k)}")
        ax.fill_between(
            g["eta"].to_numpy(),
            (g["d_knn_nmstv_d_eta_mean"] - 1.96 * g["d_knn_nmstv_d_eta_sem"]).to_numpy(),
            (g["d_knn_nmstv_d_eta_mean"] + 1.96 * g["d_knn_nmstv_d_eta_sem"]).to_numpy(),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(0.0, color="0.2", lw=1.0)
    ax.set_xlabel("label flip probability eta")
    ax.set_ylabel("d(kNN NMSTV) / d eta")
    ax.set_title("First-derivative proxy from nested eta masks")
    ax.set_xlim(-0.005, 0.505)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def plot_transition(candidates: pd.DataFrame, fig_path: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.3, 4.2), dpi=180)
    x = np.arange(len(candidates))
    labels = [f"k={int(k)}" for k in candidates["k"]]
    ax0.bar(x - 0.18, candidates["eta_cross_nmstv_0p9"], width=0.36, color="#00857a", label="NMSTV >= 0.90")
    ax0.bar(x + 0.18, candidates["eta_peak_d_knn_nmstv_d_eta"], width=0.36, color="#d27a00", label="peak derivative")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, 0.52)
    ax0.set_ylabel("eta")
    ax0.set_title("eta landmarks")
    ax0.grid(True, axis="y", color="0.88", linewidth=0.7)
    ax0.legend(frameon=False, fontsize=8)

    ax1.plot(candidates["k"], candidates["positive_slope_mass"], marker="o", color="#2451a6", lw=1.9)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(candidates["k"])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("kNN graph k")
    ax1.set_ylabel("integral positive d(NMSTV)/d eta")
    ax1.set_title("first-derivative mass")
    ax1.grid(True, color="0.88", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def plot_examples(
    paths: RunPaths,
    data: np.lib.npyio.NpzFile,
    eta_values: np.ndarray,
    y_train_eta: np.ndarray,
    u_train: np.ndarray,
) -> None:
    target_etas = [0.0, 0.1, 0.2, 0.35, 0.5]
    eta_indices = [int(np.abs(eta_values - eta).argmin()) for eta in target_etas]
    base_y = data["y_train"].astype(np.int8)
    raw = data["X_train_raw10"].reshape(-1, 10, 10)
    interesting = np.argsort(np.abs(u_train[0] - 0.2))[:10]
    fig, axes = plt.subplots(len(eta_indices), len(interesting), figsize=(10.0, 5.4), dpi=180)
    for row, eta_idx in enumerate(eta_indices):
        eta = eta_values[eta_idx]
        y_eta = y_train_eta[eta_idx, 0]
        flip = y_eta != base_y
        for col, idx in enumerate(interesting):
            ax = axes[row, col]
            ax.imshow(raw[idx], cmap="gray_r", interpolation="nearest")
            color = "#b33b34" if flip[idx] else "#2451a6"
            sign = "+" if y_eta[idx] > 0 else "-"
            digit = int(data["digit_train"][idx])
            ax.set_title(f"{digit}:{sign}", fontsize=7, color=color, pad=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color if flip[idx] else "0.75")
                spine.set_linewidth(1.2 if flip[idx] else 0.5)
        axes[row, 0].set_ylabel(f"eta={eta:.2f}", fontsize=8)
    fig.suptitle("Nested label flips on fixed MNIST examples", y=0.995, fontsize=11)
    fig.tight_layout(pad=0.35)
    fig.savefig(paths.figures / "fig04_eta_nested_label_examples.png")
    plt.close(fig)


def maybe_plot_spin_reference(paths: RunPaths) -> None:
    if not SPIN_CURVATURE_REFERENCE.exists():
        return
    ref = pd.read_csv(SPIN_CURVATURE_REFERENCE)
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=180)
    ax.plot(ref["beta"], ref["A_kappa"], marker="o", ms=3.0, lw=1.7, color="#2451a6")
    ax.set_xlabel("spin beta")
    ax.set_ylabel("positive curvature mass A_kappa")
    ax.set_title("Reference: spin-only random Gaussian curvature mass")
    ax.grid(True, color="0.88", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig05_reference_spin_positive_curvature_mass.png")
    plt.close(fig)
    ref.to_csv(paths.raw / "reference_spin_positive_curvature_mass.csv", index=False)


def write_report(
    paths: RunPaths,
    cfg: dict,
    elapsed: float,
    label_store: Path,
    candidates: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    k3 = candidates[candidates["k"] == 3].iloc[0].to_dict()
    expected_k3 = expected[expected["k"] == 3].sort_values("eta").iloc[0].to_dict()
    report = f"""# Eta Flip Pilot Report

## Run

- run_name: `{cfg['run_name']}`
- base dataset: `{cfg['base_dataset_path']}`
- eta grid: `{cfg['eta_values']}`
- replicates: `{cfg['replicates']}` with nested flip uniforms
- kNN graph k values: `{cfg['k_values']}`
- resource policy: CPU cap target {cfg['cpu_load_cap_fraction']:.0%}, GPU cap target {cfg['gpu_load_cap_fraction']:.0%}; this run used CPU only with thread_limit={cfg['thread_limit']}
- elapsed seconds: `{elapsed:.2f}`
- compact label store: `{label_store}`

## Main Read

This pilot treats eta as a label-noise knob on the fixed MNIST even/odd inputs.
For the graph-TV proxy, independent label flips have a closed-form expectation:

`q_eta = q0 + 2 * eta * (1 - eta) * (1 - 2*q0)`

where `q0` is the base edge-disagreement rate. Therefore the first derivative
is largest near eta=0 and decays toward eta=0.5 when q0<0.5. In this proxy
alone, an interior positive-curvature transition is not expected; if a
random-like transition appears in phi(d), it should come from the DNN
sampling/reference geometry rather than from graph-TV algebra alone.

## k=3 Landmarks

- NMSTV at eta=0: `{k3['knn_nmstv_eta0']:.4f}`
- NMSTV at eta=0.5: `{k3['knn_nmstv_eta0p5']:.4f}`
- eta crossing NMSTV>=0.90: `{k3['eta_cross_nmstv_0p9']:.3f}`
- peak first derivative eta: `{k3['eta_peak_d_knn_nmstv_d_eta']:.3f}`
- expected d2 cut / d eta2 at eta=0, k=3: `{expected_k3['expected_d2_cut_d_eta2']:.4f}`

## Output Tables

- `summary_by_eta_k.csv`
- `replicate_metrics.csv`
- `derivative_by_eta_k.csv`
- `eta_transition_candidates.csv`
- `expected_independent_flip_curve.csv`

## Output Figures

- `fig01_eta_knn_nmstv_by_k.png`
- `fig02_eta_first_derivative_proxy.png`
- `fig03_eta_transition_landmarks.png`
- `fig04_eta_nested_label_examples.png`
- `fig05_reference_spin_positive_curvature_mass.png` when the reference file is available
"""
    (paths.raw / "RUN_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    paths = make_paths(cfg["run_name"])

    start = time.time()
    eta_values = np.asarray(cfg["eta_values"], dtype=np.float64)
    k_values = [int(k) for k in cfg["k_values"]]
    thread_limit = int(cfg["thread_limit"])

    effective_cfg = dict(cfg)
    effective_cfg["effective_env"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", ""),
    }
    (paths.raw / "effective_config.json").write_text(json.dumps(effective_cfg, indent=2), encoding="utf-8")

    with limited_threads(thread_limit):
        data = np.load(cfg["base_dataset_path"])
        x_train = data["X_train"].astype(np.float32, copy=False)
        y_train = data["y_train"].astype(np.int8, copy=False)
        y_test = data["y_test"].astype(np.int8, copy=False)
        edges_by_k = build_knn_edges(x_train, k_values)
        base_rows = edge_metrics(y_train, edges_by_k)
        base_metrics = pd.DataFrame(base_rows).sort_values("k")

        u_train, u_test, y_train_eta, y_test_eta = generate_nested_labels(
            y_train=y_train,
            y_test=y_test,
            eta_values=eta_values,
            replicates=int(cfg["replicates"]),
            base_seed=int(cfg["base_seed"]),
        )

        label_store = save_label_store(paths, cfg, data, eta_values, u_train, u_test, y_train_eta, y_test_eta)

        rows = []
        qc_rows = []
        for eta_idx, eta in enumerate(eta_values):
            for replicate in range(int(cfg["replicates"])):
                y_eta = y_train_eta[eta_idx, replicate]
                metric_rows = edge_metrics(y_eta, edges_by_k)
                flip_rate_train = float(np.mean(y_eta != y_train))
                flip_rate_test = float(np.mean(y_test_eta[eta_idx, replicate] != y_test))
                qc_rows.append(
                    {
                        "eta": float(eta),
                        "replicate": int(replicate),
                        "flip_rate_train": flip_rate_train,
                        "flip_rate_test": flip_rate_test,
                        "abs_flip_error_train": abs(flip_rate_train - float(eta)),
                        "abs_flip_error_test": abs(flip_rate_test - float(eta)),
                    }
                )
                for row in metric_rows:
                    row.update({"eta": float(eta), "replicate": int(replicate)})
                    rows.append(row)

    rep_metrics = pd.DataFrame(rows).sort_values(["k", "replicate", "eta"]).reset_index(drop=True)
    qc = pd.DataFrame(qc_rows).sort_values(["eta", "replicate"]).reset_index(drop=True)
    summary, derivative_summary = summarize_metrics(rep_metrics, eta_values)
    expected = expected_flip_curve(base_metrics, eta_values, base_p_pos=float(np.mean(y_train > 0)))
    candidates = transition_candidates(summary, derivative_summary)

    base_metrics.to_csv(paths.raw / "base_knn_metrics.csv", index=False)
    rep_metrics.to_csv(paths.raw / "replicate_metrics.csv", index=False)
    summary.to_csv(paths.raw / "summary_by_eta_k.csv", index=False)
    derivative_summary.to_csv(paths.raw / "derivative_by_eta_k.csv", index=False)
    candidates.to_csv(paths.raw / "eta_transition_candidates.csv", index=False)
    expected.to_csv(paths.raw / "expected_independent_flip_curve.csv", index=False)
    qc.to_csv(paths.qc / "flip_rate_qc_by_eta_replicate.csv", index=False)
    qc.groupby("eta", as_index=False).agg(
        flip_rate_train_mean=("flip_rate_train", "mean"),
        flip_rate_train_sd=("flip_rate_train", "std"),
        flip_rate_test_mean=("flip_rate_test", "mean"),
        flip_rate_test_sd=("flip_rate_test", "std"),
        max_abs_flip_error_train=("abs_flip_error_train", "max"),
        max_abs_flip_error_test=("abs_flip_error_test", "max"),
    ).to_csv(paths.qc / "flip_rate_qc_summary.csv", index=False)

    plot_nmstv(summary, expected, paths.figures / "fig01_eta_knn_nmstv_by_k.png")
    plot_derivative(derivative_summary, expected, paths.figures / "fig02_eta_first_derivative_proxy.png")
    plot_transition(candidates, paths.figures / "fig03_eta_transition_landmarks.png")
    plot_examples(paths, data, eta_values, y_train_eta, u_train)
    maybe_plot_spin_reference(paths)

    elapsed = time.time() - start
    write_report(paths, cfg, elapsed, label_store, candidates, expected)
    print(json.dumps({"run_name": cfg["run_name"], "elapsed_seconds": round(elapsed, 3)}, indent=2))


if __name__ == "__main__":
    main()
