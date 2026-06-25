#!/usr/bin/env python3
"""Generate a lower-NMSTV synthetic MNIST10 label rule on the current split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
WINDOWS_ROOT = Path("/home/bjyong/Complexity/windows_project")
SOURCE_RUN_ROOT = (
    WINDOWS_ROOT
    / "02_dnn/08_mnist/runs/final/"
    "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
)
SOURCE_DATASET = SOURCE_RUN_ROOT / "01_dataset_prepare/raw_datasets/split_000/real_even_odd/dataset.npz"
SOURCE_COMPLEXITY = SOURCE_RUN_ROOT / "02_complexity_measure/complexity_by_rule_summary.csv"
DEFAULT_OUT = LOCAL_ROOT / "01_dataset_gen/raw_outputs/very_low_tv_spectral_teacher_v1"
RULE = "very_low_tv_spectral_teacher"
EXPERIMENT_ID = "mnist10_very_low_tv_spectral_teacher_v1"
K_VALUES = [8, 16, 32]


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


def normalize_labels(y: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(y) > 0, 1, -1).astype(np.int8)


def knn_weight_graph(x: np.ndarray, k: int) -> tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray, float]:
    nn = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean")
    nn.fit(np.asarray(x, dtype=np.float64))
    dist, idx = nn.kneighbors(np.asarray(x, dtype=np.float64), return_distance=True)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    sigma = float(np.median(dist[dist > 0.0]))
    edge_weight: dict[tuple[int, int], float] = {}
    for i in range(x.shape[0]):
        for d, j_raw in zip(dist[i], idx[i]):
            j = int(j_raw)
            a, b = (i, j) if i < j else (j, i)
            weight = float(np.exp(-(float(d) ** 2) / (2.0 * sigma * sigma)))
            if weight > edge_weight.get((a, b), -1.0):
                edge_weight[(a, b)] = weight
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
    mat = coo_matrix((vals, (rows, cols)), shape=(x.shape[0], x.shape[0]), dtype=np.float64)
    return mat, np.asarray(edge_i), np.asarray(edge_j), np.asarray(edge_w), sigma


def edge_tv_baseline_nmstv(y: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, edge_w: np.ndarray) -> tuple[float, float, float]:
    labels = normalize_labels(y)
    total_w = float(np.sum(edge_w))
    cut_w = float(np.sum(edge_w[labels[edge_i] != labels[edge_j]]))
    tv = cut_w / max(total_w, 1.0e-300)
    p_pos = float(np.mean(labels == 1))
    baseline = 2.0 * p_pos * (1.0 - p_pos)
    return tv, baseline, float(tv / max(baseline, 1.0e-12))


def max_digit_label_purity(y: np.ndarray, digits: np.ndarray) -> float:
    labels = normalize_labels(y)
    purities = []
    for digit in sorted(np.unique(digits)):
        mask = digits == digit
        pos = float(np.mean(labels[mask] == 1))
        purities.append(max(pos, 1.0 - pos))
    return float(max(purities))


def spectral_basis(w_mat: coo_matrix, spectral_k: int) -> tuple[np.ndarray, np.ndarray]:
    degree = np.asarray(w_mat.sum(axis=1)).ravel()
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 1.0e-300
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
    lap = eye(w_mat.shape[0], format="csr", dtype=np.float64) - diags(inv_sqrt_degree) @ w_mat.tocsr() @ diags(inv_sqrt_degree)
    eigvals, eigvecs = eigsh(lap, k=int(spectral_k) + 1, which="SM", tol=1.0e-6)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order[1 : int(spectral_k) + 1]]


def complexity_rows(
    *,
    y: np.ndarray,
    graphs: dict[int, tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray, float]],
    dataset_path: Path,
    n_train: int,
    n_test: int,
    input_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        _mat, edge_i, edge_j, edge_w, sigma = graphs[k]
        tv, baseline, nmstv = edge_tv_baseline_nmstv(y, edge_i, edge_j, edge_w)
        graph_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "mode": "synthetic_single_dataset",
                "split_id": 0,
                "rule": RULE,
                "dataset_path": str(dataset_path),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "input_dim": int(input_dim),
                "train_pos_fraction": float(np.mean(normalize_labels(y) == 1)),
                "k": int(k),
                "edge_count": int(len(edge_w)),
                "sigma_k": float(sigma),
                "tv": float(tv),
                "baseline": float(baseline),
                "nmstv": float(nmstv),
            }
        )
    graph_df = pd.DataFrame(graph_rows)
    summary = pd.DataFrame(
        [
            {
                "rule": RULE,
                "nmstv_mean": float(graph_df["nmstv"].mean()),
                "tv_mean": float(graph_df["tv"].mean()),
                "n_datasets": 1,
            }
        ]
    )
    return graph_df, summary


def choose_candidate(
    *,
    x_train: np.ndarray,
    y_even: np.ndarray,
    digit_train: np.ndarray,
    graphs: dict[int, tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray, float]],
    max_draws_per_basis: int,
    spectral_ks: list[int],
    rng_seed: int,
) -> dict[str, Any]:
    real_k16 = edge_tv_baseline_nmstv(y_even, graphs[16][1], graphs[16][2], graphs[16][3])[2]
    best: dict[str, Any] | None = None
    best_rejected: dict[str, Any] | None = None
    for spectral_k in spectral_ks:
        eigvals, basis = spectral_basis(graphs[16][0], spectral_k)
        rng = np.random.default_rng(int(rng_seed) + 1009 * int(spectral_k))
        for draw_idx in range(int(max_draws_per_basis)):
            coeff = rng.normal(size=basis.shape[1])
            score = np.asarray(basis @ coeff, dtype=np.float64)
            threshold = float(np.median(score))
            y = np.where(score >= threshold, 1, -1).astype(np.int8)
            pos = float(np.mean(y == 1))
            k16_tv, k16_baseline, k16_nmstv = edge_tv_baseline_nmstv(y, graphs[16][1], graphs[16][2], graphs[16][3])
            corr = float(np.corrcoef(y.astype(np.float64), normalize_labels(y_even).astype(np.float64))[0, 1])
            purity = max_digit_label_purity(y, digit_train)
            row = {
                "spectral_k": int(spectral_k),
                "draw_idx": int(draw_idx),
                "rng_seed": int(rng_seed) + 1009 * int(spectral_k),
                "coefficients": coeff.tolist(),
                "threshold": threshold,
                "pos_fraction": pos,
                "k16_tv": float(k16_tv),
                "k16_baseline": float(k16_baseline),
                "k16_nmstv": float(k16_nmstv),
                "corr_even_odd": corr,
                "max_digit_label_purity": purity,
                "laplacian_eigenvalues": eigvals.tolist(),
            }
            score_key = float(k16_nmstv + 0.2 * abs(corr) + 0.2 * purity)
            if best_rejected is None or score_key < float(best_rejected["score_key"]):
                best_rejected = {**row, "score_key": score_key}
            if not (0.48 <= pos <= 0.52 and k16_nmstv < 0.8 * real_k16 and abs(corr) < 0.25 and purity < 0.80):
                continue
            nmstvs = []
            tvs = []
            for k in K_VALUES:
                tv, _baseline, nmstv = edge_tv_baseline_nmstv(y, graphs[k][1], graphs[k][2], graphs[k][3])
                tvs.append(float(tv))
                nmstvs.append(float(nmstv))
            row["tv_mean"] = float(np.mean(tvs))
            row["nmstv_mean"] = float(np.mean(nmstvs))
            row["nmstv_by_k"] = {str(k): float(v) for k, v in zip(K_VALUES, nmstvs)}
            row["tv_by_k"] = {str(k): float(v) for k, v in zip(K_VALUES, tvs)}
            if best is None or float(row["nmstv_mean"]) < float(best["nmstv_mean"]):
                best = row
    if best is None:
        raise RuntimeError(f"No feasible candidate found. Best rejected candidate: {best_rejected}")
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate lower-TV spectral synthetic MNIST10 labels.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--max-draws-per-basis", type=int, default=25000)
    parser.add_argument("--spectral-ks", default="3,4,6,8,12")
    parser.add_argument("--rng-seed", type=int, default=20260618)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    dataset_dir = out_root / "01_dataset_prepare/raw_datasets/split_000" / RULE
    dataset_path = dataset_dir / "dataset.npz"
    if dataset_path.exists() and not args.force:
        print(f"Reusing existing dataset: {dataset_path}")
        return 0
    if out_root.exists() and args.force:
        # Avoid deleting the whole run root; only overwrite deterministic files below.
        pass

    payload = np.load(SOURCE_DATASET)
    x_train = payload["X_train"].astype(np.float64)
    x_test = payload["X_test"].astype(np.float64)
    y_even = normalize_labels(payload["y_train"])
    digit_train = payload["digit_train"]

    graphs = {k: knn_weight_graph(x_train, k) for k in K_VALUES}
    spectral_ks = [int(x.strip()) for x in str(args.spectral_ks).split(",") if x.strip()]
    candidate = choose_candidate(
        x_train=x_train,
        y_even=y_even,
        digit_train=digit_train,
        graphs=graphs,
        max_draws_per_basis=int(args.max_draws_per_basis),
        spectral_ks=spectral_ks,
        rng_seed=int(args.rng_seed),
    )

    _eigvals, basis = spectral_basis(graphs[16][0], int(candidate["spectral_k"]))
    coeff = np.asarray(candidate["coefficients"], dtype=np.float64)
    train_score = np.asarray(basis @ coeff, dtype=np.float64)
    threshold = float(candidate["threshold"])
    y_train = np.where(train_score >= threshold, 1, -1).astype(np.int8)

    sigma = graphs[16][4]
    nn = NearestNeighbors(n_neighbors=16, metric="euclidean")
    nn.fit(x_train)
    test_dist, test_idx = nn.kneighbors(x_test, return_distance=True)
    test_weight = np.exp(-(test_dist**2) / (2.0 * sigma * sigma))
    test_score = np.sum(test_weight * train_score[test_idx], axis=1) / np.maximum(np.sum(test_weight, axis=1), 1.0e-300)
    y_test = np.where(test_score >= threshold, 1, -1).astype(np.int8)

    ensure_dir(dataset_dir)
    np.savez_compressed(
        dataset_path,
        X_train=payload["X_train"],
        y_train=y_train,
        X_test=payload["X_test"],
        y_test=y_test,
        X_train_raw10=payload["X_train_raw10"],
        X_test_raw10=payload["X_test_raw10"],
        X_train_raw=payload["X_train_raw"],
        X_test_raw=payload["X_test_raw"],
        digit_train=payload["digit_train"],
        digit_test=payload["digit_test"],
        train_indices=payload["train_indices"],
        test_indices=payload["test_indices"],
        standardization_mean=payload["standardization_mean"],
        standardization_std=payload["standardization_std"],
        spectral_train_score=train_score.astype(np.float64),
        spectral_test_score=test_score.astype(np.float64),
    )

    graph_df, summary_df = complexity_rows(
        y=y_train,
        graphs=graphs,
        dataset_path=dataset_path,
        n_train=int(payload["X_train"].shape[0]),
        n_test=int(payload["X_test"].shape[0]),
        input_dim=int(payload["X_train"].shape[1]),
    )
    existing_complexity = pd.read_csv(SOURCE_COMPLEXITY)
    combined = pd.concat([existing_complexity, summary_df], ignore_index=True, sort=False)
    combined["complexity_rank_nmstv_low_to_high"] = combined["nmstv_mean"].rank(method="first").astype(int)

    dataset_index = pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT_ID,
                "mode": "synthetic_single_dataset",
                "split_id": 0,
                "rule": RULE,
                "dataset_path": str(dataset_path),
                "n_train": int(payload["X_train"].shape[0]),
                "n_test": int(payload["X_test"].shape[0]),
                "input_dim": int(payload["X_train"].shape[1]),
                "train_pos_fraction": float(np.mean(y_train == 1)),
            }
        ]
    )
    write_csv(out_root / "01_dataset_prepare/dataset_index.csv", dataset_index)
    write_csv(out_root / "02_complexity_measure/graph_stats_by_dataset_k.csv", graph_df)
    write_csv(out_root / "02_complexity_measure/complexity_by_dataset.csv", graph_df.groupby(["experiment_id", "mode", "split_id", "rule", "dataset_path", "n_train", "n_test", "input_dim", "train_pos_fraction"], as_index=False).agg(tv_mean=("tv", "mean"), nmstv_mean=("nmstv", "mean"), edge_count_min=("edge_count", "min")))
    write_csv(out_root / "02_complexity_measure/complexity_by_rule_summary.csv", summary_df)
    write_csv(out_root / "02_complexity_measure/complexity_with_existing_rules.csv", combined.sort_values("complexity_rank_nmstv_low_to_high"))

    metadata = {
        "rule": RULE,
        "split_id": 0,
        "definition": "very-low-frequency spectral graph teacher on the current MNIST10 train kNN graph with kNN interpolation for test labels",
        "source_dataset": str(SOURCE_DATASET),
        "selection_constraints": {
            "train_pos_fraction": "[0.48, 0.52]",
            "k16_nmstv": "< 0.8 * real_even_odd_k16_nmstv",
            "abs_corr_even_odd": "< 0.25",
            "max_digit_label_purity": "< 0.80",
        },
        "candidate": candidate,
        "train_pos_fraction": float(np.mean(y_train == 1)),
        "test_pos_fraction": float(np.mean(y_test == 1)),
        "complexity_mean": summary_df.iloc[0].to_dict(),
        "complexity_rank_with_existing": combined.sort_values("complexity_rank_nmstv_low_to_high").to_dict("records"),
    }
    write_json(dataset_dir / "dataset_metadata.json", metadata)
    write_json(out_root / "01_dataset_prepare/QC_STATUS.json", {"status": "pass", "dataset_path": str(dataset_path), "rule": RULE})
    write_json(out_root / "02_complexity_measure/QC_STATUS.json", {"status": "pass", "rule": RULE, "nmstv_mean": float(summary_df["nmstv_mean"].iloc[0])})
    write_json(out_root / "run_config_resolved.json", {"experiment_id": EXPERIMENT_ID, "rule": RULE, "source_dataset": str(SOURCE_DATASET), "out_root": str(out_root)})
    (out_root / "REPORT.md").write_text(
        "\n".join(
            [
                "# Very Low-TV Spectral Teacher Dataset",
                "",
                f"- Rule: `{RULE}`",
                f"- Dataset: `{dataset_path}`",
                f"- NMSTV mean: `{float(summary_df['nmstv_mean'].iloc[0]):.6f}`",
                f"- TV mean: `{float(summary_df['tv_mean'].iloc[0]):.6f}`",
                f"- Existing+new rank table: `02_complexity_measure/complexity_with_existing_rules.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset_path": str(dataset_path), "nmstv_mean": float(summary_df["nmstv_mean"].iloc[0]), "tv_mean": float(summary_df["tv_mean"].iloc[0])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
