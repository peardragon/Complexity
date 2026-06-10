from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLING_SRC = REPO_ROOT / "02_dnn" / "04_sampling" / "src"
if str(SAMPLING_SRC) not in sys.path:
    sys.path.insert(0, str(SAMPLING_SRC))

from dnn_model import P, ce_error_batch  # noqa: E402


SUMMARY_COLUMNS = [
    "beta",
    "dataset_id",
    "K",
    "S_ref",
    "Q_ref",
    "H_err",
    "H_CE",
    "B_CE_mean",
    "B_CE_median",
    "B_CE_q90",
    "B_err_mean",
    "B_err_median",
    "B_err_q90",
]

PAIRWISE_COLUMNS = [
    "beta",
    "dataset_id",
    "ref_i",
    "ref_j",
    "D_ij",
    "q_ij",
    "B_CE_ij",
    "B_err_ij",
]


@dataclass(frozen=True)
class DatasetPayload:
    beta: float
    cell_id: str
    dataset_tag: str
    selected_refs: list[dict[str, Any]]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_beta(cell_id: str) -> float:
    match = re.search(r"cell_beta_([0-9]+)p([0-9]+)_", cell_id)
    if not match:
        raise ValueError(f"cannot parse beta from cell_id={cell_id!r}")
    return float(f"{match.group(1)}.{match.group(2)}")


def parse_dataset_number(dataset_tag: str) -> int:
    match = re.search(r"dataset_([0-9]+)_", dataset_tag)
    if not match:
        raise ValueError(f"cannot parse dataset number from dataset_tag={dataset_tag!r}")
    return int(match.group(1))


def beta_key(beta: float) -> str:
    return f"{float(beta):.12g}"


def load_reference_payloads(config: dict[str, Any], max_datasets_per_beta: int | None) -> list[DatasetPayload]:
    manifest = load_json(repo_path(config["reference_pool_manifest"]))
    selected_betas = {round(float(x), 12) for x in config["selected_betas"]}
    per_beta_count: dict[float, int] = defaultdict(int)
    payloads: list[DatasetPayload] = []

    for item in manifest["dataset_pools"]:
        beta = round(parse_beta(str(item["cell_id"])), 12)
        if beta not in selected_betas:
            continue
        if max_datasets_per_beta is not None and per_beta_count[beta] >= max_datasets_per_beta:
            continue
        refs = list(item["selected_refs"])
        expected_k = int(config["references_per_dataset"])
        if len(refs) != expected_k:
            raise ValueError(
                f"{item['cell_id']} {item['dataset_tag']} has {len(refs)} refs; expected {expected_k}"
            )
        payloads.append(
            DatasetPayload(
                beta=float(beta),
                cell_id=str(item["cell_id"]),
                dataset_tag=str(item["dataset_tag"]),
                selected_refs=refs,
            )
        )
        per_beta_count[beta] += 1

    payloads.sort(key=lambda x: (x.beta, parse_dataset_number(x.dataset_tag)))
    return payloads


def load_theta_matrix(selected_refs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    theta_rows: list[np.ndarray] = []
    norm_rows: list[float] = []
    ref_ids: list[int] = []
    for order, ref in enumerate(selected_refs):
        theta_path = repo_path(ref["theta_path"])
        if not theta_path.exists():
            raise FileNotFoundError(theta_path)
        theta = np.load(theta_path).astype(np.float64, copy=False).reshape(-1)
        if theta.size != P:
            raise ValueError(f"{theta_path} has P={theta.size}; expected {P}")
        theta_rows.append(theta)
        norm_rows.append(float(np.linalg.norm(theta)))
        ref_ids.append(int(ref.get("ref_id", order)))
    return np.vstack(theta_rows), np.asarray(norm_rows, dtype=np.float64), ref_ids


def load_dataset(config: dict[str, Any], payload: DatasetPayload) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = repo_path(config["dataset_root"]) / payload.cell_id / payload.dataset_tag / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    with np.load(dataset_path) as data:
        x = np.asarray(data["X_train"], dtype=np.float64)
        y = np.asarray(data["y"])
    return x, y


def pairwise_geometry(theta: np.ndarray) -> dict[str, Any]:
    k = theta.shape[0]
    norms = np.linalg.norm(theta, axis=1)
    gram = theta @ theta.T
    dist_sq = np.maximum(norms[:, None] ** 2 + norms[None, :] ** 2 - 2.0 * gram, 0.0)
    d_mat = np.sqrt(dist_sq / float(P))
    denom = norms[:, None] * norms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        q_mat = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0.0)
    pair_i, pair_j = np.triu_indices(k, k=1)
    d_vals = d_mat[pair_i, pair_j]
    q_vals = q_mat[pair_i, pair_j]
    return {
        "D": d_mat,
        "q": q_mat,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "D_vals": d_vals,
        "q_vals": q_vals,
        "S_ref": float(math.sqrt(float(np.mean(d_vals**2)))),
        "Q_ref": float(np.mean(q_vals)),
    }


def eval_ce_err(theta_batch: np.ndarray, x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return ce_error_batch(
        theta_batch,
        x,
        y,
        chunk_size=int(config["eval_chunk_size"]),
        device=str(config.get("device", "cpu")),
        dtype=str(config.get("dtype", "float64")),
    )


def linear_barriers(
    theta: np.ndarray,
    ref_ce: np.ndarray,
    ref_err: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_grid = np.linspace(0.0, 1.0, int(config["t_grid_count"]), dtype=np.float64)
    interior_t = t_grid[1:-1]
    endpoint_ce_max = np.maximum(ref_ce[pair_i], ref_ce[pair_j])
    endpoint_err_max = np.maximum(ref_err[pair_i], ref_err[pair_j])
    path_ce_max = endpoint_ce_max.copy()
    path_err_max = endpoint_err_max.copy()

    if interior_t.size:
        left = theta[pair_i]
        right = theta[pair_j]
        paths = (
            (1.0 - interior_t[:, None, None]) * left[None, :, :]
            + interior_t[:, None, None] * right[None, :, :]
        )
        # Shape is (t, pair, P); transpose so each pair has its 19 interior points
        # adjacent in the result and then flatten to a standard theta batch.
        batch = np.transpose(paths, (1, 0, 2)).reshape(-1, theta.shape[1])
        ce, err = eval_ce_err(batch, x, y, config)
        ce_path = ce.reshape(pair_i.size, interior_t.size)
        err_path = err.reshape(pair_i.size, interior_t.size)
        path_ce_max = np.maximum(path_ce_max, np.max(ce_path, axis=1))
        path_err_max = np.maximum(path_err_max, np.max(err_path, axis=1))

    b_ce = path_ce_max - endpoint_ce_max
    b_ce = np.where(b_ce < 1e-12, 0.0, b_ce)
    return b_ce.astype(np.float64), path_err_max.astype(np.float64)


def compute_one_payload(config: dict[str, Any], payload: DatasetPayload) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    theta, theta_norms, ref_ids = load_theta_matrix(payload.selected_refs)
    x, y = load_dataset(config, payload)
    geom = pairwise_geometry(theta)
    ref_ce, ref_err = eval_ce_err(theta, x, y, config)
    center = np.mean(theta, axis=0, keepdims=True)
    center_ce, center_err = eval_ce_err(center, x, y, config)
    b_ce, b_err = linear_barriers(theta, ref_ce, ref_err, x, y, config, geom["pair_i"], geom["pair_j"])

    summary = {
        "beta": payload.beta,
        "dataset_id": payload.dataset_tag,
        "K": int(theta.shape[0]),
        "S_ref": geom["S_ref"],
        "Q_ref": geom["Q_ref"],
        "H_err": float(center_err[0]),
        "H_CE": float(center_ce[0] - np.median(ref_ce)),
        "B_CE_mean": float(np.mean(b_ce)),
        "B_CE_median": float(np.median(b_ce)),
        "B_CE_q90": float(np.quantile(b_ce, 0.9)),
        "B_err_mean": float(np.mean(b_err)),
        "B_err_median": float(np.median(b_err)),
        "B_err_q90": float(np.quantile(b_err, 0.9)),
    }

    pair_rows: list[dict[str, Any]] = []
    pair_i = geom["pair_i"]
    pair_j = geom["pair_j"]
    for idx, (i, j) in enumerate(zip(pair_i, pair_j, strict=True)):
        pair_rows.append(
            {
                "beta": payload.beta,
                "dataset_id": payload.dataset_tag,
                "ref_i": ref_ids[int(i)],
                "ref_j": ref_ids[int(j)],
                "D_ij": float(geom["D_vals"][idx]),
                "q_ij": float(geom["q_vals"][idx]),
                "B_CE_ij": float(b_ce[idx]),
                "B_err_ij": float(b_err[idx]),
            }
        )

    aux = {
        "cell_id": payload.cell_id,
        "dataset_id": payload.dataset_tag,
        "beta": payload.beta,
        "ref_ids": ref_ids,
        "theta_norms": theta_norms.tolist(),
        "ref_CE": ref_ce.tolist(),
        "ref_err": ref_err.tolist(),
        "center_CE": float(center_ce[0]),
        "center_err": float(center_err[0]),
    }
    return summary, pair_rows, aux


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_phi_descriptor(config: dict[str, Any], selected_betas: list[float]) -> list[dict[str, Any]]:
    phi_path = repo_path(config["proxy_summary_root"]) / "absolute_phi_by_beta_radius.csv"
    if not phi_path.exists():
        return []
    target_radius = float(config["proxy_descriptor_radius"])
    rows_by_beta: dict[float, list[dict[str, str]]] = defaultdict(list)
    with phi_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            beta = round(float(row["beta"]), 12)
            if beta in {round(float(x), 12) for x in selected_betas}:
                rows_by_beta[beta].append(row)
    out: list[dict[str, Any]] = []
    for beta in selected_betas:
        b = round(float(beta), 12)
        candidates = rows_by_beta.get(b, [])
        if not candidates:
            continue
        chosen = min(candidates, key=lambda row: abs(float(row["radius"]) - target_radius))
        out.append(
            {
                "beta": float(beta),
                "requested_radius": target_radius,
                "radius": float(chosen["radius"]),
                "phi_E": float(chosen["phi_energy"]),
                "phi_full": float(chosen["phi_full"]),
                "ref_count": int(float(chosen["ref_count"])),
                "dataset_count": int(float(chosen["dataset_count"])),
            }
        )
    return out


def mean_sem(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    sem = float(sd / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, sd, sem


def write_aggregate_tables(
    config: dict[str, Any],
    output_root: Path,
    summary_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_betas = [float(x) for x in config["selected_betas"]]
    phi_rows = read_phi_descriptor(config, selected_betas)
    phi_by_beta = {round(row["beta"], 12): row for row in phi_rows}
    by_beta: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_beta[round(float(row["beta"]), 12)].append(row)

    metrics = [
        "S_ref",
        "Q_ref",
        "H_err",
        "H_CE",
        "B_CE_mean",
        "B_CE_median",
        "B_CE_q90",
        "B_err_mean",
        "B_err_median",
        "B_err_q90",
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for beta in selected_betas:
        key = round(beta, 12)
        rows = by_beta.get(key, [])
        if not rows:
            continue
        out: dict[str, Any] = {"beta": beta, "dataset_count": len(rows), "K": int(rows[0]["K"])}
        for metric in metrics:
            mean, sd, sem = mean_sem([float(row[metric]) for row in rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_sd"] = sd
            out[f"{metric}_sem"] = sem
        phi = phi_by_beta.get(key)
        if phi:
            out["phi_E_dstar"] = phi["phi_E"]
            out["phi_E_radius"] = phi["radius"]
        aggregate_rows.append(out)

    if aggregate_rows:
        columns = list(aggregate_rows[0].keys())
        write_csv(output_root / "geometry_summary_beta_aggregate.csv", aggregate_rows, columns)
    if phi_rows:
        write_csv(
            output_root / "phi_E_descriptor_by_beta.csv",
            phi_rows,
            ["beta", "requested_radius", "radius", "phi_E", "phi_full", "ref_count", "dataset_count"],
        )
    return aggregate_rows, phi_rows


def matrix_from_pair_rows(pair_rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, list[int]]:
    refs = sorted({int(row["ref_i"]) for row in pair_rows} | {int(row["ref_j"]) for row in pair_rows})
    index = {ref_id: idx for idx, ref_id in enumerate(refs)}
    n = len(refs)
    mat = np.zeros((n, n), dtype=np.float64)
    if metric == "D_ij" or metric == "B_CE_ij" or metric == "B_err_ij":
        diag = 0.0
    elif metric == "q_ij":
        diag = 1.0
    else:
        diag = np.nan
    np.fill_diagonal(mat, diag)
    for row in pair_rows:
        i = index[int(row["ref_i"])]
        j = index[int(row["ref_j"])]
        value = float(row[metric])
        mat[i, j] = value
        mat[j, i] = value
    return mat, refs


def classical_mds(d_mat: np.ndarray) -> np.ndarray:
    n = d_mat.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d_mat**2) @ j
    vals, vecs = np.linalg.eigh((b + b.T) / 2.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order[:2]]
    vecs = vecs[:, order[:2]]
    vals = np.maximum(vals, 0.0)
    coords = vecs * np.sqrt(vals)[None, :]
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(n)])
    return coords


def selected_representatives(summary_rows: list[dict[str, Any]], selected_betas: list[float]) -> list[dict[str, Any]]:
    if not selected_betas:
        return []
    picks = [selected_betas[0], selected_betas[len(selected_betas) // 2], selected_betas[-1]]
    by_beta: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_beta[round(float(row["beta"]), 12)].append(row)
    reps: list[dict[str, Any]] = []
    for beta in picks:
        key = round(float(beta), 12)
        rows = by_beta[key]
        target_s = float(np.mean([float(row["S_ref"]) for row in rows]))
        chosen = min(rows, key=lambda row: abs(float(row["S_ref"]) - target_s))
        reps.append({"beta": float(beta), "dataset_id": chosen["dataset_id"]})
    return reps


def simple_line_with_error(ax: Any, x: np.ndarray, y: np.ndarray, sem: np.ndarray, label: str, color: str) -> None:
    ax.plot(x, y, marker="o", linewidth=1.8, color=color, label=label)
    if np.any(sem > 0):
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)
    ax.grid(True, alpha=0.25)


def generate_figures(
    config: dict[str, Any],
    output_root: Path,
    figure_root: Path,
    summary_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    phi_rows: list[dict[str, Any]],
    aux_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    figure_root.mkdir(parents=True, exist_ok=True)
    selected_betas = [float(x) for x in config["selected_betas"]]
    reps = selected_representatives(summary_rows, selected_betas)
    write_csv(output_root / "representative_datasets_for_figures.csv", reps, ["beta", "dataset_id"])

    agg = {round(float(row["beta"]), 12): row for row in aggregate_rows}
    betas = np.asarray([float(row["beta"]) for row in aggregate_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("S_ref", "S_ref", "tab:blue"),
        ("Q_ref", "Q_ref", "tab:green"),
        ("H_err", "H_err", "tab:red"),
        ("H_CE", "H_CE", "tab:purple"),
    ]
    for ax, (metric, ylabel, color) in zip(axes.ravel(), panels, strict=True):
        means = np.asarray([agg[round(b, 12)][f"{metric}_mean"] for b in betas], dtype=np.float64)
        sems = np.asarray([agg[round(b, 12)][f"{metric}_sem"] for b in betas], dtype=np.float64)
        simple_line_with_error(ax, betas, means, sems, ylabel, color)
        ax.set_xlabel("beta")
        ax.set_ylabel(ylabel)
    fig.suptitle("Reference-cloud spread, alignment, and center defect")
    fig.savefig(figure_root / "fig01_S_Q_H_vs_beta.png", dpi=220)
    plt.close(fig)

    phi_by_beta = {round(float(row["beta"]), 12): float(row["phi_E"]) for row in phi_rows}
    scatter_x = np.asarray([float(row["S_ref"]) for row in summary_rows])
    scatter_beta = np.asarray([float(row["beta"]) for row in summary_rows])
    size_values = np.asarray([abs(phi_by_beta.get(round(float(row["beta"]), 12), 0.0)) for row in summary_rows])
    if np.ptp(size_values) > 0:
        sizes = 35.0 + 90.0 * (size_values - np.min(size_values)) / np.ptp(size_values)
    else:
        sizes = np.full_like(size_values, 55.0, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "H_CE", "H_CE"),
        (axes[1], "H_err", "H_err"),
    ]:
        sc = ax.scatter(
            scatter_x,
            np.asarray([float(row[metric]) for row in summary_rows]),
            c=scatter_beta,
            s=sizes,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.82,
        )
        ax.set_xlabel("S_ref")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes, label="beta")
    fig.suptitle("S-H phase scatter; marker size uses |phi_E(d*)| when available")
    fig.savefig(figure_root / "fig02_S_H_phase_scatter.png", dpi=220)
    plt.close(fig)

    pairs_by_dataset: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        pairs_by_dataset[(round(float(row["beta"]), 12), str(row["dataset_id"]))].append(row)

    fig, axes = plt.subplots(2, len(reps), figsize=(4.2 * len(reps), 7.0), constrained_layout=True)
    if len(reps) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, rep in enumerate(reps):
        rows = pairs_by_dataset[(round(float(rep["beta"]), 12), str(rep["dataset_id"]))]
        d_mat, _ = matrix_from_pair_rows(rows, "D_ij")
        q_mat, _ = matrix_from_pair_rows(rows, "q_ij")
        order = np.argsort(classical_mds(d_mat)[:, 0])
        im0 = axes[0, col].imshow(d_mat[np.ix_(order, order)], cmap="magma")
        axes[0, col].set_title(f"beta={rep['beta']:.2f}, {rep['dataset_id']}\nD_ij")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
        im1 = axes[1, col].imshow(q_mat[np.ix_(order, order)], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[1, col].set_title("q_ij")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
    fig.suptitle("Pairwise distance and cosine heatmaps")
    fig.savefig(figure_root / "fig03_pairwise_distance_cosine_heatmaps.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(reps), figsize=(4.5 * len(reps), 4.1), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, reps, strict=True):
        rows = pairs_by_dataset[(round(float(rep["beta"]), 12), str(rep["dataset_id"]))]
        d_mat, _ = matrix_from_pair_rows(rows, "D_ij")
        b_mat, _ = matrix_from_pair_rows(rows, "B_CE_ij")
        order = np.argsort(classical_mds(d_mat)[:, 0])
        im = ax.imshow(b_mat[np.ix_(order, order)], cmap="inferno")
        ax.set_title(f"beta={rep['beta']:.2f}\n{rep['dataset_id']}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Linear CE barrier heatmaps, B_CE_ij")
    fig.savefig(figure_root / "fig04_linear_barrier_heatmap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for metric, color in [
        ("B_CE_median", "tab:orange"),
        ("B_CE_q90", "tab:red"),
        ("B_CE_mean", "tab:brown"),
    ]:
        means = np.asarray([agg[round(b, 12)][f"{metric}_mean"] for b in betas], dtype=np.float64)
        sems = np.asarray([agg[round(b, 12)][f"{metric}_sem"] for b in betas], dtype=np.float64)
        simple_line_with_error(axes[0], betas, means, sems, metric, color)
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("CE barrier")
    axes[0].legend(fontsize=8)
    for metric, color in [
        ("B_err_median", "tab:blue"),
        ("B_err_q90", "tab:purple"),
        ("B_err_mean", "tab:green"),
    ]:
        means = np.asarray([agg[round(b, 12)][f"{metric}_mean"] for b in betas], dtype=np.float64)
        sems = np.asarray([agg[round(b, 12)][f"{metric}_sem"] for b in betas], dtype=np.float64)
        simple_line_with_error(axes[1], betas, means, sems, metric, color)
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("max error along line")
    axes[1].legend(fontsize=8)
    fig.suptitle("B_lin summaries vs beta")
    fig.savefig(figure_root / "fig05_Blin_vs_beta.png", dpi=220)
    plt.close(fig)

    aux_by_dataset = {(round(float(row["beta"]), 12), str(row["dataset_id"])): row for row in aux_rows}
    phi_minmax = [float(row["phi_E"]) for row in phi_rows] or [0.0, 1.0]
    phi_min = min(phi_minmax)
    phi_max = max(phi_minmax)
    rep_plot_data: list[dict[str, Any]] = []
    all_edge_values: list[float] = []
    k_edges = int(config["knn_edges"])
    for rep in reps:
        key = (round(float(rep["beta"]), 12), str(rep["dataset_id"]))
        rows = pairs_by_dataset[key]
        d_mat, _ = matrix_from_pair_rows(rows, "D_ij")
        b_mat, _ = matrix_from_pair_rows(rows, "B_CE_ij")
        coords = classical_mds(d_mat)
        edge_pairs: set[tuple[int, int]] = set()
        for i in range(d_mat.shape[0]):
            nearest = np.argsort(d_mat[i])[1 : k_edges + 1]
            for j in nearest:
                edge_pairs.add(tuple(sorted((i, int(j)))))
        edge_pairs_sorted = sorted(edge_pairs)
        edge_values = np.asarray([b_mat[i, j] for i, j in edge_pairs_sorted], dtype=np.float64)
        all_edge_values.extend(edge_values.tolist())
        rep_plot_data.append(
            {
                "rep": rep,
                "key": key,
                "coords": coords,
                "edge_pairs": edge_pairs_sorted,
                "edge_values": edge_values,
            }
        )
    edge_min = min(all_edge_values) if all_edge_values else 0.0
    edge_max = max(all_edge_values) if all_edge_values else 1.0
    if edge_max <= edge_min:
        edge_max = edge_min + 1.0
    edge_norm = Normalize(vmin=edge_min, vmax=edge_max)
    node_norm = Normalize(vmin=phi_min, vmax=phi_max if phi_max > phi_min else phi_min + 1.0)
    fig, axes = plt.subplots(1, len(reps), figsize=(5.0 * len(reps), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, plot_data in zip(axes, rep_plot_data, strict=True):
        rep = plot_data["rep"]
        key = plot_data["key"]
        coords = plot_data["coords"]
        aux = aux_by_dataset[key]
        theta_norms = np.asarray(aux["theta_norms"], dtype=np.float64)
        if np.ptp(theta_norms) > 0:
            node_sizes = 45.0 + 115.0 * (theta_norms - np.min(theta_norms)) / np.ptp(theta_norms)
        else:
            node_sizes = np.full_like(theta_norms, 80.0)
        edge_pairs = plot_data["edge_pairs"]
        edge_values = plot_data["edge_values"]
        segments = [(coords[i], coords[j]) for i, j in edge_pairs]
        if segments:
            widths = np.full(edge_values.shape, 1.2)
            if np.ptp(edge_values) > 0:
                widths = 0.8 + 3.0 * (edge_values - np.min(edge_values)) / np.ptp(edge_values)
            lc = LineCollection(segments, cmap="inferno", norm=edge_norm, linewidths=widths, alpha=0.78)
            lc.set_array(edge_values)
            ax.add_collection(lc)
        phi = phi_by_beta.get(round(float(rep["beta"]), 12), 0.0)
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=np.full(coords.shape[0], phi),
            cmap="viridis",
            norm=node_norm,
            s=node_sizes,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        ax.set_title(f"beta={rep['beta']:.2f}\n{rep['dataset_id']}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.18)
    edge_mappable = ScalarMappable(norm=edge_norm, cmap="inferno")
    edge_mappable.set_array([])
    fig.colorbar(edge_mappable, ax=axes, fraction=0.02, pad=0.02, label="B_CE_ij on kNN edges")
    fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.07, pad=0.08, label="phi_E(d*)")
    fig.suptitle("MDS reference map with kNN linear-barrier overlay")
    fig.savefig(figure_root / "fig06_reference_map_barrier_overlay.png", dpi=220)
    plt.close(fig)


def run(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_root = repo_path(args.output_root) if args.output_root else repo_path(config["output_root"])
    figure_root = repo_path(args.figure_root) if args.figure_root else repo_path(config["figure_root"])
    qc_root = repo_path(args.qc_root) if args.qc_root else repo_path(config["qc_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    qc_root.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figure_root.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        summary_rows = load_csv_dicts(output_root / "geometry_summary_by_beta.csv")
        pairwise_rows = load_csv_dicts(output_root / "pairwise_metrics.csv")
        aggregate_rows = load_csv_dicts(output_root / "geometry_summary_beta_aggregate.csv")
        phi_rows = load_csv_dicts(output_root / "phi_E_descriptor_by_beta.csv")
        aux_rows = load_json(output_root / "reference_eval_auxiliary.json")["datasets"]
        generate_figures(config, output_root, figure_root, summary_rows, pairwise_rows, aggregate_rows, phi_rows, aux_rows)
        return {
            "figures_only": True,
            "summary_rows": len(summary_rows),
            "pairwise_rows": len(pairwise_rows),
        }

    payloads = load_reference_payloads(config, args.max_datasets_per_beta)
    start_time = time.perf_counter()
    summary_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    aux_rows: list[dict[str, Any]] = []

    for idx, payload in enumerate(payloads, start=1):
        item_start = time.perf_counter()
        summary, pair_rows, aux = compute_one_payload(config, payload)
        summary_rows.append(summary)
        pairwise_rows.extend(pair_rows)
        aux_rows.append(aux)
        elapsed = time.perf_counter() - item_start
        print(
            f"[{idx:04d}/{len(payloads):04d}] beta={payload.beta:.2f} dataset={payload.dataset_tag} "
            f"S={summary['S_ref']:.6g} Q={summary['Q_ref']:.6g} "
            f"H_CE={summary['H_CE']:.6g} B_CE_med={summary['B_CE_median']:.6g} "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )

    write_csv(output_root / "geometry_summary_by_beta.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(output_root / "pairwise_metrics.csv", pairwise_rows, PAIRWISE_COLUMNS)
    write_json(output_root / "reference_eval_auxiliary.json", {"datasets": aux_rows})
    aggregate_rows, phi_rows = write_aggregate_tables(config, output_root, summary_rows)
    if not args.no_figures:
        generate_figures(config, output_root, figure_root, summary_rows, pairwise_rows, aggregate_rows, phi_rows, aux_rows)

    elapsed = time.perf_counter() - start_time
    expected_dataset_count = len(config["selected_betas"]) * (
        int(config["datasets_per_beta"]) if args.max_datasets_per_beta is None else min(int(config["datasets_per_beta"]), args.max_datasets_per_beta)
    )
    validation = {
        "claim": "S/Q/H/B_lin reference-cloud proxy metrics only; no graph connectivity G_tau was computed.",
        "dataset_count": len(summary_rows),
        "expected_dataset_count": expected_dataset_count,
        "pairwise_row_count": len(pairwise_rows),
        "expected_pairwise_row_count": len(summary_rows) * int(config["references_per_dataset"]) * (int(config["references_per_dataset"]) - 1) // 2,
        "summary_columns": SUMMARY_COLUMNS,
        "pairwise_columns": PAIRWISE_COLUMNS,
        "t_grid_count": int(config["t_grid_count"]),
        "uses_interior_grid_plus_exact_endpoints": True,
        "passed": True,
    }
    validation["passed"] = (
        validation["dataset_count"] == validation["expected_dataset_count"]
        and validation["pairwise_row_count"] == validation["expected_pairwise_row_count"]
    )
    write_json(qc_root / "validation.json", validation)

    timing = {
        "elapsed_seconds": elapsed,
        "dataset_count": len(summary_rows),
        "seconds_per_dataset": elapsed / max(1, len(summary_rows)),
        "max_datasets_per_beta": args.max_datasets_per_beta,
        "figures_generated": not args.no_figures,
    }
    write_json(qc_root / "timing_report.json", timing)
    report_lines = [
        "# Reference Cloud Proxy Metrics Run Report",
        "",
        f"- Dataset units: {len(summary_rows)}",
        f"- Pairwise rows: {len(pairwise_rows)}",
        f"- Elapsed seconds: {elapsed:.3f}",
        f"- Seconds per dataset: {timing['seconds_per_dataset']:.3f}",
        f"- t_grid_count: {int(config['t_grid_count'])}",
        f"- Figures generated: {not args.no_figures}",
        "",
        "Claim boundary: `B_lin` is a straight-line CE/error barrier diagnostic only. It is not a nonlinear connectivity proof, and `G_tau` was not computed.",
    ]
    (qc_root / "run_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"validation": validation, "timing": timing}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Stage JSON config path.")
    parser.add_argument("--max-datasets-per-beta", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--figure-root", default=None)
    parser.add_argument("--qc-root", default=None)
    parser.add_argument("--figures-only", action="store_true", help="Regenerate figures from existing CSV and auxiliary outputs.")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_json(repo_path(args.config))
    if int(config["param_count"]) != P:
        raise ValueError(f"config param_count={config['param_count']} but dnn_model.P={P}")
    result = run(config, args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
