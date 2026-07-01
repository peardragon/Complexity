from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .graphs import mutual_knn_graph
from .layout import (
    DATASET_ID_FOR_SAMPLE,
    K_GRAPH,
    RAW_ROOT,
    SAMPLE_SUMMARY_ROOT,
    SPIN_SUMMARY_ROOT,
    beta_dirs,
    beta_from_dir_name,
    dataset_dirs,
    dataset_id_from_dir,
    dataset_label,
    find_dataset_dir,
    source_dataset_path,
    source_image_path,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_sample_summary() -> None:
    SAMPLE_SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    dirs = beta_dirs()
    if not dirs:
        raise FileNotFoundError(RAW_ROOT)

    for panel_order, beta_dir in enumerate(dirs, start=1):
        find_dataset_dir(beta_dir, DATASET_ID_FOR_SAMPLE)
        beta = beta_from_dir_name(beta_dir.name)
        rows.append(
            {
                "panel_order": panel_order,
                "beta_ising": f"{beta:.2f}",
                "beta_dir": beta_dir.name,
                "dataset_label": dataset_label(DATASET_ID_FOR_SAMPLE),
                "source_dataset_path": source_dataset_path(beta_dir, DATASET_ID_FOR_SAMPLE),
                "source_image_path": source_image_path(beta_dir, DATASET_ID_FOR_SAMPLE),
            }
        )

    _write_csv(
        SAMPLE_SUMMARY_ROOT / "selected_sample_indices.csv",
        rows,
        [
            "panel_order",
            "beta_ising",
            "beta_dir",
            "dataset_label",
            "source_dataset_path",
            "source_image_path",
        ],
    )


def _spin_alignment_for_dataset(dataset_dir: Path) -> dict[str, object]:
    data = np.load(dataset_dir / "dataset.npz")
    x_raw = np.asarray(data["X_raw"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    edges, _sigma_med = mutual_knn_graph(x_raw, K_GRAPH)
    edge_i = np.asarray([i for i, _j in edges], dtype=np.int32)
    edge_j = np.asarray([j for _i, j in edges], dtype=np.int32)
    edge_products = y[edge_i] * y[edge_j]
    return {
        "edge_alignment": float(np.mean(edge_products)),
        "domain_wall_fraction": float(np.mean(edge_products < 0.0)),
        "n_edges": int(edge_products.size),
    }


def build_spin_dynamics_summary() -> None:
    SPIN_SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    per_dataset_rows: list[dict[str, object]] = []
    grouped_rows: dict[float, list[dict[str, object]]] = defaultdict(list)
    for beta_dir in beta_dirs():
        beta = beta_from_dir_name(beta_dir.name)
        for dataset_dir in dataset_dirs(beta_dir):
            dataset_id = dataset_id_from_dir(dataset_dir)
            metrics = _spin_alignment_for_dataset(dataset_dir)
            row = {
                "beta_ising": f"{beta:.2f}",
                "temperature_inverse_beta": f"{1.0 / beta:.6f}",
                "beta_dir": beta_dir.name,
                "dataset_label": dataset_label(dataset_id),
                "edge_alignment": f"{metrics['edge_alignment']:.10f}",
                "domain_wall_fraction": f"{metrics['domain_wall_fraction']:.10f}",
                "n_edges": metrics["n_edges"],
                "source_dataset_path": source_dataset_path(beta_dir, dataset_id),
            }
            per_dataset_rows.append(row)
            grouped_rows[beta].append(row)

    by_beta_rows: list[dict[str, object]] = []
    for beta in sorted(grouped_rows):
        rows = grouped_rows[beta]
        alignments = np.asarray([float(row["edge_alignment"]) for row in rows], dtype=np.float64)
        walls = np.asarray([float(row["domain_wall_fraction"]) for row in rows], dtype=np.float64)
        edge_counts = np.asarray([int(row["n_edges"]) for row in rows], dtype=np.float64)
        std_alignment = float(np.std(alignments, ddof=1)) if alignments.size > 1 else 0.0
        std_wall = float(np.std(walls, ddof=1)) if walls.size > 1 else 0.0
        by_beta_rows.append(
            {
                "beta_ising": f"{beta:.2f}",
                "temperature_inverse_beta": f"{1.0 / beta:.6f}",
                "n_datasets": int(alignments.size),
                "mean_edge_alignment": f"{float(np.mean(alignments)):.10f}",
                "sem_edge_alignment": f"{std_alignment / math.sqrt(max(1, alignments.size)):.10f}",
                "std_edge_alignment": f"{std_alignment:.10f}",
                "mean_domain_wall_fraction": f"{float(np.mean(walls)):.10f}",
                "sem_domain_wall_fraction": f"{std_wall / math.sqrt(max(1, walls.size)):.10f}",
                "mean_n_edges": f"{float(np.mean(edge_counts)):.2f}",
            }
        )

    by_dataset_csv = SPIN_SUMMARY_ROOT / "spin_alignment_by_dataset.csv"
    by_beta_csv = SPIN_SUMMARY_ROOT / "spin_alignment_by_beta.csv"
    _write_csv(
        by_dataset_csv,
        per_dataset_rows,
        [
            "beta_ising",
            "temperature_inverse_beta",
            "beta_dir",
            "dataset_label",
            "edge_alignment",
            "domain_wall_fraction",
            "n_edges",
            "source_dataset_path",
        ],
    )
    _write_csv(
        by_beta_csv,
        by_beta_rows,
        [
            "beta_ising",
            "temperature_inverse_beta",
            "n_datasets",
            "mean_edge_alignment",
            "sem_edge_alignment",
            "std_edge_alignment",
            "mean_domain_wall_fraction",
            "sem_domain_wall_fraction",
            "mean_n_edges",
        ],
    )


def build_summarized_outputs() -> None:
    build_sample_summary()
    build_spin_dynamics_summary()
