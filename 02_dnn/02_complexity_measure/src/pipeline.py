from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from defaults import DEFAULT_CONFIG
from io_utils import load_csv_rows, load_json, now_iso, save_csv, save_json, start_verbose_print_capture
from manifest import build_roots, write_manifest
from nmstv import nmstv_from_graph_cache
from stats import nanmean, nanstd, pearson_corr, safe_float, spearman_corr
from visuals import plot_heatmap, plot_multiscale_curves, plot_scatter, plot_scatter_with_errorbars, plot_series_grid


CELL_ID_RE = re.compile(r"^cell_beta_(?P<beta>\d+p\d+)_p_(?P<p>\d+p\d+)$")


def _parse_cell_id_series_values(cell_id: str) -> tuple[float | None, float | None]:
    match = CELL_ID_RE.match(str(cell_id))
    if not match:
        return None, None
    return float(match.group("beta").replace("p", ".")), float(match.group("p").replace("p", "."))


def _cell_in_beta_series(cell_id: str) -> bool:
    _, p_val = _parse_cell_id_series_values(cell_id)
    return p_val is not None and abs(p_val) <= 1e-12


def _cell_in_p_series(cell_id: str) -> bool:
    beta_val, _ = _parse_cell_id_series_values(cell_id)
    return beta_val is not None and abs(beta_val - 0.60) <= 1e-12


def prepare_graph_cache(X: np.ndarray, k_graph: int) -> Dict[str, object]:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=int(k_graph) + 1, metric="euclidean", n_jobs=1)
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    dists_k = dists[:, 1:]
    idx_k = idx[:, 1:]
    sigma_med = float(np.median(dists_k))
    knn_sets = [set(row.tolist()) for row in idx_k]
    edges = set()
    for i in range(n):
        for j in idx_k[i]:
            jj = int(j)
            if i in knn_sets[jj]:
                a, b = (i, jj) if i < jj else (jj, i)
                edges.add((a, b))
    edge_list = sorted(edges)
    idx_i = np.asarray([i for i, _ in edge_list], dtype=np.int32)
    idx_j = np.asarray([j for _, j in edge_list], dtype=np.int32)
    diff = X[idx_i] - X[idx_j]
    dist2 = np.sum(diff * diff, axis=1)
    return {
        "edges": edge_list,
        "sigma_med": sigma_med,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "dist2": dist2,
        "n_edges": int(len(edge_list)),
        "k_graph": int(k_graph),
    }


def merged_config(config_path: Path, upstream_manifest: Path, force: bool) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        config.update(load_json(config_path, {}) or {})
    config["upstream_manifest"] = str(upstream_manifest)
    config["force"] = bool(force)
    return config


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _existing_success_manifest(summary_root: Path, *, force: bool) -> Path | None:
    manifest_path = summary_root / "manifest.json"
    if force or not manifest_path.exists():
        return None
    payload = load_json(manifest_path, {}) or {}
    if payload.get("status") != "success":
        return None
    summary_outputs = payload.get("summary_outputs", [])
    if not isinstance(summary_outputs, list):
        return None
    if not all(Path(str(path)).exists() for path in summary_outputs):
        return None
    return manifest_path


def run_pipeline(*, part_root: Path, config_path: Path, upstream_manifest: Path, force: bool, verbose: bool = False) -> Path:
    started_at = now_iso()
    upstream = load_json(upstream_manifest, {})
    dimension = int(upstream["dimension"])
    config = merged_config(config_path, upstream_manifest, force)
    run_group = "smoke_runs" if config_path.name == "smoke.json" else "runs"
    config["force"] = bool(config["force"]) or run_group == "smoke_runs"
    output_root = part_root / f"d{int(dimension)}"
    cfg_hash, raw_root, summary_root = build_roots(output_root, config, run_group=run_group, upstream_manifest=upstream_manifest)
    existing_manifest = _existing_success_manifest(summary_root, force=bool(config["force"]))
    if existing_manifest is not None:
        return existing_manifest
    log_capture = start_verbose_print_capture(summary_root, enabled=verbose)
    log_path = str(log_capture.log_path) if log_capture is not None else None
    dataset_index = load_csv_rows(Path(upstream["summary_output_root"]) / "dataset_index.csv")
    synthetic_part_root = Path(upstream["summary_output_root"]).parents[2]
    dataset_rows = []
    multiscale_dataset_rows = []
    computed_by_raw_path: dict[str, dict[str, float]] = {}
    seen_dataset_keys: set[tuple[str, int, int]] = set()
    total = len(dataset_index)
    _log(verbose, f"[complexity_measure] D={dimension}, datasets={total}")
    for idx, row in enumerate(dataset_index, start=1):
        dataset_key = (str(row["cell_id"]), int(row["dataset_id"]), int(row["seed"]))
        if dataset_key in seen_dataset_keys:
            _log(verbose, f"[complexity_measure] [{idx}/{total}] {row['cell_id']} dataset_{int(row['dataset_id']):03d}: duplicate index skip")
            continue
        seen_dataset_keys.add(dataset_key)
        _log(verbose, f"[complexity_measure] [{idx}/{total}] {row['cell_id']} dataset_{int(row['dataset_id']):03d}")
        raw_path = (synthetic_part_root / row["dataset_raw_path"]).resolve()
        cache_key = str(raw_path)
        if cache_key in computed_by_raw_path:
            comp = computed_by_raw_path[cache_key]
        else:
            npz = np.load(raw_path)
            X_raw = np.asarray(npz["X_raw"], dtype=np.float64)
            y = np.asarray(npz["y"], dtype=np.int8)
            graph_cache = prepare_graph_cache(X_raw, int(config["k_graph"]))
            comp = nmstv_from_graph_cache(y, graph_cache, list(config["nmstv_scales"]))
            computed_by_raw_path[cache_key] = comp
        dataset_rows.append(
            {
                "cell_id": row["cell_id"],
                "series": row["series"],
                "dataset_id": int(row["dataset_id"]),
                "seed": int(row["seed"]),
                "beta_ising": float(row["beta_ising"]),
                "rewire_p": float(row["rewire_p"]),
                "C": float(comp["C"]),
                "sigma_med": float(comp["sigma_med"]),
                "n_edges": int(comp["n_edges"]),
            }
        )
        for scale, c_s, rho_s in zip(comp["scales"], comp["C_s"], comp["rho_s"]):
            multiscale_dataset_rows.append(
                {
                    "cell_id": row["cell_id"],
                    "series": row["series"],
                    "dataset_id": int(row["dataset_id"]),
                    "seed": int(row["seed"]),
                    "beta_ising": float(row["beta_ising"]),
                    "rewire_p": float(row["rewire_p"]),
                    "scale": float(scale),
                    "C_s": float(c_s),
                    "rho_s": float(rho_s),
                }
            )
    by_cell = {}
    for row in dataset_rows:
        by_cell.setdefault(row["cell_id"], []).append(row)
    cell_rows = []
    multiscale_by_cell_scale: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in multiscale_dataset_rows:
        multiscale_by_cell_scale.setdefault((str(row["cell_id"]), float(row["scale"])), []).append(row)
    multiscale_cell_rows = []
    for cell_id, rows in sorted(by_cell.items()):
        cvals = [safe_float(row["C"]) for row in rows]
        cell_rows.append(
            {
                "cell_id": cell_id,
                "series": rows[0]["series"],
                "beta_ising": float(rows[0]["beta_ising"]),
                "rewire_p": float(rows[0]["rewire_p"]),
                "n_datasets": len(rows),
                "C_mean": nanmean(cvals),
                "C_std": nanstd(cvals),
            }
        )
    for (cell_id, scale), rows in sorted(multiscale_by_cell_scale.items(), key=lambda item: (item[0][0], item[0][1])):
        multiscale_cell_rows.append(
            {
                "cell_id": cell_id,
                "series": str(rows[0]["series"]),
                "beta_ising": float(rows[0]["beta_ising"]),
                "rewire_p": float(rows[0]["rewire_p"]),
                "scale": float(scale),
                "n_datasets": int(len(rows)),
                "C_s_mean": nanmean([safe_float(row["C_s"]) for row in rows]),
                "C_s_std": nanstd([safe_float(row["C_s"]) for row in rows]),
                "rho_s_mean": nanmean([safe_float(row["rho_s"]) for row in rows]),
                "rho_s_std": nanstd([safe_float(row["rho_s"]) for row in rows]),
            }
        )
    dataset_csv = summary_root / "complexity_summary_by_dataset.csv"
    cell_csv = summary_root / "complexity_summary_by_cell.csv"
    multiscale_dataset_csv = summary_root / "complexity_multiscale_by_dataset.csv"
    multiscale_cell_csv = summary_root / "complexity_multiscale_by_cell.csv"
    save_csv(dataset_csv, dataset_rows, ["cell_id", "series", "dataset_id", "seed", "beta_ising", "rewire_p", "C", "sigma_med", "n_edges"])
    save_csv(cell_csv, cell_rows, ["cell_id", "series", "beta_ising", "rewire_p", "n_datasets", "C_mean", "C_std"])
    save_csv(multiscale_dataset_csv, multiscale_dataset_rows, ["cell_id", "series", "dataset_id", "seed", "beta_ising", "rewire_p", "scale", "C_s", "rho_s"])
    save_csv(multiscale_cell_csv, multiscale_cell_rows, ["cell_id", "series", "beta_ising", "rewire_p", "scale", "n_datasets", "C_s_mean", "C_s_std", "rho_s_mean", "rho_s_std"])
    figs_dir = summary_root / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    beta_rows = [row for row in dataset_rows if str(row.get("series", "")).strip().lower() == "beta"]
    p_rows = [row for row in dataset_rows if str(row.get("series", "")).strip().lower() == "p"]
    beta_raw_fig = figs_dir / "nmstv_vs_beta_raw.png"
    p_raw_fig = figs_dir / "nmstv_vs_p_raw.png"
    plot_scatter(np.asarray([safe_float(row["beta_ising"]) for row in beta_rows]), np.asarray([safe_float(row["C"]) for row in beta_rows]), beta_raw_fig, title=f"nMSTV vs beta / D={dimension}", xlabel="beta", ylabel="C (nMSTV)")
    plot_scatter(np.asarray([safe_float(row["rewire_p"]) for row in p_rows]), np.asarray([safe_float(row["C"]) for row in p_rows]), p_raw_fig, title=f"nMSTV vs p / D={dimension}", xlabel="p", ylabel="C (nMSTV)")
    beta_cell_rows = [row for row in cell_rows if str(row.get("series", "")).strip().lower() == "beta"]
    p_cell_rows = [row for row in cell_rows if str(row.get("series", "")).strip().lower() == "p"]
    beta_fig = figs_dir / "nmstv_vs_beta.png"
    p_fig = figs_dir / "nmstv_vs_p.png"
    plot_scatter_with_errorbars(np.asarray([safe_float(row["beta_ising"]) for row in beta_cell_rows]), np.asarray([safe_float(row["C_mean"]) for row in beta_cell_rows]), np.asarray([safe_float(row["C_std"]) for row in beta_cell_rows]), beta_fig, title=f"nMSTV vs beta / D={dimension}", xlabel="beta", ylabel="C mean +/- std")
    plot_scatter_with_errorbars(np.asarray([safe_float(row["rewire_p"]) for row in p_cell_rows]), np.asarray([safe_float(row["C_mean"]) for row in p_cell_rows]), np.asarray([safe_float(row["C_std"]) for row in p_cell_rows]), p_fig, title=f"nMSTV vs p / D={dimension}", xlabel="p", ylabel="C mean +/- std")
    beta_grid = figs_dir / "beta_series_grid.png"
    beta_grid_scatter = figs_dir / "beta_series_grid_scatter.png"
    p_grid = figs_dir / "p_series_grid.png"
    p_grid_scatter = figs_dir / "p_series_grid_scatter.png"
    plot_series_grid(dataset_index, cell_rows, synthetic_part_root, beta_grid, series_name="beta", title=f"beta series / D={dimension} representative datasets", panel_mode="filled")
    plot_series_grid(dataset_index, cell_rows, synthetic_part_root, beta_grid_scatter, series_name="beta", title=f"beta series / D={dimension} representative datasets / scatter", panel_mode="scatter")
    plot_series_grid(dataset_index, cell_rows, synthetic_part_root, p_grid, series_name="p", title=f"p series / D={dimension} representative datasets", panel_mode="filled")
    plot_series_grid(dataset_index, cell_rows, synthetic_part_root, p_grid_scatter, series_name="p", title=f"p series / D={dimension} representative datasets / scatter", panel_mode="scatter")
    beta_multiscale_rows = [row for row in multiscale_cell_rows if str(row.get("series", "")).strip().lower() == "beta"]
    beta_multiscale_rows.sort(key=lambda row: (safe_float(row["beta_ising"]), safe_float(row["scale"])))
    scales = sorted({safe_float(row["scale"]) for row in beta_multiscale_rows})
    beta_values = sorted({safe_float(row["beta_ising"]) for row in beta_multiscale_rows})
    c_curve_fig = figs_dir / "nmstv_vs_scale_beta_series.png"
    rho_curve_fig = figs_dir / "rho_vs_scale_beta_series.png"
    c_heatmap_fig = figs_dir / "nmstv_scale_heatmap_beta_series.png"
    rho_heatmap_fig = figs_dir / "rho_scale_heatmap_beta_series.png"
    span_fig = figs_dir / "nmstv_scale_span_vs_beta.png"
    rank_heatmap_fig = figs_dir / "nmstv_scale_rank_heatmap_beta_series.png"
    crossing_heatmap_fig = figs_dir / "nmstv_scale_crossing_count_beta_series.png"
    robust_crossing_heatmap_fig = figs_dir / "nmstv_scale_crossing_robust_beta_series.png"
    ordering_pairs_csv = summary_root / "complexity_multiscale_ordering_pairs.csv"
    ordering_summary_json = summary_root / "complexity_multiscale_ordering_summary.json"
    beta_curve_rows = []
    rho_curve_rows = []
    c_heatmap = np.full((len(beta_values), len(scales)), np.nan, dtype=np.float64)
    rho_heatmap = np.full((len(beta_values), len(scales)), np.nan, dtype=np.float64)
    span_rows = []
    std_heatmap = np.full((len(beta_values), len(scales)), np.nan, dtype=np.float64)
    for beta in beta_values:
        rows = [row for row in beta_multiscale_rows if abs(safe_float(row["beta_ising"]) - float(beta)) <= 1.0e-12]
        rows.sort(key=lambda row: safe_float(row["scale"]))
        x = np.asarray([safe_float(row["scale"]) for row in rows], dtype=np.float64)
        c_mean = np.asarray([safe_float(row["C_s_mean"]) for row in rows], dtype=np.float64)
        c_std = np.asarray([safe_float(row["C_s_std"]) for row in rows], dtype=np.float64)
        rho_mean = np.asarray([safe_float(row["rho_s_mean"]) for row in rows], dtype=np.float64)
        rho_std = np.asarray([safe_float(row["rho_s_std"]) for row in rows], dtype=np.float64)
        beta_curve_rows.append((f"beta={float(beta):.2f}", x, c_mean, c_std))
        rho_curve_rows.append((f"beta={float(beta):.2f}", x, rho_mean, rho_std))
        if x.size:
            span_rows.append(
                {
                    "beta_ising": float(beta),
                    "C_s_span": float(np.nanmax(c_mean) - np.nanmin(c_mean)),
                    "rho_s_span": float(np.nanmax(rho_mean) - np.nanmin(rho_mean)),
                    "C_s_fine_minus_coarse": float(c_mean[0] - c_mean[-1]),
                    "rho_s_fine_minus_coarse": float(rho_mean[0] - rho_mean[-1]),
                }
            )
        for row in rows:
            beta_idx = beta_values.index(float(beta))
            scale_idx = scales.index(float(row["scale"]))
            c_heatmap[beta_idx, scale_idx] = safe_float(row["C_s_mean"])
            rho_heatmap[beta_idx, scale_idx] = safe_float(row["rho_s_mean"])
            std_heatmap[beta_idx, scale_idx] = safe_float(row["C_s_std"])
    plot_multiscale_curves(beta_curve_rows, c_curve_fig, title=f"nMSTV vs scale / beta series / D={dimension}", ylabel="C_s", xlabel="scale multiplier")
    plot_multiscale_curves(rho_curve_rows, rho_curve_fig, title=f"rho vs scale / beta series / D={dimension}", ylabel="rho_s", xlabel="scale multiplier")
    plot_heatmap(c_heatmap, [f"{beta:.2f}" for beta in beta_values], [f"{scale:.2f}" for scale in scales], c_heatmap_fig, title=f"nMSTV scale heatmap / beta series / D={dimension}", colorbar_label="C_s mean")
    plot_heatmap(rho_heatmap, [f"{beta:.2f}" for beta in beta_values], [f"{scale:.2f}" for scale in scales], rho_heatmap_fig, title=f"rho scale heatmap / beta series / D={dimension}", colorbar_label="rho_s mean")
    if span_rows:
        plot_scatter_with_errorbars(
            np.asarray([safe_float(row["beta_ising"]) for row in span_rows], dtype=np.float64),
            np.asarray([safe_float(row["C_s_span"]) for row in span_rows], dtype=np.float64),
            np.zeros(len(span_rows), dtype=np.float64),
            span_fig,
            title=f"nMSTV multiscale span vs beta / D={dimension}",
            xlabel="beta",
            ylabel="max(C_s)-min(C_s)",
        )
    rank_matrix = np.full((len(beta_values), len(scales)), np.nan, dtype=np.float64)
    for scale_idx in range(len(scales)):
        values = c_heatmap[:, scale_idx]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        order = np.argsort(-values[finite])
        finite_indices = np.flatnonzero(finite)
        for rank, local_idx in enumerate(order, start=1):
            rank_matrix[finite_indices[local_idx], scale_idx] = float(rank)
    plot_heatmap(rank_matrix, [f"{beta:.2f}" for beta in beta_values], [f"{scale:.2f}" for scale in scales], rank_heatmap_fig, title=f"nMSTV rank heatmap / beta series / D={dimension}", colorbar_label="rank (1=largest)")
    crossing_matrix = np.full((len(beta_values), len(beta_values)), np.nan, dtype=np.float64)
    robust_crossing_matrix = np.full((len(beta_values), len(beta_values)), np.nan, dtype=np.float64)
    ordering_pair_rows = []
    raw_crossing_pairs = 0
    robust_crossing_pairs = 0
    for i, beta_i in enumerate(beta_values):
        for j, beta_j in enumerate(beta_values):
            if i == j:
                crossing_matrix[i, j] = 0.0
                robust_crossing_matrix[i, j] = 0.0
                continue
            diff = c_heatmap[i, :] - c_heatmap[j, :]
            finite = np.isfinite(diff)
            signs = np.sign(diff[finite])
            signs = signs[signs != 0.0]
            crossing_count = float(np.sum(signs[1:] != signs[:-1])) if signs.size >= 2 else 0.0
            pooled_std = np.sqrt(np.maximum(std_heatmap[i, :] ** 2 + std_heatmap[j, :] ** 2, 0.0))
            robust_diff = np.divide(diff, pooled_std, out=np.zeros_like(diff), where=np.isfinite(pooled_std) & (pooled_std > 1.0e-12))
            robust_signs = np.sign(robust_diff[np.isfinite(robust_diff) & (np.abs(robust_diff) >= 1.0)])
            robust_crossing_count = float(np.sum(robust_signs[1:] != robust_signs[:-1])) if robust_signs.size >= 2 else 0.0
            crossing_matrix[i, j] = crossing_count
            robust_crossing_matrix[i, j] = robust_crossing_count
            if i < j:
                raw_crossing_pairs += int(crossing_count > 0.0)
                robust_crossing_pairs += int(robust_crossing_count > 0.0)
                ordering_pair_rows.append(
                    {
                        "beta_i": float(beta_i),
                        "beta_j": float(beta_j),
                        "crossing_count": crossing_count,
                        "robust_crossing_count": robust_crossing_count,
                        "fine_minus_coarse_gap_i": float(c_heatmap[i, 0] - c_heatmap[i, -1]),
                        "fine_minus_coarse_gap_j": float(c_heatmap[j, 0] - c_heatmap[j, -1]),
                        "mean_gap_min": float(np.nanmin(np.abs(diff))) if np.any(np.isfinite(diff)) else float("nan"),
                        "z_gap_max_abs": float(np.nanmax(np.abs(robust_diff))) if np.any(np.isfinite(robust_diff)) else float("nan"),
                    }
                )
    save_csv(ordering_pairs_csv, ordering_pair_rows, ["beta_i", "beta_j", "crossing_count", "robust_crossing_count", "fine_minus_coarse_gap_i", "fine_minus_coarse_gap_j", "mean_gap_min", "z_gap_max_abs"])
    save_json(
        ordering_summary_json,
        {
            "beta_values": [float(beta) for beta in beta_values],
            "scales": [float(scale) for scale in scales],
            "pair_count": int(len(ordering_pair_rows)),
            "raw_crossing_pair_count": int(raw_crossing_pairs),
            "robust_crossing_pair_count": int(robust_crossing_pairs),
            "ordering_preserved_all_scales": bool(raw_crossing_pairs == 0),
            "ordering_preserved_all_scales_robust": bool(robust_crossing_pairs == 0),
        },
    )
    plot_heatmap(crossing_matrix, [f"{beta:.2f}" for beta in beta_values], [f"{beta:.2f}" for beta in beta_values], crossing_heatmap_fig, title=f"pairwise crossing count / beta series / D={dimension}", colorbar_label="crossing count")
    plot_heatmap(robust_crossing_matrix, [f"{beta:.2f}" for beta in beta_values], [f"{beta:.2f}" for beta in beta_values], robust_crossing_heatmap_fig, title=f"pairwise robust crossing count / beta series / D={dimension}", colorbar_label="robust crossing count")
    beta_corr = {
        "pearson_C_beta": pearson_corr([safe_float(row["beta_ising"]) for row in beta_rows], [safe_float(row["C"]) for row in beta_rows]),
        "spearman_C_beta": spearman_corr([safe_float(row["beta_ising"]) for row in beta_rows], [safe_float(row["C"]) for row in beta_rows]),
    }
    p_corr = {
        "pearson_C_p": pearson_corr([safe_float(row["rewire_p"]) for row in p_rows], [safe_float(row["C"]) for row in p_rows]),
        "spearman_C_p": spearman_corr([safe_float(row["rewire_p"]) for row in p_rows], [safe_float(row["C"]) for row in p_rows]),
    }
    corr_json = summary_root / "complexity_series_correlations.json"
    save_json(corr_json, {"beta_series": beta_corr, "p_series": p_corr})
    save_json(summary_root / "run_config.json", config)
    summary_outputs = [str(dataset_csv), str(cell_csv), str(multiscale_dataset_csv), str(multiscale_cell_csv), str(ordering_pairs_csv), str(ordering_summary_json), str(corr_json)]
    for path in (beta_raw_fig, p_raw_fig, beta_fig, p_fig, beta_grid, beta_grid_scatter, p_grid, p_grid_scatter, c_curve_fig, rho_curve_fig, c_heatmap_fig, rho_heatmap_fig, span_fig, rank_heatmap_fig, crossing_heatmap_fig, robust_crossing_heatmap_fig):
        if path.exists():
            summary_outputs.append(str(path))
    if log_path is not None:
        summary_outputs.append(log_path)
    write_manifest(
        summary_root,
        pipeline_id=str(config["pipeline_id"]),
        methodology_id=str(config["methodology_id"]),
        cfg_hash=cfg_hash,
        config_path=str(config_path),
        raw_root=raw_root,
        upstream_refs=[str(upstream_manifest)],
        summary_outputs=summary_outputs,
        dimension=dimension,
        run_group=run_group,
        started_at=started_at,
    )
    _log(verbose, f"[complexity_measure] wrote {len(dataset_rows)} dataset row(s) and {len(cell_rows)} cell row(s)")
    _log(verbose, f"[complexity_measure] wrote manifest -> {summary_root / 'manifest.json'}")
    if log_capture is not None:
        log_capture.close()
    return summary_root / "manifest.json"

