from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from dataset_builder import make_ws_ising_dataset
from defaults import DEFAULT_CONFIG, VALID_DIMS, build_cell_specs, dataset_seed
from io_utils import ensure_dir, load_json, now_iso, save_csv, save_json, start_verbose_print_capture
from manifest import build_roots, write_manifest
from visuals import plot_dataset_series_grid, plot_dataset_view


def merged_config(config_path: Path, *, dim: int, force: bool, seed: int | None) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        config.update(load_json(config_path, {}) or {})
    config["input_dim"] = int(dim)
    if seed is not None:
        config["seed"] = int(seed)
    config["force"] = bool(force)
    if int(config["input_dim"]) not in VALID_DIMS:
        raise ValueError(f"Unsupported dim: {config['input_dim']}")
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


def run_pipeline(*, part_root: Path, config_path: Path, dim: int, force: bool, seed: int | None, verbose: bool = False) -> Path:
    started_at = now_iso()
    config = merged_config(config_path, dim=dim, force=force, seed=seed)
    run_group = "smoke_runs" if config_path.name == "smoke.json" else "runs"
    config["force"] = bool(config["force"]) or run_group == "smoke_runs"
    cfg_hash, raw_root, summary_root = build_roots(part_root, config, run_group=run_group)
    existing_manifest = _existing_success_manifest(summary_root, force=bool(config["force"]))
    if existing_manifest is not None:
        return existing_manifest
    log_capture = start_verbose_print_capture(summary_root, enabled=verbose)
    log_path = str(log_capture.log_path) if log_capture is not None else None
    raw_beta_series = config.get("beta_series", None)
    raw_p_series = config.get("p_series", None)
    cell_specs = build_cell_specs(
        beta_series=None if raw_beta_series is None else [float(x) for x in raw_beta_series],
        p_series=None if raw_p_series is None else [float(x) for x in raw_p_series],
    )
    canonical_cell_index: dict[str, int] = {}
    unique_cells = 0
    for cell_index, cell in enumerate(cell_specs):
        if cell.cell_id not in canonical_cell_index:
            canonical_cell_index[cell.cell_id] = cell_index
            unique_cells += 1
    total_datasets = unique_cells * int(config["datasets_per_cell"])
    _log(verbose, f"[dataset_gen] D={int(config['input_dim'])}, cells={len(cell_specs)}, unique_cells={unique_cells}, datasets_per_cell={int(config['datasets_per_cell'])}, total={total_datasets}")
    index_rows = []
    seen_index_keys: set[tuple[str, int, int]] = set()
    generated_meta_by_npz: dict[str, dict[str, Any]] = {}
    completed = 0
    for cell_index, cell in enumerate(cell_specs):
        for dataset_id in range(int(config["datasets_per_cell"])):
            canonical_index = canonical_cell_index[cell.cell_id] if bool(config.get("reuse_duplicate_cell_datasets", True)) else cell_index
            is_duplicate_cell = canonical_index != cell_index
            if not is_duplicate_cell:
                completed += 1
            ds_seed = dataset_seed(int(config["seed"]), canonical_index, dataset_id)
            ds_dir = raw_root / cell.cell_id / f"dataset_{dataset_id:03d}_seed_{ds_seed:06d}"
            ensure_dir(ds_dir)
            npz_path = ds_dir / "dataset.npz"
            meta_path = ds_dir / "dataset_meta.json"
            cache_key = str(npz_path.resolve())
            if cache_key in generated_meta_by_npz:
                reuse_reason = "duplicate-cell reuse" if is_duplicate_cell else "shared reuse"
                _log(verbose, f"[dataset_gen] [{completed}/{total_datasets}] {cell.cell_id} dataset_{dataset_id:03d}: {reuse_reason}")
                meta = generated_meta_by_npz[cache_key]
            elif npz_path.exists() and meta_path.exists() and not bool(config["force"]):
                reuse_reason = "duplicate-cell reuse" if is_duplicate_cell else "reuse"
                _log(verbose, f"[dataset_gen] [{completed}/{total_datasets}] {cell.cell_id} dataset_{dataset_id:03d}: {reuse_reason}")
                meta = load_json(meta_path, {})
                generated_meta_by_npz[cache_key] = meta
            else:
                _log(verbose, f"[dataset_gen] [{completed}/{total_datasets}] {cell.cell_id} dataset_{dataset_id:03d}: generate")
                data = make_ws_ising_dataset(
                    n_points=int(config["n_points"]),
                    input_dim=int(config["input_dim"]),
                    k_graph=int(config["k_graph"]),
                    rewire_p=float(cell.rewire_p),
                    rewire_mode=str(config["rewire_mode"]),
                    beta_ising=float(cell.beta_ising),
                    ising_sweeps=int(config["ising_sweeps"]),
                    seed=int(ds_seed),
                    scales=list(config["nmstv_scales"]),
                )
                np.savez_compressed(npz_path, X_raw=data["X_raw"], X_train=data["X_train"], y=data["y"])
                view_meta = plot_dataset_view(
                    data["X_raw"],
                    data["X_train"],
                    data["y"],
                    ds_dir,
                    title=f"{cell.display_label} / dataset {dataset_id:03d}",
                )
                meta = {
                    "cell_id": cell.cell_id,
                    "series": cell.series,
                    "dataset_id": int(dataset_id),
                    "seed": int(ds_seed),
                    "beta_ising": float(cell.beta_ising),
                    "rewire_p": float(cell.rewire_p),
                    "meta": {
                        **data["meta"],
                        "visualization": view_meta,
                    },
                }
                save_json(meta_path, meta)
                generated_meta_by_npz[cache_key] = meta
            index_key = (cell.cell_id, int(dataset_id), int(ds_seed))
            if index_key in seen_index_keys:
                continue
            seen_index_keys.add(index_key)
            index_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "series": cell.series,
                    "dataset_id": int(dataset_id),
                    "seed": int(ds_seed),
                    "beta_ising": float(cell.beta_ising),
                    "rewire_p": float(cell.rewire_p),
                    "dataset_raw_path": str(npz_path.relative_to(part_root)),
                    "dataset_meta_path": str(meta_path.relative_to(part_root)),
                }
            )
    dataset_index_path = summary_root / "dataset_index.csv"
    save_csv(
        dataset_index_path,
        index_rows,
        ["cell_id", "series", "dataset_id", "seed", "beta_ising", "rewire_p", "dataset_raw_path", "dataset_meta_path"],
    )
    beta_grid = summary_root / "figs" / "beta_series_grid.png"
    beta_grid_scatter = summary_root / "figs" / "beta_series_grid_scatter.png"
    p_grid = summary_root / "figs" / "p_series_grid.png"
    p_grid_scatter = summary_root / "figs" / "p_series_grid_scatter.png"
    plot_dataset_series_grid(index_rows, part_root, beta_grid, dimension=int(config["input_dim"]), series_name="beta", title=f"beta series / D={int(config['input_dim'])} representative datasets", panel_mode="filled")
    plot_dataset_series_grid(index_rows, part_root, beta_grid_scatter, dimension=int(config["input_dim"]), series_name="beta", title=f"beta series / D={int(config['input_dim'])} representative datasets / scatter", panel_mode="scatter")
    plot_dataset_series_grid(index_rows, part_root, p_grid, dimension=int(config["input_dim"]), series_name="p", title=f"p series / D={int(config['input_dim'])} representative datasets", panel_mode="filled")
    plot_dataset_series_grid(index_rows, part_root, p_grid_scatter, dimension=int(config["input_dim"]), series_name="p", title=f"p series / D={int(config['input_dim'])} representative datasets / scatter", panel_mode="scatter")
    save_json(summary_root / "run_config.json", config)
    summary_outputs = [str(dataset_index_path)]
    for path in (beta_grid, beta_grid_scatter, p_grid, p_grid_scatter):
        if path.exists():
            summary_outputs.append(str(path))
    if log_path is not None:
        summary_outputs.append(log_path)
    write_manifest(
        summary_root,
        pipeline_id=str(config["pipeline_id"]),
        methodology_id=str(config["methodology_id"]),
        config_hash_value=cfg_hash,
        config_path=str(config_path),
        raw_output_root=raw_root,
        upstream_refs=[],
        summary_outputs=summary_outputs,
        dimension=int(config["input_dim"]),
        status="success",
        run_group=run_group,
        started_at=started_at,
    )
    _log(verbose, f"[dataset_gen] wrote dataset_index -> {dataset_index_path}")
    _log(verbose, f"[dataset_gen] wrote manifest -> {summary_root / 'manifest.json'}")
    if log_capture is not None:
        log_capture.close()
    return summary_root / "manifest.json"


