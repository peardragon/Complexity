from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from utils.dataset_builder import make_ws_ising_dataset
from utils.defaults import DEFAULT_CONFIG, make_cell_id
from utils.io_utils import ensure_dir, load_json, save_json
from utils.layout import RAW_ROOT, STAGE_ROOT
from utils.visuals import plot_dataset_view


CONFIG_PATH = STAGE_ROOT / "config" / "default.json"


def _dataset_label(dataset_id: int) -> str:
    return f"dataset_{int(dataset_id):03d}"


def _source_sorted_index(beta_ising: float, beta_values: list[float]) -> int:
    min_beta = min(beta_values)
    return int(round((float(beta_ising) - float(min_beta)) / 0.01))


def _dataset_seed(beta_ising: float, dataset_id: int, beta_values: list[float]) -> int:
    source_index = _source_sorted_index(float(beta_ising), beta_values)
    return int(1000 * source_index + 100000 * int(dataset_id) + 1234)


def _existing_meta(dataset_dir: Path) -> dict[str, Any]:
    meta = load_json(dataset_dir / "dataset_meta.json", default={})
    return meta if isinstance(meta, dict) else {}


def _resolve_generation_seed(dataset_dir: Path, beta_ising: float, dataset_id: int, beta_values: list[float]) -> int:
    meta = _existing_meta(dataset_dir)
    if isinstance(meta.get("seed"), int):
        return int(meta["seed"])
    nested_meta = meta.get("meta", {})
    if isinstance(nested_meta, dict) and isinstance(nested_meta.get("seed"), int):
        return int(nested_meta["seed"])
    return _dataset_seed(beta_ising, dataset_id, beta_values)


def _load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH, default=None)
    if not isinstance(config, dict):
        raise FileNotFoundError(CONFIG_PATH)
    return config


def build_datasets(*, overwrite: bool = False) -> dict[str, int]:
    config = _load_config()
    beta_values = [float(beta) for beta in config["beta_values"]]
    datasets_per_beta = int(config["datasets_per_beta"])
    n_points = int(config.get("n_points_per_dataset", DEFAULT_CONFIG["n_points"]))
    input_dim = int(config.get("input_dim", DEFAULT_CONFIG.get("input_dim", 2)))
    k_graph = int(config.get("k_graph", DEFAULT_CONFIG["k_graph"]))
    rewire_p = float(config.get("rewire_p", 0.0))
    rewire_mode = str(config.get("rewire_mode", DEFAULT_CONFIG["rewire_mode"]))
    ising_sweeps = int(config.get("ising_sweeps", DEFAULT_CONFIG["ising_sweeps"]))
    scales = config.get("nmstv_scales", DEFAULT_CONFIG["nmstv_scales"])

    ensure_dir(RAW_ROOT)
    written = 0
    skipped = 0
    for beta_ising in beta_values:
        beta_dir = RAW_ROOT / make_cell_id(beta_ising, rewire_p)
        ensure_dir(beta_dir)
        for dataset_id in range(datasets_per_beta):
            dataset_dir = beta_dir / _dataset_label(dataset_id)
            dataset_npz = dataset_dir / "dataset.npz"
            if dataset_npz.exists() and not overwrite:
                skipped += 1
                continue

            existing_meta = _existing_meta(dataset_dir)
            seed = _resolve_generation_seed(dataset_dir, beta_ising, dataset_id, beta_values)
            dataset = make_ws_ising_dataset(
                n_points=n_points,
                input_dim=input_dim,
                k_graph=k_graph,
                rewire_p=rewire_p,
                rewire_mode=rewire_mode,
                beta_ising=beta_ising,
                ising_sweeps=ising_sweeps,
                seed=seed,
                scales=scales,
            )

            ensure_dir(dataset_dir)
            np.savez_compressed(
                dataset_npz,
                X_raw=dataset["X_raw"],
                X_train=dataset["X_train"],
                y=dataset["y"],
            )
            visualization = plot_dataset_view(
                dataset["X_raw"],
                dataset["X_train"],
                dataset["y"],
                dataset_dir,
                title=f"beta={beta_ising:.2f}, p={rewire_p:.2f} / dataset {dataset_id:03d}",
            )
            meta = {
                "beta_ising": beta_ising,
                "beta_dir": beta_dir.name,
                "cell_id": beta_dir.name,
                "dataset_id": dataset_id,
                "meta": {**dataset["meta"], "visualization": visualization},
                "rewire_p": rewire_p,
                "seed": seed,
                "series": "beta",
            }
            if isinstance(existing_meta.get("replacement_for"), str):
                meta["replacement_for"] = existing_meta["replacement_for"]
            save_json(dataset_dir / "dataset_meta.json", meta)
            written += 1

    return {"written": written, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic Ising-label datasets.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing dataset folders.")
    args = parser.parse_args()
    counts = build_datasets(overwrite=bool(args.overwrite))
    print(f"datasets written={counts['written']} skipped={counts['skipped']}")


if __name__ == "__main__":
    main()
