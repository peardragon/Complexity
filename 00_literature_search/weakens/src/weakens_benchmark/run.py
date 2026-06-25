from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_dir, stage_dir, write_json
from .figures import make_ab_clarified_figure, make_final_figure, make_schematic_figure
from .importance import run_vmf_l2_importance
from .landscape import ProxyLandscape
from .qc import build_qc_tables
from .samplers import run_all_baselines


def _records_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(config["experiment_id"])
    rng = np.random.default_rng(int(config["seed"]))
    landscape = ProxyLandscape(config)

    dataset_dir = stage_dir("dataset", experiment_id)
    region_dir = stage_dir("regions", experiment_id)
    baseline_dir = stage_dir("baselines", experiment_id)
    vmf_dir = stage_dir("vmf", experiment_id)
    qc_dir = stage_dir("qc", experiment_id)
    figure_dir = stage_dir("figures", experiment_id)

    write_json(dataset_dir / "manifest.json", _manifest(config, landscape))
    shutil.copyfile(config["_config_path"], dataset_dir / "config.json")

    grid = landscape.grid_reference(int(config["domain"]["grid_n"]))
    np.savez_compressed(
        dataset_dir / "grid_reference.npz",
        x=grid["x"],
        y=grid["y"],
        energy=grid["energy"],
        density=grid["density"],
        region_mass=grid["region_mass"],
    )
    region_rows = landscape.region_reference_frame()
    for row, mass in zip(region_rows, grid["region_mass"]):
        row["truth_mass"] = float(mass)
    _records_to_csv(region_dir / "region_reference.csv", region_rows)

    baseline_results = run_all_baselines(landscape, config, rng)
    baseline_meta = {}
    sample_payload = {}
    for method, result in baseline_results.items():
        baseline_meta[method] = result.metadata | {
            "accept_rate": result.accept_rate,
            "retained_sample_count": int(result.samples.shape[0]),
        }
        sample_payload[f"{method}_samples"] = result.samples
        sample_payload[f"{method}_energies"] = result.energies
    np.savez_compressed(baseline_dir / "baseline_samples.npz", **sample_payload)
    write_json(baseline_dir / "baseline_reproduction_metadata.json", baseline_meta)

    importance_result = run_vmf_l2_importance(landscape, config, rng)
    np.savez_compressed(
        vmf_dir / "vmf_l2_samples.npz",
        samples=importance_result.samples,
        energies=importance_result.energies,
        log_weights=importance_result.log_weights,
    )
    write_json(vmf_dir / "importance_summary.json", importance_result.metadata)

    region_mass_df, qc_df = build_qc_tables(
        landscape,
        baseline_results,
        importance_result,
        grid["region_mass"],
        config,
    )
    region_mass_df.to_csv(qc_dir / "region_mass_estimates.csv", index=False)
    qc_df.to_csv(qc_dir / "qc_summary.csv", index=False)
    write_json(
        qc_dir / "qc_report.json",
        {
            "experiment_id": experiment_id,
            "qc_thresholds": config["qc"],
            "summary": qc_df.to_dict(orient="records"),
        },
    )

    figure_path = figure_dir / "final_sampling_failure_vmf_recovery.png"
    make_final_figure(
        landscape,
        grid,
        baseline_results,
        importance_result,
        region_mass_df,
        qc_df,
        figure_path,
    )
    schematic_path = figure_dir / "schematic_problem_statement_vmf_l2.png"
    make_schematic_figure(
        landscape,
        baseline_results,
        importance_result,
        qc_df,
        schematic_path,
    )
    ab_path = figure_dir / "final_sampling_failure_vmf_recovery_AB.png"
    make_ab_clarified_figure(
        landscape,
        grid,
        baseline_results,
        importance_result,
        qc_df,
        ab_path,
    )
    return {
        "experiment_id": experiment_id,
        "figure_path": str(figure_path),
        "schematic_figure_path": str(schematic_path),
        "ab_figure_path": str(ab_path),
        "qc_summary_path": str(qc_dir / "qc_summary.csv"),
        "region_mass_path": str(qc_dir / "region_mass_estimates.csv"),
        "baseline_metadata_path": str(baseline_dir / "baseline_reproduction_metadata.json"),
        "importance_summary_path": str(vmf_dir / "importance_summary.json"),
        "qc": qc_df.to_dict(orient="records"),
    }


def _manifest(config: dict[str, Any], landscape: ProxyLandscape) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "seed": config["seed"],
        "beta": config["beta"],
        "domain": config["domain"],
        "rough_term_count": landscape.rough_count,
        "l2_scale": landscape.l2_scale,
        "target_energy": "E_proxy(z) + l2_scale * ||z||^2",
        "method_roles": {
            "random_walk_mcmc": "existing local MCMC control",
            "hmc": "existing method reproduction proxy for arXiv:2503.08266",
            "pseudo_langevin": "existing method proxy reproduction for arXiv:2603.15367",
            "vmf_l2_final": "final vMF+L2 methodology in proxy coordinates",
        },
    }
