#!/usr/bin/env python3
"""Phi(d) smoke using eta-specific exact references."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

STAGE_ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = STAGE_ROOT.parent
REPO_ROOT = STAGE_ROOT.parents[2]
PROJECT_ROOT = REPO_ROOT.parent
SAMPLING_SRC = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = STAGE_ROOT / "raw_outputs" / "shell_pool"
DEFAULT_REFERENCE_ROOT = LABEL_ROOT / "03_reference_search" / "raw_outputs"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MNIST14_DEVICE", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(STAGE_ROOT / ".cache" / "matplotlib"))
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "TORCH_NUM_INTEROP_THREADS",
):
    os.environ.setdefault(_var, "8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(SAMPLING_SRC) not in sys.path:
    sys.path.insert(0, str(SAMPLING_SRC))

import resample_mnist10_local_support as resample  # noqa: E402


P = float(resample.P)
R0 = 0.1


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def eta_token(eta: float) -> str:
    return f"eta_{float(eta):.2f}".replace(".", "p")


def eta_from_token(rule: str) -> float:
    text = str(rule)
    if not text.startswith("eta_"):
        return float("nan")
    return float(text.replace("eta_", "").replace("p", "."))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def shard_dir(run_root: Path) -> Path:
    return ensure_dir(run_root / "shards")


def logs_dir(run_root: Path) -> Path:
    return ensure_dir(run_root / "logs")


def reference_pool_dir(run_root: Path) -> Path:
    return ensure_dir(run_root / "reference_pool")


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def resolve_existing_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    for root in (PROJECT_ROOT, REPO_ROOT):
        candidate = root / path
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


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
    if pd.isna(obj):
        return None
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


def configure_resources(cpu_threads: int) -> None:
    threads = max(1, min(8, int(cpu_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except Exception:
        pass


def configure_sampling(
    run_root: Path,
    eta_rules: list[str],
    radii: list[float],
    samples: int,
    cpu_threads: int,
    save_samples_npz: bool,
    direct_derivative: bool,
    derivative_chunk_size: int,
) -> dict[str, Any]:
    resample.RULES = list(eta_rules)
    resample.RADII = list(radii)
    cfg = resample.configure_pipe(run_root)
    cfg["experiment_id"] = "mnist10_eta_specific_reference_phi_smoke"
    cfg["identity"] = run_root.name
    cfg["sampling"] = dict(cfg["sampling"])
    cfg["sampling"]["r0"] = float(R0)
    cfg["sampling"]["radii"] = list(radii)
    cfg["sampling"]["samples_per_ref_radius"] = int(samples)
    cfg["sampling"]["fallback_policies_enabled"] = False
    cfg["sampling"]["seed_offset"] = 2026062600
    cfg["sampling"]["radial_derivative_enabled"] = bool(direct_derivative)
    cfg["sampling"]["note"] = "Eta-specific exact references from the label_noise_sweep reference-search stage."
    cfg["outputs"] = dict(cfg.get("outputs", {}))
    cfg["outputs"]["run_root"] = str(run_root)
    cfg["outputs"]["save_unit_samples_npz"] = bool(save_samples_npz)
    cfg["compute"] = dict(cfg.get("compute", {}))
    cfg["compute"]["device"] = "cpu"
    cfg["compute"]["derivative_chunk_size"] = int(derivative_chunk_size)
    cfg["compute"]["chunk_size"] = min(int(cfg["compute"].get("chunk_size", 256)), 256)
    cfg["resource_policy"] = {
        "cpu_limit_target": "use <=60% by limiting shard count and thread count",
        "gpu_limit_target": "use <=50%; this run uses CPU only unless the wrapper is changed after a GPU smoke",
        "cpu_threads_per_process": max(1, min(8, int(cpu_threads))),
        "device": "cpu",
    }
    return cfg


def load_reference_rows(reference_run_root: Path, etas: list[float], ref_count: int) -> pd.DataFrame:
    canonical_path = reference_run_root / "reference_index_canonical.csv"
    if canonical_path.exists():
        refs = pd.read_csv(canonical_path)
        rows = []
        for eta in etas:
            noise_eta = f"noise_eta_{float(eta):.2f}".replace(".", "p")
            rule = eta_token(eta)
            sub = refs[refs["noise_eta"].eq(noise_eta)].sort_values("ref").head(int(ref_count)).copy()
            if len(sub) < int(ref_count):
                raise RuntimeError(f"Reference pool short for {noise_eta}: {len(sub)} < {ref_count}")
            for record in sub.to_dict("records"):
                ref_name = str(record["ref"])
                ref_id = int(ref_name.removeprefix("ref_"))
                metadata_path = resolve_existing_project_path(str(record["reference_metadata"]))
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                theta_path = (
                    DEFAULT_REFERENCE_ROOT
                    / noise_eta
                    / ref_name
                    / "theta.npy"
                )
                rows.append(
                    {
                        "split_id": 0,
                        "rule": rule,
                        "eta": float(eta),
                        "ref_id": int(record.get("source_ref_id", ref_id - 1)),
                        "output_ref_id": ref_id,
                        "theta_path": project_relative(theta_path),
                        "dataset_path": metadata.get("dataset_payload_path") or metadata.get("source_dataset_path", ""),
                        "CE_mean_train": float(metadata["CE_mean_train"]),
                        "train_error": float(metadata.get("train_error", 0.0)),
                        "test_error": float(metadata.get("test_error", float("nan"))),
                        "theta_norm": float(metadata.get("theta_norm", float("nan"))),
                        "resample_seed_offset": 2026062600,
                    }
                )
        return pd.DataFrame(rows)

    ref_path = reference_run_root / "04_exact_reference_search" / "reference_index.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"reference index not found: {canonical_path} or {ref_path}")
    refs = pd.read_csv(ref_path)
    refs["eta"] = pd.to_numeric(refs.get("eta", refs["rule"].map(eta_from_token)), errors="coerce")
    selected_rules = [eta_token(eta) for eta in etas]
    rows = []
    for rule in selected_rules:
        sub = refs[refs["rule"].eq(rule)].sort_values("ref_id").head(int(ref_count)).copy()
        if len(sub) < int(ref_count):
            raise RuntimeError(f"Reference pool short for {rule}: {len(sub)} < {ref_count}")
        sub["resample_seed_offset"] = 2026062600
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def sample_tasks(
    run_root: Path,
    cfg: dict[str, Any],
    refs: pd.DataFrame,
    radii: list[float],
    max_units: int | None,
    force: bool,
    shard_index: int = 0,
    shard_count: int = 1,
) -> pd.DataFrame:
    tasks = []
    for row in refs.sort_values(["eta", "ref_id"]).to_dict("records"):
        for radius in radii:
            tasks.append((row, float(radius)))
    all_task_count = len(tasks)
    tasks = [
        task
        for task_idx, task in enumerate(tasks)
        if task_idx % int(shard_count) == int(shard_index)
    ]
    if max_units is not None:
        tasks = tasks[: int(max_units)]
    write_csv(
        shard_dir(run_root) / f"tasks_shard{shard_index}_of_{shard_count}.csv",
        pd.DataFrame(
            [
                {
                    "task_index": idx,
                    "all_task_count": int(all_task_count),
                    "shard_index": int(shard_index),
                    "shard_count": int(shard_count),
                    "eta": float(row["eta"]),
                    "rule": str(row["rule"]),
                    "ref_id": int(row["ref_id"]),
                    "radius": float(radius),
                    "samples_per_ref_radius": int(cfg["sampling"]["samples_per_ref_radius"]),
                }
                for idx, (row, radius) in enumerate(tasks)
            ]
        ),
    )
    rows = []
    started = time.time()
    for idx, (row, radius) in enumerate(tasks, start=1):
        print(
            f"[eta-ref-phi] shard={shard_index}/{shard_count} unit={idx}/{len(tasks)} eta={float(row['eta']):.2f} "
            f"ref={int(row['ref_id']):03d} r={float(radius):.4f}",
            flush=True,
        )
        payload = resample.sample_unit(row, float(radius), cfg, run_root, force=force)
        payload["eta"] = float(row["eta"])
        rows.append(payload)
        write_csv(
            shard_dir(run_root) / f"sampling_log_latest_shard{shard_index}_of_{shard_count}.csv",
            pd.DataFrame(rows),
        )
    out = pd.DataFrame(rows)
    out["run_elapsed_s"] = float(time.time() - started)
    write_csv(shard_dir(run_root) / f"shard{shard_index}_unit_summary.csv", out)
    return out


def load_unit_payloads(run_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_root.glob("noise_eta_*/ref_*/r_*/unit_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unit_summary_path"] = str(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize(run_root: Path, cfg: dict[str, Any], etas: list[float], radii: list[float], ref_count: int) -> dict[str, Any]:
    unit_df = load_unit_payloads(run_root)
    if unit_df.empty:
        raise RuntimeError("No unit summaries found for eta reference phi smoke.")
    for col in [
        "eta",
        "ref_id",
        "radius",
        "n_samples",
        "logZ_inf_full",
        "dlogZ_inf_full_dr",
        "split_dlogZ_dr_per_P_diff",
        "weighted_total_radial_score_sd",
        "ce_replay_max_abs_diff",
        "ess_fraction",
        "split_logZ_per_P_diff",
        "weighted_ce",
        "weighted_error",
        "elapsed_s",
    ]:
        if col in unit_df.columns:
            unit_df[col] = pd.to_numeric(unit_df[col], errors="coerce")
    if "eta" not in unit_df.columns:
        unit_df["eta"] = unit_df["rule"].map(eta_from_token)
    else:
        missing_eta = ~np.isfinite(pd.to_numeric(unit_df["eta"], errors="coerce"))
        if missing_eta.any():
            unit_df.loc[missing_eta, "eta"] = unit_df.loc[missing_eta, "rule"].map(eta_from_token)
        unit_df["eta"] = pd.to_numeric(unit_df["eta"], errors="coerce")
    requested_eta = np.asarray(etas, dtype=np.float64)
    requested_radii = np.asarray(radii, dtype=np.float64)
    eta_mask = unit_df["eta"].map(lambda value: bool(np.isclose(float(value), requested_eta).any()))
    radius_mask = unit_df["radius"].map(lambda value: bool(np.isclose(float(value), requested_radii).any()))
    unit_df = unit_df[eta_mask & radius_mask].copy()
    if unit_df.empty:
        raise RuntimeError("No unit summaries matched the requested eta/radius grid.")
    unit_df["phi_energy_raw"] = unit_df["logZ_inf_full"] / P
    key = ["eta", "ref_id"]
    r0_df = (
        unit_df[np.isclose(unit_df["radius"], R0)][key + ["logZ_inf_full"]]
        .drop_duplicates(key, keep="first")
        .rename(columns={"logZ_inf_full": "logZ_r0"})
    )
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined["logZ_inf_full"] - joined["logZ_r0"]) / P
    joined["d_phi_energy_raw_dd_unit"] = np.nan
    joined["d_delta_phi_energy_dd_unit"] = np.nan
    if "dlogZ_inf_full_dr" in joined.columns:
        joined["d_phi_energy_direct_dd_unit"] = joined["dlogZ_inf_full_dr"] / P
        joined["d_delta_phi_energy_direct_dd_unit"] = joined["d_phi_energy_direct_dd_unit"]
    for direct_col in [
        "d_phi_energy_direct_dd_unit",
        "d_delta_phi_energy_direct_dd_unit",
        "split_dlogZ_dr_per_P_diff",
        "ce_replay_max_abs_diff",
    ]:
        if direct_col not in joined.columns:
            joined[direct_col] = np.nan
    for (_eta, _ref_id), sub in joined.groupby(["eta", "ref_id"], sort=False):
        sub = sub.sort_values("radius")
        x = sub["radius"].to_numpy(dtype=np.float64)
        raw = sub["phi_energy_raw"].to_numpy(dtype=np.float64)
        delta = sub["delta_phi_energy_unit"].to_numpy(dtype=np.float64)
        if len(sub) >= 2:
            joined.loc[sub.index, "d_phi_energy_raw_dd_unit"] = np.gradient(raw, x)
            if np.isfinite(delta).sum() >= 2:
                joined.loc[sub.index, "d_delta_phi_energy_dd_unit"] = np.gradient(delta, x)
    write_csv(shard_dir(run_root) / "shell_summary_by_unit_with_phi.csv", joined)
    write_csv(shard_dir(run_root) / "shell_summary_by_unit_with_phi_derivatives.csv", joined)

    summary = (
        joined.groupby(["eta", "rule", "radius"], as_index=False)
        .agg(
            n_units=("ref_id", "nunique"),
            phi_energy_raw_mean=("phi_energy_raw", "mean"),
            phi_energy_raw_sd=("phi_energy_raw", "std"),
            phi_energy_raw_sem=("phi_energy_raw", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            delta_phi_energy_mean=("delta_phi_energy_unit", "mean"),
            delta_phi_energy_sd=("delta_phi_energy_unit", "std"),
            delta_phi_energy_sem=("delta_phi_energy_unit", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            ess_fraction_min=("ess_fraction", "min"),
            ess_fraction_mean=("ess_fraction", "mean"),
            split_logZ_per_P_diff_max=("split_logZ_per_P_diff", "max"),
            weighted_ce_mean=("weighted_ce", "mean"),
            weighted_error_mean=("weighted_error", "mean"),
            elapsed_s_mean=("elapsed_s", "mean"),
            d_phi_energy_raw_dd_unit_mean=("d_phi_energy_raw_dd_unit", "mean"),
            d_phi_energy_raw_dd_unit_sd=("d_phi_energy_raw_dd_unit", "std"),
            d_phi_energy_raw_dd_unit_sem=("d_phi_energy_raw_dd_unit", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            d_delta_phi_energy_dd_unit_mean=("d_delta_phi_energy_dd_unit", "mean"),
            d_delta_phi_energy_dd_unit_sd=("d_delta_phi_energy_dd_unit", "std"),
            d_delta_phi_energy_dd_unit_sem=("d_delta_phi_energy_dd_unit", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            d_phi_energy_direct_dd_unit_mean=("d_phi_energy_direct_dd_unit", "mean"),
            d_phi_energy_direct_dd_unit_sd=("d_phi_energy_direct_dd_unit", "std"),
            d_phi_energy_direct_dd_unit_sem=("d_phi_energy_direct_dd_unit", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            split_dlogZ_dr_per_P_diff_max=("split_dlogZ_dr_per_P_diff", "max"),
            ce_replay_max_abs_diff_max=("ce_replay_max_abs_diff", "max"),
        )
        .sort_values(["eta", "radius"])
    )
    write_csv(run_root / "06_results_figures" / "eta_reference_phi_by_eta_radius.csv", summary)

    derivative_rows = []
    for eta, sub in summary.groupby("eta"):
        sub = sub.sort_values("radius")
        x = sub["radius"].to_numpy(dtype=np.float64)
        raw = sub["phi_energy_raw_mean"].to_numpy(dtype=np.float64)
        delta = sub["delta_phi_energy_mean"].to_numpy(dtype=np.float64)
        d_raw = np.gradient(raw, x) if len(sub) >= 2 else np.full(len(sub), np.nan)
        d_delta = np.gradient(delta, x) if len(sub) >= 2 else np.full(len(sub), np.nan)
        for row, val_raw, val_delta in zip(sub.to_dict("records"), d_raw, d_delta):
            derivative_rows.append(
                {
                    **row,
                    "d_phi_energy_raw_dd": float(val_raw),
                    "d_delta_phi_energy_dd": float(val_delta),
                }
            )
    derivative = pd.DataFrame(derivative_rows)
    write_csv(run_root / "06_results_figures" / "eta_reference_dphi_dd_by_eta_radius.csv", derivative)

    status = {
        "status": "complete" if int(len(joined)) >= int(len(etas) * ref_count * len(radii)) else "partial",
        "note": "Eta-specific exact-reference phi smoke.",
        "etas": etas,
        "radii": radii,
        "ref_count": int(ref_count),
        "samples_per_ref_radius": int(cfg["sampling"]["samples_per_ref_radius"]),
        "save_unit_samples_npz": bool((cfg.get("outputs") or {}).get("save_unit_samples_npz", False)),
        "radial_derivative_enabled": bool((cfg.get("sampling") or {}).get("radial_derivative_enabled", False)),
        "completed_units": int(len(joined)),
        "expected_units": int(len(etas) * ref_count * len(radii)),
        "mean_unit_elapsed_s": float(joined["elapsed_s"].mean()),
        "max_unit_elapsed_s": float(joined["elapsed_s"].max()),
        "max_split_logZ_per_P_diff": float(joined["split_logZ_per_P_diff"].max()),
        "max_split_dlogZ_dr_per_P_diff": float(joined["split_dlogZ_dr_per_P_diff"].max()) if "split_dlogZ_dr_per_P_diff" in joined.columns else float("nan"),
        "max_ce_replay_abs_diff": float(joined["ce_replay_max_abs_diff"].max()) if "ce_replay_max_abs_diff" in joined.columns else float("nan"),
        "min_ess_fraction": float(joined["ess_fraction"].min()),
        "resource_policy": cfg["resource_policy"],
    }
    write_json(logs_dir(run_root) / "aggregate_status.json", status)
    write_json(shard_dir(run_root) / "SAMPLING_STATUS.json", status)
    write_json(run_root / "06_results_figures" / "run_config_resolved.json", {**cfg, "aggregate_status": status})
    write_report(run_root, status, summary)
    plot_summary(run_root, summary, derivative)
    return status


def write_report(run_root: Path, status: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# Eta-Specific Reference Phi Smoke",
        "",
        f"- Status: `{status['status']}`",
        f"- Units: `{status['completed_units']}` / `{status['expected_units']}`",
        f"- Samples per unit: `{status['samples_per_ref_radius']}`",
        f"- Mean unit elapsed seconds: `{status['mean_unit_elapsed_s']:.3f}`",
        f"- Max split logZ/P diff: `{status['max_split_logZ_per_P_diff']:.6g}`",
        f"- Min ESS fraction: `{status['min_ess_fraction']:.6g}`",
        "",
        "This smoke uses eta-specific exact references, unlike the earlier fixed real_even_odd anchor smoke.",
        "",
        "Primary files:",
        "",
        "- `reference_pool/reference_index.csv`",
        "- `shards/shell_summary_by_unit_with_phi.csv`",
        "- `shards/shell_summary_by_unit_with_phi_derivatives.csv`",
        "- `06_results_figures/eta_reference_phi_by_eta_radius.csv`",
        "- `06_results_figures/eta_reference_dphi_dd_by_eta_radius.csv`",
        "- `06_results_figures/fig01_eta_reference_phi_energy_d1_zoom.png`",
        "- `06_results_figures/fig02_eta_reference_delta_phi_energy_d1_zoom.png`",
        "- `06_results_figures/fig03_eta_reference_dphi_dd_d1_zoom.png`",
    ]
    if not summary.empty:
        lines.extend(["", "Summary CSV preview:", "", summary.to_csv(index=False)])
    (run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(run_root: Path, summary: pd.DataFrame, derivative: pd.DataFrame) -> None:
    fig_dir = ensure_dir(run_root / "06_results_figures")
    zoom = summary[summary["radius"] >= 0.75].copy().sort_values(["eta", "radius"])
    if zoom.empty:
        zoom = summary.copy().sort_values(["eta", "radius"])

    fig, ax = plt.subplots(figsize=(7.4, 4.7), dpi=180)
    for eta, sub in zoom.groupby("eta"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["phi_energy_raw_mean"], marker="o", lw=1.8, ms=3.3, label=f"eta={eta:.2f}")
        if sub["n_units"].max() > 1:
            ax.fill_between(
                sub["radius"].to_numpy(),
                (sub["phi_energy_raw_mean"] - 1.96 * sub["phi_energy_raw_sem"]).to_numpy(),
                (sub["phi_energy_raw_mean"] + 1.96 * sub["phi_energy_raw_sem"]).to_numpy(),
                alpha=0.13,
                linewidth=0,
            )
    ax.set_xlabel("radius d")
    ax.set_ylabel("phi(d) energy raw")
    ax.set_title("Eta-specific reference phi smoke: d near 1")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_eta_reference_phi_energy_d1_zoom.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.7), dpi=180)
    for eta, sub in zoom.groupby("eta"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["delta_phi_energy_mean"], marker="o", lw=1.8, ms=3.3, label=f"eta={eta:.2f}")
        if sub["n_units"].max() > 1:
            ax.fill_between(
                sub["radius"].to_numpy(),
                (sub["delta_phi_energy_mean"] - 1.96 * sub["delta_phi_energy_sem"]).to_numpy(),
                (sub["delta_phi_energy_mean"] + 1.96 * sub["delta_phi_energy_sem"]).to_numpy(),
                alpha=0.13,
                linewidth=0,
            )
    ax.axhline(0.0, color="0.25", lw=0.9)
    ax.set_xlabel("radius d")
    ax.set_ylabel("delta phi energy from r=0.1")
    ax.set_title("Eta-specific reference delta phi smoke: d near 1")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_eta_reference_delta_phi_energy_d1_zoom.png")
    plt.close(fig)

    dzoom = derivative[derivative["radius"] >= 0.75].copy().sort_values(["eta", "radius"])
    if dzoom.empty:
        dzoom = derivative.copy().sort_values(["eta", "radius"])
    fig, ax = plt.subplots(figsize=(7.4, 4.7), dpi=180)
    for eta, sub in dzoom.groupby("eta"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["d_phi_energy_raw_dd"], marker="o", lw=1.8, ms=3.3, label=f"eta={eta:.2f}")
    ax.axhline(0.0, color="0.25", lw=0.9)
    ax.set_xlabel("radius d")
    ax.set_ylabel("d phi energy / dd")
    ax.set_title("Eta-specific reference first derivative: d near 1")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_eta_reference_dphi_dd_d1_zoom.png")
    plt.close(fig)

    unit_d = summary[summary["radius"] >= 0.75].copy().sort_values(["eta", "radius"])
    if not unit_d.empty and "d_phi_energy_raw_dd_unit_mean" in unit_d.columns:
        fig, ax = plt.subplots(figsize=(7.4, 4.7), dpi=180)
        for eta, sub in unit_d.groupby("eta"):
            sub = sub.sort_values("radius")
            ax.plot(sub["radius"], sub["d_phi_energy_raw_dd_unit_mean"], marker="o", lw=1.8, ms=3.3, label=f"eta={eta:.2f}")
            if sub["n_units"].max() > 1:
                ax.fill_between(
                    sub["radius"].to_numpy(),
                    (sub["d_phi_energy_raw_dd_unit_mean"] - 1.96 * sub["d_phi_energy_raw_dd_unit_sem"]).to_numpy(),
                    (sub["d_phi_energy_raw_dd_unit_mean"] + 1.96 * sub["d_phi_energy_raw_dd_unit_sem"]).to_numpy(),
                    alpha=0.13,
                    linewidth=0,
                )
        ax.axhline(0.0, color="0.25", lw=0.9)
        ax.set_xlabel("radius d")
        ax.set_ylabel("mean ref-level d phi energy / dd")
        ax.set_title("Eta-specific reference-level first derivative")
        ax.grid(True, color="0.88", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig04_eta_reference_reflevel_dphi_dd_d1_zoom.png")
        plt.close(fig)

    direct_d = summary[summary["radius"] >= 0.75].copy().sort_values(["eta", "radius"])
    if direct_d.empty:
        direct_d = summary.copy().sort_values(["eta", "radius"])
    if not direct_d.empty and "d_phi_energy_direct_dd_unit_mean" in direct_d.columns and np.isfinite(direct_d["d_phi_energy_direct_dd_unit_mean"]).any():
        fig, ax = plt.subplots(figsize=(7.4, 4.7), dpi=180)
        for eta, sub in direct_d.groupby("eta"):
            sub = sub.sort_values("radius")
            ax.plot(sub["radius"], sub["d_phi_energy_direct_dd_unit_mean"], marker="o", lw=1.8, ms=3.3, label=f"eta={eta:.2f}")
            if sub["n_units"].max() > 1:
                ax.fill_between(
                    sub["radius"].to_numpy(),
                    (sub["d_phi_energy_direct_dd_unit_mean"] - 1.96 * sub["d_phi_energy_direct_dd_unit_sem"]).to_numpy(),
                    (sub["d_phi_energy_direct_dd_unit_mean"] + 1.96 * sub["d_phi_energy_direct_dd_unit_sem"]).to_numpy(),
                    alpha=0.13,
                    linewidth=0,
                )
        ax.axhline(0.0, color="0.25", lw=0.9)
        ax.set_xlabel("radius d")
        ax.set_ylabel("direct sampled d phi_E / dd")
        ax.set_title("Eta-specific direct radial derivative")
        ax.grid(True, color="0.88", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig05_eta_reference_direct_dphi_dd_d1_zoom.png")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="eta_reference_phi_smoke_cpu35_gpu0")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--reference-run-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--etas", default="0.0,0.2,0.35,0.5")
    parser.add_argument("--radii", default="0.1,0.8,0.9,1.0,1.1")
    parser.add_argument("--ref-count", type=int, default=3)
    parser.add_argument("--samples-per-ref-radius", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--derivative-chunk-size", type=int, default=64)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-final-aggregate", action="store_true")
    parser.add_argument("--save-samples-npz", dest="save_samples_npz", action="store_true", default=False)
    parser.add_argument("--no-save-samples-npz", dest="save_samples_npz", action="store_false")
    parser.add_argument("--direct-derivative", action="store_true")
    args = parser.parse_args()

    configure_resources(int(args.cpu_threads))
    etas = parse_float_list(args.etas)
    radii = parse_float_list(args.radii)
    eta_rules = [eta_token(eta) for eta in etas]
    run_root = ensure_dir(args.run_root if args.run_root is not None else STAGE_ROOT / "raw_outputs" / args.run_name)
    ensure_dir(STAGE_ROOT / ".cache" / "matplotlib")
    cfg = configure_sampling(
        run_root,
        eta_rules,
        radii,
        int(args.samples_per_ref_radius),
        int(args.cpu_threads),
        bool(args.save_samples_npz),
        bool(args.direct_derivative),
        int(args.derivative_chunk_size),
    )
    cfg["reference_search"] = dict(cfg.get("reference_search", {}))
    cfg["reference_search"]["eta_reference_run_root"] = str(args.reference_run_root)
    write_json(run_root / "run_config.json", cfg)

    refs = load_reference_rows(args.reference_run_root, etas, int(args.ref_count))
    write_csv(reference_pool_dir(run_root) / "reference_index.csv", refs)
    if not args.aggregate_only:
        shard_df = sample_tasks(
            run_root,
            cfg,
            refs,
            radii,
            args.max_units,
            bool(args.force),
            int(args.shard_index),
            int(args.shard_count),
        )
        if args.no_final_aggregate:
            status = {
                "status": "sampling_shard_complete",
                "run_name": args.run_name,
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
                "shard_units_this_invocation": int(len(shard_df)),
                "etas": etas,
                "radii": radii,
                "ref_count": int(args.ref_count),
                "samples_per_ref_radius": int(args.samples_per_ref_radius),
                "save_unit_samples_npz": bool(args.save_samples_npz),
                "resource_policy": cfg["resource_policy"],
            }
            write_json(
                shard_dir(run_root) / f"shard{args.shard_index}_status.json",
                status,
            )
            print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
            return 0

    status = summarize(run_root, cfg, etas, radii, int(args.ref_count))
    print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
