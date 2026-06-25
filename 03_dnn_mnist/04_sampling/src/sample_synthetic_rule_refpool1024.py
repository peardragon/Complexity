#!/usr/bin/env python3
"""Mechanical 1024-sample PM-SAIS run for one synthetic MNIST10 rule."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_RUN_ROOT = LOCAL_ROOT / "03_reference_search/raw_outputs/very_low_tv_spectral_teacher_v1"
REFERENCE_INDEX = REFERENCE_RUN_ROOT / "04_exact_reference_search/reference_index.csv"
DEFAULT_RUN_ROOT = LOCAL_ROOT / "04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_90ref"
RULE = "very_low_tv_spectral_teacher"
RADII = [round(idx / 10.0, 1) for idx in range(1, 26)]
R0 = 0.1
P = 2461.0
SPLIT_GATE = 0.004
ESS_GATE = 0.04
FINITE_FRACTION_GATE = 0.95
BOOTSTRAP_GATE = 0.012


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import resample_mnist10_local_support as resample  # noqa: E402


resample.RULES = [RULE]
resample.RADII = list(RADII)


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


def configure_resources(cpu_threads: int, device: str) -> None:
    threads = max(1, int(cpu_threads))
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MNIST14_DEVICE"] = "cpu"
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
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except Exception:
        pass


def configure_sampler(args: argparse.Namespace, run_root: Path, cpu_threads: int) -> dict[str, Any]:
    cfg = resample.configure_pipe(run_root)
    cfg["experiment_id"] = "mnist10_very_low_tv_refpool1024_mechanical_all_radii"
    cfg["identity"] = run_root.name
    cfg["dataset"] = dict(cfg["dataset"])
    cfg["dataset"]["label_rules"] = [RULE]
    cfg["sampling"] = dict(cfg["sampling"])
    cfg["sampling"]["r0"] = float(R0)
    cfg["sampling"]["radii"] = list(RADII)
    cfg["sampling"]["radius_grid_kind"] = "mechanical_0p1_to_2p5_step0p1"
    cfg["sampling"]["samples_per_ref_radius"] = int(args.samples_per_ref_radius)
    cfg["sampling"]["fallback_policies_enabled"] = False
    cfg["sampling"]["seed_offset"] = int(args.seed_offset)
    cfg["sampling"]["task_policy"] = "single_synthetic_rule_all_ref_radius_no_qc_gate"
    cfg["reference_search"] = dict(cfg["reference_search"])
    cfg["reference_search"]["target_pool_refs_per_rule"] = int(args.target_refs)
    cfg["reference_search"]["reference_index"] = str(args.reference_index)
    cfg["compute"] = dict(cfg.get("compute", {}))
    cfg["compute"]["chunk_size"] = int(args.chunk_size)
    cfg["compute"]["device"] = str(args.device)
    cfg["outputs"] = dict(cfg["outputs"])
    cfg["outputs"]["run_root"] = str(run_root)
    cfg["outputs"]["source_reference_run_root"] = str(Path(args.reference_index).parent.parent)
    cfg["outputs"]["save_unit_samples_npz"] = bool(args.save_samples_npz)
    cfg["qc"] = dict(cfg.get("qc", {}))
    cfg["qc"]["finite_unit_fraction_min"] = FINITE_FRACTION_GATE
    cfg["qc"]["q05_ess_fraction_min"] = ESS_GATE
    cfg["qc"]["max_split_logZ_per_P_diff"] = SPLIT_GATE
    cfg["qc"]["bootstrap_sd_phi_max"] = BOOTSTRAP_GATE
    cfg["resource_policy"] = {
        "cpu_limit_target": "caller should keep total shard_threads <= 70% machine CPUs",
        "gpu_limit_target": "default device=cpu keeps GPU at 0%",
        "cpu_threads_per_process": int(cpu_threads),
        "device": str(args.device),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
    }
    cfg["resolved_at_unix"] = time.time()
    return cfg


def load_pool(reference_index: Path, target_refs: int) -> pd.DataFrame:
    refs = pd.read_csv(reference_index)
    refs = refs[refs["rule"].eq(RULE)].copy()
    refs["ref_id"] = refs["ref_id"].astype(int)
    refs = refs.sort_values("ref_id").drop_duplicates(["rule", "ref_id"], keep="first").head(int(target_refs)).reset_index(drop=True)
    if len(refs) < int(target_refs):
        raise RuntimeError(f"Reference pool short for {RULE}: {len(refs)} < {int(target_refs)}")
    refs["pool_rank"] = np.arange(1, len(refs) + 1)
    return refs


def bootstrap_sd(values: np.ndarray, seed: int, n_boot: int = 300) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for idx in range(int(n_boot)):
        means[idx] = np.mean(rng.choice(values, size=values.size, replace=True))
    return float(np.std(means, ddof=1))


def load_unit_payloads(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = run_root / "05_pool2_pm_sais_sampling/unit_summaries"
    for path in sorted(root.rglob("unit_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unit_summary_path"] = str(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize(run_root: Path, cfg: dict[str, Any], pool_df: pd.DataFrame) -> dict[str, Any]:
    stage05 = ensure_dir(run_root / "05_pool2_pm_sais_sampling")
    out_dir = ensure_dir(run_root / "06_results_figures")
    unit = load_unit_payloads(run_root)
    if unit.empty:
        raise RuntimeError("No unit summaries found")
    for col in ["split_id", "ref_id", "radius", "n_samples", "logZ", "logZ_inf_full", "ess_fraction", "split_logZ_per_P_diff", "weighted_ce", "weighted_error"]:
        if col in unit.columns:
            unit[col] = pd.to_numeric(unit[col], errors="coerce")
    unit["rule"] = unit["rule"].astype(str)
    unit["phi_energy_raw"] = unit["logZ_inf_full"] / float(P)
    write_csv(stage05 / "shell_summary_by_unit.csv", unit)
    write_csv(stage05 / "shell_summary_by_unit_with_phi.csv", unit)

    target_refs = int(cfg["reference_search"]["target_pool_refs_per_rule"])
    expected_refs = set(pool_df["ref_id"].astype(int).tolist())
    rows: list[dict[str, Any]] = []
    phi_rows: list[dict[str, Any]] = []
    for radius in RADII:
        sub = unit[unit["rule"].eq(RULE) & np.isclose(unit["radius"], radius)].copy()
        observed_refs = set(sub["ref_id"].dropna().astype(int).tolist()) if len(sub) else set()
        missing_refs = sorted(expected_refs - observed_refs)
        finite_values = sub["phi_energy_raw"].to_numpy(dtype=np.float64) if len(sub) else np.asarray([], dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        finite_fraction = float(len(finite_values) / target_refs) if target_refs else 0.0
        q05_ess = float(np.quantile(sub["ess_fraction"].dropna(), 0.05)) if len(sub) else float("nan")
        max_split = float(sub["split_logZ_per_P_diff"].max()) if len(sub) else float("nan")
        boot_sd = bootstrap_sd(finite_values, 1204000 + int(round(float(radius) * 1000))) if len(finite_values) else float("nan")
        complete = len(observed_refs) == target_refs
        qc_pass = bool(
            complete
            and finite_fraction >= FINITE_FRACTION_GATE
            and np.isfinite(q05_ess)
            and q05_ess >= ESS_GATE
            and np.isfinite(max_split)
            and max_split <= SPLIT_GATE
            and np.isfinite(boot_sd)
            and boot_sd <= BOOTSTRAP_GATE
        )
        mean_raw = float(np.mean(finite_values)) if len(finite_values) else float("nan")
        sd_raw = float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0
        row = {
            "rule": RULE,
            "radius": float(radius),
            "target_ref_count": int(target_refs),
            "observed_ref_count": int(len(observed_refs)),
            "missing_ref_count": int(len(missing_refs)),
            "finite_unit_count": int(len(finite_values)),
            "finite_unit_fraction": finite_fraction,
            "q05_ess_fraction": q05_ess,
            "max_split_logZ_per_P_diff": max_split,
            "bootstrap_sd_phi_energy_raw": boot_sd,
            "qc_diagnostic_pass": qc_pass,
            "sampling_status": "complete" if complete else "partial_missing_units",
            "missing_ref_ids": ",".join(str(x) for x in missing_refs[:40]),
            "phi_energy_raw": mean_raw,
            "phi_energy_raw_sd": sd_raw,
            "phi_energy_raw_sem": float(sd_raw / math.sqrt(len(finite_values))) if len(finite_values) > 1 else 0.0,
            "weighted_ce_mean": float(sub["weighted_ce"].mean()) if len(sub) else float("nan"),
            "weighted_error_mean": float(sub["weighted_error"].mean()) if len(sub) else float("nan"),
        }
        rows.append(row)
        phi_rows.append(
            {
                "rule": RULE,
                "radius": float(radius),
                "phi_energy_raw": mean_raw,
                "phi_energy_raw_sd": sd_raw,
                "phi_energy_raw_sem": row["phi_energy_raw_sem"],
                "n_units": int(len(sub)),
                "target_ref_count": int(target_refs),
                "sampling_status": row["sampling_status"],
                "qc_diagnostic_pass": qc_pass,
            }
        )
    summary = pd.DataFrame(rows)
    phi = pd.DataFrame(phi_rows)
    write_csv(stage05 / "shell_summary_by_rule_radius.csv", summary)
    write_csv(stage05 / "qc_diagnostics_by_rule_radius.csv", summary)
    write_csv(out_dir / "phi_energy_by_rule_radius.csv", phi)
    write_csv(out_dir / "phi_raw_by_rule_radius.csv", phi[["rule", "radius", "phi_energy_raw", "n_units", "target_ref_count", "sampling_status"]])

    completed_units = int(len(unit.drop_duplicates(["rule", "ref_id", "radius"])))
    expected_units = int(target_refs * len(RADII))
    status = {
        "status": "complete" if completed_units >= expected_units and int((summary["sampling_status"] == "complete").sum()) == len(RADII) else "partial",
        "rule": RULE,
        "radii": RADII,
        "samples_per_ref_radius": int(cfg["sampling"]["samples_per_ref_radius"]),
        "target_refs_per_rule": int(target_refs),
        "completed_units": completed_units,
        "expected_units": expected_units,
        "complete_rule_radius_rows": int((summary["sampling_status"] == "complete").sum()),
        "total_rule_radius_rows": int(len(RADII)),
        "qc_diagnostic_pass_rows": int(summary["qc_diagnostic_pass"].sum()),
        "save_unit_samples_npz": bool(cfg["outputs"].get("save_unit_samples_npz", False)),
    }
    write_json(run_root / "SAMPLING_STATUS.json", status)
    write_json(stage05 / "SAMPLING_STATUS.json", status)
    write_json(out_dir / "run_config_resolved.json", {**cfg, "aggregate_status": status})
    (run_root / "REPORT.md").write_text(
        f"# Very Low-TV Refpool1024 Sampling\n\n- Status: `{status['status']}`\n- Units: `{completed_units}` / `{expected_units}`\n- Samples per unit: `{int(cfg['sampling']['samples_per_ref_radius'])}`\n- Raw phi table: `06_results_figures/phi_energy_by_rule_radius.csv`\n",
        encoding="utf-8",
    )
    return status


def tasks(pool_df: pd.DataFrame) -> list[tuple[dict[str, Any], float]]:
    out = []
    for row in pool_df.to_dict("records"):
        item = dict(row)
        item["rule"] = RULE
        item["ref_id"] = int(item["ref_id"])
        item["split_id"] = int(item.get("split_id", 0))
        for radius in RADII:
            out.append((item, float(radius)))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    cpu_threads = int(args.cpu_threads)
    configure_resources(cpu_threads, str(args.device))
    cfg = configure_sampler(args, run_root, cpu_threads)
    pool = load_pool(Path(args.reference_index), int(args.target_refs))
    write_csv(run_root / "04_reference_pool/reference_pool_index.csv", pool)
    write_json(run_root / "04_reference_pool/POOL_STATUS.json", {"rule": RULE, "target_refs_per_rule": int(args.target_refs), "reference_index": str(args.reference_index)})
    write_json(run_root / "run_config_resolved.json", cfg)

    all_tasks = tasks(pool)
    shard_tasks = [task for idx, task in enumerate(all_tasks) if idx % int(args.shard_count) == int(args.shard_index)]
    if args.max_units is not None:
        shard_tasks = shard_tasks[: int(args.max_units)]
    write_csv(
        run_root / "05_pool2_pm_sais_sampling" / f"tasks_shard{args.shard_index}_of_{args.shard_count}.csv",
        pd.DataFrame(
            [
                {"task_index": idx, "rule": row["rule"], "ref_id": int(row["ref_id"]), "pool_rank": int(row.get("pool_rank", -1)), "radius": float(radius)}
                for idx, (row, radius) in enumerate(shard_tasks)
            ]
        ),
    )
    rows = []
    started = time.time()
    for idx, (row, radius) in enumerate(shard_tasks, start=1):
        print(f"[synthetic sampling] shard={args.shard_index}/{args.shard_count} unit={idx}/{len(shard_tasks)} ref={int(row['ref_id']):03d} r={radius:.1f}", flush=True)
        payload = resample.sample_unit(row, float(radius), cfg, run_root, force=bool(args.force))
        rows.append(
            {
                "rule": RULE,
                "ref_id": int(row["ref_id"]),
                "pool_rank": int(row.get("pool_rank", -1)),
                "radius": float(radius),
                "reused": bool(payload.get("reused", False)),
                "finite": bool(payload.get("finite", False)),
                "logZ_inf_full": float(payload.get("logZ_inf_full", float("nan"))),
                "ess_fraction": float(payload.get("ess_fraction", float("nan"))),
                "split_logZ_per_P_diff": float(payload.get("split_logZ_per_P_diff", float("nan"))),
                "elapsed_s": float(payload.get("elapsed_s", float("nan"))),
            }
        )
        if args.aggregate_every and idx % int(args.aggregate_every) == 0:
            summarize(run_root, cfg, pool)
    write_csv(run_root / "05_pool2_pm_sais_sampling" / f"shard{args.shard_index}_unit_summary.csv", pd.DataFrame(rows))
    if args.no_final_aggregate:
        status = {
            "status": "sampling_shard_complete",
            "shard_units_this_invocation": int(len(rows)),
            "elapsed_s": float(time.time() - started),
        }
    else:
        status = summarize(run_root, cfg, pool)
        status["shard_units_this_invocation"] = int(len(rows))
        status["elapsed_s"] = float(time.time() - started)
    write_json(run_root / "05_pool2_pm_sais_sampling" / f"shard{args.shard_index}_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic-rule 1024-particle sampling.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--reference-index", default=str(REFERENCE_INDEX))
    parser.add_argument("--target-refs", type=int, default=90)
    parser.add_argument("--samples-per-ref-radius", type=int, default=1024)
    parser.add_argument("--seed-offset", type=int, default=2026061900)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--aggregate-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-final-aggregate", action="store_true")
    parser.add_argument("--save-samples-npz", dest="save_samples_npz", action="store_true", default=True)
    parser.add_argument("--no-save-samples-npz", dest="save_samples_npz", action="store_false")
    args = parser.parse_args(argv)

    if args.aggregate_only:
        configure_resources(int(args.cpu_threads), str(args.device))
        cfg = configure_sampler(args, Path(args.run_root), int(args.cpu_threads))
        pool = load_pool(Path(args.reference_index), int(args.target_refs))
        status = summarize(Path(args.run_root), cfg, pool)
        print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
        return 0 if status["status"] == "complete" else 2
    status = run(args)
    return 0 if status["status"] in {"complete", "sampling_shard_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
