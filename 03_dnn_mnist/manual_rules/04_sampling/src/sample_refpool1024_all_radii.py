#!/usr/bin/env python3
"""Mechanical MNIST PM-SAIS sampling over the full reference pool.

Defaults:
- 60 references per rule from the existing exact reference pool.
- radii 0.1, 0.2, ..., 2.5.
- 1024 particles per rule/reference/radius unit.
- no QC-gated task selection and no fallback escalation.
- unit_summary.json plus samples.npz, matching the compact 02_dnn layout.
"""

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


LOCAL_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_ROOT = Path("/home/bjyong/Complexity/windows_project")
MNIST_ROOT = WINDOWS_ROOT / "02_dnn" / "08_mnist"
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_RUN_ROOT = (
    MNIST_ROOT
    / "runs"
    / "final"
    / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
)
REFERENCE_INDEX = REFERENCE_RUN_ROOT / "04_exact_reference_search" / "reference_index.csv"
VERY_LOW_REFERENCE_RUN_ROOT = LOCAL_ROOT / "03_reference_search" / "raw_outputs" / "very_low_tv_spectral_teacher_v1"
VERY_LOW_REFERENCE_INDEX = VERY_LOW_REFERENCE_RUN_ROOT / "04_exact_reference_search" / "reference_index.csv"
DEFAULT_EXTRA_REFERENCE_RUN_ROOT = LOCAL_ROOT / "03_reference_search" / "raw_outputs" / "extra_reference_pool"
EXTRA_REFERENCE_INDEX_NAME = "extra_reference_index.csv"
DEFAULT_RUN_ROOT = LOCAL_ROOT / "04_sampling" / "raw_outputs" / "shell_pool"

RULES = [
    "very_low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
RULE_IDS = {
    "very_low_tv_spectral_teacher": "rule_001",
    "real_even_odd": "rule_002",
    "teacher_nn": "rule_003",
    "random_label": "rule_004",
}
DEPRECATED_RULES = ["low_tv_spectral_teacher"]
SEED_OFFSETS_BY_RULE = {
    "very_low_tv_spectral_teacher": 2026061900,
    "real_even_odd": 2026061800,
    "teacher_nn": 2026061800,
    "random_label": 2026061800,
}
PRODUCTION_RADII = [round(idx / 10.0, 1) for idx in range(1, 26)]
ADVANCED_RADII = [round(idx / 20.0, 2) for idx in range(2, 51)]
RADII = list(PRODUCTION_RADII)
R0 = 0.1
P = 2461.0

SPLIT_GATE = 0.004
ESS_GATE = 0.04
FINITE_FRACTION_GATE = 0.95
BOOTSTRAP_GATE = 0.012

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
LABEL_SAMPLING_SRC = SCRIPT_DIR.parents[2] / "label_noise_sweep" / "04_sampling" / "src"
if str(LABEL_SAMPLING_SRC) not in sys.path:
    sys.path.insert(0, str(LABEL_SAMPLING_SRC))

import resample_mnist10_local_support as resample  # noqa: E402


resample.RULES = list(RULES)


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


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def parse_radii_text(text: str) -> list[float]:
    radii = sorted({round(float(part.strip()), 6) for part in str(text).split(",") if part.strip()})
    if not radii:
        raise ValueError("--radii was provided but no valid radius values were found")
    invalid = [radius for radius in radii if radius <= 0]
    if invalid:
        raise ValueError(f"Radii must be positive: {invalid}")
    if round(float(R0), 6) not in radii:
        raise ValueError(f"Radii must include the anchor radius r0={R0}")
    return radii


def resolve_radii(radius_grid: str, radii_text: str = "") -> list[float]:
    if str(radii_text).strip():
        return parse_radii_text(radii_text)
    if radius_grid == "production":
        return list(PRODUCTION_RADII)
    if radius_grid == "advanced":
        return list(ADVANCED_RADII)
    raise ValueError(f"Unknown radius grid: {radius_grid}")


def activate_radii(radius_grid: str, radii_text: str = "") -> list[float]:
    global RADII
    RADII = resolve_radii(radius_grid, radii_text)
    resample.RADII = list(RADII)
    return list(RADII)


def radius_grid_kind(radius_grid: str, radii_text: str = "") -> str:
    if str(radii_text).strip():
        return f"custom_{len(RADII)}_radii"
    if radius_grid == "advanced":
        return "advanced_mechanical_0p1_to_2p5_step0p05"
    return "mechanical_0p1_to_2p5_step0p1"


def default_cpu_threads() -> int:
    count = os.cpu_count() or 1
    return max(1, min(4, int(math.floor(0.70 * count))))


def configure_resources(cpu_threads: int, device: str) -> None:
    threads = max(1, int(cpu_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = str(threads)
    if device:
        os.environ["MNIST14_DEVICE"] = str(device)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
    except Exception:
        pass


def load_reference_pool(extra_reference_run_root: Path, target_refs: int) -> pd.DataFrame:
    refs = pd.read_csv(REFERENCE_INDEX)
    refs = refs[~refs["rule"].astype(str).isin(DEPRECATED_RULES)].copy()
    very_low = pd.read_csv(VERY_LOW_REFERENCE_INDEX)
    refs = pd.concat([very_low, refs], ignore_index=True, sort=False)
    refs["ref_id"] = refs["ref_id"].astype(int)
    extra_path = extra_reference_run_root / "04_extra_reference_search" / EXTRA_REFERENCE_INDEX_NAME
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        extra["ref_id"] = extra["ref_id"].astype(int)
        refs = pd.concat([refs, extra], ignore_index=True, sort=False)
    rows: list[pd.DataFrame] = []
    missing: dict[str, int] = {}
    for rule in RULES:
        sub = (
            refs[refs["rule"].eq(rule)]
            .sort_values("ref_id")
            .drop_duplicates(["rule", "ref_id"], keep="first")
            .reset_index(drop=True)
        )
        if len(sub) < int(target_refs):
            missing[rule] = int(target_refs) - int(len(sub))
        rows.append(sub.head(int(target_refs)))
    if missing:
        raise RuntimeError(f"Reference pool is short for target_refs={target_refs}: {missing}")
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["pool_rank"] = out.groupby("rule").cumcount() + 1
    out["rule_id"] = out["rule"].map(RULE_IDS)
    out["resample_seed_offset"] = out["rule"].map(SEED_OFFSETS_BY_RULE).fillna(2026061800).astype(int)
    out["_rule_order"] = pd.Categorical(out["rule"], categories=RULES, ordered=True)
    return out.sort_values(["_rule_order", "pool_rank"]).drop(columns=["_rule_order"]).reset_index(drop=True)


def configure_sampler(args: argparse.Namespace, run_root: Path, cpu_threads: int) -> dict[str, Any]:
    resample.RADII = list(RADII)
    cfg = resample.configure_pipe(run_root)
    cfg["experiment_id"] = "mnist10_refpool1024_mechanical_all_radii"
    cfg["identity"] = run_root.name
    cfg["sampling"] = dict(cfg["sampling"])
    cfg["sampling"]["r0"] = float(R0)
    cfg["sampling"]["radii"] = list(RADII)
    cfg["sampling"]["radius_grid_kind"] = radius_grid_kind(str(args.radius_grid), str(args.radii))
    cfg["sampling"]["samples_per_ref_radius"] = int(args.samples_per_ref_radius)
    cfg["sampling"]["fallback_policies_enabled"] = False
    cfg["sampling"]["seed_offset"] = int(args.seed_offset)
    cfg["sampling"]["seed_offsets_by_rule"] = dict(SEED_OFFSETS_BY_RULE)
    cfg["sampling"]["radial_derivative_enabled"] = bool(args.direct_derivative)
    cfg["sampling"]["task_policy"] = "mechanical_all_rule_ref_radius_no_qc_gate"
    cfg["sampling"]["note"] = (
        "All configured rule/ref/radius units are sampled mechanically. QC diagnostics are reported "
        "afterward but do not select or skip tasks."
    )
    cfg["reference_search"] = dict(cfg["reference_search"])
    cfg["reference_search"]["target_pool_refs_per_rule"] = int(args.target_refs)
    cfg["reference_search"]["pool_source"] = str(REFERENCE_INDEX)
    cfg["reference_search"]["very_low_pool_source"] = str(VERY_LOW_REFERENCE_INDEX)
    cfg["reference_search"]["deprecated_rules_excluded"] = list(DEPRECATED_RULES)
    cfg["compute"] = dict(cfg.get("compute", {}))
    cfg["compute"]["chunk_size"] = int(args.chunk_size)
    cfg["compute"]["derivative_chunk_size"] = int(args.derivative_chunk_size)
    cfg["compute"]["device"] = str(args.device or os.environ.get("MNIST14_DEVICE", "cpu"))
    cfg["outputs"] = dict(cfg["outputs"])
    cfg["outputs"]["run_root"] = str(run_root)
    cfg["outputs"]["source_reference_run_root"] = str(REFERENCE_RUN_ROOT)
    cfg["outputs"]["very_low_source_reference_run_root"] = str(VERY_LOW_REFERENCE_RUN_ROOT)
    cfg["outputs"]["extra_reference_run_root"] = str(args.extra_reference_run_root)
    cfg["outputs"]["save_unit_samples_npz"] = bool(args.save_samples_npz)
    cfg["outputs"]["unit_layout"] = "05_pool2_pm_sais_sampling/unit_summaries/split_000/<rule>/ref_xxx/r_xxxx/{unit_summary.json,samples.npz}"
    cfg["qc"] = dict(cfg.get("qc", {}))
    cfg["qc"]["finite_unit_fraction_min"] = FINITE_FRACTION_GATE
    cfg["qc"]["q05_ess_fraction_min"] = ESS_GATE
    cfg["qc"]["max_split_logZ_per_P_diff"] = SPLIT_GATE
    cfg["qc"]["bootstrap_sd_phi_max"] = BOOTSTRAP_GATE
    cfg["resource_policy"] = {
        "cpu_limit_target": "use <=70% by limiting process/thread count",
        "gpu_limit_target": "default device=cpu keeps GPU at 0%; if cuda is requested, run one shard/process at a time",
        "cpu_threads_per_process": int(cpu_threads),
        "device": str(args.device or os.environ.get("MNIST14_DEVICE", "cpu")),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
    }
    cfg["resolved_at_unix"] = time.time()
    return cfg


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
    strict_paths = sorted(run_root.glob("rule_*/ref_*/r_*/unit_summary.json"))
    legacy_root = run_root / "05_pool2_pm_sais_sampling" / "unit_summaries"
    paths = strict_paths or sorted(legacy_root.rglob("unit_summary.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["unit_summary_path"] = str(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def add_derivative_columns(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    derivative_cols: list[str] = []
    for value_col in value_cols:
        derivative_cols.extend([f"d_{value_col}_dd", f"d2_{value_col}_dd2"])
    for col in derivative_cols:
        out[col] = np.nan
    if out.empty:
        return out
    for _, sub in out.groupby(group_cols, sort=False, dropna=False):
        sub = sub.sort_values("radius")
        x = pd.to_numeric(sub["radius"], errors="coerce").to_numpy(dtype=np.float64)
        for value_col in value_cols:
            y = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            valid_idx = sub.index[mask]
            if len(valid_idx) < 2:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            order = np.argsort(x_valid)
            sorted_idx = valid_idx.to_numpy()[order]
            x_sorted = x_valid[order]
            y_sorted = y_valid[order]
            d1 = np.gradient(y_sorted, x_sorted)
            out.loc[sorted_idx, f"d_{value_col}_dd"] = d1
            if len(sorted_idx) >= 3:
                out.loc[sorted_idx, f"d2_{value_col}_dd2"] = np.gradient(d1, x_sorted)
    return out


def derivative_columns(value_cols: list[str]) -> list[str]:
    cols: list[str] = []
    for value_col in value_cols:
        cols.extend([f"d_{value_col}_dd", f"d2_{value_col}_dd2"])
    return cols


def summarize_and_write(run_root: Path, cfg: dict[str, Any], pool_df: pd.DataFrame) -> dict[str, Any]:
    stage05 = ensure_dir(run_root / "05_pool2_pm_sais_sampling")
    out_dir = ensure_dir(run_root / "06_results_figures")
    unit_df = load_unit_payloads(run_root)
    if unit_df.empty:
        raise RuntimeError(f"No unit summaries found under {stage05}")
    for col in [
        "split_id",
        "ref_id",
        "radius",
        "n_samples",
        "logZ",
        "logZ_inf_full",
        "dlogZ_inf_full_dr",
        "split_dlogZ_dr_per_P_diff",
        "weighted_total_radial_score_sd",
        "ce_replay_max_abs_diff",
        "ess_fraction",
        "split_logZ_per_P_diff",
        "weighted_ce",
        "weighted_error",
    ]:
        if col in unit_df.columns:
            unit_df[col] = pd.to_numeric(unit_df[col], errors="coerce")
    unit_df["rule"] = unit_df["rule"].astype(str)
    write_csv(stage05 / "shell_summary_by_unit.csv", unit_df)

    key = ["split_id", "rule", "ref_id"]
    r0_df = (
        unit_df[np.isclose(unit_df["radius"], R0)][key + ["logZ_inf_full"]]
        .drop_duplicates(key, keep="first")
        .rename(columns={"logZ_inf_full": "logZ_r0"})
    )
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["phi_energy_raw"] = joined["logZ_inf_full"] / float(P)
    if "dlogZ_inf_full_dr" in joined.columns:
        joined["d_phi_energy_direct_dd_unit"] = joined["dlogZ_inf_full_dr"] / float(P)
        joined["d_delta_phi_energy_direct_dd_unit"] = joined["d_phi_energy_direct_dd_unit"]
    for direct_col in [
        "d_phi_energy_direct_dd_unit",
        "d_delta_phi_energy_direct_dd_unit",
        "split_dlogZ_dr_per_P_diff",
        "ce_replay_max_abs_diff",
    ]:
        if direct_col not in joined.columns:
            joined[direct_col] = np.nan
    joined["delta_phi_energy_unit"] = (joined["logZ_inf_full"] - joined["logZ_r0"]) / float(P)
    joined["delta_phi_full_unit"] = np.where(
        joined["radius"] > 0,
        ((P - 1.0) / P) * np.log(joined["radius"] / float(R0)) + joined["delta_phi_energy_unit"],
        np.nan,
    )
    write_csv(stage05 / "shell_summary_by_unit_with_phi.csv", joined)
    unit_derivative_value_cols = ["phi_energy_raw", "delta_phi_energy_unit", "delta_phi_full_unit"]
    joined_with_derivatives = add_derivative_columns(joined, ["split_id", "rule", "ref_id"], unit_derivative_value_cols)
    write_csv(stage05 / "shell_summary_by_unit_with_phi_derivatives.csv", joined_with_derivatives)

    target_refs = int(cfg["reference_search"]["target_pool_refs_per_rule"])
    summary_rows: list[dict[str, Any]] = []
    phi_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for rule in RULES:
        expected_refs = set(pool_df[pool_df["rule"].eq(rule)]["ref_id"].astype(int).tolist())
        for radius in RADII:
            sub = joined[joined["rule"].eq(rule) & np.isclose(joined["radius"], radius)].copy()
            observed_refs = set(sub["ref_id"].dropna().astype(int).tolist()) if len(sub) else set()
            missing_refs = sorted(expected_refs - observed_refs)
            finite_mask = np.isfinite(pd.to_numeric(sub["logZ_inf_full"], errors="coerce"))
            finite_count = int(finite_mask.sum()) if len(sub) else 0
            finite_fraction = float(finite_count / target_refs) if target_refs else 0.0
            q05_ess = float(np.quantile(sub["ess_fraction"].dropna(), 0.05)) if len(sub) else float("nan")
            max_split = float(sub["split_logZ_per_P_diff"].max()) if len(sub) else float("nan")
            raw_values = sub["phi_energy_raw"].to_numpy(dtype=np.float64) if len(sub) else np.asarray([], dtype=np.float64)
            delta_values = sub["delta_phi_energy_unit"].to_numpy(dtype=np.float64) if len(sub) else np.asarray([], dtype=np.float64)
            boot_sd = bootstrap_sd(delta_values, 912000 + RULES.index(rule) * 1000 + int(round(float(radius) * 1000))) if len(sub) else float("nan")
            complete = len(observed_refs) == target_refs
            qc_diagnostic_pass = bool(
                complete
                and finite_fraction >= FINITE_FRACTION_GATE
                and np.isfinite(q05_ess)
                and q05_ess >= ESS_GATE
                and np.isfinite(max_split)
                and max_split <= SPLIT_GATE
                and np.isfinite(boot_sd)
                and boot_sd <= BOOTSTRAP_GATE
            )
            mean_raw = float(np.mean(raw_values[np.isfinite(raw_values)])) if np.isfinite(raw_values).any() else float("nan")
            mean_delta = float(np.mean(delta_values[np.isfinite(delta_values)])) if np.isfinite(delta_values).any() else float("nan")
            mean_full = (
                float(((P - 1.0) / P) * math.log(float(radius) / float(R0)) + mean_delta)
                if np.isfinite(mean_delta)
                else float("nan")
            )
            row = {
                "rule": rule,
                "radius": float(radius),
                "d0": float(R0),
                "target_ref_count": int(target_refs),
                "observed_ref_count": int(len(observed_refs)),
                "missing_ref_count": int(len(missing_refs)),
                "finite_unit_count": int(finite_count),
                "finite_unit_fraction": float(finite_fraction),
                "q05_ess_fraction": q05_ess,
                "max_split_logZ_per_P_diff": max_split,
                "max_split_dlogZ_dr_per_P_diff": float(sub["split_dlogZ_dr_per_P_diff"].max()) if "split_dlogZ_dr_per_P_diff" in sub.columns else float("nan"),
                "max_ce_replay_abs_diff": float(sub["ce_replay_max_abs_diff"].max()) if "ce_replay_max_abs_diff" in sub.columns else float("nan"),
                "bootstrap_sd_delta_phi_energy": boot_sd,
                "qc_diagnostic_pass": qc_diagnostic_pass,
                "sampling_status": "complete" if complete else "partial_missing_units",
                "missing_ref_ids": ",".join(str(ref_id) for ref_id in missing_refs[:40]),
            }
            summary_rows.append(
                {
                    **row,
                    "mean_logZ_inf_full": float(sub["logZ_inf_full"].mean()) if len(sub) else float("nan"),
                    "mean_phi_energy_raw": mean_raw,
                    "mean_delta_phi_energy": mean_delta,
                    "mean_delta_phi_full": mean_full,
                    "mean_d_phi_energy_direct_dd": float(sub["d_phi_energy_direct_dd_unit"].mean()) if len(sub) else float("nan"),
                    "sd_d_phi_energy_direct_dd": float(sub["d_phi_energy_direct_dd_unit"].std(ddof=1)) if len(sub) > 1 else float("nan"),
                    "sem_d_phi_energy_direct_dd": float(sub["d_phi_energy_direct_dd_unit"].std(ddof=1) / math.sqrt(len(sub))) if len(sub) > 1 else 0.0,
                    "weighted_ce_mean": float(sub["weighted_ce"].mean()) if len(sub) else float("nan"),
                    "weighted_error_mean": float(sub["weighted_error"].mean()) if len(sub) else float("nan"),
                }
            )
            phi_rows.append(
                {
                    "rule": rule,
                    "radius": float(radius),
                    "d0": float(R0),
                    "phi_energy_raw": mean_raw,
                    "delta_phi_energy": mean_delta,
                    "delta_phi_full": mean_full,
                    "d_phi_energy_direct_dd": float(sub["d_phi_energy_direct_dd_unit"].mean()) if len(sub) else float("nan"),
                    "d_phi_energy_direct_dd_sem": float(sub["d_phi_energy_direct_dd_unit"].std(ddof=1) / math.sqrt(len(sub))) if len(sub) > 1 else 0.0,
                    "n_units": int(len(sub)),
                    "target_ref_count": int(target_refs),
                    "sampling_status": row["sampling_status"],
                    "qc_diagnostic_pass": qc_diagnostic_pass,
                }
            )
            qc_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["rule", "radius"]).reset_index(drop=True)
    phi_df = pd.DataFrame(phi_rows).sort_values(["rule", "radius"]).reset_index(drop=True)
    phi_df = add_derivative_columns(phi_df, ["rule"], ["phi_energy_raw", "delta_phi_energy", "delta_phi_full"])
    qc_df = pd.DataFrame(qc_rows).sort_values(["rule", "radius"]).reset_index(drop=True)
    write_csv(stage05 / "shell_summary_by_rule_radius.csv", summary_df)
    write_csv(stage05 / "qc_diagnostics_by_rule_radius.csv", qc_df)
    write_csv(out_dir / "phi_by_rule_radius.csv", phi_df)
    write_csv(out_dir / "phi_raw_by_rule_radius.csv", phi_df[["rule", "radius", "phi_energy_raw", "n_units", "target_ref_count", "sampling_status"]])
    dphi_cols = ["rule", "radius", "n_units", "target_ref_count", "sampling_status", "qc_diagnostic_pass"] + derivative_columns(
        ["phi_energy_raw", "delta_phi_energy", "delta_phi_full"]
    )
    dphi_cols.extend(["d_phi_energy_direct_dd", "d_phi_energy_direct_dd_sem"])
    write_csv(out_dir / "dphi_dd_by_rule_radius.csv", phi_df[dphi_cols])

    completed_units = int(len(joined.drop_duplicates(["rule", "ref_id", "radius"])))
    expected_units = int(len(RULES) * target_refs * len(RADII))
    complete_rule_radius = int((summary_df["sampling_status"] == "complete").sum())
    status = {
        "status": "complete" if completed_units >= expected_units and complete_rule_radius == len(RULES) * len(RADII) else "partial",
        "rules": RULES,
        "radii": RADII,
        "r0": float(R0),
        "samples_per_ref_radius": int(cfg["sampling"]["samples_per_ref_radius"]),
        "target_refs_per_rule": int(target_refs),
        "completed_units": completed_units,
        "expected_units": expected_units,
        "complete_rule_radius_rows": complete_rule_radius,
        "total_rule_radius_rows": int(len(RULES) * len(RADII)),
        "qc_diagnostic_pass_rows": int(qc_df["qc_diagnostic_pass"].sum()),
        "save_unit_samples_npz": bool(cfg["outputs"].get("save_unit_samples_npz", False)),
        "radial_derivative_enabled": bool(cfg["sampling"].get("radial_derivative_enabled", False)),
        "max_split_dlogZ_dr_per_P_diff": float(joined["split_dlogZ_dr_per_P_diff"].max()),
        "max_ce_replay_abs_diff": float(joined["ce_replay_max_abs_diff"].max()),
        "radius_grid_kind": str(cfg["sampling"].get("radius_grid_kind", "")),
        "derivative_outputs": {
            "unit_table": "05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv",
            "rule_radius_table": "06_results_figures/dphi_dd_by_rule_radius.csv",
        },
    }
    write_json(run_root / "SAMPLING_STATUS.json", status)
    write_json(stage05 / "SAMPLING_STATUS.json", status)
    write_json(out_dir / "run_config_resolved.json", {**cfg, "aggregate_status": status})
    write_report(run_root, status, summary_df)
    return status


def write_report(run_root: Path, status: dict[str, Any], summary_df: pd.DataFrame) -> None:
    by_rule = (
        summary_df.groupby("rule", as_index=False)
        .agg(
            complete_radii=("sampling_status", lambda s: int((s == "complete").sum())),
            observed_units=("observed_ref_count", "sum"),
            missing_units=("missing_ref_count", "sum"),
            qc_diagnostic_pass_radii=("qc_diagnostic_pass", "sum"),
        )
        .sort_values("rule")
    )
    def markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in df.iterrows():
            values: list[str] = []
            for col in cols:
                value = row[col]
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value).replace("|", "\\|").replace("\n", " "))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# MNIST Refpool 1024 Mechanical Sampling",
        "",
        f"- Status: `{status['status']}`",
        f"- Units: `{status['completed_units']}` / `{status['expected_units']}`",
        f"- References per rule: `{status['target_refs_per_rule']}`",
        f"- Radius grid: `{status.get('radius_grid_kind', 'mechanical_0p1_to_2p5_step0p1')}`",
        f"- Radii: `{RADII[0]:.4g}..{RADII[-1]:.4g}` ({len(RADII)} values)",
        f"- Samples per unit: `{status['samples_per_ref_radius']}`",
        f"- QC diagnostic pass rows: `{status['qc_diagnostic_pass_rows']}` / `{status['total_rule_radius_rows']}`",
        "",
        markdown_table(by_rule),
        "",
        "QC diagnostics are reported but are not used to select or skip sampling units.",
        "",
        "Primary files:",
        "",
        "- `04_reference_pool/reference_pool_index.csv`",
        "- `05_pool2_pm_sais_sampling/shell_summary_by_unit.csv`",
        "- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi.csv`",
        "- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv`",
        "- `05_pool2_pm_sais_sampling/shell_summary_by_rule_radius.csv`",
        "- `05_pool2_pm_sais_sampling/qc_diagnostics_by_rule_radius.csv`",
        "- `06_results_figures/phi_by_rule_radius.csv`",
        "- `06_results_figures/dphi_dd_by_rule_radius.csv`",
        "- `SAMPLING_STATUS.json`",
    ]
    write_text(run_root / "REPORT.md", "\n".join(lines) + "\n")


def task_rows(pool_df: pd.DataFrame) -> list[tuple[dict[str, Any], float]]:
    rows: list[tuple[dict[str, Any], float]] = []
    for row in pool_df.to_dict("records"):
        item = dict(row)
        item["rule"] = str(item["rule"])
        item["ref_id"] = int(item["ref_id"])
        item["split_id"] = int(item.get("split_id", 0))
        for radius in RADII:
            rows.append((item, float(radius)))
    return rows


def run_sampling(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    cpu_threads = int(args.cpu_threads or default_cpu_threads())
    configure_resources(cpu_threads, str(args.device))
    cfg = configure_sampler(args, run_root, cpu_threads)
    pool_df = load_reference_pool(Path(args.extra_reference_run_root), int(args.target_refs))
    write_csv(run_root / "04_reference_pool" / "reference_pool_index.csv", pool_df)
    write_json(
        run_root / "04_reference_pool" / "POOL_STATUS.json",
        {
            "target_refs_per_rule": int(args.target_refs),
            "available_refs_per_rule": pool_df.groupby("rule")["ref_id"].nunique().astype(int).to_dict(),
            "reference_index": str(REFERENCE_INDEX),
            "very_low_reference_index": str(VERY_LOW_REFERENCE_INDEX),
            "extra_reference_run_root": str(args.extra_reference_run_root),
            "deprecated_rules_excluded": list(DEPRECATED_RULES),
        },
    )
    write_json(run_root / "run_config_resolved.json", cfg)

    tasks_all = task_rows(pool_df)
    tasks = [
        task
        for idx, task in enumerate(tasks_all)
        if idx % int(args.shard_count) == int(args.shard_index)
    ]
    if args.max_units is not None:
        tasks = tasks[: int(args.max_units)]
    write_csv(
        run_root / "05_pool2_pm_sais_sampling" / f"tasks_shard{args.shard_index}_of_{args.shard_count}.csv",
        pd.DataFrame(
            [
                {
                    "task_index": idx,
                    "rule": row["rule"],
                    "ref_id": int(row["ref_id"]),
                    "pool_rank": int(row.get("pool_rank", -1)),
                    "radius": float(radius),
                    "resample_seed_offset": int(row.get("resample_seed_offset", cfg["sampling"]["seed_offset"])),
                }
                for idx, (row, radius) in enumerate(tasks)
            ]
        ),
    )

    rows: list[dict[str, Any]] = []
    shard_log_latest = run_root / "05_pool2_pm_sais_sampling" / f"sampling_log_latest_shard{args.shard_index}_of_{args.shard_count}.csv"
    started = time.time()
    for idx, (row, radius) in enumerate(tasks, start=1):
        print(
            f"[refpool1024] shard={args.shard_index}/{args.shard_count} unit={idx}/{len(tasks)} "
            f"rule={row['rule']} ref={int(row['ref_id']):03d} r={radius:.2f}",
            flush=True,
        )
        payload = resample.sample_unit(row, float(radius), cfg, run_root, force=bool(args.force))
        rows.append(
            {
                "rule": str(row["rule"]),
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
            summarize_and_write(run_root, cfg, pool_df)
            write_csv(shard_log_latest, pd.DataFrame(rows))

    write_csv(run_root / "05_pool2_pm_sais_sampling" / f"shard{args.shard_index}_unit_summary.csv", pd.DataFrame(rows))
    write_csv(shard_log_latest, pd.DataFrame(rows))
    if args.no_final_aggregate:
        status = {
            "status": "sampling_shard_complete",
            "shard_units_this_invocation": int(len(rows)),
            "elapsed_s": float(time.time() - started),
        }
    else:
        status = summarize_and_write(run_root, cfg, pool_df)
        status["shard_units_this_invocation"] = int(len(rows))
        status["elapsed_s"] = float(time.time() - started)
        write_json(run_root / "SAMPLING_STATUS.json", status)
    print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run mechanical 1024-particle sampling over the MNIST reference pool.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--extra-reference-run-root", default=str(DEFAULT_EXTRA_REFERENCE_RUN_ROOT))
    parser.add_argument("--radius-grid", choices=["production", "advanced"], default="production")
    parser.add_argument("--radii", default="", help="Optional comma-separated custom radius grid; must include r0=0.1.")
    parser.add_argument("--target-refs", type=int, default=60)
    parser.add_argument("--samples-per-ref-radius", type=int, default=1024)
    parser.add_argument("--seed-offset", type=int, default=2026061800)
    parser.add_argument("--device", default=os.environ.get("MNIST14_DEVICE", "cpu"))
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--derivative-chunk-size", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--aggregate-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-final-aggregate", action="store_true")
    parser.add_argument("--save-samples-npz", dest="save_samples_npz", action="store_true", default=True)
    parser.add_argument("--no-save-samples-npz", dest="save_samples_npz", action="store_false")
    parser.add_argument("--direct-derivative", action="store_true")
    args = parser.parse_args(argv)
    activate_radii(str(args.radius_grid), str(args.radii))

    run_root = Path(args.run_root)
    cpu_threads = int(args.cpu_threads or default_cpu_threads())
    configure_resources(cpu_threads, str(args.device))
    if args.aggregate_only:
        cfg = configure_sampler(args, run_root, cpu_threads)
        pool_df = load_reference_pool(Path(args.extra_reference_run_root), int(args.target_refs))
        status = summarize_and_write(run_root, cfg, pool_df)
        print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
        return 0 if status["status"] == "complete" else 2
    status = run_sampling(args)
    return 0 if status["status"] in {"complete", "sampling_shard_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
