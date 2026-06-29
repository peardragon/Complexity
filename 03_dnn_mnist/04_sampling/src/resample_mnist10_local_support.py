#!/usr/bin/env python3
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


REPO_ROOT = Path("/home/bjyong/Complexity/windows_project")
MNIST_ROOT = REPO_ROOT / "02_dnn" / "08_mnist"
LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
LOCAL_SRC_CACHE = LOCAL_ROOT / ".src_cache_mnist"
SRC_DIR = LOCAL_SRC_CACHE if LOCAL_SRC_CACHE.exists() else MNIST_ROOT / "src"
SOURCE_RUN_ROOT = MNIST_ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
ANALYSIS_ROOT = SOURCE_RUN_ROOT / "07_reference_family_analysis"
DEFAULT_RUN_ROOT = MNIST_ROOT / "runs" / "final" / "local_support_dmax0p65_all_rules_resampled"
SELECTOR = "dense_qc_stable_ref30"
RULES = ["low_tv_spectral_teacher", "real_even_odd", "teacher_nn", "random_label"]
RADII = [
    0.010,
    0.011,
    0.012,
    0.013,
    0.014,
    0.016,
    0.018,
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,
    0.065,
    0.080,
    0.120,
    0.150,
    0.200,
    0.300,
    0.450,
    0.650,
]

SPLIT_GATE = 0.004
ESS_GATE = 0.04
BOOTSTRAP_GATE = 0.012
FINITE_FRACTION_GATE = 0.95
SEED_OFFSET = 2026061600


if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mnist10_allrule_sparse_to_2p50_pipeline as sparse  # noqa: E402


pipe = sparse.base.pipe
if SRC_DIR == LOCAL_SRC_CACHE:
    sparse.REPO_ROOT = REPO_ROOT
    sparse.base.REPO_ROOT = REPO_ROOT
    sparse.base.pipe.REPO_ROOT = REPO_ROOT
P = pipe.P


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


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


def radius_token(radius: float) -> str:
    return f"r_{float(radius):.4f}".replace(".", "p")


def configure_pipe(run_root: Path) -> dict[str, Any]:
    sparse.RUN_ROOT = run_root
    sparse.RADII = list(RADII)
    sparse.configure_pipe()
    cfg = pipe.load_config()
    cfg["experiment_id"] = "mnist10_local_support_dmax0p65_all_rules_resampled"
    cfg["identity"] = run_root.name
    cfg["sampling"] = dict(cfg["sampling"])
    cfg["sampling"]["radii"] = list(RADII)
    cfg["sampling"]["radius_grid_kind"] = "dense_local_support_0p010_to_0p650_resampled"
    cfg["sampling"]["seed_offset"] = SEED_OFFSET
    cfg["sampling"]["resampling_note"] = (
        "Fresh PM-SAIS unit summaries using existing exact references and selector membership. "
        "Procedure and fallback policies are inherited from 08_mnist; seeds are offset for an independent rerun."
    )
    cfg["reference_search"] = dict(cfg["reference_search"])
    cfg["reference_search"]["selected_refs_per_dataset"] = 30
    cfg["outputs"] = dict(cfg["outputs"])
    cfg["outputs"]["run_root"] = rel(run_root)
    cfg["outputs"]["source_run_root"] = rel(SOURCE_RUN_ROOT)
    cfg["outputs"]["analysis_root"] = rel(ANALYSIS_ROOT)
    cfg["qc"] = dict(cfg["qc"])
    cfg["qc"]["finite_unit_fraction_min"] = FINITE_FRACTION_GATE
    cfg["qc"]["q05_ess_fraction_min"] = ESS_GATE
    cfg["qc"]["max_split_logZ_per_P_diff"] = SPLIT_GATE
    cfg["qc"]["bootstrap_sd_phi_max"] = BOOTSTRAP_GATE
    cfg["resolved_at_unix"] = time.time()
    return cfg


def unit_summary_path(run_root: Path, row: dict[str, Any], radius: float) -> Path:
    return (
        run_root
        / "05_pool2_pm_sais_sampling"
        / "unit_summaries"
        / "split_000"
        / str(row["rule"])
        / f"ref_{int(row['ref_id']):03d}"
        / radius_token(radius)
        / "unit_summary.json"
    )


def parse_radii_filter(text: str) -> list[float]:
    if not text.strip():
        return list(RADII)
    requested = [float(x.strip()) for x in text.split(",") if x.strip()]
    valid = {round(float(r), 4): float(r) for r in RADII}
    missing = [r for r in requested if round(float(r), 4) not in valid]
    if missing:
        raise ValueError(f"Requested radii are not in the configured grid: {missing}")
    return [valid[round(float(r), 4)] for r in requested]


def parse_rules_filter(text: str) -> list[str]:
    if not text.strip():
        return list(RULES)
    requested = [x.strip() for x in text.split(",") if x.strip()]
    missing = [rule for rule in requested if rule not in RULES]
    if missing:
        raise ValueError(f"Requested rules are not in the configured rules: {missing}")
    return requested


def load_selected_tasks(
    ref_ids: set[int] | None = None,
    radii: list[float] | None = None,
    rules: list[str] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    membership_path = ANALYSIS_ROOT / "selector_membership.csv"
    reference_path = SOURCE_RUN_ROOT / "04_exact_reference_search" / "reference_index.csv"
    membership = pd.read_csv(membership_path)
    references = pd.read_csv(reference_path)
    selected_rules = list(RULES if rules is None else rules)
    membership = membership[
        membership["selector"].eq(SELECTOR)
        & membership["rule"].isin(selected_rules)
    ].copy()
    if ref_ids is not None:
        membership = membership[membership["ref_id"].astype(int).isin(ref_ids)].copy()
    selected = membership[["rule", "ref_id"]].drop_duplicates()
    references["ref_id"] = references["ref_id"].astype(int)
    rows = references.merge(selected, on=["rule", "ref_id"], how="inner")
    rows["rule"] = pd.Categorical(rows["rule"], categories=RULES, ordered=True)
    rows = rows.sort_values(["rule", "ref_id"]).reset_index(drop=True)
    tasks: list[tuple[dict[str, Any], float]] = []
    selected_radii = list(RADII if radii is None else radii)
    for row in rows.to_dict("records"):
        row["rule"] = str(row["rule"])
        for radius in selected_radii:
            tasks.append((row, float(radius)))
    return tasks


def samples_npz_path_from_summary(path: Path) -> Path:
    return path.with_name("samples.npz")


def save_samples_npz(path: Path, samples: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npz")
    arrays = {str(key): np.asarray(value) for key, value in samples.items()}
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


def reusable_resampled_payload(
    path: Path,
    radius: float,
    seed_offset: int,
    force: bool,
    *,
    n_samples: int | None = None,
    require_samples_npz: bool = False,
    require_direct_derivative: bool = False,
) -> dict[str, Any] | None:
    if force or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        abs(float(payload.get("radius", float("nan"))) - float(radius)) <= 1e-12
        and int(payload.get("resample_seed_offset", -1)) == int(seed_offset)
        and (n_samples is None or int(payload.get("n_samples", payload.get("n_samples_total", -1))) == int(n_samples))
        and (not require_samples_npz or samples_npz_path_from_summary(path).exists())
        and math.isfinite(float(payload.get("logZ_inf_full", float("nan"))))
        and math.isfinite(float(payload.get("split_logZ_per_P_diff", float("nan"))))
        and (not require_direct_derivative or math.isfinite(float(payload.get("dlogZ_inf_full_dr", float("nan")))))
    ):
        payload["reused"] = True
        return payload
    return None


def sample_unit(row: dict[str, Any], radius: float, cfg: dict[str, Any], run_root: Path, *, force: bool) -> dict[str, Any]:
    path = unit_summary_path(run_root, row, radius)
    save_unit_samples = bool((cfg.get("outputs") or {}).get("save_unit_samples_npz", False))
    seed_offset = int(row.get("resample_seed_offset", cfg["sampling"]["seed_offset"]))
    cached = reusable_resampled_payload(
        path,
        radius,
        seed_offset,
        force,
        n_samples=int(cfg["sampling"]["samples_per_ref_radius"]),
        require_samples_npz=save_unit_samples,
        require_direct_derivative=bool(cfg.get("sampling", {}).get("radial_derivative_enabled", False)),
    )
    if cached is not None:
        return cached

    policy = pipe.fallback_policy_for(str(row["rule"]), float(radius), int(row["ref_id"])) if bool(cfg["sampling"].get("fallback_policies_enabled", True)) else None
    lambda_reg = float(cfg["sampling"]["lambda_reg"])
    if policy is None:
        n_samples = int(cfg["sampling"]["samples_per_ref_radius"])
        ds = pipe.load_dataset(row["dataset_path"])
        theta_ref = np.load(REPO_ROOT / str(row["theta_path"])).astype(np.float64).reshape(-1)
        seed = seed_offset + 3900000 + RULES.index(str(row["rule"])) * 100000 + int(row["ref_id"]) * 1000 + int(round(float(radius) * 10000))
        started = time.time()
        smc = pipe.run_smc_split(theta_ref, ds, float(radius), n_samples, lambda_reg, seed, cfg, float(row["CE_mean_train"]))
        samples_npz = smc.pop("_samples_npz", None)
        payload = {
            "split_id": int(row["split_id"]),
            "rule": str(row["rule"]),
            "ref_id": int(row["ref_id"]),
            "radius": float(radius),
            "n_samples": n_samples,
            "seed": seed,
            "resample_seed_offset": seed_offset,
            "lambda_reg": lambda_reg,
            "theta_path": str(row["theta_path"]),
            "dataset_path": str(row["dataset_path"]),
            "theta_ref_norm": float(np.linalg.norm(theta_ref)),
            "sampler_method": "exact_shell_l2_vmf_adaptive_ce_tempered_smc",
            "fallback_policy_name": "baseline",
            "finite": bool(np.isfinite(smc["logZ"]) and np.isfinite(smc["logZ_inf_full"])),
            "elapsed_s": float(time.time() - started),
            "reused": False,
            **smc,
        }
        if save_unit_samples and samples_npz is not None:
            samples_path = save_samples_npz(samples_npz_path_from_summary(path), samples_npz)
            payload["samples_path"] = str(samples_path)
    else:
        fallback_cfg = pipe.cfg_for_fallback_policy(cfg, policy)
        seed = seed_offset + 9900000 + RULES.index(str(row["rule"])) * 100000 + int(row["ref_id"]) * 1000 + int(round(float(radius) * 10000))
        started = time.time()
        payload = pipe.run_replicated_smc(
            row,
            float(radius),
            fallback_cfg,
            n_samples_each=int(policy["n_samples_each"]),
            replicates=int(policy["replicates"]),
            lambda_reg=lambda_reg,
            seed=seed,
        )
        payload["fallback_policy_name"] = str(policy["name"])
        payload["fallback_target_cess_fraction"] = float(policy["target_cess_fraction"])
        payload["fallback_mh_sweeps"] = int(policy["mh_sweeps"])
        payload["fallback_move_kappa_factor"] = float(policy["move_kappa_factor"])
        payload["seed"] = seed
        payload["resample_seed_offset"] = seed_offset
        payload["elapsed_s"] = float(time.time() - started)
        payload["finite"] = bool(np.isfinite(payload["logZ_inf_full"]))
        payload["reused"] = False

    write_json(path, payload)
    return payload


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
    for path in sorted((run_root / "05_pool2_pm_sais_sampling" / "unit_summaries").rglob("unit_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unit_summary_path"] = rel(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize_and_write(run_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(run_root / "06_results_figures")
    stage05 = ensure_dir(run_root / "05_pool2_pm_sais_sampling")
    unit_df = load_unit_payloads(run_root)
    if unit_df.empty:
        raise RuntimeError(f"No unit_summary.json files found under {stage05}")
    for col in ["split_id", "ref_id", "radius", "logZ_inf_full", "ess_fraction", "split_logZ_per_P_diff", "weighted_ce", "weighted_error"]:
        if col in unit_df.columns:
            unit_df[col] = pd.to_numeric(unit_df[col], errors="coerce")
    unit_df["rule"] = unit_df["rule"].astype(str)
    write_csv(stage05 / "shell_summary_by_unit.csv", unit_df)

    key = ["split_id", "rule", "ref_id"]
    r0 = float(cfg["sampling"]["r0"])
    r0_df = unit_df[np.isclose(unit_df["radius"], r0)][key + ["logZ_inf_full"]].rename(columns={"logZ_inf_full": "logZ_r0"})
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined["logZ_inf_full"] - joined["logZ_r0"]) / float(P)
    joined["delta_phi_full_unit"] = np.where(
        joined["radius"] > 0,
        ((P - 1.0) / P) * np.log(joined["radius"] / r0) + joined["delta_phi_energy_unit"],
        np.nan,
    )
    write_csv(stage05 / "shell_summary_by_unit_with_phi.csv", joined)

    selected_tasks = load_selected_tasks()
    selected_count = int(len({(row["rule"], int(row["ref_id"])) for row, _ in selected_tasks}) / len(RULES))
    summary_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    phi_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for rule in RULES:
        for radius in RADII:
            sub = joined[(joined["rule"].eq(rule)) & np.isclose(joined["radius"], radius)].copy()
            observed_refs = int(sub["ref_id"].nunique()) if len(sub) else 0
            finite_count = int(np.isfinite(sub["delta_phi_energy_unit"]).sum()) if len(sub) else 0
            finite_fraction = float(finite_count / selected_count) if selected_count else 0.0
            q05_ess = float(np.quantile(sub["ess_fraction"].dropna(), 0.05)) if len(sub) else float("nan")
            max_split = float(sub["split_logZ_per_P_diff"].max()) if len(sub) else float("nan")
            values = sub["delta_phi_energy_unit"].to_numpy(dtype=np.float64) if len(sub) else np.asarray([], dtype=np.float64)
            boot_sd = bootstrap_sd(values, 812000 + RULES.index(rule) * 1000 + int(round(float(radius) * 10000))) if len(sub) else float("nan")
            complete = observed_refs == selected_count
            pass_qc = bool(
                complete
                and finite_fraction >= FINITE_FRACTION_GATE
                and np.isfinite(q05_ess)
                and q05_ess >= ESS_GATE
                and np.isfinite(max_split)
                and max_split <= SPLIT_GATE
                and np.isfinite(boot_sd)
                and boot_sd <= BOOTSTRAP_GATE
            )
            claim_status = "claimable_rule_radius" if pass_qc else ("no_claim_missing_units" if not complete else "no_claim_qc_fail")
            mean_energy = float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")
            mean_full = float(((P - 1.0) / P) * math.log(float(radius) / r0) + mean_energy) if np.isfinite(mean_energy) else float("nan")
            weighted_ce = float(sub["weighted_ce"].mean()) if len(sub) else float("nan")
            weighted_error = float(sub["weighted_error"].mean()) if len(sub) else float("nan")
            row = {
                "selector": SELECTOR,
                "rule": rule,
                "radius": float(radius),
                "d0": r0,
                "selected_ref_count": selected_count,
                "observed_ref_count": observed_refs,
                "finite_unit_count": finite_count,
                "finite_unit_fraction": finite_fraction,
                "q05_ess_fraction": q05_ess,
                "max_split_logZ_per_P_diff": max_split,
                "bootstrap_sd_phi": boot_sd,
                "weighted_ce_mean": weighted_ce,
                "weighted_error_mean": weighted_error,
                "qc_pass": pass_qc,
                "claim_status": claim_status,
            }
            summary_rows.append({
                **row,
                "mean_logZ": float(sub["logZ_inf_full"].mean()) if len(sub) else float("nan"),
                "mean_delta_phi_energy": mean_energy,
            })
            qc_rows.append(row)
            phi_rows.append({
                "selector": SELECTOR,
                "rule": rule,
                "radius": float(radius),
                "d0": r0,
                "delta_phi_energy": mean_energy,
                "delta_phi_full": mean_full,
                "n_units": int(len(sub)),
                "qc_pass": pass_qc,
                "claim_status": claim_status,
            })
            boot_rows.append({
                "rule": rule,
                "radius": float(radius),
                "delta_phi_energy_mean": mean_energy,
                "bootstrap_sd": boot_sd,
                "ci95_low": mean_energy - 1.96 * boot_sd if np.isfinite(mean_energy) and np.isfinite(boot_sd) else float("nan"),
                "ci95_high": mean_energy + 1.96 * boot_sd if np.isfinite(mean_energy) and np.isfinite(boot_sd) else float("nan"),
            })

    summary_df = pd.DataFrame(summary_rows)
    qc_df = pd.DataFrame(qc_rows)
    phi_df = pd.DataFrame(phi_rows)
    boot_df = pd.DataFrame(boot_rows)
    write_csv(stage05 / "shell_summary_by_rule_radius.csv", summary_df)
    write_csv(stage05 / "qc_by_rule_radius.csv", qc_df)
    write_csv(out_dir / "phi_by_rule_radius.csv", phi_df)
    write_csv(out_dir / "phi_bootstrap_by_rule_radius.csv", boot_df)
    write_csv(out_dir / "qc_pass_by_rule_radius.csv", qc_df)

    dphi_rows: list[dict[str, Any]] = []
    for rule, sub in phi_df[np.isfinite(phi_df["delta_phi_energy"])].groupby("rule"):
        sub = sub.sort_values("radius")
        x = sub["radius"].to_numpy(dtype=np.float64)
        y = sub["delta_phi_energy"].to_numpy(dtype=np.float64)
        if len(sub) == 1:
            deriv = np.asarray([float("nan")])
        else:
            deriv = np.gradient(y, x)
        for row, value in zip(sub.to_dict("records"), deriv):
            dphi_rows.append({
                "selector": SELECTOR,
                "rule": str(rule),
                "radius": float(row["radius"]),
                "d_delta_phi_energy_dd": float(value),
                "qc_pass": bool(row["qc_pass"]),
                "claim_status": str(row["claim_status"]),
            })
    dphi_df = pd.DataFrame(dphi_rows)
    write_csv(out_dir / "dphi_dd_energy_by_rule_radius.csv", dphi_df)

    claim_df = phi_df.merge(qc_df[["rule", "radius", "selected_ref_count", "observed_ref_count", "q05_ess_fraction", "max_split_logZ_per_P_diff", "bootstrap_sd_phi", "weighted_ce_mean", "weighted_error_mean"]], on=["rule", "radius"], how="left")
    common = qc_df.groupby("radius")["qc_pass"].all().rename("all_rule_common_qc_pass_at_radius").reset_index()
    claim_df = claim_df.merge(common, on="radius", how="left")
    claim_df["comparison_claim_status"] = np.where(
        claim_df["all_rule_common_qc_pass_at_radius"] & claim_df["qc_pass"],
        "claimable_all_rule_comparison_radius",
        np.where(claim_df["qc_pass"], "per_rule_qc_pass_but_no_all_rule_comparison_claim", claim_df["claim_status"]),
    )
    write_csv(out_dir / "final_claim_table.csv", claim_df)

    completed_units = int(len(joined))
    expected_units = len(RULES) * selected_count * len(RADII)
    status = {
        "status": "complete" if completed_units == expected_units else "partial",
        "selector": SELECTOR,
        "rules": RULES,
        "radii": RADII,
        "selected_refs_per_rule": selected_count,
        "completed_units": completed_units,
        "expected_units": expected_units,
        "all_rule_common_qc_pass_radii": [float(r) for r in sorted(common[common["all_rule_common_qc_pass_at_radius"]]["radius"].tolist())],
        "no_claim_by_rule_radius": qc_df[~qc_df["qc_pass"]][["rule", "radius", "claim_status"]].to_dict("records"),
    }
    write_json(run_root / "QC_STATUS.json", status)
    write_json(out_dir / "run_config_resolved.json", {**cfg, "aggregate_status": status})
    return status


def run_units(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    cfg = configure_pipe(run_root)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    if args.device:
        os.environ["MNIST14_DEVICE"] = args.device
    ensure_dir(run_root / "logs")
    write_json(run_root / "run_config_resolved.json", {
        **cfg,
        "resource_policy": {
            "cpu_threads_per_process": int(args.cpu_threads),
            "device": os.environ.get("MNIST14_DEVICE", "auto"),
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
        },
    })
    ref_filter = {int(x) for x in args.ref_ids.split(",") if x.strip()} if args.ref_ids else None
    radii_filter = parse_radii_filter(args.radii)
    rules_filter = parse_rules_filter(args.rules)
    tasks = load_selected_tasks(ref_filter, radii_filter, rules_filter)
    tasks = [task for idx, task in enumerate(tasks) if idx % int(args.shard_count) == int(args.shard_index)]
    if args.max_units is not None:
        tasks = tasks[: int(args.max_units)]
    write_csv(
        run_root / "05_pool2_pm_sais_sampling" / f"tasks_shard{args.shard_index}_of_{args.shard_count}.csv",
        pd.DataFrame([
            {"task_index": idx, "rule": row["rule"], "ref_id": int(row["ref_id"]), "radius": float(radius)}
            for idx, (row, radius) in enumerate(tasks)
        ]),
    )
    rows: list[dict[str, Any]] = []
    started = time.time()
    for idx, (row, radius) in enumerate(tasks, start=1):
        print(
            f"[resample] shard={args.shard_index}/{args.shard_count} unit={idx}/{len(tasks)} "
            f"rule={row['rule']} ref={int(row['ref_id']):03d} r={radius:.4f}",
            flush=True,
        )
        rows.append(sample_unit(row, radius, cfg, run_root, force=bool(args.force)))
        if args.aggregate_every and idx % int(args.aggregate_every) == 0:
            summarize_and_write(run_root, cfg)
    shard_df = pd.DataFrame(rows)
    write_csv(run_root / "05_pool2_pm_sais_sampling" / f"shard{args.shard_index}_unit_summary.csv", shard_df)
    if args.no_final_aggregate:
        existing_units = len(list((run_root / "05_pool2_pm_sais_sampling" / "unit_summaries").rglob("unit_summary.json")))
        status = {
            "status": "sampling_shard_complete",
            "selector": SELECTOR,
            "rules": RULES,
            "radii": RADII,
            "requested_radii_this_invocation": radii_filter,
            "requested_rules_this_invocation": rules_filter,
            "completed_units_visible": int(existing_units),
            "expected_units": int(len(RULES) * 30 * len(RADII)),
        }
    else:
        status = summarize_and_write(run_root, cfg)
    status.update({
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "shard_units_this_invocation": int(len(rows)),
        "elapsed_s": float(time.time() - started),
    })
    write_json(run_root / "05_pool2_pm_sais_sampling" / f"shard{args.shard_index}_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fresh MNIST10 local-support PM-SAIS resampling runner.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--device", default=os.environ.get("MNIST14_DEVICE", ""))
    parser.add_argument("--cpu-threads", type=int, default=int(os.environ.get("MNIST10_CPU_THREADS", "4")))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--aggregate-every", type=int, default=10)
    parser.add_argument("--ref-ids", default="")
    parser.add_argument("--radii", default="")
    parser.add_argument("--rules", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-final-aggregate", action="store_true")
    args = parser.parse_args(argv)
    if args.aggregate_only:
        cfg = configure_pipe(Path(args.run_root))
        status = summarize_and_write(Path(args.run_root), cfg)
        print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
        return 0
    run_units(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
