from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
RUN_ROOT = LOCAL_ROOT / "04_sampling" / "raw_outputs" / "refpool1024_all_radii_90ref"
OUT_ROOT = RUN_ROOT / "06_results_figures" / "replicate_stability_probe"
P = 2461.0
SPLIT_GATE = 0.004
RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sample_refpool1024_all_radii as refpool  # noqa: E402
import resample_mnist10_local_support as resample  # noqa: E402


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_rules(text: str) -> list[str]:
    if not text.strip():
        return list(RULES)
    rules = [part.strip() for part in text.split(",") if part.strip()]
    missing = [rule for rule in rules if rule not in RULES]
    if missing:
        raise ValueError(f"unknown rules: {missing}")
    return rules


def configure_runtime(cpu_threads: int, device: str) -> None:
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[name] = str(cpu_threads)
    os.environ["MNIST14_DEVICE"] = str(device)
    try:
        import torch

        torch.set_num_threads(int(cpu_threads))
        torch.set_num_interop_threads(int(cpu_threads))
    except Exception:
        pass


def base_config(run_root: Path, samples_per_ref_radius: int, chunk_size: int, device: str) -> dict[str, Any]:
    cfg = resample.configure_pipe(run_root)
    cfg["sampling"] = dict(cfg.get("sampling", {}))
    cfg["sampling"]["samples_per_ref_radius"] = int(samples_per_ref_radius)
    cfg["sampling"]["fallback_policies_enabled"] = False
    cfg["compute"] = dict(cfg.get("compute", {}))
    cfg["compute"]["chunk_size"] = int(chunk_size)
    cfg["compute"]["device"] = str(device)
    cfg["outputs"] = dict(cfg.get("outputs", {}))
    cfg["outputs"]["save_unit_samples_npz"] = False
    cfg["replicate_probe"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_root": str(RUN_ROOT),
        "samples_per_ref_radius": int(samples_per_ref_radius),
    }
    return cfg


def load_source_units() -> pd.DataFrame:
    path = RUN_ROOT / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit.csv"
    df = pd.read_csv(path)
    df["rule"] = df["rule"].astype(str)
    df["ref_id"] = df["ref_id"].astype(int)
    df["radius"] = df["radius"].astype(float)
    return df


def select_probe_units(units: pd.DataFrame, pool: pd.DataFrame, rules: list[str], radii: list[float], refs_per_cell: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    refs_per_cell = max(1, int(refs_per_cell))
    for rule in rules:
        for radius in radii:
            sub = units[units["rule"].eq(rule) & np.isclose(units["radius"], float(radius))].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("split_logZ_per_P_diff").reset_index(drop=True)
            if refs_per_cell == 1:
                indices = [len(sub) // 2]
                labels = ["median_split"]
            elif refs_per_cell == 2:
                indices = [0, len(sub) - 1]
                labels = ["low_split", "high_split"]
            else:
                indices = np.linspace(0, len(sub) - 1, refs_per_cell).round().astype(int).tolist()
                labels = [f"split_rank_{idx + 1}_of_{refs_per_cell}" for idx in range(refs_per_cell)]
                labels[0] = "low_split"
                labels[-1] = "high_split"
                if refs_per_cell >= 3:
                    labels[refs_per_cell // 2] = "median_split"
            for label, idx in zip(labels, indices):
                unit = sub.iloc[int(idx)].to_dict()
                ref_match = pool[pool["rule"].eq(rule) & pool["ref_id"].eq(int(unit["ref_id"]))]
                if ref_match.empty:
                    continue
                row = ref_match.iloc[0].to_dict()
                row["probe_label"] = label
                row["source_split_logZ_per_P_diff"] = float(unit["split_logZ_per_P_diff"])
                row["source_logZ_inf_full"] = float(unit["logZ_inf_full"])
                row["radius"] = float(radius)
                rows.append(row)
    out = pd.DataFrame(rows).drop_duplicates(["rule", "ref_id", "radius", "probe_label"])
    out["rule"] = out["rule"].astype(str)
    out["ref_id"] = out["ref_id"].astype(int)
    out["radius"] = out["radius"].astype(float)
    return out.sort_values(["rule", "radius", "probe_label", "ref_id"]).reset_index(drop=True)


def run_one(row: dict[str, Any], replicate_idx: int, cfg: dict[str, Any], base_seed: int) -> dict[str, Any]:
    rule = str(row["rule"])
    ref_id = int(row["ref_id"])
    radius = float(row["radius"])
    theta_path = resample.REPO_ROOT / str(row["theta_path"])
    dataset_path = str(row["dataset_path"])
    ds = resample.pipe.load_dataset(dataset_path)
    theta_ref = np.load(theta_path).astype(np.float64).reshape(-1)
    seed = (
        int(base_seed)
        + int(replicate_idx) * 10_000_000
        + RULES.index(rule) * 1_000_000
        + ref_id * 1000
        + int(round(radius * 10000))
    )
    started = time.time()
    smc = resample.pipe.run_smc_split(
        theta_ref,
        ds,
        radius,
        int(cfg["sampling"]["samples_per_ref_radius"]),
        float(cfg["sampling"].get("lambda_reg", 1.0)),
        seed,
        cfg,
        float(row["CE_mean_train"]),
    )
    smc.pop("_samples_npz", None)
    return {
        "rule": rule,
        "ref_id": ref_id,
        "radius": radius,
        "probe_label": str(row["probe_label"]),
        "replicate_idx": int(replicate_idx),
        "seed": int(seed),
        "n_samples": int(cfg["sampling"]["samples_per_ref_radius"]),
        "source_split_logZ_per_P_diff": float(row["source_split_logZ_per_P_diff"]),
        "source_logZ_inf_full": float(row["source_logZ_inf_full"]),
        "elapsed_s": float(time.time() - started),
        "finite": bool(np.isfinite(smc["logZ_inf_full"])),
        "logZ_inf_full": float(smc["logZ_inf_full"]),
        "phi_energy": float(smc["logZ_inf_full"]) / P,
        "split_logZ_per_P_diff": float(smc["split_logZ_per_P_diff"]),
        "ess_fraction": float(smc["ess_fraction"]),
        "weighted_ce": float(smc["weighted_ce"]),
        "weighted_error": float(smc["weighted_error"]),
        "smc_step_count": int(smc["smc_step_count"]),
        "smc_total_step_count": int(smc["smc_total_step_count"]),
        "smc_mean_mh_acceptance": float(smc["smc_mean_mh_acceptance"]),
    }


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()
    unit_summary = (
        results.groupby(["rule", "ref_id", "radius", "probe_label"])
        .agg(
            replicate_count=("replicate_idx", "nunique"),
            logZ_inf_full_mean=("logZ_inf_full", "mean"),
            logZ_inf_full_sd=("logZ_inf_full", "std"),
            phi_energy_mean=("phi_energy", "mean"),
            phi_energy_sd=("phi_energy", "std"),
            split_q50=("split_logZ_per_P_diff", "median"),
            split_q95=("split_logZ_per_P_diff", lambda s: float(np.quantile(s, 0.95))),
            split_max=("split_logZ_per_P_diff", "max"),
            split_fail_rate=("split_logZ_per_P_diff", lambda s: float(np.mean(np.asarray(s) > SPLIT_GATE))),
            ess_min=("ess_fraction", "min"),
            elapsed_s_sum=("elapsed_s", "sum"),
            source_split_logZ_per_P_diff=("source_split_logZ_per_P_diff", "first"),
            source_logZ_inf_full=("source_logZ_inf_full", "first"),
        )
        .reset_index()
        .sort_values(["rule", "radius", "probe_label", "ref_id"])
    )
    rule_radius = (
        unit_summary.groupby(["rule", "radius"])
        .agg(
            probe_unit_count=("ref_id", "nunique"),
            mean_replicate_phi_sd=("phi_energy_sd", "mean"),
            max_replicate_phi_sd=("phi_energy_sd", "max"),
            mean_split_q95=("split_q95", "mean"),
            max_split_q95=("split_q95", "max"),
            mean_split_fail_rate=("split_fail_rate", "mean"),
            max_split_fail_rate=("split_fail_rate", "max"),
        )
        .reset_index()
        .sort_values(["rule", "radius"])
    )
    return unit_summary, rule_radius


def write_report(out_root: Path, tasks: pd.DataFrame, results: pd.DataFrame, unit_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Replicate Stability Probe",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Design",
        "",
        "- Runs fresh independent SMC replicates for selected `rule/ref/radius` units.",
        "- This is the appropriate follow-up when arbitrary random re-split logZ cannot be reconstructed from saved normalized sample weights.",
        f"- Replicates per probe unit: `{int(args.replicates)}`.",
        f"- Samples per replicate: `{int(args.samples_per_ref_radius)}`.",
        f"- Probe radii: `{args.radii}`.",
        f"- Refs per rule/radius cell: `{int(args.refs_per_cell)}` selected by source split rank.",
        "",
        "## Current Run",
        "",
        f"- Probe units selected: `{len(tasks)}`.",
        f"- Replicate rows completed: `{len(results)}`.",
        f"- Total elapsed SMC seconds: `{float(results['elapsed_s'].sum()) if len(results) else 0.0:.1f}`.",
        "",
    ]
    if len(unit_summary):
        lines.extend(["## Unit Summary", "", "| rule | radius | ref | label | phi sd | split q95 | split fail rate |", "| --- | ---: | ---: | --- | ---: | ---: | ---: |"])
        for _, row in unit_summary.iterrows():
            phi_sd = float(row["phi_energy_sd"]) if np.isfinite(row["phi_energy_sd"]) else 0.0
            lines.append(
                f"| {row['rule']} | {float(row['radius']):.1f} | {int(row['ref_id'])} | {row['probe_label']} | "
                f"{phi_sd:.6g} | {float(row['split_q95']):.6g} | {float(row['split_fail_rate']):.3f} |"
            )
    lines.extend(["", "## Outputs", "", "- `probe_tasks.csv`", "- `replicate_unit_results.csv`", "- `replicate_unit_summary.csv`", "- `replicate_rule_radius_summary.csv`", ""])
    (out_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run independent SMC replicate stability probes for refpool1024 units.")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--radii", default="0.3,1.0,2.5")
    parser.add_argument("--rules", default="")
    parser.add_argument("--refs-per-cell", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--samples-per-ref-radius", type=int, default=1024)
    parser.add_argument("--target-refs", type=int, default=90)
    parser.add_argument("--seed", type=int, default=2026061800)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-probe-units", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    ensure_dir(out_root)
    configure_runtime(int(args.cpu_threads), str(args.device))
    cfg = base_config(out_root, int(args.samples_per_ref_radius), int(args.chunk_size), str(args.device))
    write_json(out_root / "run_config_resolved.json", {"args": vars(args), "cfg": cfg.get("replicate_probe", {})})

    units = load_source_units()
    pool = refpool.load_reference_pool(Path(refpool.DEFAULT_EXTRA_REFERENCE_RUN_ROOT), int(args.target_refs))
    tasks = select_probe_units(units, pool, parse_csv_rules(args.rules), parse_csv_floats(args.radii), int(args.refs_per_cell))
    if args.max_probe_units is not None:
        tasks = tasks.head(int(args.max_probe_units)).copy()
    write_csv(out_root / "probe_tasks.csv", tasks)

    result_path = out_root / "replicate_unit_results.csv"
    if result_path.exists() and not args.force:
        results = pd.read_csv(result_path)
    else:
        rows: list[dict[str, Any]] = []
        for task_idx, row in enumerate(tasks.to_dict("records"), start=1):
            for replicate_idx in range(int(args.replicates)):
                print(
                    f"[replicate_probe] task={task_idx}/{len(tasks)} rep={replicate_idx + 1}/{int(args.replicates)} "
                    f"rule={row['rule']} ref={int(row['ref_id']):03d} r={float(row['radius']):.1f} label={row['probe_label']}",
                    flush=True,
                )
                rows.append(run_one(row, replicate_idx, cfg, int(args.seed)))
                write_csv(result_path, pd.DataFrame(rows))
        results = pd.DataFrame(rows)

    unit_summary, rule_radius = summarize(results)
    write_csv(out_root / "replicate_unit_summary.csv", unit_summary)
    write_csv(out_root / "replicate_rule_radius_summary.csv", rule_radius)
    write_report(out_root, tasks, results, unit_summary, args)
    print(f"wrote replicate probe to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
