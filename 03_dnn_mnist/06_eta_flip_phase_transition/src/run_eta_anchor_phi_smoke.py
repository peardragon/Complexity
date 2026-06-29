#!/usr/bin/env python3
"""Eta label-flip phi(d) smoke using fixed real_even_odd references as anchors."""

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
LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
WINDOWS_ROOT = Path("/home/bjyong/Complexity/windows_project")
MNIST_ROOT = WINDOWS_ROOT / "02_dnn" / "08_mnist"
SOURCE_RUN_ROOT = MNIST_ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
REFERENCE_INDEX = SOURCE_RUN_ROOT / "04_exact_reference_search" / "reference_index.csv"
BASE_DATASET = SOURCE_RUN_ROOT / "01_dataset_prepare" / "raw_datasets" / "split_000" / "real_even_odd" / "dataset.npz"

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

SAMPLING_SRC = LOCAL_ROOT / "04_sampling" / "src"
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


def radius_token(radius: float) -> str:
    return f"r_{float(radius):.4f}".replace(".", "p")


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


def load_base_dataset() -> dict[str, np.ndarray]:
    payload = np.load(BASE_DATASET)
    return {key: payload[key] for key in payload.files}


def materialize_eta_dataset(base: dict[str, np.ndarray], eta: float, seed: int, out_path: Path) -> dict[str, Any]:
    if out_path.exists():
        payload = np.load(out_path)
        y_train = payload["y_train"].astype(np.int8, copy=False)
        y_test = payload["y_test"].astype(np.int8, copy=False)
        return {
            "dataset_path": str(out_path),
            "flip_rate_train": float(np.mean(y_train != base["y_train"])),
            "flip_rate_test": float(np.mean(y_test != base["y_test"])),
            "reused": True,
        }
    rng = np.random.default_rng(int(seed))
    flip_train = rng.random(base["y_train"].shape[0]) < float(eta)
    flip_test = rng.random(base["y_test"].shape[0]) < float(eta)
    arrays = {key: np.asarray(value) for key, value in base.items()}
    arrays["y_train"] = (base["y_train"].astype(np.int8) * np.where(flip_train, -1, 1).astype(np.int8)).astype(np.int8)
    arrays["y_test"] = (base["y_test"].astype(np.int8) * np.where(flip_test, -1, 1).astype(np.int8)).astype(np.int8)
    arrays["eta_flip_mask_train"] = flip_train.astype(np.bool_)
    arrays["eta_flip_mask_test"] = flip_test.astype(np.bool_)
    arrays["eta"] = np.asarray(float(eta), dtype=np.float32)
    arrays["eta_seed"] = np.asarray(int(seed), dtype=np.int64)
    ensure_dir(out_path.parent)
    tmp = out_path.with_name(f"{out_path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out_path)
    return {
        "dataset_path": str(out_path),
        "flip_rate_train": float(np.mean(flip_train)),
        "flip_rate_test": float(np.mean(flip_test)),
        "reused": False,
    }


def configure_sampling(run_root: Path, eta_rules: list[str], radii: list[float], samples: int, cpu_threads: int) -> dict[str, Any]:
    resample.RULES = list(eta_rules)
    resample.RADII = list(radii)
    cfg = resample.configure_pipe(run_root)
    cfg["experiment_id"] = "mnist10_eta_anchor_phi_smoke"
    cfg["identity"] = run_root.name
    cfg["sampling"] = dict(cfg["sampling"])
    cfg["sampling"]["r0"] = float(R0)
    cfg["sampling"]["radii"] = list(radii)
    cfg["sampling"]["samples_per_ref_radius"] = int(samples)
    cfg["sampling"]["fallback_policies_enabled"] = False
    cfg["sampling"]["seed_offset"] = 2026062500
    cfg["sampling"]["note"] = "Smoke only: fixed real_even_odd theta references reused as anchors for eta-flipped labels."
    cfg["outputs"] = dict(cfg.get("outputs", {}))
    cfg["outputs"]["run_root"] = str(run_root)
    cfg["outputs"]["save_unit_samples_npz"] = False
    cfg["compute"] = dict(cfg.get("compute", {}))
    cfg["compute"]["device"] = "cpu"
    cfg["compute"]["chunk_size"] = min(int(cfg["compute"].get("chunk_size", 256)), 256)
    cfg["resource_policy"] = {
        "cpu_limit_target": "use <=35% by limiting thread count",
        "gpu_limit_target": "use <=25%; this smoke uses CPU only",
        "cpu_threads_per_process": int(cpu_threads),
        "device": "cpu",
    }
    return cfg


def recompute_reference_ce(theta_path: str, dataset_path: str, cfg: dict[str, Any]) -> tuple[float, float]:
    theta = np.load(resample.REPO_ROOT / str(theta_path)).astype(np.float64).reshape(1, -1)
    ds = resample.pipe.load_dataset(dataset_path)
    ce, err = resample.pipe.ce_error_batch_torch(theta, ds["X_train"], ds["y_train"], chunk_size=int(cfg["compute"]["chunk_size"]))
    return float(ce[0]), float(err[0])


def build_anchor_rows(
    cfg: dict[str, Any],
    run_root: Path,
    etas: list[float],
    ref_count: int,
    eta_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = load_base_dataset()
    ref_df = pd.read_csv(REFERENCE_INDEX)
    ref_df = ref_df[ref_df["rule"].eq("real_even_odd")].sort_values("ref_id").head(int(ref_count)).copy()
    dataset_rows = []
    anchor_rows = []
    for eta in etas:
        rule = eta_token(eta)
        dataset_path = run_root / "01_dataset_gen" / "split_000" / rule / "rep_000" / "dataset.npz"
        dataset_info = materialize_eta_dataset(base, eta, eta_seed + int(round(eta * 10000)), dataset_path)
        dataset_rows.append({"eta": float(eta), "rule": rule, **dataset_info})
        for _, ref in ref_df.iterrows():
            row = ref.to_dict()
            ce_ref, err_ref = recompute_reference_ce(str(row["theta_path"]), str(dataset_path), cfg)
            row["anchor_rule"] = "real_even_odd"
            row["rule"] = rule
            row["eta"] = float(eta)
            row["dataset_path"] = str(dataset_path)
            row["CE_mean_train"] = ce_ref
            row["CE_sum_train"] = ce_ref * float(base["X_train"].shape[0])
            row["train_error"] = err_ref
            row["resample_seed_offset"] = int(cfg["sampling"]["seed_offset"])
            anchor_rows.append(row)
    datasets = pd.DataFrame(dataset_rows)
    anchors = pd.DataFrame(anchor_rows)
    write_csv(run_root / "01_dataset_gen" / "eta_dataset_manifest.csv", datasets)
    write_csv(run_root / "04_reference_pool" / "anchor_reference_index.csv", anchors)
    return datasets, anchors


def sample_tasks(
    run_root: Path,
    cfg: dict[str, Any],
    anchors: pd.DataFrame,
    radii: list[float],
    max_units: int | None,
    force: bool,
) -> pd.DataFrame:
    tasks = []
    for row in anchors.sort_values(["eta", "ref_id"]).to_dict("records"):
        for radius in radii:
            tasks.append((row, float(radius)))
    if max_units is not None:
        tasks = tasks[: int(max_units)]
    write_csv(
        run_root / "05_pool2_pm_sais_sampling" / "tasks.csv",
        pd.DataFrame(
            [
                {
                    "task_index": idx,
                    "eta": row["eta"],
                    "rule": row["rule"],
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
            f"[eta-anchor-smoke] unit={idx}/{len(tasks)} eta={float(row['eta']):.2f} "
            f"ref={int(row['ref_id']):03d} r={float(radius):.4f}",
            flush=True,
        )
        payload = resample.sample_unit(row, float(radius), cfg, run_root, force=force)
        payload["eta"] = float(row["eta"])
        payload["anchor_rule"] = str(row["anchor_rule"])
        rows.append(payload)
        write_csv(run_root / "05_pool2_pm_sais_sampling" / "sampling_log_latest.csv", pd.DataFrame(rows))
    out = pd.DataFrame(rows)
    out["run_elapsed_s"] = float(time.time() - started)
    write_csv(run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit.csv", out)
    return out


def load_unit_payloads(run_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((run_root / "05_pool2_pm_sais_sampling" / "unit_summaries").rglob("unit_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unit_summary_path"] = str(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize(run_root: Path, cfg: dict[str, Any], etas: list[float], radii: list[float], ref_count: int) -> dict[str, Any]:
    unit_df = load_unit_payloads(run_root)
    if unit_df.empty:
        raise RuntimeError("No unit summaries found for eta anchor smoke.")
    for col in ["eta", "ref_id", "radius", "n_samples", "logZ_inf_full", "ess_fraction", "split_logZ_per_P_diff", "weighted_ce", "weighted_error", "elapsed_s"]:
        if col in unit_df.columns:
            unit_df[col] = pd.to_numeric(unit_df[col], errors="coerce")
    if "eta" not in unit_df.columns:
        unit_df["eta"] = unit_df["rule"].map(eta_from_token)
    else:
        missing_eta = ~np.isfinite(pd.to_numeric(unit_df["eta"], errors="coerce"))
        if missing_eta.any():
            unit_df.loc[missing_eta, "eta"] = unit_df.loc[missing_eta, "rule"].map(eta_from_token)
        unit_df["eta"] = pd.to_numeric(unit_df["eta"], errors="coerce")
    unit_df["phi_energy_raw"] = unit_df["logZ_inf_full"] / P
    key = ["eta", "ref_id"]
    r0_df = (
        unit_df[np.isclose(unit_df["radius"], R0)][key + ["logZ_inf_full"]]
        .drop_duplicates(key, keep="first")
        .rename(columns={"logZ_inf_full": "logZ_r0"})
    )
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined["logZ_inf_full"] - joined["logZ_r0"]) / P
    write_csv(run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv", joined)

    summary = (
        joined.groupby(["eta", "rule", "radius"], as_index=False)
        .agg(
            n_units=("ref_id", "nunique"),
            phi_energy_raw_mean=("phi_energy_raw", "mean"),
            phi_energy_raw_sd=("phi_energy_raw", "std"),
            delta_phi_energy_mean=("delta_phi_energy_unit", "mean"),
            ess_fraction_min=("ess_fraction", "min"),
            ess_fraction_mean=("ess_fraction", "mean"),
            split_logZ_per_P_diff_max=("split_logZ_per_P_diff", "max"),
            weighted_ce_mean=("weighted_ce", "mean"),
            weighted_error_mean=("weighted_error", "mean"),
            elapsed_s_mean=("elapsed_s", "mean"),
        )
        .sort_values(["eta", "radius"])
    )
    write_csv(run_root / "06_results_figures" / "eta_anchor_phi_by_eta_radius.csv", summary)

    derivative_rows = []
    for eta, sub in summary.groupby("eta"):
        sub = sub.sort_values("radius")
        if len(sub) >= 2:
            d1 = np.gradient(sub["phi_energy_raw_mean"].to_numpy(dtype=np.float64), sub["radius"].to_numpy(dtype=np.float64))
        else:
            d1 = np.full(len(sub), np.nan)
        for row, val in zip(sub.to_dict("records"), d1):
            derivative_rows.append({**row, "d_phi_energy_raw_dd": float(val)})
    derivative = pd.DataFrame(derivative_rows)
    write_csv(run_root / "06_results_figures" / "eta_anchor_dphi_dd_by_eta_radius.csv", derivative)

    status = {
        "status": "complete" if int(len(joined)) >= int(len(etas) * ref_count * len(radii)) else "partial",
        "note": "Anchor smoke: eta labels are sampled around fixed real_even_odd references; eta-specific reference search not performed.",
        "etas": etas,
        "radii": radii,
        "ref_count": int(ref_count),
        "samples_per_ref_radius": int(cfg["sampling"]["samples_per_ref_radius"]),
        "completed_units": int(len(joined)),
        "expected_units": int(len(etas) * ref_count * len(radii)),
        "mean_unit_elapsed_s": float(joined["elapsed_s"].mean()),
        "max_unit_elapsed_s": float(joined["elapsed_s"].max()),
        "estimated_20_unit_elapsed_s": float(joined["elapsed_s"].mean() * 20.0),
        "resource_policy": cfg["resource_policy"],
    }
    write_json(run_root / "SAMPLING_STATUS.json", status)
    write_json(run_root / "05_pool2_pm_sais_sampling" / "SAMPLING_STATUS.json", status)
    write_json(run_root / "06_results_figures" / "run_config_resolved.json", {**cfg, "aggregate_status": status})
    write_report(run_root, status, summary)
    plot_summary(run_root, summary)
    return status


def write_report(run_root: Path, status: dict[str, Any], summary: pd.DataFrame) -> None:
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
        "# Eta Anchor Phi Smoke",
        "",
        f"- Status: `{status['status']}`",
        f"- Units: `{status['completed_units']}` / `{status['expected_units']}`",
        f"- Samples per unit: `{status['samples_per_ref_radius']}`",
        f"- Mean unit elapsed seconds: `{status['mean_unit_elapsed_s']:.3f}`",
        f"- Max unit elapsed seconds: `{status['max_unit_elapsed_s']:.3f}`",
        "",
        "This is not an eta-specific reference-search result. It reuses real_even_odd references as fixed anchors,",
        "then evaluates eta-flipped label datasets around those anchors to measure timing and rough phi(d) behavior.",
        "",
        "Primary files:",
        "",
        "- `01_dataset_gen/eta_dataset_manifest.csv`",
        "- `04_reference_pool/anchor_reference_index.csv`",
        "- `05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi.csv`",
        "- `06_results_figures/eta_anchor_phi_by_eta_radius.csv`",
        "- `06_results_figures/eta_anchor_dphi_dd_by_eta_radius.csv`",
        "- `06_results_figures/fig01_eta_anchor_phi_energy.png`",
    ]
    if not summary.empty:
        preview = summary[["eta", "radius", "n_units", "phi_energy_raw_mean", "weighted_ce_mean", "elapsed_s_mean"]].copy()
        lines.extend(["", "Summary preview:", "", markdown_table(preview)])
    (run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(run_root: Path, summary: pd.DataFrame) -> None:
    fig_dir = ensure_dir(run_root / "06_results_figures")
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=180)
    if summary["radius"].nunique() > 1:
        for eta, sub in summary.groupby("eta"):
            sub = sub.sort_values("radius")
            ax.plot(sub["radius"], sub["phi_energy_raw_mean"], marker="o", lw=1.7, ms=3.2, label=f"eta={eta:.2f}")
        ax.set_xlabel("radius d")
    else:
        sub = summary.sort_values("eta")
        ax.plot(sub["eta"], sub["phi_energy_raw_mean"], marker="o", lw=1.7, ms=3.2, color="#2451a6")
        ax.set_xlabel("eta")
    ax.set_ylabel("phi(d) energy raw")
    ax.set_title("Eta anchor phi smoke")
    ax.grid(True, color="0.88", linewidth=0.7)
    if summary["eta"].nunique() > 1 and summary["radius"].nunique() > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_eta_anchor_phi_energy.png")
    plt.close(fig)

    if summary["radius"].nunique() > 1:
        zoom = summary[summary["radius"] >= 0.75].copy().sort_values(["eta", "radius"])
        if not zoom.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
            for eta, sub in zoom.groupby("eta"):
                ax.plot(sub["radius"], sub["phi_energy_raw_mean"], marker="o", lw=1.7, ms=3.2, label=f"eta={eta:.2f}")
            ax.set_xlabel("radius d")
            ax.set_ylabel("phi(d) energy raw")
            ax.set_title("Eta anchor phi smoke: d near 1")
            ax.grid(True, color="0.88", linewidth=0.7)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "fig02_eta_anchor_phi_energy_d1_zoom.png")
            plt.close(fig)

            finite_delta = zoom[np.isfinite(zoom["delta_phi_energy_mean"])]
            if not finite_delta.empty:
                fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
                for eta, sub in finite_delta.groupby("eta"):
                    ax.plot(sub["radius"], sub["delta_phi_energy_mean"], marker="o", lw=1.7, ms=3.2, label=f"eta={eta:.2f}")
                ax.axhline(0.0, color="0.25", lw=0.9)
                ax.set_xlabel("radius d")
                ax.set_ylabel("delta phi energy from r=0.1")
                ax.set_title("Eta anchor delta phi smoke: d near 1")
                ax.grid(True, color="0.88", linewidth=0.7)
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                fig.savefig(fig_dir / "fig03_eta_anchor_delta_phi_energy_d1_zoom.png")
                plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="eta_anchor_phi_smoke_cpu35_gpu0")
    parser.add_argument("--etas", default="0.0")
    parser.add_argument("--radii", default="1.0")
    parser.add_argument("--ref-count", type=int, default=1)
    parser.add_argument("--samples-per-ref-radius", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--eta-seed", type=int, default=2026062500)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    cpu_threads = max(1, min(8, int(args.cpu_threads)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS", "TORCH_NUM_INTEROP_THREADS"):
        os.environ[name] = str(cpu_threads)
    try:
        import torch

        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
    except Exception:
        pass

    etas = parse_float_list(args.etas)
    radii = parse_float_list(args.radii)
    eta_rules = [eta_token(eta) for eta in etas]
    run_root = ensure_dir(STAGE_ROOT / "raw_outputs" / args.run_name)
    ensure_dir(STAGE_ROOT / ".cache" / "matplotlib")
    cfg = configure_sampling(run_root, eta_rules, radii, int(args.samples_per_ref_radius), cpu_threads)
    write_json(run_root / "run_config_resolved.json", cfg)

    if not args.aggregate_only:
        _, anchors = build_anchor_rows(cfg, run_root, etas, int(args.ref_count), int(args.eta_seed))
        sample_tasks(run_root, cfg, anchors, radii, args.max_units, bool(args.force))

    status = summarize(run_root, cfg, etas, radii, int(args.ref_count))
    print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
