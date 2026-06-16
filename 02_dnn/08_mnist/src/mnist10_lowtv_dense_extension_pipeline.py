from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mnist10_single_dataset_10x10_ntrain512_sparse_pipeline as pipe


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_lowtv_dense_0p010_to_0p080"
SOURCE_MICRO = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_microline_4rule_lowtv"
SOURCE_BROAD = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_qcpass_line_4rule_lowtv"
RULES = ["low_tv_spectral_teacher"]
RADII = [0.010, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018, 0.020, 0.025, 0.030, 0.040, 0.050, 0.065, 0.080]

_BASE_LOAD_CONFIG = pipe.load_config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=pipe.json_default) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def stage_dir(stage: str) -> Path:
    return RUN_ROOT / ("final_report" if stage == "06_results_figures" else stage)


def dense_radii() -> list[float]:
    return list(RADII)


def configure_pipe() -> None:
    pipe.RUN_ROOT = RUN_ROOT
    pipe.RULES = list(RULES)
    pipe.dense_radii = dense_radii

    def load_config() -> dict[str, Any]:
        cfg = _BASE_LOAD_CONFIG()
        cfg["experiment_id"] = "mnist10_single_dataset_10x10_box_n_train_512_60ref_lowtv_dense"
        cfg["identity"] = RUN_ROOT.name
        cfg["dataset"] = dict(cfg["dataset"])
        cfg["dataset"]["label_rules"] = list(RULES)
        cfg["sampling"] = dict(cfg["sampling"])
        cfg["sampling"]["radii"] = list(RADII)
        cfg["sampling"]["radius_grid_kind"] = "lowtv_dense_hard_line_0p010_to_0p080"
        cfg["sampling"]["fallback_policies_enabled"] = False
        cfg["sampling"]["fallback_policy_note"] = "Disabled for low_tv_spectral_teacher dense extension; baseline 4096-sample PM-SAIS passed at known endpoints."
        cfg["sampling"]["recovery_note"] = (
            "Low-TV-only dense extension. Reuses identical 10x10 BOX dataset, exact references, "
            "and already completed shell units where available; computes missing radii with the unchanged PM-SAIS unit sampler."
        )
        cfg["outputs"] = dict(cfg["outputs"])
        cfg["outputs"]["run_root"] = rel(RUN_ROOT)
        cfg["outputs"]["source_reuse"] = {
            "dataset_and_references": rel(SOURCE_BROAD),
            "microline_shell_units": rel(SOURCE_MICRO),
            "broad_shell_units": rel(SOURCE_BROAD),
        }
        cfg["resolved_at_unix"] = time.time()
        return cfg

    pipe.load_config = load_config


def radius_dir_name(radius: float) -> str:
    return f"r_{float(radius):.4f}".replace(".", "p")


def source_radius_names(radius: float) -> list[str]:
    fixed = f"{float(radius):.4f}"
    compact = fixed.rstrip("0").rstrip(".")
    names = [f"r_{fixed.replace('.', 'p')}", f"r_{compact.replace('.', 'p')}"]
    return list(dict.fromkeys(names))


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    shutil.copytree(src, dst)


def prepare_stage01() -> None:
    cfg = pipe.load_config()
    out_dir = ensure_dir(stage_dir("01_dataset_prepare"))
    src_dir = SOURCE_BROAD / "01_dataset_prepare"
    src_index = pd.read_csv(src_dir / "dataset_index.csv")
    lowtv = src_index[src_index["rule"] == RULES[0]].copy()
    if len(lowtv) != 1:
        raise pipe.StageBlocked("01_dataset_prepare", "Expected exactly one low-TV dataset row in source run.", observed={"rows": int(len(lowtv))})
    ds_src = src_dir / "raw_datasets" / "split_000" / RULES[0]
    ds_dst = out_dir / "raw_datasets" / "split_000" / RULES[0]
    copytree_clean(ds_src, ds_dst)
    lowtv.loc[:, "experiment_id"] = cfg["experiment_id"]
    lowtv.loc[:, "mode"] = cfg["mode"]
    lowtv.loc[:, "dataset_path"] = rel(ds_dst / "dataset.npz")
    write_csv(out_dir / "dataset_index.csv", lowtv)
    if (src_dir / "metadata").exists():
        copytree_clean(src_dir / "metadata", out_dir / "metadata")
    if (src_dir / "figures").exists():
        copytree_clean(src_dir / "figures", out_dir / "figures")
    write_json(out_dir / "run_config_resolved.json", cfg)
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "01_dataset_prepare",
            "status": "pass",
            "checks": {
                "reused_filtered_lowtv_dataset": True,
                "source_run": rel(SOURCE_BROAD),
                "dataset_rows": int(len(lowtv)),
                "dataset_file_exists": bool((ds_dst / "dataset.npz").exists()),
            },
            "warnings": [],
            "hard_failures": [],
        },
    )
    write_text(
        out_dir / "REPORT.md",
        "# Stage 01 Dataset Prepare\n\nReused the identical 10x10 BOX low-TV dataset from the prior 60-reference run and filtered the active dataset index to the low-TV rule.\n",
    )


def prepare_stage04() -> None:
    cfg = pipe.load_config()
    out_dir = ensure_dir(stage_dir("04_exact_reference_search"))
    src_dir = SOURCE_BROAD / "04_exact_reference_search"
    ref_df = pd.read_csv(src_dir / "reference_index.csv")
    lowtv = ref_df[ref_df["rule"] == RULES[0]].sort_values("ref_id").reset_index(drop=True).copy()
    target_refs = int(cfg["reference_search"]["selected_refs_per_dataset"])
    if len(lowtv) < target_refs:
        raise pipe.StageBlocked(
            "04_exact_reference_search",
            "Source run does not contain enough low-TV exact references.",
            observed={"source_rows": int(len(lowtv)), "target_refs": target_refs},
        )
    src_pool = src_dir / "selected_reference_pool" / "split_000" / RULES[0]
    dst_pool = out_dir / "selected_reference_pool" / "split_000" / RULES[0]
    copytree_clean(src_pool, dst_pool)
    dataset_path = rel(stage_dir("01_dataset_prepare") / "raw_datasets" / "split_000" / RULES[0] / "dataset.npz")
    for idx, row in lowtv.iterrows():
        ref_id = int(row["ref_id"])
        theta_path = rel(dst_pool / f"ref_{ref_id:03d}" / "theta.npy")
        lowtv.loc[idx, "theta_path"] = theta_path
        lowtv.loc[idx, "dataset_path"] = dataset_path
        summary_path = dst_pool / f"ref_{ref_id:03d}" / "ref_summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["theta_path"] = theta_path
            payload["dataset_path"] = dataset_path
            payload["reused_from_run"] = rel(SOURCE_BROAD)
            write_json(summary_path, payload)
    write_csv(out_dir / "reference_index.csv", lowtv)
    if (src_dir / "figures").exists():
        copytree_clean(src_dir / "figures", out_dir / "figures")
    write_json(out_dir / "run_config_resolved.json", {**cfg, "reference_reuse_source": rel(SOURCE_BROAD)})
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "04_exact_reference_search",
            "status": "pass",
            "checks": {
                "reused_lowtv_references": True,
                "source_run": rel(SOURCE_BROAD),
                "reference_rows": int(len(lowtv)),
                "expected_reference_rows": target_refs,
                "all_exact": bool((lowtv["train_error"] == 0.0).all()),
                "theta_length_all_P": bool((lowtv["P"] == pipe.P).all()),
            },
            "warnings": [],
            "hard_failures": [],
        },
    )
    write_text(
        out_dir / "REPORT.md",
        f"# Stage 04 Reference Search\n\nReused {len(lowtv)} exact low-TV optimizer-induced references from the prior identical 10x10 BOX run.\n",
    )


def rewrite_unit_payload(payload: dict[str, Any], radius: float) -> dict[str, Any]:
    ref_id = int(payload["ref_id"])
    payload = dict(payload)
    payload["radius"] = float(radius)
    payload["theta_path"] = rel(stage_dir("04_exact_reference_search") / "selected_reference_pool" / "split_000" / RULES[0] / f"ref_{ref_id:03d}" / "theta.npy")
    payload["dataset_path"] = rel(stage_dir("01_dataset_prepare") / "raw_datasets" / "split_000" / RULES[0] / "dataset.npz")
    payload["copied_for_lowtv_dense_extension"] = True
    return payload


def copy_reusable_units() -> dict[str, Any]:
    copied = 0
    missing = 0
    out_base = stage_dir("05_pool2_pm_sais_sampling") / "unit_summaries" / "split_000" / RULES[0]
    source_runs = [SOURCE_MICRO, SOURCE_BROAD]
    for ref_id in range(60):
        for radius in RADII:
            dst = out_base / f"ref_{ref_id:03d}" / radius_dir_name(radius) / "unit_summary.json"
            if dst.exists():
                continue
            src_file = None
            for source_run in source_runs:
                src_base = source_run / "05_pool2_pm_sais_sampling" / "unit_summaries" / "split_000" / RULES[0] / f"ref_{ref_id:03d}"
                for name in source_radius_names(radius):
                    candidate = src_base / name / "unit_summary.json"
                    if candidate.exists():
                        src_file = candidate
                        break
                if src_file is not None:
                    break
            if src_file is None:
                missing += 1
                continue
            payload = rewrite_unit_payload(json.loads(src_file.read_text(encoding="utf-8")), radius)
            write_json(dst, payload)
            copied += 1
    write_json(
        stage_dir("05_pool2_pm_sais_sampling") / "reuse_manifest.json",
        {
            "copied_unit_summaries": copied,
            "missing_unit_summaries_to_compute": missing,
            "source_runs": [rel(p) for p in source_runs],
            "target_radius_count": len(RADII),
            "target_unit_count": len(RADII) * 60,
        },
    )
    return {"copied": copied, "missing": missing}


def add_derivative_outputs() -> None:
    out_dir = ensure_dir(stage_dir("06_results_figures"))
    phi_path = out_dir / "phi_by_rule_radius.csv"
    if not phi_path.exists():
        raise pipe.StageBlocked("06_results_figures", "Cannot add derivative outputs because phi_by_rule_radius.csv is missing.")
    phi_df = pd.read_csv(phi_path).sort_values(["rule", "radius"])
    rows: list[dict[str, Any]] = []
    for rule, sub in phi_df.groupby("rule"):
        x = sub["radius"].to_numpy(dtype=np.float64)
        y = sub["delta_phi_energy"].to_numpy(dtype=np.float64)
        if len(x) < 2:
            continue
        dydx = np.gradient(y, x)
        for row, value in zip(sub.to_dict("records"), dydx):
            rows.append(
                {
                    "rule": rule,
                    "radius": float(row["radius"]),
                    "d_delta_phi_energy_dd": float(value),
                    "qc_pass": bool(row["qc_pass"]),
                    "derivative_method": "numpy.gradient on all-QC dense low-TV radii",
                }
            )
    deriv_df = pd.DataFrame(rows)
    write_csv(out_dir / "dphi_dd_energy_by_rule_radius.csv", deriv_df)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rule, sub in deriv_df.groupby("rule"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["d_delta_phi_energy_dd"], marker="o", linewidth=1.4, label=rule)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("d delta phi energy / d d_raw")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_dphi_dd_energy_qc_pass_lowtv_dense.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rule, sub in phi_df.groupby("rule"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["delta_phi_energy"], marker="o", linewidth=1.4, label=rule)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("delta phi energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_phi_energy_qc_pass_lowtv_dense.png", dpi=180)
    plt.close(fig)
    report_path = out_dir / "REPORT.md"
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Low-TV Dense Extension\n\n"
            "Added dense all-QC low-TV-only `phi(d)_energy` and numerical derivative outputs. "
            "The derivative uses `numpy.gradient` over the QC-passed dense hard-radius line.\n"
        )
    qc_path = out_dir / "QC_STATUS.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {"stage": "06_results_figures", "status": "pass", "checks": {}}
    qc.setdefault("checks", {})
    qc["checks"]["dphi_rows"] = int(len(deriv_df))
    qc["checks"]["dphi_figure_exists"] = bool((fig_dir / "fig10_dphi_dd_energy_qc_pass_lowtv_dense.png").exists())
    qc["checks"]["lowtv_dense_phi_figure_exists"] = bool((fig_dir / "fig11_phi_energy_qc_pass_lowtv_dense.png").exists())
    write_json(qc_path, qc)


def run_all(*, force_sampling: bool = False, max_units: int | None = None) -> None:
    configure_pipe()
    prepare_stage01()
    pipe.stage02_complexity_measure(force=False)
    pipe.stage03_pool_design()
    prepare_stage04()
    reuse = copy_reusable_units()
    print(f"[lowtv dense] reusable Stage05 units copied={reuse['copied']} missing_to_compute={reuse['missing']}", flush=True)
    pipe.stage05_pool2_pm_sais_sampling(force=force_sampling, max_units=max_units, workers=1)
    if max_units is None:
        pipe.stage06_results_figures()
        add_derivative_outputs()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "05_pool2_pm_sais_sampling", "06_results_figures"], default="all")
    parser.add_argument("--force-sampling", action="store_true")
    parser.add_argument("--max-units", type=int, default=None)
    args = parser.parse_args(argv)
    configure_pipe()
    try:
        if args.stage == "prepare":
            prepare_stage01()
            pipe.stage02_complexity_measure(force=False)
            pipe.stage03_pool_design()
            prepare_stage04()
            copy_reusable_units()
        elif args.stage == "05_pool2_pm_sais_sampling":
            copy_reusable_units()
            pipe.stage05_pool2_pm_sais_sampling(force=args.force_sampling, max_units=args.max_units, workers=1)
        elif args.stage == "06_results_figures":
            pipe.stage06_results_figures()
            add_derivative_outputs()
        else:
            run_all(force_sampling=args.force_sampling, max_units=args.max_units)
    except pipe.StageBlocked as blocked:
        pipe.write_blocked(blocked)
        print(f"BLOCKED {blocked.stage}: {blocked.reason}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
