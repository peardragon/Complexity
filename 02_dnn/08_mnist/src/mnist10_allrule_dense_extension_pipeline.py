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
RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_dense_0p010_to_0p080"
SOURCE_LOWTV_DENSE = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_lowtv_dense_0p010_to_0p080"
SOURCE_MICRO = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_microline_4rule_lowtv"
SOURCE_BROAD = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_qcpass_line_4rule_lowtv"
RULES = ["low_tv_spectral_teacher", "real_even_odd", "teacher_nn", "random_label"]
RADII = [0.010, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018, 0.020, 0.025, 0.030, 0.040, 0.050, 0.065, 0.080]

KNOWN_BAD_UNITS = {
    ("random_label", 4, 0.020),
    ("random_label", 7, 0.030),
    ("random_label", 18, 0.030),
    ("real_even_odd", 0, 0.050),
    ("real_even_odd", 27, 0.050),
}

_BASE_LOAD_CONFIG = pipe.load_config
_BASE_FALLBACK_POLICY_FOR = pipe.fallback_policy_for


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


def strong_outlier_policy() -> dict[str, Any]:
    return {
        "name": "rep16_n4096_cess95_mh2_outlier_recompute",
        "replicates": 16,
        "n_samples_each": 4096,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 260,
    }


def broad_random_policy() -> dict[str, Any]:
    return {
        "name": "rep8_n2048_cess95_mh2",
        "replicates": 8,
        "n_samples_each": 2048,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 240,
    }


def configure_pipe() -> None:
    pipe.RUN_ROOT = RUN_ROOT
    pipe.RULES = list(RULES)
    pipe.dense_radii = dense_radii

    def load_config() -> dict[str, Any]:
        cfg = _BASE_LOAD_CONFIG()
        cfg["experiment_id"] = "mnist10_single_dataset_10x10_box_n_train_512_60ref_allrule_dense"
        cfg["identity"] = RUN_ROOT.name
        cfg["dataset"] = dict(cfg["dataset"])
        cfg["dataset"]["label_rules"] = list(RULES)
        cfg["sampling"] = dict(cfg["sampling"])
        cfg["sampling"]["radii"] = list(RADII)
        cfg["sampling"]["radius_grid_kind"] = "allrule_dense_hard_line_0p010_to_0p080"
        cfg["sampling"]["fallback_policies_enabled"] = True
        cfg["sampling"]["fallback_policy_note"] = (
            "Existing fallback policies retained; known bad broad-run outlier units are recomputed "
            "with rep16_n4096_cess95_mh2_outlier_recompute."
        )
        cfg["sampling"]["recovery_note"] = (
            "All-rule dense extension. Reuses identical 10x10 BOX datasets, exact references, "
            "strong microline r0 units, broad endpoint units, and the completed low-TV dense run where available. "
            "Known bad broad outliers are intentionally not copied."
        )
        cfg["outputs"] = dict(cfg["outputs"])
        cfg["outputs"]["run_root"] = rel(RUN_ROOT)
        cfg["outputs"]["source_reuse"] = {
            "lowtv_dense_shell_units": rel(SOURCE_LOWTV_DENSE),
            "microline_shell_units": rel(SOURCE_MICRO),
            "broad_shell_units": rel(SOURCE_BROAD),
            "bad_units_skipped_for_recompute": sorted([f"{r}|ref_{ref:03d}|{rad:.4f}" for r, ref, rad in KNOWN_BAD_UNITS]),
        }
        cfg["resolved_at_unix"] = time.time()
        return cfg

    def fallback_policy_for(rule: str, radius: float, ref_id: int | None = None) -> dict[str, Any] | None:
        key = (str(rule), int(ref_id) if ref_id is not None else -1, round(float(radius), 4))
        if key in {(r, ref, round(rad, 4)) for r, ref, rad in KNOWN_BAD_UNITS}:
            return strong_outlier_policy()
        if ref_id is not None and str(rule) == "random_label" and int(ref_id) >= 5 and round(float(radius), 4) in {0.0200, 0.0300, 0.0500, 0.0800}:
            return broad_random_policy()
        return _BASE_FALLBACK_POLICY_FOR(rule, radius, ref_id)

    pipe.load_config = load_config
    pipe.fallback_policy_for = fallback_policy_for


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
    rows = src_index[src_index["rule"].isin(RULES)].sort_values(["split_id", "rule"]).copy()
    if len(rows) != len(RULES):
        raise pipe.StageBlocked("01_dataset_prepare", "Expected one source dataset row per rule.", observed={"rows": int(len(rows)), "rules": rows["rule"].tolist()})
    for rule in RULES:
        ds_src = src_dir / "raw_datasets" / "split_000" / rule
        ds_dst = out_dir / "raw_datasets" / "split_000" / rule
        copytree_clean(ds_src, ds_dst)
        rows.loc[rows["rule"] == rule, "dataset_path"] = rel(ds_dst / "dataset.npz")
    rows.loc[:, "experiment_id"] = cfg["experiment_id"]
    rows.loc[:, "mode"] = cfg["mode"]
    write_csv(out_dir / "dataset_index.csv", rows)
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
                "reused_filtered_allrule_datasets": True,
                "source_run": rel(SOURCE_BROAD),
                "dataset_rows": int(len(rows)),
                "dataset_files_exist": bool(all((out_dir / "raw_datasets" / "split_000" / rule / "dataset.npz").exists() for rule in RULES)),
            },
            "warnings": [],
            "hard_failures": [],
        },
    )
    write_text(out_dir / "REPORT.md", "# Stage 01 Dataset Prepare\n\nReused the identical 10x10 BOX datasets for all four rules from the prior 60-reference run.\n")


def prepare_stage04() -> None:
    cfg = pipe.load_config()
    out_dir = ensure_dir(stage_dir("04_exact_reference_search"))
    src_dir = SOURCE_BROAD / "04_exact_reference_search"
    ref_df = pd.read_csv(src_dir / "reference_index.csv")
    rows = ref_df[ref_df["rule"].isin(RULES)].sort_values(["rule", "ref_id"]).reset_index(drop=True).copy()
    target_refs = int(cfg["reference_search"]["selected_refs_per_dataset"])
    counts = rows.groupby("rule").size()
    if len(rows) != len(RULES) * target_refs or int(counts.min()) < target_refs:
        raise pipe.StageBlocked(
            "04_exact_reference_search",
            "Source run does not contain enough exact references for every rule.",
            observed={"source_rows": int(len(rows)), "counts": counts.to_dict(), "target_refs": target_refs},
        )
    src_pool = src_dir / "selected_reference_pool" / "split_000"
    dst_pool = out_dir / "selected_reference_pool" / "split_000"
    copytree_clean(src_pool, dst_pool)
    for idx, row in rows.iterrows():
        rule = str(row["rule"])
        ref_id = int(row["ref_id"])
        theta_path = rel(dst_pool / rule / f"ref_{ref_id:03d}" / "theta.npy")
        dataset_path = rel(stage_dir("01_dataset_prepare") / "raw_datasets" / "split_000" / rule / "dataset.npz")
        rows.loc[idx, "theta_path"] = theta_path
        rows.loc[idx, "dataset_path"] = dataset_path
        summary_path = dst_pool / rule / f"ref_{ref_id:03d}" / "ref_summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["theta_path"] = theta_path
            payload["dataset_path"] = dataset_path
            payload["reused_from_run"] = rel(SOURCE_BROAD)
            write_json(summary_path, payload)
    write_csv(out_dir / "reference_index.csv", rows)
    if (src_dir / "figures").exists():
        copytree_clean(src_dir / "figures", out_dir / "figures")
    write_json(out_dir / "run_config_resolved.json", {**cfg, "reference_reuse_source": rel(SOURCE_BROAD)})
    write_json(
        out_dir / "QC_STATUS.json",
        {
            "stage": "04_exact_reference_search",
            "status": "pass",
            "checks": {
                "reused_allrule_references": True,
                "source_run": rel(SOURCE_BROAD),
                "reference_rows": int(len(rows)),
                "expected_reference_rows": len(RULES) * target_refs,
                "min_refs_per_rule": int(counts.min()),
                "all_exact": bool((rows["train_error"] == 0.0).all()),
                "theta_length_all_P": bool((rows["P"] == pipe.P).all()),
            },
            "warnings": [],
            "hard_failures": [],
        },
    )
    write_text(out_dir / "REPORT.md", f"# Stage 04 Reference Search\n\nReused {len(rows)} exact optimizer-induced references across all four rules from the prior identical 10x10 BOX run.\n")


def rewrite_unit_payload(payload: dict[str, Any], rule: str, radius: float) -> dict[str, Any]:
    ref_id = int(payload["ref_id"])
    payload = dict(payload)
    payload["rule"] = str(rule)
    payload["radius"] = float(radius)
    payload["theta_path"] = rel(stage_dir("04_exact_reference_search") / "selected_reference_pool" / "split_000" / rule / f"ref_{ref_id:03d}" / "theta.npy")
    payload["dataset_path"] = rel(stage_dir("01_dataset_prepare") / "raw_datasets" / "split_000" / rule / "dataset.npz")
    payload["copied_for_allrule_dense_extension"] = True
    return payload


def source_runs_for_rule(rule: str) -> list[Path]:
    if rule == "low_tv_spectral_teacher":
        return [SOURCE_LOWTV_DENSE, SOURCE_MICRO, SOURCE_BROAD]
    return [SOURCE_MICRO, SOURCE_BROAD]


def copy_reusable_units() -> dict[str, Any]:
    copied = 0
    missing = 0
    skipped_bad = 0
    for rule in RULES:
        out_base = stage_dir("05_pool2_pm_sais_sampling") / "unit_summaries" / "split_000" / rule
        for ref_id in range(60):
            for radius in RADII:
                dst = out_base / f"ref_{ref_id:03d}" / radius_dir_name(radius) / "unit_summary.json"
                if dst.exists():
                    continue
                if (rule, ref_id, round(float(radius), 4)) in {(r, ref, round(rad, 4)) for r, ref, rad in KNOWN_BAD_UNITS}:
                    skipped_bad += 1
                    missing += 1
                    continue
                src_file = None
                src_run_used = None
                for source_run in source_runs_for_rule(rule):
                    src_base = source_run / "05_pool2_pm_sais_sampling" / "unit_summaries" / "split_000" / rule / f"ref_{ref_id:03d}"
                    for name in source_radius_names(radius):
                        candidate = src_base / name / "unit_summary.json"
                        if candidate.exists():
                            src_file = candidate
                            src_run_used = source_run
                            break
                    if src_file is not None:
                        break
                if src_file is None:
                    missing += 1
                    continue
                payload = rewrite_unit_payload(json.loads(src_file.read_text(encoding="utf-8")), rule, radius)
                payload["copied_from_run"] = rel(src_run_used) if src_run_used is not None else None
                write_json(dst, payload)
                copied += 1
    write_json(
        stage_dir("05_pool2_pm_sais_sampling") / "reuse_manifest.json",
        {
            "copied_unit_summaries": copied,
            "missing_unit_summaries_to_compute": missing,
            "known_bad_units_skipped": skipped_bad,
            "source_runs": [rel(p) for p in [SOURCE_LOWTV_DENSE, SOURCE_MICRO, SOURCE_BROAD]],
            "target_rule_count": len(RULES),
            "target_radius_count": len(RADII),
            "target_unit_count": len(RULES) * len(RADII) * 60,
            "known_bad_units": sorted([f"{r}|ref_{ref:03d}|{rad:.4f}" for r, ref, rad in KNOWN_BAD_UNITS]),
        },
    )
    return {"copied": copied, "missing": missing, "skipped_bad": skipped_bad}


def add_derivative_outputs() -> None:
    out_dir = ensure_dir(stage_dir("06_results_figures"))
    phi_path = out_dir / "phi_by_rule_radius.csv"
    raw_path = out_dir / "phi_by_rule_radius_raw_diagnostic.csv"
    if not phi_path.exists() or not raw_path.exists():
        raise pipe.StageBlocked("06_results_figures", "Cannot add derivative outputs because phi CSV outputs are missing.")
    phi_df = pd.read_csv(phi_path).sort_values(["rule", "radius"])
    raw_df = pd.read_csv(raw_path).sort_values(["rule", "radius"])
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
                    "derivative_method": "numpy.gradient on all-QC dense all-rule radii",
                }
            )
    deriv_df = pd.DataFrame(rows)
    write_csv(out_dir / "dphi_dd_energy_by_rule_radius.csv", deriv_df)
    raw_deriv_rows: list[dict[str, Any]] = []
    for rule, sub in raw_df.groupby("rule"):
        x = sub["radius"].to_numpy(dtype=np.float64)
        y = sub["delta_phi_energy"].to_numpy(dtype=np.float64)
        if len(x) < 2:
            continue
        dydx = np.gradient(y, x)
        for row, value in zip(sub.to_dict("records"), dydx):
            raw_deriv_rows.append(
                {
                    "rule": rule,
                    "radius": float(row["radius"]),
                    "d_delta_phi_energy_dd": float(value),
                    "qc_pass": bool(row["qc_pass"]),
                    "diagnostic_status": row.get("diagnostic_status", "raw_diagnostic"),
                    "derivative_method": "numpy.gradient on raw dense all-rule radii",
                }
            )
    raw_deriv_df = pd.DataFrame(raw_deriv_rows)
    write_csv(out_dir / "dphi_dd_energy_raw_diagnostic_by_rule_radius.csv", raw_deriv_df)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(9, 5))
    for rule, sub in phi_df.groupby("rule"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["delta_phi_energy"], marker="o", linewidth=1.3, label=rule)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("delta phi energy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_phi_energy_qc_pass_allrule_dense.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for rule, sub in deriv_df.groupby("rule"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["d_delta_phi_energy_dd"], marker="o", linewidth=1.3, label=rule)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("d delta phi energy / d d_raw")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_dphi_dd_energy_qc_pass_allrule_dense.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    for ax, rule in zip(axes.ravel(), RULES):
        sub = raw_df[raw_df["rule"] == rule].sort_values("radius")
        if sub.empty:
            continue
        ax.plot(sub["radius"], sub["delta_phi_energy"], linewidth=1.2)
        pass_sub = sub[sub["qc_pass"]]
        fail_sub = sub[~sub["qc_pass"]]
        if not pass_sub.empty:
            ax.scatter(pass_sub["radius"], pass_sub["delta_phi_energy"], s=24, marker="o", label="QC pass")
        if not fail_sub.empty:
            ax.scatter(fail_sub["radius"], fail_sub["delta_phi_energy"], s=38, marker="x", label="QC fail/no-claim")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
        ax.set_title(rule)
        ax.set_xlabel("d_raw")
        ax.set_ylabel("delta phi energy")
        ax.grid(True, linewidth=0.35, alpha=0.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("MNIST10 BOX all-rule dense phi(d)_energy")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig12_phi_energy_allrule_dense_panels.png", dpi=180)
    plt.close(fig)
    with (out_dir / "REPORT.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## All-Rule Dense Extension\n\n"
            "Added all-rule dense `phi(d)_energy` and numerical derivative outputs. "
            "Derivative outputs use `numpy.gradient` over the dense hard-radius line. "
            "If Stage 06 passed, `dphi_dd_energy_by_rule_radius.csv` is based only on QC-passed rows; "
            "raw diagnostics are also retained with QC labels.\n"
        )
    qc_path = out_dir / "QC_STATUS.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {"stage": "06_results_figures", "status": "pass", "checks": {}}
    qc.setdefault("checks", {})
    qc["checks"]["dphi_rows"] = int(len(deriv_df))
    qc["checks"]["raw_dphi_rows"] = int(len(raw_deriv_df))
    qc["checks"]["allrule_dphi_figure_exists"] = bool((fig_dir / "fig10_dphi_dd_energy_qc_pass_allrule_dense.png").exists())
    qc["checks"]["allrule_phi_figure_exists"] = bool((fig_dir / "fig11_phi_energy_qc_pass_allrule_dense.png").exists())
    qc["checks"]["allrule_panel_figure_exists"] = bool((fig_dir / "fig12_phi_energy_allrule_dense_panels.png").exists())
    write_json(qc_path, qc)


def run_all(*, force_sampling: bool = False, max_units: int | None = None) -> None:
    configure_pipe()
    prepare_stage01()
    pipe.stage02_complexity_measure(force=False)
    pipe.stage03_pool_design()
    prepare_stage04()
    reuse = copy_reusable_units()
    print(
        f"[allrule dense] reusable Stage05 units copied={reuse['copied']} "
        f"missing_to_compute={reuse['missing']} known_bad_skipped={reuse['skipped_bad']}",
        flush=True,
    )
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
