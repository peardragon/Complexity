from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mnist10_allrule_dense_extension_pipeline as base


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
SOURCE_ALLRULE_DENSE = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_dense_0p010_to_0p080"
RULES = ["low_tv_spectral_teacher", "real_even_odd", "teacher_nn", "random_label"]
RADII = [0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.150, 0.200, 0.300, 0.450, 0.650, 0.850, 1.000, 1.250, 1.500, 1.750, 2.000, 2.250, 2.500]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=base.pipe.json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def dense_radii() -> list[float]:
    return list(RADII)


def rep8_n2048_policy() -> dict[str, Any]:
    return {
        "name": "sparse_rep8_n2048_cess95_mh2",
        "replicates": 8,
        "n_samples_each": 2048,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 240,
    }


def rep16_n2048_policy() -> dict[str, Any]:
    return {
        "name": "sparse_rep16_n2048_cess95_mh2",
        "replicates": 16,
        "n_samples_each": 2048,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 260,
    }


def rep16_n4096_policy() -> dict[str, Any]:
    return {
        "name": "sparse_rep16_n4096_cess95_mh2",
        "replicates": 16,
        "n_samples_each": 4096,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 300,
    }


def rep32_n4096_policy() -> dict[str, Any]:
    return {
        "name": "sparse_rep32_n4096_cess95_mh2",
        "replicates": 32,
        "n_samples_each": 4096,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 320,
    }


def rep64_n4096_policy() -> dict[str, Any]:
    return {
        "name": "sparse_rep64_n4096_cess95_mh2",
        "replicates": 64,
        "n_samples_each": 4096,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 2,
        "move_kappa_factor": 80.0,
        "max_steps": 340,
    }


def teacher_kernel_move160_policy() -> dict[str, Any]:
    return {
        "name": "sparse_teacher_rep16_n1024_cess95_mh4_move160",
        "replicates": 16,
        "n_samples_each": 1024,
        "target_cess_fraction": 0.95,
        "mh_sweeps": 4,
        "move_kappa_factor": 160.0,
        "max_steps": 280,
    }


def sparse_policy_for(rule: str, radius: float) -> dict[str, Any] | None:
    r = round(float(radius), 4)
    extreme_rep64 = {
        "low_tv_spectral_teacher": {1.0000, 1.2500, 1.5000, 1.7500, 2.0000, 2.2500, 2.5000},
    }
    ultrahard_rep32 = {
        "low_tv_spectral_teacher": set(),
    }
    superhard_rep16 = {
        "low_tv_spectral_teacher": {1.0000, 1.2500, 2.0000, 2.2500, 2.5000},
    }
    hard_rep16 = {
        "low_tv_spectral_teacher": {0.6500, 0.8500},
        "real_even_odd": {2.0000, 2.2500, 2.5000},
        "random_label": {0.2000, 0.4500},
    }
    near_rep8 = {
        "low_tv_spectral_teacher": {0.4500},
        "real_even_odd": {0.4500, 0.8500, 1.2500, 1.5000, 1.7500},
        "random_label": {0.1200, 0.3000, 0.6500, 0.8500, 1.0000, 1.5000, 2.0000, 2.5000},
        "teacher_nn": {0.3000, 2.2500},
    }
    if r in extreme_rep64.get(str(rule), set()):
        return rep64_n4096_policy()
    if r in ultrahard_rep32.get(str(rule), set()):
        return rep32_n4096_policy()
    if r in superhard_rep16.get(str(rule), set()):
        return rep16_n4096_policy()
    if r in hard_rep16.get(str(rule), set()):
        return rep16_n2048_policy()
    if str(rule) == "teacher_nn" and r == 2.5000:
        return teacher_kernel_move160_policy()
    if r in near_rep8.get(str(rule), set()):
        return rep8_n2048_policy()
    return None


def sparse_sample_stage05_unit_worker(args: tuple[dict[str, Any], float, dict[str, Any], bool]) -> dict[str, Any]:
    configure_pipe()
    row, radius, cfg, force = args
    return base.pipe.sample_stage05_unit(row, radius, cfg, force=force)


def configure_pipe() -> None:
    base.RUN_ROOT = RUN_ROOT
    base.SOURCE_LOWTV_DENSE = SOURCE_ALLRULE_DENSE
    base.SOURCE_BROAD = SOURCE_ALLRULE_DENSE
    base.RULES = list(RULES)
    base.RADII = list(RADII)
    base.KNOWN_BAD_UNITS = set()
    base.pipe.RUN_ROOT = RUN_ROOT
    base.pipe.RULES = list(RULES)
    base.pipe.dense_radii = dense_radii
    base.dense_radii = dense_radii

    def load_config() -> dict[str, Any]:
        cfg = base._BASE_LOAD_CONFIG()
        cfg["experiment_id"] = "mnist10_single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_to_2p50"
        cfg["identity"] = RUN_ROOT.name
        cfg["dataset"] = dict(cfg["dataset"])
        cfg["dataset"]["label_rules"] = list(RULES)
        cfg["sampling"] = dict(cfg["sampling"])
        cfg["sampling"]["radii"] = list(RADII)
        cfg["sampling"]["radius_grid_kind"] = "allrule_sparse_hard_line_0p010_to_2p500"
        cfg["sampling"]["fallback_policies_enabled"] = True
        cfg["sampling"]["fallback_policy_note"] = (
            "Sparse 2.50 extension starts from the all-rule dense QC-pass run. "
            "Overlapping dense radii are reused. Larger sparse radii use the unchanged hard-shell PM-SAIS "
            "estimator with targeted replicate/particle fallback policies selected from the one-reference "
            "sparse pilot; remaining failed rule/radius rows are no-claim until targeted recomputation passes."
        )
        cfg["sampling"]["recovery_note"] = (
            "All-rule sparse-to-2.50 extension. Reuses identical 10x10 BOX datasets, exact references, "
            "and overlapping QC-passed dense shell units from the all-rule dense run. "
            "New sparse radii beyond 0.08 are computed with the unchanged hard-shell PM-SAIS skeleton."
        )
        cfg["outputs"] = dict(cfg["outputs"])
        cfg["outputs"]["run_root"] = rel(RUN_ROOT)
        cfg["outputs"]["source_reuse"] = {
            "allrule_dense_source": rel(SOURCE_ALLRULE_DENSE),
            "overlap_radii_expected_reuse": [0.010, 0.020, 0.030, 0.050, 0.080],
            "new_sparse_radii": [r for r in RADII if r not in {0.010, 0.020, 0.030, 0.050, 0.080}],
        }
        cfg["resolved_at_unix"] = time.time()
        return cfg

    def fallback_policy_for(rule: str, radius: float, ref_id: int | None = None) -> dict[str, Any] | None:
        sparse_policy = sparse_policy_for(rule, radius)
        if sparse_policy is not None:
            return sparse_policy
        return base._BASE_FALLBACK_POLICY_FOR(rule, radius, ref_id)

    base.pipe.load_config = load_config
    base.pipe.fallback_policy_for = fallback_policy_for
    base.pipe.sample_stage05_unit_worker = sparse_sample_stage05_unit_worker


def stage05_pilot_one_ref_per_rule(*, force: bool = False) -> None:
    configure_pipe()
    cfg = base.pipe.load_config()
    out_dir = ensure_dir(RUN_ROOT / "05_pool2_pm_sais_sampling" / "pilot_sparse_one_ref_per_rule")
    ref_path = RUN_ROOT / "04_exact_reference_search" / "reference_index.csv"
    if not ref_path.exists():
        raise base.pipe.StageBlocked("05_pool2_pm_sais_sampling", "Reference index missing for sparse pilot.", observed={"missing": rel(ref_path)})
    ref_df = pd.read_csv(ref_path).groupby("rule").head(1).reset_index(drop=True)
    rows = []
    started = time.time()
    for row in ref_df.to_dict("records"):
        for radius in RADII:
            print(f"[sparse pilot] rule={row['rule']} ref={row['ref_id']} r={radius:.4f}", flush=True)
            rows.append(base.pipe.sample_stage05_unit(row, float(radius), cfg, force=force))
    df = pd.DataFrame(rows)
    write_csv(out_dir / "pilot_unit_summary.csv", df)
    new_df = df[~df.get("reused", False).astype(bool)] if "reused" in df.columns else df
    estimate_units = int(len(RULES) * int(cfg["reference_search"]["selected_refs_per_dataset"]) * len([r for r in RADII if r not in {0.010, 0.020, 0.030, 0.050, 0.080}]))
    mean_new_s = float(new_df["elapsed_s"].mean()) if len(new_df) else 0.0
    write_json(
        out_dir / "runtime_estimate.json",
        {
            "pilot_units": int(len(df)),
            "new_pilot_units": int(len(new_df)),
            "pilot_elapsed_s": float(time.time() - started),
            "mean_new_unit_elapsed_s": mean_new_s,
            "estimated_new_full_units": estimate_units,
            "estimated_new_full_sampling_hours": float(mean_new_s * estimate_units / 3600.0),
            "max_split_logZ_per_P_diff": float(df["split_logZ_per_P_diff"].max()),
            "q05_ess_fraction": float(np.quantile(df["ess_fraction"], 0.05)),
        },
    )
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rule, sub in df.groupby("rule"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["split_logZ_per_P_diff"], marker="o", linewidth=1.0, label=rule)
    ax.axhline(float(cfg["qc"]["max_split_logZ_per_P_diff"]), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("split logZ/P diff")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_sparse_pilot_split_logz.png", dpi=160)
    plt.close(fig)


def add_sparse_report_note() -> None:
    out_dir = RUN_ROOT / "final_report"
    report_path = out_dir / "REPORT.md"
    if not report_path.exists():
        return
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Sparse 2.50 Extension\n\n"
            "This run extends the all-rule 10x10 BOX hard-shell PM-SAIS line to sparse radii through d_raw=2.50. "
            "Overlapping radii are copied from the all-rule dense QC-pass source run; larger sparse radii are first-pass PM-SAIS computations. "
            "Any failed Stage 05 QC row is no-claim and should be targeted by a stronger recomputation pass.\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "pilot_one_ref_per_rule", "05_pool2_pm_sais_sampling", "06_results_figures"], default="all")
    parser.add_argument("--force-sampling", action="store_true")
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    configure_pipe()
    try:
        if args.stage == "prepare":
            base.prepare_stage01()
            base.pipe.stage02_complexity_measure(force=False)
            base.pipe.stage03_pool_design()
            base.prepare_stage04()
            base.copy_reusable_units()
        elif args.stage == "pilot_one_ref_per_rule":
            base.prepare_stage01()
            base.pipe.stage02_complexity_measure(force=False)
            base.pipe.stage03_pool_design()
            base.prepare_stage04()
            base.copy_reusable_units()
            stage05_pilot_one_ref_per_rule(force=args.force_sampling)
        elif args.stage == "05_pool2_pm_sais_sampling":
            base.copy_reusable_units()
            base.pipe.stage05_pool2_pm_sais_sampling(force=args.force_sampling, max_units=args.max_units, workers=args.workers)
        elif args.stage == "06_results_figures":
            base.pipe.stage06_results_figures()
            base.add_derivative_outputs()
            add_sparse_report_note()
        else:
            base.prepare_stage01()
            base.pipe.stage02_complexity_measure(force=False)
            base.pipe.stage03_pool_design()
            base.prepare_stage04()
            reuse = base.copy_reusable_units()
            print(
                f"[sparse 2.50] reusable Stage05 units copied={reuse['copied']} missing_to_compute={reuse['missing']}",
                flush=True,
            )
            base.pipe.stage05_pool2_pm_sais_sampling(force=args.force_sampling, max_units=args.max_units, workers=args.workers)
            if args.max_units is None:
                base.pipe.stage06_results_figures()
                base.add_derivative_outputs()
                add_sparse_report_note()
    except base.pipe.StageBlocked as blocked:
        base.pipe.write_blocked(blocked)
        print(f"BLOCKED {blocked.stage}: {blocked.reason}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
