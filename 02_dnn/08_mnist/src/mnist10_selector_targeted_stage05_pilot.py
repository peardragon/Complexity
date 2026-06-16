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

import mnist10_allrule_sparse_to_2p50_pipeline as sparse


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
ANALYSIS_ROOT = SOURCE_RUN_ROOT / "07_reference_family_analysis"
PILOT_RUN_ROOT = ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_ref30_selector_targeted_pilot"
DEFAULT_SELECTOR = "l2_min_norm_ref30"
DEFAULT_RULE = "low_tv_spectral_teacher"


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
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def configure_pilot_pipe() -> dict[str, Any]:
    sparse.RUN_ROOT = PILOT_RUN_ROOT
    sparse.configure_pipe()
    cfg = sparse.base.pipe.load_config()
    cfg["experiment_id"] = "mnist10_l2_min_norm_ref30_selector_targeted_stage05_pilot"
    cfg["identity"] = PILOT_RUN_ROOT.name
    cfg["outputs"] = dict(cfg["outputs"])
    cfg["outputs"]["run_root"] = rel(PILOT_RUN_ROOT)
    cfg["outputs"]["source_run_root"] = rel(SOURCE_RUN_ROOT)
    cfg["outputs"]["analysis_root"] = rel(ANALYSIS_ROOT)
    return cfg


def existing_source_unit_index() -> set[tuple[str, int, float]]:
    unit_path = ANALYSIS_ROOT / "unit_summary_long.csv"
    if not unit_path.exists():
        return set()
    df = pd.read_csv(unit_path, usecols=["rule", "ref_id", "radius"])
    return {(str(row.rule), int(row.ref_id), round(float(row.radius), 4)) for row in df.itertuples()}


def load_selector_tasks(selector: str, rule: str, radii: list[float], *, only_missing_source: bool, ref_ids: list[int] | None = None) -> list[tuple[dict[str, Any], float]]:
    membership_path = ANALYSIS_ROOT / "selector_membership.csv"
    ref_path = SOURCE_RUN_ROOT / "04_exact_reference_search" / "reference_index.csv"
    if not membership_path.exists():
        raise FileNotFoundError(membership_path)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    membership = pd.read_csv(membership_path)
    refs = pd.read_csv(ref_path)
    selected = membership[(membership["selector"] == selector) & (membership["rule"] == rule)]["ref_id"].astype(int).tolist()
    if not selected:
        raise ValueError(f"No selected refs for selector={selector} rule={rule}")
    if ref_ids is not None:
        ref_set = set(int(r) for r in ref_ids)
        selected = [int(r) for r in selected if int(r) in ref_set]
        if not selected:
            raise ValueError(f"Requested ref_ids are not in selector={selector} rule={rule}: {sorted(ref_set)}")
    source_index = existing_source_unit_index() if only_missing_source else set()
    rows = refs[(refs["rule"] == rule) & (refs["ref_id"].astype(int).isin(selected))].copy()
    rows["ref_id"] = rows["ref_id"].astype(int)
    rows = rows.sort_values("ref_id")
    tasks: list[tuple[dict[str, Any], float]] = []
    for row in rows.to_dict("records"):
        for radius in radii:
            key = (str(rule), int(row["ref_id"]), round(float(radius), 4))
            if only_missing_source and key in source_index:
                continue
            tasks.append((row, float(radius)))
    return tasks


def summarize_pilot(df: pd.DataFrame, selector: str, rule: str, radii: list[float], out_dir: Path, started: float) -> dict[str, Any]:
    if df.empty:
        summary = {
            "selector": selector,
            "rule": rule,
            "requested_radii": radii,
            "unit_rows": 0,
            "elapsed_s": float(time.time() - started),
            "decision": "no_units_to_run",
        }
        write_json(out_dir / "pilot_summary.json", summary)
        return summary
    split_gate = 0.004
    df["split_gate_pass"] = pd.to_numeric(df["split_logZ_per_P_diff"], errors="coerce") <= split_gate
    df["ess_gate_pass"] = pd.to_numeric(df["ess_fraction"], errors="coerce") >= 0.04
    df["unit_qc_pass"] = df["split_gate_pass"] & df["ess_gate_pass"] & np.isfinite(pd.to_numeric(df["logZ_inf_full"], errors="coerce"))
    by_radius = (
        df.groupby("radius", as_index=False)
        .agg(
            unit_rows=("ref_id", "size"),
            observed_ref_count=("ref_id", "nunique"),
            pass_count=("unit_qc_pass", "sum"),
            max_split_logZ_per_P_diff=("split_logZ_per_P_diff", "max"),
            q05_ess_fraction=("ess_fraction", lambda x: float(np.quantile(pd.to_numeric(x, errors="coerce").dropna(), 0.05)) if len(pd.to_numeric(x, errors="coerce").dropna()) else float("nan")),
            mean_elapsed_s=("elapsed_s", "mean"),
        )
        .sort_values("radius")
    )
    by_radius["all_observed_pass"] = by_radius["pass_count"].astype(int) == by_radius["unit_rows"].astype(int)
    write_csv(out_dir / "pilot_qc_by_radius.csv", by_radius)
    mean_elapsed = float(pd.to_numeric(df["elapsed_s"], errors="coerce").mean())
    remaining_30_ref_10_large_units = 30 * 10
    summary = {
        "selector": selector,
        "rule": rule,
        "requested_radii": radii,
        "unit_rows": int(len(df)),
        "elapsed_s": float(time.time() - started),
        "mean_unit_elapsed_s": mean_elapsed,
        "estimated_30ref_10_large_radius_hours_at_mean": float(mean_elapsed * remaining_30_ref_10_large_units / 3600.0),
        "max_split_logZ_per_P_diff": float(pd.to_numeric(df["split_logZ_per_P_diff"], errors="coerce").max()),
        "q05_ess_fraction": float(np.quantile(pd.to_numeric(df["ess_fraction"], errors="coerce").dropna(), 0.05)),
        "all_observed_units_pass": bool(df["unit_qc_pass"].all()),
        "decision": "pilot_pass_observed_units" if bool(df["unit_qc_pass"].all()) else "pilot_qc_fail_observed_units",
    }
    write_json(out_dir / "pilot_summary.json", summary)
    fig_dir = ensure_dir(out_dir / "figures")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ref_id, sub in df.groupby("ref_id"):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub["split_logZ_per_P_diff"], marker="o", linewidth=1.0, label=f"ref {int(ref_id)}")
    ax.axhline(split_gate, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("d_raw")
    ax.set_ylabel("split logZ/P diff")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_targeted_pilot_split_logz.png", dpi=170)
    plt.close(fig)
    return summary


def write_report(out_dir: Path, summary: dict[str, Any], tasks: list[tuple[dict[str, Any], float]]) -> None:
    task_rows = "\n".join(f"- {row['rule']} ref_{int(row['ref_id']):03d} d_raw={radius:.4f}" for row, radius in tasks[:50])
    if len(tasks) > 50:
        task_rows += f"\n- ... {len(tasks) - 50} more"
    report = f"""# Targeted Selector Stage05 Pilot

Selector: `{summary["selector"]}`

Rule: `{summary["rule"]}`

Decision: `{summary["decision"]}`

Observed unit rows: `{summary["unit_rows"]}`

Mean unit elapsed seconds: `{summary.get("mean_unit_elapsed_s", "n/a")}`

Estimated 30-ref x 10-large-radius hours at this mean: `{summary.get("estimated_30ref_10_large_radius_hours_at_mean", "n/a")}`

## Tasks

{task_rows if task_rows else "- none"}

## Outputs

- `pilot_unit_summary.csv`
- `pilot_qc_by_radius.csv`
- `pilot_summary.json`
- `figures/fig01_targeted_pilot_split_logz.png`
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", default=DEFAULT_SELECTOR)
    parser.add_argument("--rule", default=DEFAULT_RULE)
    parser.add_argument("--radii", default="0.45")
    parser.add_argument("--ref-ids", default="")
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--only-missing-source", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    radii = [float(x.strip()) for x in str(args.radii).split(",") if x.strip()]
    ref_ids = [int(x.strip()) for x in str(args.ref_ids).split(",") if x.strip()] or None
    cfg = configure_pilot_pipe()
    if str(args.run_label).strip():
        out_dir = PILOT_RUN_ROOT / "05_pool2_pm_sais_sampling" / "targeted_selector_pilot" / args.selector / args.rule / str(args.run_label).strip()
    else:
        out_dir = PILOT_RUN_ROOT / "05_pool2_pm_sais_sampling" / "targeted_selector_pilot" / args.selector / args.rule / ("r_" + "_".join(f"{r:.4f}".replace(".", "p") for r in radii))
    out_dir = ensure_dir(out_dir)
    tasks = load_selector_tasks(args.selector, args.rule, radii, only_missing_source=bool(args.only_missing_source), ref_ids=ref_ids)
    if args.max_units is not None:
        tasks = tasks[: int(args.max_units)]
    write_json(
        out_dir / "run_config_resolved.json",
        {
            **cfg,
            "selector": args.selector,
            "rule": args.rule,
            "radii": radii,
            "max_units": args.max_units,
            "only_missing_source": bool(args.only_missing_source),
            "ref_ids": ref_ids,
            "task_count": len(tasks),
            "source_run_root": rel(SOURCE_RUN_ROOT),
        },
    )
    rows: list[dict[str, Any]] = []
    started = time.time()
    for idx, (row, radius) in enumerate(tasks, start=1):
        print(f"[targeted selector pilot] unit {idx}/{len(tasks)} selector={args.selector} rule={row['rule']} ref={int(row['ref_id']):03d} r={radius:.4f}", flush=True)
        rows.append(sparse.base.pipe.sample_stage05_unit(row, radius, cfg, force=bool(args.force)))
    df = pd.DataFrame(rows)
    write_csv(out_dir / "pilot_unit_summary.csv", df)
    summary = summarize_pilot(df, args.selector, args.rule, radii, out_dir, started)
    write_report(out_dir, summary, tasks)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
