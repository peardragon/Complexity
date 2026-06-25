#!/usr/bin/env python3
"""Fill MNIST unit-QC holes with new baseline-4096 reference units and plot phi(d).

The default policy matches the current sampling table: a unit is usable when it
passes finite/ESS/split QC, regardless of whether the old run used a fallback
replicate policy. Replacement units are always sampled with the baseline
4096-particle policy and fallback disabled.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
WINDOWS_ROOT = Path("/home/bjyong/Complexity/windows_project")
MNIST_ROOT = WINDOWS_ROOT / "02_dnn" / "08_mnist"
LOCAL_SRC_CACHE = LOCAL_ROOT / ".src_cache_mnist"
SRC_DIR = LOCAL_SRC_CACHE if LOCAL_SRC_CACHE.exists() else MNIST_ROOT / "src"
SOURCE_RUN_ROOT = MNIST_ROOT / "runs" / "final" / "local_support_dmax0p65_all_rules_resampled"
REFERENCE_RUN_ROOT = (
    MNIST_ROOT
    / "runs"
    / "final"
    / "single_dataset_10x10_box_n_train_512_60ref_allrule_sparse_0p010_to_2p500"
)
SOURCE_UNITS = SOURCE_RUN_ROOT / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv"
REFERENCE_INDEX = REFERENCE_RUN_ROOT / "04_exact_reference_search" / "reference_index.csv"
SELECTOR_MEMBERSHIP = REFERENCE_RUN_ROOT / "07_reference_family_analysis" / "selector_membership.csv"
DEFAULT_REPLACEMENT_RUN_ROOT = LOCAL_ROOT / "04_sampling" / "raw_outputs" / "qc_fill_baseline4096_replacement_units"
DEFAULT_OUT = LOCAL_ROOT / "05_proxy_local_entropy" / "raw_outputs" / "qc_filled_phi_dmax0p65"
EXTRA_REFERENCE_INDEX_NAME = "extra_reference_index.csv"

P = 2461.0
R0 = 0.010
SPLIT_GATE = 0.004
ESS_GATE = 0.04
SELECTOR = "dense_qc_stable_ref30"
SEED_OFFSET = 2026061700

RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
LABELS = {
    "low_tv_spectral_teacher": "low_tv",
    "real_even_odd": "even_odd",
    "teacher_nn": "teacher_nn",
    "random_label": "random",
}
COLORS = {
    "low_tv_spectral_teacher": "#2f6b9a",
    "real_even_odd": "#4c8c4a",
    "teacher_nn": "#b0782d",
    "random_label": "#9a3b58",
}
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

SAMPLING_SRC = LOCAL_ROOT / "04_sampling" / "src"
if str(SAMPLING_SRC) not in sys.path:
    sys.path.insert(0, str(SAMPLING_SRC))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_RESAMPLE_MODULE: Any | None = None


def resample_module() -> Any:
    global _RESAMPLE_MODULE
    if _RESAMPLE_MODULE is None:
        print("[setup] importing resample_mnist10_local_support", flush=True)
        import resample_mnist10_local_support as resample  # noqa: PLC0415

        _RESAMPLE_MODULE = resample
        print("[setup] imported resample_mnist10_local_support", flush=True)
    return _RESAMPLE_MODULE


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def radius_token(radius: float) -> str:
    return f"r_{float(radius):.4f}".replace(".", "p")


def unit_path(run_root: Path, rule: str, ref_id: int, radius: float) -> Path:
    return (
        run_root
        / "05_pool2_pm_sais_sampling"
        / "unit_summaries"
        / "split_000"
        / rule
        / f"ref_{int(ref_id):03d}"
        / radius_token(radius)
        / "unit_summary.json"
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


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


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
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


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda value: str(value).lower() == "true").fillna(False)


def unit_qc_pass(df: pd.DataFrame) -> pd.Series:
    finite = to_bool(df["finite"]) if "finite" in df.columns else pd.Series(False, index=df.index)
    ess = pd.to_numeric(df["ess_fraction"], errors="coerce") >= ESS_GATE
    split = pd.to_numeric(df["split_logZ_per_P_diff"], errors="coerce") <= SPLIT_GATE
    logz = np.isfinite(pd.to_numeric(df["logZ_inf_full"], errors="coerce"))
    return finite & ess & split & logz


def is_baseline4096(df: pd.DataFrame) -> pd.Series:
    n_samples = pd.to_numeric(df.get("n_samples", pd.Series(np.nan, index=df.index)), errors="coerce")
    sampler = df.get("sampler_method", pd.Series("", index=df.index)).astype(str)
    return n_samples.eq(4096) & sampler.eq("exact_shell_l2_vmf_adaptive_ce_tempered_smc")


def normalize_unit_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in [
        "split_id",
        "ref_id",
        "radius",
        "n_samples",
        "logZ_inf_full",
        "ess_fraction",
        "split_logZ_per_P_diff",
        "weighted_ce",
        "weighted_error",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["rule"] = out["rule"].astype(str)
    out["source"] = source
    out["unit_qc_pass"] = unit_qc_pass(out)
    out["baseline4096"] = is_baseline4096(out)
    out["phi_energy_raw"] = out["logZ_inf_full"] / P
    return out


def load_source_units() -> pd.DataFrame:
    units = pd.read_csv(SOURCE_UNITS)
    units = normalize_unit_frame(units, "source_selected")
    units["selected_ref"] = True
    return units


def load_replacement_units(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = run_root / "05_pool2_pm_sais_sampling" / "unit_summaries"
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.rglob("unit_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["unit_summary_path"] = str(path)
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    units = normalize_unit_frame(pd.DataFrame(rows), "replacement_baseline4096")
    units["selected_ref"] = False
    return units


def load_reference_tables(replacement_run_root: Path = DEFAULT_REPLACEMENT_RUN_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = pd.read_csv(REFERENCE_INDEX)
    refs["ref_id"] = refs["ref_id"].astype(int)
    extra_path = replacement_run_root / "04_extra_reference_search" / EXTRA_REFERENCE_INDEX_NAME
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        extra["ref_id"] = extra["ref_id"].astype(int)
        refs = pd.concat([refs, extra], ignore_index=True, sort=False)
    membership = pd.read_csv(SELECTOR_MEMBERSHIP)
    membership = membership[membership["selector"].eq(SELECTOR)].copy()
    membership["ref_id"] = membership["ref_id"].astype(int)
    return refs, membership


def selected_ref_sets(membership: pd.DataFrame) -> dict[str, set[int]]:
    return {
        rule: set(int(x) for x in sub["ref_id"].tolist())
        for rule, sub in membership.groupby("rule")
    }


def candidate_refs(refs: pd.DataFrame, membership: pd.DataFrame) -> dict[str, list[int]]:
    selected = selected_ref_sets(membership)
    result: dict[str, list[int]] = {}
    for rule in RULES:
        sub = refs[refs["rule"].eq(rule)].sort_values("ref_id")
        result[rule] = [
            int(ref_id)
            for ref_id in sub["ref_id"].tolist()
            if int(ref_id) not in selected.get(rule, set())
        ]
    return result


def row_for_ref(refs: pd.DataFrame, rule: str, ref_id: int) -> dict[str, Any]:
    row = refs[(refs["rule"].eq(rule)) & (refs["ref_id"].eq(int(ref_id)))]
    if row.empty:
        raise KeyError(f"No reference_index row for {rule} ref_{int(ref_id):03d}")
    item = row.iloc[0].to_dict()
    item["rule"] = str(item["rule"])
    item["ref_id"] = int(item["ref_id"])
    item["split_id"] = int(item.get("split_id", 0))
    return item


def eligible_units(units: pd.DataFrame, policy: str) -> pd.DataFrame:
    if units.empty:
        return units.copy()
    mask = units["unit_qc_pass"].fillna(False)
    if policy == "strict4096":
        mask = mask & units["baseline4096"].fillna(False)
    out = units[mask].copy()
    return out.sort_values(["rule", "radius", "source", "ref_id"]).reset_index(drop=True)


def attach_delta_phi(units: pd.DataFrame) -> pd.DataFrame:
    if units.empty:
        return units.copy()
    out = units.copy()
    key = ["rule", "ref_id"]
    anchors = (
        out[np.isclose(out["radius"], R0) & out["unit_qc_pass"]]
        .sort_values(["source", "ref_id"])
        .drop_duplicates(key, keep="first")[key + ["logZ_inf_full"]]
        .rename(columns={"logZ_inf_full": "anchor_logZ_inf_full"})
    )
    out = out.merge(anchors, on=key, how="left")
    self_anchor = np.isclose(out["radius"], R0) & out["anchor_logZ_inf_full"].isna()
    out.loc[self_anchor, "anchor_logZ_inf_full"] = out.loc[self_anchor, "logZ_inf_full"]
    out["has_anchor"] = np.isfinite(out["anchor_logZ_inf_full"])
    out["delta_phi_energy"] = (out["logZ_inf_full"] - out["anchor_logZ_inf_full"]) / P
    out["delta_phi_full"] = np.where(
        out["radius"] > 0,
        ((P - 1.0) / P) * np.log(out["radius"] / R0) + out["delta_phi_energy"],
        np.nan,
    )
    return out


def build_selection(units: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = eligible_units(attach_delta_phi(units), policy)
    eligible = eligible[eligible["has_anchor"]].copy()
    rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for rule in RULES:
        for radius in RADII:
            sub = eligible[eligible["rule"].eq(rule) & np.isclose(eligible["radius"], radius)].copy()
            sub["source_rank"] = np.where(sub["source"].eq("source_selected"), 0, 1)
            sub = sub.sort_values(["source_rank", "ref_id"]).drop_duplicates(["rule", "ref_id", "radius"], keep="first")
            chosen = sub.head(30).copy()
            chosen["filled_selection_rank"] = np.arange(1, len(chosen) + 1)
            rows.append(chosen)
            summary_rows.append(
                {
                    "rule": rule,
                    "radius": float(radius),
                    "usable_pass_count": int(sub["ref_id"].nunique()),
                    "selected_count": int(chosen["ref_id"].nunique()),
                    "deficit_to_30": int(max(0, 30 - chosen["ref_id"].nunique())),
                    "replacement_selected_count": int(chosen["source"].eq("replacement_baseline4096").sum()),
                    "fill_complete": bool(chosen["ref_id"].nunique() == 30),
                    "mean_delta_phi_energy": float(chosen["delta_phi_energy"].mean()) if len(chosen) else float("nan"),
                    "mean_delta_phi_full": float(chosen["delta_phi_full"].mean()) if len(chosen) else float("nan"),
                    "mean_phi_energy_raw": float(chosen["phi_energy_raw"].mean()) if len(chosen) else float("nan"),
                    "max_split_logZ_per_P_diff": float(chosen["split_logZ_per_P_diff"].max()) if len(chosen) else float("nan"),
                    "q05_ess_fraction": float(np.quantile(chosen["ess_fraction"], 0.05)) if len(chosen) else float("nan"),
                }
            )
    selection = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values(["rule", "radius"]).reset_index(drop=True)
    return selection, summary


def counts_by_policy(units: pd.DataFrame, policy: str) -> pd.DataFrame:
    eligible = eligible_units(attach_delta_phi(units), policy)
    eligible = eligible[eligible["has_anchor"]].copy()
    rows = []
    for rule in RULES:
        for radius in RADII:
            sub = eligible[eligible["rule"].eq(rule) & np.isclose(eligible["radius"], radius)]
            rows.append(
                {
                    "policy": policy,
                    "rule": rule,
                    "radius": float(radius),
                    "pass_ref_radius_count": int(sub["ref_id"].nunique()),
                    "deficit_to_30": int(max(0, 30 - sub["ref_id"].nunique())),
                }
            )
    return pd.DataFrame(rows)


def configure_replacement_sampler(run_root: Path, cpu_threads: int, device: str) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    if device:
        os.environ["MNIST14_DEVICE"] = device
    resample = resample_module()
    print("[setup] calling resample.configure_pipe", flush=True)
    cfg = resample.configure_pipe(run_root)
    print("[setup] configured replacement sampler", flush=True)
    cfg["sampling"] = dict(cfg["sampling"])
    cfg["sampling"]["fallback_policies_enabled"] = False
    cfg["sampling"]["seed_offset"] = SEED_OFFSET
    cfg["sampling"]["replacement_fill_note"] = (
        "Targeted replacement units for unit-QC holes. Baseline 4096 PM-SAIS only; fallback policies disabled."
    )
    cfg["outputs"] = dict(cfg["outputs"])
    cfg["outputs"]["run_root"] = str(run_root)
    write_json(run_root / "run_config_resolved.json", cfg)
    return cfg


def sample_one(
    refs: pd.DataFrame,
    cfg: dict[str, Any],
    run_root: Path,
    rule: str,
    ref_id: int,
    radius: float,
    *,
    force: bool,
) -> dict[str, Any]:
    resample = resample_module()
    row = row_for_ref(refs, rule, ref_id)
    payload = resample.sample_unit(row, float(radius), cfg, run_root, force=force)
    return payload


def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame([payload])
    frame["selected_ref"] = False
    return normalize_unit_frame(frame, "replacement_baseline4096")


def run_fill_sampling(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    print("[setup] loading reference tables", flush=True)
    run_root = Path(args.replacement_run_root)
    refs, membership = load_reference_tables(run_root)
    candidates = candidate_refs(refs, membership)
    print("[setup] configuring baseline-4096 replacement sampler", flush=True)
    cfg = configure_replacement_sampler(run_root, int(args.cpu_threads), args.device)
    print("[setup] loading existing source and replacement units", flush=True)
    source_units = load_source_units()
    replacement_units = load_replacement_units(run_root)
    all_units = pd.concat([source_units, replacement_units], ignore_index=True, sort=False)
    sample_log: list[dict[str, Any]] = []
    started = time.time()
    units_sampled = 0

    def has_pass(rule: str, ref_id: int, radius: float, units: pd.DataFrame) -> bool:
        sub = units[
            units["rule"].eq(rule)
            & units["ref_id"].eq(int(ref_id))
            & np.isclose(units["radius"], float(radius))
        ]
        if sub.empty:
            return False
        sub = attach_delta_phi(sub)
        return bool((sub["unit_qc_pass"] & sub["baseline4096"]).any())

    while True:
        selection, summary = build_selection(all_units, args.existing_policy)
        deficient = summary[summary["deficit_to_30"] > 0].copy()
        if deficient.empty:
            break
        made_attempt = False
        for _, deficit in deficient.sort_values(["rule", "radius"]).iterrows():
            rule = str(deficit["rule"])
            radius = float(deficit["radius"])
            if int(deficit["deficit_to_30"]) <= 0:
                continue
            for ref_id in candidates[rule]:
                if args.max_new_units is not None and units_sampled >= int(args.max_new_units):
                    write_csv(pd.DataFrame(sample_log), out_dir / "replacement_sampling_log.csv")
                    return all_units
                current_selection, current_summary = build_selection(all_units, args.existing_policy)
                current = current_summary[
                    current_summary["rule"].eq(rule) & np.isclose(current_summary["radius"], radius)
                ].iloc[0]
                if int(current["deficit_to_30"]) <= 0:
                    break
                already = all_units[
                    all_units["rule"].eq(rule)
                    & all_units["ref_id"].eq(int(ref_id))
                    & np.isclose(all_units["radius"], radius)
                    & all_units["source"].eq("replacement_baseline4096")
                ]
                if not already.empty and not bool(args.force):
                    continue
                if not np.isclose(radius, R0) and not has_pass(rule, ref_id, R0, all_units):
                    print(f"[anchor] rule={rule} ref={ref_id:03d} r={R0:.4f}", flush=True)
                    payload = sample_one(refs, cfg, run_root, rule, ref_id, R0, force=bool(args.force))
                    frame = payload_to_frame(payload)
                    all_units = pd.concat([all_units, frame], ignore_index=True, sort=False)
                    units_sampled += int(not bool(payload.get("reused", False)))
                    sample_log.append(
                        {
                            "elapsed_s_total": time.time() - started,
                            "kind": "anchor",
                            "rule": rule,
                            "ref_id": int(ref_id),
                            "radius": float(R0),
                            "unit_qc_pass": bool(frame.iloc[0]["unit_qc_pass"]),
                            "baseline4096": bool(frame.iloc[0]["baseline4096"]),
                            "reused": bool(payload.get("reused", False)),
                            "split_logZ_per_P_diff": float(frame.iloc[0]["split_logZ_per_P_diff"]),
                            "ess_fraction": float(frame.iloc[0]["ess_fraction"]),
                        }
                    )
                    made_attempt = True
                    if not bool(frame.iloc[0]["unit_qc_pass"] and frame.iloc[0]["baseline4096"]):
                        continue
                    if args.max_new_units is not None and units_sampled >= int(args.max_new_units):
                        write_csv(pd.DataFrame(sample_log), out_dir / "replacement_sampling_log.csv")
                        return all_units
                print(f"[fill] rule={rule} ref={ref_id:03d} r={radius:.4f}", flush=True)
                payload = sample_one(refs, cfg, run_root, rule, ref_id, radius, force=bool(args.force))
                frame = payload_to_frame(payload)
                all_units = pd.concat([all_units, frame], ignore_index=True, sort=False)
                units_sampled += int(not bool(payload.get("reused", False)))
                sample_log.append(
                    {
                        "elapsed_s_total": time.time() - started,
                        "kind": "fill",
                        "rule": rule,
                        "ref_id": int(ref_id),
                        "radius": float(radius),
                        "unit_qc_pass": bool(frame.iloc[0]["unit_qc_pass"]),
                        "baseline4096": bool(frame.iloc[0]["baseline4096"]),
                        "reused": bool(payload.get("reused", False)),
                        "split_logZ_per_P_diff": float(frame.iloc[0]["split_logZ_per_P_diff"]),
                        "ess_fraction": float(frame.iloc[0]["ess_fraction"]),
                    }
                )
                made_attempt = True
                break
        if not made_attempt:
            break
    write_csv(pd.DataFrame(sample_log), out_dir / "replacement_sampling_log.csv")
    return all_units


def plot_mean(summary: pd.DataFrame, value_col: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for rule in RULES:
        sub = summary[summary["rule"].eq(rule)].sort_values("radius")
        complete = sub[sub["fill_complete"]]
        incomplete = sub[~sub["fill_complete"]]
        ax.plot(sub["radius"], sub[value_col], color=COLORS[rule], lw=1.7, alpha=0.72)
        ax.scatter(complete["radius"], complete[value_col], color=COLORS[rule], s=34, label=LABELS[rule], zorder=3)
        if not incomplete.empty:
            ax.scatter(incomplete["radius"], incomplete[value_col], facecolors="white", edgecolors=COLORS[rule], s=42, zorder=3)
    ax.axhline(0.0, color="#777777", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("radius d")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, which="both", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_count_grid(summary: pd.DataFrame, path: Path) -> None:
    pivot = (
        summary.pivot(index="rule", columns="radius", values="selected_count")
        .reindex(RULES)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(11.2, 3.7))
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=30, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)), [LABELS[r] for r in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_xlabel("radius d")
    ax.set_title("Final selected QC-passed unit count")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = int(pivot.iat[i, j])
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color="white" if val < 18 else "black")
    fig.colorbar(im, ax=ax, label="selected units")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    out_dir: Path,
    current_counts: pd.DataFrame,
    strict_counts: pd.DataFrame,
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    policy: str,
) -> None:
    current_total = int(current_counts["pass_ref_radius_count"].sum())
    current_deficit = int(current_counts["deficit_to_30"].sum())
    strict_total = int(strict_counts["pass_ref_radius_count"].sum())
    strict_deficit = int(strict_counts["deficit_to_30"].sum())
    final_complete = bool(summary["fill_complete"].all())
    final_deficit = int(summary["deficit_to_30"].sum())
    by_rule = (
        summary.groupby("rule", as_index=False)
        .agg(
            final_selected_units=("selected_count", "sum"),
            final_deficit=("deficit_to_30", "sum"),
            replacement_units_used=("replacement_selected_count", "sum"),
            complete_radii=("fill_complete", "sum"),
        )
        .sort_values("rule")
    )
    lines = [
        "# MNIST QC-Filled Phi(d)",
        "",
        f"Existing-policy used for the final filled set: `{policy}`.",
        "Replacement sampling policy: baseline `n_samples == 4096`, fallback disabled.",
        "",
        "## Counts Before Fill",
        "",
        f"- Current unit-QC pass count: `{current_total}` / `2400`; deficit to 30 per rule/radius: `{current_deficit}`.",
        f"- Strict baseline-4096 unit-QC pass count: `{strict_total}` / `2400`; deficit to 30 per rule/radius: `{strict_deficit}`.",
        "",
        "## Final Filled Set",
        "",
        f"- Complete all 4 rules x 20 radii x 30 units: `{final_complete}`.",
        f"- Remaining deficit: `{final_deficit}`.",
        "",
        markdown_table(by_rule),
        "",
        "Averaging rule: simple arithmetic mean over the selected QC-passed units at each `(rule, radius)`.",
        "",
        "Files:",
        "",
        "- `counts_current_qc_existing.csv`",
        "- `counts_strict4096_existing.csv`",
        "- `replacement_units_qc_table.csv`",
        "- `filled_unit_selection.csv`",
        "- `filled_rule_radius_summary.csv`",
        "- `fig01_filled_mean_delta_phi_energy.png`",
        "- `fig02_filled_mean_delta_phi_full.png`",
        "- `fig03_filled_mean_raw_phi_energy.png`",
        "- `fig04_final_pass_count_grid.png`",
    ]
    if not final_complete:
        missing = summary[summary["deficit_to_30"] > 0][["rule", "radius", "selected_count", "deficit_to_30"]]
        lines.extend(["", "## Remaining Missing", "", markdown_table(missing, max_rows=80)])
    write_text(out_dir / "REPORT.md", "\n".join(lines) + "\n")


def finalize_outputs(args: argparse.Namespace, all_units: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    source_units = all_units[all_units["source"].eq("source_selected")].copy()
    current_counts = counts_by_policy(source_units, "current_qc")
    strict_counts = counts_by_policy(source_units, "strict4096")
    replacement = all_units[all_units["source"].eq("replacement_baseline4096")].copy()
    replacement = attach_delta_phi(replacement)
    selection, summary = build_selection(all_units, args.existing_policy)

    write_csv(current_counts, out_dir / "counts_current_qc_existing.csv")
    write_csv(strict_counts, out_dir / "counts_strict4096_existing.csv")
    write_csv(replacement, out_dir / "replacement_units_qc_table.csv")
    write_csv(selection, out_dir / "filled_unit_selection.csv")
    write_csv(summary, out_dir / "filled_rule_radius_summary.csv")

    plot_mean(
        summary,
        "mean_delta_phi_energy",
        "mean delta phi_energy",
        "QC-filled mean phi_energy(d), relative to d0=0.01",
        out_dir / "fig01_filled_mean_delta_phi_energy.png",
    )
    plot_mean(
        summary,
        "mean_delta_phi_full",
        "mean delta phi_full",
        "QC-filled mean full phi(d), relative to d0=0.01",
        out_dir / "fig02_filled_mean_delta_phi_full.png",
    )
    plot_mean(
        summary,
        "mean_phi_energy_raw",
        "mean logZ_inf_full / P",
        "QC-filled mean raw phi_energy(d)",
        out_dir / "fig03_filled_mean_raw_phi_energy.png",
    )
    plot_count_grid(summary, out_dir / "fig04_final_pass_count_grid.png")
    write_report(out_dir, current_counts, strict_counts, summary, selection, args.existing_policy)

    status = {
        "existing_policy": args.existing_policy,
        "final_complete": bool(summary["fill_complete"].all()),
        "remaining_deficit": int(summary["deficit_to_30"].sum()),
        "current_qc_existing_pass_count": int(current_counts["pass_ref_radius_count"].sum()),
        "current_qc_existing_deficit_to_30": int(current_counts["deficit_to_30"].sum()),
        "strict4096_existing_pass_count": int(strict_counts["pass_ref_radius_count"].sum()),
        "strict4096_existing_deficit_to_30": int(strict_counts["deficit_to_30"].sum()),
        "replacement_unit_rows": int(len(replacement)),
        "replacement_qc_pass_rows": int((replacement["unit_qc_pass"] & replacement["baseline4096"]).sum()) if len(replacement) else 0,
    }
    write_json(out_dir / "STATUS.json", status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill MNIST unit-QC holes and plot mean phi(d).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--replacement-run-root", default=str(DEFAULT_REPLACEMENT_RUN_ROOT))
    parser.add_argument("--existing-policy", choices=["current_qc", "strict4096"], default="current_qc")
    parser.add_argument("--sample", action="store_true", help="Sample replacement units until the selected set is complete or candidates are exhausted.")
    parser.add_argument("--max-new-units", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=int(os.environ.get("MNIST10_CPU_THREADS", "4")))
    parser.add_argument("--device", default=os.environ.get("MNIST14_DEVICE", ""))
    args = parser.parse_args(argv)

    out_dir = ensure_dir(Path(args.out))
    if args.sample:
        all_units = run_fill_sampling(args, out_dir)
    else:
        source_units = load_source_units()
        replacement_units = load_replacement_units(Path(args.replacement_run_root))
        all_units = pd.concat([source_units, replacement_units], ignore_index=True, sort=False)
    status = finalize_outputs(args, all_units, out_dir)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["final_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
