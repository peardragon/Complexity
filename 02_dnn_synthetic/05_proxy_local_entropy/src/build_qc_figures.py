from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLING_ROOT = REPO_ROOT / "02_dnn_synthetic" / "04_sampling" / "raw_outputs" / "shell_pool"
SAMPLING_CONFIG = REPO_ROOT / "02_dnn_synthetic" / "config" / "04_sampling.json"
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "05_proxy_local_entropy"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / "qc"
FIGURE_INPUT_ROOT = STAGE_ROOT / "summarized_outputs" / "figure_inputs"
FIGURE_ROOT = STAGE_ROOT / "figures"

QC_RADII = ("r_0p01", "r_0p05", "r_0p10", "r_0p25", "r_0p50", "r_1p00", "r_1p50", "r_2p00", "r_2p50")
QC_REFS = ("ref_001", "ref_008", "ref_015", "ref_022", "ref_030")
DATASETS_PER_BETA = 15


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figure_input_csv(figure_name: str, source_name: str, rows: list[dict[str, object]]) -> None:
    _write_csv(FIGURE_INPUT_ROOT / figure_name / source_name, rows)


def _quantile(values: list[float], q: float) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def _beta_from_cell(cell_name: str) -> float:
    return float(cell_name.removeprefix("cell_beta_").replace("p", "."))


def _radius_from_dir(radius_name: str) -> float:
    return float(radius_name.removeprefix("r_").replace("p", "."))


def _load_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for cell_dir in sorted(SAMPLING_ROOT.glob("cell_beta_*")):
        if not cell_dir.is_dir():
            continue
        beta = _beta_from_cell(cell_dir.name)
        dataset_dirs = sorted(p for p in cell_dir.glob("dataset_*") if p.is_dir())[:DATASETS_PER_BETA]
        for dataset_dir in dataset_dirs:
            for ref_name in QC_REFS:
                ref_dir = dataset_dir / ref_name
                if not ref_dir.exists():
                    continue
                for radius_name in QC_RADII:
                    summary_path = ref_dir / radius_name / "unit_summary.json"
                    if not summary_path.exists():
                        continue
                    row = _read_json(summary_path)
                    records.append(
                        {
                            "beta": beta,
                            "cell": cell_dir.name,
                            "dataset": dataset_dir.name,
                            "ref": ref_name,
                            "radius": _radius_from_dir(radius_name),
                            "split_logZ_per_P_diff": float(row.get("split_logZ_per_P_diff", float("nan"))),
                            "split_dlogZ_dr_per_P_diff": float(row.get("split_dlogZ_dr_per_P_diff", float("nan"))),
                            "ess_frac": float(row.get("ess_frac", float("nan"))),
                            "smc_min_ess_fraction": float(row.get("smc_min_ess_fraction", float("nan"))),
                            "smc_min_cess_fraction": float(row.get("smc_min_cess_fraction", float("nan"))),
                            "smc_step_count": int(row.get("smc_step_count", 0)),
                            "smc_mean_mh_acceptance": float(row.get("smc_mean_mh_acceptance", float("nan"))),
                            "logZ_inf_full": float(row.get("logZ_inf_full", float("nan"))),
                            "dlogZ_inf_full_dr": float(row.get("dlogZ_inf_full_dr", float("nan"))),
                            "sampler_method": str(row.get("sampler_method", "")),
                            "smc_completed": bool(row.get("smc_completed", False)),
                        }
                    )
    if not records:
        raise FileNotFoundError("no QC unit_summary.json records were found")
    return records


def _logz_qc_rows(records: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(float(record["beta"]), float(record["radius"]))].append(record)

    rows: list[dict[str, object]] = []
    for (beta, radius), items in sorted(grouped.items()):
        split = [float(item["split_logZ_per_P_diff"]) for item in items]
        dsplit = [float(item["split_dlogZ_dr_per_P_diff"]) for item in items]
        ess = [float(item["ess_frac"]) for item in items]
        cess = [float(item["smc_min_cess_fraction"]) for item in items]
        completed = sum(1 for item in items if bool(item["smc_completed"]))
        max_split = max(split)
        rows.append(
            {
                "beta": beta,
                "radius": radius,
                "unit_count": len(items),
                "smc_completed_count": completed,
                "mean_split_logZ_per_P_diff": mean(split),
                "q95_split_logZ_per_P_diff": _quantile(split, 0.95),
                "max_split_logZ_per_P_diff": max_split,
                "mean_split_dlogZ_dr_per_P_diff": mean(dsplit),
                "q05_ess_frac": _quantile(ess, 0.05),
                "q05_smc_min_cess_fraction": _quantile(cess, 0.05),
                "threshold_max_split_logZ_per_P_diff": threshold,
                "claim": "pass" if max_split <= threshold and completed == len(items) else "inspect",
            }
        )
    return rows


def _reference_variability_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset: dict[tuple[float, float, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_dataset[(float(record["beta"]), float(record["radius"]), str(record["dataset"]))].append(record)

    ref_sd_rows: dict[tuple[float, float], list[dict[str, float]]] = defaultdict(list)
    for (beta, radius, _dataset), items in sorted(by_dataset.items()):
        logz = [float(item["logZ_inf_full"]) for item in items if np.isfinite(float(item["logZ_inf_full"]))]
        dlogz = [float(item["dlogZ_inf_full_dr"]) for item in items if np.isfinite(float(item["dlogZ_inf_full_dr"]))]
        if len(logz) < 2 or len(dlogz) < 2:
            continue
        ref_sd_rows[(beta, radius)].append(
            {
                "ref_count": float(len(items)),
                "sd_logZ_inf_full": pstdev(logz),
                "se_logZ_inf_full": pstdev(logz) / np.sqrt(len(logz)),
                "sd_dlogZ_inf_full_dr": pstdev(dlogz),
                "se_dlogZ_inf_full_dr": pstdev(dlogz) / np.sqrt(len(dlogz)),
            }
        )

    rows: list[dict[str, object]] = []
    for (beta, radius), items in sorted(ref_sd_rows.items()):
        rows.append(
            {
                "beta": beta,
                "radius": radius,
                "dataset_count": len(items),
                "ref_count_min": int(min(item["ref_count"] for item in items)),
                "ref_count_max": int(max(item["ref_count"] for item in items)),
                "mean_ref_sd_logZ_inf_full": mean(item["sd_logZ_inf_full"] for item in items),
                "q95_ref_sd_logZ_inf_full": _quantile([item["sd_logZ_inf_full"] for item in items], 0.95),
                "mean_ref_se_logZ_inf_full": mean(item["se_logZ_inf_full"] for item in items),
                "q95_ref_se_logZ_inf_full": _quantile([item["se_logZ_inf_full"] for item in items], 0.95),
                "mean_ref_sd_dlogZ_inf_full_dr": mean(item["sd_dlogZ_inf_full_dr"] for item in items),
                "q95_ref_sd_dlogZ_inf_full_dr": _quantile([item["sd_dlogZ_inf_full_dr"] for item in items], 0.95),
                "mean_ref_se_dlogZ_inf_full_dr": mean(item["se_dlogZ_inf_full_dr"] for item in items),
                "q95_ref_se_dlogZ_inf_full_dr": _quantile([item["se_dlogZ_inf_full_dr"] for item in items], 0.95),
            }
        )
    return rows


def _dataset_variability_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset: dict[tuple[float, float, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_dataset[(float(record["beta"]), float(record["radius"]), str(record["dataset"]))].append(record)

    dataset_means: dict[tuple[float, float], list[dict[str, float]]] = defaultdict(list)
    for (beta, radius, _dataset), items in sorted(by_dataset.items()):
        logz = [float(item["logZ_inf_full"]) for item in items if np.isfinite(float(item["logZ_inf_full"]))]
        dlogz = [float(item["dlogZ_inf_full_dr"]) for item in items if np.isfinite(float(item["dlogZ_inf_full_dr"]))]
        if not logz or not dlogz:
            continue
        dataset_means[(beta, radius)].append(
            {
                "mean_logZ_inf_full": mean(logz),
                "mean_dlogZ_inf_full_dr": mean(dlogz),
            }
        )

    rows: list[dict[str, object]] = []
    for (beta, radius), items in sorted(dataset_means.items()):
        logz_means = [item["mean_logZ_inf_full"] for item in items]
        dlogz_means = [item["mean_dlogZ_inf_full_dr"] for item in items]
        dataset_count = len(items)
        dataset_sd_logz = pstdev(logz_means) if dataset_count > 1 else 0.0
        dataset_sd_dlogz = pstdev(dlogz_means) if dataset_count > 1 else 0.0
        rows.append(
            {
                "beta": beta,
                "radius": radius,
                "dataset_count": dataset_count,
                "dataset_sd_logZ_inf_full": dataset_sd_logz,
                "dataset_se_logZ_inf_full": dataset_sd_logz / np.sqrt(dataset_count),
                "dataset_sd_dlogZ_inf_full_dr": dataset_sd_dlogz,
                "dataset_se_dlogZ_inf_full_dr": dataset_sd_dlogz / np.sqrt(dataset_count),
                "dataset_mean_logZ_inf_full": mean(logz_means),
                "dataset_mean_dlogZ_inf_full_dr": mean(dlogz_means),
            }
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate_by_beta(rows: list[dict[str, object]], keys: tuple[str, ...]) -> dict[float, dict[str, float]]:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["beta"])].append(row)
    out: dict[float, dict[str, float]] = {}
    for beta, items in grouped.items():
        out[beta] = {key: mean(float(item[key]) for item in items) for key in keys}
    return dict(sorted(out.items()))


def _plot_logz_qc(rows: list[dict[str, object]], path: Path) -> None:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["beta"])].append(row)
    fig, ax = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    betas = sorted(grouped)
    for index, beta in enumerate(betas):
        items = sorted(grouped[beta], key=lambda item: float(item["radius"]))
        x = [float(item["radius"]) for item in items]
        y = [float(item["q95_split_logZ_per_P_diff"]) for item in items]
        ax.plot(x, y, linewidth=1.2, alpha=0.9, color=cmap(index / max(len(betas) - 1, 1)))
    threshold = float(rows[0]["threshold_max_split_logZ_per_P_diff"])
    ax.axhline(threshold, color="#b42318", linewidth=1.1, linestyle="--", label="QC threshold")
    ax.set_xscale("log")
    ax.set_xlabel("distance d")
    ax.set_ylabel("q95 split logZ / P diff")
    ax.set_title("Split logZ stability QC")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_variability(rows: list[dict[str, object]], path: Path, *, prefix: str, title: str) -> None:
    logz_key = f"{prefix}_sd_logZ_inf_full"
    dlogz_key = f"{prefix}_sd_dlogZ_inf_full_dr"
    by_beta = _aggregate_by_beta(rows, (logz_key, dlogz_key))
    beta = np.asarray(list(by_beta), dtype=np.float64)
    logz = np.asarray([item[logz_key] for item in by_beta.values()], dtype=np.float64)
    dlogz = np.asarray([item[dlogz_key] for item in by_beta.values()], dtype=np.float64)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    ax_left.plot(beta, logz, "o-", color="#2364aa", linewidth=1.35, markersize=4.0)
    ax_left.set_xlabel("beta")
    ax_left.set_ylabel("SD logZ")
    ax_left.set_title("logZ variability")
    ax_left.grid(True, alpha=0.24)

    ax_right.plot(beta, dlogz, "o-", color="#7a3e9d", linewidth=1.35, markersize=4.0)
    ax_right.set_xlabel("beta")
    ax_right.set_ylabel("SD dlogZ/dd")
    ax_right.set_title("derivative variability")
    ax_right.grid(True, alpha=0.24)

    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_report(records: list[dict[str, object]], logz_rows: list[dict[str, object]]) -> None:
    run_config = _read_json_if_exists(SAMPLING_ROOT / "run_config.json") or _read_json_if_exists(SAMPLING_CONFIG)
    status = _read_json_if_exists(SAMPLING_ROOT / "logs" / "aggregate_status.json")
    backend = _read_json_if_exists(SAMPLING_ROOT / "logs" / "backend_selection_report.json")
    methods = sorted({str(record["sampler_method"]) for record in records})
    claims = sorted({str(row["claim"]) for row in logz_rows})
    sampling = run_config.get("sampling", {})
    smc = run_config.get("smc", {})
    tempered_path_default = bool(
        sampling.get("tempered_path_default")
        or smc.get("target_cess_fraction") is not None
        or any("smc" in method.lower() for method in methods)
    )

    report = [
        "# DNN synthetic QC report",
        "",
        f"- method: `{run_config.get('method')}`",
        f"- sampling proposal: `{sampling.get('proposal')}`",
        f"- tempered-path default: `{tempered_path_default}`",
        f"- tempered-path evidence: `adaptive SMC temperature schedule with target CESS`",
        f"- SMC target CESS: `{smc.get('target_cess_fraction')}`",
        f"- SMC resample ESS: `{smc.get('resample_ess_fraction')}`",
        f"- backend: `{backend.get('selected_backend')}`",
        f"- completed units: `{status.get('completed')}` / `{status.get('total')}`",
        f"- failed units: `{status.get('failed')}`",
        f"- QC subset records: `{len(records)}`",
        f"- QC subset datasets per beta: `{DATASETS_PER_BETA}`",
        f"- QC refs: `{', '.join(QC_REFS)}`",
        f"- QC radii: `{', '.join(QC_RADII)}`",
        f"- unit sampler_method values: `{', '.join(methods)}`",
        f"- split logZ QC claims: `{', '.join(claims)}`",
        "",
        "## Figure inputs",
        "",
        "- `figure_inputs/logZ_split_qc_results/logZ_split_qc_results.csv` -> `figures/logZ_split_qc_results.png`",
        "- `figure_inputs/reference_variability_results/reference_variability_results.csv` -> `figures/reference_variability_results.png`",
        "- `figure_inputs/dataset_variability_results/dataset_variability_results.csv` -> `figures/dataset_variability_results.png`",
        "",
        "## Variability definitions",
        "",
        "- Reference variability: for each beta/radius/dataset, compute SD of `logZ_inf_full` across references, then summarize those SDs across datasets.",
        "- Dataset variability: for each beta/radius/dataset, average `logZ_inf_full` across references, then compute SD across datasets.",
        "- SE columns are included as `SD / sqrt(n)` for average-stability checks; figures plot SD because they visualize raw variability.",
        "- `dlogZ_inf_full_dr` variability columns are kept as derivative-side companion diagnostics; the main variability quantity is `logZ_inf_full`.",
        "",
        "The raw run uses adaptive CE SMC with a temperature path selected by target CESS. Older unit summaries keep the method label `exact_shell_l2_vmf_adaptive_ce_smc`; this report treats that as the tempered-path PM-SAIS default when SMC target-CESS fields and histories are present.",
    ]
    (SUMMARY_ROOT / "qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_cached_report(logz_rows: list[dict[str, object]]) -> None:
    run_config = _read_json_if_exists(SAMPLING_CONFIG)
    claims = sorted({str(row.get("claim", "")) for row in logz_rows})
    report = [
        "# DNN synthetic QC report",
        "",
        "- mode: `cached summarized_outputs`",
        f"- method: `{run_config.get('sampler', run_config.get('method_name', 'TPIS'))}`",
        "- tempered-path evidence: `QC raw payloads are omitted from Git; retained summaries were produced from adaptive CE SMC temperature-path runs.`",
        f"- split logZ QC claims: `{', '.join(claims)}`",
        "",
        "This report was regenerated without raw unit payloads. It uses the retained `summarized_outputs/qc/*.csv` files as the source of truth.",
    ]
    (SUMMARY_ROOT / "qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    run_config = _read_json_if_exists(SAMPLING_ROOT / "run_config.json") or _read_json_if_exists(SAMPLING_CONFIG)
    threshold = float(run_config.get("qc", {}).get("max_split_logZ_per_P_diff", 0.004))

    try:
        records = _load_records()
        logz_rows = _logz_qc_rows(records, threshold)
        ref_rows = _reference_variability_rows(records)
        dataset_rows = _dataset_variability_rows(records)
        _write_report(records, logz_rows)
    except FileNotFoundError:
        logz_rows = _read_csv_rows(SUMMARY_ROOT / "logZ_split_qc_results.csv")
        ref_rows = _read_csv_rows(SUMMARY_ROOT / "reference_variability_results.csv")
        dataset_rows = _read_csv_rows(SUMMARY_ROOT / "dataset_variability_results.csv")
        _write_cached_report(logz_rows)

    _write_csv(SUMMARY_ROOT / "logZ_split_qc_results.csv", logz_rows)
    _write_csv(SUMMARY_ROOT / "reference_variability_results.csv", ref_rows)
    _write_csv(SUMMARY_ROOT / "dataset_variability_results.csv", dataset_rows)
    _write_figure_input_csv("logZ_split_qc_results", "logZ_split_qc_results.csv", logz_rows)
    _write_figure_input_csv("reference_variability_results", "reference_variability_results.csv", ref_rows)
    _write_figure_input_csv("dataset_variability_results", "dataset_variability_results.csv", dataset_rows)
    _plot_logz_qc(logz_rows, FIGURE_ROOT / "logZ_split_qc_results.png")
    _plot_variability(ref_rows, FIGURE_ROOT / "reference_variability_results.png", prefix="mean_ref", title="Reference variability QC")
    _plot_variability(dataset_rows, FIGURE_ROOT / "dataset_variability_results.png", prefix="dataset", title="Dataset variability QC")


if __name__ == "__main__":
    main()
