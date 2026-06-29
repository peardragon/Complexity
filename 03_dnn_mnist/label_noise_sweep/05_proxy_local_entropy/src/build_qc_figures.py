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


REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLING_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "04_sampling" / "raw_outputs" / "shell_pool"
STAGE_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "05_proxy_local_entropy"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / "qc"
FIGURE_INPUT_ROOT = STAGE_ROOT / "summarized_outputs" / "figure_inputs"
FIGURE_ROOT = STAGE_ROOT / "figures"
QC_RADII = ("r_0p0100", "r_0p0500", "r_0p1000", "r_0p2500", "r_0p5000", "r_0p7500", "r_1p0000")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_json_if_exists(path: Path) -> dict:
    if path.exists():
        return _read_json(path)
    return {}


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


def _radius_from_dir(name: str) -> float:
    return float(name.removeprefix("r_").replace("p", "."))


def _eta_from_dir(name: str) -> float:
    return float(name.removeprefix("noise_eta_").replace("p", "."))


def _quantile(values: list[float], q: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def _load_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for eta_dir in sorted(SAMPLING_ROOT.glob("noise_eta_*")):
        eta = _eta_from_dir(eta_dir.name)
        for ref_dir in sorted(eta_dir.glob("ref_*")):
            for radius_name in QC_RADII:
                summary_path = ref_dir / radius_name / "unit_summary.json"
                if not summary_path.exists():
                    continue
                row = _read_json(summary_path)
                records.append(
                    {
                        "noise_eta": eta_dir.name,
                        "eta": eta,
                        "ref": ref_dir.name,
                        "radius": _radius_from_dir(radius_name),
                        "split_logZ_per_P_diff": float(row.get("split_logZ_per_P_diff", float("nan"))),
                        "split_dlogZ_dr_per_P_diff": float(row.get("split_dlogZ_dr_per_P_diff", float("nan"))),
                        "ess_fraction": float(row.get("ess_fraction", float("nan"))),
                        "smc_min_cess_fraction": float(row.get("smc_min_cess_fraction", float("nan"))),
                        "smc_completed": bool(row.get("smc_completed", False)),
                        "sampler_method": str(row.get("sampler_method", "")),
                        "logZ_inf_full": float(row.get("logZ_inf_full", float("nan"))),
                    }
                )
    if not records:
        raise FileNotFoundError(f"no QC unit_summary.json records under {SAMPLING_ROOT}")
    return records


def _logz_qc_rows(records: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["noise_eta"]), float(record["radius"]))].append(record)
    rows: list[dict[str, object]] = []
    for (noise_eta, radius), items in sorted(grouped.items()):
        split = [float(item["split_logZ_per_P_diff"]) for item in items]
        dsplit = [float(item["split_dlogZ_dr_per_P_diff"]) for item in items]
        ess = [float(item["ess_fraction"]) for item in items]
        cess = [float(item["smc_min_cess_fraction"]) for item in items]
        completed = sum(1 for item in items if bool(item["smc_completed"]))
        max_split = max(split)
        rows.append(
            {
                "noise_eta": noise_eta,
                "eta": _eta_from_dir(noise_eta),
                "radius": radius,
                "ref_count": len(items),
                "smc_completed_count": completed,
                "mean_split_logZ_per_P_diff": mean(split),
                "q95_split_logZ_per_P_diff": _quantile(split, 0.95),
                "max_split_logZ_per_P_diff": max_split,
                "mean_split_dlogZ_dr_per_P_diff": mean(dsplit),
                "q95_split_dlogZ_dr_per_P_diff": _quantile(dsplit, 0.95),
                "max_split_dlogZ_dr_per_P_diff": max(dsplit),
                "q05_ess_fraction": _quantile(ess, 0.05),
                "q05_smc_min_cess_fraction": _quantile(cess, 0.05),
                "threshold_max_split_logZ_per_P_diff": threshold,
                "claim": "pass" if max_split <= threshold and completed == len(items) else "inspect",
            }
        )
    return rows


def _reference_variability_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["noise_eta"]), float(record["radius"]))].append(record)
    rows: list[dict[str, object]] = []
    for (noise_eta, radius), items in sorted(grouped.items()):
        logz = [float(item["logZ_inf_full"]) for item in items if np.isfinite(float(item["logZ_inf_full"]))]
        sd = pstdev(logz) if len(logz) > 1 else 0.0
        rows.append(
            {
                "noise_eta": noise_eta,
                "eta": _eta_from_dir(noise_eta),
                "radius": radius,
                "ref_count": len(logz),
                "reference_sd_logZ_inf_full": sd,
                "reference_se_logZ_inf_full": sd / np.sqrt(len(logz)) if logz else float("nan"),
                "reference_mean_logZ_inf_full": mean(logz) if logz else float("nan"),
            }
        )
    return rows


def _plot_logz(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)
    for eta in sorted({float(row["eta"]) for row in rows}):
        items = sorted([row for row in rows if float(row["eta"]) == eta], key=lambda row: float(row["radius"]))
        ax.plot(
            [float(row["radius"]) for row in items],
            [float(row["q95_split_logZ_per_P_diff"]) for row in items],
            marker="o",
            linewidth=1.4,
            markersize=4.0,
            label=f"eta={eta:.2f}",
        )
    threshold = float(rows[0]["threshold_max_split_logZ_per_P_diff"])
    ax.axhline(threshold, color="#b42318", linestyle="--", linewidth=1.1, label="QC threshold")
    ax.set_xscale("log")
    ax.set_xlabel("distance d")
    ax.set_ylabel("q95 split logZ / P diff")
    ax.set_title("MNIST label-noise split logZ QC")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_reference_variability(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)
    for eta in sorted({float(row["eta"]) for row in rows}):
        items = sorted([row for row in rows if float(row["eta"]) == eta], key=lambda row: float(row["radius"]))
        ax.plot(
            [float(row["radius"]) for row in items],
            [float(row["reference_sd_logZ_inf_full"]) for row in items],
            marker="o",
            linewidth=1.4,
            markersize=4.0,
            label=f"eta={eta:.2f}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("distance d")
    ax.set_ylabel("SD logZ across references")
    ax.set_title("MNIST label-noise reference variability QC")
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_report(records: list[dict[str, object]], logz_rows: list[dict[str, object]]) -> None:
    run_config = _read_json_if_exists(SAMPLING_ROOT / "run_metadata" / "run_config_resolved.json")
    status = _read_json_if_exists(SAMPLING_ROOT / "run_metadata" / "SAMPLING_STATUS.json")
    methods = sorted({str(record["sampler_method"]) for record in records})
    claims = sorted({str(row["claim"]) for row in logz_rows})
    claim_counts = {claim: sum(1 for row in logz_rows if str(row["claim"]) == claim) for claim in claims}
    inspect_rows = [row for row in logz_rows if str(row["claim"]) == "inspect"]
    report = [
        "# MNIST label-noise QC report",
        "",
        f"- sampling proposal: `{run_config.get('sampling', {}).get('proposal')}`",
        "- radial derivative: `sampling-time direct radial score derivative`.",
        f"- tempered-path default: `{'tempered' in ' '.join(methods).lower()}`",
        f"- SMC target CESS: `{run_config.get('smc', {}).get('target_cess_fraction')}`",
        f"- completed units: `{status.get('completed_units', 'unknown')}` / `{status.get('expected_units', 'unknown')}`",
        f"- QC subset records: `{len(records)}`",
        f"- unit sampler_method values: `{', '.join(methods)}`",
        f"- split logZ QC claims: `{', '.join(claims)}`",
        f"- split logZ QC claim counts: `{claim_counts}`",
        "",
        "## Figure inputs",
        "",
        "- `figure_inputs/logZ_split_qc_results/logZ_split_qc_results.csv` -> `figures/logZ_split_qc_results.png`",
        "- `figure_inputs/reference_variability_results/reference_variability_results.csv` -> `figures/reference_variability_results.png`",
        "",
        "## Variability definitions",
        "",
        "- Reference variability: for each eta/radius, compute SD of `logZ_inf_full` across references.",
        "- Dataset variability is not plotted for this stage because the MNIST label-noise sweep uses one dataset split.",
        "- SE columns are included as `SD / sqrt(n)`; figures plot SD because they visualize raw variability.",
    ]
    if inspect_rows:
        report.extend(
            [
                "",
                "## Split LogZ Inspect Cells",
                "",
                "| noise_eta | radius | max split logZ/P diff | threshold | q05 SMC CESS |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in inspect_rows:
            report.append(
                "| "
                f"{row['noise_eta']} | "
                f"{float(row['radius']):.2f} | "
                f"{float(row['max_split_logZ_per_P_diff']):.6g} | "
                f"{float(row['threshold_max_split_logZ_per_P_diff']):.6g} | "
                f"{float(row['q05_smc_min_cess_fraction']):.6g} |"
            )
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    (SUMMARY_ROOT / "qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_cached_report(logz_rows: list[dict[str, object]]) -> None:
    claims = sorted({str(row.get("claim", "")) for row in logz_rows})
    claim_counts = {claim: sum(1 for row in logz_rows if str(row.get("claim", "")) == claim) for claim in claims}
    report = [
        "# MNIST label-noise QC report",
        "",
        "- mode: `cached summarized_outputs`",
        "- radial derivative: `sampling-time direct radial score derivative`.",
        "- tempered-path evidence: `raw unit payloads are omitted from Git; retained summaries were produced from exact_shell_l2_vmf_adaptive_ce_tempered_smc runs.`",
        f"- split logZ QC claims: `{', '.join(claims)}`",
        f"- split logZ QC claim counts: `{claim_counts}`",
        "",
        "This report was regenerated without raw unit payloads. It uses the retained `summarized_outputs/qc/*.csv` files as the source of truth.",
    ]
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    (SUMMARY_ROOT / "qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    run_config = _read_json_if_exists(SAMPLING_ROOT / "run_config.json")
    threshold = float(run_config.get("qc", {}).get("max_split_logZ_per_P_diff", 0.004))
    try:
        records = _load_records()
        logz_rows = _logz_qc_rows(records, threshold)
        ref_rows = _reference_variability_rows(records)
        _write_report(records, logz_rows)
    except FileNotFoundError:
        logz_rows = _read_csv_rows(SUMMARY_ROOT / "logZ_split_qc_results.csv")
        ref_rows = _read_csv_rows(SUMMARY_ROOT / "reference_variability_results.csv")
        _write_cached_report(logz_rows)
    _write_csv(SUMMARY_ROOT / "logZ_split_qc_results.csv", logz_rows)
    _write_csv(SUMMARY_ROOT / "reference_variability_results.csv", ref_rows)
    _write_figure_input_csv("logZ_split_qc_results", "logZ_split_qc_results.csv", logz_rows)
    _write_figure_input_csv("reference_variability_results", "reference_variability_results.csv", ref_rows)
    _plot_logz(logz_rows, FIGURE_ROOT / "logZ_split_qc_results.png")
    _plot_reference_variability(ref_rows, FIGURE_ROOT / "reference_variability_results.png")


if __name__ == "__main__":
    main()
