from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER_PATH = REPO_ROOT / "02_dnn_synthetic" / "05_proxy_local_entropy" / "src" / "build_six_figures.py"
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "05_proxy_local_entropy"
RUN_ROOT = STAGE_ROOT / "summarized_outputs" / "gaussian_random_90_dataset_30_reference" / "d_0.01_to_2.50_dense"
SUMMARY_ROOT = RUN_ROOT / "summary_tables"
FIGURE_ROOT = STAGE_ROOT / "figures"
RANDOM_COMPLEXITY_POINT = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "06_random_baseline"
    / "02_complexity_measure"
    / "summarized_outputs"
    / "gaussian_random_90_dataset_30_reference"
    / "summary_tables"
    / "random_baseline_complexity_point.csv"
)


def _load_helper():
    spec = importlib.util.spec_from_file_location("synthetic_ple_figures", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complexity_by_beta() -> dict[float, float]:
    with RANDOM_COMPLEXITY_POINT.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    beta_tag = round(float(row.get("source_beta_tag") or 0.05), 8)
    return {beta_tag: float(row["complexity_mean"])}


def _random_complexity_point() -> dict[str, str]:
    with RANDOM_COMPLEXITY_POINT.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def main() -> None:
    helper = _load_helper()
    if (FIGURE_ROOT / "gaussian_random_90_dataset_30_reference").exists():
        backup = FIGURE_ROOT / "legacy_nested_gaussian_random_90_dataset_30_reference"
        (FIGURE_ROOT / "gaussian_random_90_dataset_30_reference").rename(backup)
    for output_name, source_name, value_key, ylabel, title in helper.FIGURE_SPECS:
        rows = helper._read_csv(SUMMARY_ROOT / source_name)
        helper._plot_curves(rows, value_key, ylabel, title, FIGURE_ROOT / output_name)

    dphi_rows = helper._read_csv(SUMMARY_ROOT / "dphi_dr_by_beta_radius.csv")
    a_rows = helper._a_measure_rows(dphi_rows)
    complexity_lookup = _complexity_by_beta()
    random_point = _random_complexity_point()
    enriched_a_rows: list[dict[str, object]] = []
    for row in a_rows:
        enriched = dict(row)
        enriched["complexity_mean"] = complexity_lookup.get(round(float(row["beta"]), 8), float("nan"))
        enriched["series"] = "gaussian_random_baseline"
        enriched["beta_role"] = "source_tag_only_not_sweep_axis"
        enriched["source_beta_tag"] = random_point.get("source_beta_tag", "")
        enriched["complexity_metric"] = random_point.get("complexity_metric", "3nn_label_disagreement")
        enriched_a_rows.append(enriched)
    helper._write_csv(SUMMARY_ROOT / "phase_like_A_measure.csv", enriched_a_rows)
    helper._plot_phase_panel(
        dphi_rows,
        enriched_a_rows,
        FIGURE_ROOT / "phase_like_A_by_beta.png",
        x_key="beta",
        x_label=r"$\beta$",
        title="Random-baseline A measure by beta",
    )
    helper._plot_phase_panel(
        dphi_rows,
        enriched_a_rows,
        FIGURE_ROOT / "phase_like_A_by_complexity.png",
        x_key="complexity_mean",
        x_label="3-NN complexity",
        title="Random-baseline A measure by complexity",
    )


if __name__ == "__main__":
    main()
