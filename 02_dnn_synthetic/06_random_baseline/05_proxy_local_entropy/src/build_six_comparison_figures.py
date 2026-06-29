from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER_PATH = REPO_ROOT / "02_dnn_synthetic" / "05_proxy_local_entropy" / "src" / "build_six_figures.py"
SPIN_SUMMARY = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "05_proxy_local_entropy"
    / "summarized_outputs"
    / "18_beta_cell_90_dataset_30_reference"
    / "d_0.01_to_2.50_dense"
    / "summary_tables"
)
RANDOM_SUMMARY = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "06_random_baseline"
    / "05_proxy_local_entropy"
    / "summarized_outputs"
    / "gaussian_random_90_dataset_30_reference"
    / "d_0.01_to_2.50_dense"
    / "summary_tables"
)
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "05_proxy_local_entropy"
FIGURE_ROOT = STAGE_ROOT / "figures" / "with_random_baseline"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / "with_random_baseline"

FIGURE_SPECS = (
    ("phi_d_curve_with_random_baseline.png", "absolute_phi_by_beta_radius.csv", "phi_full", r"$\phi(d)$"),
    ("phi_energetic_d_curve_with_random_baseline.png", "absolute_phi_by_beta_radius.csv", "phi_energy", r"energetic $\phi(d)$"),
    ("derivative_phi_d_curve_with_random_baseline.png", "dphi_dr_by_beta_radius.csv", "dphi_full_dr", r"$d\phi/dd$"),
    ("derivative_phi_energetic_d_curve_with_random_baseline.png", "dphi_dr_by_beta_radius.csv", "dphi_energy_dr", r"energetic $d\phi/dd$"),
)


def _load_helper():
    spec = importlib.util.spec_from_file_location("synthetic_ple_figures", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_overlay(helper, source_name: str, value_key: str, ylabel: str, path: Path) -> None:
    spin_rows = helper._read_csv(SPIN_SUMMARY / source_name)
    random_rows = helper._read_csv(RANDOM_SUMMARY / source_name)
    spin_curves = helper._group_curves(spin_rows, value_key)
    random_curves = helper._group_curves(random_rows, value_key)
    random_radius, random_value = next(iter(random_curves.values()))

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, (beta, (radius, value)) in enumerate(spin_curves.items()):
        ax.plot(radius, value, linewidth=1.1, alpha=0.78, color=cmap(index / max(len(spin_curves) - 1, 1)))
    ax.plot(random_radius, random_value, color="black", linewidth=2.3, label="random baseline")
    ax.set_xscale("log")
    ax.set_xlabel(r"distance $d$")
    ax.set_ylabel(ylabel)
    ax.set_title("Spin synthetic with random baseline")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_a_overlay(helper, x_key: str, x_label: str, path: Path) -> None:
    spin_rows = helper._read_csv(SPIN_SUMMARY / "phase_like_A_measure.csv")
    random_rows = helper._read_csv(RANDOM_SUMMARY / "phase_like_A_measure.csv")
    spin_x = np.asarray([float(row[x_key]) for row in spin_rows], dtype=np.float64)
    spin_y = np.asarray([float(row["A_transition_total_variation"]) for row in spin_rows], dtype=np.float64)
    order = np.argsort(spin_x)
    random_x = float(random_rows[0][x_key])
    random_y = float(random_rows[0]["A_transition_total_variation"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(spin_x[order], spin_y[order], "o-", color="#7b3f8c", linewidth=1.4, markersize=4.5, label="spin synthetic")
    ax.scatter([random_x], [random_y], color="black", s=52, marker="s", label="random baseline", zorder=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("A measure")
    ax.set_title("A measure with random baseline")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    helper = _load_helper()
    for output_name, source_name, value_key, ylabel in FIGURE_SPECS:
        _plot_overlay(helper, source_name, value_key, ylabel, FIGURE_ROOT / output_name)

    spin_a = []
    for row in helper._read_csv(SPIN_SUMMARY / "phase_like_A_measure.csv"):
        clean = dict(row)
        clean["series"] = "spin_synthetic"
        clean["beta_role"] = "spin_sweep_axis"
        clean["source_beta_tag"] = ""
        spin_a.append(clean)
    random_a = []
    for row in helper._read_csv(RANDOM_SUMMARY / "phase_like_A_measure.csv"):
        clean = dict(row)
        clean["series"] = "gaussian_random_baseline"
        clean["beta_role"] = "source_tag_only_not_sweep_axis"
        clean["source_beta_tag"] = clean.get("source_beta_tag", clean.get("beta", ""))
        random_a.append(clean)
    _write_csv(SUMMARY_ROOT / "phase_like_A_with_random_baseline.csv", spin_a + random_a)
    _plot_a_overlay(helper, "beta", r"$\beta$", FIGURE_ROOT / "phase_like_A_by_beta_with_random_baseline.png")
    _plot_a_overlay(helper, "complexity_mean", "3-NN complexity", FIGURE_ROOT / "phase_like_A_by_complexity_with_random_baseline.png")


if __name__ == "__main__":
    main()
