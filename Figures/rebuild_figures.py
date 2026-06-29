from __future__ import annotations

import argparse
import runpy
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FIGURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURES_DIR.parent

ANALYTIC_CSV = (
    PROJECT_ROOT
    / "01_theory"
    / "01_theory_analytic"
    / "summarized_outputs"
    / "fig01_phi_by_analytic_solution_alpha0p1.csv"
)
SAMPLING_CSV = (
    PROJECT_ROOT
    / "01_theory"
    / "02_theory_sampling"
    / "summarized_outputs"
    / "fig01_sampling_phi_by_distance.csv"
)

OUTPUTS = {
    "analytic": FIGURES_DIR / "fig01_theory_analytic_phi_by_distance_alpha0p1.png",
    "sampling": FIGURES_DIR / "fig02_theory_sampling_phi_by_distance_alpha0p1.png",
    "combined": FIGURES_DIR / "fig03_theory_sampling_vs_analytic_phi_by_distance_alpha0p1.png",
    "theory_stage_combined": PROJECT_ROOT
    / "01_theory"
    / "figures"
    / "fig01_sampling_vs_analytic_phi_by_distance_alpha0p1.png",
}

DNN_PROXY_SCRIPT = PROJECT_ROOT / "02_dnn_synthetic" / "05_proxy_local_entropy" / "src" / "build_six_figures.py"
DNN_QC_SCRIPT = PROJECT_ROOT / "02_dnn_synthetic" / "05_proxy_local_entropy" / "src" / "build_qc_figures.py"
DNN_FIGURES_DIR = PROJECT_ROOT / "02_dnn_synthetic" / "05_proxy_local_entropy" / "figures"
DNN_FIGURE_OUTPUTS = {
    "dnn_synthetic_phi_d_curve": "phi_d_curve.png",
    "dnn_synthetic_phi_energetic_d_curve": "phi_energetic_d_curve.png",
    "dnn_synthetic_derivative_phi_d_curve": "derivative_phi_d_curve.png",
    "dnn_synthetic_derivative_phi_energetic_d_curve": "derivative_phi_energetic_d_curve.png",
    "dnn_synthetic_phase_like_A_by_beta": "phase_like_A_by_beta.png",
    "dnn_synthetic_phase_like_A_by_complexity": "phase_like_A_by_complexity.png",
    "dnn_synthetic_logZ_split_qc_results": "logZ_split_qc_results.png",
    "dnn_synthetic_reference_variability_results": "reference_variability_results.png",
    "dnn_synthetic_dataset_variability_results": "dataset_variability_results.png",
}

RANDOM_COMPLEXITY_SCRIPT = (
    PROJECT_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "02_complexity_measure" / "src" / "build_complexity_summary.py"
)
RANDOM_DATASET_PREVIEW_SCRIPT = (
    PROJECT_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "01_dataset" / "src" / "build_dataset_preview.py"
)
RANDOM_PROXY_SCRIPT = (
    PROJECT_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "05_proxy_local_entropy" / "src" / "build_six_figures.py"
)
RANDOM_QC_SCRIPT = (
    PROJECT_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "05_proxy_local_entropy" / "src" / "build_qc_figures.py"
)
RANDOM_COMPARISON_SCRIPT = (
    PROJECT_ROOT
    / "02_dnn_synthetic"
    / "06_random_baseline"
    / "05_proxy_local_entropy"
    / "src"
    / "build_six_comparison_figures.py"
)
RANDOM_COMPLEXITY_FIGURES_DIR = PROJECT_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "02_complexity_measure" / "figures"
RANDOM_DATASET_FIGURES_DIR = PROJECT_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "01_dataset" / "figures"
RANDOM_FIGURES_DIR = PROJECT_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "05_proxy_local_entropy" / "figures"
RANDOM_COMPARISON_FIGURES_DIR = RANDOM_FIGURES_DIR / "with_random_baseline"
RANDOM_FIGURE_OUTPUTS = {
    "dnn_synthetic_random_baseline_dataset_example": (
        RANDOM_DATASET_FIGURES_DIR,
        "gaussian_random_dataset_001_example.png",
    ),
    "dnn_synthetic_random_baseline_complexity_summary": (
        RANDOM_COMPLEXITY_FIGURES_DIR,
        "random_baseline_complexity_summary.png",
    ),
    "dnn_synthetic_random_baseline_phi_d_curve": (RANDOM_FIGURES_DIR, "phi_d_curve.png"),
    "dnn_synthetic_random_baseline_phi_energetic_d_curve": (RANDOM_FIGURES_DIR, "phi_energetic_d_curve.png"),
    "dnn_synthetic_random_baseline_derivative_phi_d_curve": (RANDOM_FIGURES_DIR, "derivative_phi_d_curve.png"),
    "dnn_synthetic_random_baseline_derivative_phi_energetic_d_curve": (
        RANDOM_FIGURES_DIR,
        "derivative_phi_energetic_d_curve.png",
    ),
    "dnn_synthetic_random_baseline_phase_like_A_by_beta": (RANDOM_FIGURES_DIR, "phase_like_A_by_beta.png"),
    "dnn_synthetic_random_baseline_phase_like_A_by_complexity": (
        RANDOM_FIGURES_DIR,
        "phase_like_A_by_complexity.png",
    ),
    "dnn_synthetic_random_baseline_logZ_split_qc_results": (RANDOM_FIGURES_DIR, "logZ_split_qc_results.png"),
    "dnn_synthetic_random_baseline_reference_variability_results": (
        RANDOM_FIGURES_DIR,
        "reference_variability_results.png",
    ),
    "dnn_synthetic_random_baseline_dataset_variability_results": (
        RANDOM_FIGURES_DIR,
        "dataset_variability_results.png",
    ),
    "dnn_synthetic_with_random_baseline_phi_d_curve": (
        RANDOM_COMPARISON_FIGURES_DIR,
        "phi_d_curve_with_random_baseline.png",
    ),
    "dnn_synthetic_with_random_baseline_phi_energetic_d_curve": (
        RANDOM_COMPARISON_FIGURES_DIR,
        "phi_energetic_d_curve_with_random_baseline.png",
    ),
    "dnn_synthetic_with_random_baseline_derivative_phi_d_curve": (
        RANDOM_COMPARISON_FIGURES_DIR,
        "derivative_phi_d_curve_with_random_baseline.png",
    ),
    "dnn_synthetic_with_random_baseline_derivative_phi_energetic_d_curve": (
        RANDOM_COMPARISON_FIGURES_DIR,
        "derivative_phi_energetic_d_curve_with_random_baseline.png",
    ),
    "dnn_synthetic_with_random_baseline_phase_like_A_by_beta": (
        RANDOM_COMPARISON_FIGURES_DIR,
        "phase_like_A_by_beta_with_random_baseline.png",
    ),
    "dnn_synthetic_with_random_baseline_phase_like_A_by_complexity": (
        RANDOM_COMPARISON_FIGURES_DIR,
        "phase_like_A_by_complexity_with_random_baseline.png",
    ),
}

MNIST_LABEL_DATASET_SCRIPT = PROJECT_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "01_dataset" / "src" / "build_dataset_summary.py"
MNIST_LABEL_COMPLEXITY_SCRIPT = (
    PROJECT_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "02_complexity_measure" / "src" / "build_complexity_summary.py"
)
MNIST_LABEL_PROXY_SCRIPT = (
    PROJECT_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "05_proxy_local_entropy" / "src" / "build_six_figures.py"
)
MNIST_LABEL_QC_SCRIPT = (
    PROJECT_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "05_proxy_local_entropy" / "src" / "build_qc_figures.py"
)
MNIST_LABEL_DATASET_FIGURES_DIR = PROJECT_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "01_dataset" / "figures"
MNIST_LABEL_COMPLEXITY_FIGURES_DIR = PROJECT_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "02_complexity_measure" / "figures"
MNIST_LABEL_FIGURES_DIR = PROJECT_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "05_proxy_local_entropy" / "figures"
MNIST_LABEL_FIGURE_OUTPUTS = {
    "dnn_mnist_label_noise_dataset_eta_sweep": (MNIST_LABEL_DATASET_FIGURES_DIR, "label_noise_eta_sweep.png"),
    "dnn_mnist_label_noise_complexity_by_eta": (
        MNIST_LABEL_COMPLEXITY_FIGURES_DIR,
        "label_noise_complexity_by_eta.png",
    ),
    "dnn_mnist_label_noise_phi_d_curve": (MNIST_LABEL_FIGURES_DIR, "phi_d_curve.png"),
    "dnn_mnist_label_noise_phi_energetic_d_curve": (MNIST_LABEL_FIGURES_DIR, "phi_energetic_d_curve.png"),
    "dnn_mnist_label_noise_derivative_phi_d_curve": (MNIST_LABEL_FIGURES_DIR, "derivative_phi_d_curve.png"),
    "dnn_mnist_label_noise_derivative_phi_energetic_d_curve": (
        MNIST_LABEL_FIGURES_DIR,
        "derivative_phi_energetic_d_curve.png",
    ),
    "dnn_mnist_label_noise_phase_like_A_by_eta": (MNIST_LABEL_FIGURES_DIR, "phase_like_A_by_eta.png"),
    "dnn_mnist_label_noise_phase_like_A_by_complexity": (
        MNIST_LABEL_FIGURES_DIR,
        "phase_like_A_by_complexity.png",
    ),
    "dnn_mnist_label_noise_logZ_split_qc_results": (MNIST_LABEL_FIGURES_DIR, "logZ_split_qc_results.png"),
    "dnn_mnist_label_noise_reference_variability_results": (
        MNIST_LABEL_FIGURES_DIR,
        "reference_variability_results.png",
    ),
}

MNIST_MANUAL_PROXY_SCRIPT = (
    PROJECT_ROOT / "03_dnn_mnist" / "manual_rules" / "05_proxy_local_entropy" / "src" / "build_six_figures.py"
)
MNIST_MANUAL_QC_SCRIPT = (
    PROJECT_ROOT / "03_dnn_mnist" / "manual_rules" / "05_proxy_local_entropy" / "src" / "build_qc_figures.py"
)
MNIST_MANUAL_FIGURES_DIR = PROJECT_ROOT / "03_dnn_mnist" / "manual_rules" / "05_proxy_local_entropy" / "figures"
MNIST_MANUAL_FIGURE_OUTPUTS = {
    "dnn_mnist_manual_rules_phi_d_curve": (MNIST_MANUAL_FIGURES_DIR, "phi_d_curve.png"),
    "dnn_mnist_manual_rules_phi_energetic_d_curve": (MNIST_MANUAL_FIGURES_DIR, "phi_energetic_d_curve.png"),
    "dnn_mnist_manual_rules_derivative_phi_d_curve": (
        MNIST_MANUAL_FIGURES_DIR,
        "derivative_phi_d_curve.png",
    ),
    "dnn_mnist_manual_rules_derivative_phi_energetic_d_curve": (
        MNIST_MANUAL_FIGURES_DIR,
        "derivative_phi_energetic_d_curve.png",
    ),
    "dnn_mnist_manual_rules_phase_like_A_by_rule": (MNIST_MANUAL_FIGURES_DIR, "phase_like_A_by_rule.png"),
    "dnn_mnist_manual_rules_phase_like_A_by_complexity": (
        MNIST_MANUAL_FIGURES_DIR,
        "phase_like_A_by_complexity.png",
    ),
    "dnn_mnist_manual_rules_logZ_split_qc_results": (MNIST_MANUAL_FIGURES_DIR, "logZ_split_qc_results.png"),
    "dnn_mnist_manual_rules_reference_variability_results": (
        MNIST_MANUAL_FIGURES_DIR,
        "reference_variability_results.png",
    ),
}


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def analytic_relative_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_rel" in frame.columns:
        return frame["phi_rel"]
    ordered = frame.sort_values("r")
    return ordered["phi"] - ordered["phi"].iloc[0]


def sampling_relative_phi(frame: pd.DataFrame) -> pd.Series:
    if "phi_emp_rel" in frame.columns:
        return frame["phi_emp_rel"]
    ordered = frame.sort_values("r")
    return ordered["phi_emp"] - ordered["phi_emp"].iloc[0]


def write_analytic_figure(analytic: pd.DataFrame, output: Path) -> None:
    analytic = analytic.sort_values("r")
    plt.figure(figsize=(7.4, 4.8))
    plt.plot(analytic["r"], analytic_relative_phi(analytic), color="black", linewidth=2.2)
    plt.xlabel("d")
    plt.ylabel("phi(d) - phi(d0)")
    plt.title("Analytic full-RS phi(d), alpha=0.1")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(output, dpi=240)
    plt.close()


def write_sampling_figure(sampling: pd.DataFrame, output: Path) -> None:
    sampling = sampling.sort_values(["N", "r"])
    plt.figure(figsize=(7.4, 4.8))
    for n_value, group in sampling.groupby("N", sort=True):
        group = group.sort_values("r")
        plt.plot(group["r"], sampling_relative_phi(group), marker="o", linewidth=1.7, label=f"N={int(n_value)}")
    plt.xlabel("d")
    plt.ylabel("empirical phi(d) - phi(d0)")
    plt.title("Two-pool shell sampling, alpha=0.1")
    plt.grid(True, alpha=0.28)
    plt.legend(title="system size", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=240)
    plt.close()


def write_combined_figure(analytic: pd.DataFrame, sampling: pd.DataFrame, output: Path) -> None:
    analytic = analytic.sort_values("r")
    sampling = sampling.sort_values(["N", "r"])
    plt.figure(figsize=(8.5, 5.1))
    plt.plot(
        analytic["r"],
        analytic_relative_phi(analytic),
        color="black",
        linewidth=2.4,
        label="analytic full-RS",
    )
    for n_value, group in sampling.groupby("N", sort=True):
        group = group.sort_values("r")
        plt.plot(
            group["r"],
            sampling_relative_phi(group),
            marker="o",
            markersize=3.2,
            linewidth=1.55,
            label=f"N={int(n_value)} sampling",
        )
    plt.xlabel("d")
    plt.ylabel("phi(d) - phi(d0)")
    plt.title("Analytic vs shell sampling, alpha=0.1")
    plt.grid(True, alpha=0.28)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def rebuild_figures() -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    analytic = read_required_csv(ANALYTIC_CSV)
    sampling = read_required_csv(SAMPLING_CSV)
    write_analytic_figure(analytic, OUTPUTS["analytic"])
    write_sampling_figure(sampling, OUTPUTS["sampling"])
    write_combined_figure(analytic, sampling, OUTPUTS["combined"])
    write_combined_figure(analytic, sampling, OUTPUTS["theory_stage_combined"])
    outputs = list(OUTPUTS.values())

    runpy.run_path(str(DNN_PROXY_SCRIPT), run_name="__main__")
    runpy.run_path(str(DNN_QC_SCRIPT), run_name="__main__")
    for output_stem, source_name in DNN_FIGURE_OUTPUTS.items():
        source = DNN_FIGURES_DIR / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        target = FIGURES_DIR / f"{output_stem}.png"
        shutil.copy2(source, target)
        outputs.append(target)

    runpy.run_path(str(RANDOM_DATASET_PREVIEW_SCRIPT), run_name="__main__")
    runpy.run_path(str(RANDOM_COMPLEXITY_SCRIPT), run_name="__main__")
    runpy.run_path(str(RANDOM_PROXY_SCRIPT), run_name="__main__")
    runpy.run_path(str(RANDOM_QC_SCRIPT), run_name="__main__")
    runpy.run_path(str(RANDOM_COMPARISON_SCRIPT), run_name="__main__")
    for output_stem, (source_dir, source_name) in RANDOM_FIGURE_OUTPUTS.items():
        source = source_dir / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        target = FIGURES_DIR / f"{output_stem}.png"
        shutil.copy2(source, target)
        outputs.append(target)

    runpy.run_path(str(MNIST_LABEL_DATASET_SCRIPT), run_name="__main__")
    runpy.run_path(str(MNIST_LABEL_COMPLEXITY_SCRIPT), run_name="__main__")
    runpy.run_path(str(MNIST_LABEL_PROXY_SCRIPT), run_name="__main__")
    runpy.run_path(str(MNIST_LABEL_QC_SCRIPT), run_name="__main__")
    for output_stem, (source_dir, source_name) in MNIST_LABEL_FIGURE_OUTPUTS.items():
        source = source_dir / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        target = FIGURES_DIR / f"{output_stem}.png"
        shutil.copy2(source, target)
        outputs.append(target)

    runpy.run_path(str(MNIST_MANUAL_PROXY_SCRIPT), run_name="__main__")
    runpy.run_path(str(MNIST_MANUAL_QC_SCRIPT), run_name="__main__")
    for output_stem, (source_dir, source_name) in MNIST_MANUAL_FIGURE_OUTPUTS.items():
        source = source_dir / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        target = FIGURES_DIR / f"{output_stem}.png"
        shutil.copy2(source, target)
        outputs.append(target)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate only the top-level Complexity/Figures outputs from summarized CSVs."
    )
    parser.parse_args()
    for output in rebuild_figures():
        print(output)


if __name__ == "__main__":
    main()
