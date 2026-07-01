from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "reference_quality"
FIGURE_ROOT = STAGE_ROOT / "figures"
FIGURE_SUMMARY_PATH = FIGURE_INPUT_ROOT / "reference_quality_by_eta.csv"
FIGURE_PER_REF_PATH = FIGURE_INPUT_ROOT / "reference_quality_by_ref.csv"
FIGURE_PATH = FIGURE_ROOT / "reference_quality_by_eta.png"


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FIGURE_SUMMARY_PATH.exists() or not FIGURE_PER_REF_PATH.exists():
        raise FileNotFoundError("run src/make_summarized_outputs.py first")
    return pd.read_csv(FIGURE_SUMMARY_PATH), pd.read_csv(FIGURE_PER_REF_PATH)


def build_figures() -> None:
    summary, per_ref = _read_inputs()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    eta = pd.to_numeric(summary["eta"], errors="coerce").to_numpy(dtype=float)
    test_mean = pd.to_numeric(summary["test_error_mean"], errors="coerce").to_numpy(dtype=float)
    test_sem = pd.to_numeric(summary["test_error_sem"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    norm_mean = pd.to_numeric(summary["theta_norm_mean"], errors="coerce").to_numpy(dtype=float)
    norm_sem = pd.to_numeric(summary["theta_norm_sem"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)

    axes[0].errorbar(eta, test_mean, yerr=test_sem, fmt="o-", color="#284f8f", ecolor="#8aa7d6", capsize=3)
    for eta_value, sub in per_ref.groupby("eta", sort=True):
        x = np.full(len(sub), float(eta_value), dtype=float)
        y = pd.to_numeric(sub["test_error"], errors="coerce").to_numpy(dtype=float)
        axes[0].scatter(x, y, s=14, color="#284f8f", alpha=0.20, linewidth=0)
    axes[0].set_xlabel("label noise eta")
    axes[0].set_ylabel("test error")
    axes[0].set_title("Reference generalization")
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(eta, norm_mean, yerr=norm_sem, fmt="o-", color="#7a4f1d", ecolor="#c49b6b", capsize=3)
    for eta_value, sub in per_ref.groupby("eta", sort=True):
        x = np.full(len(sub), float(eta_value), dtype=float)
        y = pd.to_numeric(sub["theta_norm"], errors="coerce").to_numpy(dtype=float)
        axes[1].scatter(x, y, s=14, color="#7a4f1d", alpha=0.20, linewidth=0)
    axes[1].set_xlabel("label noise eta")
    axes[1].set_ylabel("theta norm")
    axes[1].set_title("Reference norm")
    axes[1].grid(True, alpha=0.25)

    fig.savefig(FIGURE_PATH, dpi=240)
    plt.close(fig)


def main() -> None:
    build_figures()
    print(f"figure={FIGURE_PATH}")


if __name__ == "__main__":
    main()
