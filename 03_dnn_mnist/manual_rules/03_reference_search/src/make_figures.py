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
SUMMARY_PATH = FIGURE_INPUT_ROOT / "reference_quality_by_rule.csv"
PER_REF_PATH = FIGURE_INPUT_ROOT / "reference_quality_by_ref.csv"
FIGURE_PATH = FIGURE_ROOT / "reference_quality_by_rule.png"


def _rule_label(rule: str) -> str:
    return str(rule).replace("_", " ")


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SUMMARY_PATH.exists() or not PER_REF_PATH.exists():
        raise FileNotFoundError("run src/make_summarized_outputs.py first")
    return pd.read_csv(SUMMARY_PATH), pd.read_csv(PER_REF_PATH)


def build_figures() -> None:
    summary, per_ref = _read_inputs()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    labels = [_rule_label(value) for value in summary["rule"].astype(str)]
    x = np.arange(len(summary), dtype=float)
    test_mean = pd.to_numeric(summary["test_error_mean"], errors="coerce").to_numpy(dtype=float)
    test_sem = pd.to_numeric(summary["test_error_sem"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    norm_mean = pd.to_numeric(summary["theta_norm_mean"], errors="coerce").to_numpy(dtype=float)
    norm_sem = pd.to_numeric(summary["theta_norm_sem"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    axes[0].errorbar(x, test_mean, yerr=test_sem, fmt="o", color="#284f8f", ecolor="#8aa7d6", capsize=3)
    for idx, rule in enumerate(summary["rule"].astype(str)):
        sub = per_ref[per_ref["rule"].astype(str).eq(rule)]
        y = pd.to_numeric(sub["test_error"], errors="coerce").to_numpy(dtype=float)
        jitter = np.linspace(-0.13, 0.13, len(y), dtype=float) if len(y) else np.asarray([], dtype=float)
        axes[0].scatter(np.full(len(y), idx, dtype=float) + jitter, y, s=14, color="#284f8f", alpha=0.22, linewidth=0)
    axes[0].set_ylabel("test error")
    axes[0].set_title("Reference generalization")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].errorbar(x, norm_mean, yerr=norm_sem, fmt="o", color="#3a7f59", ecolor="#9bc3ac", capsize=3)
    for idx, rule in enumerate(summary["rule"].astype(str)):
        sub = per_ref[per_ref["rule"].astype(str).eq(rule)]
        y = pd.to_numeric(sub["theta_norm"], errors="coerce").to_numpy(dtype=float)
        jitter = np.linspace(-0.13, 0.13, len(y), dtype=float) if len(y) else np.asarray([], dtype=float)
        axes[1].scatter(np.full(len(y), idx, dtype=float) + jitter, y, s=14, color="#3a7f59", alpha=0.22, linewidth=0)
    axes[1].set_ylabel("theta L2 norm")
    axes[1].set_title("Reference norm spread")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    build_figures()
    print(f"figure={FIGURE_PATH}")


if __name__ == "__main__":
    main()
