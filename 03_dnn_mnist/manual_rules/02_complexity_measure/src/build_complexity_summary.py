from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROOT = STAGE_ROOT.parent
RULE_MAPPING = MANUAL_ROOT / "config" / "rule_mapping.csv"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_ROOT = STAGE_ROOT / "figures"
NMS_TV = {
    "very_low_tv_spectral_teacher": 0.3245703473792008,
    "real_even_odd": 0.4932864276461805,
    "teacher_nn": 0.6843772639598127,
    "random_label": 0.985558573825462,
}


def build_complexity_summary() -> pd.DataFrame:
    rules = pd.read_csv(RULE_MAPPING)
    rules["rule_order"] = range(1, len(rules) + 1)
    rules["nmstv_mean"] = rules["rule_name"].map(NMS_TV)
    out = rules[["rule_id", "rule_name", "label", "rule_order", "nmstv_mean"]].copy()
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUMMARY_ROOT / "complexity_by_rule.csv", index=False)
    return out


def build_figure(frame: pd.DataFrame) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURE_ROOT / "manual_rule_complexity_by_rule.png"
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot(frame["rule_order"], frame["nmstv_mean"], marker="o", linewidth=1.6, color="#4C78A8")
    ax.set_xticks(frame["rule_order"])
    ax.set_xticklabels(frame["label"], rotation=20, ha="right")
    ax.set_xlabel("manual rule")
    ax.set_ylabel("3-NN MNIST complexity")
    ax.set_title("Manual-rule complexity axis")
    ax.grid(True, alpha=0.24)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    frame = build_complexity_summary()
    print(SUMMARY_ROOT / "complexity_by_rule.csv")
    print(build_figure(frame))


if __name__ == "__main__":
    main()
