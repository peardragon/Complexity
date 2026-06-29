from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROOT = STAGE_ROOT.parent
RULE_MAPPING = MANUAL_ROOT / "config" / "rule_mapping.csv"
RAW_ROOT = STAGE_ROOT / "raw_outputs"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_ROOT = STAGE_ROOT / "figures"


def _label_balance(y: np.ndarray) -> str:
    values, counts = np.unique(y, return_counts=True)
    return ";".join(f"{int(v)}:{int(c)}" for v, c in zip(values, counts))


def build_dataset_index() -> pd.DataFrame:
    rules = pd.read_csv(RULE_MAPPING)
    rows: list[dict[str, object]] = []
    for row in rules.itertuples(index=False):
        path = RAW_ROOT / row.rule_id / "dataset.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        rows.append(
            {
                "rule_id": row.rule_id,
                "rule_name": row.rule_name,
                "label": row.label,
                "dataset_path": f"Complexity/03_dnn_mnist/manual_rules/01_dataset/raw_outputs/{row.rule_id}/dataset.npz",
                "n_train": int(data["X_train"].shape[0]),
                "n_test": int(data["X_test"].shape[0]),
                "feature_dim": int(data["X_train"].shape[1]),
                "train_label_balance": _label_balance(data["y_train"]),
                "test_label_balance": _label_balance(data["y_test"]),
                "keys": ";".join(data.files),
            }
        )
        metadata = {
            "rule_id": row.rule_id,
            "rule_name": row.rule_name,
            "label": row.label,
            "dataset_path": (
                "Complexity/03_dnn_mnist/manual_rules/01_dataset/raw_outputs/"
                f"{row.rule_id}/dataset.npz"
            ),
            "n_train": int(data["X_train"].shape[0]),
            "n_test": int(data["X_test"].shape[0]),
            "feature_dim": int(data["X_train"].shape[1]),
            "keys": list(data.files),
        }
        for name in ["dataset_metadata.json", "canonical_dataset_metadata.json"]:
            (RAW_ROOT / row.rule_id / name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    out = pd.DataFrame(rows)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUMMARY_ROOT / "dataset_index.csv", index=False)
    return out


def build_sample_figure() -> Path:
    rules = pd.read_csv(RULE_MAPPING)
    fig, axes = plt.subplots(len(rules), 8, figsize=(9.6, 4.8), constrained_layout=True)
    for r_idx, row in enumerate(rules.itertuples(index=False)):
        data = np.load(RAW_ROOT / row.rule_id / "dataset.npz")
        images = data["X_train_raw10"][:8].reshape(-1, 10, 10)
        labels = data["y_train"][:8]
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            ax.imshow(images[c_idx], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if c_idx == 0:
                ax.set_ylabel(row.label, fontsize=9)
            ax.set_title(str(int(labels[c_idx])), fontsize=8)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURE_ROOT / "sample_figure.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    print(SUMMARY_ROOT / "dataset_index.csv")
    build_dataset_index()
    print(build_sample_figure())


if __name__ == "__main__":
    main()
