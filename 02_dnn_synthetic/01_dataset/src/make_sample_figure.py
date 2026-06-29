from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "01_dataset"
RAW_ROOT = STAGE_ROOT / "raw_outputs" / "18_beta_cell_90_dataset" / "raw_datasets"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / "18_beta_cell_90_dataset"
FIGURE_PATH = STAGE_ROOT / "figures" / "sample_figure.png"

PANEL_SPECS = (
    ("beta=0.05", "cell_beta_0p05_p_0p00", "dataset_000_seed_001234"),
    ("beta=0.21", "cell_beta_0p21_p_0p00", "dataset_000_seed_017234"),
    ("beta=0.37", "cell_beta_0p37_p_0p00", "dataset_000_seed_033234"),
)


def main() -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(9.6, 3.4), constrained_layout=True)
    manifest_rows: list[dict[str, str]] = []
    for ax, (label, cell_id, dataset_id) in zip(axes, PANEL_SPECS):
        image_path = RAW_ROOT / cell_id / dataset_id / "region_fill_d2.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        manifest_rows.append(
            {
                "panel": label,
                "cell_id": cell_id,
                "dataset_id": dataset_id,
                "source_image": str(image_path.relative_to(REPO_ROOT)),
            }
        )

    fig.suptitle("Synthetic 3-NN dataset examples", y=1.03)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)

    manifest_path = SUMMARY_ROOT / "sample_figure_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["panel", "cell_id", "dataset_id", "source_image"])
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
