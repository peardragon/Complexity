from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "01_dataset"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
RAW_ROOT = STAGE_ROOT / "raw_outputs"
FIGURE_ROOT = STAGE_ROOT / "figures"
SPEC_PATH = SUMMARY_ROOT / "eta_dataset_spec.json"


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    eta_values = [float(value) for value in spec["eta_values"]]

    rows = [
        {
            "noise_eta": f"noise_eta_{str(value).replace('.', 'p')}",
            "eta": value,
            "n_train": int(spec["n_train"]),
            "input_shape": spec["input_shape"],
            "dataset_path": f"03_dnn_mnist/label_noise_sweep/01_dataset/raw_outputs/noise_eta_{str(value).replace('.', 'p')}/dataset.npz",
            "raw_payload_status": "present",
        }
        for value in eta_values
    ]
    with (SUMMARY_ROOT / "eta_dataset_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    raw_manifest = {
        "stage": "01_dataset",
        "dataset_policy": spec["dataset_policy"],
        "raw_payload_status": "present",
        "canonical_noise_eta_values": [row["noise_eta"] for row in rows],
        "dataset_payloads": {
            row["noise_eta"]: row["dataset_path"]
            for row in rows
        },
        "downstream_sampling": "03_dnn_mnist/label_noise_sweep/04_sampling/raw_outputs/shell_pool/noise_eta_*",
    }
    (RAW_ROOT / "raw_payload_manifest.json").write_text(json.dumps(raw_manifest, indent=2) + "\n", encoding="utf-8")

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(eta_values, [value * 100.0 for value in eta_values], "o-", color="#2364aa", linewidth=1.6)
    ax.set_xlabel("label noise eta")
    ax.set_ylabel("expected flipped labels (%)")
    ax.set_title("MNIST label-noise sweep")
    ax.grid(True, alpha=0.25)
    fig.savefig(FIGURE_ROOT / "label_noise_eta_sweep.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
