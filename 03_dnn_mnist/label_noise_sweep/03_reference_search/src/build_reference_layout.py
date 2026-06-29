from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT.parent
STAGE_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "03_reference_search"
RAW_ROOT = STAGE_ROOT / "raw_outputs"
DATASET_RAW_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "01_dataset" / "raw_outputs"
PROVENANCE_ROOT = RAW_ROOT / "provenance"
REFERENCE_INDEX = (
    PROVENANCE_ROOT
    / "eta_reference_search_gapfill_0p02_0p05_0p15_0p25_30ref_cpu60_gpu0"
    / "04_exact_reference_search"
    / "reference_index.csv"
)


def _noise_eta(rule: str) -> str:
    return rule.replace("eta_", "noise_eta_")


def _canonical_ref(ref_id: str) -> str:
    return f"ref_{int(ref_id) + 1:03d}"


def main() -> None:
    rows: list[dict[str, object]] = []
    with REFERENCE_INDEX.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            noise_eta = _noise_eta(row["rule"])
            ref = _canonical_ref(row["ref_id"])
            ref_dir = RAW_ROOT / noise_eta / ref
            ref_dir.mkdir(parents=True, exist_ok=True)
            local_theta = ref_dir / "theta.npy"
            local_theta_init = ref_dir / "theta_init.npy"
            local_dataset = DATASET_RAW_ROOT / noise_eta / "dataset.npz"
            source_theta = Path(row["theta_path"])
            source_dataset = Path(row["dataset_path"])
            metadata = {
                "noise_eta": noise_eta,
                "eta": float(row["eta"]),
                "ref": ref,
                "source_ref_id": int(row["ref_id"]),
                "attempt_seed": int(row["attempt_seed"]),
                "optimizer_chain": row["optimizer_chain"],
                "P": int(row["P"]),
                "train_error": float(row["train_error"]),
                "test_error": float(row["test_error"]),
                "CE_mean_train": float(row["CE_mean_train"]),
                "theta_norm": float(row["theta_norm"]),
                "theta_payload_path": str(local_theta.relative_to(RAW_ROOT)),
                "theta_payload_exists": local_theta.exists(),
                "theta_init_payload_path": str(local_theta_init.relative_to(RAW_ROOT)),
                "theta_init_payload_exists": local_theta_init.exists(),
                "dataset_payload_path": str(local_dataset.relative_to(PROJECT_ROOT)),
                "dataset_payload_exists": local_dataset.exists(),
                "source_theta_path": row["theta_path"],
                "source_theta_exists": source_theta.exists(),
                "source_dataset_path": row["dataset_path"],
                "source_dataset_exists": source_dataset.exists(),
            }
            (ref_dir / "reference_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            clean = {
                "noise_eta": noise_eta,
                "eta": float(row["eta"]),
                "ref": ref,
                "source_ref_id": int(row["ref_id"]),
                "reference_metadata": str(
                    (ref_dir / "reference_metadata.json").relative_to(PROJECT_ROOT)
                ),
                "theta_payload_status": "present" if local_theta.exists() else "missing_in_current_payload",
                "theta_init_payload_status": "present" if local_theta_init.exists() else "not_retained_in_source_payload",
                "source_theta_exists": source_theta.exists(),
                "dataset_payload_status": "present" if local_dataset.exists() else "missing_in_current_payload",
                "source_dataset_exists": source_dataset.exists(),
                "train_error": float(row["train_error"]),
                "test_error": float(row["test_error"]),
                "theta_norm": float(row["theta_norm"]),
            }
            rows.append(clean)

    with (RAW_ROOT / "reference_index_canonical.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "stage": "03_reference_search",
        "canonical_layout": "raw_outputs/noise_eta_*/ref_*/theta.npy",
        "metadata_layout": "raw_outputs/noise_eta_*/ref_*/reference_metadata.json",
        "theta_payload_status": {
            "present": sum(1 for row in rows if row["theta_payload_status"] == "present"),
            "missing_in_current_payload": sum(1 for row in rows if row["theta_payload_status"] != "present"),
        },
        "theta_init_payload_status": {
            "present": sum(1 for row in rows if row["theta_init_payload_status"] == "present"),
            "not_retained_in_source_payload": sum(
                1 for row in rows if row["theta_init_payload_status"] != "present"
            ),
        },
        "reference_count": len(rows),
        "dataset_payload_status": {
            "present": sum(1 for row in rows if row["dataset_payload_status"] == "present"),
            "missing_in_current_payload": sum(1 for row in rows if row["dataset_payload_status"] != "present"),
        },
        "noise_eta_values": sorted({row["noise_eta"] for row in rows}),
        "refs_per_noise_eta": 30,
    }
    (RAW_ROOT / "raw_payload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
