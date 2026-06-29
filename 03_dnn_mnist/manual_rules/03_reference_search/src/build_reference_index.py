from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROOT = STAGE_ROOT.parent
RULE_MAPPING = MANUAL_ROOT / "config" / "rule_mapping.csv"
RAW_ROOT = STAGE_ROOT / "raw_outputs"


def build_reference_index() -> pd.DataFrame:
    rules = pd.read_csv(RULE_MAPPING)
    rows: list[dict[str, object]] = []
    for row in rules.itertuples(index=False):
        for ref_dir in sorted((RAW_ROOT / row.rule_id).glob("ref_*")):
            theta_path = ref_dir / "theta.npy"
            metadata_path = ref_dir / "reference_metadata.json"
            if not theta_path.exists():
                raise FileNotFoundError(theta_path)
            theta = np.load(theta_path)
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            canonical_metadata = {
                "rule_id": row.rule_id,
                "rule_name": row.rule_name,
                "label": row.label,
                "ref_id": ref_dir.name,
                "theta_path": (
                    "Complexity/03_dnn_mnist/manual_rules/03_reference_search/raw_outputs/"
                    f"{row.rule_id}/{ref_dir.name}/theta.npy"
                ),
                "dataset_path": (
                    "Complexity/03_dnn_mnist/manual_rules/01_dataset/raw_outputs/"
                    f"{row.rule_id}/dataset.npz"
                ),
                "theta_shape": list(theta.shape),
                "theta_norm": float(np.linalg.norm(theta)),
                "source_ref_id": metadata.get("ref_id"),
                "source_seed": metadata.get("seed"),
            }
            metadata_path.write_text(json.dumps(canonical_metadata, indent=2) + "\n", encoding="utf-8")
            rows.append(
                {
                    "rule_id": row.rule_id,
                    "rule_name": row.rule_name,
                    "label": row.label,
                    "ref_id": ref_dir.name,
                    "theta_path": (
                        "Complexity/03_dnn_mnist/manual_rules/03_reference_search/raw_outputs/"
                        f"{row.rule_id}/{ref_dir.name}/theta.npy"
                    ),
                    "reference_metadata_path": (
                        "Complexity/03_dnn_mnist/manual_rules/03_reference_search/raw_outputs/"
                        f"{row.rule_id}/{ref_dir.name}/reference_metadata.json"
                    ),
                    "theta_shape": "x".join(str(dim) for dim in theta.shape),
                    "theta_norm": float(np.linalg.norm(theta)),
                    "source_ref_id": metadata.get("ref_id"),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RAW_ROOT / "reference_index.csv", index=False)
    return out


def main() -> None:
    print(RAW_ROOT / "reference_index.csv")
    build_reference_index()


if __name__ == "__main__":
    main()
