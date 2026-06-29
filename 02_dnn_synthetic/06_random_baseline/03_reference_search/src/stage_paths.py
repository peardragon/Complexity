from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_ROOT = REPO_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "03_reference_search" / "raw_outputs"


def reference_dir(dataset_id: int, ref_id: int) -> Path:
    return RAW_ROOT / f"dataset_{dataset_id:03d}" / f"ref_{ref_id:03d}"


def required_reference_files(dataset_id: int, ref_id: int) -> tuple[Path, Path]:
    ref_root = reference_dir(dataset_id, ref_id)
    return ref_root / "theta_init.npy", ref_root / "theta.npy"


def validate_reference_layout(dataset_count: int = 90, ref_count: int = 30) -> None:
    missing: list[Path] = []
    for dataset_id in range(1, dataset_count + 1):
        for ref_id in range(1, ref_count + 1):
            for path in required_reference_files(dataset_id, ref_id):
                if not path.exists():
                    missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} reference files:\n{preview}")


if __name__ == "__main__":
    validate_reference_layout()
