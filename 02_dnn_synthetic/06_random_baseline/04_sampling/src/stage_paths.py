from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_ROOT = REPO_ROOT / "02_dnn_synthetic" / "06_random_baseline" / "04_sampling" / "raw_outputs"
SHELL_POOL_ROOT = RAW_ROOT / "shell_pool"


def sample_dir(dataset_id: int, ref_id: int, radius_name: str) -> Path:
    return SHELL_POOL_ROOT / f"dataset_{dataset_id:03d}" / f"ref_{ref_id:03d}" / radius_name


def sample_file(dataset_id: int, ref_id: int, radius_name: str) -> Path:
    return sample_dir(dataset_id, ref_id, radius_name) / "samples.npz"


def validate_sampling_layout(dataset_count: int = 90, ref_count: int = 30, radius_count: int = 250) -> None:
    missing: list[Path] = []
    expected_radii = [f"r_{index / 100:.2f}".replace(".", "p") for index in range(1, radius_count + 1)]
    for dataset_id in range(1, dataset_count + 1):
        for ref_id in range(1, ref_count + 1):
            for radius_name in expected_radii:
                path = sample_file(dataset_id, ref_id, radius_name)
                if not path.exists():
                    missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} sample files:\n{preview}")


if __name__ == "__main__":
    validate_sampling_layout()
