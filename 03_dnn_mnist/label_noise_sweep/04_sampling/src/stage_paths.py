from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT.parent
RAW_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "04_sampling" / "raw_outputs"
SHELL_POOL_ROOT = RAW_ROOT / "shell_pool"
REFERENCE_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "03_reference_search" / "raw_outputs"
DATASET_MANIFEST = (
    REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "01_dataset" / "raw_outputs" / "raw_payload_manifest.json"
)
DATASET_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "01_dataset" / "raw_outputs"
NOISE_ETAS = ("noise_eta_0p05", "noise_eta_0p15", "noise_eta_0p25")


def sample_dir(noise_eta: str, ref_id: int, radius_index: int) -> Path:
    return SHELL_POOL_ROOT / noise_eta / f"ref_{ref_id:03d}" / f"r_{radius_index / 100:.4f}".replace(".", "p")


def sample_file(noise_eta: str, ref_id: int, radius_index: int) -> Path:
    return sample_dir(noise_eta, ref_id, radius_index) / "samples.npz"


def _project_relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def validate_sampling_layout(ref_count: int = 30, radius_count: int = 100) -> None:
    missing: list[Path] = []
    for noise_eta in NOISE_ETAS:
        for ref_id in range(ref_count):
            for radius_index in range(1, radius_count + 1):
                for path in [
                    sample_file(noise_eta, ref_id, radius_index),
                    sample_dir(noise_eta, ref_id, radius_index) / "unit_summary.json",
                ]:
                    if not path.exists():
                        missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} sampling files:\n{preview}")


def write_sampling_index(ref_count: int = 30, radius_count: int = 100) -> None:
    rows: list[dict[str, object]] = []
    for noise_eta in NOISE_ETAS:
        eta = float(noise_eta.removeprefix("noise_eta_").replace("p", "."))
        for ref_id in range(ref_count):
            for radius_index in range(1, radius_count + 1):
                radius = radius_index / 100
                unit_dir = sample_dir(noise_eta, ref_id, radius_index)
                radius_dir = f"r_{radius:.4f}".replace(".", "p")
                rows.append(
                    {
                        "noise_eta": noise_eta,
                        "eta": eta,
                        "ref": f"ref_{ref_id:03d}",
                        "radius": radius,
                        "samples_path": str(
                            Path("03_dnn_mnist/label_noise_sweep/04_sampling/raw_outputs/shell_pool")
                            / noise_eta
                            / f"ref_{ref_id:03d}"
                            / radius_dir
                            / "samples.npz"
                        ),
                        "unit_summary_path": str(
                            Path("03_dnn_mnist/label_noise_sweep/04_sampling/raw_outputs/shell_pool")
                            / noise_eta
                            / f"ref_{ref_id:03d}"
                            / radius_dir
                            / "unit_summary.json"
                        ),
                    }
                )
    out = RAW_ROOT / "sampling_unit_index.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_unit_summary_metadata(ref_count: int = 30, radius_count: int = 100) -> None:
    for noise_eta in NOISE_ETAS:
        for ref_id in range(ref_count):
            theta_path = REFERENCE_ROOT / noise_eta / f"ref_{ref_id:03d}" / "theta.npy"
            for radius_index in range(1, radius_count + 1):
                unit_dir = sample_dir(noise_eta, ref_id, radius_index)
                summary_path = unit_dir / "unit_summary.json"
                payload = json.loads(summary_path.read_text(encoding="utf-8"))

                for key in ("dataset_path", "samples_path", "theta_path"):
                    source_key = f"source_{key}"
                    if key in payload and source_key not in payload:
                        payload[source_key] = payload[key]

                payload["samples_path"] = _project_relative(unit_dir / "samples.npz")
                payload["unit_summary_path"] = _project_relative(summary_path)
                payload["theta_path"] = _project_relative(theta_path)
                payload["theta_payload_exists"] = theta_path.exists()
                payload["dataset_payload_manifest"] = _project_relative(DATASET_MANIFEST)
                dataset_path = DATASET_ROOT / noise_eta / "dataset.npz"
                payload["dataset_path"] = _project_relative(dataset_path)
                payload["dataset_payload_exists"] = dataset_path.exists()
                payload["dataset_payload_status"] = "present" if dataset_path.exists() else "missing_in_current_payload"

                summary_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    validate_sampling_layout()
    normalize_unit_summary_metadata()
    write_sampling_index()
