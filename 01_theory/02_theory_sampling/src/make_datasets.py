from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_OUTPUT_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "raw_outputs"
DEFAULT_OUTPUT_ROOT = RAW_OUTPUT_ROOT / "dataset_pool"
DEFAULT_N_VALUES = (40, 80, 160, 320)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def dataset_seed(n_value: int, dataset_id: int) -> int:
    return int(20_274_541 + 1_000_003 * int(n_value) + 9_176 * int(dataset_id))


def make_signed_gaussian_dataset(*, n_value: int, alpha: float, seed: int) -> np.ndarray:
    m_value = int(round(float(alpha) * int(n_value)))
    rng = np.random.default_rng(int(seed))
    patterns = rng.normal(size=(m_value, int(n_value)))
    labels = rng.choice([-1.0, 1.0], size=m_value)
    return (labels[:, None] * patterns).astype(np.float64)


def write_dataset(path: Path, *, a_matrix: np.ndarray, seed: int, alpha: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, A=a_matrix.astype(np.float64), seed=np.asarray(seed), alpha=np.asarray(alpha))
    tmp.replace(path)


def build_dataset_pool(
    *,
    output_root: Path,
    n_values: tuple[int, ...],
    dataset_count: int,
    alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_value in n_values:
        m_value = int(round(float(alpha) * int(n_value)))
        for dataset_id in range(int(dataset_count)):
            seed = dataset_seed(int(n_value), int(dataset_id))
            a_matrix = make_signed_gaussian_dataset(n_value=int(n_value), alpha=float(alpha), seed=seed)
            dataset_name = f"dataset_{int(dataset_id) + 1:03d}"
            dataset_path = output_root / f"N_{int(n_value)}" / dataset_name / "dataset.npz"
            write_dataset(dataset_path, a_matrix=a_matrix, seed=seed, alpha=float(alpha))
            rows.append(
                {
                    "N": int(n_value),
                    "M": int(m_value),
                    "dataset_id": int(dataset_id),
                    "dataset_name": dataset_name,
                    "alpha": float(alpha),
                    "seed": int(seed),
                    "dataset_path": str(dataset_path.resolve().relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the theory signed-Gaussian perceptron dataset pool.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-values", type=str, default=",".join(str(value) for value in DEFAULT_N_VALUES))
    parser.add_argument("--dataset-count", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    output_root = project_path(args.output_root).resolve()
    rows = build_dataset_pool(
        output_root=output_root,
        n_values=parse_ints(args.n_values),
        dataset_count=int(args.dataset_count),
        alpha=float(args.alpha),
    )
    print(output_root)
    print(f"dataset_count={len(rows)}")


if __name__ == "__main__":
    main()
