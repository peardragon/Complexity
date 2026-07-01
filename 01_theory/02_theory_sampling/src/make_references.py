from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_OUTPUT_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "raw_outputs"
DEFAULT_DATASET_ROOT = RAW_OUTPUT_ROOT / "dataset_pool"
DEFAULT_OUTPUT_ROOT = RAW_OUTPUT_ROOT / "reference_pool"
DEFAULT_N_VALUES = (40, 80, 160, 320)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def stable_softplus_neg_margin(theta_batch: np.ndarray, a_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dim = int(theta_batch.shape[1])
    margins = (theta_batch @ np.asarray(a_matrix, dtype=np.float64).T) / np.sqrt(dim)
    ce = np.logaddexp(0.0, -margins).sum(axis=1)
    err = np.mean(margins <= 0.0, axis=1)
    return ce.astype(np.float64), err.astype(np.float64)


def reference_seed(n_value: int, dataset_id: int) -> int:
    return int(91_337_219 + 1_000_033 * int(n_value) + 19_937 * int(dataset_id))


def initial_feasible_reference(a_matrix: np.ndarray) -> np.ndarray:
    dim = int(a_matrix.shape[1])
    target = np.full(int(a_matrix.shape[0]), np.sqrt(dim), dtype=np.float64)
    theta, *_ = np.linalg.lstsq(a_matrix, target, rcond=None)
    margins = (a_matrix @ theta) / np.sqrt(dim)
    if np.min(margins) <= 1.0e-8:
        raise RuntimeError("failed to construct an interior feasible reference")
    return theta.astype(np.float64)


def sample_truncated_normal(mean: float, lower: float, upper: float, rng: np.random.Generator) -> float:
    lo = ndtr(float(lower) - float(mean)) if np.isfinite(lower) else 0.0
    hi = ndtr(float(upper) - float(mean)) if np.isfinite(upper) else 1.0
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    if hi <= lo:
        return float(np.clip(mean, lower, upper))
    u = float(rng.uniform(lo, hi))
    u = float(np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
    return float(mean + ndtri(u))


def hit_and_run_references(
    a_matrix: np.ndarray,
    *,
    reference_count: int,
    burn: int,
    thin: int,
    seed: int,
    margin_epsilon: float = 1.0e-10,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    theta = initial_feasible_reference(a_matrix)
    dim = int(a_matrix.shape[1])
    raw_margin = float(margin_epsilon) * np.sqrt(dim)
    total_steps = int(burn) + int(reference_count) * int(thin)
    refs: list[np.ndarray] = []
    for step in range(total_steps):
        direction = rng.normal(size=dim)
        direction /= np.linalg.norm(direction)
        current = a_matrix @ theta
        slope = a_matrix @ direction
        lower = -np.inf
        upper = np.inf
        positive = slope > 1.0e-14
        negative = slope < -1.0e-14
        if np.any(positive):
            lower = max(lower, float(np.max((raw_margin - current[positive]) / slope[positive])))
        if np.any(negative):
            upper = min(upper, float(np.min((raw_margin - current[negative]) / slope[negative])))
        if lower >= upper:
            theta = initial_feasible_reference(a_matrix)
            continue
        step_size = sample_truncated_normal(float(-theta @ direction), lower, upper, rng)
        theta = theta + step_size * direction
        if step >= int(burn) and (step - int(burn)) % int(thin) == 0:
            refs.append(theta.astype(np.float64).copy())
    return np.asarray(refs[: int(reference_count)], dtype=np.float64)


def summarize_references(
    *,
    n_value: int,
    dataset_id: int,
    references: np.ndarray,
    a_matrix: np.ndarray,
    method: str,
    burn: int,
    thin: int,
    lambda_ref: float,
) -> list[dict[str, object]]:
    dim = int(a_matrix.shape[1])
    margins = (references @ a_matrix.T) / np.sqrt(dim)
    ce, err = stable_softplus_neg_margin(references, a_matrix)
    rows: list[dict[str, object]] = []
    for ref_id, theta in enumerate(references):
        ref_name = f"ref_{int(ref_id) + 1:03d}"
        rows.append(
            {
                "N": int(n_value),
                "M": int(a_matrix.shape[0]),
                "dataset_id": int(dataset_id),
                "dataset_name": f"dataset_{int(dataset_id) + 1:03d}",
                "ref_id": int(ref_id),
                "ref_name": ref_name,
                "lambda_ref": float(lambda_ref),
                "reference_method": method,
                "burn": int(burn),
                "thin": int(thin),
                "ref_norm": float(np.linalg.norm(theta)),
                "ref_norm_sq": float(theta @ theta),
                "min_margin": float(np.min(margins[ref_id])),
                "mean_margin": float(np.mean(margins[ref_id])),
                "ce_ref": float(ce[ref_id]),
                "err_ref": float(err[ref_id]),
            }
        )
    return rows


def load_imported_references(import_from: Path, *, n_value: int, dataset_id: int) -> np.ndarray:
    refs: list[np.ndarray] = []
    dataset_dir = import_from / f"N_{int(n_value)}" / f"dataset_{int(dataset_id) + 1:03d}"
    for ref_dir in sorted(dataset_dir.glob("ref_*")):
        path = ref_dir / "reference.npz"
        if path.exists():
            refs.append(np.load(path)["theta"].astype(np.float64))
    if refs:
        return np.asarray(refs, dtype=np.float64)
    raise FileNotFoundError(dataset_dir)


def copy_or_generate_references(
    *,
    dataset_root: Path,
    output_root: Path,
    n_values: tuple[int, ...],
    dataset_count: int,
    reference_count: int,
    burn: int,
    thin: int,
    lambda_ref: float,
    import_from: Path | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_value in n_values:
        for dataset_id in range(int(dataset_count)):
            n_dir = f"N_{int(n_value)}"
            dataset_name = f"dataset_{int(dataset_id) + 1:03d}"
            dataset_dir = dataset_root / n_dir / dataset_name
            dataset_path = dataset_dir / "dataset.npz"
            if not dataset_path.exists():
                raise FileNotFoundError(dataset_path)
            a_matrix = np.load(dataset_path)["A"].astype(np.float64)
            if import_from is not None:
                references = load_imported_references(import_from, n_value=int(n_value), dataset_id=int(dataset_id))
                method = "weight_space_hit_and_run"
            else:
                references = hit_and_run_references(
                    a_matrix,
                    reference_count=int(reference_count),
                    burn=int(burn),
                    thin=int(thin),
                    seed=reference_seed(int(n_value), int(dataset_id)),
                )
                method = "weight_space_hit_and_run_truncated_gaussian"
            for ref_id, theta in enumerate(references[: int(reference_count)]):
                out_path = output_root / n_dir / dataset_name / f"ref_{int(ref_id) + 1:03d}" / "reference.npz"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_name(out_path.name + ".tmp")
                with tmp.open("wb") as handle:
                    np.savez_compressed(handle, theta=np.asarray(theta, dtype=np.float64))
                tmp.replace(out_path)
            rows.extend(
                summarize_references(
                    n_value=int(n_value),
                    dataset_id=int(dataset_id),
                    references=references[: int(reference_count)],
                    a_matrix=a_matrix,
                    method=method,
                    burn=int(burn),
                    thin=int(thin),
                    lambda_ref=float(lambda_ref),
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or import the theory perceptron reference pool.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--import-from", type=Path, default=None)
    parser.add_argument("--n-values", type=str, default=",".join(str(value) for value in DEFAULT_N_VALUES))
    parser.add_argument("--dataset-count", type=int, default=10)
    parser.add_argument("--reference-count", type=int, default=10)
    parser.add_argument("--burn", type=int, default=5000)
    parser.add_argument("--thin", type=int, default=200)
    parser.add_argument("--lambda-ref", type=float, default=1.0)
    args = parser.parse_args()

    rows = copy_or_generate_references(
        dataset_root=project_path(args.dataset_root).resolve(),
        output_root=project_path(args.output_root).resolve(),
        n_values=parse_ints(args.n_values),
        dataset_count=int(args.dataset_count),
        reference_count=int(args.reference_count),
        burn=int(args.burn),
        thin=int(args.thin),
        lambda_ref=float(args.lambda_ref),
        import_from=project_path(args.import_from).resolve() if args.import_from is not None else None,
    )
    print(project_path(args.output_root).resolve())
    print(f"reference_count={len(rows)}")


if __name__ == "__main__":
    main()
