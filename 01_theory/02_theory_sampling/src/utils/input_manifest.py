"""Build and verify the immutable copied perceptron input manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
STAGE_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling"
DEFAULT_DATASET_ROOT = STAGE_ROOT / "raw_outputs" / "dataset_pool"
DEFAULT_REFERENCE_ROOT = STAGE_ROOT / "raw_outputs" / "reference_pool"
DEFAULT_MANIFEST = STAGE_ROOT / "raw_outputs" / "manifests" / "input_files.sha256"
DEFAULT_SUMMARY = (
    STAGE_ROOT / "raw_outputs" / "manifests" / "input_manifest_summary.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def expected_paths(
    dataset_root: Path, reference_root: Path
) -> tuple[list[Path], list[Path]]:
    datasets: list[Path] = []
    references: list[Path] = []
    for dimension in (40, 80, 160, 320):
        for dataset_id in range(10):
            datasets.append(
                dataset_root
                / f"N_{dimension}"
                / f"dataset_{dataset_id + 1:03d}"
                / "dataset.npz"
            )
            for reference_id in range(10):
                references.append(
                    reference_root
                    / f"N_{dimension}"
                    / f"dataset_{dataset_id + 1:03d}"
                    / f"ref_{reference_id + 1:03d}"
                    / "reference.npz"
                )
    return datasets, references


def validate_inputs(
    dataset_root: Path, reference_root: Path
) -> dict[str, Any]:
    datasets, references = expected_paths(dataset_root, reference_root)
    missing = [str(path) for path in [*datasets, *references] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} copied input files")
    actual_datasets = sorted(dataset_root.rglob("dataset.npz"))
    actual_references = sorted(reference_root.rglob("reference.npz"))
    if actual_datasets != sorted(datasets):
        raise RuntimeError("dataset_pool contains files outside the canonical 40-file set")
    if actual_references != sorted(references):
        raise RuntimeError(
            "reference_pool contains files outside the canonical 400-file set"
        )

    minimum_margin = float("inf")
    maximum_error = 0.0
    unique_reference_hashes: set[str] = set()
    dataset_rows: dict[tuple[int, int], np.ndarray] = {}
    for dimension in (40, 80, 160, 320):
        expected_m = int(round(0.1 * dimension))
        for dataset_id in range(10):
            path = (
                dataset_root
                / f"N_{dimension}"
                / f"dataset_{dataset_id + 1:03d}"
                / "dataset.npz"
            )
            with np.load(path, allow_pickle=False) as payload:
                if set(payload.files) != {"A", "seed", "alpha"}:
                    raise RuntimeError(f"unexpected dataset fields: {path}")
                a_matrix = np.asarray(payload["A"], dtype=np.float64)
                alpha = float(payload["alpha"])
            if a_matrix.shape != (expected_m, dimension):
                raise RuntimeError(f"invalid dataset shape {a_matrix.shape}: {path}")
            if not np.isfinite(a_matrix).all() or not math.isclose(
                alpha, 0.1, rel_tol=0.0, abs_tol=1.0e-15
            ):
                raise RuntimeError(f"invalid dataset payload: {path}")
            dataset_rows[(dimension, dataset_id)] = a_matrix

            for reference_id in range(10):
                reference_path = (
                    reference_root
                    / f"N_{dimension}"
                    / f"dataset_{dataset_id + 1:03d}"
                    / f"ref_{reference_id + 1:03d}"
                    / "reference.npz"
                )
                with np.load(reference_path, allow_pickle=False) as payload:
                    if payload.files != ["theta"]:
                        raise RuntimeError(
                            f"unexpected reference fields: {reference_path}"
                        )
                    theta = np.asarray(payload["theta"], dtype=np.float64)
                if theta.shape != (dimension,) or not np.isfinite(theta).all():
                    raise RuntimeError(f"invalid reference payload: {reference_path}")
                signed_margins = (a_matrix @ theta) / math.sqrt(dimension)
                error = float(np.mean(signed_margins <= 0.0))
                minimum_margin = min(minimum_margin, float(np.min(signed_margins)))
                maximum_error = max(maximum_error, error)
                unique_reference_hashes.add(
                    hashlib.sha256(theta.tobytes(order="C")).hexdigest()
                )
    if maximum_error != 0.0 or minimum_margin <= 0.0:
        raise RuntimeError("copied reference pool is not strictly hard-feasible")
    if len(unique_reference_hashes) != 400:
        raise RuntimeError("duplicate reference vectors detected")
    return {
        "dataset_count": len(datasets),
        "reference_count": len(references),
        "unique_reference_vector_count": len(unique_reference_hashes),
        "minimum_signed_margin": minimum_margin,
        "maximum_reference_error": maximum_error,
    }


def build_manifest(
    dataset_root: Path,
    reference_root: Path,
    manifest_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    summary = validate_inputs(dataset_root, reference_root)
    datasets, references = expected_paths(dataset_root, reference_root)
    paths = sorted([*datasets, *references])
    rows = [
        f"{sha256_file(path)}  {path.resolve().relative_to(PROJECT_ROOT)}"
        for path in paths
    ]
    manifest_bytes = ("\n".join(rows) + "\n").encode("utf-8")
    atomic_write(manifest_path, manifest_bytes)
    payload = {
        "schema_version": 1,
        "source_kind": "verified_existing_perceptron_inputs",
        "reference_measure": "hard_feasible_truncated_gaussian",
        "manifest_file_count": len(rows),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        **summary,
    }
    atomic_write(
        summary_path,
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8"),
    )
    return payload


def verify_manifest(manifest_path: Path) -> None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = [line.rstrip("\n") for line in handle if line.strip()]
    if len(rows) != 440:
        raise RuntimeError(f"expected 440 manifest rows, found {len(rows)}")
    for row in rows:
        digest, relative = row.split("  ", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"SHA-256 verification failed: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_manifest(args.manifest.resolve())
        payload = validate_inputs(
            args.dataset_root.resolve(), args.reference_root.resolve()
        )
    else:
        payload = build_manifest(
            args.dataset_root.resolve(),
            args.reference_root.resolve(),
            args.manifest.resolve(),
            args.summary.resolve(),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
