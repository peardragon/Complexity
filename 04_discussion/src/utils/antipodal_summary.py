"""Appendix D aggregation from the retained matched-antipodal evaluations."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


class AntipodalDataError(ValueError):
    """Raised when an Appendix D raw input violates the published design."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise AntipodalDataError(message)


def _mean_se_ci(values: Sequence[float], dataset_count: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.shape == (dataset_count,),
        f"dataset-block inference requires {dataset_count} values, found {array.shape}",
    )
    _require(np.all(np.isfinite(array)), "dataset-block values are non-finite")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(dataset_count))
    critical = float(stats.t.ppf(0.975, dataset_count - 1))
    return {
        "mean": mean,
        "se": se,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
    }


def _exact_sign_flip_p(values: Sequence[float], dataset_count: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.shape == (dataset_count,),
        f"sign-flip test requires {dataset_count} values, found {array.shape}",
    )
    observed = abs(float(array.sum()))
    tolerance = 1.0e-14 * max(1.0, observed)
    exceedances = sum(
        abs(float(np.dot(array, np.asarray(signs, dtype=np.float64))))
        >= observed - tolerance
        for signs in itertools.product((-1.0, 1.0), repeat=dataset_count)
    )
    return float(exceedances / (2**dataset_count))


def _load_condition(
    path: Path,
    *,
    condition: str,
    radii: np.ndarray,
    dataset_count: int,
    references_per_dataset: int,
    direction_count: int,
    base_seed: int,
    configured_parameter_count: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as loaded:
        required = {"curvature", "radii", "condition", "direction_count", "base_seed"}
        missing = required.difference(loaded.files)
        _require(not missing, f"{path}: missing arrays {sorted(missing)}")
        curvature = np.asarray(loaded["curvature"], dtype=np.float64)
        observed_radii = np.asarray(loaded["radii"], dtype=np.float64)
        observed_condition = str(np.asarray(loaded["condition"]).item())
        observed_directions = int(np.asarray(loaded["direction_count"]).item())
        observed_seed = int(np.asarray(loaded["base_seed"]).item())
        direction_hash = (
            np.asarray(loaded["direction_hash"])
            if "direction_hash" in loaded.files
            else None
        )
        if {"plus_increment", "minus_increment"}.issubset(loaded.files):
            plus = np.asarray(loaded["plus_increment"], dtype=np.float64)
            minus = np.asarray(loaded["minus_increment"], dtype=np.float64)
            expected_curvature = (plus + minus) / (
                float(configured_parameter_count)
                * observed_radii[None, None, :, None] ** 2
            )
            _require(
                plus.shape == curvature.shape == minus.shape,
                f"{path}: increment/curvature shape mismatch",
            )
            _require(
                np.allclose(curvature, expected_curvature, rtol=2e-13, atol=2e-13),
                f"{path}: curvature does not equal the stored antipodal increments",
            )

    expected_shape = (
        dataset_count,
        references_per_dataset,
        len(radii),
        direction_count,
    )
    _require(
        curvature.shape == expected_shape,
        f"{path}: expected {expected_shape}, found {curvature.shape}",
    )
    _require(np.all(np.isfinite(curvature)), f"{path}: non-finite curvature")
    _require(np.array_equal(observed_radii, radii), f"{path}: radius grid differs from config")
    _require(observed_condition == condition, f"{path}: condition is {observed_condition!r}")
    _require(observed_directions == direction_count, f"{path}: direction count differs from config")
    _require(observed_seed == base_seed, f"{path}: base seed differs from config")
    if direction_hash is not None:
        _require(
            direction_hash.shape == (dataset_count, references_per_dataset),
            f"{path}: direction-hash grid has shape {direction_hash.shape}",
        )
    return curvature, direction_hash


def aggregate_raw_antipodal(
    raw_paths: Mapping[str, Path], config: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Reproduce the four published Study-22 K tables from the two NPZ files."""
    design = config["design"]
    evaluation = config["evaluation"]
    conditions = tuple(str(value) for value in design["conditions"])
    _require(len(conditions) == 2, "Appendix D requires clean and randomized endpoints")
    dataset_count = int(design["datasets_per_condition"])
    reference_count = int(design["references_per_dataset"])
    direction_count = int(design["matched_antipodal_directions_per_reference"])
    _require(dataset_count >= 2, "at least two dataset blocks are required")
    _require(
        reference_count > 0 and direction_count > 0,
        "reference/direction counts must be positive",
    )
    _require(
        direction_count == int(evaluation["directions"]),
        "design/evaluation direction counts differ",
    )
    radii = np.asarray(evaluation["radii"], dtype=np.float64)
    _require(radii.ndim == 1 and radii.size >= 2, "at least two radii are required")
    _require(
        np.all(np.isfinite(radii)) and np.all(radii > 0.0),
        "radii must be finite and positive",
    )
    _require(np.all(np.diff(radii) > 0.0), "radii must be unique and strictly increasing")

    parts: list[np.ndarray] = []
    direction_hashes: list[np.ndarray | None] = []
    for condition in conditions:
        path = raw_paths[condition]
        curvature, direction_hash = _load_condition(
            path,
            condition=condition,
            radii=radii,
            dataset_count=dataset_count,
            references_per_dataset=reference_count,
            direction_count=direction_count,
            base_seed=int(evaluation["base_seed"]),
            configured_parameter_count=int(evaluation["parameter_count"]),
        )
        parts.append(curvature)
        direction_hashes.append(direction_hash)
    if all(value is not None for value in direction_hashes):
        _require(
            np.array_equal(direction_hashes[0], direction_hashes[1]),
            "clean/random direction banks are not exactly matched",
        )
    curvature = np.stack(parts, axis=0)
    baseline = curvature[:, :, :, 0, :]

    summary_rows: list[dict[str, Any]] = []
    for endpoint, condition in enumerate(conditions):
        for radius_index in range(1, radii.size):
            drift = curvature[endpoint, :, :, radius_index, :] - baseline[endpoint]
            dataset_mean = drift.mean(axis=(1, 2))
            dataset_fraction = (drift > 0.0).mean(axis=(1, 2))
            drift_inference = _mean_se_ci(dataset_mean, dataset_count)
            fraction_inference = _mean_se_ci(dataset_fraction, dataset_count)
            pooled = drift.reshape(-1)
            summary_rows.append(
                {
                    "condition": condition,
                    "near_radius": float(radii[0]),
                    "far_radius": float(radii[radius_index]),
                    "mean_delta_K": drift_inference["mean"],
                    "se_delta_K": drift_inference["se"],
                    "ci95_delta_K_low": drift_inference["ci95_low"],
                    "ci95_delta_K_high": drift_inference["ci95_high"],
                    "positive_dataset_count": int(np.sum(dataset_mean > 0.0)),
                    "exact_two_sided_sign_flip_p": _exact_sign_flip_p(dataset_mean, dataset_count),
                    "mean_hardening_direction_fraction": fraction_inference["mean"],
                    "ci95_hardening_fraction_low": fraction_inference["ci95_low"],
                    "ci95_hardening_fraction_high": fraction_inference["ci95_high"],
                    "pooled_delta_K_q05_descriptive": float(np.quantile(pooled, 0.05)),
                    "pooled_delta_K_median_descriptive": float(np.median(pooled)),
                    "pooled_delta_K_q95_descriptive": float(np.quantile(pooled, 0.95)),
                }
            )

    contrast_rows: list[dict[str, Any]] = []
    for radius_index in range(1, radii.size):
        clean = (curvature[0, :, :, radius_index, :] - baseline[0]).mean(axis=(1, 2))
        randomized = (curvature[1, :, :, radius_index, :] - baseline[1]).mean(axis=(1, 2))
        difference = randomized - clean
        inference = _mean_se_ci(difference, dataset_count)
        contrast_rows.append(
            {
                "contrast": f"{conditions[1]}_minus_{conditions[0]}",
                "near_radius": float(radii[0]),
                "far_radius": float(radii[radius_index]),
                "mean_extra_delta_K": inference["mean"],
                "se_extra_delta_K": inference["se"],
                "ci95_low": inference["ci95_low"],
                "ci95_high": inference["ci95_high"],
                "positive_dataset_count": int(np.sum(difference > 0.0)),
                "exact_two_sided_sign_flip_p": _exact_sign_flip_p(difference, dataset_count),
            }
        )

    profile_rows: list[dict[str, Any]] = []
    peak_radius_by_dataset = np.empty((2, dataset_count), dtype=np.float64)
    dataset_profiles: list[np.ndarray] = []
    for endpoint, condition in enumerate(conditions):
        dataset_profile = curvature[endpoint].mean(axis=(1, 3))
        dataset_profiles.append(dataset_profile)
        peak_radius_by_dataset[endpoint] = radii[np.argmax(dataset_profile, axis=1)]
        for radius_index, radius in enumerate(radii):
            inference = _mean_se_ci(dataset_profile[:, radius_index], dataset_count)
            profile_rows.append(
                {
                    "condition": condition,
                    "radius": float(radius),
                    "mean_K": inference["mean"],
                    "se_K": inference["se"],
                    "ci95_K_low": inference["ci95_low"],
                    "ci95_K_high": inference["ci95_high"],
                }
            )

    peak_radius_difference = peak_radius_by_dataset[1] - peak_radius_by_dataset[0]
    peak_clean = np.max(dataset_profiles[0], axis=1)
    peak_random = np.max(dataset_profiles[1], axis=1)
    peak_height_difference = peak_random - peak_clean
    baseline_clean = curvature[0, :, :, 0, :].mean(axis=(1, 2))
    baseline_random = curvature[1, :, :, 0, :].mean(axis=(1, 2))
    hardening_difference = (peak_random - baseline_random) - (peak_clean - baseline_clean)
    peak_rows: list[dict[str, Any]] = []
    for metric, values in (
        ("peak_radius", peak_radius_difference),
        ("peak_absolute_K", peak_height_difference),
        ("peak_hardening_delta_K", hardening_difference),
    ):
        inference = _mean_se_ci(values, dataset_count)
        peak_rows.append(
            {
                "contrast": f"{conditions[1]}_minus_{conditions[0]}",
                "metric": metric,
                "mean_difference": inference["mean"],
                "se_difference": inference["se"],
                "ci95_low": inference["ci95_low"],
                "ci95_high": inference["ci95_high"],
                "positive_dataset_count": int(np.sum(values > 0.0)),
                "negative_dataset_count": int(np.sum(values < 0.0)),
                "exact_two_sided_sign_flip_p": _exact_sign_flip_p(values, dataset_count),
            }
        )
    return {
        "profiles": profile_rows,
        "summary": summary_rows,
        "contrast": contrast_rows,
        "peak": peak_rows,
    }
