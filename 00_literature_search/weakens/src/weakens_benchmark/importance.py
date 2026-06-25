from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import i0e, logsumexp, ndtr

from .landscape import ProxyLandscape


@dataclass
class ImportanceResult:
    samples: np.ndarray
    energies: np.ndarray
    log_weights: np.ndarray
    ess_fraction: float
    metadata: dict[str, Any]


def _log_von_mises_pdf(theta: np.ndarray, mean: float, kappa: float) -> np.ndarray:
    return kappa * np.cos(theta - mean) - (np.log(2.0 * np.pi) + np.log(i0e(kappa)) + abs(kappa))


def _sample_truncnorm(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    out = np.empty(size, dtype=np.float64)
    filled = 0
    while filled < size:
        draw = rng.normal(loc=mean, scale=sd, size=max(128, 2 * (size - filled)))
        draw = draw[(draw >= low) & (draw <= high)]
        take = min(draw.size, size - filled)
        if take:
            out[filled : filled + take] = draw[:take]
            filled += take
    return out


def _log_truncnorm_pdf(r: np.ndarray, mean: float, sd: float, low: float, high: float) -> np.ndarray:
    z = (r - mean) / sd
    norm_const = ndtr((high - mean) / sd) - ndtr((low - mean) / sd)
    log_pdf = -0.5 * z * z - np.log(sd) - 0.5 * np.log(2.0 * np.pi) - np.log(norm_const)
    return np.where((r >= low) & (r <= high), log_pdf, -np.inf)


def _proposal_components(landscape: ProxyLandscape, config: dict[str, Any]) -> list[dict[str, Any]]:
    vmf_cfg = config["vmf_l2"]
    broad_weight = float(vmf_cfg.get("broad_weight", 0.1))
    broad_weight = min(max(broad_weight, 0.0), 0.95)
    anchor_weight = (1.0 - broad_weight) / len(landscape.basins)
    components: list[dict[str, Any]] = []
    for row in landscape.region_reference_frame():
        components.append(
            {
                "kind": "vmf_l2_anchor",
                "region": row["region"],
                "weight": anchor_weight,
                "angle": float(row["angle"]),
                "radius": max(0.08, float(row["radius"])),
                "kappa": float(vmf_cfg["kappa"]),
                "radius_sd": float(vmf_cfg["radius_sd"]),
            }
        )
    components.append(
        {
            "kind": "broad_uniform",
            "region": "broad",
            "weight": broad_weight,
        }
    )
    return components


def _log_proposal_area_density(
    z: np.ndarray,
    components: list[dict[str, Any]],
    max_radius: float,
) -> np.ndarray:
    r = np.linalg.norm(z, axis=1)
    theta = np.arctan2(z[:, 1], z[:, 0])
    log_terms = []
    for comp in components:
        log_weight = np.log(max(float(comp["weight"]), 1.0e-300))
        if comp["kind"] == "broad_uniform":
            log_r = np.where((r > 0.0) & (r <= max_radius), -np.log(max_radius), -np.inf)
            log_theta = -np.log(2.0 * np.pi)
        else:
            log_r = _log_truncnorm_pdf(
                r,
                mean=float(comp["radius"]),
                sd=float(comp["radius_sd"]),
                low=0.0,
                high=max_radius,
            )
            log_theta = _log_von_mises_pdf(theta, float(comp["angle"]), float(comp["kappa"]))
        log_terms.append(log_weight + log_r + log_theta)
    log_polar = logsumexp(np.column_stack(log_terms), axis=1)
    return log_polar - np.log(np.maximum(r, 1.0e-300))


def run_vmf_l2_importance(
    landscape: ProxyLandscape,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> ImportanceResult:
    start_time = time.perf_counter()
    vmf_cfg = config["vmf_l2"]
    n_samples = int(vmf_cfg["n_samples"])
    max_radius = float(vmf_cfg["max_radius"])
    components = _proposal_components(landscape, config)
    weights = np.asarray([float(c["weight"]) for c in components], dtype=np.float64)
    weights = weights / np.sum(weights)
    choices = rng.choice(len(components), size=n_samples, p=weights)
    radii = np.empty(n_samples, dtype=np.float64)
    theta = np.empty(n_samples, dtype=np.float64)
    for comp_id, comp in enumerate(components):
        mask = choices == comp_id
        count = int(np.sum(mask))
        if count == 0:
            continue
        if comp["kind"] == "broad_uniform":
            radii[mask] = rng.uniform(0.0, max_radius, size=count)
            theta[mask] = rng.uniform(-np.pi, np.pi, size=count)
        else:
            radii[mask] = _sample_truncnorm(
                rng,
                mean=float(comp["radius"]),
                sd=float(comp["radius_sd"]),
                low=0.0,
                high=max_radius,
                size=count,
            )
            theta[mask] = rng.vonmises(float(comp["angle"]), float(comp["kappa"]), size=count)
    samples = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    energies = landscape.energy(samples)
    log_q = _log_proposal_area_density(samples, components, max_radius=max_radius)
    log_weights = -landscape.beta * energies - log_q
    log_weights = np.where(np.isfinite(log_weights), log_weights, -np.inf)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        ess_fraction = 0.0
    else:
        lw = log_weights[finite]
        ess = float(np.exp(2.0 * logsumexp(lw) - logsumexp(2.0 * lw)))
        ess_fraction = ess / max(1, lw.size)
    metadata = {
        "method_role": "final_method_proxy_instantiation",
        "source_paper": "project final vMF+L2 methodology",
        "proxy_mapping": "von Mises direction proposal plus truncated-normal L2 shell proposal",
        "n_samples": n_samples,
        "attempted_sample_count": n_samples,
        "max_radius": max_radius,
        "components": components,
        "ess_fraction": ess_fraction,
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    return ImportanceResult(
        samples=samples,
        energies=energies,
        log_weights=log_weights,
        ess_fraction=ess_fraction,
        metadata=metadata,
    )


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_weights)
    out = np.zeros_like(log_weights, dtype=np.float64)
    if np.any(finite):
        out[finite] = np.exp(log_weights[finite] - logsumexp(log_weights[finite]))
    return out
