"""Exact-shell algebra and direct first-radial-derivative utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .objective import ObjectiveContract


def shell_l2_components(
    theta_ref: np.ndarray,
    radius: float,
    *,
    contract: ObjectiveContract,
) -> dict[str, float | int]:
    theta = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    parameter_count = int(theta.size)
    if parameter_count <= 0:
        raise ValueError("theta_ref cannot be empty")
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    norm = float(np.linalg.norm(theta))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("theta_ref norm must be positive and finite")
    kappa = (
        contract.lambda_reg
        * float(radius)
        * norm
        / math.sqrt(parameter_count)
    )
    return {
        "parameter_count": parameter_count,
        "theta_ref_norm": norm,
        "kappa": float(kappa),
        "reference_l2_log_weight": float(
            -contract.lambda_reg * norm * norm / (2.0 * parameter_count)
        ),
        "radius_l2_log_weight": float(
            -contract.lambda_reg * radius * radius / 2.0
        ),
    }


def expanded_l2_penalty(
    theta_ref: np.ndarray,
    directions: np.ndarray,
    radius: float,
    *,
    contract: ObjectiveContract,
) -> np.ndarray:
    ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    unit = np.asarray(directions, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != ref.size:
        raise ValueError("directions must have shape (particles, P)")
    if (
        ref.size == 0
        or not np.all(np.isfinite(ref))
        or not np.all(np.isfinite(unit))
    ):
        raise ValueError("theta_ref and directions must be finite and non-empty")
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    if not np.allclose(
        np.linalg.norm(unit, axis=1), 1.0, rtol=0.0, atol=2.0e-6
    ):
        raise ValueError("exact-shell directions must have unit norm")
    theta = ref[None, :] + math.sqrt(ref.size) * float(radius) * unit
    return (
        contract.lambda_reg
        * np.sum(theta * theta, axis=1)
        / (2.0 * ref.size)
    )


def direct_radial_score_numpy(
    theta_ref: np.ndarray,
    directions: np.ndarray,
    ce_gradients: np.ndarray,
    radius: float,
    *,
    contract: ObjectiveContract,
) -> dict[str, np.ndarray]:
    """Evaluate the direct score, never a finite difference of a radius curve."""

    ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    unit = np.asarray(directions, dtype=np.float64)
    gradients = np.asarray(ce_gradients, dtype=np.float64)
    if unit.ndim != 2 or gradients.shape != unit.shape:
        raise ValueError("directions and CE gradients must share shape (n, P)")
    if unit.shape[1] != ref.size:
        raise ValueError("theta_ref dimension differs from particle dimension")
    if (
        ref.size == 0
        or not np.all(np.isfinite(ref))
        or not np.all(np.isfinite(unit))
        or not np.all(np.isfinite(gradients))
    ):
        raise ValueError(
            "theta_ref, directions, and CE gradients must be finite and non-empty"
        )
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    norms = np.linalg.norm(unit, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError("direct radial score requires unit directions")
    root_p = math.sqrt(ref.size)
    ce = -contract.gamma_ce * root_p * np.sum(gradients * unit, axis=1)
    ref_dot = unit @ ref / root_p
    l2 = -contract.lambda_reg * (float(radius) + ref_dot)
    return {
        "ce_radial_score": ce,
        "l2_radial_score": l2,
        "total_radial_score": ce + l2,
        "theta_ref_dot_u_over_sqrt_p": ref_dot,
    }


def direct_radial_score_torch(
    theta_ref: Any,
    directions: Any,
    ce_gradients: Any,
    radius: float,
    *,
    contract: ObjectiveContract,
) -> dict[str, Any]:
    """Torch equivalent of :func:`direct_radial_score_numpy`.

    ``ce_gradients`` must be the gradient of CE *mean* with respect to each
    shell particle.  No adjacent-radius values enter this computation.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("torch is required for direct_radial_score_torch") from exc
    ref = theta_ref.reshape(-1)
    unit = directions
    gradients = ce_gradients
    if unit.ndim != 2 or gradients.shape != unit.shape:
        raise ValueError("directions and CE gradients must share shape (n, P)")
    if unit.shape[1] != ref.numel() or ref.numel() == 0:
        raise ValueError("theta_ref dimension differs from particle dimension")
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    if not bool(torch.isfinite(ref).all().item()):
        raise ValueError("theta_ref contains non-finite values")
    if not bool(torch.isfinite(unit).all().item()) or not bool(
        torch.isfinite(gradients).all().item()
    ):
        raise ValueError("directions or CE gradients contain non-finite values")
    norms = torch.linalg.vector_norm(unit, dim=1)
    if not bool(
        torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=0.0,
            atol=2.0e-6,
        )
    ):
        raise ValueError("direct radial score requires unit directions")
    root_p = math.sqrt(ref.numel())
    ce = (
        -float(contract.gamma_ce)
        * root_p
        * torch.sum(gradients * unit, dim=1)
    )
    ref_dot = unit @ ref / root_p
    l2 = -float(contract.lambda_reg) * (float(radius) + ref_dot)
    return {
        "ce_radial_score": ce,
        "l2_radial_score": l2,
        "total_radial_score": ce + l2,
        "theta_ref_dot_u_over_sqrt_p": ref_dot,
    }


def normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("log weights must be finite and non-empty")
    maximum = float(np.max(values))
    shifted = np.exp(values - maximum)
    return shifted / float(np.sum(shifted))


def weighted_direct_derivative(
    scores: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    log_weights: np.ndarray | None = None,
) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite and non-empty")
    if (weights is None) == (log_weights is None):
        raise ValueError("provide exactly one of weights or log_weights")
    if log_weights is not None:
        probability = normalise_log_weights(log_weights)
        if probability.shape != values.shape:
            raise ValueError("score/log-weight shape mismatch")
    else:
        probability = np.asarray(weights, dtype=np.float64).reshape(-1)
        if probability.shape != values.shape:
            raise ValueError("score/weight shape mismatch")
        if np.any(probability < 0.0) or not np.all(np.isfinite(probability)):
            raise ValueError("weights must be finite and non-negative")
        total = float(np.sum(probability))
        if total <= 0.0:
            raise ValueError("weights must have positive mass")
        probability = probability / total
    return float(np.sum(probability * values))


def centred_profiles(
    logz_angular: np.ndarray,
    radii: np.ndarray,
    *,
    parameter_count: int,
    baseline_index: int,
) -> dict[str, np.ndarray]:
    logz = np.asarray(logz_angular, dtype=np.float64).reshape(-1)
    radius = np.asarray(radii, dtype=np.float64).reshape(-1)
    if logz.shape != radius.shape:
        raise ValueError("logZ and radius arrays must have equal shapes")
    if (
        logz.size == 0
        or not np.all(np.isfinite(logz))
        or not np.all(np.isfinite(radius))
        or np.any(radius <= 0.0)
    ):
        raise ValueError("logZ must be finite and radii positive and finite")
    if not 0 <= int(baseline_index) < radius.size:
        raise IndexError("baseline index lies outside the radius grid")
    if int(parameter_count) <= 0:
        raise ValueError("parameter_count must be positive")
    p = float(parameter_count)
    energy = (logz - logz[int(baseline_index)]) / p
    # delta(||theta-theta_ref||^2 / P - r^2) contributes r^(P-2).
    entropy = ((p - 2.0) / p) * np.log(
        radius / radius[int(baseline_index)]
    )
    return {
        "phi_energy_centered": energy,
        "phi_shell_entropy_centered": entropy,
        "phi_shell_centered": energy + entropy,
    }


def profile_derivatives(
    dlogz_dr: np.ndarray,
    radii: np.ndarray,
    *,
    parameter_count: int,
) -> dict[str, np.ndarray]:
    direct = np.asarray(dlogz_dr, dtype=np.float64).reshape(-1)
    radius = np.asarray(radii, dtype=np.float64).reshape(-1)
    if direct.shape != radius.shape:
        raise ValueError("dlogZ/dr and radius arrays must have equal shapes")
    if (
        direct.size == 0
        or not np.all(np.isfinite(direct))
        or not np.all(np.isfinite(radius))
        or np.any(radius <= 0.0)
    ):
        raise ValueError(
            "dlogZ/dr must be finite and radii positive and finite"
        )
    if int(parameter_count) <= 0:
        raise ValueError("parameter_count must be positive")
    p = float(parameter_count)
    energy = direct / p
    entropy = (p - 2.0) / (p * radius)
    return {
        "dphi_energy_dr_direct": energy,
        "dphi_shell_entropy_dr_analytic": entropy,
        "dphi_shell_dr_direct": energy + entropy,
    }


def combine_split_log_normalizers(
    split_logz: np.ndarray,
    split_particle_counts: np.ndarray,
) -> float:
    logz = np.asarray(split_logz, dtype=np.float64).reshape(-1)
    counts = np.asarray(split_particle_counts, dtype=np.float64).reshape(-1)
    if logz.shape != counts.shape or logz.size == 0:
        raise ValueError("split logZ and count arrays must share a non-empty shape")
    if np.any(counts <= 0.0) or not np.all(np.isfinite(logz)):
        raise ValueError("split counts must be positive and logZ finite")
    values = np.log(split_mixture_probabilities(logz, counts)) + logz
    maximum = float(np.max(values))
    return float(maximum + math.log(float(np.sum(np.exp(values - maximum)))))


def split_mixture_probabilities(
    split_logz: np.ndarray,
    split_particle_counts: np.ndarray,
) -> np.ndarray:
    """Linear-space split fractions before normalizer values are applied."""

    logz = np.asarray(split_logz, dtype=np.float64).reshape(-1)
    counts = np.asarray(split_particle_counts, dtype=np.float64).reshape(-1)
    if logz.shape != counts.shape or logz.size == 0:
        raise ValueError("split logZ and count arrays must share a non-empty shape")
    if (
        np.any(counts <= 0.0)
        or not np.all(np.isfinite(counts))
        or not np.all(np.isfinite(logz))
    ):
        raise ValueError("split counts must be positive and all inputs finite")
    return counts / float(np.sum(counts))


def split_target_mixture_probabilities(
    split_logz: np.ndarray,
    split_particle_counts: np.ndarray,
) -> np.ndarray:
    """Weights for derivatives/observables of the combined split normalizer.

    If ``Z_hat=sum_s a_s Z_s``, then
    ``d log Z_hat = sum_s softmax(log(a_s)+log Z_s) d log Z_s``.
    """

    logz = np.asarray(split_logz, dtype=np.float64).reshape(-1)
    fractions = split_mixture_probabilities(logz, split_particle_counts)
    values = np.log(fractions) + logz
    maximum = float(np.max(values))
    unnormalised = np.exp(values - maximum)
    return unnormalised / float(np.sum(unnormalised))


def combine_split_direct_derivatives(
    split_derivatives: np.ndarray,
    split_logz: np.ndarray,
    split_particle_counts: np.ndarray,
) -> float:
    """Direct derivative consistent with the linear-space combined ``Z``."""

    derivatives = np.asarray(split_derivatives, dtype=np.float64).reshape(-1)
    mixture = split_target_mixture_probabilities(
        split_logz, split_particle_counts
    )
    if derivatives.shape != mixture.shape or not np.all(
        np.isfinite(derivatives)
    ):
        raise ValueError("split derivatives must be finite and match split logZ")
    return float(np.sum(mixture * derivatives))
