"""Production shell sampler for the frozen 01 perceptron construction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln, ive, logsumexp

from perceptron import (
    assert_unit_directions,
    ce_sum,
    classification_error,
    energy,
    l2_matched_parameters,
    margins,
    radial_log_target_score,
    shell_theta,
)


METHOD = "l2_matched_overlap_controlled_pool_smc_v2"


@dataclass(frozen=True)
class SMCParams:
    target_step_overlap: float
    resample_pool_overlap: float
    max_temperature_events: int
    minimum_temperature_increment: float
    bisection_iterations: int
    mh_sweeps_per_event: int
    move_kappa_factor: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SMCParams":
        smc = config["smc"]
        params = cls(
            target_step_overlap=float(smc["target_step_overlap"]),
            resample_pool_overlap=float(smc["resample_pool_overlap"]),
            max_temperature_events=int(smc["max_temperature_events"]),
            minimum_temperature_increment=float(
                smc["minimum_temperature_increment"]
            ),
            bisection_iterations=int(smc["bisection_iterations"]),
            mh_sweeps_per_event=int(smc["mh_sweeps_per_event"]),
            move_kappa_factor=float(smc["move_kappa_factor"]),
        )
        if not 0.0 < params.target_step_overlap < 1.0:
            raise ValueError("target_step_overlap must lie in (0,1)")
        if not 0.0 < params.resample_pool_overlap <= 1.0:
            raise ValueError("resample_pool_overlap must lie in (0,1]")
        if params.max_temperature_events < 1:
            raise ValueError("max_temperature_events must be positive")
        if params.minimum_temperature_increment <= 0.0:
            raise ValueError("minimum_temperature_increment must be positive")
        if params.bisection_iterations < 1:
            raise ValueError("bisection_iterations must be positive")
        if params.mh_sweeps_per_event < 1:
            raise ValueError("mutation must run after every event")
        if params.move_kappa_factor <= 0.0:
            raise ValueError("move_kappa_factor must be positive")
        return params


def log_sphere_mgf(dimension: int, kappa: float) -> float:
    """``log E_Haar exp(kappa mu.u)`` with a scaled-Bessel evaluation."""
    kappa = abs(float(kappa))
    if kappa == 0.0:
        return 0.0
    nu = 0.5 * float(dimension) - 1.0
    scaled_bessel = float(ive(nu, kappa))
    if not np.isfinite(scaled_bessel) or scaled_bessel <= 0.0:
        raise FloatingPointError(
            f"scaled Bessel evaluation failed for N={dimension}, kappa={kappa}"
        )
    return float(
        gammaln(0.5 * float(dimension))
        + nu * math.log(2.0 / kappa)
        + math.log(scaled_bessel)
        + kappa
    )


def sample_vmf(
    mu: np.ndarray, kappa: float, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Wood's exact vMF sampler with deterministic NumPy RNG."""
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(mu))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("vMF mean direction must be finite and nonzero")
    mu = mu / norm
    dimension = int(mu.size)
    count = int(count)
    if dimension < 2 or count < 1:
        raise ValueError("vMF requires dimension >=2 and count >=1")
    if float(kappa) <= 1.0e-12:
        result = rng.normal(size=(count, dimension))
        result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result

    # Rationalized form avoids catastrophic cancellation at
    # kappa_move=80*N.
    b = float(dimension - 1) / (
        math.sqrt(4.0 * float(kappa) ** 2 + float(dimension - 1) ** 2)
        + 2.0 * float(kappa)
    )
    x0 = (1.0 - b) / (1.0 + b)
    constant = float(kappa) * x0 + float(dimension - 1) * math.log(
        max(1.0 - x0 * x0, 1.0e-300)
    )
    axial = np.empty(count, dtype=np.float64)
    alpha = 0.5 * float(dimension - 1)
    filled = 0
    while filled < count:
        draw_count = max(1024, int(math.ceil(1.25 * (count - filled))))
        beta_draw = rng.beta(alpha, alpha, size=draw_count)
        candidate = (1.0 - (1.0 + b) * beta_draw) / (
            1.0 - (1.0 - b) * beta_draw
        )
        log_acceptance = (
            float(kappa) * candidate
            + float(dimension - 1)
            * np.log(np.maximum(1.0 - x0 * candidate, 1.0e-300))
            - constant
        )
        accepted = candidate[np.log(rng.random(draw_count)) <= log_acceptance]
        take = min(accepted.size, count - filled)
        if take:
            axial[filled : filled + take] = accepted[:take]
            filled += take

    tangent = rng.normal(size=(count, dimension))
    tangent -= (tangent @ mu)[:, None] * mu[None, :]
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    bad = tangent_norm[:, 0] <= 0.0
    while np.any(bad):
        tangent[bad] = rng.normal(size=(int(np.sum(bad)), dimension))
        tangent[bad] -= (tangent[bad] @ mu)[:, None] * mu[None, :]
        tangent_norm[bad] = np.linalg.norm(tangent[bad], axis=1, keepdims=True)
        bad = tangent_norm[:, 0] <= 0.0
    tangent /= tangent_norm
    result = (
        axial[:, None] * mu[None, :]
        + np.sqrt(np.maximum(1.0 - axial * axial, 0.0))[:, None] * tangent
    )
    assert_unit_directions(result)
    return result


def sample_vmf_batch(
    means: np.ndarray, kappa: float, rng: np.random.Generator
) -> np.ndarray:
    """Row-wise vMF proposals sharing one concentration."""
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2:
        raise ValueError("means must be a matrix")
    count, dimension = means.shape
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("proposal means must be finite and nonzero")
    unit_means = means / norms
    if float(kappa) <= 1.0e-12:
        result = rng.normal(size=(count, dimension))
        result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result

    b = float(dimension - 1) / (
        math.sqrt(4.0 * float(kappa) ** 2 + float(dimension - 1) ** 2)
        + 2.0 * float(kappa)
    )
    x0 = (1.0 - b) / (1.0 + b)
    constant = float(kappa) * x0 + float(dimension - 1) * math.log(
        max(1.0 - x0 * x0, 1.0e-300)
    )
    axial = np.empty(count, dtype=np.float64)
    alpha = 0.5 * float(dimension - 1)
    filled = 0
    while filled < count:
        draw_count = max(1024, int(math.ceil(1.25 * (count - filled))))
        beta_draw = rng.beta(alpha, alpha, size=draw_count)
        candidate = (1.0 - (1.0 + b) * beta_draw) / (
            1.0 - (1.0 - b) * beta_draw
        )
        log_acceptance = (
            float(kappa) * candidate
            + float(dimension - 1)
            * np.log(np.maximum(1.0 - x0 * candidate, 1.0e-300))
            - constant
        )
        accepted = candidate[np.log(rng.random(draw_count)) <= log_acceptance]
        take = min(accepted.size, count - filled)
        if take:
            axial[filled : filled + take] = accepted[:take]
            filled += take

    tangent = rng.normal(size=(count, dimension))
    tangent -= np.sum(tangent * unit_means, axis=1, keepdims=True) * unit_means
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    bad = tangent_norm[:, 0] <= 0.0
    while np.any(bad):
        tangent[bad] = rng.normal(size=(int(np.sum(bad)), dimension))
        tangent[bad] -= (
            np.sum(tangent[bad] * unit_means[bad], axis=1, keepdims=True)
            * unit_means[bad]
        )
        tangent_norm[bad] = np.linalg.norm(tangent[bad], axis=1, keepdims=True)
        bad = tangent_norm[:, 0] <= 0.0
    tangent /= tangent_norm
    result = (
        axial[:, None] * unit_means
        + np.sqrt(np.maximum(1.0 - axial * axial, 0.0))[:, None] * tangent
    )
    assert_unit_directions(result)
    return result


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    return np.asarray(log_weights, dtype=np.float64) - logsumexp(log_weights)


def step_overlap(
    log_weights: np.ndarray, ce_values: np.ndarray, delta_t: float
) -> float:
    """Inverse-chi-square overlap between current and tentative probabilities."""
    log_weights = normalize_log_weights(log_weights)
    log_factor = -float(delta_t) * np.asarray(ce_values, dtype=np.float64)
    return float(
        np.exp(
            2.0 * logsumexp(log_weights + log_factor)
            - logsumexp(log_weights + 2.0 * log_factor)
        )
    )


def pool_overlap(log_weights: np.ndarray) -> float:
    """Inverse-chi-square overlap with the uniform particle distribution."""
    log_weights = normalize_log_weights(log_weights)
    count = int(log_weights.size)
    return float(np.exp(-logsumexp(2.0 * log_weights)) / count)


def choose_temperature(
    temperature: float,
    ce_values: np.ndarray,
    log_weights: np.ndarray,
    params: SMCParams,
) -> tuple[float, float, bool]:
    direct_overlap = step_overlap(
        log_weights, ce_values, 1.0 - float(temperature)
    )
    if direct_overlap >= params.target_step_overlap:
        return 1.0, direct_overlap, True
    minimum_candidate = min(
        1.0,
        float(temperature) + params.minimum_temperature_increment,
    )
    minimum_overlap = step_overlap(
        log_weights,
        ce_values,
        minimum_candidate - float(temperature),
    )
    if minimum_overlap < params.target_step_overlap:
        raise RuntimeError(
            "minimum temperature increment violates target step overlap: "
            f"t={float(temperature):.17g}, "
            f"minimum_delta={minimum_candidate - float(temperature):.17g}, "
            f"overlap={minimum_overlap:.17g}, "
            f"target={params.target_step_overlap:.17g}"
        )
    lower, upper = float(temperature), 1.0
    for _ in range(params.bisection_iterations):
        midpoint = 0.5 * (lower + upper)
        overlap = step_overlap(
            log_weights, ce_values, midpoint - float(temperature)
        )
        if overlap >= params.target_step_overlap:
            lower = midpoint
        else:
            upper = midpoint
    selected = max(lower, minimum_candidate)
    selected_overlap = step_overlap(
        log_weights,
        ce_values,
        selected - float(temperature),
    )
    if selected_overlap < params.target_step_overlap:
        raise RuntimeError(
            "temperature search returned a step below the overlap target: "
            f"overlap={selected_overlap:.17g}, "
            f"target={params.target_step_overlap:.17g}"
        )
    return float(selected), float(selected_overlap), False


def systematic_resample(
    log_weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    probabilities = np.exp(normalize_log_weights(log_weights))
    count = int(probabilities.size)
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    positions = (rng.random() + np.arange(count, dtype=np.float64)) / count
    return np.searchsorted(cdf, positions, side="left").astype(np.int64)


def weighted_mean(values: np.ndarray, log_weights: np.ndarray) -> float:
    probability = np.exp(normalize_log_weights(log_weights))
    return float(np.sum(probability * np.asarray(values, dtype=np.float64)))


def split_target_mixture_weights(split_logz: np.ndarray) -> np.ndarray:
    """Normalizer-proportional weights for equal-budget independent splits."""

    values = np.asarray(split_logz, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("split log normalizers must be finite and non-empty")
    return np.exp(normalize_log_weights(values))


def combine_split_values(
    split_values: np.ndarray,
    split_logz: np.ndarray,
) -> float:
    """Combine target expectations consistently with mean split normalizers."""

    values = np.asarray(split_values, dtype=np.float64).reshape(-1)
    mixture = split_target_mixture_weights(split_logz)
    if values.shape != mixture.shape or not np.all(np.isfinite(values)):
        raise ValueError("split values must be finite and match split logZ")
    return float(np.sum(mixture * values))


def mutate(
    *,
    directions: np.ndarray,
    ce_values: np.ndarray,
    errors: np.ndarray,
    theta_ref: np.ndarray,
    a_matrix: np.ndarray,
    radius: float,
    mu: np.ndarray,
    base_kappa: float,
    temperature: float,
    params: SMCParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    dimension = int(theta_ref.size)
    move_kappa = params.move_kappa_factor * dimension
    current_projection = directions @ mu
    acceptance_rates: list[float] = []
    for _ in range(params.mh_sweeps_per_event):
        proposal = sample_vmf_batch(directions, move_kappa, rng)
        proposal_theta = shell_theta(theta_ref, radius, proposal)
        proposal_ce = ce_sum(proposal_theta, a_matrix)
        proposal_error = classification_error(proposal_theta, a_matrix)
        proposal_projection = proposal @ mu
        log_ratio = (
            base_kappa * (proposal_projection - current_projection)
            - float(temperature) * (proposal_ce - ce_values)
        )
        accept = np.log(rng.random(ce_values.size)) <= np.minimum(0.0, log_ratio)
        if np.any(accept):
            directions[accept] = proposal[accept]
            ce_values[accept] = proposal_ce[accept]
            errors[accept] = proposal_error[accept]
            current_projection[accept] = proposal_projection[accept]
        acceptance_rates.append(float(np.mean(accept)))
    assert_unit_directions(directions)
    return directions, ce_values, errors, acceptance_rates


def run_split(
    *,
    theta_ref: np.ndarray,
    a_matrix: np.ndarray,
    radius: float,
    particles: int,
    seed: int,
    params: SMCParams,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    mu, base_kappa = l2_matched_parameters(theta_ref, radius)
    directions = sample_vmf(mu, base_kappa, int(particles), rng)
    theta = shell_theta(theta_ref, radius, directions)
    ce_values = ce_sum(theta, a_matrix)
    errors = classification_error(theta, a_matrix)
    log_weights = np.full(
        int(particles), -math.log(int(particles)), dtype=np.float64
    )
    temperature = 0.0
    logz_ce = 0.0
    history: list[dict[str, Any]] = []
    resample_count = 0

    for event in range(params.max_temperature_events):
        if temperature >= 1.0 - 1.0e-12:
            break
        next_temperature, overlap, was_direct = choose_temperature(
            temperature, ce_values, log_weights, params
        )
        delta_t = next_temperature - temperature
        log_factor = -delta_t * ce_values
        increment = float(logsumexp(log_weights + log_factor))
        logz_ce += increment
        log_weights = normalize_log_weights(log_weights + log_factor)
        overlap_uniform = pool_overlap(log_weights)
        resampled = overlap_uniform < params.resample_pool_overlap
        if resampled:
            ancestors = systematic_resample(log_weights, rng)
            directions = directions[ancestors].copy()
            ce_values = ce_values[ancestors].copy()
            errors = errors[ancestors].copy()
            log_weights = np.full(
                int(particles), -math.log(int(particles)), dtype=np.float64
            )
            resample_count += 1
        directions, ce_values, errors, acceptance = mutate(
            directions=directions,
            ce_values=ce_values,
            errors=errors,
            theta_ref=theta_ref,
            a_matrix=a_matrix,
            radius=radius,
            mu=mu,
            base_kappa=base_kappa,
            temperature=next_temperature,
            params=params,
            rng=rng,
        )
        history.append(
            {
                "event": event + 1,
                "t_start": temperature,
                "t_end": next_temperature,
                "delta_t": delta_t,
                "step_overlap": overlap,
                "pool_overlap_before_reconstruction": overlap_uniform,
                "resampled": resampled,
                "direct_transition": was_direct,
                "mh_acceptance": float(np.mean(acceptance)),
            }
        )
        temperature = next_temperature

    completed = temperature >= 1.0 - 1.0e-12
    if not completed:
        raise RuntimeError(
            f"SMC did not reach t=1 in {params.max_temperature_events} events"
        )
    theta = shell_theta(theta_ref, radius, directions)
    radial_scores = radial_log_target_score(
        theta_ref, radius, directions, a_matrix
    )
    l2_values = 0.5 * np.sum(theta * theta, axis=1)
    return {
        "logZ_CE": float(logz_ce),
        "dlogZ_dr_direct": weighted_mean(radial_scores, log_weights),
        "weighted_ce": weighted_mean(ce_values, log_weights),
        "weighted_error": weighted_mean(errors, log_weights),
        "weighted_l2": weighted_mean(l2_values, log_weights),
        "weighted_energy": weighted_mean(energy(theta, a_matrix), log_weights),
        "temperature_event_count": len(history),
        "resample_count": resample_count,
        "minimum_step_overlap": min(
            float(row["step_overlap"]) for row in history
        ),
        "minimum_pool_overlap": min(
            float(row["pool_overlap_before_reconstruction"]) for row in history
        ),
        "mean_mh_acceptance": float(
            np.mean([float(row["mh_acceptance"]) for row in history])
        ),
        "completed": True,
    }


def load_unit_inputs(row: dict[str, Any], project_root: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = project_root / row["dataset_path"]
    reference_path = project_root / row["reference_path"]
    with np.load(dataset_path, allow_pickle=False) as payload:
        a_matrix = np.asarray(payload["A"], dtype=np.float64)
    with np.load(reference_path, allow_pickle=False) as payload:
        theta_ref = np.asarray(payload["theta"], dtype=np.float64)
    expected_n = int(row["N"])
    if a_matrix.shape != (int(row["M"]), expected_n):
        raise ValueError(f"dataset shape mismatch: {dataset_path}")
    if theta_ref.shape != (expected_n,):
        raise ValueError(f"reference shape mismatch: {reference_path}")
    return a_matrix, theta_ref


def run_unit(
    *,
    row: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    params = SMCParams.from_config(config)
    if int(row["particles_per_split"]) * 2 != int(row["total_particles"]):
        raise ValueError("manifest particle split is not equal")
    split_seeds = [int(value) for value in row["split_seeds"]]
    if len(split_seeds) != 2 or split_seeds[0] == split_seeds[1]:
        raise ValueError("manifest must contain two distinct split seeds")
    a_matrix, theta_ref = load_unit_inputs(row, project_root)
    dimension = int(row["N"])
    radius = float(row["radius"])
    reference_margin = margins(theta_ref, a_matrix)[0]
    if float(np.min(reference_margin)) <= 0.0:
        raise RuntimeError("reference is not strictly hard-feasible")
    reference_ce = float(ce_sum(theta_ref, a_matrix)[0])
    reference_error = float(classification_error(theta_ref, a_matrix)[0])
    mu, base_kappa = l2_matched_parameters(theta_ref, radius)
    splits = [
        run_split(
            theta_ref=theta_ref,
            a_matrix=a_matrix,
            radius=radius,
            particles=int(row["particles_per_split"]),
            seed=split_seeds[split_id],
            params=params,
        )
        for split_id in range(2)
    ]
    split_logz_ce = np.asarray([split["logZ_CE"] for split in splits], float)
    logz_ce = float(logsumexp(split_logz_ce) - math.log(2.0))
    ref_norm_sq = float(theta_ref @ theta_ref)
    log_m = log_sphere_mgf(dimension, base_kappa)
    log_prefactor = -0.5 * dimension * radius * radius + log_m
    reference_l2_log_weight = -0.5 * ref_norm_sq
    split_logz_full = reference_l2_log_weight + log_prefactor + split_logz_ce
    logz_full = reference_l2_log_weight + log_prefactor + logz_ce
    split_derivative = np.asarray(
        [split["dlogZ_dr_direct"] for split in splits], float
    )
    split_mixture = split_target_mixture_weights(split_logz_ce)
    derivative = combine_split_values(split_derivative, split_logz_ce)

    scalar = {
        **row,
        "sampler_method": METHOD,
        "objective": "CE_sum+||w||^2/2",
        "beta": 1.0,
        "lambda_ref": 1.0,
        "lambda_shell": 1.0,
        "reference_ce_sum": reference_ce,
        "reference_error": reference_error,
        "reference_min_margin": float(np.min(reference_margin)),
        "reference_norm": math.sqrt(ref_norm_sq),
        "kappa_r": base_kappa,
        "logM_P_kappa": log_m,
        "reference_l2_log_weight": reference_l2_log_weight,
        "radial_l2_log_prefactor": -0.5 * dimension * radius * radius,
        "logZ_CE": logz_ce,
        "split0_logZ_CE": float(split_logz_ce[0]),
        "split1_logZ_CE": float(split_logz_ce[1]),
        "logZ_angular_full": logz_full,
        "logZ_angular_stripped": log_prefactor + logz_ce,
        "split0_logZ_angular_full": float(split_logz_full[0]),
        "split1_logZ_angular_full": float(split_logz_full[1]),
        "split0_target_mixture_weight": float(split_mixture[0]),
        "split1_target_mixture_weight": float(split_mixture[1]),
        "split_logZ_per_N_difference": float(
            abs(split_logz_full[0] - split_logz_full[1]) / dimension
        ),
        "dlogZ_dr_direct": derivative,
        "split0_dlogZ_dr_direct": float(split_derivative[0]),
        "split1_dlogZ_dr_direct": float(split_derivative[1]),
        "split_dlogZ_dr_per_N_difference": float(
            abs(split_derivative[0] - split_derivative[1]) / dimension
        ),
        "phi_energy_raw": logz_full / dimension,
        "phi_shell_raw": (
            logz_full / dimension
            + (dimension - 2.0) / dimension * math.log(radius)
        ),
        "dphi_energy_dr_direct": derivative / dimension,
        "dphi_shell_dr_direct": (
            derivative / dimension
            + (dimension - 2.0) / (dimension * radius)
        ),
        "weighted_ce_sum": combine_split_values(
            np.asarray([split["weighted_ce"] for split in splits], float),
            split_logz_ce,
        ),
        "weighted_error": combine_split_values(
            np.asarray([split["weighted_error"] for split in splits], float),
            split_logz_ce,
        ),
        "weighted_l2": combine_split_values(
            np.asarray([split["weighted_l2"] for split in splits], float),
            split_logz_ce,
        ),
        "weighted_energy": combine_split_values(
            np.asarray([split["weighted_energy"] for split in splits], float),
            split_logz_ce,
        ),
        "smc_completed": True,
        "temperature_event_count_max": max(
            int(split["temperature_event_count"]) for split in splits
        ),
        "temperature_event_count_total": sum(
            int(split["temperature_event_count"]) for split in splits
        ),
        "resample_count_total": sum(
            int(split["resample_count"]) for split in splits
        ),
        "minimum_step_overlap": min(
            float(split["minimum_step_overlap"]) for split in splits
        ),
        "minimum_pool_overlap": min(
            float(split["minimum_pool_overlap"]) for split in splits
        ),
        "mean_mh_acceptance": float(
            np.mean([split["mean_mh_acceptance"] for split in splits])
        ),
        "derivative_method": "direct_particle_radial_score",
        "finite_difference_used_for_first_derivative": False,
    }

    return scalar, None
