"""Frozen perceptron objective and exact L2 shell identities for 01_theory."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class PerceptronObjective:
    """The original theory objective; it is intentionally not DNN ``H_0.01``."""

    beta: float = 1.0
    lambda_ref: float = 1.0
    lambda_shell: float = 1.0

    def validate(self) -> None:
        if (self.beta, self.lambda_ref, self.lambda_shell) != (1.0, 1.0, 1.0):
            raise ValueError(
                "01_theory is frozen to beta=lambda_ref=lambda_shell=1"
            )


FROZEN_OBJECTIVE = PerceptronObjective()


def margins(theta_batch: np.ndarray, a_matrix: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_batch, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[None, :]
    a = np.asarray(a_matrix, dtype=np.float64)
    if theta.ndim != 2 or a.ndim != 2 or theta.shape[1] != a.shape[1]:
        raise ValueError(
            f"incompatible theta/A shapes: theta={theta.shape}, A={a.shape}"
        )
    return (theta @ a.T) / math.sqrt(theta.shape[1])


def ce_sum(theta_batch: np.ndarray, a_matrix: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, -margins(theta_batch, a_matrix)).sum(axis=1)


def classification_error(
    theta_batch: np.ndarray, a_matrix: np.ndarray
) -> np.ndarray:
    return np.mean(margins(theta_batch, a_matrix) <= 0.0, axis=1)


def energy(theta_batch: np.ndarray, a_matrix: np.ndarray) -> np.ndarray:
    """Return ``sum_mu softplus(-h_mu) + ||w||^2/2``."""
    theta = np.asarray(theta_batch, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[None, :]
    return ce_sum(theta, a_matrix) + 0.5 * np.sum(theta * theta, axis=1)


def ce_gradient(theta_batch: np.ndarray, a_matrix: np.ndarray) -> np.ndarray:
    """Exact vectorized gradient of the CE sum."""
    theta = np.asarray(theta_batch, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[None, :]
    a = np.asarray(a_matrix, dtype=np.float64)
    h = margins(theta, a)
    return -(expit(-h) @ a) / math.sqrt(theta.shape[1])


def l2_matched_parameters(
    theta_ref: np.ndarray, radius: float
) -> tuple[np.ndarray, float]:
    theta_ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(theta_ref))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("reference norm must be positive and finite")
    dimension = int(theta_ref.size)
    mu = -theta_ref / norm
    kappa = math.sqrt(dimension) * float(radius) * norm
    return mu, float(kappa)


def shell_theta(
    theta_ref: np.ndarray, radius: float, directions: np.ndarray
) -> np.ndarray:
    theta_ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[None, :]
    if directions.shape[1] != theta_ref.size:
        raise ValueError("direction/reference dimension mismatch")
    return theta_ref[None, :] + math.sqrt(theta_ref.size) * float(radius) * directions


def factorized_negative_energy(
    theta_ref: np.ndarray,
    radius: float,
    directions: np.ndarray,
    a_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return direct and L2-matched-factorized values of ``-E``.

    The identity checked here is

    ``-E(w_ref+sqrt(N) r u) =
      -||w_ref||^2/2 - N r^2/2 + kappa mu.u - CE_sum``.
    """
    theta_ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[None, :]
    theta = shell_theta(theta_ref, radius, directions)
    direct = -energy(theta, a_matrix)
    mu, kappa = l2_matched_parameters(theta_ref, radius)
    factorized = (
        -0.5 * float(theta_ref @ theta_ref)
        - 0.5 * theta_ref.size * float(radius) ** 2
        + kappa * (directions @ mu)
        - ce_sum(theta, a_matrix)
    )
    return direct, factorized


def radial_log_target_score(
    theta_ref: np.ndarray,
    radius: float,
    directions: np.ndarray,
    a_matrix: np.ndarray,
) -> np.ndarray:
    """Direct first derivative ``d[-E(w_ref+sqrt(N)ru)]/dr``.

    This is a genuine radial derivative evaluated at each final particle.  It
    is not a finite difference across neighboring radii.
    """
    theta_ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[None, :]
    theta = shell_theta(theta_ref, radius, directions)
    gradient = ce_gradient(theta, a_matrix)
    dimension = theta_ref.size
    return (
        -math.sqrt(dimension) * np.sum(gradient * directions, axis=1)
        -math.sqrt(dimension) * (directions @ theta_ref)
        -dimension * float(radius)
    )


def assert_unit_directions(
    directions: np.ndarray, *, tolerance: float = 2.0e-10
) -> None:
    norm = np.linalg.norm(np.asarray(directions, dtype=np.float64), axis=1)
    if not np.all(np.isfinite(norm)) or not np.allclose(
        norm, 1.0, rtol=0.0, atol=tolerance
    ):
        raise RuntimeError("SMC directions left the unit sphere")
