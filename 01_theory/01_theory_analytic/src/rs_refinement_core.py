"""PDF-consistent replica-saddle quadrature and constrained refinements.

This module is the reusable numerical core for the validated production
solver.  It addresses three implementation details identified by the appendix
audit:

* the exact geometric lower endpoint of the Q interval;
* the selected-reference Gaussian integral, evaluated as a conditional
  truncated-normal expectation rather than an indicator on a tensor GH rule;
* the full feasible ``A >= 0`` direction with the physical
  ``max_Q min_(s, eta)`` contour.

The public entry points are :class:`RSRefinementCore`, :class:`RSConfig`, and
the scalar geometry/parameter helpers below.  The energetic kernels are Numba
compiled when Numba is available.  No function in this file writes output.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import ndtri_exp, ndtr, roots_hermitenorm, roots_legendre

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - only used on environments without Numba.
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if args and callable(args[0]):
            return args[0]

        def decorate(func):
            return func

        return decorate


SQRT_2PI = math.sqrt(2.0 * math.pi)


def gh_normal(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normalized weights for a standard Gaussian expectation."""
    x, w = roots_hermitenorm(int(order))
    return np.asarray(x, float), np.asarray(w / SQRT_2PI, float)


def solve_q_ref(alpha: float, gh_order: int = 80) -> float:
    """Solve the scalar reference-overlap saddle equation by bracketing."""
    z, w = gh_normal(gh_order)

    def equation(q: float) -> float:
        q = float(np.clip(q, 1.0e-12, 1.0 - 1.0e-12))
        a = -math.sqrt(q / (1.0 - q)) * z
        tail = np.maximum(ndtr(-a), 1.0e-300)
        density = np.exp(-0.5 * a * a) / SQRT_2PI
        integ = float(np.sum(w * np.square(density / tail)))
        return float(q / (1.0 - q) ** 2 - alpha * integ / (1.0 - q))

    grid = np.linspace(1.0e-7, 0.999, 500)
    values = np.asarray([equation(float(q)) for q in grid])
    brackets = np.flatnonzero(values[:-1] * values[1:] < 0.0)
    if brackets.size == 0:
        raise RuntimeError("Reference-overlap equation has no sign-changing bracket")
    index = int(brackets[0])
    return float(brentq(equation, float(grid[index]), float(grid[index + 1]), xtol=1.0e-14))


def q_ref_residual(q: float, alpha: float, gh_order: int = 160) -> float:
    """Independent high-order residual of the reference saddle equation."""
    z, w = gh_normal(gh_order)
    q = float(q)
    a = -math.sqrt(q / (1.0 - q)) * z
    tail = np.maximum(ndtr(-a), 1.0e-300)
    density = np.exp(-0.5 * a * a) / SQRT_2PI
    integ = float(np.sum(w * np.square(density / tail)))
    return float(q / (1.0 - q) ** 2 - alpha * integ / (1.0 - q))


def q_geometric_bounds(radius: float, q_ref_norm: float) -> tuple[float, float]:
    """Exact inclusive geometric Q endpoints implied by ``|c_d| <= 1``."""
    root = math.sqrt(float(q_ref_norm))
    radius = float(radius)
    return abs(root - radius) ** 2, (root + radius) ** 2


def q_interior_bounds(
    radius: float,
    q_ref_norm: float,
    relative_margin: float = 1.0e-7,
) -> tuple[float, float]:
    """Numerically safe open interval, formed in ``sqrt(Q)`` coordinates."""
    root = math.sqrt(float(q_ref_norm))
    x_lo = abs(root - float(radius))
    x_hi = root + float(radius)
    width = x_hi - x_lo
    margin = max(float(relative_margin) * width, 1.0e-10)
    left = x_lo + margin
    right = x_hi - margin
    if not left < right:
        raise ValueError(f"Empty interior Q interval at radius={radius}")
    return left * left, right * right


def distance_cosine(Q: float, radius: float, q_ref_norm: float) -> float:
    return float(
        (float(Q) + float(q_ref_norm) - float(radius) ** 2)
        / (2.0 * math.sqrt(float(Q) * float(q_ref_norm)))
    )


def full_parameters(cd: float, q_ref: float, s: float, eta: float) -> dict[str, float]:
    """Stable feasible parameterization used in Replica Appendix Eqs. 243--244."""
    cd = float(cd)
    q_ref = float(q_ref)
    s = float(s)
    eta = float(eta)
    one_minus_c2 = max(1.0 - cd * cd, 0.0)
    one_minus_s2 = max(1.0 - s * s, 0.0)
    p_min = cd * cd + s * s * one_minus_c2
    slack = one_minus_c2 * one_minus_s2
    A = eta * slack
    one_minus_p = (1.0 - eta) * slack
    p = 1.0 - one_minus_p
    t = q_ref * cd + s * math.sqrt(max(q_ref * (1.0 - q_ref) * one_minus_c2, 0.0))
    return {
        "p_min": float(p_min),
        "p": float(p),
        "t": float(t),
        "A": float(A),
        "one_minus_p": float(one_minus_p),
    }


def stable_shell_entropy(cd: float, q_ref: float, s: float, eta: float) -> float:
    """Algebraically reduced shell entropy, stable near the feasible boundary."""
    one_minus_s2 = 1.0 - float(s) ** 2
    one_minus_c2 = 1.0 - float(cd) ** 2
    one_minus_eta = 1.0 - float(eta)
    one_minus_q = 1.0 - float(q_ref)
    if min(one_minus_s2, one_minus_c2, one_minus_eta, one_minus_q) <= 0.0:
        return float("inf")
    rational = (one_minus_q + float(q_ref) * float(s) ** 2) / (
        2.0 * one_minus_eta * one_minus_s2 * one_minus_q
    )
    log_one_minus_p = (
        math.log1p(-float(eta))
        + math.log1p(-float(cd) ** 2)
        + math.log1p(-float(s) ** 2)
    )
    return float(rational + 0.5 * math.log(2.0 * math.pi) + 0.5 * log_one_minus_p)


def conditional_z1_rule(
    q_ref: float,
    z0: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre rule for ``z1 | sqrt(q)z0+sqrt(1-q)z1>0``.

    The survival-probability transform is used in log space:
    ``z1=-Phi^{-1}(u Phi(-a))`` with
    ``a=-sqrt(q/(1-q))*z0``.  The returned Legendre weights integrate a
    normalized conditional expectation and therefore sum to one exactly up to
    roundoff for each ``z0`` row.
    """
    nodes, weights = roots_legendre(int(order))
    u = 0.5 * (np.asarray(nodes, float) + 1.0)
    weights = 0.5 * np.asarray(weights, float)
    a = -math.sqrt(float(q_ref) / (1.0 - float(q_ref))) * np.asarray(z0, float)
    log_survival_a = np.asarray(np.log(ndtr(-a)), float)
    log_survival = log_survival_a[:, None] + np.log(u)[None, :]
    z1 = -ndtri_exp(log_survival)
    if not np.isfinite(z1).all():
        raise FloatingPointError("Non-finite conditional z1 quadrature node")
    return np.asarray(z1, float), weights


def conditional_normalization(z0_weights: np.ndarray, conditional_weights: np.ndarray) -> float:
    """Constant-integrand check for the conditional quadrature."""
    return float(np.sum(z0_weights) * np.sum(conditional_weights))


@njit(cache=True)
def _softplus_negative_field(h: float) -> float:
    if h >= 0.0:
        return math.log1p(math.exp(-h))
    return -h + math.log1p(math.exp(h))


@njit(cache=True)
def boundary_energetic_numba(
    Q: float,
    cd: float,
    s: float,
    beta: float,
    q_ref: float,
    z0: np.ndarray,
    w0: np.ndarray,
    z1_conditional: np.ndarray,
    w_conditional: np.ndarray,
    z3: np.ndarray,
    logw3: np.ndarray,
) -> float:
    one_minus_c2 = max(1.0 - cd * cd, 0.0)
    one_minus_s2 = max(1.0 - s * s, 0.0)
    sqrt_q = math.sqrt(q_ref)
    sqrt_one_minus_q = math.sqrt(1.0 - q_ref)
    b0 = sqrt_q * cd + s * math.sqrt((1.0 - q_ref) * one_minus_c2)
    b1 = sqrt_one_minus_q * cd - s * math.sqrt(q_ref * one_minus_c2)
    sqrt_Q = math.sqrt(Q)
    sqrt_one_minus_p = math.sqrt(max(one_minus_c2 * one_minus_s2, 0.0))
    total = 0.0
    for i0 in range(z0.shape[0]):
        conditional_sum = 0.0
        for iu in range(w_conditional.shape[0]):
            base = sqrt_Q * (b0 * z0[i0] + b1 * z1_conditional[i0, iu])
            max_log = -1.0e300
            for i3 in range(z3.shape[0]):
                h = base + sqrt_Q * sqrt_one_minus_p * z3[i3]
                value = logw3[i3] - beta * _softplus_negative_field(h)
                if value > max_log:
                    max_log = value
            exp_sum = 0.0
            for i3 in range(z3.shape[0]):
                h = base + sqrt_Q * sqrt_one_minus_p * z3[i3]
                value = logw3[i3] - beta * _softplus_negative_field(h)
                exp_sum += math.exp(value - max_log)
            conditional_sum += w_conditional[iu] * (max_log + math.log(exp_sum))
        total += w0[i0] * conditional_sum
    return total


@njit(cache=True)
def full_energetic_numba(
    Q: float,
    cd: float,
    s: float,
    eta: float,
    beta: float,
    q_ref: float,
    z0: np.ndarray,
    w0: np.ndarray,
    z1_conditional: np.ndarray,
    w_conditional: np.ndarray,
    z2: np.ndarray,
    w2: np.ndarray,
    z3: np.ndarray,
    logw3: np.ndarray,
) -> float:
    one_minus_c2 = max(1.0 - cd * cd, 0.0)
    one_minus_s2 = max(1.0 - s * s, 0.0)
    slack = one_minus_c2 * one_minus_s2
    A = eta * slack
    one_minus_p = (1.0 - eta) * slack
    sqrt_q = math.sqrt(q_ref)
    sqrt_one_minus_q = math.sqrt(1.0 - q_ref)
    b0 = sqrt_q * cd + s * math.sqrt((1.0 - q_ref) * one_minus_c2)
    b1 = sqrt_one_minus_q * cd - s * math.sqrt(q_ref * one_minus_c2)
    sqrt_Q = math.sqrt(Q)
    sqrt_A = math.sqrt(max(A, 0.0))
    sqrt_one_minus_p = math.sqrt(max(one_minus_p, 0.0))
    total = 0.0
    for i0 in range(z0.shape[0]):
        conditional_sum = 0.0
        for iu in range(w_conditional.shape[0]):
            base = sqrt_Q * (b0 * z0[i0] + b1 * z1_conditional[i0, iu])
            z2_sum = 0.0
            for i2 in range(z2.shape[0]):
                shifted = base + sqrt_Q * sqrt_A * z2[i2]
                max_log = -1.0e300
                for i3 in range(z3.shape[0]):
                    h = shifted + sqrt_Q * sqrt_one_minus_p * z3[i3]
                    value = logw3[i3] - beta * _softplus_negative_field(h)
                    if value > max_log:
                        max_log = value
                exp_sum = 0.0
                for i3 in range(z3.shape[0]):
                    h = shifted + sqrt_Q * sqrt_one_minus_p * z3[i3]
                    value = logw3[i3] - beta * _softplus_negative_field(h)
                    exp_sum += math.exp(value - max_log)
                z2_sum += w2[i2] * (max_log + math.log(exp_sum))
            conditional_sum += w_conditional[iu] * z2_sum
        total += w0[i0] * conditional_sum
    return total


@dataclass(frozen=True)
class RSConfig:
    alpha: float = 0.1
    beta: float = 1.0
    lambda_ref: float = 1.0
    lambda_shell: float = 1.0
    n0: int = 24
    ncond: int = 24
    n2: int = 16
    n3: int = 36
    boundary_epsilon: float = 1.0e-7


@dataclass
class SaddlePoint:
    r: float
    phi: float
    Q: float
    s: float
    eta: float
    A: float
    p: float
    t: float
    cd: float
    qref: float
    G_S: float
    G_E: float
    inner_success: bool = True
    outer_success: bool = True
    evaluations: int = 0

    def to_dict(self) -> dict[str, float | bool | int]:
        return asdict(self)


class RSRefinementCore:
    """Conditional-quadrature evaluator and nested constrained solver."""

    def __init__(self, config: RSConfig = RSConfig(), q_ref: float | None = None) -> None:
        self.config = config
        self.q_ref = float(solve_q_ref(config.alpha) if q_ref is None else q_ref)
        self.q_ref_norm = 1.0 / float(config.lambda_ref)
        self.z0, self.w0 = gh_normal(config.n0)
        self.z1_conditional, self.w_conditional = conditional_z1_rule(
            self.q_ref, self.z0, config.ncond
        )
        self.z2, self.w2 = gh_normal(config.n2)
        self.z3, self.w3 = gh_normal(config.n3)
        self.logw3 = np.log(self.w3)
        self.evaluations = 0

    @property
    def normalization(self) -> float:
        return conditional_normalization(self.w0, self.w_conditional)

    def _valid(self, radius: float, Q: float, s: float, eta: float) -> bool:
        if not (Q > 0.0 and -1.0 < s < 1.0 and 0.0 <= eta < 1.0):
            return False
        cd = distance_cosine(Q, radius, self.q_ref_norm)
        return bool(-1.0 < cd < 1.0)

    def evaluate(
        self,
        radius: float,
        Q: float,
        s: float,
        eta: float = 0.0,
        *,
        force_full: bool = False,
    ) -> SaddlePoint:
        """Evaluate the stable action at one feasible point."""
        radius, Q, s, eta = map(float, (radius, Q, s, eta))
        self.evaluations += 1
        if not self._valid(radius, Q, s, eta):
            return SaddlePoint(
                radius, float("nan"), Q, s, eta, float("nan"), float("nan"),
                float("nan"), float("nan"), self.q_ref, float("nan"), float("nan"),
                inner_success=False, outer_success=False, evaluations=self.evaluations,
            )
        cd = distance_cosine(Q, radius, self.q_ref_norm)
        params = full_parameters(cd, self.q_ref, s, eta)
        G_S = stable_shell_entropy(cd, self.q_ref, s, eta)
        if force_full or eta > 0.0:
            G_E = float(
                full_energetic_numba(
                    Q, cd, s, eta, self.config.beta, self.q_ref,
                    self.z0, self.w0, self.z1_conditional, self.w_conditional,
                    self.z2, self.w2, self.z3, self.logw3,
                )
            )
        else:
            G_E = float(
                boundary_energetic_numba(
                    Q, cd, s, self.config.beta, self.q_ref,
                    self.z0, self.w0, self.z1_conditional, self.w_conditional,
                    self.z3, self.logw3,
                )
            )
        phi = (
            0.5 * math.log(Q)
            - 0.5 * self.config.beta * self.config.lambda_shell * Q
            + G_S
            + self.config.alpha * G_E
        )
        return SaddlePoint(
            r=radius,
            phi=float(phi),
            Q=Q,
            s=s,
            eta=eta,
            A=float(params["A"]),
            p=float(params["p"]),
            t=float(params["t"]),
            cd=float(cd),
            qref=self.q_ref,
            G_S=float(G_S),
            G_E=float(G_E),
            evaluations=self.evaluations,
        )

    @staticmethod
    def _local_minimum_indices(values: np.ndarray) -> list[int]:
        finite = np.isfinite(values)
        indices: list[int] = []
        for index in range(1, len(values) - 1):
            if finite[index] and values[index] <= values[index - 1] and values[index] <= values[index + 1]:
                indices.append(index)
        if np.any(finite):
            global_index = int(np.nanargmin(values))
            if global_index not in indices:
                indices.append(global_index)
        return sorted(indices)

    @staticmethod
    def _local_maximum_indices(values: np.ndarray) -> list[int]:
        finite = np.isfinite(values)
        indices: list[int] = []
        for index in range(1, len(values) - 1):
            if finite[index] and values[index] >= values[index - 1] and values[index] >= values[index + 1]:
                indices.append(index)
        if np.any(finite):
            global_index = int(np.nanargmax(values))
            if global_index not in indices:
                indices.append(global_index)
        return sorted(indices)

    def minimize_boundary_s(
        self,
        radius: float,
        Q: float,
        *,
        s_seed_count: int = 65,
        refine: bool = True,
    ) -> SaddlePoint:
        eps = float(self.config.boundary_epsilon)
        s_grid = np.linspace(-1.0 + eps, 1.0 - eps, int(s_seed_count))
        points = [self.evaluate(radius, Q, float(s), 0.0) for s in s_grid]
        values = np.asarray([point.phi for point in points], float)
        candidates = [points[int(np.nanargmin(values))]]
        if refine:
            for index in self._local_minimum_indices(values):
                left = float(s_grid[max(index - 1, 0)])
                right = float(s_grid[min(index + 1, len(s_grid) - 1)])
                if not left < right:
                    continue
                result = minimize_scalar(
                    lambda sval: self.evaluate(radius, Q, float(sval), 0.0).phi,
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 2.0e-9, "maxiter": 100},
                )
                if result.success and np.isfinite(result.fun):
                    point = self.evaluate(radius, Q, float(result.x), 0.0)
                    point.inner_success = bool(result.success)
                    candidates.append(point)
        return min(candidates, key=lambda point: point.phi)

    def minimize_full_s_eta(
        self,
        radius: float,
        Q: float,
        *,
        s_seed_count: int = 17,
        eta_seeds: Sequence[float] = (0.0, 0.001, 0.01, 0.05, 0.2, 0.6),
        max_starts: int = 5,
        maxiter: int = 120,
    ) -> SaddlePoint:
        """Joint physical inner minimization, including the ``eta=0`` face."""
        eps = float(self.config.boundary_epsilon)
        boundary = self.minimize_boundary_s(
            radius, Q, s_seed_count=max(17, int(s_seed_count)), refine=True
        )
        s_grid = np.linspace(-1.0 + eps, 1.0 - eps, int(s_seed_count))
        seed_points: list[tuple[float, float, float]] = [(boundary.phi, boundary.s, 0.0)]
        for eta in eta_seeds:
            eta_value = float(np.clip(eta, 0.0, 1.0 - eps))
            for s_value in s_grid:
                point = self.evaluate(radius, Q, float(s_value), eta_value, force_full=True)
                if np.isfinite(point.phi):
                    seed_points.append((point.phi, float(s_value), eta_value))
        seed_points.sort(key=lambda item: item[0])
        selected: list[tuple[float, float]] = [(boundary.s, 0.0)]
        for _, s_value, eta_value in seed_points:
            if all(abs(s_value - old_s) > 0.08 or abs(eta_value - old_eta) > 0.02 for old_s, old_eta in selected):
                selected.append((s_value, eta_value))
            if len(selected) >= int(max_starts):
                break

        candidates = [boundary]
        bounds = [(-1.0 + eps, 1.0 - eps), (0.0, 1.0 - eps)]
        for seed in selected:
            result = minimize(
                lambda x: self.evaluate(
                    radius, Q, float(x[0]), float(x[1]), force_full=True
                ).phi,
                np.asarray(seed, float),
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "xatol": 2.0e-8,
                    "fatol": 2.0e-10,
                    "maxiter": int(maxiter),
                },
            )
            if result.success and np.isfinite(result.fun):
                point = self.evaluate(
                    radius, Q, float(result.x[0]), float(result.x[1]), force_full=True
                )
                point.inner_success = bool(result.success)
                candidates.append(point)
        return min(candidates, key=lambda point: point.phi)

    def _sqrtQ_grid(self, radius: float, count: int) -> np.ndarray:
        qlo, qhi = q_interior_bounds(
            radius, self.q_ref_norm, self.config.boundary_epsilon
        )
        return np.linspace(math.sqrt(qlo), math.sqrt(qhi), int(count))

    def solve_boundary(
        self,
        radius: float,
        *,
        q_scan_count: int = 129,
        s_seed_count: int = 65,
        refine_inner: bool = True,
        refine_outer: bool = True,
    ) -> SaddlePoint:
        x_grid = self._sqrtQ_grid(radius, q_scan_count)
        points = [
            self.minimize_boundary_s(
                radius, float(x * x), s_seed_count=s_seed_count, refine=refine_inner
            )
            for x in x_grid
        ]
        values = np.asarray([point.phi for point in points], float)
        candidates = [points[int(np.nanargmax(values))]]
        if refine_outer:
            for index in self._local_maximum_indices(values):
                left = float(x_grid[max(index - 1, 0)])
                right = float(x_grid[min(index + 1, len(x_grid) - 1)])
                if not left < right:
                    continue

                def outer_objective(x_value: float) -> float:
                    return -self.minimize_boundary_s(
                        radius,
                        float(x_value * x_value),
                        s_seed_count=s_seed_count,
                        refine=refine_inner,
                    ).phi

                result = minimize_scalar(
                    outer_objective,
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 2.0e-8, "maxiter": 100},
                )
                if result.success and np.isfinite(result.fun):
                    point = self.minimize_boundary_s(
                        radius,
                        float(result.x * result.x),
                        s_seed_count=s_seed_count,
                        refine=refine_inner,
                    )
                    point.outer_success = bool(result.success)
                    candidates.append(point)
        return max(candidates, key=lambda point: point.phi)

    def solve_full(
        self,
        radius: float,
        *,
        q_scan_count: int = 25,
        s_seed_count: int = 13,
        eta_seeds: Sequence[float] = (0.0, 0.001, 0.01, 0.05, 0.2, 0.6),
        max_starts: int = 4,
        refine_outer: bool = True,
        maxiter: int = 100,
    ) -> SaddlePoint:
        x_grid = self._sqrtQ_grid(radius, q_scan_count)

        def inner(x_value: float) -> SaddlePoint:
            return self.minimize_full_s_eta(
                radius,
                float(x_value * x_value),
                s_seed_count=s_seed_count,
                eta_seeds=eta_seeds,
                max_starts=max_starts,
                maxiter=maxiter,
            )

        points = [inner(float(x)) for x in x_grid]
        values = np.asarray([point.phi for point in points], float)
        candidates = [points[int(np.nanargmax(values))]]
        if refine_outer:
            for index in self._local_maximum_indices(values):
                left = float(x_grid[max(index - 1, 0)])
                right = float(x_grid[min(index + 1, len(x_grid) - 1)])
                if not left < right:
                    continue
                result = minimize_scalar(
                    lambda x_value: -inner(float(x_value)).phi,
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 1.0e-7, "maxiter": 70},
                )
                if result.success and np.isfinite(result.fun):
                    point = inner(float(result.x))
                    point.outer_success = bool(result.success)
                    candidates.append(point)
        return max(candidates, key=lambda point: point.phi)

    def eta_forward_derivative(self, point: SaddlePoint, step: float = 1.0e-5) -> float:
        if point.eta != 0.0:
            raise ValueError("Forward boundary derivative is defined here only at eta=0")
        f0 = self.evaluate(point.r, point.Q, point.s, 0.0, force_full=True).phi
        f1 = self.evaluate(point.r, point.Q, point.s, float(step), force_full=True).phi
        return float((f1 - f0) / float(step))

    def stationarity_residuals(self, point: SaddlePoint, step: float = 1.0e-5) -> dict[str, float]:
        """Central/one-sided finite-difference diagnostics at a selected point."""
        h = float(step)
        x = math.sqrt(point.Q)
        fp = self.evaluate(point.r, (x + h) ** 2, point.s, point.eta, force_full=True).phi
        fm = self.evaluate(point.r, (x - h) ** 2, point.s, point.eta, force_full=True).phi
        sp = self.evaluate(point.r, point.Q, point.s + h, point.eta, force_full=True).phi
        sm = self.evaluate(point.r, point.Q, point.s - h, point.eta, force_full=True).phi
        result = {
            "dF_dsqrtQ": float((fp - fm) / (2.0 * h)),
            "dF_ds": float((sp - sm) / (2.0 * h)),
        }
        if point.eta <= h:
            f0 = self.evaluate(point.r, point.Q, point.s, 0.0, force_full=True).phi
            fe = self.evaluate(point.r, point.Q, point.s, h, force_full=True).phi
            result["dF_deta"] = float((fe - f0) / h)
        else:
            ep = self.evaluate(point.r, point.Q, point.s, point.eta + h, force_full=True).phi
            em = self.evaluate(point.r, point.Q, point.s, point.eta - h, force_full=True).phi
            result["dF_deta"] = float((ep - em) / (2.0 * h))
        return result


def relative_curve(points: Iterable[SaddlePoint]) -> np.ndarray:
    values = np.asarray([point.phi for point in points], float)
    return values - values[0]
