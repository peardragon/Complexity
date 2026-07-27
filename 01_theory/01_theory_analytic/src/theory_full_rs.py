"""Replica-symmetric analytic curve with a validated full-``A >= 0`` saddle.

The production path uses the exact geometric ``Q`` interval, a full
``s in (-1, 1)`` boundary search, conditional-CDF quadrature for the selected
reference, continuous saddle refinement, and a constrained ``eta`` coordinate
for ``A >= 0``.  The stationary coordinates are confirmed at the 32-point
tier and the reported action is evaluated at the 48-point reference tier.

``FullRS`` is retained as the historical masked-quadrature ``A=0`` facade for
notebooks and audit scripts that import its public methods.  ``compute_rows``
selects the corrected production path when ``solver_mode`` says so.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import gammaln, ive, logsumexp, ndtr, roots_hermitenorm

# Keep file-based imports used by the appendix audit compatible: unlike normal
# script execution, importlib does not automatically add this file's directory
# to sys.path.
SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from rs_refinement_core import (
    NUMBA_AVAILABLE,
    RSConfig,
    RSRefinementCore,
    SaddlePoint,
    q_geometric_bounds,
    q_interior_bounds,
)


DEFAULT_RADII = tuple(round(0.15 + 0.05 * idx, 10) for idx in range(42))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "01_theory" / "01_theory_analytic" / "config" / "default.json"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "01_theory"
    / "01_theory_analytic"
    / "summarized_outputs"
    / "phi_by_analytic_solution_alpha0p1.csv"
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Historical A=0 compatibility facade
# ---------------------------------------------------------------------------


def stable_ce_sum(h: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, -h).sum(axis=-1)


def log_M_sphere(dim: int, kappa: float) -> float:
    if kappa < 1.0e-10:
        return 0.0
    nu = dim / 2.0 - 1.0
    val = ive(nu, kappa)
    if val <= 0 or not np.isfinite(val):
        log_i = kappa - 0.5 * np.log(2.0 * np.pi * kappa)
    else:
        log_i = np.log(val) + kappa
    return float(gammaln(dim / 2.0) + nu * np.log(2.0 / kappa) + log_i)


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("-inf")
    return float(logsumexp(values) - np.log(values.size))


def gh_norm(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = roots_hermitenorm(n)
    return x, w / np.sqrt(2.0 * np.pi)


def h_tail(x: np.ndarray) -> np.ndarray:
    return ndtr(-x)


def phi_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def solve_q_ref(alpha: float, gh_order: int = 80) -> float:
    """Historical public reference-overlap helper."""
    z, w = gh_norm(gh_order)

    def f(q: float) -> float:
        q = float(np.clip(q, 1.0e-9, 1.0 - 1.0e-9))
        a = -np.sqrt(q / (1.0 - q)) * z
        integ = np.sum(w * (phi_pdf(a) / np.maximum(h_tail(a), 1.0e-300)) ** 2)
        return float(q / (1.0 - q) ** 2 - alpha / (1.0 - q) * integ)

    xs = np.linspace(1.0e-6, 0.999, 300)
    vals = np.asarray([f(float(x)) for x in xs])
    for idx in range(len(xs) - 1):
        if vals[idx] * vals[idx + 1] < 0:
            return float(brentq(f, xs[idx], xs[idx + 1]))
    return float(xs[int(np.argmin(np.abs(vals)))])


def convert_qs(q_norm: float, s_value: float, cd: float, q_ref: float) -> tuple[float, float]:
    p = cd * cd + s_value * s_value * (1.0 - cd * cd)
    t = q_ref * cd + s_value * np.sqrt(
        max(q_ref * (1.0 - q_ref) * (1.0 - cd * cd), 0.0)
    )
    return float(p), float(t)


def gs_entropy(p: float, t: float, cd: float, q_ref: float) -> float:
    den = 2.0 * (1.0 - p) * (1.0 - q_ref) ** 2
    num = (
        (1.0 - cd**2) * (1.0 - 2.0 * q_ref)
        + q_ref * q_ref
        - 2.0 * q_ref * cd * t
        + t * t
    )
    return float(
        num / den + 0.5 * np.log(2.0 * np.pi) + 0.5 * np.log(1.0 - p)
    )


class FullRS:
    """Historical tensor-GH, narrow-grid, ``A=0`` implementation.

    This class intentionally preserves its pre-promotion method signatures and
    numerical meaning.  The corrected production solver is
    :class:`CorrectedFullRS` below.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        beta: float = 1.0,
        lambda_ref: float = 1.0,
        lambda_shell: float = 1.0,
        n0: int = 13,
        n1: int = 13,
        n3: int = 19,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_ref = float(lambda_ref)
        self.lambda_shell = float(lambda_shell)
        self.q_ref = solve_q_ref(self.alpha)
        self.z0, self.w0 = gh_norm(n0)
        self.z1, self.w1 = gh_norm(n1)
        self.z3, self.w3 = gh_norm(n3)
        self.z0_grid = self.z0[:, None, None]
        self.z1_grid = self.z1[None, :, None]
        self.z3_grid = self.z3[None, None, :]
        self.w1_cond = self.w1[None, :]
        self.logw3 = np.log(self.w3)[None, None, :]

    def ge_energy(self, q_norm: float, s_value: float, cd: float) -> float:
        q_ref = self.q_ref
        p, t = convert_qs(q_norm, s_value, cd, q_ref)
        if p >= 1.0 or p < -1.0:
            return float("nan")
        uref = (
            np.sqrt(q_ref) * self.z0_grid
            + np.sqrt(1.0 - q_ref) * self.z1_grid
        )
        mask = (uref[:, :, 0] > 0).astype(float)
        denom = np.maximum(
            h_tail(-np.sqrt(q_ref / (1.0 - q_ref)) * self.z0), 1.0e-300
        )
        base = (
            (t / np.sqrt(q_ref)) * self.z0_grid
            + ((cd - t) / np.sqrt(1.0 - q_ref)) * self.z1_grid
        )
        h = np.sqrt(q_norm) * (
            base + np.sqrt(max(1.0 - p, 1.0e-14)) * self.z3_grid
        )
        logs = self.logw3 - self.beta * np.logaddexp(0.0, -h)
        maximum = np.max(logs, axis=2, keepdims=True)
        inner = np.squeeze(maximum, 2) + np.log(
            np.sum(np.exp(logs - maximum), axis=2)
        )
        sums = np.sum(self.w1_cond * mask * inner, axis=1) / denom
        return float(np.sum(self.w0 * sums))

    def action(self, q_norm: float, s_value: float, radius: float) -> float:
        q_ref_norm = 1.0 / self.lambda_ref
        if q_norm <= 0.0:
            return float("nan")
        cd = (q_norm + q_ref_norm - radius * radius) / (
            2.0 * np.sqrt(q_norm * q_ref_norm)
        )
        if not (-1.0 < cd < 1.0):
            return float("nan")
        p, t = convert_qs(q_norm, s_value, cd, self.q_ref)
        if p >= 1.0:
            return float("nan")
        return float(
            0.5 * np.log(q_norm)
            - 0.5 * self.beta * self.lambda_shell * q_norm
            + gs_entropy(p, t, cd, self.q_ref)
            + self.alpha * self.ge_energy(q_norm, s_value, cd)
        )

    def solve_radius(
        self,
        radius: float,
        *,
        q_grid_count: int,
        s_grid_count: int,
        s_abs_max: float,
    ) -> dict[str, float]:
        q_ref_norm = 1.0 / self.lambda_ref
        lo = max(0.0, np.sqrt(q_ref_norm) - radius) ** 2 + 1.0e-4
        hi = (np.sqrt(q_ref_norm) + radius) ** 2 - 1.0e-4
        q_grid = np.linspace(lo, hi, int(q_grid_count))
        s_grid = np.linspace(-float(s_abs_max), float(s_abs_max), int(s_grid_count))
        best: tuple[float, float, float] | None = None
        for q_norm in q_grid:
            vals = np.asarray(
                [self.action(float(q_norm), float(s), radius) for s in s_grid]
            )
            if not np.any(np.isfinite(vals)):
                continue
            idx = int(np.nanargmin(vals))
            val = float(vals[idx])
            if best is None or val > best[0]:
                best = (val, float(q_norm), float(s_grid[idx]))
        if best is None:
            raise RuntimeError(f"No finite historical A=0 branch for radius={radius}")
        val, q_norm, s_value = best
        cd = (q_norm + q_ref_norm - radius * radius) / (
            2.0 * np.sqrt(q_norm * q_ref_norm)
        )
        p, t = convert_qs(q_norm, s_value, cd, self.q_ref)
        return {
            "r": float(radius),
            "phi": val,
            "Q": q_norm,
            "p": p,
            "t": t,
            "cd": float(cd),
            "s": s_value,
            "qref": self.q_ref,
        }


# ---------------------------------------------------------------------------
# Validated corrected production path
# ---------------------------------------------------------------------------


def _action_from_vector(
    core: RSRefinementCore,
    radius: float,
    vector: Sequence[float],
    *,
    full: bool,
) -> float:
    sqrt_q, s_value = float(vector[0]), float(vector[1])
    eta = float(vector[2]) if full else 0.0
    return core.evaluate(
        radius, sqrt_q * sqrt_q, s_value, eta, force_full=full
    ).phi


def _numerical_gradient(
    core: RSRefinementCore,
    radius: float,
    vector: Sequence[float],
    *,
    full: bool,
) -> np.ndarray:
    values = np.asarray(vector, float)
    qlo, qhi = q_interior_bounds(
        radius, core.q_ref_norm, core.config.boundary_epsilon
    )
    sqrt_q_width = math.sqrt(qhi) - math.sqrt(qlo)
    steps = [max(2.0e-5, 2.0e-5 * sqrt_q_width), 2.0e-5]
    if full:
        steps.append(min(2.0e-5, max(2.0e-8, 0.35 * values[2])))
    gradient = np.empty(len(values), float)
    for index, step in enumerate(steps):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            _action_from_vector(core, radius, plus, full=full)
            - _action_from_vector(core, radius, minus, full=full)
        ) / (2.0 * step)
    return gradient


def _numerical_hessian(
    core: RSRefinementCore,
    radius: float,
    vector: Sequence[float],
    *,
    full: bool,
) -> np.ndarray:
    values = np.asarray(vector, float)
    steps = np.asarray(
        [2.0e-4, 2.0e-4]
        + ([min(1.0e-4, 0.25 * values[2])] if full else []),
        float,
    )
    steps = np.maximum(steps, 2.0e-7)
    base = _action_from_vector(core, radius, values, full=full)
    hessian = np.empty((len(values), len(values)), float)
    for i in range(len(values)):
        ei = np.zeros_like(values)
        ei[i] = steps[i]
        hessian[i, i] = (
            _action_from_vector(core, radius, values + ei, full=full)
            - 2.0 * base
            + _action_from_vector(core, radius, values - ei, full=full)
        ) / steps[i] ** 2
        for j in range(i):
            ej = np.zeros_like(values)
            ej[j] = steps[j]
            hessian[i, j] = hessian[j, i] = (
                _action_from_vector(core, radius, values + ei + ej, full=full)
                - _action_from_vector(core, radius, values + ei - ej, full=full)
                - _action_from_vector(core, radius, values - ei + ej, full=full)
                + _action_from_vector(core, radius, values - ei - ej, full=full)
            ) / (4.0 * steps[i] * steps[j])
    return hessian


def _curvature_diagnostics(hessian: np.ndarray) -> dict[str, float | bool]:
    inner = hessian[1:, 1:]
    inner_eigenvalues = np.linalg.eigvalsh(inner)
    try:
        reduced_outer = float(
            hessian[0, 0]
            - hessian[0, 1:] @ np.linalg.solve(inner, hessian[1:, 0])
        )
    except np.linalg.LinAlgError:
        reduced_outer = float("nan")
    return {
        "inner_hessian_min_eigenvalue": float(inner_eigenvalues.min()),
        "inner_hessian_max_eigenvalue": float(inner_eigenvalues.max()),
        "reduced_outer_curvature": reduced_outer,
        "physical_curvature_signature": bool(
            inner_eigenvalues.min() > 0.0
            and np.isfinite(reduced_outer)
            and reduced_outer < 0.0
        ),
    }


def _refine_stationary(
    core: RSRefinementCore,
    radius: float,
    seed: Sequence[float],
    *,
    full: bool,
    max_nfev: int,
) -> tuple[SaddlePoint, dict[str, Any]]:
    qlo, qhi = q_interior_bounds(
        radius, core.q_ref_norm, core.config.boundary_epsilon
    )
    lower = [math.sqrt(qlo), -1.0 + core.config.boundary_epsilon]
    upper = [math.sqrt(qhi), 1.0 - core.config.boundary_epsilon]
    seed_array = np.asarray(seed, float).copy()
    if full:
        lower.append(1.0e-8)
        upper.append(0.95)
        seed_array[2] = float(np.clip(seed_array[2], 2.0e-7, 0.94))
    result = least_squares(
        lambda vector: _numerical_gradient(core, radius, vector, full=full),
        seed_array,
        bounds=(np.asarray(lower), np.asarray(upper)),
        xtol=2.0e-10,
        ftol=2.0e-10,
        gtol=2.0e-10,
        x_scale="jac",
        max_nfev=int(max_nfev),
    )
    vector = np.asarray(result.x, float)
    point = core.evaluate(
        radius,
        float(vector[0] ** 2),
        float(vector[1]),
        float(vector[2]) if full else 0.0,
        force_full=full,
    )
    gradient = _numerical_gradient(core, radius, vector, full=full)
    hessian = _numerical_hessian(core, radius, vector, full=full)
    diagnostics: dict[str, Any] = {
        "root_success": bool(result.success),
        "root_status": int(result.status),
        "root_nfev": int(result.nfev),
        "root_cost": float(result.cost),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "dF_dsqrtQ": float(gradient[0]),
        "dF_ds": float(gradient[1]),
        "dF_deta": float(gradient[2]) if full else float("nan"),
    }
    diagnostics.update(_curvature_diagnostics(hessian))
    return point, diagnostics


def _orders(
    config: dict[str, Any], key: str, default: Sequence[int]
) -> tuple[int, int, int, int]:
    raw = config.get(key, default)
    if isinstance(raw, dict):
        values = [raw[name] for name in ("n0", "ncond", "n2", "n3")]
    else:
        values = list(raw)
    if len(values) != 4 or any(int(value) < 2 for value in values):
        raise ValueError(f"{key} must contain four quadrature orders >= 2")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _core_config(config: dict[str, Any], orders: Sequence[int]) -> RSConfig:
    return RSConfig(
        alpha=float(config.get("alpha", 0.1)),
        beta=float(config.get("beta", 1.0)),
        lambda_ref=float(config.get("lambda_ref", 1.0)),
        lambda_shell=float(config.get("lambda_shell", 1.0)),
        n0=int(orders[0]),
        ncond=int(orders[1]),
        n2=int(orders[2]),
        n3=int(orders[3]),
        boundary_epsilon=float(config.get("boundary_epsilon", 1.0e-7)),
    )


def _tier_name(prefix: str, orders: Sequence[int]) -> str:
    return f"{prefix}_{int(orders[0])}_{int(orders[1])}_{int(orders[2])}_{int(orders[3])}"


class CorrectedFullRS:
    """Deterministic production solver for ``max_Q min_(s, eta) F``."""

    def __init__(self, config: dict[str, Any]) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "corrected_full_A requires Numba; the pure-Python nested "
                "quadrature fallback is not viable for a production sweep"
            )
        self.config = dict(config)
        self.screen_orders = _orders(
            config, "boundary_screen_orders", (16, 16, 12, 24)
        )
        self.standard_orders = _orders(
            config, "standard_orders", (24, 24, 16, 36)
        )
        self.confirmation_orders = _orders(
            config, "confirmation_orders", (32, 32, 24, 48)
        )
        self.reference_orders = _orders(
            config, "reference_orders", (48, 48, 32, 72)
        )
        self.screen = RSRefinementCore(_core_config(config, self.screen_orders))
        self.q_ref = self.screen.q_ref
        self.boundary_tiers: list[tuple[str, RSRefinementCore]] = []
        for prefix, orders in (
            ("standard", self.standard_orders),
            ("confirmation", self.confirmation_orders),
            ("reference", self.reference_orders),
        ):
            self.boundary_tiers.append(
                (
                    _tier_name(prefix, orders),
                    RSRefinementCore(_core_config(config, orders), q_ref=self.q_ref),
                )
            )
        self.confirmation_name, self.confirmation = self.boundary_tiers[1]
        self.reference_name, self.reference = self.boundary_tiers[2]
        self.q_scan_count = int(config.get("boundary_q_scan_count", 129))
        self.s_seed_count = int(config.get("boundary_s_seed_count", 65))
        self.boundary_root_max_nfev = int(
            config.get("boundary_root_max_nfev", 45)
        )
        self.full_root_max_nfev = int(config.get("full_root_max_nfev", 50))
        self.full_eta_initial = float(config.get("full_eta_initial", 1.0e-3))
        self.boundary_gradient_tolerance = float(
            config.get("boundary_gradient_tolerance", 5.0e-5)
        )
        self.confirmation_gradient_tolerance = float(
            config.get("confirmation_gradient_tolerance", 5.0e-5)
        )
        self.reference_gradient_tolerance = float(
            config.get("reference_gradient_tolerance", 5.0e-5)
        )
        if self.q_scan_count < 3 or self.s_seed_count < 3:
            raise ValueError("Boundary Q and s seed counts must both be >= 3")
        if not 1.0e-8 < self.full_eta_initial < 0.95:
            raise ValueError("full_eta_initial must lie in (1e-8, 0.95)")
        for core in [self.screen, *(item[1] for item in self.boundary_tiers)]:
            if not np.isfinite(core.normalization) or abs(core.normalization - 1.0) > 5.0e-13:
                raise RuntimeError(
                    "Conditional-CDF quadrature failed its constant-integrand check"
                )

    def solve_radius(self, radius: float) -> dict[str, Any]:
        radius = float(radius)

        # Global branch selection on the exact open geometric interval and the
        # full physical s domain, followed by continuous saddle continuation.
        boundary = self.screen.solve_boundary(
            radius,
            q_scan_count=self.q_scan_count,
            s_seed_count=self.s_seed_count,
            refine_inner=True,
            refine_outer=True,
        )
        boundary_diagnostic: dict[str, Any] = {}
        for tier_name, core in self.boundary_tiers:
            boundary, boundary_diagnostic = _refine_stationary(
                core,
                radius,
                (math.sqrt(boundary.Q), boundary.s),
                full=False,
                max_nfev=self.boundary_root_max_nfev,
            )
            boundary_gradient_max = float(
                boundary_diagnostic["gradient_max_abs"]
            )
            if not boundary_diagnostic["root_success"]:
                raise RuntimeError(
                    f"Boundary saddle root failed at {tier_name}, radius={radius}"
                )
            if not boundary_diagnostic["physical_curvature_signature"]:
                raise RuntimeError(
                    "Boundary saddle has the wrong curvature signature at "
                    f"{tier_name}, radius={radius}"
                )
            if (
                not np.isfinite(boundary_gradient_max)
                or boundary_gradient_max > self.boundary_gradient_tolerance
            ):
                raise RuntimeError(
                    "Boundary saddle residual exceeds tolerance at "
                    f"{tier_name}, radius={radius}"
                )

        # A direct confirmation-tier root from the high-order A=0 saddle was
        # checked against the longer screen -> standard -> confirmation route
        # at all 42 production radii.  It selects the same physical branch and
        # agrees in the action to < 7e-15.
        confirmed, confirmation_diagnostic = _refine_stationary(
            self.confirmation,
            radius,
            (math.sqrt(boundary.Q), boundary.s, self.full_eta_initial),
            full=True,
            max_nfev=self.full_root_max_nfev,
        )
        if not confirmation_diagnostic["root_success"]:
            raise RuntimeError(f"Full-A saddle root failed at radius={radius}")
        if not confirmation_diagnostic["physical_curvature_signature"]:
            raise RuntimeError(
                f"Full-A saddle has the wrong curvature signature at radius={radius}"
            )
        confirmation_gradient_max = float(
            confirmation_diagnostic["gradient_max_abs"]
        )
        if (
            not np.isfinite(confirmation_gradient_max)
            or confirmation_gradient_max > self.confirmation_gradient_tolerance
        ):
            raise RuntimeError(
                f"Full-A confirmation residual exceeds tolerance at radius={radius}"
            )

        # The accepted validation curve is intentionally hybrid: stationary
        # coordinates are selected at the confirmation tier, while the action
        # and reported residual are evaluated at the higher reference tier.
        reference = self.reference.evaluate(
            radius,
            confirmed.Q,
            confirmed.s,
            confirmed.eta,
            force_full=True,
        )
        reference_vector = np.asarray(
            [math.sqrt(confirmed.Q), confirmed.s, confirmed.eta], float
        )
        reference_gradient = _numerical_gradient(
            self.reference, radius, reference_vector, full=True
        )
        reference_gradient_max = float(np.max(np.abs(reference_gradient)))
        if (
            not np.isfinite(reference_gradient_max)
            or reference_gradient_max > self.reference_gradient_tolerance
        ):
            raise RuntimeError(
                f"Reference-tier residual exceeds tolerance at radius={radius}"
            )

        q_lower, q_upper = q_geometric_bounds(radius, self.reference.q_ref_norm)
        return {
            "r": radius,
            "phi": float(reference.phi),
            "Q": float(reference.Q),
            "p": float(reference.p),
            "t": float(reference.t),
            "cd": float(reference.cd),
            "s": float(reference.s),
            "qref": float(reference.qref),
            "eta": float(reference.eta),
            "A": float(reference.A),
            "G_S": float(reference.G_S),
            "G_E": float(reference.G_E),
            "alpha_G_E": float(self.reference.config.alpha * reference.G_E),
            "quadrature_tier": self.reference_name,
            "gradient_max_abs": reference_gradient_max,
            "dF_dsqrtQ": float(reference_gradient[0]),
            "dF_ds": float(reference_gradient[1]),
            "dF_deta": float(reference_gradient[2]),
            "confirmation_quadrature_tier": self.confirmation_name,
            "confirmation_gradient_max_abs": float(
                confirmation_diagnostic["gradient_max_abs"]
            ),
            "confirmation_dF_dsqrtQ": float(
                confirmation_diagnostic["dF_dsqrtQ"]
            ),
            "confirmation_dF_ds": float(confirmation_diagnostic["dF_ds"]),
            "confirmation_dF_deta": float(confirmation_diagnostic["dF_deta"]),
            "root_success": bool(confirmation_diagnostic["root_success"]),
            "root_status": int(confirmation_diagnostic["root_status"]),
            "root_nfev": int(confirmation_diagnostic["root_nfev"]),
            "root_cost": float(confirmation_diagnostic["root_cost"]),
            "inner_hessian_min_eigenvalue": float(
                confirmation_diagnostic["inner_hessian_min_eigenvalue"]
            ),
            "inner_hessian_max_eigenvalue": float(
                confirmation_diagnostic["inner_hessian_max_eigenvalue"]
            ),
            "reduced_outer_curvature": float(
                confirmation_diagnostic["reduced_outer_curvature"]
            ),
            "physical_curvature_signature": bool(
                confirmation_diagnostic["physical_curvature_signature"]
            ),
            "boundary_Q": float(boundary.Q),
            "boundary_s": float(boundary.s),
            "boundary_gradient_max_abs": float(
                boundary_diagnostic["gradient_max_abs"]
            ),
            "Q_lower_exact": float(q_lower),
            "Q_upper_exact": float(q_upper),
            "geometry_margin_lower": float(reference.Q - q_lower),
            "geometry_margin_upper": float(q_upper - reference.Q),
            "conditional_normalization": float(self.reference.normalization),
            "quadrature_method": "conditional_CDF_GH_GL",
            "solver_mode": "corrected_full_A",
            "boundary_q_scan_count": self.q_scan_count,
            "boundary_s_seed_count": self.s_seed_count,
            "full_eta_initial": self.full_eta_initial,
        }


def _legacy_raw_rows(config: dict[str, Any], radii: Sequence[float]) -> list[dict[str, Any]]:
    calc = FullRS(
        alpha=float(config.get("alpha", 0.5)),
        beta=float(config.get("beta", 1.0)),
        lambda_ref=float(config.get("lambda_ref", 1.0)),
        lambda_shell=float(config.get("lambda_shell", 1.0)),
        n0=int(config.get("n0", 13)),
        n1=int(config.get("n1", 13)),
        n3=int(config.get("n3", 19)),
    )
    rows: list[dict[str, Any]] = []
    for radius in radii:
        row = calc.solve_radius(
            float(radius),
            q_grid_count=int(config.get("q_grid_count", 45)),
            s_grid_count=int(config.get("s_grid_count", 31)),
            s_abs_max=float(config.get("s_abs_max", 0.25)),
        )
        row["solver_mode"] = "legacy_masked_grid_A0"
        rows.append(row)
    return rows


def _attach_derived_columns(
    raw_rows: Iterable[dict[str, Any]], alpha: float
) -> list[dict[str, Any]]:
    rows = list(raw_rows)
    if not rows:
        raise ValueError("At least one radius is required")
    base_phi = float(rows[0]["phi"])
    base_radius = float(rows[0]["r"])
    historical = ("r", "phi", "Q", "p", "t", "cd", "s", "qref")
    result: list[dict[str, Any]] = []
    for raw in rows:
        radius = float(raw["r"])
        phi = float(raw["phi"])
        row = {name: raw[name] for name in historical}
        row["phi_rel"] = float(phi - base_phi)
        row["phi_radius"] = float(math.log(radius))
        row["phi_radius_rel"] = float(math.log(radius / base_radius))
        row["phi_energy"] = float(phi - row["phi_radius"])
        row["phi_energy_rel"] = float(row["phi_rel"] - row["phi_radius_rel"])
        row["alpha"] = float(alpha)
        for key, value in raw.items():
            if key not in row:
                row[key] = value
        result.append(row)
    return result


def _validate_rows(
    rows: Sequence[dict[str, Any]],
    requested_radii: Sequence[float],
    *,
    corrected: bool,
) -> None:
    if len(rows) != len(requested_radii):
        raise RuntimeError("The solver did not return exactly one row per radius")
    actual = np.asarray([float(row["r"]) for row in rows], float)
    expected = np.asarray(requested_radii, float)
    if not np.array_equal(actual, expected):
        raise RuntimeError("Output radius ordering differs from the requested ordering")
    if np.unique(actual).size != actual.size:
        raise RuntimeError("Output radii must be unique")
    required = (
        "r",
        "phi",
        "Q",
        "p",
        "t",
        "cd",
        "s",
        "qref",
        "phi_rel",
        "phi_radius",
        "phi_radius_rel",
        "phi_energy",
        "phi_energy_rel",
        "alpha",
    )
    for row in rows:
        values = np.asarray([float(row[name]) for name in required], float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite core output at radius={row['r']}")
        if not math.isclose(
            float(row["phi_energy"]),
            float(row["phi"]) - math.log(float(row["r"])),
            rel_tol=0.0,
            abs_tol=2.0e-14,
        ):
            raise RuntimeError("phi_energy no longer equals phi - log(r)")
        if corrected:
            if not (0.0 <= float(row["eta"]) < 1.0):
                raise RuntimeError(f"Infeasible eta at radius={row['r']}")
            if float(row["A"]) < -1.0e-14:
                raise RuntimeError(f"Infeasible A at radius={row['r']}")
            corrected_values = np.asarray(
                [
                    float(row[name])
                    for name in (
                        "eta",
                        "A",
                        "G_S",
                        "G_E",
                        "gradient_max_abs",
                        "dF_dsqrtQ",
                        "dF_ds",
                        "dF_deta",
                        "confirmation_gradient_max_abs",
                        "conditional_normalization",
                    )
                ],
                float,
            )
            if not np.isfinite(corrected_values).all():
                raise RuntimeError(
                    f"Non-finite corrected diagnostic at radius={row['r']}"
                )
            if abs(float(row["conditional_normalization"]) - 1.0) > 5.0e-13:
                raise RuntimeError("Conditional quadrature normalization failed")
            if not bool(row["physical_curvature_signature"]):
                raise RuntimeError("Nonphysical saddle curvature in output")
    if abs(float(rows[0]["phi_rel"])) > 1.0e-14:
        raise RuntimeError("The first requested radius must define phi_rel=0")
    if abs(float(rows[0]["phi_energy_rel"])) > 1.0e-14:
        raise RuntimeError("The first requested radius must define phi_energy_rel=0")


def compute_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute one deterministic output row for each configured radius.

    A missing ``solver_mode`` retains the historical behavior for external
    legacy configs.  The promoted project config explicitly requests
    ``corrected_full_A``.
    """
    radii = [float(value) for value in config.get("radii", DEFAULT_RADII)]
    if not radii:
        raise ValueError("The radius list cannot be empty")
    mode = str(config.get("solver_mode", "legacy_masked_grid_A0")).lower()
    corrected_modes = {"corrected_full_a", "corrected_full", "full_a_constrained"}
    legacy_modes = {"legacy_masked_grid_a0", "legacy", "legacy_a0"}
    if mode in corrected_modes:
        if not bool(config.get("allow_unvalidated_domain", False)):
            validated_parameters = {
                "alpha": 0.1,
                "beta": 1.0,
                "lambda_ref": 1.0,
                "lambda_shell": 1.0,
            }
            changed = [
                name
                for name, expected in validated_parameters.items()
                if not math.isclose(
                    float(config.get(name, expected)),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-14,
                )
            ]
            off_grid = [
                radius
                for radius in radii
                if not any(
                    math.isclose(radius, allowed, rel_tol=0.0, abs_tol=1.0e-12)
                    for allowed in DEFAULT_RADII
                )
            ]
            if changed or off_grid:
                raise ValueError(
                    "The single-start corrected_full_A branch guard was validated "
                    "only for alpha=0.1, beta=lambda_ref=lambda_shell=1 and subsets "
                    "of the 42 production radii. Use a legacy-mode config or set "
                    "allow_unvalidated_domain=true only after an independent global "
                    f"branch check. Changed parameters={changed}, off-grid radii={off_grid}."
                )
        calc = CorrectedFullRS(config)
        raw_rows: list[dict[str, Any]] = []
        progress = bool(config.get("progress", True))
        for index, radius in enumerate(radii, start=1):
            row = calc.solve_radius(radius)
            raw_rows.append(row)
            if progress:
                print(
                    f"corrected-full-A {index:02d}/{len(radii):02d} "
                    f"r={radius:.2f} eta={float(row['eta']):.6g} "
                    f"A={float(row['A']):.6g} "
                    f"|grad_ref|={float(row['gradient_max_abs']):.2e}",
                    file=sys.stderr,
                    flush=True,
                )
        rows = _attach_derived_columns(raw_rows, float(config.get("alpha", 0.1)))
        _validate_rows(rows, radii, corrected=True)
        return rows
    if mode in legacy_modes:
        raw_rows = _legacy_raw_rows(config, radii)
        rows = _attach_derived_columns(raw_rows, float(config.get("alpha", 0.5)))
        _validate_rows(rows, radii, corrected=False)
        return rows
    raise ValueError(
        f"Unknown solver_mode={config.get('solver_mode')!r}; expected corrected_full_A or legacy_masked_grid_A0"
    )


def parse_radii(value: str | None) -> tuple[float, ...]:
    if value is None or not str(value).strip():
        return ()
    return tuple(float(x.strip()) for x in str(value).split(",") if x.strip())


def load_config(path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path is not None and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if args.alpha is not None:
        payload["alpha"] = args.alpha
    cli_radii = parse_radii(args.radii)
    if cli_radii:
        payload["radii"] = cli_radii
    return payload


def _atomic_write_csv(rows: Sequence[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--radii", type=str, default=None, help="Comma-separated radii override."
    )
    args = parser.parse_args()
    config = load_config(project_path(args.config), args)
    rows = compute_rows(config)
    output_value = config.get("output_csv", DEFAULT_OUTPUT_CSV)
    out = (
        project_path(args.out)
        if args.out is not None
        else project_path(Path(output_value))
    )
    _atomic_write_csv(rows, out)
    print(out)


if __name__ == "__main__":
    main()
