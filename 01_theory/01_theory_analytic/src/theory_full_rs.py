"""Replica-symmetric analytic curve with the sole corrected full-``A >= 0`` path.

The production path uses the exact geometric ``Q`` interval, a full
``s in (-1, 1)`` boundary search, conditional-CDF quadrature for the selected
reference, continuous saddle refinement, and a constrained ``eta`` coordinate
for ``A >= 0``.  Stationary-root and curvature admissibility are checked at
the ``(32, 32, 24, 48)`` confirmation tier; the reported action and analytic
gradient use the ``(96, 96, 64, 144)`` final tier and are independently
validated at ``(128, 128, 96, 192)``.

The production solver covers the full feasible ``A >= 0`` domain. Numerical
stationarity, curvature, and quadrature checks are part of the calculation.
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
from scipy.optimize import least_squares

# Keep file-based imports used by the appendix audit compatible: unlike normal
# script execution, importlib does not automatically add this file's directory
# to sys.path.
SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from utils.rs_refinement_core import (
    NUMBA_AVAILABLE,
    RSConfig,
    RSRefinementCore,
    SaddlePoint,
    q_geometric_bounds,
    q_interior_bounds,
)


DEFAULT_RADII = tuple(round(0.15 + 0.05 * idx, 10) for idx in range(42))
CONFIRMATION_ORDERS = (32, 32, 24, 48)
FINAL_REFERENCE_ORDERS = (96, 96, 64, 144)
VALIDATION_ORDERS = (128, 128, 96, 192)
QUADRATURE_CONVERGENCE_TOLERANCE = 5.0e-7
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "01_theory" / "01_theory_analytic" / "config" / "default.json"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "01_theory"
    / "01_theory_analytic"
    / "candidate_outputs"
    / "phi_by_analytic_solution_alpha0p1.csv"
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def _stationary_bounds(
    core: RSRefinementCore,
    radius: float,
    *,
    full: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically safe bounds for the complete open physical domain."""

    qlo, qhi = q_interior_bounds(
        radius, core.q_ref_norm, core.config.boundary_epsilon
    )
    epsilon = float(core.config.boundary_epsilon)
    lower = [math.sqrt(qlo), -1.0 + epsilon]
    upper = [math.sqrt(qhi), 1.0 - epsilon]
    if full:
        # eta=0 is audited separately as the exact A=0 face.  The interior
        # root search reaches the opposite physical boundary up to the same
        # numerical epsilon used for Q and s; no eta=.95 truncation remains.
        lower.append(max(1.0e-8, 0.1 * epsilon))
        upper.append(1.0 - epsilon)
    return np.asarray(lower, float), np.asarray(upper, float)


def _interior_difference_steps(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: np.ndarray,
) -> np.ndarray:
    """Choose central-difference steps that cannot leave physical bounds."""

    room = np.minimum(values - lower, upper - values)
    if np.any(~np.isfinite(room)) or np.any(room <= 1.0e-12):
        raise ValueError("stationary diagnostic point touches a physical bound")
    steps = np.minimum(np.asarray(nominal, float), 0.25 * room)
    if np.any(steps <= 0.0) or np.any(~np.isfinite(steps)):
        raise ValueError("could not construct an interior difference stencil")
    return steps


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
    nominal = [max(2.0e-5, 2.0e-5 * sqrt_q_width), 2.0e-5]
    if full:
        nominal.append(2.0e-5)
    lower, upper = _stationary_bounds(core, radius, full=full)
    steps = _interior_difference_steps(
        values,
        lower,
        upper,
        np.asarray(nominal, float),
    )
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
    nominal = np.asarray(
        [2.0e-4, 2.0e-4] + ([1.0e-4] if full else []),
        dtype=float,
    )
    lower, upper = _stationary_bounds(core, radius, full=full)
    steps = _interior_difference_steps(
        values,
        lower,
        upper,
        nominal,
    )
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
    lower, upper = _stationary_bounds(core, radius, full=full)
    seed_array = np.asarray(seed, float).copy()
    if seed_array.shape != lower.shape:
        raise ValueError("stationary seed dimension does not match solver mode")
    widths = upper - lower
    guards = np.maximum(1.0e-10, 1.0e-8 * widths)
    seed_array = np.minimum(
        np.maximum(seed_array, lower + guards),
        upper - guards,
    )
    result = least_squares(
        lambda vector: _numerical_gradient(core, radius, vector, full=full),
        seed_array,
        bounds=(lower, upper),
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


def validate_quadrature_contract(config: dict[str, Any]) -> None:
    """Fail closed unless the production and independent validation tiers match."""

    required_orders = {
        "confirmation_orders": CONFIRMATION_ORDERS,
        "reference_orders": FINAL_REFERENCE_ORDERS,
        "validation_orders": VALIDATION_ORDERS,
    }
    for key, expected in required_orders.items():
        if key not in config or _orders(config, key, expected) != expected:
            raise ValueError(f"{key} must equal {list(expected)}")
    if not math.isclose(
        float(config.get("quadrature_convergence_tolerance", float("nan"))),
        QUADRATURE_CONVERGENCE_TOLERANCE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "quadrature_convergence_tolerance must equal "
            f"{QUADRATURE_CONVERGENCE_TOLERANCE}"
        )
    for key in ("final_gradient_tolerance", "validation_gradient_tolerance"):
        value = float(config.get(key, float("nan")))
        if not math.isfinite(value) or value <= 0.0 or value > 5.0e-5:
            raise ValueError(f"{key} must lie in (0, 5e-5]")


class CorrectedFullRS:
    """Deterministic production solver for ``max_Q min_(s, eta) F``."""

    def __init__(self, config: dict[str, Any]) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "corrected_full_A requires Numba; the pure-Python nested "
                "quadrature fallback is not viable for a production sweep"
            )
        validate_quadrature_contract(config)
        self.config = dict(config)
        self.screen_orders = _orders(
            config, "boundary_screen_orders", (16, 16, 12, 24)
        )
        self.standard_orders = _orders(
            config, "standard_orders", (24, 24, 16, 36)
        )
        self.confirmation_orders = _orders(
            config, "confirmation_orders", CONFIRMATION_ORDERS
        )
        self.reference_orders = _orders(
            config, "reference_orders", FINAL_REFERENCE_ORDERS
        )
        self.validation_orders = _orders(
            config, "validation_orders", VALIDATION_ORDERS
        )
        self.screen = RSRefinementCore(_core_config(config, self.screen_orders))
        self.q_ref = self.screen.q_ref
        self.boundary_tiers: list[tuple[str, RSRefinementCore]] = []
        for prefix, orders in (
            ("standard", self.standard_orders),
            ("confirmation", self.confirmation_orders),
        ):
            self.boundary_tiers.append(
                (
                    _tier_name(prefix, orders),
                    RSRefinementCore(_core_config(config, orders), q_ref=self.q_ref),
                )
            )
        self.confirmation_name, self.confirmation = self.boundary_tiers[1]
        self.reference = RSRefinementCore(
            _core_config(config, self.reference_orders), q_ref=self.q_ref
        )
        self.reference_name = _tier_name("final", self.reference_orders)
        self.validation_name = _tier_name(
            "validation", self.validation_orders
        )
        self.validation = RSRefinementCore(
            _core_config(config, self.validation_orders), q_ref=self.q_ref
        )
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
        self.final_gradient_tolerance = float(
            config["final_gradient_tolerance"]
        )
        self.validation_gradient_tolerance = float(
            config["validation_gradient_tolerance"]
        )
        self.quadrature_convergence_tolerance = float(
            config["quadrature_convergence_tolerance"]
        )
        if self.q_scan_count < 3 or self.s_seed_count < 3:
            raise ValueError("Boundary Q and s seed counts must both be >= 3")
        eta_upper = 1.0 - float(self.screen.config.boundary_epsilon)
        if not 1.0e-8 < self.full_eta_initial < eta_upper:
            raise ValueError(
                "full_eta_initial must lie inside the complete physical "
                f"eta interval (1e-8, {eta_upper})"
            )
        for core in [
            self.screen,
            *(item[1] for item in self.boundary_tiers),
            self.validation,
        ]:
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

        # Stationary coordinates and curvature are selected at the
        # confirmation tier.  The reported action is evaluated at the fixed
        # final tier, and an independent higher tier must agree at the same
        # coordinates before the row can enter the staged curve.
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
            or reference_gradient_max > self.final_gradient_tolerance
        ):
            raise RuntimeError(
                f"Final-tier residual exceeds tolerance at radius={radius}"
            )
        validation = self.validation.evaluate(
            radius,
            confirmed.Q,
            confirmed.s,
            confirmed.eta,
            force_full=True,
        )
        validation_gradient = _numerical_gradient(
            self.validation, radius, reference_vector, full=True
        )
        validation_gradient_max = float(
            np.max(np.abs(validation_gradient))
        )
        if (
            not np.isfinite(validation_gradient_max)
            or validation_gradient_max > self.validation_gradient_tolerance
        ):
            raise RuntimeError(
                f"Validation-tier residual exceeds tolerance at radius={radius}"
            )
        quadrature_gap = abs(float(reference.phi) - float(validation.phi))
        if (
            not np.isfinite(quadrature_gap)
            or quadrature_gap > self.quadrature_convergence_tolerance
        ):
            raise RuntimeError(
                "Final/validation quadrature gap exceeds tolerance at "
                f"radius={radius}: {quadrature_gap}"
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
            "final_quadrature_orders": list(self.reference_orders),
            "gradient_max_abs": reference_gradient_max,
            "final_gradient_max_abs": reference_gradient_max,
            "dF_dsqrtQ": float(reference_gradient[0]),
            "dF_ds": float(reference_gradient[1]),
            "dF_deta": float(reference_gradient[2]),
            "validation_phi": float(validation.phi),
            "validation_quadrature_tier": self.validation_name,
            "validation_quadrature_orders": list(self.validation_orders),
            "validation_gradient_max_abs": validation_gradient_max,
            "validation_dF_dsqrtQ": float(validation_gradient[0]),
            "validation_dF_ds": float(validation_gradient[1]),
            "validation_dF_deta": float(validation_gradient[2]),
            "quadrature_convergence_abs": float(quadrature_gap),
            "quadrature_convergence_tolerance": float(
                self.quadrature_convergence_tolerance
            ),
            "quadrature_convergence_passed": True,
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
                        "final_gradient_max_abs",
                        "validation_phi",
                        "validation_gradient_max_abs",
                        "validation_dF_dsqrtQ",
                        "validation_dF_ds",
                        "validation_dF_deta",
                        "quadrature_convergence_abs",
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
            if not bool(row["quadrature_convergence_passed"]):
                raise RuntimeError("Quadrature convergence gate is not passing")
            if float(row["quadrature_convergence_abs"]) > float(
                row["quadrature_convergence_tolerance"]
            ):
                raise RuntimeError("Quadrature convergence tolerance exceeded")
    if abs(float(rows[0]["phi_rel"])) > 1.0e-14:
        raise RuntimeError("The first requested radius must define phi_rel=0")
    if abs(float(rows[0]["phi_energy_rel"])) > 1.0e-14:
        raise RuntimeError("The first requested radius must define phi_energy_rel=0")


def validate_production_config(config: dict[str, Any]) -> list[float]:
    """Validate the complete fixed production contract before heavy work."""

    radii = [float(value) for value in config.get("radii", DEFAULT_RADII)]
    if not radii:
        raise ValueError("The radius list cannot be empty")
    if "solver_mode" not in config:
        raise ValueError("solver_mode is required and must equal 'corrected_full_A'")
    if str(config["solver_mode"]) != "corrected_full_A":
        raise ValueError(
            f"unsupported solver_mode={config['solver_mode']!r}; "
            "the revised production path accepts only 'corrected_full_A'"
        )

    validated_parameters = {
        "alpha": 0.1,
        "beta": 1.0,
        "lambda_ref": 1.0,
        "lambda_shell": 1.0,
    }
    changed = [
        name
        for name, expected in validated_parameters.items()
        if name not in config
        or not math.isclose(
            float(config[name]), expected, rel_tol=0.0, abs_tol=1.0e-14
        )
    ]
    if changed:
        raise ValueError(
            "01_theory is frozen to the perceptron construction "
            "(alpha=0.1, beta=lambda_ref=lambda_shell=1); "
            f"invalid or missing keys={changed}"
        )
    if len(radii) != len(DEFAULT_RADII) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(radii, DEFAULT_RADII)
    ):
        raise ValueError("production analytic config must contain the canonical 42 radii")
    validate_quadrature_contract(config)
    return radii


def compute_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute one corrected full-A row for every configured radius.

    ``solver_mode`` is mandatory and has exactly one accepted value.  This
    fail-closed contract prevents a missing key or typo from selecting a
    historical approximation.
    """
    radii = validate_production_config(config)

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
                f"|grad_final|={float(row['gradient_max_abs']):.2e} "
                f"|phi_final-phi_validation|="
                f"{float(row['quadrature_convergence_abs']):.2e}",
                file=sys.stderr,
                flush=True,
            )
    rows = _attach_derived_columns(raw_rows, float(config["alpha"]))
    _validate_rows(rows, radii, corrected=True)
    return rows


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
    parser.add_argument("--force", action="store_true", help="Recompute an existing result file.")
    args = parser.parse_args()
    config = load_config(project_path(args.config), args)
    output_value = config.get("output_csv", DEFAULT_OUTPUT_CSV)
    out = (
        project_path(args.out)
        if args.out is not None
        else project_path(Path(output_value))
    )
    if out.is_file() and not args.force:
        print(f"{out}: skipped_existing")
        return
    rows = compute_rows(config)
    _atomic_write_csv(rows, out)
    print(out)


if __name__ == "__main__":
    main()
