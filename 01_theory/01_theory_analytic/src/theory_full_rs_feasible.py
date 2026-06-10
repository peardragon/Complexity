"""Full feasible-domain RS shell solver with A >= 0.

This module leaves ``theory_full_rs.py`` as the retained A=0 baseline and
adds the eta direction for the full feasible covariance cone. The energetic
quadrature intentionally extends the legacy normalization convention from the
baseline solver so that eta=0 remains directly comparable to the retained CSV.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from theory_full_rs import gh_norm, gs_entropy, h_tail, solve_q_ref

try:  # pragma: no cover - exercised by runtime availability.
    with contextlib.redirect_stderr(io.StringIO()):
        from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]


DEFAULT_BASELINE_CSV = Path("01_theory/01_theory_analytic/raw_outputs/theory_full_rs_alpha0p1.csv")
DEFAULT_RADII = tuple(round(0.15 + 0.05 * idx, 10) for idx in range(42))
EPS = 1.0e-12


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def cd_from_Q(Q: float, radius: float, Qref: float) -> float:
    return float((Q + Qref - radius * radius) / (2.0 * math.sqrt(Q * Qref)))


def q_bounds_for_radius(radius: float, Qref: float, eps: float = 1.0e-4) -> tuple[float, float]:
    lo = max(0.0, math.sqrt(Qref) - float(radius)) ** 2 + eps
    hi = (math.sqrt(Qref) + float(radius)) ** 2 - eps
    if not (lo < hi):
        raise ValueError(f"empty Q interval for radius={radius}, Qref={Qref}: {lo} >= {hi}")
    return lo, hi


def convert_q_s_eta(
    *,
    Q: float,
    radius: float,
    q_ref: float,
    lambda_ref: float,
    s: float,
    eta: float,
) -> dict[str, float]:
    """Feasible parametrization of (p, t, A)."""
    Qref = 1.0 / float(lambda_ref)
    cd = cd_from_Q(Q, radius, Qref)
    if not (-1.0 < cd < 1.0):
        return {"valid": 0.0, "cd": cd}

    q = float(q_ref)
    one_minus_c2 = max(1.0 - cd * cd, 0.0)
    s = float(np.clip(s, -1.0, 1.0))
    eta = float(np.clip(eta, 0.0, 1.0 - 1.0e-10))

    t = q * cd + s * math.sqrt(max(q * (1.0 - q) * one_minus_c2, 0.0))
    p_min = cd * cd + s * s * one_minus_c2
    if p_min >= 1.0:
        return {"valid": 0.0, "cd": cd, "t": t, "p_min": p_min}

    p = p_min + eta * (1.0 - p_min)
    A = eta * (1.0 - p_min)
    one_minus_p = 1.0 - p
    if A < -1.0e-10 or one_minus_p <= 0.0 or p >= 1.0:
        return {"valid": 0.0, "cd": cd, "t": t, "p_min": p_min, "p": p, "A": A}

    return {
        "valid": 1.0,
        "cd": float(cd),
        "t": float(t),
        "p_min": float(p_min),
        "p": float(p),
        "A": float(max(A, 0.0)),
        "one_minus_p": float(one_minus_p),
        "s": float(s),
        "eta": float(eta),
    }


@dataclass(frozen=True)
class SolverConfig:
    alpha: float = 0.1
    beta: float = 1.0
    lambda_ref: float = 1.0
    lambda_shell: float = 1.0
    n0: int = 13
    n1: int = 13
    n2: int = 13
    n3: int = 19
    q_grid_count: int = 45
    s_grid_count: int = 41
    eta_grid_count: int = 21
    eta_max: float = 0.98
    radii: tuple[float, ...] = DEFAULT_RADII
    backend: str = "auto"


if NUMBA_AVAILABLE:

    @njit
    def _softplus_neg(h: float) -> float:
        if h >= 0.0:
            return math.log1p(math.exp(-h))
        return -h + math.log1p(math.exp(h))


    @njit
    def _gs_entropy_numba(p: float, t: float, cd: float, q_ref: float) -> float:
        den = 2.0 * (1.0 - p) * (1.0 - q_ref) ** 2
        if den <= 0.0:
            return math.nan
        num = (1.0 - cd * cd) * (1.0 - 2.0 * q_ref) + q_ref * q_ref - 2.0 * q_ref * cd * t + t * t
        return num / den + 0.5 * math.log(2.0 * math.pi) + 0.5 * math.log(1.0 - p)


    @njit
    def _ge_energy_legacy_numba(
        Q: float,
        p: float,
        t: float,
        cd: float,
        A: float,
        beta: float,
        q_ref: float,
        z0: np.ndarray,
        w0: np.ndarray,
        z1: np.ndarray,
        w1: np.ndarray,
        z2: np.ndarray,
        w2: np.ndarray,
        z3: np.ndarray,
        logw3: np.ndarray,
        mask: np.ndarray,
        denom: np.ndarray,
    ) -> float:
        if p >= 1.0 or A < -1.0e-10:
            return math.nan

        sqrt_q = math.sqrt(q_ref)
        sqrt_1mq = math.sqrt(1.0 - q_ref)
        sqrt_Q = math.sqrt(Q)
        sqrt_A = math.sqrt(max(A, 0.0))
        sqrt_1mp = math.sqrt(max(1.0 - p, 1.0e-14))
        total = 0.0

        for i0 in range(z0.shape[0]):
            denom_i = max(denom[i0], 1.0e-300)
            z0_i = z0[i0]
            sum_z1 = 0.0
            for i1 in range(z1.shape[0]):
                if mask[i0, i1] <= 0.0:
                    continue
                base = (t / sqrt_q) * z0_i + ((cd - t) / sqrt_1mq) * z1[i1]
                sum_z2 = 0.0
                for i2 in range(z2.shape[0]):
                    shifted = sqrt_Q * (base + sqrt_A * z2[i2])
                    max_log = -1.0e300
                    for i3 in range(z3.shape[0]):
                        h = shifted + sqrt_Q * sqrt_1mp * z3[i3]
                        val = logw3[i3] - beta * _softplus_neg(h)
                        if val > max_log:
                            max_log = val

                    exp_sum = 0.0
                    for i3 in range(z3.shape[0]):
                        h = shifted + sqrt_Q * sqrt_1mp * z3[i3]
                        val = logw3[i3] - beta * _softplus_neg(h)
                        exp_sum += math.exp(val - max_log)
                    sum_z2 += w2[i2] * (max_log + math.log(exp_sum))
                sum_z1 += w1[i1] * sum_z2

            # Legacy baseline convention: keep the extra w0 factor used by
            # theory_full_rs.py so eta=0 reproduces retained outputs.
            total += w0[i0] * (w0[i0] * sum_z1 / denom_i)
        return total


    @njit(parallel=True)
    def _scan_radius_numba(
        radius: float,
        q_ref_norm: float,
        q_ref: float,
        alpha: float,
        beta: float,
        lambda_shell: float,
        q_grid: np.ndarray,
        s_grid: np.ndarray,
        eta_grid: np.ndarray,
        z0: np.ndarray,
        w0: np.ndarray,
        z1: np.ndarray,
        w1: np.ndarray,
        z2: np.ndarray,
        w2: np.ndarray,
        z3: np.ndarray,
        logw3: np.ndarray,
        mask: np.ndarray,
        denom: np.ndarray,
    ) -> np.ndarray:
        nq = q_grid.shape[0]
        per_q = np.empty((nq, 3, 11), dtype=np.float64)
        for iq in prange(nq):
            for branch_idx in range(3):
                for col in range(11):
                    per_q[iq, branch_idx, col] = math.nan
                per_q[iq, branch_idx, 0] = 1.0e300 if branch_idx < 2 else -1.0e300

            Q = q_grid[iq]
            if Q <= 0.0:
                continue
            cd = (Q + q_ref_norm - radius * radius) / (2.0 * math.sqrt(Q * q_ref_norm))
            if not (-1.0 < cd < 1.0):
                continue

            one_minus_c2 = max(1.0 - cd * cd, 0.0)
            for is_idx in range(s_grid.shape[0]):
                s = s_grid[is_idx]
                t = q_ref * cd + s * math.sqrt(max(q_ref * (1.0 - q_ref) * one_minus_c2, 0.0))
                p_min = cd * cd + s * s * one_minus_c2
                if p_min >= 1.0:
                    continue

                for ie in range(eta_grid.shape[0]):
                    eta = eta_grid[ie]
                    p = p_min + eta * (1.0 - p_min)
                    A = eta * (1.0 - p_min)
                    if A < -1.0e-10 or p >= 1.0 or 1.0 - p <= 0.0:
                        continue

                    gs = _gs_entropy_numba(p, t, cd, q_ref)
                    if not math.isfinite(gs):
                        continue
                    ge = _ge_energy_legacy_numba(
                        Q,
                        p,
                        t,
                        cd,
                        A,
                        beta,
                        q_ref,
                        z0,
                        w0,
                        z1,
                        w1,
                        z2,
                        w2,
                        z3,
                        logw3,
                        mask,
                        denom,
                    )
                    if not math.isfinite(ge):
                        continue

                    F = 0.5 * math.log(Q) - 0.5 * beta * lambda_shell * Q + gs + alpha * ge

                    if ie == 0 and F < per_q[iq, 0, 0]:
                        per_q[iq, 0, 0] = F
                        per_q[iq, 0, 1] = Q
                        per_q[iq, 0, 2] = s
                        per_q[iq, 0, 3] = eta
                        per_q[iq, 0, 4] = A
                        per_q[iq, 0, 5] = p
                        per_q[iq, 0, 6] = t
                        per_q[iq, 0, 7] = cd
                        per_q[iq, 0, 8] = gs
                        per_q[iq, 0, 9] = ge
                        per_q[iq, 0, 10] = p_min

                    if F < per_q[iq, 1, 0]:
                        per_q[iq, 1, 0] = F
                        per_q[iq, 1, 1] = Q
                        per_q[iq, 1, 2] = s
                        per_q[iq, 1, 3] = eta
                        per_q[iq, 1, 4] = A
                        per_q[iq, 1, 5] = p
                        per_q[iq, 1, 6] = t
                        per_q[iq, 1, 7] = cd
                        per_q[iq, 1, 8] = gs
                        per_q[iq, 1, 9] = ge
                        per_q[iq, 1, 10] = p_min

                    if F > per_q[iq, 2, 0]:
                        per_q[iq, 2, 0] = F
                        per_q[iq, 2, 1] = Q
                        per_q[iq, 2, 2] = s
                        per_q[iq, 2, 3] = eta
                        per_q[iq, 2, 4] = A
                        per_q[iq, 2, 5] = p
                        per_q[iq, 2, 6] = t
                        per_q[iq, 2, 7] = cd
                        per_q[iq, 2, 8] = gs
                        per_q[iq, 2, 9] = ge
                        per_q[iq, 2, 10] = p_min
        return per_q


class FullRSFeasible:
    def __init__(self, cfg: SolverConfig) -> None:
        self.cfg = cfg
        self.alpha = float(cfg.alpha)
        self.beta = float(cfg.beta)
        self.lambda_ref = float(cfg.lambda_ref)
        self.lambda_shell = float(cfg.lambda_shell)
        self.q_ref = solve_q_ref(self.alpha)
        if not (0.0 < self.q_ref < 1.0):
            raise ValueError(f"q_ref must be in (0,1), got {self.q_ref}")

        self.z0, self.w0 = gh_norm(int(cfg.n0))
        self.z1, self.w1 = gh_norm(int(cfg.n1))
        self.z2, self.w2 = gh_norm(int(cfg.n2))
        self.z3, self.w3 = gh_norm(int(cfg.n3))
        self.logw3 = np.log(self.w3)

        uref = math.sqrt(self.q_ref) * self.z0[:, None] + math.sqrt(1.0 - self.q_ref) * self.z1[None, :]
        self.mask = (uref > 0.0).astype(np.float64)
        self.denom = np.maximum(h_tail(-math.sqrt(self.q_ref / (1.0 - self.q_ref)) * self.z0), 1.0e-300)

        self.use_numba = self._resolve_backend(cfg.backend)

    def _resolve_backend(self, backend: str) -> bool:
        backend = str(backend).lower()
        if backend == "numba":
            if not NUMBA_AVAILABLE:
                raise RuntimeError("backend='numba' requested, but numba is not available")
            return True
        if backend == "numpy":
            return False
        if backend == "auto":
            return bool(NUMBA_AVAILABLE)
        raise ValueError(f"unknown backend: {backend}")

    def ge_energy_numpy(self, *, Q: float, p: float, t: float, cd: float, A: float) -> float:
        q = self.q_ref
        if p >= 1.0 or A < -1.0e-10:
            return float("nan")

        z0_grid = self.z0[:, None, None, None]
        z1_grid = self.z1[None, :, None, None]
        z2_grid = self.z2[None, None, :, None]
        z3_grid = self.z3[None, None, None, :]
        base = (t / math.sqrt(q)) * z0_grid + ((cd - t) / math.sqrt(1.0 - q)) * z1_grid
        g = base + math.sqrt(max(A, 0.0)) * z2_grid + math.sqrt(max(1.0 - p, 1.0e-14)) * z3_grid
        h = math.sqrt(Q) * g
        logs = self.logw3[None, None, None, :] - self.beta * np.logaddexp(0.0, -h)
        max_logs = np.max(logs, axis=3, keepdims=True)
        inner_z3 = np.squeeze(max_logs, axis=3) + np.log(np.sum(np.exp(logs - max_logs), axis=3))
        inner_z2 = np.sum(self.w2[None, None, :] * inner_z3, axis=2)
        sums = np.sum((self.w0[:, None] * self.w1[None, :]) * self.mask * inner_z2, axis=1) / self.denom
        return float(np.sum(self.w0 * sums))

    def action(self, *, Q: float, radius: float, s: float, eta: float) -> dict[str, float]:
        conv = convert_q_s_eta(
            Q=Q,
            radius=radius,
            q_ref=self.q_ref,
            lambda_ref=self.lambda_ref,
            s=s,
            eta=eta,
        )
        if conv.get("valid", 0.0) <= 0.0:
            return {"valid": 0.0, "F": float("nan")}

        cd = conv["cd"]
        p = conv["p"]
        t = conv["t"]
        A = conv["A"]
        gs = gs_entropy(p, t, cd, self.q_ref)
        ge = self.ge_energy_numpy(Q=Q, p=p, t=t, cd=cd, A=A)
        if not np.isfinite(gs) or not np.isfinite(ge):
            return {"valid": 0.0, "F": float("nan"), **conv}

        F = 0.5 * math.log(Q) - 0.5 * self.beta * self.lambda_shell * Q + gs + self.alpha * ge
        return {
            "valid": 1.0,
            "F": float(F),
            "G_S": float(gs),
            "G_E": float(ge),
            "Q": float(Q),
            "r": float(radius),
            **conv,
        }

    def _solve_radius_all_numba(self, radius: float) -> list[dict[str, float | str]]:
        Qref = 1.0 / self.lambda_ref
        lo, hi = q_bounds_for_radius(radius, Qref)
        q_grid = np.linspace(lo, hi, int(self.cfg.q_grid_count))
        s_grid = np.linspace(-1.0, 1.0, int(self.cfg.s_grid_count))
        eta_grid = np.linspace(0.0, float(self.cfg.eta_max), int(self.cfg.eta_grid_count))
        per_q = _scan_radius_numba(
            float(radius),
            float(Qref),
            float(self.q_ref),
            float(self.alpha),
            float(self.beta),
            float(self.lambda_shell),
            q_grid.astype(np.float64),
            s_grid.astype(np.float64),
            eta_grid.astype(np.float64),
            self.z0.astype(np.float64),
            self.w0.astype(np.float64),
            self.z1.astype(np.float64),
            self.w1.astype(np.float64),
            self.z2.astype(np.float64),
            self.w2.astype(np.float64),
            self.z3.astype(np.float64),
            self.logw3.astype(np.float64),
            self.mask.astype(np.float64),
            self.denom.astype(np.float64),
        )

        specs = [
            ("boundary_mixed_eta0", 0, "max"),
            ("full_mixed_maxQ_min_s_eta", 1, "max"),
            ("full_max_envelope", 2, "max"),
        ]
        rows: list[dict[str, float | str]] = []
        for branch, branch_idx, selector in specs:
            values = per_q[:, branch_idx, 0]
            finite = np.isfinite(values)
            if branch_idx < 2:
                finite &= values < 1.0e200
            else:
                finite &= values > -1.0e200
            if not np.any(finite):
                raise RuntimeError(f"No {branch} solution at radius={radius}")
            idxs = np.flatnonzero(finite)
            best_i = idxs[int(np.argmax(values[finite]))] if selector == "max" else idxs[int(np.argmin(values[finite]))]
            rows.append(self._row_from_solution(branch, radius, per_q[best_i, branch_idx, :]))
        return rows

    def _solve_radius_all_numpy(self, radius: float) -> list[dict[str, float | str]]:
        Qref = 1.0 / self.lambda_ref
        lo, hi = q_bounds_for_radius(radius, Qref)
        q_grid = np.linspace(lo, hi, int(self.cfg.q_grid_count))
        s_grid = np.linspace(-1.0, 1.0, int(self.cfg.s_grid_count))
        eta_grid = np.linspace(0.0, float(self.cfg.eta_max), int(self.cfg.eta_grid_count))

        boundary_best: dict[str, float] | None = None
        full_mixed_best: dict[str, float] | None = None
        full_max_best: dict[str, float] | None = None
        for Q in q_grid:
            boundary_local: dict[str, float] | None = None
            full_local: dict[str, float] | None = None
            for s in s_grid:
                for eta in eta_grid:
                    row = self.action(Q=float(Q), radius=radius, s=float(s), eta=float(eta))
                    if row.get("valid", 0.0) <= 0.0 or not np.isfinite(row["F"]):
                        continue
                    if eta == eta_grid[0] and (boundary_local is None or row["F"] < boundary_local["F"]):
                        boundary_local = dict(row)
                    if full_local is None or row["F"] < full_local["F"]:
                        full_local = dict(row)
                    if full_max_best is None or row["F"] > full_max_best["F"]:
                        full_max_best = dict(row)
            if boundary_local is not None and (boundary_best is None or boundary_local["F"] > boundary_best["F"]):
                boundary_best = dict(boundary_local)
            if full_local is not None and (full_mixed_best is None or full_local["F"] > full_mixed_best["F"]):
                full_mixed_best = dict(full_local)

        out = []
        for branch, row in (
            ("boundary_mixed_eta0", boundary_best),
            ("full_mixed_maxQ_min_s_eta", full_mixed_best),
            ("full_max_envelope", full_max_best),
        ):
            if row is None:
                raise RuntimeError(f"No {branch} solution at radius={radius}")
            row = dict(row)
            row["phi"] = float(row["F"])
            row["branch"] = branch
            out.append(row)
        return out

    def _row_from_solution(self, branch: str, radius: float, sol: np.ndarray) -> dict[str, float | str]:
        return {
            "branch": branch,
            "r": float(radius),
            "phi": float(sol[0]),
            "Q": float(sol[1]),
            "s": float(sol[2]),
            "eta": float(sol[3]),
            "A": float(sol[4]),
            "p": float(sol[5]),
            "t": float(sol[6]),
            "cd": float(sol[7]),
            "G_S": float(sol[8]),
            "G_E": float(sol[9]),
            "p_min": float(sol[10]),
        }

    def solve_radius_all(self, radius: float) -> list[dict[str, float | str]]:
        if self.use_numba:
            return self._solve_radius_all_numba(radius)
        return self._solve_radius_all_numpy(radius)

    def compute_rows(self, *, progress: bool = True) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        radii = [float(x) for x in self.cfg.radii]
        start = time.perf_counter()
        for idx, radius in enumerate(radii, start=1):
            if progress:
                elapsed = time.perf_counter() - start
                print(f"[{idx}/{len(radii)}] r={radius:.6g} elapsed={elapsed:.1f}s", flush=True)
            rows.extend(self.solve_radius_all(float(radius)))

        df = pd.DataFrame(rows)
        df["qref"] = self.q_ref
        df["alpha"] = self.alpha
        df["beta"] = self.beta
        df["lambda_ref"] = self.lambda_ref
        df["lambda_shell"] = self.lambda_shell
        df["phi_rel"] = np.nan
        for branch, idxs in df.groupby("branch").groups.items():
            sub = df.loc[list(idxs)].sort_values("r")
            base = float(sub.iloc[0]["phi"])
            df.loc[sub.index, "phi_rel"] = sub["phi"].astype(float) - base

        columns = [
            "branch",
            "r",
            "phi",
            "phi_rel",
            "Q",
            "s",
            "eta",
            "A",
            "p",
            "t",
            "cd",
            "G_S",
            "G_E",
            "qref",
            "alpha",
            "beta",
            "lambda_ref",
            "lambda_shell",
        ]
        return df[columns].sort_values(["branch", "r"]).reset_index(drop=True)


def parse_radii(value: str | None) -> tuple[float, ...]:
    if value is None or not str(value).strip():
        return ()
    return tuple(float(x.strip()) for x in str(value).split(",") if x.strip())


def read_radii_from_csv(path: Path) -> tuple[float, ...]:
    if not path.exists():
        return ()
    df = pd.read_csv(path)
    if "r" not in df.columns:
        return ()
    radii = sorted({finite_float(x) for x in df["r"]})
    return tuple(float(x) for x in radii if np.isfinite(x))


def load_cfg(path: Path | None, args: argparse.Namespace) -> SolverConfig:
    payload: dict[str, Any] = {}
    if path is not None and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))

    cli_radii = parse_radii(args.radii)
    if cli_radii:
        radii = cli_radii
    elif "radii" in payload:
        radii = tuple(float(x) for x in payload["radii"])
    else:
        radii = read_radii_from_csv(args.radii_from_csv) or DEFAULT_RADII

    return SolverConfig(
        alpha=float(args.alpha if args.alpha is not None else payload.get("alpha", 0.1)),
        beta=float(args.beta if args.beta is not None else payload.get("beta", 1.0)),
        lambda_ref=float(args.lambda_ref if args.lambda_ref is not None else payload.get("lambda_ref", 1.0)),
        lambda_shell=float(args.lambda_shell if args.lambda_shell is not None else payload.get("lambda_shell", 1.0)),
        n0=int(args.n0 if args.n0 is not None else payload.get("n0", 13)),
        n1=int(args.n1 if args.n1 is not None else payload.get("n1", 13)),
        n2=int(args.n2 if args.n2 is not None else payload.get("n2", 13)),
        n3=int(args.n3 if args.n3 is not None else payload.get("n3", 19)),
        q_grid_count=int(args.q_grid_count if args.q_grid_count is not None else payload.get("q_grid_count", 45)),
        s_grid_count=int(args.s_grid_count if args.s_grid_count is not None else payload.get("s_grid_count", 41)),
        eta_grid_count=int(args.eta_grid_count if args.eta_grid_count is not None else payload.get("eta_grid_count", 21)),
        eta_max=float(args.eta_max if args.eta_max is not None else payload.get("eta_max", 0.98)),
        radii=radii,
        backend=str(args.backend if args.backend is not None else payload.get("backend", "auto")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full feasible-domain RS shell solver with A>=0.")
    parser.add_argument("--config", type=Path, default=Path("01_theory/01_theory_analytic/config/default.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lambda-ref", type=float, default=None)
    parser.add_argument("--lambda-shell", type=float, default=None)
    parser.add_argument("--radii", type=str, default=None, help="Comma-separated radii override.")
    parser.add_argument("--radii-from-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--n0", type=int, default=None)
    parser.add_argument("--n1", type=int, default=None)
    parser.add_argument("--n2", type=int, default=None)
    parser.add_argument("--n3", type=int, default=None)
    parser.add_argument("--q-grid-count", type=int, default=None)
    parser.add_argument("--s-grid-count", type=int, default=None)
    parser.add_argument("--eta-grid-count", type=int, default=None)
    parser.add_argument("--eta-max", type=float, default=None)
    parser.add_argument("--backend", choices=("auto", "numba", "numpy"), default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config, args)
    solver = FullRSFeasible(cfg)
    print(
        json.dumps(
            {
                "alpha": cfg.alpha,
                "backend": "numba" if solver.use_numba else "numpy",
                "eta_grid_count": cfg.eta_grid_count,
                "eta_max": cfg.eta_max,
                "q_grid_count": cfg.q_grid_count,
                "radius_count": len(cfg.radii),
                "s_grid_count": cfg.s_grid_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    df = solver.compute_rows(progress=not args.no_progress)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(args.out)
    print(df[["branch", "r", "phi_rel", "Q", "s", "eta", "A", "p", "t", "cd"]].to_string(index=False))


if __name__ == "__main__":
    main()
