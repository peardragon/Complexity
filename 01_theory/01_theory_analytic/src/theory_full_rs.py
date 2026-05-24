"""Standalone full-RS branch calculator for the theory stage."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr, roots_hermitenorm


def gh_norm(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = roots_hermitenorm(n)
    return x, w / np.sqrt(2.0 * np.pi)


def h_tail(x: np.ndarray) -> np.ndarray:
    return ndtr(-x)


def phi_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def solve_q_ref(alpha: float, gh_order: int = 80) -> float:
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
    t = q_ref * cd + s_value * np.sqrt(max(q_ref * (1.0 - q_ref) * (1.0 - cd * cd), 0.0))
    return float(p), float(t)


def gs_entropy(p: float, t: float, cd: float, q_ref: float) -> float:
    den = 2.0 * (1.0 - p) * (1.0 - q_ref) ** 2
    num = (1.0 - cd**2) * (1.0 - 2.0 * q_ref) + q_ref * q_ref - 2.0 * q_ref * cd * t + t * t
    return float(num / den + 0.5 * np.log(2.0 * np.pi) + 0.5 * np.log(1.0 - p))


class FullRS:
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
        self.w01 = self.w0[:, None] * self.w1[None, :]
        self.logw3 = np.log(self.w3)[None, None, :]

    def ge_energy(self, q_norm: float, s_value: float, cd: float) -> float:
        q_ref = self.q_ref
        p, t = convert_qs(q_norm, s_value, cd, q_ref)
        if p >= 1.0 or p < -1.0:
            return float("nan")
        uref = np.sqrt(q_ref) * self.z0_grid + np.sqrt(1.0 - q_ref) * self.z1_grid
        mask = (uref[:, :, 0] > 0).astype(float)
        denom = np.maximum(h_tail(-np.sqrt(q_ref / (1.0 - q_ref)) * self.z0), 1.0e-300)
        base = (t / np.sqrt(q_ref)) * self.z0_grid + ((cd - t) / np.sqrt(1.0 - q_ref)) * self.z1_grid
        h = np.sqrt(q_norm) * (base + np.sqrt(max(1.0 - p, 1.0e-14)) * self.z3_grid)
        logs = self.logw3 - self.beta * np.logaddexp(0.0, -h)
        m = np.max(logs, axis=2, keepdims=True)
        inner = np.squeeze(m, 2) + np.log(np.sum(np.exp(logs - m), axis=2))
        sums = np.sum(self.w01 * mask * inner, axis=1) / denom
        return float(np.sum(self.w0 * sums))

    def action(self, q_norm: float, s_value: float, radius: float) -> float:
        q_ref_norm = 1.0 / self.lambda_ref
        if q_norm <= 0.0:
            return float("nan")
        cd = (q_norm + q_ref_norm - radius * radius) / (2.0 * np.sqrt(q_norm * q_ref_norm))
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

    def solve_radius(self, radius: float, *, q_grid_count: int, s_grid_count: int, s_abs_max: float) -> dict[str, float]:
        q_ref_norm = 1.0 / self.lambda_ref
        lo = max(0.0, np.sqrt(q_ref_norm) - radius) ** 2 + 1.0e-4
        hi = (np.sqrt(q_ref_norm) + radius) ** 2 - 1.0e-4
        q_grid = np.linspace(lo, hi, int(q_grid_count))
        s_grid = np.linspace(-float(s_abs_max), float(s_abs_max), int(s_grid_count))
        best: tuple[float, float, float] | None = None
        for q_norm in q_grid:
            vals = np.asarray([self.action(float(q_norm), float(s), radius) for s in s_grid])
            if not np.any(np.isfinite(vals)):
                continue
            idx = int(np.nanargmin(vals))
            val = float(vals[idx])
            if best is None or val > best[0]:
                best = (val, float(q_norm), float(s_grid[idx]))
        if best is None:
            raise RuntimeError(f"No finite full-RS branch for radius={radius}")
        val, q_norm, s_value = best
        cd = (q_norm + q_ref_norm - radius * radius) / (2.0 * np.sqrt(q_norm * q_ref_norm))
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


def compute_rows(config: dict) -> list[dict[str, float]]:
    calc = FullRS(
        alpha=float(config.get("alpha", 0.5)),
        beta=float(config.get("beta", 1.0)),
        lambda_ref=float(config.get("lambda_ref", 1.0)),
        lambda_shell=float(config.get("lambda_shell", 1.0)),
        n0=int(config.get("n0", 13)),
        n1=int(config.get("n1", 13)),
        n3=int(config.get("n3", 19)),
    )
    radii = [float(value) for value in config.get("radii", [0.15, 0.25, 0.40, 0.60, 0.85, 1.10, 1.35, 1.60, 1.90, 2.20])]
    rows = [
        calc.solve_radius(
            radius,
            q_grid_count=int(config.get("q_grid_count", 45)),
            s_grid_count=int(config.get("s_grid_count", 31)),
            s_abs_max=float(config.get("s_abs_max", 0.25)),
        )
        for radius in radii
    ]
    base = float(rows[0]["phi"])
    for row in rows:
        row["phi_rel"] = float(row["phi"] - base)
        row["alpha"] = float(config.get("alpha", 0.5))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    rows = compute_rows({"alpha": args.alpha})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
