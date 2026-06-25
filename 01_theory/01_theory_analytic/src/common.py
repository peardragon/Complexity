from __future__ import annotations

import numpy as np
from scipy.special import gammaln, ive, logsumexp


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
