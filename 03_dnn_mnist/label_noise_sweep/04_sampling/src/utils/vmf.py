from __future__ import annotations

import numpy as np
from scipy.special import gammaln, ive


def log_sphere_mgf(dim: int, kappa: float) -> float:
    k = float(abs(kappa))
    if k == 0.0:
        return 0.0
    nu = float(dim) / 2.0 - 1.0
    scaled = ive(nu, k)
    if scaled <= 0.0 or not np.isfinite(scaled):
        return float((k * k) / (2.0 * float(dim)))
    return float(gammaln(float(dim) / 2.0) + nu * np.log(2.0 / k) + np.log(scaled) + k)


def _sample_vmf_wood(mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    dim = int(mu.size)
    if dim < 2:
        raise ValueError("vMF dimension must be at least 2")
    b = (-2.0 * float(kappa) + np.sqrt(4.0 * float(kappa) * float(kappa) + float(dim - 1) ** 2)) / float(dim - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = float(kappa) * x0 + float(dim - 1) * np.log(max(1.0 - x0 * x0, 1.0e-300))
    w = np.empty(int(n), dtype=np.float64)
    filled = 0
    alpha = 0.5 * float(dim - 1)
    while filled < int(n):
        draw = max(1024, int((int(n) - filled) * 1.25))
        z = rng.beta(alpha, alpha, size=draw)
        candidate = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        log_accept = float(kappa) * candidate + float(dim - 1) * np.log(np.maximum(1.0 - x0 * candidate, 1.0e-300)) - c
        accepted = candidate[np.log(rng.random(size=draw)) <= log_accept]
        take = min(accepted.size, int(n) - filled)
        if take:
            w[filled : filled + take] = accepted[:take]
            filled += take
    tangent = rng.normal(size=(int(n), dim))
    tangent -= (tangent @ mu)[:, None] * mu[None, :]
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    bad = tangent_norm[:, 0] <= 0.0
    while np.any(bad):
        tangent[bad] = rng.normal(size=(int(np.sum(bad)), dim))
        tangent[bad] -= (tangent[bad] @ mu)[:, None] * mu[None, :]
        tangent_norm[bad] = np.linalg.norm(tangent[bad], axis=1, keepdims=True)
        bad = tangent_norm[:, 0] <= 0.0
    tangent /= tangent_norm
    return w[:, None] * mu[None, :] + np.sqrt(np.maximum(1.0 - w * w, 0.0))[:, None] * tangent


def sample_vmf(mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(mu))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("vMF mean direction has zero or non-finite norm")
    mu = mu / norm
    if float(kappa) <= 1.0e-12:
        x = rng.normal(size=(int(n), mu.size))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x.astype(np.float64)
    try:
        return _sample_vmf_wood(mu, float(kappa), int(n), rng).astype(np.float64)
    except Exception:
        from scipy.stats import vonmises_fisher

        out = vonmises_fisher(mu=mu, kappa=float(kappa)).rvs(size=int(n), random_state=rng)
        return np.asarray(out, dtype=np.float64).reshape(int(n), mu.size)


def sample_vmf_batch(mus: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    mus = np.asarray(mus, dtype=np.float64)
    if mus.ndim != 2:
        raise ValueError(f"expected mus to be a matrix, got shape {mus.shape}")
    out = np.empty_like(mus, dtype=np.float64)
    for idx, mu in enumerate(mus):
        out[idx] = sample_vmf(mu, kappa, 1, rng)[0]
    return out

