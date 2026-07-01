from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .io_utils import now_iso
from .graphs import prepare_graph_cache, rewire_graph
from .ising import kawasaki_ising_sample_cached_fields


def normalize_features(X_raw: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    X_raw = np.asarray(X_raw, dtype=np.float64)
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    X_norm = (X_raw - mean[None, :]) / std[None, :]
    return X_norm.astype(np.float64), {
        "x_mean": mean.astype(np.float64).tolist(),
        "x_std": std.astype(np.float64).tolist(),
    }


def make_ws_ising_dataset(*, n_points: int, input_dim: int, k_graph: int, rewire_p: float, rewire_mode: str, beta_ising: float, ising_sweeps: int, seed: int, scales: Sequence[float]) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    X_raw = rng.uniform(low=-1.0, high=1.0, size=(int(n_points), int(input_dim))).astype(np.float64)
    graph_cache = prepare_graph_cache(X_raw, int(k_graph))
    edges_local = graph_cache["edges"]
    edges_int = rewire_graph(edges_local, int(n_points), float(rewire_p), rng, mode=rewire_mode)
    y = kawasaki_ising_sample_cached_fields(int(n_points), edges_int, float(beta_ising), rng, sweeps=int(ising_sweeps))
    X_train, norm_stats = normalize_features(X_raw)
    return {
        "X_raw": X_raw,
        "X_train": X_train,
        "y": y,
        "meta": {
            "seed": int(seed),
            "n_points": int(n_points),
            "input_dim": int(input_dim),
            "beta_ising": float(beta_ising),
            "rewire_p": float(rewire_p),
            "rewire_mode": str(rewire_mode),
            "ising_sweeps": int(ising_sweeps),
            "created_at": now_iso(),
            **norm_stats,
        },
    }


__all__ = ["make_ws_ising_dataset", "normalize_features"]


