from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def nmstv_from_graph_cache(y: np.ndarray, graph_cache: Dict[str, Any], scales: Sequence[float]) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    idx_i = np.asarray(graph_cache["idx_i"], dtype=np.int32)
    idx_j = np.asarray(graph_cache["idx_j"], dtype=np.int32)
    dist2 = np.asarray(graph_cache["dist2"], dtype=np.float64)
    sigma_med = float(graph_cache["sigma_med"])
    yprod = y[idx_i].astype(np.float64) * y[idx_j].astype(np.float64)
    Cs = []
    rhos = []
    for scale in scales:
        denom = 2.0 * (float(scale) * sigma_med) ** 2 + 1e-300
        weights = np.exp(-dist2 / denom)
        Z = float(np.sum(weights)) + 1e-300
        rho = float(np.sum(weights * yprod) / Z)
        C_s = 0.5 * (1.0 - rho)
        Cs.append(float(C_s))
        rhos.append(float(rho))
    return {
        "C": float(np.mean(Cs)),
        "C_s": Cs,
        "rho_s": rhos,
        "sigma_med": float(sigma_med),
        "n_edges": int(graph_cache["n_edges"]),
        "k_graph": int(graph_cache["k_graph"]),
        "scales": [float(scale) for scale in scales],
    }


__all__ = ["nmstv_from_graph_cache"]
