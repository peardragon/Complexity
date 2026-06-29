from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from graphs import build_neighbors


def kawasaki_ising_sample_cached_fields(n: int, edges: Sequence[Tuple[int, int]], beta_ising: float, rng: np.random.Generator, *, sweeps: int) -> np.ndarray:
    y = np.ones(int(n), dtype=np.int8)
    y[n // 2 :] = -1
    rng.shuffle(y)
    neighbors = build_neighbors(int(n), edges)
    edge_i = np.asarray([int(i) for i, _ in edges], dtype=np.int32)
    edge_j = np.asarray([int(j) for _, j in edges], dtype=np.int32)
    has_edge = np.zeros((int(n), int(n)), dtype=bool)
    if edge_i.size > 0:
        has_edge[edge_i, edge_j] = True
        has_edge[edge_j, edge_i] = True
    local_field = np.zeros(int(n), dtype=np.float64)
    for i in range(int(n)):
        if neighbors[i].size > 0:
            local_field[i] = float(np.sum(y[neighbors[i]]))
    for _sweep in range(int(sweeps)):
        for _ in range(int(n)):
            i = int(rng.integers(0, int(n)))
            for _attempt in range(20):
                j = int(rng.integers(0, int(n)))
                if y[i] != y[j]:
                    break
            else:
                continue
            yi = float(y[i])
            yj = float(y[j])
            connected = bool(has_edge[i, j])
            s_i = local_field[i] - (yj if connected else 0.0)
            s_j = local_field[j] - (yi if connected else 0.0)
            delta_e = 2.0 * (yi * s_i + yj * s_j)
            if delta_e <= 0.0 or rng.uniform() < math.exp(-float(beta_ising) * delta_e):
                if neighbors[i].size > 0:
                    local_field[neighbors[i]] -= 2.0 * yi
                if neighbors[j].size > 0:
                    local_field[neighbors[j]] -= 2.0 * yj
                y[i] = -y[i]
                y[j] = -y[j]
    return y.astype(np.int8)


__all__ = ["kawasaki_ising_sample_cached_fields"]


