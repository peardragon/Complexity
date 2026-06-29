from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


def mutual_knn_graph(X: np.ndarray, k: int) -> Tuple[List[Tuple[int, int]], float]:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if k < 1 or k >= n:
        raise ValueError("k_graph must satisfy 1 <= k < n")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    dists_k = dists[:, 1:]
    idx_k = idx[:, 1:]
    sigma_med = float(np.median(dists_k))
    knn_sets = [set(row.tolist()) for row in idx_k]
    edges_set = set()
    for i in range(n):
        for j in idx_k[i]:
            j = int(j)
            if i in knn_sets[j]:
                a, b = (i, j) if i < j else (j, i)
                edges_set.add((a, b))
    return sorted(edges_set), sigma_med


def rewire_graph(edges: Sequence[Tuple[int, int]], n_nodes: int, p_rewire: float, rng: np.random.Generator, *, mode: str) -> List[Tuple[int, int]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(n_nodes)))
    graph.add_edges_from([(int(i), int(j)) for i, j in edges])
    if mode == "degree_preserve":
        nswap = int(round(float(p_rewire) * graph.number_of_edges()))
        if nswap <= 0:
            return sorted((min(i, j), max(i, j)) for i, j in graph.edges())
        try:
            nx.double_edge_swap(graph, nswap=nswap, max_tries=max(10000, 10 * nswap), seed=rng)
        except Exception:
            pass
        return sorted((min(i, j), max(i, j)) for i, j in graph.edges())
    raise ValueError("Only degree_preserve rewiring is implemented")


def build_neighbors(n: int, edges: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
    adj = [[] for _ in range(int(n))]
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return [np.asarray(row, dtype=np.int32) for row in adj]


def prepare_graph_cache(X: np.ndarray, k_graph: int) -> Dict[str, object]:
    edges, sigma_med = mutual_knn_graph(X, k_graph)
    idx_i = np.asarray([i for i, _ in edges], dtype=np.int32)
    idx_j = np.asarray([j for _, j in edges], dtype=np.int32)
    diff = X[idx_i] - X[idx_j]
    dist2 = np.sum(diff * diff, axis=1)
    return {
        "edges": edges,
        "sigma_med": float(sigma_med),
        "idx_i": idx_i,
        "idx_j": idx_j,
        "dist2": dist2,
        "n_edges": int(len(edges)),
        "k_graph": int(k_graph),
    }


__all__ = ["build_neighbors", "mutual_knn_graph", "prepare_graph_cache", "rewire_graph"]
