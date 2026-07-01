from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


ORIGINAL_MNIST_DIRNAME = "original_mnist"
ORIGINAL_MNIST_FILENAME = "mnist_openml_uint8.npz"
ORIGINAL_MNIST_MANIFEST = "source_manifest.json"

INPUT_SIDE = 10
INPUT_DIM = INPUT_SIDE * INPUT_SIDE
DEFAULT_N_TRAIN = 512
DEFAULT_N_TEST = 2048
DEFAULT_SPLIT_SEED = 20260610
TEACHER_SEED = 31001
RANDOM_LABEL_TRAIN_SEED = 41001
RANDOM_LABEL_TEST_SEED = 42001

VERY_LOW_TV_RULE = "very_low_tv_spectral_teacher"
VERY_LOW_TV_K_VALUES = [8, 16, 32]
VERY_LOW_TV_SPECTRAL_KS = [3, 4, 6, 8, 12]
VERY_LOW_TV_MAX_DRAWS_PER_BASIS = 25000
VERY_LOW_TV_RNG_SEED = 20260618


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_npz_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def original_mnist_dir(raw_root: Path) -> Path:
    return raw_root / ORIGINAL_MNIST_DIRNAME


def original_mnist_path(raw_root: Path) -> Path:
    return original_mnist_dir(raw_root) / ORIGINAL_MNIST_FILENAME


def _load_mnist_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path) as payload:
        if "X" not in payload.files or "y" not in payload.files:
            raise KeyError(f"{cache_path} must contain X and y arrays")
        x = np.asarray(payload["X"], dtype=np.uint8).reshape(-1, 784)
        y = np.asarray(payload["y"], dtype=np.int16).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{cache_path} has mismatched X/y rows: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[1] != 784:
        raise ValueError(f"{cache_path} has X shape {x.shape}; expected (*, 784)")
    return x, y


def _manifest(cache_path: Path, *, status: str, download_performed: bool, source_cache: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cache_path": str(cache_path),
        "source": "openml_mnist_784",
        "source_status": status,
        "download_performed": bool(download_performed),
        "payload_keys": ["X", "y"],
        "payload_shape": {"X": [70000, 784], "y": [70000]},
    }
    if source_cache is not None:
        payload["source_cache"] = str(source_cache)
    return payload


def ensure_original_mnist(
    raw_root: Path,
    *,
    source_cache: str | Path | None = None,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Ensure this stage has a local OpenML MNIST uint8 cache under raw_outputs."""

    cache_path = original_mnist_path(raw_root)
    manifest_path = cache_path.parent / ORIGINAL_MNIST_MANIFEST
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        x, y = _load_mnist_cache(cache_path)
        manifest = _manifest(cache_path, status="local_cache", download_performed=False)
        write_json_atomic(manifest_path, manifest)
        return x, y, manifest

    if source_cache is not None:
        source_path = Path(source_cache).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        _load_mnist_cache(source_path)
        shutil.copy2(source_path, cache_path)
        x, y = _load_mnist_cache(cache_path)
        manifest = _manifest(cache_path, status="copied_from_source_cache", download_performed=False, source_cache=source_path)
        write_json_atomic(manifest_path, manifest)
        return x, y, manifest

    if not download:
        raise FileNotFoundError(
            f"{cache_path} does not exist. Provide --source-cache or allow OpenML download."
        )

    try:
        from sklearn.datasets import fetch_openml

        fetched = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="auto",
            data_home=str(cache_path.parent / "openml"),
        )
    except Exception as exc:  # pragma: no cover - depends on external network/OpenML.
        raise RuntimeError(
            f"MNIST data are not available locally and OpenML fetch failed for {cache_path}."
        ) from exc

    x = np.asarray(fetched.data, dtype=np.uint8).reshape(-1, 784)
    y = np.asarray(fetched.target, dtype=np.int16).reshape(-1)
    write_npz_atomic(cache_path, {"X": x, "y": y})
    manifest = _manifest(cache_path, status="downloaded_from_openml", download_performed=True)
    write_json_atomic(manifest_path, manifest)
    return x, y, manifest


def box_downscale_10(x784: np.ndarray) -> np.ndarray:
    x = np.asarray(x784, dtype=np.uint8).reshape(-1, 28, 28)
    out = np.empty((x.shape[0], INPUT_DIM), dtype=np.float32)
    for i, image in enumerate(x):
        small = Image.fromarray(image).resize((INPUT_SIDE, INPUT_SIDE), Image.Resampling.BOX)
        out[i] = np.asarray(small, dtype=np.float32).reshape(-1)
    return out


def build_mnist10_base_payload(
    raw28: np.ndarray,
    digits: np.ndarray,
    *,
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = DEFAULT_N_TEST,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if int(n_train) % 2 != 0 or int(n_test) % 2 != 0:
        raise ValueError("MNIST10 split requires even n_train and n_test for balanced even/odd sampling")

    digits = np.asarray(digits, dtype=np.int16).reshape(-1)
    even_idx = np.flatnonzero((digits % 2) == 0)
    odd_idx = np.flatnonzero((digits % 2) == 1)
    if even_idx.size < int(n_train) // 2 + int(n_test) // 2:
        raise ValueError("Not enough even-digit MNIST examples for requested split")
    if odd_idx.size < int(n_train) // 2 + int(n_test) // 2:
        raise ValueError("Not enough odd-digit MNIST examples for requested split")

    rng = np.random.default_rng(int(split_seed))
    even_perm = rng.permutation(even_idx)
    odd_perm = rng.permutation(odd_idx)
    train_idx = np.concatenate([even_perm[: int(n_train) // 2], odd_perm[: int(n_train) // 2]])
    test_idx = np.concatenate(
        [
            even_perm[int(n_train) // 2 : int(n_train) // 2 + int(n_test) // 2],
            odd_perm[int(n_train) // 2 : int(n_train) // 2 + int(n_test) // 2],
        ]
    )
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    x_train_raw = box_downscale_10(np.asarray(raw28)[train_idx])
    x_test_raw = box_downscale_10(np.asarray(raw28)[test_idx])
    mean = x_train_raw.mean(axis=0, keepdims=True)
    train_std = x_train_raw.std(axis=0, keepdims=True)
    std = np.where(train_std < 1.0e-6, 1.0, train_std)
    x_train = ((x_train_raw - mean) / std).astype(np.float32)
    x_test = ((x_test_raw - mean) / std).astype(np.float32)
    digit_train = digits[train_idx].astype(np.int16)
    digit_test = digits[test_idx].astype(np.int16)

    payload = {
        "X_train": x_train,
        "X_test": x_test,
        "X_train_raw10": x_train_raw.astype(np.float32),
        "X_test_raw10": x_test_raw.astype(np.float32),
        "X_train_raw": x_train_raw.astype(np.float32),
        "X_test_raw": x_test_raw.astype(np.float32),
        "digit_train": digit_train,
        "digit_test": digit_test,
        "train_indices": train_idx.astype(np.int64),
        "test_indices": test_idx.astype(np.int64),
        "standardization_mean": mean.astype(np.float32),
        "standardization_std": std.astype(np.float32),
    }
    metadata = {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "split_seed": int(split_seed),
        "input_shape": [INPUT_SIDE, INPUT_SIDE],
        "downscale": "PIL.Image.resize((10, 10), Image.Resampling.BOX)",
        "standardization": "train mean/std on X_train_raw10, std floor 1e-6 -> 1.0",
        "train_even_fraction": float(np.mean((digit_train % 2) == 0)),
        "test_even_fraction": float(np.mean((digit_test % 2) == 0)),
    }
    return payload, metadata


def normalize_labels(y: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(y) > 0, 1, -1).astype(np.int8)


def even_odd_labels(digits: np.ndarray) -> np.ndarray:
    return np.where((np.asarray(digits) % 2) == 0, 1, -1).astype(np.int8)


def _init_theta(seed: int, *, input_dim: int = INPUT_DIM, hidden_width: int = 20) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(seed))
    d = int(input_dim)
    h = int(hidden_width)
    w1 = rng.normal(0.0, 1.0 / math.sqrt(d), size=(h, d)).astype(np.float64)
    b1 = np.zeros(h, dtype=np.float64)
    w2 = rng.normal(0.0, 1.0 / math.sqrt(h), size=(h, h)).astype(np.float64)
    b2 = np.zeros(h, dtype=np.float64)
    w3 = rng.normal(0.0, 1.0 / math.sqrt(h), size=(1, h)).astype(np.float64)
    b3 = np.zeros(1, dtype=np.float64)
    return w1, b1, w2, b2, w3, b3


def teacher_logits(x: np.ndarray, seed: int = TEACHER_SEED) -> np.ndarray:
    w1, b1, w2, b2, w3, b3 = _init_theta(seed, input_dim=np.asarray(x).shape[1], hidden_width=20)
    x64 = np.asarray(x, dtype=np.float64)
    h1 = np.tanh(x64 @ w1.T + b1)
    h2 = np.tanh(h1 @ w2.T + b2)
    return (h2 @ w3.T + b3).reshape(-1)


def balanced_pm1(n: int, seed: int) -> np.ndarray:
    if int(n) % 2 != 0:
        raise ValueError("balanced_pm1 requires even n")
    y = np.concatenate([np.ones(int(n) // 2, dtype=np.int8), -np.ones(int(n) // 2, dtype=np.int8)])
    rng = np.random.default_rng(int(seed))
    rng.shuffle(y)
    return y


def _base_payload_with_labels(base: dict[str, np.ndarray], y_train: np.ndarray, y_test: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "X_train": base["X_train"],
        "y_train": np.asarray(y_train, dtype=np.int8),
        "X_test": base["X_test"],
        "y_test": np.asarray(y_test, dtype=np.int8),
        "X_train_raw10": base["X_train_raw10"],
        "X_test_raw10": base["X_test_raw10"],
        "X_train_raw": base["X_train_raw"],
        "X_test_raw": base["X_test_raw"],
        "digit_train": base["digit_train"],
        "digit_test": base["digit_test"],
        "train_indices": base["train_indices"],
        "test_indices": base["test_indices"],
        "standardization_mean": base["standardization_mean"],
        "standardization_std": base["standardization_std"],
    }


def knn_weight_graph(x: np.ndarray, k: int) -> tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray, float]:
    x64 = np.asarray(x, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean")
    nn.fit(x64)
    dist, idx = nn.kneighbors(x64, return_distance=True)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    positive = dist[dist > 0.0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0

    edge_weight: dict[tuple[int, int], float] = {}
    for i in range(x64.shape[0]):
        for d, j_raw in zip(dist[i], idx[i]):
            j = int(j_raw)
            a, b = (i, j) if i < j else (j, i)
            weight = float(np.exp(-(float(d) ** 2) / (2.0 * sigma * sigma)))
            if weight > edge_weight.get((a, b), -1.0):
                edge_weight[(a, b)] = weight

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    edge_i: list[int] = []
    edge_j: list[int] = []
    edge_w: list[float] = []
    for (a, b), weight in edge_weight.items():
        rows.extend([a, b])
        cols.extend([b, a])
        vals.extend([weight, weight])
        edge_i.append(a)
        edge_j.append(b)
        edge_w.append(weight)
    mat = coo_matrix((vals, (rows, cols)), shape=(x64.shape[0], x64.shape[0]), dtype=np.float64)
    return mat, np.asarray(edge_i), np.asarray(edge_j), np.asarray(edge_w), sigma


def edge_tv_baseline_nmstv(
    y: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_w: np.ndarray,
) -> tuple[float, float, float]:
    labels = normalize_labels(y)
    total_w = float(np.sum(edge_w))
    cut_w = float(np.sum(edge_w[labels[edge_i] != labels[edge_j]]))
    tv = cut_w / max(total_w, 1.0e-300)
    p_pos = float(np.mean(labels == 1))
    baseline = 2.0 * p_pos * (1.0 - p_pos)
    return tv, baseline, float(tv / max(baseline, 1.0e-12))


def max_digit_label_purity(y: np.ndarray, digits: np.ndarray) -> float:
    labels = normalize_labels(y)
    purities = []
    for digit in sorted(np.unique(digits)):
        mask = np.asarray(digits) == digit
        pos = float(np.mean(labels[mask] == 1))
        purities.append(max(pos, 1.0 - pos))
    return float(max(purities))


def spectral_basis(w_mat: coo_matrix, spectral_k: int) -> tuple[np.ndarray, np.ndarray]:
    degree = np.asarray(w_mat.sum(axis=1)).ravel()
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 1.0e-300
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
    lap = eye(w_mat.shape[0], format="csr", dtype=np.float64) - diags(inv_sqrt_degree) @ w_mat.tocsr() @ diags(inv_sqrt_degree)
    eigvals, eigvecs = eigsh(lap, k=int(spectral_k) + 1, which="SM", tol=1.0e-6)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order[1 : int(spectral_k) + 1]]


def _choose_very_low_tv_candidate(
    *,
    x_train: np.ndarray,
    y_even: np.ndarray,
    digit_train: np.ndarray,
    graphs: dict[int, tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray, float]],
) -> dict[str, Any]:
    real_k16 = edge_tv_baseline_nmstv(y_even, graphs[16][1], graphs[16][2], graphs[16][3])[2]
    best: dict[str, Any] | None = None
    best_rejected: dict[str, Any] | None = None
    for spectral_k in VERY_LOW_TV_SPECTRAL_KS:
        eigvals, basis = spectral_basis(graphs[16][0], spectral_k)
        rng_seed = VERY_LOW_TV_RNG_SEED + 1009 * int(spectral_k)
        rng = np.random.default_rng(rng_seed)
        for draw_idx in range(VERY_LOW_TV_MAX_DRAWS_PER_BASIS):
            coeff = rng.normal(size=basis.shape[1])
            score = np.asarray(basis @ coeff, dtype=np.float64)
            threshold = float(np.median(score))
            y = np.where(score >= threshold, 1, -1).astype(np.int8)
            pos = float(np.mean(y == 1))
            k16_tv, k16_baseline, k16_nmstv = edge_tv_baseline_nmstv(y, graphs[16][1], graphs[16][2], graphs[16][3])
            corr = float(np.corrcoef(y.astype(np.float64), normalize_labels(y_even).astype(np.float64))[0, 1])
            purity = max_digit_label_purity(y, digit_train)
            row = {
                "spectral_k": int(spectral_k),
                "draw_idx": int(draw_idx),
                "rng_seed": int(rng_seed),
                "coefficients": coeff.tolist(),
                "threshold": threshold,
                "pos_fraction": pos,
                "k16_tv": float(k16_tv),
                "k16_baseline": float(k16_baseline),
                "k16_nmstv": float(k16_nmstv),
                "corr_even_odd": corr,
                "max_digit_label_purity": purity,
                "laplacian_eigenvalues": eigvals.tolist(),
            }
            score_key = float(k16_nmstv + 0.2 * abs(corr) + 0.2 * purity)
            if best_rejected is None or score_key < float(best_rejected["score_key"]):
                best_rejected = {**row, "score_key": score_key}
            if not (0.48 <= pos <= 0.52 and k16_nmstv < 0.8 * real_k16 and abs(corr) < 0.25 and purity < 0.80):
                continue
            nmstvs = []
            tvs = []
            for k in VERY_LOW_TV_K_VALUES:
                tv, _baseline, nmstv = edge_tv_baseline_nmstv(y, graphs[k][1], graphs[k][2], graphs[k][3])
                tvs.append(float(tv))
                nmstvs.append(float(nmstv))
            row["tv_mean"] = float(np.mean(tvs))
            row["nmstv_mean"] = float(np.mean(nmstvs))
            row["nmstv_by_k"] = {str(k): float(v) for k, v in zip(VERY_LOW_TV_K_VALUES, nmstvs)}
            row["tv_by_k"] = {str(k): float(v) for k, v in zip(VERY_LOW_TV_K_VALUES, tvs)}
            if best is None or float(row["nmstv_mean"]) < float(best["nmstv_mean"]):
                best = row
    if best is None:
        raise RuntimeError(f"No feasible very-low-TV candidate found. Best rejected candidate: {best_rejected}")
    return best


def very_low_tv_spectral_teacher(
    base: dict[str, np.ndarray],
    y_even_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x_train = np.asarray(base["X_train"], dtype=np.float64)
    x_test = np.asarray(base["X_test"], dtype=np.float64)
    digit_train = base["digit_train"]
    graphs = {k: knn_weight_graph(x_train, k) for k in VERY_LOW_TV_K_VALUES}
    candidate = _choose_very_low_tv_candidate(
        x_train=x_train,
        y_even=normalize_labels(y_even_train),
        digit_train=digit_train,
        graphs=graphs,
    )

    _eigvals, basis = spectral_basis(graphs[16][0], int(candidate["spectral_k"]))
    coeff = np.asarray(candidate["coefficients"], dtype=np.float64)
    train_score = np.asarray(basis @ coeff, dtype=np.float64)
    threshold = float(candidate["threshold"])
    y_train = np.where(train_score >= threshold, 1, -1).astype(np.int8)

    sigma = graphs[16][4]
    nn = NearestNeighbors(n_neighbors=16, metric="euclidean")
    nn.fit(x_train)
    test_dist, test_idx = nn.kneighbors(x_test, return_distance=True)
    test_weight = np.exp(-(test_dist**2) / (2.0 * sigma * sigma))
    test_score = np.sum(test_weight * train_score[test_idx], axis=1) / np.maximum(np.sum(test_weight, axis=1), 1.0e-300)
    y_test = np.where(test_score >= threshold, 1, -1).astype(np.int8)

    metadata = {
        "rule": VERY_LOW_TV_RULE,
        "definition": "very-low-frequency spectral graph teacher on the MNIST10 train kNN graph with kNN interpolation for test labels",
        "selection_constraints": {
            "train_pos_fraction": "[0.48, 0.52]",
            "k16_nmstv": "< 0.8 * real_even_odd_k16_nmstv",
            "abs_corr_even_odd": "< 0.25",
            "max_digit_label_purity": "< 0.80",
        },
        "candidate": candidate,
        "train_pos_fraction": float(np.mean(y_train == 1)),
        "test_pos_fraction": float(np.mean(y_test == 1)),
    }
    return y_train, y_test, train_score.astype(np.float64), test_score.astype(np.float64), metadata


def manual_rule_payload(
    rule_name: str,
    base: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    y_even_train = even_odd_labels(base["digit_train"])
    y_even_test = even_odd_labels(base["digit_test"])

    if rule_name == VERY_LOW_TV_RULE:
        y_train, y_test, train_score, test_score, metadata = very_low_tv_spectral_teacher(base, y_even_train)
        payload = _base_payload_with_labels(base, y_train, y_test)
        payload["spectral_train_score"] = train_score
        payload["spectral_test_score"] = test_score
        return payload, metadata

    if rule_name == "real_even_odd":
        return _base_payload_with_labels(base, y_even_train, y_even_test), {
            "rule": rule_name,
            "definition": "even digit +1, odd digit -1",
        }

    if rule_name == "teacher_nn":
        teacher_train = teacher_logits(base["X_train"], TEACHER_SEED)
        teacher_test = teacher_logits(base["X_test"], TEACHER_SEED)
        threshold = float(np.median(teacher_train))
        y_train = np.where(teacher_train >= threshold, 1, -1).astype(np.int8)
        y_test = np.where(teacher_test >= threshold, 1, -1).astype(np.int8)
        return _base_payload_with_labels(base, y_train, y_test), {
            "rule": rule_name,
            "teacher_seed": TEACHER_SEED,
            "teacher_architecture": "100-20-20-1-tanh",
            "train_median_logit_threshold": threshold,
        }

    if rule_name == "random_label":
        y_train = balanced_pm1(base["X_train"].shape[0], RANDOM_LABEL_TRAIN_SEED)
        y_test = balanced_pm1(base["X_test"].shape[0], RANDOM_LABEL_TEST_SEED)
        return _base_payload_with_labels(base, y_train, y_test), {
            "rule": rule_name,
            "train_seed": RANDOM_LABEL_TRAIN_SEED,
            "test_seed": RANDOM_LABEL_TEST_SEED,
            "definition": "balanced random +/-1 labels independent of digit",
        }

    raise ValueError(f"Unsupported MNIST manual rule: {rule_name}")


def eta_noise_payload(
    base: dict[str, np.ndarray],
    *,
    eta: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    y_train = even_odd_labels(base["digit_train"])
    y_test = even_odd_labels(base["digit_test"])
    rng = np.random.default_rng(int(seed))
    train_mask = rng.random(y_train.shape[0]) < float(eta)
    test_mask = rng.random(y_test.shape[0]) < float(eta)
    noisy_train = y_train.copy()
    noisy_test = y_test.copy()
    noisy_train[train_mask] *= -1
    noisy_test[test_mask] *= -1
    payload = _base_payload_with_labels(base, noisy_train, noisy_test)
    payload["eta_flip_mask_train"] = train_mask.astype(bool)
    payload["eta_flip_mask_test"] = test_mask.astype(bool)
    payload["eta"] = np.asarray(float(eta), dtype=np.float32)
    payload["eta_seed"] = np.asarray(int(seed), dtype=np.int64)
    metadata = {
        "base_rule": "real_even_odd",
        "eta": float(eta),
        "eta_seed": int(seed),
        "train_flip_count": int(np.sum(train_mask)),
        "test_flip_count": int(np.sum(test_mask)),
    }
    return payload, metadata

