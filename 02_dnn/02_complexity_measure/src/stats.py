from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def finite_or_default(x: np.ndarray, default: float = 0.0) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return a
    return np.where(np.isfinite(a), a, float(default))


def nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


def nanstd(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size >= 2:
        return float(np.std(arr, ddof=1))
    if arr.size == 1:
        return 0.0
    return float("nan")


def logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    summed = np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)
    out = a_max_safe + np.log(summed + 1e-300)
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 2:
        return float("nan")
    aa = a[m]
    bb = b[m]
    if np.std(aa) <= 1e-15 or np.std(bb) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 2:
        return float("nan")
    order_a = np.argsort(a[m], kind="mergesort")
    order_b = np.argsort(b[m], kind="mergesort")
    ranks_a = np.empty_like(order_a, dtype=np.float64)
    ranks_b = np.empty_like(order_b, dtype=np.float64)
    ranks_a[order_a] = np.arange(1, order_a.size + 1, dtype=np.float64)
    ranks_b[order_b] = np.arange(1, order_b.size + 1, dtype=np.float64)
    return pearson_corr(ranks_a, ranks_b)


def finite_mean_std_axis0(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    counts = np.sum(valid, axis=0)
    mean = np.full(x.shape[1], np.nan, dtype=np.float64)
    mask_mean = counts > 0
    if np.any(mask_mean):
        sums = np.sum(np.where(valid, x, 0.0), axis=0)
        mean[mask_mean] = sums[mask_mean] / counts[mask_mean]
    std = np.full(x.shape[1], np.nan, dtype=np.float64)
    mask_std = counts >= 2
    if np.any(mask_std):
        diff = np.where(valid, x - mean[None, :], 0.0)
        ss = np.sum(diff * diff, axis=0)
        std[mask_std] = np.sqrt(ss[mask_std] / (counts[mask_std] - 1.0))
    std[counts == 1] = 0.0
    return mean, std


def safe_interp(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 2:
        return float("nan")
    return float(np.interp(float(x0), x[m], y[m], left=np.nan, right=np.nan))


def mean_median_from_logp(d: np.ndarray, logp: np.ndarray) -> Tuple[float, float]:
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    logp = np.asarray(logp, dtype=np.float64).reshape(-1)
    m = np.isfinite(d) & np.isfinite(logp)
    if np.sum(m) < 3:
        return float("nan"), float("nan")
    dd = d[m]
    lp = logp[m]
    delta = float(np.mean(np.diff(dd)))
    p = np.exp(np.clip(lp, -800.0, 800.0))
    p = finite_or_default(p, default=0.0)
    Z = float(np.sum(p) * delta)
    if Z <= 0.0:
        return float("nan"), float("nan")
    p = p / Z
    mean_d = float(np.sum(dd * p) * delta)
    cdf = np.cumsum(p) * delta
    med_d = float(np.interp(0.5, np.clip(cdf, 0.0, 1.0), dd))
    return mean_d, med_d


def common_shift_anchor(curves: Sequence[Tuple[np.ndarray, np.ndarray]]) -> float:
    anchors = []
    for d, y in curves:
        d = np.asarray(d, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(d) & np.isfinite(y)
        if np.sum(m) < 2:
            return float("nan")
        anchors.append(float(d[m][-1] - 0.02))
    if not anchors:
        return float("nan")
    anchor = float(min(anchors))
    if anchor <= 0.0:
        return float("nan")
    for d, y in curves:
        if not np.isfinite(safe_interp(d, y, anchor)):
            return float("nan")
    return anchor


__all__ = [
    "common_shift_anchor",
    "finite_mean_std_axis0",
    "finite_or_default",
    "logsumexp",
    "mean_median_from_logp",
    "nanmean",
    "nanstd",
    "pearson_corr",
    "safe_float",
    "safe_interp",
    "spearman_corr",
]
