from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DNNArch:
    input_dim: int = 2
    hidden_width: int = 48
    hidden_layers: int = 2
    activation: str = "tanh"

    @property
    def param_count(self) -> int:
        return (
            self.input_dim * self.hidden_width
            + self.hidden_width
            + self.hidden_width * self.hidden_width
            + self.hidden_width
            + self.hidden_width
            + 1
        )


ARCH = DNNArch()
P = ARCH.param_count

_TORCH_XY_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
_TORCH_XY_CACHE_MAX = int(os.environ.get("COMPLEXITY_TORCH_XY_CACHE_MAX", "128"))


def _array_cache_key(arr: np.ndarray) -> tuple[Any, ...]:
    base = np.asarray(arr)
    ptr = int(base.__array_interface__.get("data", (0,))[0])
    return (id(base), ptr, tuple(base.shape), str(base.dtype), tuple(base.strides))


def _get_torch_xy_tensors(
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: Any,
    dtype: Any,
    np_dtype: Any,
) -> tuple[Any, Any]:
    import torch

    x_base = np.asarray(x)
    y_base = np.asarray(y)
    key = (_array_cache_key(x_base), _array_cache_key(y_base), str(device), str(dtype))
    try:
        cached = _TORCH_XY_CACHE.pop(key)
    except KeyError:
        x_t = torch.as_tensor(np.asarray(x_base, dtype=np_dtype), device=device, dtype=dtype)
        y_t = torch.as_tensor(normalize_labels(y_base).astype(np_dtype), device=device, dtype=dtype)
        cached = (x_t, y_t)
    _TORCH_XY_CACHE[key] = cached
    while len(_TORCH_XY_CACHE) > _TORCH_XY_CACHE_MAX:
        _TORCH_XY_CACHE.popitem(last=False)
    return cached


def clear_runtime_caches() -> None:
    _TORCH_XY_CACHE.clear()


def unpack_theta(theta: np.ndarray, arch: DNNArch = ARCH) -> tuple[np.ndarray, ...]:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.size != arch.param_count:
        raise ValueError(f"expected P={arch.param_count}, got {theta.size}")
    idx = 0
    h = int(arch.hidden_width)
    d = int(arch.input_dim)
    w1 = theta[idx : idx + d * h].reshape(h, d)
    idx += d * h
    b1 = theta[idx : idx + h]
    idx += h
    w2 = theta[idx : idx + h * h].reshape(h, h)
    idx += h * h
    b2 = theta[idx : idx + h]
    idx += h
    w3 = theta[idx : idx + h].reshape(1, h)
    idx += h
    b3 = theta[idx : idx + 1]
    idx += 1
    if idx != theta.size:
        raise AssertionError("theta unpack did not consume all parameters")
    return w1, b1, w2, b2, w3, b3


def flatten_parts(parts: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in parts])


def logits(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    w1, b1, w2, b2, w3, b3 = unpack_theta(theta)
    h1 = np.tanh(np.asarray(x, dtype=np.float64) @ w1.T + b1)
    h2 = np.tanh(h1 @ w2.T + b2)
    return (h2 @ w3.T + b3).reshape(-1)


def ce_and_error(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    z = logits(theta, x)
    y_pm1 = normalize_labels(y)
    yz = y_pm1 * z
    ce = float(np.mean(np.logaddexp(0.0, -yz)))
    err = float(np.mean(yz <= 0.0))
    return ce, err


def normalize_labels(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    values = set(np.unique(arr).tolist())
    if values.issubset({0.0, 1.0}):
        arr = 2.0 * arr - 1.0
    return np.where(arr >= 0.0, 1.0, -1.0)


def _ce_error_batch_torch(
    theta_batch: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_size: int,
    device_name: str,
    dtype_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float32 if str(dtype_name).lower() in {"float32", "fp32"} else torch.float64
    np_dtype = np.float32 if dtype is torch.float32 else np.float64
    x_t, y_t = _get_torch_xy_tensors(x, y, device=device, dtype=dtype, np_dtype=np_dtype)
    theta_np = np.asarray(theta_batch, dtype=np.float32 if dtype is torch.float32 else np.float64)
    ce_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    h = ARCH.hidden_width
    d = ARCH.input_dim
    with torch.no_grad():
        for start in range(0, theta_np.shape[0], max(1, int(chunk_size))):
            batch = torch.as_tensor(theta_np[start : start + max(1, int(chunk_size))], device=device, dtype=dtype)
            idx = 0
            w1 = batch[:, idx : idx + d * h].reshape(batch.shape[0], h, d)
            idx += d * h
            b1 = batch[:, idx : idx + h]
            idx += h
            w2 = batch[:, idx : idx + h * h].reshape(batch.shape[0], h, h)
            idx += h * h
            b2 = batch[:, idx : idx + h]
            idx += h
            w3 = batch[:, idx : idx + h].reshape(batch.shape[0], 1, h)
            idx += h
            b3 = batch[:, idx : idx + 1].reshape(batch.shape[0])
            h1 = torch.tanh(torch.einsum("nd,bhd->bnh", x_t, w1) + b1[:, None, :])
            h2 = torch.tanh(torch.einsum("bnh,bkh->bnk", h1, w2) + b2[:, None, :])
            z = torch.einsum("bnh,bh->bn", h2, w3[:, 0, :]) + b3[:, None]
            yz = z * y_t[None, :]
            ce = torch.mean(torch.nn.functional.softplus(-yz), dim=1)
            err = torch.mean((yz <= 0.0).to(dtype), dim=1)
            ce_rows.append(ce.detach().cpu().numpy().astype(np.float64))
            err_rows.append(err.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(ce_rows), np.concatenate(err_rows)


def ce_error_batch(
    theta_batch: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_size: int = 64,
    device: str = "cpu",
    dtype: str = "float64",
) -> tuple[np.ndarray, np.ndarray]:
    theta_batch = np.asarray(theta_batch, dtype=np.float64)
    if theta_batch.ndim == 1:
        theta_batch = theta_batch.reshape(1, -1)
    if theta_batch.shape[1] != P:
        raise ValueError(f"expected theta batch shape (*,{P}), got {theta_batch.shape}")
    if str(device).lower() in {"auto", "cuda"}:
        try:
            return _ce_error_batch_torch(theta_batch, x, y, chunk_size=chunk_size, device_name=str(device).lower(), dtype_name=str(dtype))
        except Exception:
            if str(device).lower() == "cuda":
                raise
    x = np.asarray(x, dtype=np.float64)
    y_pm1 = normalize_labels(y)
    ce_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    h = ARCH.hidden_width
    d = ARCH.input_dim
    for start in range(0, theta_batch.shape[0], max(1, int(chunk_size))):
        batch = theta_batch[start : start + max(1, int(chunk_size))]
        idx = 0
        w1 = batch[:, idx : idx + d * h].reshape(batch.shape[0], h, d)
        idx += d * h
        b1 = batch[:, idx : idx + h]
        idx += h
        w2 = batch[:, idx : idx + h * h].reshape(batch.shape[0], h, h)
        idx += h * h
        b2 = batch[:, idx : idx + h]
        idx += h
        w3 = batch[:, idx : idx + h].reshape(batch.shape[0], 1, h)
        idx += h
        b3 = batch[:, idx : idx + 1].reshape(batch.shape[0])
        a1 = np.einsum("nd,bhd->bnh", x, w1, optimize=True) + b1[:, None, :]
        h1 = np.tanh(a1)
        a2 = np.einsum("bnh,bkh->bnk", h1, w2, optimize=True) + b2[:, None, :]
        h2 = np.tanh(a2)
        z = np.einsum("bnh,bh->bn", h2, w3[:, 0, :], optimize=True) + b3[:, None]
        yz = z * y_pm1[None, :]
        ce_rows.append(np.mean(np.logaddexp(0.0, -yz), axis=1))
        err_rows.append(np.mean(yz <= 0.0, axis=1))
    return np.concatenate(ce_rows), np.concatenate(err_rows)


def ce_radial_grad_batch(
    theta_batch: np.ndarray,
    directions: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_size: int = 64,
    device: str = "cpu",
    dtype: str = "float64",
) -> tuple[np.ndarray, np.ndarray]:
    """Return CE and grad_theta CE dot radial direction for each theta row."""
    import torch

    theta_batch = np.asarray(theta_batch, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if theta_batch.ndim == 1:
        theta_batch = theta_batch.reshape(1, -1)
    if directions.ndim == 1:
        directions = directions.reshape(1, -1)
    if theta_batch.shape != directions.shape:
        raise ValueError(f"theta and direction batches must match, got {theta_batch.shape} and {directions.shape}")
    if theta_batch.shape[1] != P:
        raise ValueError(f"expected theta batch shape (*,{P}), got {theta_batch.shape}")

    device_name = str(device).lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for radial derivative, but torch.cuda.is_available() is false")
    torch_device = torch.device(device_name)
    torch_dtype = torch.float32 if str(dtype).lower() in {"float32", "fp32"} else torch.float64
    np_dtype = np.float32 if torch_dtype is torch.float32 else np.float64
    x_t, y_t = _get_torch_xy_tensors(x, y, device=torch_device, dtype=torch_dtype, np_dtype=np_dtype)
    theta_np = theta_batch.astype(np_dtype, copy=False)
    direction_np = directions.astype(np_dtype, copy=False)

    ce_rows: list[np.ndarray] = []
    radial_rows: list[np.ndarray] = []
    h = ARCH.hidden_width
    d = ARCH.input_dim
    for start in range(0, theta_np.shape[0], max(1, int(chunk_size))):
        batch_np = theta_np[start : start + max(1, int(chunk_size))]
        direction_chunk = torch.as_tensor(direction_np[start : start + max(1, int(chunk_size))], device=torch_device, dtype=torch_dtype)
        batch = torch.as_tensor(batch_np, device=torch_device, dtype=torch_dtype).detach().requires_grad_(True)
        idx = 0
        w1 = batch[:, idx : idx + d * h].reshape(batch.shape[0], h, d)
        idx += d * h
        b1 = batch[:, idx : idx + h]
        idx += h
        w2 = batch[:, idx : idx + h * h].reshape(batch.shape[0], h, h)
        idx += h * h
        b2 = batch[:, idx : idx + h]
        idx += h
        w3 = batch[:, idx : idx + h].reshape(batch.shape[0], 1, h)
        idx += h
        b3 = batch[:, idx : idx + 1].reshape(batch.shape[0])
        h1 = torch.tanh(torch.einsum("nd,bhd->bnh", x_t, w1) + b1[:, None, :])
        h2 = torch.tanh(torch.einsum("bnh,bkh->bnk", h1, w2) + b2[:, None, :])
        z = torch.einsum("bnh,bh->bn", h2, w3[:, 0, :]) + b3[:, None]
        yz = z * y_t[None, :]
        ce = torch.mean(torch.nn.functional.softplus(-yz), dim=1)
        grad = torch.autograd.grad(ce.sum(), batch)[0]
        radial_grad = torch.sum(grad * direction_chunk, dim=1)
        ce_rows.append(ce.detach().cpu().numpy().astype(np.float64))
        radial_rows.append(radial_grad.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(ce_rows), np.concatenate(radial_rows)
