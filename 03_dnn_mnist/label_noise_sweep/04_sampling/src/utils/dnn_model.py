from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MNIST10Arch:
    input_dim: int = 100
    hidden_width: int = 20
    hidden_layers: int = 2
    activation: str = "tanh"

    @property
    def param_count(self) -> int:
        h = int(self.hidden_width)
        d = int(self.input_dim)
        return d * h + h + h * h + h + h + 1


ARCH = MNIST10Arch()
P = ARCH.param_count
_TORCH_DATA_CACHE: dict[tuple[int, int, str, str, tuple[int, ...], tuple[int, ...]], tuple[object, object]] = {}


def normalize_labels(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    values = set(np.unique(arr).tolist())
    if values.issubset({0.0, 1.0}):
        arr = 2.0 * arr - 1.0
    return np.where(arr >= 0.0, 1.0, -1.0)


def unpack_theta(theta: np.ndarray, arch: MNIST10Arch = ARCH) -> tuple[np.ndarray, ...]:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.size != arch.param_count:
        raise ValueError(f"expected P={arch.param_count}, got {theta.size}")
    h = int(arch.hidden_width)
    d = int(arch.input_dim)
    idx = 0
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
    return w1, b1, w2, b2, w3, b3


def logits_np(theta: np.ndarray, x: np.ndarray, arch: MNIST10Arch = ARCH) -> np.ndarray:
    w1, b1, w2, b2, w3, b3 = unpack_theta(theta, arch)
    x64 = np.asarray(x, dtype=np.float64)
    h1 = np.tanh(x64 @ w1.T + b1)
    h2 = np.tanh(h1 @ w2.T + b2)
    return (h2 @ w3.T + b3).reshape(-1)


def ce_and_error_np(theta: np.ndarray, x: np.ndarray, y: np.ndarray, arch: MNIST10Arch = ARCH) -> tuple[float, float]:
    z = logits_np(theta, x, arch)
    yz = normalize_labels(y) * z
    ce = float(np.mean(np.logaddexp(0.0, -yz)))
    err = float(np.mean(yz <= 0.0))
    return ce, err


def _torch_xy(x: np.ndarray, y: np.ndarray, device: str, dtype: str) -> tuple[object, object, object, object]:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    torch_dtype = torch.float32 if str(dtype).lower() in {"float32", "fp32"} else torch.float64
    np_dtype = np.float32 if torch_dtype is torch.float32 else np.float64
    x_arr = np.asarray(x, dtype=np_dtype)
    y_arr = np.asarray(y)
    cache_key = (id(x), id(y), str(torch_device), str(torch_dtype), tuple(x_arr.shape), tuple(y_arr.shape))
    cached = _TORCH_DATA_CACHE.get(cache_key)
    if cached is None:
        x_t = torch.as_tensor(x_arr, device=torch_device, dtype=torch_dtype)
        y_t = torch.as_tensor(normalize_labels(y_arr).astype(np_dtype), device=torch_device, dtype=torch_dtype)
        _TORCH_DATA_CACHE[cache_key] = (x_t, y_t)
    else:
        x_t, y_t = cached
    return x_t, y_t, torch_device, torch_dtype


def _ce_err_from_theta_tensor(batch, x_t, y_t, arch: MNIST10Arch):
    import torch

    h = int(arch.hidden_width)
    d = int(arch.input_dim)
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
    err = torch.mean((yz <= 0.0).to(batch.dtype), dim=1)
    return ce, err


def ce_error_batch_torch(
    theta_batch: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_size: int = 128,
    device: str = "auto",
    dtype: str = "float32",
    arch: MNIST10Arch = ARCH,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch_dtype_np = np.float32 if str(dtype).lower() in {"float32", "fp32"} else np.float64
    theta_batch = np.asarray(theta_batch, dtype=torch_dtype_np)
    if theta_batch.ndim == 1:
        theta_batch = theta_batch.reshape(1, -1)
    if theta_batch.shape[1] != arch.param_count:
        raise ValueError(f"expected theta batch shape (*,{arch.param_count}), got {theta_batch.shape}")
    x_t, y_t, torch_device, torch_dtype = _torch_xy(x, y, device, dtype)
    ce_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, theta_batch.shape[0], max(1, int(chunk_size))):
            batch = torch.as_tensor(theta_batch[start : start + int(chunk_size)], device=torch_device, dtype=torch_dtype)
            ce, err = _ce_err_from_theta_tensor(batch, x_t, y_t, arch)
            ce_rows.append(ce.detach().cpu().numpy().astype(np.float64))
            err_rows.append(err.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(ce_rows), np.concatenate(err_rows)


def ce_radial_grad_batch_torch(
    theta_batch: np.ndarray,
    directions: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_size: int = 128,
    device: str = "auto",
    dtype: str = "float32",
    arch: MNIST10Arch = ARCH,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch_dtype_np = np.float32 if str(dtype).lower() in {"float32", "fp32"} else np.float64
    theta_batch = np.asarray(theta_batch, dtype=torch_dtype_np)
    directions = np.asarray(directions, dtype=torch_dtype_np)
    x_t, y_t, torch_device, torch_dtype = _torch_xy(x, y, device, dtype)
    ce_rows: list[np.ndarray] = []
    radial_rows: list[np.ndarray] = []
    for start in range(0, theta_batch.shape[0], max(1, int(chunk_size))):
        batch_np = theta_batch[start : start + int(chunk_size)]
        direction_np = directions[start : start + int(chunk_size)]
        batch = torch.as_tensor(batch_np, device=torch_device, dtype=torch_dtype).clone().detach().requires_grad_(True)
        direction = torch.as_tensor(direction_np, device=torch_device, dtype=torch_dtype)
        ce, _err = _ce_err_from_theta_tensor(batch, x_t, y_t, arch)
        grad = torch.autograd.grad(ce.sum(), batch, retain_graph=False, create_graph=False)[0]
        radial = torch.sum(grad * direction, dim=1)
        ce_rows.append(ce.detach().cpu().numpy().astype(np.float64))
        radial_rows.append(radial.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(ce_rows), np.concatenate(radial_rows)


DNNArch = MNIST10Arch
ce_and_error = ce_and_error_np
ce_error_batch = ce_error_batch_torch
ce_radial_grad_batch = ce_radial_grad_batch_torch
