from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model_types import DNNArch


def activation_forward(x: torch.Tensor, name: str) -> torch.Tensor:
    if str(name).lower() == "softplus":
        return F.softplus(x)
    if str(name).lower() == "tanh":
        return torch.tanh(x)
    raise ValueError(f"Unsupported activation: {name!r}")


def activation_prime(x: torch.Tensor, name: str, activated: Optional[torch.Tensor] = None) -> torch.Tensor:
    if str(name).lower() == "softplus":
        return torch.sigmoid(x)
    if str(name).lower() == "tanh":
        out = torch.tanh(x) if activated is None else activated
        return 1.0 - out * out
    raise ValueError(f"Unsupported activation: {name!r}")


def loss_forward(logits: torch.Tensor, y_pm1: torch.Tensor, name: str, *, margin: float) -> torch.Tensor:
    if str(name).lower() == "logistic":
        return F.softplus(-y_pm1 * logits)
    if str(name).lower() == "squared_hinge":
        slack = torch.clamp(float(margin) - y_pm1 * logits, min=0.0)
        return slack * slack
    if str(name).lower() == "exact_sign":
        # Steep sign surrogate: approximates the fraction of sign mistakes.
        return torch.sigmoid(-25.0 * y_pm1 * logits)
    raise ValueError(f"Unsupported loss: {name!r}")


def init_param(rng: np.random.Generator, shape: Tuple[int, ...], scale: float) -> np.ndarray:
    return (scale * rng.normal(size=shape)).astype(np.float64)


def init_3layer_params(arch: DNNArch, rng: np.random.Generator, *, init_scale_multiplier: float = 1.0) -> Dict[str, np.ndarray]:
    mult = float(init_scale_multiplier)
    W1 = init_param(rng, (arch.width1, arch.input_dim), scale=mult * (0.2 / math.sqrt(arch.input_dim)))
    b1 = np.zeros((arch.width1,), dtype=np.float64)
    W2 = init_param(rng, (arch.width2, arch.width1), scale=mult * (0.2 / math.sqrt(arch.width1)))
    b2 = np.zeros((arch.width2,), dtype=np.float64)
    W3 = init_param(rng, (1, arch.width2), scale=mult * (0.2 / math.sqrt(arch.width2)))
    b3 = np.zeros((1,), dtype=np.float64)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def flatten_params(params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list[tuple[str, tuple[int, ...], int, int]]]:
    flat_parts = []
    spec = []
    idx = 0
    for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        arr = np.asarray(params[name], dtype=np.float64)
        n = int(arr.size)
        flat_parts.append(arr.reshape(-1))
        spec.append((name, tuple(arr.shape), idx, idx + n))
        idx += n
    return np.concatenate(flat_parts, axis=0), spec


def build_param_spec(arch: DNNArch) -> list[tuple[str, tuple[int, ...], int, int]]:
    params = init_3layer_params(arch, np.random.default_rng(0))
    _, spec = flatten_params(params)
    return spec


def unflatten_params(theta: torch.Tensor, spec: Sequence[tuple[str, tuple[int, ...], int, int]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, shape, a, b in spec:
        out[name] = theta[a:b].view(shape)
    return out


def forward_3layer(X: torch.Tensor, theta: torch.Tensor, spec: Sequence[tuple[str, tuple[int, ...], int, int]], activation: str) -> torch.Tensor:
    params = unflatten_params(theta, spec)
    h1 = activation_forward(X @ params["W1"].T + params["b1"], activation)
    h2 = activation_forward(h1 @ params["W2"].T + params["b2"], activation)
    logits = (h2 @ params["W3"].T + params["b3"]).squeeze(-1)
    return logits


__all__ = [
    "activation_forward",
    "activation_prime",
    "build_param_spec",
    "flatten_params",
    "forward_3layer",
    "init_3layer_params",
    "init_param",
    "loss_forward",
    "unflatten_params",
]



