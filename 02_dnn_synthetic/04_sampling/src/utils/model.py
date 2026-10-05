from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


PARAMETER_ORDER = ("W1", "b1", "W2", "b2", "W3", "b3")
PARAMETER_SHAPES: Mapping[str, tuple[int, ...]] = {
    "W1": (48, 2),
    "b1": (48,),
    "W2": (48, 48),
    "b2": (48,),
    "W3": (1, 48),
    "b3": (1,),
}


@dataclass(frozen=True)
class ParameterSlice:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int


def parameter_spec() -> tuple[ParameterSlice, ...]:
    result: list[ParameterSlice] = []
    cursor = 0
    for name in PARAMETER_ORDER:
        shape = PARAMETER_SHAPES[name]
        size = int(np.prod(shape))
        result.append(ParameterSlice(name, shape, cursor, cursor + size))
        cursor += size
    return tuple(result)


PARAMETER_SPEC = parameter_spec()
PARAMETER_COUNT = PARAMETER_SPEC[-1].stop
if PARAMETER_COUNT != 2545:
    raise RuntimeError(f"model parameter count drifted: {PARAMETER_COUNT}")


def initialize_theta(seed: int, *, base_scale: float = 0.2, multiplier: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    values: list[np.ndarray] = []
    fan_in_by_weight = {"W1": 2, "W2": 48, "W3": 48}
    for item in PARAMETER_SPEC:
        if item.name.startswith("W"):
            scale = float(multiplier) * float(base_scale) / math.sqrt(fan_in_by_weight[item.name])
            value = rng.normal(loc=0.0, scale=scale, size=item.shape)
        else:
            value = np.zeros(item.shape, dtype=np.float64)
        values.append(np.asarray(value, dtype=np.float64).reshape(-1))
    return np.concatenate(values)


def unpack_batch(theta_batch: torch.Tensor) -> dict[str, torch.Tensor]:
    if theta_batch.ndim == 1:
        theta_batch = theta_batch.unsqueeze(0)
    if theta_batch.ndim != 2 or theta_batch.shape[1] != PARAMETER_COUNT:
        raise ValueError(f"theta must have shape (batch,{PARAMETER_COUNT}), got {tuple(theta_batch.shape)}")
    return {
        item.name: theta_batch[:, item.start : item.stop].reshape((theta_batch.shape[0],) + item.shape)
        for item in PARAMETER_SPEC
    }


def logits_batch(theta_batch: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    params = unpack_batch(theta_batch)
    hidden1 = torch.tanh(torch.einsum("nd,bhd->bnh", x, params["W1"]) + params["b1"][:, None, :])
    hidden2 = torch.tanh(
        torch.einsum("bnh,bkh->bnk", hidden1, params["W2"]) + params["b2"][:, None, :]
    )
    return (
        torch.einsum("bnk,bok->bno", hidden2, params["W3"]) + params["b3"][:, None, :]
    ).squeeze(-1)


def logits(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if theta.ndim != 1:
        raise ValueError("single-theta logits expects a one-dimensional parameter vector")
    return logits_batch(theta.unsqueeze(0), x).squeeze(0)
