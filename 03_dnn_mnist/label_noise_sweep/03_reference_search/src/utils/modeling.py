from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterLayout:
    input_dim: int = 100
    hidden1: int = 20
    hidden2: int = 20
    output_dim: int = 1

    @property
    def parameter_count(self) -> int:
        return (
            self.hidden1 * self.input_dim
            + self.hidden1
            + self.hidden2 * self.hidden1
            + self.hidden2
            + self.output_dim * self.hidden2
            + self.output_dim
        )

    @classmethod
    def from_config(cls, model: Mapping[str, Any]) -> "ParameterLayout":
        hidden = tuple(int(value) for value in model["hidden_dims"])
        if len(hidden) != 2:
            raise ValueError("the revised MNIST model requires exactly two hidden layers")
        layout = cls(
            input_dim=int(model["input_dim"]),
            hidden1=hidden[0],
            hidden2=hidden[1],
            output_dim=int(model["output_dim"]),
        )
        declared = int(model["parameter_count"])
        if layout.parameter_count != declared:
            raise ValueError(
                f"model parameter-count mismatch: computed {layout.parameter_count}, "
                f"declared {declared}"
            )
        if layout != cls():
            raise ValueError(f"production architecture drifted from 100-20-20-1: {layout}")
        if str(model["activation"]).lower() != "tanh":
            raise ValueError("production activation must be tanh")
        return layout


LAYOUT = ParameterLayout()
P = LAYOUT.parameter_count


def parameter_slices(layout: ParameterLayout = LAYOUT) -> dict[str, slice]:
    offset = 0
    result: dict[str, slice] = {}
    for name, size in (
        ("w1", layout.hidden1 * layout.input_dim),
        ("b1", layout.hidden1),
        ("w2", layout.hidden2 * layout.hidden1),
        ("b2", layout.hidden2),
        ("w3", layout.output_dim * layout.hidden2),
        ("b3", layout.output_dim),
    ):
        result[name] = slice(offset, offset + size)
        offset += size
    if offset != layout.parameter_count:
        raise AssertionError("internal parameter layout error")
    return result


SLICES = parameter_slices()


def fresh_fan_in_vector(seed: int, *, layout: ParameterLayout = LAYOUT) -> np.ndarray:
    """Historical production initialization: fan-in normal weights, zero biases."""
    rng = np.random.default_rng(int(seed))
    theta = np.zeros(layout.parameter_count, dtype=np.float64)
    theta[SLICES["w1"]] = rng.normal(
        scale=1.0 / math.sqrt(layout.input_dim),
        size=layout.hidden1 * layout.input_dim,
    )
    theta[SLICES["w2"]] = rng.normal(
        scale=1.0 / math.sqrt(layout.hidden1),
        size=layout.hidden2 * layout.hidden1,
    )
    theta[SLICES["w3"]] = rng.normal(
        scale=1.0 / math.sqrt(layout.hidden2),
        size=layout.output_dim * layout.hidden2,
    )
    return theta


def torch_logits_single(theta, x):
    """Forward one parameter vector; Torch is imported by the caller."""
    w1 = theta[SLICES["w1"]].reshape(LAYOUT.hidden1, LAYOUT.input_dim)
    b1 = theta[SLICES["b1"]]
    w2 = theta[SLICES["w2"]].reshape(LAYOUT.hidden2, LAYOUT.hidden1)
    b2 = theta[SLICES["b2"]]
    w3 = theta[SLICES["w3"]].reshape(LAYOUT.output_dim, LAYOUT.hidden2)
    b3 = theta[SLICES["b3"]]
    hidden1 = (x @ w1.T + b1).tanh()
    hidden2 = (hidden1 @ w2.T + b2).tanh()
    return (hidden2 @ w3.T + b3).reshape(-1)


def torch_logits_batched(theta_batch, x):
    """Forward independent networks for a batch of flat parameter vectors."""
    w1 = theta_batch[:, SLICES["w1"]].reshape(
        -1, LAYOUT.hidden1, LAYOUT.input_dim
    )
    b1 = theta_batch[:, SLICES["b1"]]
    w2 = theta_batch[:, SLICES["w2"]].reshape(
        -1, LAYOUT.hidden2, LAYOUT.hidden1
    )
    b2 = theta_batch[:, SLICES["b2"]]
    w3 = theta_batch[:, SLICES["w3"]].reshape(
        -1, LAYOUT.output_dim, LAYOUT.hidden2
    )
    b3 = theta_batch[:, SLICES["b3"]]
    hidden1 = (x.unsqueeze(0) @ w1.transpose(1, 2) + b1[:, None, :]).tanh()
    hidden2 = (
        hidden1 @ w2.transpose(1, 2) + b2[:, None, :]
    ).tanh()
    return (
        hidden2 @ w3.transpose(1, 2) + b3[:, None, :]
    ).squeeze(-1)


def numpy_logits(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64)
    if theta.size != P:
        raise ValueError(f"expected {P} parameters, found {theta.size}")
    w1 = theta[SLICES["w1"]].reshape(LAYOUT.hidden1, LAYOUT.input_dim)
    b1 = theta[SLICES["b1"]]
    w2 = theta[SLICES["w2"]].reshape(LAYOUT.hidden2, LAYOUT.hidden1)
    b2 = theta[SLICES["b2"]]
    w3 = theta[SLICES["w3"]].reshape(LAYOUT.output_dim, LAYOUT.hidden2)
    b3 = theta[SLICES["b3"]]
    h1 = np.tanh(x @ w1.T + b1)
    h2 = np.tanh(h1 @ w2.T + b2)
    return (h2 @ w3.T + b3).reshape(-1)


def replay_metrics_numpy(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Dependency-light float64 replay used by artifact validators."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    logits = numpy_logits(theta, x)
    margins = y * logits
    ce_mean = float(np.logaddexp(0.0, -margins).mean())
    n_wrong = int(np.count_nonzero(margins <= 0.0))
    return {
        "ce_mean": ce_mean,
        "n_wrong": n_wrong,
        "min_margin": float(margins.min()),
        "mean_margin": float(margins.mean()),
        "theta_norm_sq": float(theta @ theta),
    }
