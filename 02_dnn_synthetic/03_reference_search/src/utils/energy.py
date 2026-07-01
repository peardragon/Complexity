from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .network import forward_3layer, loss_forward


class DNNBinaryEnergy0:
    def __init__(self, X: np.ndarray, y_pm1: np.ndarray, spec: Sequence[tuple[str, tuple[int, ...], int, int]], *, activation: str, loss: str, weight_decay: float, margin: float, device: str, dtype: torch.dtype):
        self.device = torch.device(device)
        self.dtype = dtype
        self.X = torch.tensor(np.asarray(X, dtype=np.float64), device=self.device, dtype=self.dtype)
        self.y = torch.tensor(np.asarray(y_pm1, dtype=np.float64).reshape(-1), device=self.device, dtype=self.dtype)
        self.spec = list(spec)
        self.activation = str(activation)
        self.loss = str(loss)
        self.weight_decay = float(weight_decay)
        self.margin = float(margin)
        self.P = int(self.spec[-1][3])

    def energy0_and_grad(self, theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        theta_np = np.asarray(theta_np, dtype=np.float64).reshape(-1)
        theta = torch.tensor(theta_np, device=self.device, dtype=self.dtype, requires_grad=True)
        logits = forward_3layer(self.X, theta, self.spec, self.activation)
        per_ex = loss_forward(logits, self.y, self.loss, margin=self.margin)
        loss_mean = per_ex.mean()
        prior = 0.5 * self.weight_decay * torch.mean(theta * theta)
        energy = loss_mean + prior
        grad = torch.autograd.grad(energy, theta, create_graph=False, retain_graph=False)[0]
        return float(energy.detach().cpu().item()), grad.detach().cpu().numpy().astype(np.float64)

    def energy0(self, theta_np: np.ndarray) -> float:
        return self.energy0_and_grad(theta_np)[0]

    def predict_logits(self, theta_np: np.ndarray) -> np.ndarray:
        theta = torch.tensor(np.asarray(theta_np, dtype=np.float64).reshape(-1), device=self.device, dtype=self.dtype)
        with torch.no_grad():
            logits = forward_3layer(self.X, theta, self.spec, self.activation)
        return logits.detach().cpu().numpy()

    def classification_error(self, theta_np: np.ndarray) -> float:
        logits = self.predict_logits(theta_np)
        y = self.y.detach().cpu().numpy()
        signed_margin = y * logits
        return float(np.mean(signed_margin <= 0.0))

    def margin_stats(self, theta_np: np.ndarray) -> dict[str, float]:
        logits = self.predict_logits(theta_np)
        y = self.y.detach().cpu().numpy()
        margins = np.where(np.isfinite(y * logits), y * logits, 0.0)
        return {
            "margin_mean": float(np.mean(margins)),
            "margin_min": float(np.min(margins)),
            "margin_p10": float(np.quantile(margins, 0.1)),
            "margin_p50": float(np.quantile(margins, 0.5)),
            "margin_std": float(np.std(margins)),
        }


class ScaledEnergyModel:
    def __init__(self, base: DNNBinaryEnergy0, beta: float):
        self.base = base
        self.beta = float(beta)

    def energy_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        e0, g0 = self.base.energy0_and_grad(theta)
        return self.beta * e0, self.beta * g0


__all__ = ["DNNBinaryEnergy0", "ScaledEnergyModel"]


