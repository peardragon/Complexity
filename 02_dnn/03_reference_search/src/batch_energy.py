from __future__ import annotations

import torch
import torch.nn.functional as F


def sampling_torch_dtype(dtype_name: str) -> torch.dtype:
    if str(dtype_name).lower() in ("float32", "fp32"):
        return torch.float32
    if str(dtype_name).lower() in ("float64", "fp64", "double"):
        return torch.float64
    raise ValueError(f"Unsupported sampling dtype: {dtype_name!r}")


class ManualTorchBatchEnergyModel:
    def __init__(self, model, *, sampling_device: str, sampling_dtype: str):
        base = model.base
        self.device = torch.device(str(sampling_device).lower())
        self.dtype = sampling_torch_dtype(sampling_dtype)
        self.activation_name = str(base.activation).lower()
        self.loss_name = str(base.loss).lower()
        self.weight_decay = float(base.weight_decay)
        self.global_beta = float(getattr(model, "beta", 1.0))
        self.X = base.X.detach().to(device=self.device, dtype=self.dtype)
        self.y = base.y.detach().to(device=self.device, dtype=self.dtype)
        self.spec = list(base.spec)
        self.P = int(self.spec[-1][3])
        self.name_to_slice = {name: (int(a), int(b), tuple(shape)) for name, shape, a, b in self.spec}

    def _activation(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "softplus":
            return F.softplus(z)
        if self.activation_name == "tanh":
            return torch.tanh(z)
        raise ValueError(f"Unsupported activation: {self.activation_name!r}")

    def _activation_prime(self, z: torch.Tensor, activated: torch.Tensor | None = None) -> torch.Tensor:
        if self.activation_name == "softplus":
            return torch.sigmoid(z)
        if self.activation_name == "tanh":
            out = torch.tanh(z) if activated is None else activated
            return 1.0 - out * out
        raise ValueError(f"Unsupported activation: {self.activation_name!r}")

    def _loss_and_grad_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.y.unsqueeze(0)
        n = float(self.y.numel())
        yz = y * logits
        if self.loss_name == "logistic":
            loss_mean = F.softplus(-yz).mean(dim=1)
            grad_logits = (-y * torch.sigmoid(-yz)) / n
            return loss_mean, grad_logits
        raise ValueError(f"Unsupported loss: {self.loss_name!r}")

    def _unpack(self, theta_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for name, shape, a, b in self.spec:
            out[name] = theta_batch[:, a:b].reshape((theta_batch.shape[0],) + tuple(shape))
        return out
