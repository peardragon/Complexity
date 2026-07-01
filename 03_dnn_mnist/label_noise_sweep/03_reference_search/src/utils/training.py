from __future__ import annotations

from typing import Any

import numpy as np

from .mnist10_model import ARCH, P, init_theta, normalize_labels


def torch_logits_batch(theta: Any, x_t: Any) -> Any:
    import torch

    h = ARCH.hidden_width
    d = ARCH.input_dim
    idx = 0
    w1 = theta[:, idx : idx + d * h].reshape(theta.shape[0], h, d)
    idx += d * h
    b1 = theta[:, idx : idx + h]
    idx += h
    w2 = theta[:, idx : idx + h * h].reshape(theta.shape[0], h, h)
    idx += h * h
    b2 = theta[:, idx : idx + h]
    idx += h
    w3 = theta[:, idx : idx + h].reshape(theta.shape[0], 1, h)
    idx += h
    b3 = theta[:, idx : idx + 1].reshape(theta.shape[0])
    h1 = torch.tanh(torch.einsum("nd,bhd->bnh", x_t, w1) + b1[:, None, :])
    h2 = torch.tanh(torch.einsum("bnh,bkh->bnk", h1, w2) + b2[:, None, :])
    return torch.einsum("bnh,bh->bn", h2, w3[:, 0, :]) + b3[:, None]


def train_attempt_batch(
    x: np.ndarray,
    y: np.ndarray,
    seeds: list[int],
    *,
    max_epochs: int = 4200,
    lr: float = 0.022,
    device: str = "auto",
) -> list[dict[str, Any]]:
    import torch

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    dtype = torch.float32
    theta0 = np.vstack(
        [init_theta(seed, scale_multiplier=1.0 + 0.5 * ((seed % 5) / 4.0)) for seed in seeds]
    ).astype(np.float32)
    theta = torch.tensor(theta0, device=torch_device, dtype=dtype, requires_grad=True)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=torch_device, dtype=dtype)
    y_t = torch.as_tensor(normalize_labels(y).astype(np.float32), device=torch_device, dtype=dtype)
    opt = torch.optim.Adam([theta], lr=float(lr))
    solved: dict[int, dict[str, Any]] = {}
    best: dict[int, tuple[float, np.ndarray, float]] = {}
    milestones = {int(max_epochs * 0.55), int(max_epochs * 0.80)}
    for epoch in range(1, int(max_epochs) + 1):
        opt.zero_grad(set_to_none=True)
        logits = torch_logits_batch(theta, x_t)
        yz = logits * y_t[None, :]
        loss_rows = torch.nn.functional.softplus(-yz).mean(dim=1)
        loss = loss_rows.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 1000.0)
        opt.step()
        if epoch in milestones:
            for group in opt.param_groups:
                group["lr"] *= 0.35
        if epoch == 1 or epoch % 25 == 0 or epoch == max_epochs:
            with torch.no_grad():
                logits_eval = torch_logits_batch(theta, x_t)
                yz_eval = logits_eval * y_t[None, :]
                err = torch.mean((yz_eval <= 0.0).to(dtype), dim=1).detach().cpu().numpy()
                ce = torch.nn.functional.softplus(-yz_eval).mean(dim=1).detach().cpu().numpy()
                theta_np = theta.detach().cpu().numpy().astype(np.float64)
                for idx, seed in enumerate(seeds):
                    if seed not in best or float(err[idx]) < best[seed][0]:
                        best[seed] = (float(err[idx]), theta_np[idx].copy(), float(ce[idx]))
                    if float(err[idx]) == 0.0 and seed not in solved:
                        solved[seed] = {
                            "seed": seed,
                            "theta": theta_np[idx].copy(),
                            "train_error": 0.0,
                            "ce_mean_train": float(ce[idx]),
                            "epoch": int(epoch),
                            "phase": "adam",
                        }
            if len(solved) == len(seeds):
                break
    out: list[dict[str, Any]] = []
    for seed in seeds:
        if seed in solved:
            out.append(solved[seed])
        else:
            err, theta_best, ce = best[seed]
            out.append(
                {
                    "seed": seed,
                    "theta": theta_best,
                    "train_error": float(err),
                    "ce_mean_train": float(ce),
                    "epoch": int(max_epochs),
                    "phase": "adam_best",
                }
            )
    return out


def select_reference(selected: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
    theta = np.asarray(candidate["theta"], dtype=np.float64).reshape(-1)
    if theta.size != P or float(candidate["train_error"]) != 0.0:
        return False
    for row in selected:
        other = np.asarray(row["theta"], dtype=np.float64).reshape(-1)
        if float(np.linalg.norm(theta - other)) <= 1.0e-6:
            return False
    selected.append(candidate)
    return True

