from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from .batch_energy import ManualTorchBatchEnergyModel
from .energy import DNNBinaryEnergy0
from .model_types import DNNArch, TrainConfig
from .network import flatten_params, forward_3layer, init_3layer_params, loss_forward
from .seed import set_global_seed


def _manual_batch_stub(base: DNNBinaryEnergy0) -> object:
    return type("ManualBatchStub", (), {"base": base, "beta": 1.0})()


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _logit_stats(logits: np.ndarray, y_pm1: np.ndarray) -> dict[str, float]:
    signed_margin = np.asarray(y_pm1, dtype=np.float64).reshape(-1) * np.asarray(logits, dtype=np.float64).reshape(-1)
    wrong = signed_margin <= 0.0
    n_wrong = int(np.sum(wrong))
    cls_err = float(n_wrong / max(1, signed_margin.size))
    return {
        "final_cls_err": float(cls_err),
        "n_wrong": float(n_wrong),
        "final_train_accuracy": float(1.0 - cls_err),
        "min_signed_margin": float(np.min(signed_margin)),
    }


def _batch_forward(manual: ManualTorchBatchEnergyModel, theta_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    params = manual._unpack(theta_batch)
    a1 = torch.einsum("nd,rhd->rnh", manual.X, params["W1"]) + params["b1"].unsqueeze(1)
    h1 = manual._activation(a1)
    a2 = torch.einsum("rnh,rkh->rnk", h1, params["W2"]) + params["b2"].unsqueeze(1)
    h2 = manual._activation(a2)
    logits = torch.einsum("rnk,rk->rn", h2, params["W3"][:, 0, :]) + params["b3"][:, 0].unsqueeze(1)
    loss_mean, g_logits = manual._loss_and_grad_logits(logits)
    gW3 = torch.einsum("rn,rnk->rk", g_logits, h2).unsqueeze(1)
    gb3 = torch.sum(g_logits, dim=1, keepdim=True)
    gh2 = g_logits.unsqueeze(2) * params["W3"][:, 0, :].unsqueeze(1)
    ga2 = gh2 * manual._activation_prime(a2, h2)
    gW2 = torch.einsum("rnk,rnh->rkh", ga2, h1)
    gb2 = torch.sum(ga2, dim=1)
    gh1 = torch.einsum("rnk,rkh->rnh", ga2, params["W2"])
    ga1 = gh1 * manual._activation_prime(a1, h1)
    gW1 = torch.einsum("rnh,nd->rhd", ga1, manual.X)
    gb1 = torch.sum(ga1, dim=1)
    grad = torch.zeros_like(theta_batch)
    for name, g in (("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2), ("W3", gW3), ("b3", gb3)):
        a, b, _shape = manual.name_to_slice[name]
        grad[:, a:b] = g.reshape(theta_batch.shape[0], -1)
    if manual.weight_decay > 0.0:
        prior = 0.5 * manual.weight_decay * torch.mean(theta_batch * theta_batch, dim=1)
        grad = grad + (manual.weight_decay / float(manual.P)) * theta_batch
    else:
        prior = torch.zeros_like(loss_mean)
    energy0 = loss_mean + prior
    return energy0, grad, logits


def _clip_grad_per_row(grad: torch.Tensor, *, max_norm: float) -> torch.Tensor:
    grad_norm = torch.linalg.vector_norm(grad, ord=2, dim=1, keepdim=True)
    scale = torch.clamp(float(max_norm) / torch.clamp(grad_norm, min=1.0e-12), max=1.0)
    return grad * scale


class _ExactStop(Exception):
    pass


def _run_lbfgs_scipy_simple(
    theta_start: np.ndarray,
    base: DNNBinaryEnergy0,
    cfg: TrainConfig,
    y_pm1_np: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    max_iter = int(cfg.lbfgs_max_iter)
    theta0 = np.asarray(theta_start, dtype=np.float64).reshape(-1)
    if max_iter <= 0:
        return theta0, {
            "lbfgs_iters_completed": 0,
            "lbfgs_early_stop_epoch": None,
            "lbfgs_early_stop_reached": False,
        }
    latest = {"theta": theta0.copy()}
    lbfgs_state = {
        "iters": 0,
        "early_stop_epoch": None,
        "early_stop_reached": False,
    }

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        e, g = base.energy0_and_grad(np.asarray(x, dtype=np.float64))
        return float(e), np.asarray(g, dtype=np.float64)

    def callback(xk: np.ndarray) -> None:
        lbfgs_state["iters"] += 1
        latest["theta"] = np.asarray(xk, dtype=np.float64).copy()
        stats = _logit_stats(base.predict_logits(latest["theta"]), y_pm1_np)
        if float(stats["final_cls_err"]) == 0.0:
            lbfgs_state["early_stop_reached"] = True
            lbfgs_state["early_stop_epoch"] = int(lbfgs_state["iters"])
            raise _ExactStop()

    try:
        result = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={"maxiter": max_iter},
        )
        theta_out = np.asarray(result.x if result.x is not None else latest["theta"], dtype=np.float64).reshape(-1)
    except _ExactStop:
        theta_out = np.asarray(latest["theta"], dtype=np.float64).reshape(-1)

    return theta_out, {
        "lbfgs_iters_completed": int(lbfgs_state["iters"]),
        "lbfgs_early_stop_epoch": lbfgs_state["early_stop_epoch"],
        "lbfgs_early_stop_reached": bool(lbfgs_state["early_stop_reached"]),
    }


def _run_lbfgs_torch_simple(
    theta_start: np.ndarray,
    base: DNNBinaryEnergy0,
    cfg: TrainConfig,
    y_pm1_np: np.ndarray,
    *,
    chunk_iter: int = 50,
) -> tuple[np.ndarray, dict[str, Any]]:
    max_iter = int(cfg.lbfgs_max_iter)
    theta0 = np.asarray(theta_start, dtype=np.float64).reshape(-1)
    if max_iter <= 0:
        return theta0, {
            "lbfgs_iters_completed": 0,
            "lbfgs_early_stop_epoch": None,
            "lbfgs_early_stop_reached": False,
        }
    theta = torch.tensor(theta0, device=base.device, dtype=base.dtype, requires_grad=True)
    lbfgs_state = {
        "iters": 0,
        "early_stop_epoch": None,
        "early_stop_reached": False,
    }

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = forward_3layer(base.X, theta, base.spec, base.activation)
        per_ex = loss_forward(logits, base.y, base.loss, margin=base.margin)
        loss_mean = per_ex.mean()
        prior = 0.5 * base.weight_decay * torch.mean(theta * theta)
        energy = loss_mean + prior
        energy.backward()
        return energy

    remaining = max_iter
    while remaining > 0:
        step_iter = min(int(chunk_iter), remaining)
        optimizer = torch.optim.LBFGS(
            [theta],
            lr=1.0,
            max_iter=step_iter,
            max_eval=max(1, step_iter * 2),
            tolerance_grad=1.0e-7,
            tolerance_change=1.0e-9,
            line_search_fn="strong_wolfe",
        )
        try:
            optimizer.step(closure)
        except RuntimeError:
            break
        lbfgs_state["iters"] += int(step_iter)
        remaining -= int(step_iter)
        with torch.no_grad():
            logits = forward_3layer(base.X, theta, base.spec, base.activation)
            stats = _logit_stats(logits.detach().cpu().numpy(), y_pm1_np)
        if float(stats["final_cls_err"]) == 0.0:
            lbfgs_state["early_stop_reached"] = True
            lbfgs_state["early_stop_epoch"] = int(lbfgs_state["iters"])
            break

    theta_out = theta.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return theta_out, {
        "lbfgs_iters_completed": int(lbfgs_state["iters"]),
        "lbfgs_early_stop_epoch": lbfgs_state["early_stop_epoch"],
        "lbfgs_early_stop_reached": bool(lbfgs_state["early_stop_reached"]),
    }


def _build_summary(
    *,
    attempt_id: int,
    theta_final: np.ndarray,
    theta_init: np.ndarray,
    base: DNNBinaryEnergy0,
    arch: DNNArch,
    cfg: TrainConfig,
    optimizer_chain: str,
    adam_epochs_completed: int,
    adam_early_stop_epoch: int | None,
    adam_early_stop_reached: bool,
    lbfgs_info: dict[str, Any],
) -> dict[str, Any]:
    stats = _logit_stats(base.predict_logits(theta_final), base.y.detach().cpu().numpy())
    cls_err = float(stats["final_cls_err"])
    early_phase = None
    early_epoch = None
    if bool(adam_early_stop_reached):
        early_phase = "adam"
        early_epoch = int(adam_early_stop_epoch) if adam_early_stop_epoch is not None else None
    elif bool(lbfgs_info.get("lbfgs_early_stop_reached", False)):
        early_phase = "lbfgs"
        early_epoch = int(lbfgs_info["lbfgs_early_stop_epoch"])
    return {
        "attempt_id": int(attempt_id),
        "arch": {
            "input_dim": int(arch.input_dim),
            "width1": int(arch.width1),
            "width2": int(arch.width2),
        },
        "train_cfg": {
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "momentum": float(cfg.momentum),
            "epochs": int(cfg.epochs),
            "seed": int(cfg.seed),
            "optimizer_name": str(cfg.optimizer_name),
            "lbfgs_max_iter": int(cfg.lbfgs_max_iter),
            "init_scale_multiplier": float(cfg.init_scale_multiplier),
            "activation": str(cfg.activation),
            "loss": str(cfg.loss),
            "margin": float(cfg.margin),
        },
        "optimizer_chain": str(optimizer_chain),
        "final_train_loss": float(base.energy0(theta_final)),
        "final_cls_err": float(cls_err),
        "n_wrong": int(stats["n_wrong"]),
        "final_train_accuracy": float(stats["final_train_accuracy"]),
        "final_train_accuracy_percent": float(100.0 * stats["final_train_accuracy"]),
        "min_signed_margin": float(stats["min_signed_margin"]),
        "is_exact_solution": bool(cls_err == 0.0),
        "sampler_eligible": bool(cls_err == 0.0),
        "P_params": int(theta_final.size),
        "adam_epochs_completed": int(adam_epochs_completed),
        "adam_early_stop_epoch": int(adam_early_stop_epoch) if adam_early_stop_epoch is not None else None,
        "adam_early_stop_reached": bool(adam_early_stop_reached),
        "lbfgs_iters_completed": int(lbfgs_info.get("lbfgs_iters_completed", 0)),
        "lbfgs_early_stop_epoch": lbfgs_info.get("lbfgs_early_stop_epoch"),
        "lbfgs_early_stop_reached": bool(lbfgs_info.get("lbfgs_early_stop_reached", False)),
        "early_stop_phase": early_phase,
        "early_stop_epoch": early_epoch,
        "theta_norm": float(np.linalg.norm(theta_final)),
        "theta_init_norm": float(np.linalg.norm(theta_init)),
    }


def _run_adam_batched_simple(
    theta_init_batch: np.ndarray,
    base: DNNBinaryEnergy0,
    cfgs: Sequence[TrainConfig],
    *,
    verbose: bool = False,
    progress_label: str = "",
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if not cfgs:
        return [], []
    manual = ManualTorchBatchEnergyModel(_manual_batch_stub(base), sampling_device=str(base.device), sampling_dtype="float64")
    theta_batch = torch.tensor(np.asarray(theta_init_batch, dtype=np.float64), device=manual.device, dtype=manual.dtype)
    m = torch.zeros_like(theta_batch)
    v = torch.zeros_like(theta_batch)
    step_count = torch.zeros((theta_batch.shape[0], 1), device=manual.device, dtype=manual.dtype)
    lr = torch.tensor([float(cfg.lr) for cfg in cfgs], device=manual.device, dtype=manual.dtype).unsqueeze(1)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    total_epochs = int(cfgs[0].epochs)
    milestones = {max(1, int(total_epochs * 0.60)), max(1, int(total_epochs * 0.85))}
    active = torch.ones((theta_batch.shape[0], 1), device=manual.device, dtype=torch.bool)
    y_pm1_np = base.y.detach().cpu().numpy()
    state_rows = [
        {
            "adam_epochs_completed": 0,
            "adam_early_stop_epoch": None,
            "adam_early_stop_reached": False,
        }
        for _ in cfgs
    ]
    log_every = max(1, total_epochs // 10)

    with torch.no_grad():
        for epoch_idx in range(1, total_epochs + 1):
            loss_mean, grad, logits = _batch_forward(manual, theta_batch)
            grad = _clip_grad_per_row(grad, max_norm=1.0e4)
            active_f = active.to(dtype=manual.dtype)
            step_count = step_count + active_f
            m = m * beta1 + (1.0 - beta1) * grad * active_f
            v = v * beta2 + (1.0 - beta2) * grad * grad * active_f
            step_safe = torch.clamp(step_count, min=1.0)
            m_hat = m / (1.0 - beta1**step_safe)
            v_hat = v / (1.0 - beta2**step_safe)
            theta_batch = theta_batch - lr * m_hat / (torch.sqrt(v_hat) + eps) * active_f
            if epoch_idx in milestones:
                lr = lr * 0.3
            logits_np = logits.detach().cpu().numpy()
            newly_solved = 0
            for idx in range(theta_batch.shape[0]):
                if bool(active[idx, 0]):
                    state_rows[idx]["adam_epochs_completed"] = int(epoch_idx)
                    stats = _logit_stats(logits_np[idx], y_pm1_np)
                    if float(stats["final_cls_err"]) == 0.0:
                        state_rows[idx]["adam_early_stop_epoch"] = int(epoch_idx)
                        state_rows[idx]["adam_early_stop_reached"] = True
                        active[idx, 0] = False
                        newly_solved += 1
            if verbose and (epoch_idx == 1 or epoch_idx % log_every == 0 or epoch_idx == total_epochs or newly_solved > 0):
                active_count = int(torch.sum(active).item())
                exact_count = int(sum(1 for row in state_rows if bool(row["adam_early_stop_reached"])))
                best_acc = float(np.max([100.0 * (1.0 - _logit_stats(logits_np[idx], y_pm1_np)["final_cls_err"]) for idx in range(theta_batch.shape[0])]))
                _log(verbose, f"[simple_reference_search] {progress_label} adam {epoch_idx}/{total_epochs} active={active_count} exact={exact_count}/{theta_batch.shape[0]} best_acc={best_acc:.2f}%")
            if not bool(torch.any(active)):
                break
    theta_rows = theta_batch.detach().cpu().numpy().astype(np.float64)
    return [theta_rows[idx].copy() for idx in range(theta_rows.shape[0])], state_rows


def train_reference_solutions_simple_batched(
    X: np.ndarray,
    y_pm1: np.ndarray,
    arch: DNNArch,
    cfgs: Sequence[TrainConfig],
    *,
    device: str,
    verbose: bool = False,
    progress_label: str = "",
) -> list[dict[str, Any]]:
    if not cfgs:
        return []
    set_global_seed(int(cfgs[0].seed))
    theta_init_rows: list[np.ndarray] = []
    spec = None
    for cfg in cfgs:
        rng = np.random.default_rng(int(cfg.seed))
        params = init_3layer_params(arch, rng, init_scale_multiplier=float(cfg.init_scale_multiplier))
        theta_init, spec_local = flatten_params(params)
        theta_init_rows.append(theta_init.astype(np.float64))
        if spec is None:
            spec = spec_local
    assert spec is not None
    base = DNNBinaryEnergy0(
        X=X,
        y_pm1=y_pm1,
        spec=spec,
        activation=str(cfgs[0].activation),
        loss=str(cfgs[0].loss),
        weight_decay=float(cfgs[0].weight_decay),
        margin=float(cfgs[0].margin),
        device=device,
        dtype=torch.float64,
    )
    base_cpu = DNNBinaryEnergy0(
        X=X,
        y_pm1=y_pm1,
        spec=spec,
        activation=str(cfgs[0].activation),
        loss=str(cfgs[0].loss),
        weight_decay=float(cfgs[0].weight_decay),
        margin=float(cfgs[0].margin),
        device="cpu",
        dtype=torch.float64,
    )
    theta_after_adam, adam_state_rows = _run_adam_batched_simple(
        np.asarray(theta_init_rows, dtype=np.float64),
        base,
        cfgs,
        verbose=verbose,
        progress_label=progress_label,
    )
    y_pm1_np = np.asarray(y_pm1, dtype=np.float64).reshape(-1)
    outputs: list[dict[str, Any]] = []
    for idx, cfg in enumerate(cfgs):
        theta_init = theta_init_rows[idx].copy()
        theta_final = theta_after_adam[idx].copy()
        adam_state = adam_state_rows[idx]
        if bool(adam_state["adam_early_stop_reached"]):
            optimizer_chain = "adam_early_stop_exact"
            lbfgs_info = {
                "lbfgs_iters_completed": 0,
                "lbfgs_early_stop_epoch": None,
                "lbfgs_early_stop_reached": False,
            }
        else:
            if base.device.type == "cuda":
                theta_final, lbfgs_info = _run_lbfgs_torch_simple(theta_final, base, cfg, y_pm1_np)
                optimizer_chain = "adam_then_torch_cuda_lbfgs"
            else:
                theta_final, lbfgs_info = _run_lbfgs_scipy_simple(theta_final, base_cpu, cfg, y_pm1_np)
                optimizer_chain = "adam_then_scipy_lbfgs"
        summary = _build_summary(
            attempt_id=idx,
            theta_final=theta_final,
            theta_init=theta_init,
            base=base_cpu,
            arch=arch,
            cfg=cfg,
            optimizer_chain=optimizer_chain,
            adam_epochs_completed=int(adam_state["adam_epochs_completed"]),
            adam_early_stop_epoch=adam_state["adam_early_stop_epoch"],
            adam_early_stop_reached=bool(adam_state["adam_early_stop_reached"]),
            lbfgs_info=lbfgs_info,
        )
        if verbose:
            _log(verbose, f"[simple_reference_search] {progress_label} attempt={idx + 1}/{len(cfgs)} acc={float(summary['final_train_accuracy_percent']):.2f}% cls_err={float(summary['final_cls_err']):.4f} early={summary['early_stop_phase'] or 'none'}")
        outputs.append(
            {
                "attempt_id": int(idx),
                "theta": theta_final.copy(),
                "theta_init": theta_init.copy(),
                "summary": summary,
            }
        )
    return outputs


__all__ = ["train_reference_solutions_simple_batched"]
