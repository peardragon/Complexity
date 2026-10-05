"""Thin, fail-fast adapter to this stage's canonical DNN objective.

The scientific source of truth is the stage-local ``src/utils/shared`` package
and ``config/objective.json``.  The contract records the
base loss and its one-time shell-beta expansion; these vectorized functions
consume only the already-expanded effective coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys

import numpy as np
import torch
import torch.nn.functional as torch_functional

from .config import DEFAULT_PROTOCOL_PATH, LOCAL_SOURCE_ROOT, STAGE_ROOT
from .model import PARAMETER_COUNT, logits, logits_batch


if str(LOCAL_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE_ROOT))
try:
    from utils.objective_contract import ObjectiveContract
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "the domain-local shared.objective is mandatory; there is no fallback"
    ) from exc


def _load_contract() -> ObjectiveContract:
    protocol = json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    objective = protocol["objective"]
    path = (STAGE_ROOT / objective["contract_path"]).resolve()
    contract = ObjectiveContract.from_json(path)
    if contract.objective_id != objective["expected_objective_id"]:
        raise RuntimeError("shared objective_id differs from the synthetic protocol")
    return contract


CONTRACT = _load_contract()
GAMMA_CE = CONTRACT.gamma_ce
LAMBDA_REG = CONTRACT.lambda_reg
OBJECTIVE_ID = CONTRACT.objective_id
OBJECTIVE_FINGERPRINT = CONTRACT.fingerprint


@dataclass(frozen=True)
class NumpyObjectiveEvaluation:
    """One canonical fp64 objective evaluation and its reusable forward output.

    ``detached_logits`` shares storage with the logits used to compute
    ``total_loss`` and ``gradient`` but retains no autograd graph.  This lets
    optimizer callbacks derive strict-margin diagnostics without repeating the
    model forward pass.
    """

    total_loss: float
    gradient: np.ndarray
    detached_logits: torch.Tensor


def ce_mean_batch(theta_batch: torch.Tensor, x: torch.Tensor, y_pm1: torch.Tensor) -> torch.Tensor:
    values = logits_batch(theta_batch, x)
    return torch_functional.softplus(-y_pm1.reshape(1, -1) * values).mean(dim=1)


def l2_per_parameter_batch(theta_batch: torch.Tensor) -> torch.Tensor:
    if theta_batch.ndim == 1:
        theta_batch = theta_batch.unsqueeze(0)
    return CONTRACT.lambda_reg * torch.sum(theta_batch.square(), dim=1) / (2.0 * PARAMETER_COUNT)


def total_loss_batch(
    theta_batch: torch.Tensor,
    x: torch.Tensor,
    y_pm1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = ce_mean_batch(theta_batch, x, y_pm1)
    l2 = l2_per_parameter_batch(theta_batch)
    return CONTRACT.gamma_ce * ce + l2, ce, l2


def total_loss(
    theta: torch.Tensor,
    x: torch.Tensor,
    y_pm1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if theta.ndim != 1:
        raise ValueError("total_loss expects a one-dimensional parameter vector")
    model_logits = logits(theta, x)
    return CONTRACT.torch_terms(model_logits, y_pm1, [theta])


def replay_metrics(theta: torch.Tensor, x: torch.Tensor, y_pm1: torch.Tensor) -> dict[str, float | int | bool]:
    with torch.no_grad():
        prediction = logits(theta, x)
        signed_margin = y_pm1 * prediction
        n_wrong = int(torch.count_nonzero(signed_margin <= 0.0).item())
        total, ce, l2 = CONTRACT.torch_terms(prediction, y_pm1, [theta])
    minimum = float(torch.min(signed_margin).cpu().item())
    return {
        **CONTRACT.metadata(parameter_count=PARAMETER_COUNT),
        "total_loss": float(total.cpu().item()),
        "ce_mean": float(ce.cpu().item()),
        "l2_per_parameter": float(l2.cpu().item()),
        "n_wrong": n_wrong,
        "min_signed_margin": minimum,
        "exact": bool(n_wrong == 0 and minimum > 0.0),
    }


def numpy_objective_evaluation(
    theta_np: np.ndarray,
    x: torch.Tensor,
    y_pm1: torch.Tensor,
) -> NumpyObjectiveEvaluation:
    """Evaluate canonical total loss/gradient once and retain its logits.

    The arithmetic order is the same as :func:`total_loss`: model logits,
    shared-contract terms, then ``autograd.grad``.  The scalar and gradient
    are materialized before the detached forward output is exposed so adding
    callback diagnostics cannot perturb the values returned to SciPy.
    """

    theta = torch.tensor(
        np.asarray(theta_np, dtype=np.float64),
        device=x.device,
        dtype=x.dtype,
        requires_grad=True,
    )
    model_logits = logits(theta, x)
    total, _ce, _l2 = CONTRACT.torch_terms(model_logits, y_pm1, [theta])
    gradient = torch.autograd.grad(total, theta, create_graph=False, retain_graph=False)[0]
    total_value = float(total.detach().cpu().item())
    gradient_value = gradient.detach().cpu().numpy().astype(np.float64)
    detached_logits = model_logits.detach()
    return NumpyObjectiveEvaluation(
        total_loss=total_value,
        gradient=gradient_value,
        detached_logits=detached_logits,
    )


def numpy_total_and_grad(
    theta_np: np.ndarray,
    x: torch.Tensor,
    y_pm1: torch.Tensor,
) -> tuple[float, np.ndarray]:
    evaluation = numpy_objective_evaluation(theta_np, x, y_pm1)
    return evaluation.total_loss, evaluation.gradient
