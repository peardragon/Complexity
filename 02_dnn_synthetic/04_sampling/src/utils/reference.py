from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
import torch

from .config import (
    SYNTHETIC_ROOT,
    canonical_json_bytes,
    stage_config_fingerprint,
    stage_source_fingerprint,
)
from .fingerprints import reference_seed_projection, reference_task_projection, strict_fingerprints
from .io import atomic_json, atomic_npz, sha256_array, sha256_file
from .manifests import ReferenceTask, dataset_metadata_path, dataset_path, iter_reference_tasks, reference_metadata_path, reference_pack_path
from .model import PARAMETER_COUNT, initialize_theta, logits, logits_batch
from .objective import OBJECTIVE_FINGERPRINT, OBJECTIVE_ID, numpy_objective_evaluation, replay_metrics, total_loss_batch
from .seeds import reference_seed
from .runtime import runtime_fingerprint, runtime_signature


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float64", "double", "fp64"}:
        return torch.float64
    if normalized in {"float32", "float", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _load_dataset(config: dict, task: Any, synthetic_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    payload_path = dataset_path(synthetic_root, task)
    metadata_path = dataset_metadata_path(synthetic_root, task)
    with np.load(payload_path, allow_pickle=False) as payload:
        x_raw = np.asarray(payload["X_raw"], dtype=np.float64)
        x = np.asarray(payload["X_train"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64).reshape(-1)
    if x.shape != (512, 2) or y.shape != (512,) or not np.all(np.isfinite(x)):
        raise ValueError(f"invalid dataset dimensions or values: {payload_path}")
    if set(np.unique(y)) != {-1, 1} or np.count_nonzero(y == 1) != 256:
        raise ValueError(f"invalid balanced binary labels: {payload_path}")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if "payload_sha256" not in metadata:
        metadata["payload_sha256"] = sha256_array(x_raw, x, y.astype(np.int8))
    return x, y, metadata


def _strict_exact_from_logits(
    model_logits: torch.Tensor,
    y: torch.Tensor,
) -> tuple[bool, int, float]:
    with torch.no_grad():
        margins = y * model_logits
        wrong = int(torch.count_nonzero(margins <= 0.0).item())
        minimum = float(torch.min(margins).cpu().item())
    return bool(wrong == 0 and minimum > 0.0), wrong, minimum


def _post_update_exact(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[bool, int, float]:
    with torch.no_grad():
        model_logits = logits(theta, x)
    return _strict_exact_from_logits(model_logits, y)


def _matching_parameter_vector(current: np.ndarray, cached: np.ndarray | None) -> bool:
    """Require an exact shape/dtype/value match before reusing cached logits."""

    if cached is None:
        return False
    current_array = np.asarray(current)
    cached_array = np.asarray(cached)
    return bool(
        current_array.shape == cached_array.shape
        and current_array.dtype == cached_array.dtype
        and np.array_equal(current_array, cached_array)
    )


def _batched_adam_phase(
    theta_initial: np.ndarray,
    x: torch.Tensor,
    y: torch.Tensor,
    settings: dict,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    initial = np.asarray(theta_initial, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != PARAMETER_COUNT:
        raise ValueError(f"batched Adam expects shape (attempts,{PARAMETER_COUNT})")
    theta = torch.tensor(initial, device=x.device, dtype=x.dtype, requires_grad=True)
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    maximum = int(settings["max_epochs"])
    minimum = int(settings["min_epochs"])
    patience = int(settings["patience"])
    tolerance = float(settings["relative_improvement_tolerance"])
    clip_norm = float(settings["gradient_clip_norm"])
    beta1 = float(settings["beta1"])
    beta2 = float(settings["beta2"])
    epsilon = float(settings["epsilon"])
    count = int(initial.shape[0])
    best = torch.full((count,), float("inf"), device=x.device, dtype=x.dtype)
    stale = torch.zeros(count, device=x.device, dtype=torch.int64)
    active = torch.ones(count, device=x.device, dtype=torch.bool)
    first_exact = torch.full((count,), -1, device=x.device, dtype=torch.int64)
    completed = torch.zeros(count, device=x.device, dtype=torch.int64)
    last_gradient_norm = torch.full((count,), float("inf"), device=x.device, dtype=x.dtype)
    stop_reasons = ["max_epochs"] * count
    for epoch in range(1, maximum + 1):
        if not bool(torch.any(active)):
            break
        total, _ce, _l2 = total_loss_batch(theta, x, y)
        if not bool(torch.all(torch.isfinite(total))):
            raise FloatingPointError(f"non-finite Adam objective at epoch {epoch}")
        gradient = torch.autograd.grad(torch.sum(total[active]), theta, create_graph=False, retain_graph=False)[0]
        with torch.no_grad():
            row_norm = torch.linalg.vector_norm(gradient, dim=1)
            last_gradient_norm[active] = row_norm[active]
            scale = torch.clamp(clip_norm / torch.clamp(row_norm, min=1.0e-30), max=1.0)
            gradient = gradient * scale[:, None]
            active_float = active[:, None].to(theta.dtype)
            first_moment = torch.where(
                active[:, None],
                beta1 * first_moment + (1.0 - beta1) * gradient,
                first_moment,
            )
            second_moment = torch.where(
                active[:, None],
                beta2 * second_moment + (1.0 - beta2) * gradient.square(),
                second_moment,
            )
            corrected_first = first_moment / (1.0 - beta1**epoch)
            corrected_second = second_moment / (1.0 - beta2**epoch)
            theta -= (
                float(settings["learning_rate"])
                * corrected_first
                / (torch.sqrt(corrected_second) + epsilon)
                * active_float
            )
            completed[active] = epoch

            post_total, _post_ce, _post_l2 = total_loss_batch(theta, x, y)
            post_logits = logits_batch(theta, x)
            margins = y[None, :] * post_logits
            exact = torch.all(margins > 0.0, dim=1)
            newly_exact = active & exact & (first_exact < 0)
            first_exact[newly_exact] = epoch
            finite_best = torch.isfinite(best)
            threshold = tolerance * torch.maximum(torch.ones_like(best), torch.where(finite_best, torch.abs(best), 0.0))
            improved = active & (~finite_best | (post_total < best - threshold))
            best[improved] = post_total[improved]
            stale[improved] = 0
            stale[active & ~improved] += 1
            newly_stopped = active & ~newly_exact & (epoch >= minimum) & (stale >= patience)
            if bool(torch.any(newly_exact)):
                for row_id in torch.nonzero(newly_exact).reshape(-1).cpu().tolist():
                    stop_reasons[int(row_id)] = "first_exact_post_update"
                active[newly_exact] = False
            if bool(torch.any(newly_stopped)):
                for row_id in torch.nonzero(newly_stopped).reshape(-1).cpu().tolist():
                    stop_reasons[int(row_id)] = "objective_patience"
                active[newly_stopped] = False
        theta.requires_grad_(True)
    summaries: list[dict[str, Any]] = []
    best_cpu = best.detach().cpu().numpy()
    completed_cpu = completed.cpu().numpy()
    first_exact_cpu = first_exact.cpu().numpy()
    gradient_cpu = last_gradient_norm.cpu().numpy()
    for row_id in range(count):
        summaries.append(
            {
                "epochs_completed": int(completed_cpu[row_id]),
                "stop_reason": stop_reasons[row_id],
                "best_total_loss": float(best_cpu[row_id]),
                "last_gradient_norm_before_clip": float(gradient_cpu[row_id]),
                "first_exact_epoch_post_update": (
                    int(first_exact_cpu[row_id]) if int(first_exact_cpu[row_id]) >= 0 else None
                ),
            }
        )
    return theta.detach().cpu().numpy().astype(np.float64), summaries


def _adam_phase(
    theta_init: np.ndarray,
    x: torch.Tensor,
    y: torch.Tensor,
    settings: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    values, summaries = _batched_adam_phase(np.asarray(theta_init)[None, :], x, y, settings)
    return values[0], summaries[0]


def _lbfgs_phase(
    theta_start: np.ndarray,
    x: torch.Tensor,
    y: torch.Tensor,
    settings: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    class _FirstExactStop(RuntimeError):
        pass

    state: dict[str, Any] = {
        "iterations_completed": 0,
        "first_exact_iteration_post_update": None,
        "first_exact_theta": None,
        "callback_exact_cache_hits": 0,
        "callback_exact_fallbacks": 0,
    }
    objective_cache: dict[str, np.ndarray | torch.Tensor | None] = {
        "x": None,
        "detached_logits": None,
    }

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        evaluation = numpy_objective_evaluation(value, x, y)
        objective_cache["x"] = np.asarray(value).copy()
        objective_cache["detached_logits"] = evaluation.detached_logits
        return evaluation.total_loss, evaluation.gradient

    def callback(value: np.ndarray) -> None:
        state["iterations_completed"] = int(state["iterations_completed"]) + 1
        cached_x = objective_cache["x"]
        cached_logits = objective_cache["detached_logits"]
        if (
            isinstance(cached_x, np.ndarray)
            and isinstance(cached_logits, torch.Tensor)
            and _matching_parameter_vector(value, cached_x)
        ):
            state["callback_exact_cache_hits"] = int(state["callback_exact_cache_hits"]) + 1
            exact, _wrong, _minimum = _strict_exact_from_logits(cached_logits, y)
        else:
            state["callback_exact_fallbacks"] = int(state["callback_exact_fallbacks"]) + 1
            theta = torch.tensor(value, device=x.device, dtype=x.dtype)
            exact, _wrong, _minimum = _post_update_exact(theta, x, y)
        if exact and state["first_exact_iteration_post_update"] is None:
            state["first_exact_iteration_post_update"] = int(state["iterations_completed"])
            state["first_exact_theta"] = np.asarray(value, dtype=np.float64).copy()
            raise _FirstExactStop

    try:
        result = minimize(
            objective,
            np.asarray(theta_start, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": int(settings["max_iterations"]),
                "maxls": int(settings["max_line_search_iterations"]),
                "gtol": float(settings["gradient_tolerance"]),
                "ftol": float(settings["function_tolerance"]),
                "maxcor": int(settings["history_size"]),
            },
        )
        theta_final = np.asarray(result.x, dtype=np.float64)
        solver_success = bool(result.success)
        solver_status = int(result.status)
        solver_message = str(result.message)
        function_evaluations = int(result.nfev)
        stopped_on_first_exact = False
    except _FirstExactStop:
        if state["first_exact_theta"] is None:
            raise RuntimeError("L-BFGS first-exact stop did not retain its endpoint")
        theta_final = np.asarray(state["first_exact_theta"], dtype=np.float64)
        solver_success = False
        solver_status = -100
        solver_message = "stopped_at_first_exact_post_update"
        function_evaluations = None
        stopped_on_first_exact = True
    final_value, final_gradient = objective(theta_final)
    gradient_norm = float(np.linalg.norm(final_gradient))
    gradient_threshold_met = gradient_norm <= float(settings["diagnostic_gradient_norm_threshold"])
    return theta_final, {
        "iterations_completed": int(state["iterations_completed"]),
        "first_exact_iteration_post_update": state["first_exact_iteration_post_update"],
        "callback_exact_cache_hits": int(state["callback_exact_cache_hits"]),
        "callback_exact_fallbacks": int(state["callback_exact_fallbacks"]),
        "stopped_on_first_exact": stopped_on_first_exact,
        "solver_success": solver_success,
        "solver_status": solver_status,
        "solver_message": solver_message,
        "function_evaluations": function_evaluations,
        "final_total_loss": final_value,
        "final_gradient_norm": gradient_norm,
        "diagnostic_gradient_threshold": float(settings["diagnostic_gradient_norm_threshold"]),
        "diagnostic_gradient_threshold_met": gradient_threshold_met,
        "diagnostic_converged": bool(solver_success or gradient_threshold_met),
    }


def _finite_replay(theta: np.ndarray, metrics: dict[str, Any]) -> bool:
    scalar_keys = ("total_loss", "ce_mean", "l2_per_parameter", "min_signed_margin")
    return bool(
        np.all(np.isfinite(np.asarray(theta, dtype=np.float64)))
        and all(math.isfinite(float(metrics[key])) for key in scalar_keys)
    )


def _replay_consistent(
    metrics: dict[str, Any],
    *,
    optimizer_total_loss: float,
    absolute_tolerance: float,
) -> bool:
    return bool(
        math.isfinite(float(optimizer_total_loss))
        and abs(float(metrics["total_loss"]) - float(optimizer_total_loss))
        <= float(absolute_tolerance)
    )


def _attempt_counters(
    attempt_summaries: list[dict[str, Any]],
    selected_attempt_ids: list[int],
) -> dict[str, int]:
    return {
        "counted_attempts": len(attempt_summaries),
        "adam_first_exact": sum(
            bool(summary.get("adam_endpoint", {}).get("exact"))
            for summary in attempt_summaries
        ),
        "lbfgs_runs_for_adam_unresolved": sum(
            bool(summary.get("lbfgs", {}).get("run"))
            for summary in attempt_summaries
        ),
        "lbfgs_first_exact": sum(
            summary.get("first_exact_diagnostic", {}).get("phase") == "lbfgs"
            for summary in attempt_summaries
            if isinstance(summary.get("first_exact_diagnostic"), dict)
        ),
        "raw_final_exact": sum(
            bool(summary.get("final", {}).get("exact"))
            for summary in attempt_summaries
        ),
        "finite_final_endpoint": sum(
            bool(summary.get("finite_final_endpoint"))
            for summary in attempt_summaries
        ),
        "replay_consistent": sum(
            bool(summary.get("replay_consistent"))
            for summary in attempt_summaries
        ),
        "diagnostic_converged": sum(
            bool(summary.get("lbfgs", {}).get("diagnostic_converged"))
            for summary in attempt_summaries
        ),
        "duplicate_exact_endpoint": sum(
            bool(summary.get("final", {}).get("exact"))
            and bool(summary.get("duplicate_selected_endpoint"))
            for summary in attempt_summaries
        ),
        "eligible_unique_exact": sum(
            bool(summary.get("eligible_exact_endpoint"))
            for summary in attempt_summaries
        ),
        "selected": len(selected_attempt_ids),
    }


def run_attempt(
    config: dict,
    task: ReferenceTask,
    attempt_id: int,
    x_np: np.ndarray,
    y_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    settings = config["reference_search"]
    seed = reference_seed(config, task.beta_index, task.dataset_id, int(attempt_id))
    initialization = settings["initialization"]
    theta_init = initialize_theta(
        seed,
        base_scale=float(initialization["base_scale"]),
        multiplier=float(initialization["weight_scale_multiplier"]),
    )
    device_name = str(settings["device"])
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("reference config requires CUDA, but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    dtype = _torch_dtype(settings["dtype"])
    x = torch.tensor(x_np, device=device, dtype=dtype)
    y = torch.tensor(y_np, device=device, dtype=dtype)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False

    theta_adam, adam = _adam_phase(theta_init, x, y, settings["adam"])
    adam_tensor = torch.tensor(theta_adam, device=device, dtype=dtype)
    adam_metrics = replay_metrics(adam_tensor, x, y)
    if bool(adam_metrics["exact"]):
        theta_final = theta_adam
        lbfgs = {
            "run": False,
            "reason": "adam_first_exact_endpoint",
            "diagnostic_converged": None,
        }
        optimizer_total_loss = float(adam_metrics["total_loss"])
    else:
        theta_final, lbfgs_result = _lbfgs_phase(theta_adam, x, y, settings["lbfgs"])
        lbfgs = {"run": True, **lbfgs_result}
        optimizer_total_loss = float(lbfgs_result["final_total_loss"])
    final_tensor = torch.tensor(theta_final, device=device, dtype=dtype)
    metrics = replay_metrics(final_tensor, x, y)
    first_exact = None
    if adam["first_exact_epoch_post_update"] is not None:
        first_exact = {"phase": "adam", "step": int(adam["first_exact_epoch_post_update"])}
    elif lbfgs.get("first_exact_iteration_post_update") is not None:
        first_exact = {"phase": "lbfgs", "step": int(lbfgs["first_exact_iteration_post_update"])}
    finite = _finite_replay(theta_final, metrics)
    replay_consistent = _replay_consistent(
        metrics,
        optimizer_total_loss=optimizer_total_loss,
        absolute_tolerance=float(settings["replay_scalar_abs_tolerance"]),
    )
    eligible = bool(metrics["exact"]) and finite and replay_consistent
    summary = {
        "attempt_id": int(attempt_id),
        "seed": int(seed),
        "initialization": initialization["name"],
        "optimizer_chain": "fresh_random_init_then_adam_first_exact_else_lbfgs_first_exact_same_canonical_objective",
        "objective_id": OBJECTIVE_ID,
        "objective_fingerprint": OBJECTIVE_FINGERPRINT,
        "first_exact_diagnostic": first_exact,
        "adam": adam,
        "adam_endpoint": adam_metrics,
        "lbfgs": lbfgs,
        "final": metrics,
        "finite_final_endpoint": finite,
        "replay_consistent": replay_consistent,
        "convergence_is_diagnostic_only": True,
        "eligible_exact_endpoint": eligible,
        "theta_final_sha256": sha256_array(theta_final),
    }
    return theta_init, theta_final, summary


def _expected_metadata(
    config: dict,
    task: ReferenceTask,
    dataset_metadata: dict,
    pack_payload_hash: str,
) -> dict[str, Any]:
    optimizer_projection = config["reference_search"]
    fingerprints = strict_fingerprints(
        config,
        dataset_fingerprint=str(dataset_metadata["payload_sha256"]),
        reference_fingerprint=pack_payload_hash,
        task=reference_task_projection(task),
        seed=reference_seed_projection(config, task),
    )
    return {
        "schema_version": 1,
        "record_type": "synthetic_reference_pack",
        "protocol_id": config["protocol_id"],
        "protocol_sha256": stage_config_fingerprint(config, "reference"),
        "source_sha256": stage_source_fingerprint("reference"),
        "reference_stage_config_sha256": stage_config_fingerprint(config, "reference"),
        "reference_stage_source_sha256": stage_source_fingerprint("reference"),
        "runtime_fingerprint": runtime_fingerprint(),
        "runtime_signature": runtime_signature(),
        "objective_id": OBJECTIVE_ID,
        "objective_fingerprint": OBJECTIVE_FINGERPRINT,
        "beta_index": int(task.beta_index),
        "data_beta": float(task.beta),
        "dataset_id": int(task.dataset_id),
        "dataset_payload_sha256": dataset_metadata["payload_sha256"],
        "optimizer_config_sha256": __import__("hashlib").sha256(canonical_json_bytes(optimizer_projection)).hexdigest(),
        "selected_reference_count": int(config["reference_search"]["references_per_dataset"]),
        "selection_policy": "first_exact_attempt_order",
        "pack_payload_sha256": pack_payload_hash,
        "fingerprints": fingerprints,
    }


def validate_attempt_bookkeeping(
    attempt_summaries: list[dict[str, Any]],
    selected_attempt_ids: list[int],
    *,
    target: int,
) -> None:
    observed_ids = [int(summary["attempt_id"]) for summary in attempt_summaries]
    if observed_ids != list(range(len(observed_ids))):
        raise RuntimeError("counted reference attempts must be contiguous in attempt-id order")
    if selected_attempt_ids != sorted(selected_attempt_ids):
        raise RuntimeError("selected references are not in deterministic attempt-id order")
    if len(selected_attempt_ids) != int(target) or not set(selected_attempt_ids).issubset(observed_ids):
        raise RuntimeError("selected reference IDs are not a target-sized subset of counted attempts")
    expected_chain = (
        "fresh_random_init_then_batched_adam_first_exact_else_"
        "lbfgs_first_exact_same_canonical_objective"
    )
    for summary in attempt_summaries:
        if summary.get("optimizer_chain") != expected_chain:
            raise RuntimeError("counted attempt has the wrong canonical first-exact optimizer chain")
        adam_exact = bool(summary.get("adam_endpoint", {}).get("exact"))
        lbfgs_run = bool(summary.get("lbfgs", {}).get("run"))
        if adam_exact == lbfgs_run:
            raise RuntimeError(
                "L-BFGS must run exactly for Adam-unresolved attempts, never for an Adam-exact endpoint"
            )
    selected_summaries = [
        summary for summary in attempt_summaries if summary.get("selected_ref_id") is not None
    ]
    if [int(summary["selected_ref_id"]) for summary in selected_summaries] != list(range(int(target))):
        raise RuntimeError("selected ref IDs must be assigned consecutively in attempt order")
    if not all(
        bool(summary.get("eligible_exact_endpoint"))
        and bool(summary.get("final", {}).get("exact"))
        and int(summary.get("final", {}).get("n_wrong", -1)) == 0
        and float(summary.get("final", {}).get("min_signed_margin", 0.0)) > 0.0
        and bool(summary.get("finite_final_endpoint"))
        and bool(summary.get("replay_consistent"))
        for summary in selected_summaries
    ):
        raise RuntimeError("a selected reference is not a finite replay-consistent exact endpoint")


def search_dataset(
    config: dict,
    task: ReferenceTask,
    *,
    device: str | None = None,
    synthetic_root: Path = SYNTHETIC_ROOT,
) -> str:
    output_path = reference_pack_path(synthetic_root, task)
    metadata_path = reference_metadata_path(synthetic_root, task)
    if output_path.is_file():
        return "skipped_existing"
    reference_count = int(config["reference_search"]["references_per_dataset"])
    x, y, dataset_metadata = _load_dataset(config, task, synthetic_root)
    settings = dict(config["reference_search"])
    settings["device"] = str(device or settings["device"])

    target = int(config["reference_search"]["references_per_dataset"])
    maximum = int(config["reference_search"]["max_attempts_per_dataset"])
    selected_initial: list[np.ndarray] = []
    selected_final: list[np.ndarray] = []
    selected_attempts: list[int] = []
    attempt_summaries: list[dict[str, Any]] = []
    prefetched_adam_rows_not_counted = 0
    endpoint_hashes: set[str] = set()
    batch_size = int(settings["attempt_batch_size"])
    if batch_size <= 1 or "lbfgs_candidates_per_attempt_batch" in settings:
        raise ValueError("production requires batched Adam without Adam-quality candidate filtering")
    device_name = str(settings["device"])
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("reference config requires CUDA, but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    dtype = _torch_dtype(settings["dtype"])
    x_tensor = torch.tensor(x, device=device, dtype=dtype)
    y_tensor = torch.tensor(y, device=device, dtype=dtype)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False

    for batch_start in range(0, maximum, batch_size):
        batch_stop = min(maximum, batch_start + batch_size)
        attempt_ids = list(range(batch_start, batch_stop))
        seeds = [reference_seed(config, task.beta_index, task.dataset_id, attempt_id) for attempt_id in attempt_ids]
        initialization = settings["initialization"]
        theta_initial = np.stack(
            [
                initialize_theta(
                    seed,
                    base_scale=float(initialization["base_scale"]),
                    multiplier=float(initialization["weight_scale_multiplier"]),
                )
                for seed in seeds
            ]
        )
        theta_adam, adam_summaries = _batched_adam_phase(
            theta_initial,
            x_tensor,
            y_tensor,
            settings["adam"],
        )
        with torch.no_grad():
            adam_tensor = torch.tensor(theta_adam, device=device, dtype=dtype)
            adam_total, adam_ce, adam_l2 = total_loss_batch(adam_tensor, x_tensor, y_tensor)
            adam_margins = y_tensor[None, :] * logits_batch(adam_tensor, x_tensor)
            adam_wrong = torch.count_nonzero(adam_margins <= 0.0, dim=1).cpu().numpy()
            adam_min_margin = torch.min(adam_margins, dim=1).values.cpu().numpy()
            adam_total_np = adam_total.cpu().numpy()
            adam_ce_np = adam_ce.cpu().numpy()
            adam_l2_np = adam_l2.cpu().numpy()
        batch_summaries: dict[int, dict[str, Any]] = {}
        for row, attempt_id in enumerate(attempt_ids):
            batch_summaries[attempt_id] = {
                "attempt_id": int(attempt_id),
                "seed": int(seeds[row]),
                "attempt_batch_start": int(batch_start),
                "lbfgs_required_only_if_adam_unresolved": True,
                "initialization": initialization["name"],
                "objective_id": OBJECTIVE_ID,
                "objective_fingerprint": OBJECTIVE_FINGERPRINT,
                "adam": adam_summaries[row],
                "adam_endpoint": {
                    "total_loss": float(adam_total_np[row]),
                    "ce_mean": float(adam_ce_np[row]),
                    "l2_per_parameter": float(adam_l2_np[row]),
                    "n_wrong": int(adam_wrong[row]),
                    "min_signed_margin": float(adam_min_margin[row]),
                    "exact": bool(adam_wrong[row] == 0 and adam_min_margin[row] > 0.0),
                },
                "lbfgs": {
                    "run": False,
                    "reason": "pending_attempt_id_order",
                },
                "convergence_is_diagnostic_only": True,
                "eligible_exact_endpoint": False,
                "selected_ref_id": None,
            }
        # Rows are consumed in attempt-ID order. An Adam first-exact endpoint is
        # authoritative and skips L-BFGS; only unresolved rows receive the
        # same-objective L-BFGS phase, which also stops at its first exact endpoint.
        for row, attempt_id in enumerate(attempt_ids):
            adam_exact = bool(
                adam_wrong[row] == 0
                and float(adam_min_margin[row]) > 0.0
            )
            if adam_exact:
                theta_final = np.asarray(theta_adam[row], dtype=np.float64)
                lbfgs = {
                    "run": False,
                    "reason": "adam_first_exact_endpoint",
                    "diagnostic_converged": None,
                }
                optimizer_total_loss = float(adam_total_np[row])
            else:
                theta_final, lbfgs_result = _lbfgs_phase(
                    theta_adam[row],
                    x_tensor,
                    y_tensor,
                    settings["lbfgs"],
                )
                lbfgs = {"run": True, **lbfgs_result}
                optimizer_total_loss = float(lbfgs_result["final_total_loss"])
            final_tensor = torch.tensor(theta_final, device=device, dtype=dtype)
            metrics = replay_metrics(final_tensor, x_tensor, y_tensor)
            first_exact = None
            if adam_summaries[row]["first_exact_epoch_post_update"] is not None:
                first_exact = {
                    "phase": "adam",
                    "step": int(adam_summaries[row]["first_exact_epoch_post_update"]),
                }
            elif lbfgs.get("first_exact_iteration_post_update") is not None:
                first_exact = {
                    "phase": "lbfgs",
                    "step": int(lbfgs["first_exact_iteration_post_update"]),
                }
            finite = _finite_replay(theta_final, metrics)
            replay_consistent = _replay_consistent(
                metrics,
                optimizer_total_loss=optimizer_total_loss,
                absolute_tolerance=float(settings["replay_scalar_abs_tolerance"]),
            )
            endpoint_hash = sha256_array(theta_final)
            relative_tolerance = float(settings["endpoint_duplicate_relative_l2_tolerance"])
            numerically_duplicate = any(
                np.linalg.norm(theta_final - selected)
                / max(1.0, np.linalg.norm(theta_final), np.linalg.norm(selected))
                <= relative_tolerance
                for selected in selected_final
            )
            duplicate = endpoint_hash in endpoint_hashes or numerically_duplicate
            eligible = (
                bool(metrics["exact"])
                and finite
                and replay_consistent
                and not duplicate
            )
            summary = batch_summaries[attempt_id]
            summary.update(
                {
                    "optimizer_chain": (
                        "fresh_random_init_then_batched_adam_first_exact_else_"
                        "lbfgs_first_exact_same_canonical_objective"
                    ),
                    "lbfgs": lbfgs,
                    "first_exact_diagnostic": first_exact,
                    "final": metrics,
                    "finite_final_endpoint": finite,
                    "replay_consistent": replay_consistent,
                    "theta_final_sha256": endpoint_hash,
                    "duplicate_selected_endpoint": duplicate,
                    "eligible_exact_endpoint": eligible,
                }
            )
            if eligible:
                endpoint_hashes.add(endpoint_hash)
                selected_initial.append(theta_initial[row])
                selected_final.append(theta_final)
                selected_attempts.append(attempt_id)
                summary["selected_ref_id"] = len(selected_final) - 1
            attempt_summaries.append(summary)
            if len(selected_final) == target:
                prefetched_adam_rows_not_counted += len(attempt_ids) - row - 1
                break
        if len(selected_final) == target:
            break
    if len(selected_final) != target:
        counters = _attempt_counters(attempt_summaries, selected_attempts)
        failure_path = metadata_path.with_name("reference_failure.json")
        failure_fingerprints = strict_fingerprints(
            config,
            dataset_fingerprint=str(dataset_metadata["payload_sha256"]),
            reference_fingerprint=None,
            task=reference_task_projection(task),
            seed=reference_seed_projection(config, task),
        )
        atomic_json(
            failure_path,
            {
                "schema_version": 1,
                "record_type": "synthetic_reference_search_failure",
                "status": "failed_at_hard_attempt_cap",
                "protocol_id": config["protocol_id"],
                "protocol_sha256": stage_config_fingerprint(config, "reference"),
                "source_sha256": stage_source_fingerprint("reference"),
                "reference_stage_config_sha256": stage_config_fingerprint(config, "reference"),
                "reference_stage_source_sha256": stage_source_fingerprint("reference"),
                "runtime_fingerprint": runtime_fingerprint(),
                "runtime_signature": runtime_signature(),
                "objective_id": OBJECTIVE_ID,
                "objective_fingerprint": OBJECTIVE_FINGERPRINT,
                "beta_index": int(task.beta_index),
                "data_beta": float(task.beta),
                "dataset_id": int(task.dataset_id),
                "dataset_payload_sha256": dataset_metadata["payload_sha256"],
                "target_unique_exact": target,
                "hard_attempt_cap": maximum,
                "selection_policy": "first_exact_attempt_order",
                "convergence_is_diagnostic_only": True,
                "attempt_counters": counters,
                "attempt_summaries": attempt_summaries,
                "fingerprints": failure_fingerprints,
            },
        )
        raise RuntimeError(
            f"reference search failed without replacement/rescue: beta={task.beta:.2f} "
            f"dataset={task.dataset_id:03d}, raw_final_exact={counters['raw_final_exact']}, "
            f"unique_exact={len(selected_final)}/{target}, attempts={maximum}; "
            f"diagnostics={failure_path}"
        )
    initial_array = np.stack(selected_initial).astype(np.float64)
    final_array = np.stack(selected_final).astype(np.float64)
    attempt_array = np.asarray(selected_attempts, dtype=np.int32)
    pack_payload_hash = sha256_array(initial_array, final_array, attempt_array)
    expected = _expected_metadata(
        config,
        task,
        dataset_metadata,
        pack_payload_hash,
    )
    if initial_array.shape != (target, PARAMETER_COUNT) or final_array.shape != (target, PARAMETER_COUNT):
        raise RuntimeError("reference pack shape invariant failed")
    validate_attempt_bookkeeping(attempt_summaries, selected_attempts, target=target)
    atomic_npz(
        output_path,
        theta_init=initial_array,
        theta_final=final_array,
        selected_attempt_id=attempt_array,
    )
    metadata = {
        **expected,
        "attempt_count": len(attempt_summaries),
        "prefetched_adam_rows_not_counted": int(prefetched_adam_rows_not_counted),
        "selected_attempt_ids": selected_attempts,
        "selected_initial_sha256": [sha256_array(row) for row in initial_array],
        "selected_endpoint_sha256": [sha256_array(row) for row in final_array],
        "selected_replay_metrics": [
            replay_metrics(torch.tensor(row, device=device, dtype=dtype), x_tensor, y_tensor)
            for row in final_array
        ],
        "attempt_counters": _attempt_counters(attempt_summaries, selected_attempts),
        "attempt_summaries": attempt_summaries,
        "pack_payload_sha256": pack_payload_hash,
        "pack_file_sha256": sha256_file(output_path),
        "stop_on_first_exact": True,
        "convergence_is_diagnostic_only": True,
        "historical_import": False,
        "replacement_used": False,
        "alternate_objective_rescue_used": False,
        "execution_device": str(settings["device"]),
    }
    if not all(bool(item["exact"]) for item in metadata["selected_replay_metrics"]):
        raise RuntimeError("selected-reference full replay failed exactness before metadata commit")
    metadata["selected_replay_sha256"] = __import__("hashlib").sha256(
        canonical_json_bytes(metadata["selected_replay_metrics"])
    ).hexdigest()
    atomic_json(metadata_path, metadata)
    return "written"


def search_range(
    config: dict,
    *,
    start: int,
    stop: int | None,
    device: str | None = None,
    synthetic_root: Path = SYNTHETIC_ROOT,
) -> dict[str, int]:
    counts = {"written": 0, "skipped_existing": 0}
    for task in iter_reference_tasks(config, start=start, stop=stop):
        status = search_dataset(config, task, device=device, synthetic_root=synthetic_root)
        counts[status] += 1
    return counts
