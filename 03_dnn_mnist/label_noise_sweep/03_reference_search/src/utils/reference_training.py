"""Fresh-reference search on the canonical shared total objective."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .datasets import load_condition_dataset
from .fingerprints import scoped_config_fingerprint, source_fingerprint
from .io_utils import MNIST_ROOT, atomic_write_npz, resolve_project_path
from .manifests import read_reference_attempt_rows
from .modeling import (
    P,
    fresh_fan_in_vector,
    numpy_logits,
    replay_metrics_numpy,
    torch_logits_single,
)
from .protocol import (
    condition_names,
    load_objective_contract,
    validate_config,
)

from .provenance import atomic_write_json, fingerprint_mapping, sha256_array, sha256_file, runtime_fingerprint
from .shards import write_jsonl_gzip


REFERENCE_ACCEPTANCE_POLICY = (
    "first_manifest_order_authoritative_float64_exact_replay_unique"
)


def resolve_device(requested: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - full-run dependency guard.
        raise RuntimeError("PyTorch is required for reference search") from exc
    value = str(requested)
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device


def _objective_state(theta, x, y, contract):
    logits = torch_logits_single(theta, x)
    return contract.torch_terms(logits, y, [theta])


def _torch_diagnostics(theta, x, y, contract) -> dict[str, float | int]:
    import torch

    with torch.enable_grad():
        total, ce_mean, l2 = _objective_state(theta, x, y, contract)
        gradient = torch.autograd.grad(total, theta, create_graph=False)[0]
    with torch.no_grad():
        margins = y * torch_logits_single(theta, x)
        return {
            "total_loss": float(total.detach().cpu()),
            "ce_mean": float(ce_mean.detach().cpu()),
            "l2": float(l2.detach().cpu()),
            "gradient_inf_norm": float(gradient.detach().abs().max().cpu()),
            "n_wrong": int(torch.count_nonzero(margins <= 0.0).cpu()),
            "min_margin": float(margins.min().cpu()),
        }


def _torch_snapshot(theta, x, y, contract) -> dict[str, float | int]:
    """Cheap post-update objective/classification check without a gradient pass."""

    import torch

    with torch.no_grad():
        total, ce_mean, l2 = _objective_state(theta, x, y, contract)
        margins = y * torch_logits_single(theta, x)
        return {
            "total_loss": float(total.detach().cpu()),
            "ce_mean": float(ce_mean.detach().cpu()),
            "l2": float(l2.detach().cpu()),
            "n_wrong": int(torch.count_nonzero(margins <= 0.0).cpu()),
            "min_margin": float(margins.min().cpu()),
        }


def _authoritative_replay_candidate(
    theta,
    dataset: Mapping[str, np.ndarray],
    *,
    contract,
    exact_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one immutable endpoint candidate in authoritative float64."""

    import torch

    candidate = theta.detach().to(dtype=torch.float64).clone().requires_grad_(True)
    x_train = torch.as_tensor(
        np.asarray(dataset["x_train"], dtype=np.float64),
        dtype=torch.float64,
        device=candidate.device,
    )
    y_train = torch.as_tensor(
        np.asarray(dataset["y_train"], dtype=np.float64),
        dtype=torch.float64,
        device=candidate.device,
    )
    final_torch = _torch_diagnostics(candidate, x_train, y_train, contract)
    theta64 = candidate.detach().cpu().numpy().astype(np.float64)
    replay_train = replay_metrics_numpy(
        theta64, dataset["x_train"], dataset["y_train"]
    )
    replay_test = replay_metrics_numpy(theta64, dataset["x_test"], dataset["y_test"])
    logits64 = numpy_logits(theta64, dataset["x_train"])
    total64, ce64, l264 = contract.total_numpy(
        logits64,
        np.asarray(dataset["y_train"], dtype=np.float64),
        theta64,
    )
    with torch.no_grad():
        torch_logits64 = (
            torch_logits_single(candidate, x_train)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    replay_logit_max_abs = float(np.max(np.abs(logits64 - torch_logits64)))
    finite_candidate = bool(
        np.all(np.isfinite(theta64))
        and np.all(np.isfinite(logits64))
        and np.isfinite(total64)
        and np.isfinite(ce64)
        and np.isfinite(l264)
        and np.isfinite(float(final_torch["gradient_inf_norm"]))
        and np.isfinite(float(replay_train["min_margin"]))
    )
    tolerance = float(exact_cfg["classification_margin_tolerance"])
    exact_float64_replay = bool(
        int(replay_train["n_wrong"]) == int(exact_cfg["n_wrong"])
        and float(replay_train["min_margin"]) > tolerance
    )
    replay_consistent = bool(
        replay_logit_max_abs <= float(exact_cfg["float64_replay_logit_atol"])
    )
    return {
        "accepted": bool(
            finite_candidate and exact_float64_replay and replay_consistent
        ),
        "theta": theta64,
        "finite_candidate": finite_candidate,
        "exact_float64_replay": exact_float64_replay,
        "replay_consistent": replay_consistent,
        "replay_logit_max_abs": replay_logit_max_abs,
        "final_torch": final_torch,
        "final_float64_objective": {
            "total_loss": total64,
            "ce_mean": ce64,
            "l2": l264,
        },
        "train_replay": replay_train,
        "test_replay": replay_test,
    }


def _minimum_endpoint_distance(
    theta: np.ndarray,
    selected_thetas: Sequence[np.ndarray],
) -> tuple[float | None, int | None]:
    """Return the nearest selected endpoint and its deterministic index."""

    if not selected_thetas:
        return None, None
    distances = np.asarray(
        [
            np.linalg.norm(
                np.asarray(theta, dtype=np.float64)
                - np.asarray(selected, dtype=np.float64)
            )
            for selected in selected_thetas
        ],
        dtype=np.float64,
    )
    index = int(np.argmin(distances))
    return float(distances[index]), index


def train_attempt(
    config: Mapping[str, Any],
    dataset: Mapping[str, np.ndarray],
    *,
    attempt_seed: int,
    device: str,
) -> dict[str, Any]:
    """Return the first authoritative exact endpoint from one fresh attempt."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - full-run dependency guard.
        raise RuntimeError("PyTorch is required for reference search") from exc

    contract = load_objective_contract()
    resolved_device = resolve_device(device)
    torch.manual_seed(int(attempt_seed))
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(int(attempt_seed))
    initial = fresh_fan_in_vector(int(attempt_seed))
    theta = torch.tensor(
        initial,
        dtype=torch.float32,
        device=resolved_device,
        requires_grad=True,
    )
    x_train = torch.as_tensor(
        np.asarray(dataset["x_train"], dtype=np.float32),
        dtype=torch.float32,
        device=resolved_device,
    )
    y_train = torch.as_tensor(
        np.asarray(dataset["y_train"], dtype=np.float32),
        dtype=torch.float32,
        device=resolved_device,
    )
    reference_cfg = config["reference_search"]
    convergence_cfg = reference_cfg["convergence_diagnostics"]
    exact_cfg = reference_cfg["exact_replay"]
    first_observed_exact_step: str | None = None
    accepted_exact_step: str | None = None
    candidate_replay_checks = 0
    rejected_candidate_replays = 0
    previous_checkpoint_loss: float | None = None
    objective_change_from_previous_check: float | None = None

    def finish(
        candidate: Mapping[str, Any],
        *,
        termination_reason: str,
        adam_steps_completed: int,
        lbfgs_outer_completed: int,
    ) -> dict[str, Any]:
        gradient_inf = float(candidate["final_torch"]["gradient_inf_norm"])
        objective_stable = bool(
            objective_change_from_previous_check is not None
            and objective_change_from_previous_check
            <= float(convergence_cfg["objective_change"])
        )
        gradient_stable = bool(
            gradient_inf <= float(convergence_cfg["gradient_inf_norm"])
        )
        accepted = bool(candidate["accepted"])
        return {
            "accepted": accepted,
            "theta": (
                np.asarray(candidate["theta"], dtype=np.float64)
                if accepted
                else None
            ),
            "theta_init": initial.astype(np.float64) if accepted else None,
            "attempt_seed": int(attempt_seed),
            "acceptance_policy": REFERENCE_ACCEPTANCE_POLICY,
            "convergence_is_acceptance_gate": False,
            "first_observed_exact_step": first_observed_exact_step,
            "accepted_exact_step": accepted_exact_step,
            "first_exact_step": accepted_exact_step,
            "adam_steps_completed": adam_steps_completed,
            "lbfgs_outer_completed": lbfgs_outer_completed,
            "candidate_replay_checks": candidate_replay_checks,
            "rejected_candidate_replays": rejected_candidate_replays,
            "termination_reason": termination_reason,
            "finite_candidate": bool(candidate["finite_candidate"]),
            "exact_float64_replay": bool(candidate["exact_float64_replay"]),
            "replay_consistent": bool(candidate["replay_consistent"]),
            "replay_logit_max_abs": float(candidate["replay_logit_max_abs"]),
            "convergence_diagnostics": {
                "role": "diagnostic_only_not_acceptance_gate",
                "objective_change_from_previous_check": (
                    None
                    if objective_change_from_previous_check is None
                    else float(objective_change_from_previous_check)
                ),
                "objective_change_threshold": float(
                    convergence_cfg["objective_change"]
                ),
                "objective_change_below_threshold": objective_stable,
                "gradient_inf_norm": gradient_inf,
                "gradient_inf_norm_threshold": float(
                    convergence_cfg["gradient_inf_norm"]
                ),
                "gradient_inf_norm_below_threshold": gradient_stable,
                "optimizer_stable_diagnostic": bool(
                    objective_stable and gradient_stable
                ),
            },
            "final_torch": dict(candidate["final_torch"]),
            "final_float64_objective": dict(
                candidate["final_float64_objective"]
            ),
            "train_replay": dict(candidate["train_replay"]),
            "test_replay": dict(candidate["test_replay"]),
        }

    def inspect_checkpoint(
        theta_value,
        snapshot: Mapping[str, Any],
        *,
        step_label: str,
        adam_steps_completed: int,
        lbfgs_outer_completed: int,
    ) -> dict[str, Any] | None:
        nonlocal accepted_exact_step
        nonlocal candidate_replay_checks
        nonlocal first_observed_exact_step
        nonlocal objective_change_from_previous_check
        nonlocal previous_checkpoint_loss
        nonlocal rejected_candidate_replays

        current_loss = float(snapshot["total_loss"])
        objective_change_from_previous_check = (
            None
            if previous_checkpoint_loss is None
            else abs(current_loss - previous_checkpoint_loss)
        )
        previous_checkpoint_loss = current_loss
        if not (
            int(snapshot["n_wrong"]) == int(exact_cfg["n_wrong"])
            and float(snapshot["min_margin"])
            > float(exact_cfg["classification_margin_tolerance"])
        ):
            return None
        if first_observed_exact_step is None:
            first_observed_exact_step = step_label
        candidate_replay_checks += 1
        candidate = _authoritative_replay_candidate(
            theta_value,
            dataset,
            contract=contract,
            exact_cfg=exact_cfg,
        )
        if candidate["accepted"]:
            accepted_exact_step = step_label
            return finish(
                candidate,
                termination_reason="first_authoritative_exact_replay",
                adam_steps_completed=adam_steps_completed,
                lbfgs_outer_completed=lbfgs_outer_completed,
            )
        rejected_candidate_replays += 1
        return None

    adam_cfg = reference_cfg["adam"]
    optimizer = torch.optim.Adam(
        [theta],
        lr=float(adam_cfg["learning_rate"]),
        weight_decay=0.0,
    )
    adam_steps_completed = 0
    initial_snapshot = _torch_snapshot(theta, x_train, y_train, contract)
    initial_result = inspect_checkpoint(
        theta,
        initial_snapshot,
        step_label="adam:0",
        adam_steps_completed=0,
        lbfgs_outer_completed=0,
    )
    if initial_result is not None:
        return initial_result
    for step in range(1, int(adam_cfg["max_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = _objective_state(theta, x_train, y_train, contract)
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite canonical objective during Adam")
        total.backward()
        optimizer.step()
        adam_steps_completed = step
        if step % int(adam_cfg["check_every"]) != 0:
            continue
        snapshot = _torch_snapshot(theta, x_train, y_train, contract)
        result = inspect_checkpoint(
            theta,
            snapshot,
            step_label=f"adam:{step}",
            adam_steps_completed=step,
            lbfgs_outer_completed=0,
        )
        if result is not None:
            return result

    # L-BFGS is reached only if Adam did not produce an authoritative exact
    # endpoint. It optimizes the same H_0.01 and also stops at its first exact
    # replay; convergence measurements never gate selection.
    theta = theta.detach().to(dtype=torch.float64).requires_grad_(True)
    x_train = x_train.to(dtype=torch.float64)
    y_train = y_train.to(dtype=torch.float64)
    previous_checkpoint_loss = None
    objective_change_from_previous_check = None
    lbfgs_cfg = reference_cfg["lbfgs"]
    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=int(lbfgs_cfg["max_iter_per_outer"]),
        history_size=int(lbfgs_cfg["history_size"]),
        tolerance_grad=float(lbfgs_cfg["tolerance_grad"]),
        tolerance_change=float(lbfgs_cfg["tolerance_change"]),
        line_search_fn=str(lbfgs_cfg["line_search_fn"]),
    )
    lbfgs_outer_completed = 0
    lbfgs_initial_snapshot = _torch_snapshot(theta, x_train, y_train, contract)
    lbfgs_initial_result = inspect_checkpoint(
        theta,
        lbfgs_initial_snapshot,
        step_label="lbfgs:0",
        adam_steps_completed=adam_steps_completed,
        lbfgs_outer_completed=0,
    )
    if lbfgs_initial_result is not None:
        return lbfgs_initial_result
    for outer in range(1, int(lbfgs_cfg["outer_steps"]) + 1):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            total, _, _ = _objective_state(theta, x_train, y_train, contract)
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite canonical objective during L-BFGS")
            total.backward()
            return total

        optimizer.step(closure)
        lbfgs_outer_completed = outer
        snapshot = _torch_snapshot(theta, x_train, y_train, contract)
        result = inspect_checkpoint(
            theta,
            snapshot,
            step_label=f"lbfgs:{outer}",
            adam_steps_completed=adam_steps_completed,
            lbfgs_outer_completed=outer,
        )
        if result is not None:
            return result

    # A complete final replay is retained for failed-attempt diagnostics too.
    candidate_replay_checks += 1
    final_candidate = _authoritative_replay_candidate(
        theta,
        dataset,
        contract=contract,
        exact_cfg=exact_cfg,
    )
    if final_candidate["accepted"]:
        accepted_exact_step = f"lbfgs:{lbfgs_outer_completed}:final"
        return finish(
            final_candidate,
            termination_reason="final_authoritative_exact_replay",
            adam_steps_completed=adam_steps_completed,
            lbfgs_outer_completed=lbfgs_outer_completed,
        )
    return finish(
        final_candidate,
        termination_reason="optimizer_budget_exhausted_without_authoritative_exact",
        adam_steps_completed=adam_steps_completed,
        lbfgs_outer_completed=lbfgs_outer_completed,
    )


def _condition_root(config: Mapping[str, Any], condition: str) -> Path:
    return resolve_project_path(config["paths"]["reference_root"]) / condition


def _pack_path(config: Mapping[str, Any], condition: str) -> Path:
    return _condition_root(config, condition) / "selected_reference_pack.json"


def _dataset_hash(dataset: Mapping[str, np.ndarray]) -> str:
    return str(np.asarray(dataset["dataset_hash"]).reshape(()).item())


def _pack_contract(config: Mapping[str, Any], dataset_hash: str) -> dict[str, Any]:
    contract = load_objective_contract()
    return {
        "protocol": str(config["protocol"]),
        "config_fingerprint": scoped_config_fingerprint(
            config, scope="reference"
        ),
        "objective_fingerprint": contract.fingerprint,
        "dataset_hash": dataset_hash,
        "source_fingerprint": source_fingerprint(),
        "runtime_environment_signature": runtime_fingerprint(),
        "reference_acceptance_policy": REFERENCE_ACCEPTANCE_POLICY,
        "references_per_condition": int(
            config["reference_search"]["references_per_condition"]
        ),
    }


def _runtime_environment_snapshot() -> dict[str, Any]:
    from .provenance import runtime_signature

    snapshot = runtime_signature()
    try:
        import torch

        snapshot["torch_import_version"] = str(torch.__version__)
        snapshot["torch_cuda_version"] = (
            None if torch.version.cuda is None else str(torch.version.cuda)
        )
    except ImportError:  # pragma: no cover - reference runtime requires torch.
        snapshot["torch_import_version"] = None
        snapshot["torch_cuda_version"] = None
    return snapshot


def run_reference_condition(
    config: Mapping[str, Any],
    condition: str,
    *,
    device: str,
    resume: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    if condition not in condition_names(config):
        raise ValueError(f"unknown production condition {condition!r}")
    pack_path = _pack_path(config, condition)
    if pack_path.is_file():
        return json.loads(pack_path.read_text())
    dataset = load_condition_dataset(config, condition)
    dataset_hash = _dataset_hash(dataset)
    attempt_rows = [
        row
        for row in read_reference_attempt_rows(config)
        if row["condition"] == condition
    ]
    target = int(config["reference_search"]["references_per_condition"])
    maximum = int(config["reference_search"]["max_attempts_per_condition"])
    if len(attempt_rows) != maximum:
        raise ValueError(f"{condition}: attempt manifest is incomplete")
    contract = load_objective_contract()
    attempt_summaries: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    selected_thetas: list[np.ndarray] = []
    minimum_distance = float(
        config["reference_search"]["deduplication"]["minimum_l2_distance"]
    )
    attempt_log_path = (
        _condition_root(config, condition) / "attempt_summaries.jsonl.gz"
    )

    def persist_attempt_log() -> None:
        write_jsonl_gzip(attempt_log_path, attempt_summaries)

    for row in attempt_rows:
        try:
            result = train_attempt(
                config,
                dataset,
                attempt_seed=int(row["attempt_seed"]),
                device=device,
            )
        except Exception as exc:
            attempt_summaries.append(
                {
                    "attempt_id": row["attempt_id"],
                    "attempt_index": int(row["attempt_index"]),
                    "attempt_seed": int(row["attempt_seed"]),
                    "condition": condition,
                    "accepted": False,
                    "selected": False,
                    "selection_rejection_reason": "attempt_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            # Preserve all completed-attempt diagnostics, including the
            # terminal error row, before fail-fast propagation.
            persist_attempt_log()
            raise
        summary = {
            key: value
            for key, value in result.items()
            if key not in ("theta", "theta_init")
        }
        summary.update(
            {
                "attempt_id": row["attempt_id"],
                "attempt_index": int(row["attempt_index"]),
                "condition": condition,
                "selected": False,
            }
        )
        if not result["accepted"]:
            summary["selection_rejection_reason"] = (
                "authoritative_exact_replay_not_satisfied"
            )
            attempt_summaries.append(summary)
            persist_attempt_log()
            continue
        reference_index = len(selected)
        reference_dir = _condition_root(config, condition) / f"ref_{reference_index:03d}"
        theta_path = reference_dir / "reference.npz"
        theta = np.asarray(result["theta"], dtype=np.float64)
        theta_init = np.asarray(result["theta_init"], dtype=np.float64)
        theta_hash = sha256_array(theta)
        theta_init_hash = sha256_array(theta_init)
        nearest_distance, nearest_index = _minimum_endpoint_distance(
            theta, selected_thetas
        )
        summary["minimum_l2_distance_to_selected"] = nearest_distance
        summary["nearest_selected_reference_index"] = nearest_index
        if nearest_distance is not None and nearest_distance <= minimum_distance:
            summary["accepted"] = False
            summary["duplicate_endpoint_rejected"] = True
            summary["selection_rejection_reason"] = "duplicate_endpoint"
            attempt_summaries.append(summary)
            persist_attempt_log()
            continue
        metadata = {
            "schema_version": "complexity-revised.mnist.reference.v2",
            "protocol": str(config["protocol"]),
            "condition": condition,
            "reference_index": reference_index,
            "source_attempt_id": row["attempt_id"],
            "source_attempt_index": int(row["attempt_index"]),
            "attempt_seed": int(row["attempt_seed"]),
            "config_fingerprint": scoped_config_fingerprint(
                config, scope="reference"
            ),
            "objective": contract.metadata(parameter_count=P),
            "dataset_hash": dataset_hash,
            "acceptance_policy": REFERENCE_ACCEPTANCE_POLICY,
            "first_exact_step": result["first_exact_step"],
            "first_observed_exact_step": result["first_observed_exact_step"],
            "accepted_exact_step": result["accepted_exact_step"],
            "final_float64_objective": result["final_float64_objective"],
            "train_replay": result["train_replay"],
            "test_replay": result["test_replay"],
            "convergence_is_acceptance_gate": False,
            "convergence_diagnostics": result["convergence_diagnostics"],
        }
        from .fingerprints import strict_fingerprints

        row_fingerprints = strict_fingerprints(
            config,
            dataset_fingerprint=dataset_hash,
            reference_fingerprint=theta_hash,
            task={
                "stage": "03_reference_search",
                "protocol": str(config["protocol"]),
                "condition": condition,
                "reference_index": reference_index,
                "source_attempt_id": str(row["attempt_id"]),
            },
            seed={
                "attempt_seed": int(row["attempt_seed"]),
                "theta_init_sha256": theta_init_hash,
            },
        )
        metadata["fingerprints"] = row_fingerprints
        atomic_write_npz(
            theta_path,
            theta=theta,
            theta_init=theta_init,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        selected.append(
            {
                "reference_index": reference_index,
                "source_attempt_id": row["attempt_id"],
                "attempt_seed": int(row["attempt_seed"]),
                "theta_path": str(theta_path.relative_to(MNIST_ROOT)),
                "theta_array_sha256": theta_hash,
                "theta_init_array_sha256": theta_init_hash,
                "theta_file_sha256": sha256_file(theta_path),
                "fingerprints": row_fingerprints,
                "total_loss": float(result["final_float64_objective"]["total_loss"]),
                "min_margin": float(result["train_replay"]["min_margin"]),
                "n_wrong": int(result["train_replay"]["n_wrong"]),
            }
        )
        selected_thetas.append(theta)
        summary["selected"] = True
        summary["selected_reference_index"] = reference_index
        summary["selection_rejection_reason"] = None
        attempt_summaries.append(summary)
        persist_attempt_log()
        if len(selected) == target:
            break

    if len(selected) != target:
        persist_attempt_log()
        exact_replay_count = sum(
            bool(row.get("exact_float64_replay"))
            and bool(row.get("replay_consistent"))
            and bool(row.get("finite_candidate"))
            for row in attempt_summaries
        )
        duplicate_count = sum(
            bool(row.get("duplicate_endpoint_rejected"))
            for row in attempt_summaries
        )
        raise RuntimeError(
            f"{condition}: obtained {len(selected)}/{target} unique authoritative "
            f"exact references ({exact_replay_count} finite replay-consistent exact "
            f"endpoints, {duplicate_count} duplicate rejections) after "
            f"{len(attempt_summaries)}/{maximum} fresh attempts; "
            "fallback and replacement are forbidden"
        )
    pack = {
        "schema_version": "complexity-revised.mnist.reference-pack.v2",
        **_pack_contract(config, dataset_hash),
        "condition": condition,
        "attempts_used": len(attempt_summaries),
        "attempt_log_path": str(attempt_log_path.relative_to(MNIST_ROOT)),
        "attempt_log_sha256": sha256_file(attempt_log_path),
        "selected_references": selected,
        "selection_policy": REFERENCE_ACCEPTANCE_POLICY,
        "optimizer_convergence_role": "diagnostic_only_not_acceptance_gate",
        "fallback": False,
        "replacement": False,
    }
    from .fingerprints import strict_fingerprints
    from .provenance import fingerprint_mapping

    pack["fingerprints"] = strict_fingerprints(
        config,
        dataset_fingerprint=dataset_hash,
        reference_fingerprint=fingerprint_mapping(
            {"theta_hashes": [row["theta_array_sha256"] for row in selected]}
        ),
        task={
            "stage": "03_reference_search",
            "protocol": str(config["protocol"]),
            "condition": condition,
            "selected_reference_count": len(selected),
        },
        seed={"attempt_seeds": [int(row["attempt_seed"]) for row in selected]},
    )
    pack["runtime_environment_snapshot"] = _runtime_environment_snapshot()
    atomic_write_json(pack_path, pack)
    return pack


def run_reference_search(
    config: Mapping[str, Any],
    *,
    device: str,
    condition: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    conditions = condition_names(config)
    selected = conditions if condition is None else [condition]
    results = [
        run_reference_condition(
            config,
            value,
            device=device,
            resume=resume,
        )
        for value in selected
    ]
    return {
        "protocol": str(config["protocol"]),
        "conditions_completed": len(results),
        "selected_reference_count": sum(
            len(result["selected_references"]) for result in results
        ),
    }


def load_selected_reference(
    config: Mapping[str, Any],
    condition: str,
    reference_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    pack = load_selected_reference_pack(config, condition)
    return pack[int(reference_index)]


def load_selected_reference_pack(config: Mapping[str, Any], condition: str) -> list[tuple[np.ndarray, dict[str, Any]]]:
    pack_path = _pack_path(config, condition)
    pack = json.loads(pack_path.read_text())
    selected = list(pack["selected_references"])
    count = int(config["reference_search"]["references_per_condition"])
    if len(selected) != count:
        raise ValueError(f"expected {count} references: {pack_path}")
    compact = "arrays_path" in pack
    if compact:
        with np.load(resolve_project_path(pack["arrays_path"]), allow_pickle=False) as arrays:
            theta_pack = np.asarray(arrays["theta"], dtype=np.float64)
        if theta_pack.shape != (count, P):
            raise ValueError(f"invalid reference dimensions: {pack_path}")
    output = []
    for index, row in enumerate(selected):
        if int(row["reference_index"]) != index:
            raise ValueError(f"invalid reference ordering: {pack_path}")
        if compact:
            theta = theta_pack[index]
        else:
            with np.load(resolve_project_path(row["theta_path"]), allow_pickle=False) as payload:
                theta = np.asarray(payload["theta"], dtype=np.float64)
        if theta.shape != (P,) or not np.all(np.isfinite(theta)):
            raise ValueError(f"invalid reference values: {pack_path}")
        metadata = dict(row)
        metadata["reference_provenance"] = {"mode": "existing_filename"}
        output.append((theta, metadata))
    return output
