from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn.functional as torch_functional

from .config import (
    LOCAL_SOURCE_ROOT,
    SYNTHETIC_ROOT,
    canonical_json_bytes,
    protocol_fingerprint,
    radii_from_config,
    source_fingerprint,
)
from .fingerprints import shell_shard_seed_projection, shell_shard_task_projection, strict_fingerprints
from .io import atomic_gzip_jsonl
from .manifests import ShellTask, dataset_count, iter_shell_pass_tasks, reference_metadata_path, reference_pack_path, shell_pass_order, shell_pass_unit_count, shell_radius_indices
from .model import PARAMETER_COUNT, logits_batch
from .objective import CONTRACT, OBJECTIVE_FINGERPRINT, OBJECTIVE_ID
from .reference import _load_dataset
from .vmf import (
    log_sphere_mgf,
    sample_vmf_resident,
    sample_vmf_resident_batch,
)
from .runtime import runtime_fingerprint, runtime_signature


if str(LOCAL_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE_ROOT))
from utils.radial import (  # noqa: E402
    combine_split_log_normalizers,
    shell_l2_components,
)
from utils.provenance import stable_seed  # noqa: E402


SAMPLER_ID = "l2_matched_overlap_controlled_pool_smc_resident_c2_v2"
SAMPLER_ENGINE_ID = "cuda_resident_two_split_c2_v1"
RNG_STREAM_SCHEME_ID = "domain_separated_three_stream_v1"
RNG_STREAM_NAMESPACE = "complexity_revised.synthetic_shell.resident_c2_v1"
BACKEND_COMPATIBILITY_SCOPE_ID = "homogeneous_cuda_backend_no_ordinal_v1"
DERIVATIVE_ID = "combined_resident_direct_autograd_radial_score_v2"
ODD_PASS_COMPLETION_RECEIPT_SCHEMA = 1
ODD_PASS_COMPLETION_RECEIPT_ID = "atomic_strict_odd_completion_receipt_v1"
_SHARD_EXECUTION_DIAGNOSTIC_KEYS = frozenset(
    {"sampler_backend_execution"}
)


@dataclass(frozen=True)
class SMCSettings:
    particles_per_split: int
    step_overlap_threshold: float
    pool_overlap_threshold: float
    bisection_iterations: int
    minimum_temperature_increment: float
    maximum_temperature_events: int
    mh_sweeps: int
    move_kappa: float
    device: str
    dtype: str
    ce_replay_tolerance: float
    engine_id: str
    direction_dtype: str
    combined_particle_batch_size: int
    rng_stream_scheme_id: str
    rng_stream_namespace: str
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    float32_matmul_precision: str

    @classmethod
    def from_config(cls, config: dict, *, device: str | None = None) -> "SMCSettings":
        shell = config["shell"]
        tempering = shell["tempering"]
        mutation = shell["mutation"]
        evaluation = shell["evaluation"]
        derivative = shell["derivative"]
        engine = shell["engine"]
        rng = engine["rng"]
        expected_engine = {
            "id": SAMPLER_ENGINE_ID,
            "resident_directions": True,
            "combine_independent_splits": True,
            "combined_particle_batch_size": 2 * int(shell["particles"]["per_split"]),
            "direction_dtype": "float64",
        }
        for key, expected in expected_engine.items():
            if engine.get(key) != expected:
                raise ValueError(
                    f"shell.engine.{key} must be {expected!r}; got {engine.get(key)!r}"
                )
        expected_rng = {
            "scheme_id": RNG_STREAM_SCHEME_ID,
            "namespace": RNG_STREAM_NAMESPACE,
            "seed_derivation": "shared.provenance.stable_seed_v1_bits63",
            "axial": "numpy_pcg64_wood_beta_rejection_only",
            "tangent": "torch_device_generator_float64_normal_only",
            "decision": "numpy_pcg64_systematic_resampling_and_mh_uniform_only",
        }
        for key, expected in expected_rng.items():
            if rng.get(key) != expected:
                raise ValueError(
                    f"shell.engine.rng.{key} must be {expected!r}; got {rng.get(key)!r}"
                )
        expected_evaluation = {
            "single_forward_ce": True,
            "resident_dataset_tensors": True,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
        for key, expected in expected_evaluation.items():
            if evaluation.get(key) != expected:
                raise ValueError(
                    f"shell.evaluation.{key} must be {expected!r}; "
                    f"got {evaluation.get(key)!r}"
                )
        combined_particle_batch_size = int(engine["combined_particle_batch_size"])
        if int(evaluation["particle_chunk_size"]) != combined_particle_batch_size:
            raise ValueError(
                "shell.evaluation.particle_chunk_size must equal the combined "
                "two-split particle count"
            )
        return cls(
            particles_per_split=int(shell["particles"]["per_split"]),
            step_overlap_threshold=float(tempering["step_overlap_threshold"]),
            pool_overlap_threshold=float(tempering["pool_overlap_threshold"]),
            bisection_iterations=int(tempering["bisection_iterations"]),
            minimum_temperature_increment=float(tempering["minimum_temperature_increment"]),
            maximum_temperature_events=int(tempering["maximum_temperature_events"]),
            mh_sweeps=int(mutation["sweeps_after_every_temperature_event"]),
            move_kappa=float(mutation["move_kappa_factor_times_parameter_count"]) * PARAMETER_COUNT,
            device=str(device or evaluation["device"]),
            dtype=str(evaluation["dtype"]),
            ce_replay_tolerance=float(derivative["ce_replay_max_abs_tolerance"]),
            engine_id=str(engine["id"]),
            direction_dtype=str(engine["direction_dtype"]),
            combined_particle_batch_size=combined_particle_batch_size,
            rng_stream_scheme_id=str(rng["scheme_id"]),
            rng_stream_namespace=str(rng["namespace"]),
            deterministic_algorithms=bool(evaluation["deterministic_algorithms"]),
            cudnn_deterministic=bool(evaluation["cudnn_deterministic"]),
            cudnn_benchmark=bool(evaluation["cudnn_benchmark"]),
            cuda_matmul_allow_tf32=bool(
                evaluation["cuda_matmul_allow_tf32"]
            ),
            cudnn_allow_tf32=bool(evaluation["cudnn_allow_tf32"]),
            float32_matmul_precision=str(evaluation["float32_matmul_precision"]),
        )


def normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("log weights must be finite and non-empty")
    return values - logsumexp(values)


def step_overlap(
    log_weights_normalized: np.ndarray,
    ce_values: np.ndarray,
    *,
    delta_temperature: float,
) -> float:
    log_weights = normalise_log_weights(log_weights_normalized)
    ce = np.asarray(ce_values, dtype=np.float64).reshape(-1)
    if ce.shape != log_weights.shape or not np.all(np.isfinite(ce)):
        raise ValueError("CE and log weights must share a finite shape")
    incremental = -float(delta_temperature) * CONTRACT.gamma_ce * ce
    log_numerator = 2.0 * logsumexp(log_weights + incremental)
    log_denominator = logsumexp(log_weights + 2.0 * incremental)
    return float(np.exp(log_numerator - log_denominator))


def pool_overlap(log_weights_normalized: np.ndarray) -> float:
    log_weights = normalise_log_weights(log_weights_normalized)
    return float(np.exp(-logsumexp(2.0 * log_weights)) / log_weights.size)


def choose_next_temperature(
    current_temperature: float,
    ce_values: np.ndarray,
    log_weights_normalized: np.ndarray,
    settings: SMCSettings,
) -> tuple[float, float, bool]:
    current = float(current_temperature)
    if not 0.0 <= current < 1.0:
        raise ValueError("current temperature must lie in [0,1)")
    direct = step_overlap(log_weights_normalized, ce_values, delta_temperature=1.0 - current)
    if direct >= settings.step_overlap_threshold:
        return 1.0, direct, True
    low = current
    high = 1.0
    for _ in range(settings.bisection_iterations):
        midpoint = 0.5 * (low + high)
        overlap = step_overlap(log_weights_normalized, ce_values, delta_temperature=midpoint - current)
        if overlap >= settings.step_overlap_threshold:
            low = midpoint
        else:
            high = midpoint
    if low - current < settings.minimum_temperature_increment:
        candidate = min(1.0, current + settings.minimum_temperature_increment)
        candidate_overlap = step_overlap(
            log_weights_normalized,
            ce_values,
            delta_temperature=candidate - current,
        )
        if candidate_overlap < settings.step_overlap_threshold:
            raise RuntimeError(
                "minimum temperature increment violates the overlap threshold; "
                "the sampler refuses an unprincipled forced transition"
            )
        low = candidate
    actual = step_overlap(log_weights_normalized, ce_values, delta_temperature=low - current)
    overlap_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(settings.step_overlap_threshold))
    )
    if actual < settings.step_overlap_threshold - overlap_tolerance:
        raise RuntimeError(
            "temperature search returned a transition below the configured "
            f"overlap threshold: actual={actual}, "
            f"threshold={settings.step_overlap_threshold}"
        )
    return float(low), actual, False


def systematic_resample(log_weights_normalized: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probabilities = np.exp(normalise_log_weights(log_weights_normalized))
    count = int(probabilities.size)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(count, dtype=np.float64)) / count
    return np.searchsorted(cumulative, positions, side="left").astype(np.int64)


def _active_torch_backend_flags() -> dict[str, Any]:
    return {
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(
            torch.get_float32_matmul_precision()
        ),
    }


class ParticleEvaluator:
    def __init__(self, x: np.ndarray, y: np.ndarray, settings: SMCSettings):
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("shell config requires CUDA, but torch.cuda.is_available() is false")
        self.device = torch.device(settings.device)
        if settings.dtype.lower() in {"float32", "fp32", "float"}:
            self.dtype = torch.float32
        elif settings.dtype.lower() in {"float64", "fp64", "double"}:
            self.dtype = torch.float64
        else:
            raise ValueError(f"unsupported shell dtype: {settings.dtype}")
        if settings.direction_dtype.lower() not in {"float64", "fp64", "double"}:
            raise ValueError("resident shell directions must use float64")
        self.direction_dtype = torch.float64
        self.combined_particle_batch_size = int(settings.combined_particle_batch_size)
        if (
            self.device.type == "cuda"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
        ):
            raise RuntimeError(
                "resident deterministic CUDA shell requires "
                "CUBLAS_WORKSPACE_CONFIG=':4096:8' or ':16:8' before CUDA initialization"
            )
        torch.use_deterministic_algorithms(
            bool(settings.deterministic_algorithms),
            warn_only=False,
        )
        torch.backends.cudnn.deterministic = bool(settings.cudnn_deterministic)
        torch.backends.cudnn.benchmark = bool(settings.cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(
            settings.cuda_matmul_allow_tf32
        )
        torch.backends.cudnn.allow_tf32 = bool(settings.cudnn_allow_tf32)
        torch.set_float32_matmul_precision(settings.float32_matmul_precision)
        expected_backend_flags = {
            "deterministic_algorithms": bool(
                settings.deterministic_algorithms
            ),
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": bool(settings.cudnn_deterministic),
            "cudnn_benchmark": bool(settings.cudnn_benchmark),
            "cuda_matmul_allow_tf32": bool(
                settings.cuda_matmul_allow_tf32
            ),
            "cudnn_allow_tf32": bool(settings.cudnn_allow_tf32),
            "float32_matmul_precision": str(
                settings.float32_matmul_precision
            ),
        }
        active_backend_flags = _active_torch_backend_flags()
        if active_backend_flags != expected_backend_flags:
            raise RuntimeError(
                "resident shell backend flags do not match the configured "
                f"deterministic contract: active={active_backend_flags!r} "
                f"expected={expected_backend_flags!r}"
            )
        self.x = torch.tensor(np.asarray(x), device=self.device, dtype=self.dtype)
        self.y = torch.tensor(np.asarray(y), device=self.device, dtype=self.dtype)

    def _reference_tensor(self, theta_ref: np.ndarray) -> torch.Tensor:
        values = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
        if values.shape != (PARAMETER_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError("theta_ref must be one finite canonical parameter vector")
        return torch.tensor(
            values,
            device=self.device,
            dtype=self.direction_dtype,
        )

    def ce_resident(
        self,
        theta_ref: np.ndarray,
        direction_parts: list[torch.Tensor],
        radius: float,
    ) -> list[torch.Tensor]:
        """Evaluate all active split pools in one row-separable CE forward."""

        if not direction_parts:
            raise ValueError("resident CE requires at least one active split")
        counts = [int(part.shape[0]) for part in direction_parts]
        if (
            any(
                part.ndim != 2
                or part.shape[1] != PARAMETER_COUNT
                or part.dtype != self.direction_dtype
                or part.device != self.device
                for part in direction_parts
            )
            or sum(counts) > self.combined_particle_batch_size
        ):
            raise ValueError("resident CE received an invalid split-pool batch")
        directions = torch.cat(direction_parts, dim=0)
        reference = self._reference_tensor(theta_ref)
        theta = (
            reference[None, :]
            + math.sqrt(PARAMETER_COUNT) * float(radius) * directions
        ).to(self.dtype)
        with torch.no_grad():
            model_logits = logits_batch(theta, self.x)
            ce = torch_functional.softplus(
                -self.y.reshape(1, -1) * model_logits
            ).mean(dim=1)
        return list(torch.split(ce, counts, dim=0))

    def direct_scores_resident(
        self,
        theta_ref: np.ndarray,
        direction_parts: list[torch.Tensor],
        radius: float,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate both split derivatives in one direct-autograd graph."""

        if not direction_parts:
            raise ValueError("resident direct scores require at least one split")
        counts = [int(part.shape[0]) for part in direction_parts]
        if (
            any(
                part.ndim != 2
                or part.shape[1] != PARAMETER_COUNT
                or part.dtype != self.direction_dtype
                or part.device != self.device
                for part in direction_parts
            )
            or sum(counts) > self.combined_particle_batch_size
        ):
            raise ValueError("resident direct scores received an invalid split-pool batch")
        directions = torch.cat(direction_parts, dim=0)
        reference = self._reference_tensor(theta_ref)
        theta = (
            reference[None, :]
            + math.sqrt(PARAMETER_COUNT) * float(radius) * directions
        ).to(self.dtype).detach().requires_grad_(True)
        model_logits = logits_batch(theta, self.x)
        ce = torch_functional.softplus(
            -self.y.reshape(1, -1) * model_logits
        ).mean(dim=1)
        gradients = torch.autograd.grad(
            ce.sum(),
            theta,
            create_graph=False,
            retain_graph=False,
        )[0]
        root_p = math.sqrt(PARAMETER_COUNT)
        ce_scores = (
            -float(CONTRACT.gamma_ce)
            * root_p
            * torch.sum(gradients.to(torch.float64) * directions, dim=1)
        )
        reference_dot = directions @ reference / root_p
        l2_scores = -float(CONTRACT.lambda_reg) * (
            float(radius) + reference_dot
        )
        packed = torch.stack(
            (
                ce.to(torch.float64),
                ce_scores,
                l2_scores,
                ce_scores + l2_scores,
                (self.y.reshape(1, -1) * model_logits > 0).to(torch.float64).mean(dim=1),
            ),
            dim=1,
        ).detach().cpu().numpy()
        return [
            (
                part[:, 0].copy(),
                part[:, 1].copy(),
                part[:, 2].copy(),
                part[:, 3].copy(),
                part[:, 4].copy(),
            )
            for part in np.split(packed, np.cumsum(counts)[:-1], axis=0)
        ]

def weighted_mean(values: np.ndarray, log_weights_normalized: np.ndarray) -> float:
    probabilities = np.exp(normalise_log_weights(log_weights_normalized))
    return float(np.sum(probabilities * np.asarray(values, dtype=np.float64)))


@dataclass
class _ResidentSplitState:
    split_id: int
    axial_rng: np.random.Generator
    decision_rng: np.random.Generator
    tangent_generator: torch.Generator
    directions: torch.Tensor
    log_weights: np.ndarray
    ce_gpu: torch.Tensor | None = None
    ce: np.ndarray | None = None
    temperature: float = 0.0
    logz_ce: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    resampling_count: int = 0
    direct_transition_count: int = 0


def _split_rng_stream_seeds(
    task: ShellTask,
    split_id: int,
    settings: SMCSettings,
) -> dict[str, int]:
    sid = int(split_id)
    if sid not in (0, 1):
        raise ValueError("synthetic shell split_id must be 0 or 1")
    common = (
        int(task.split_seeds[sid]),
        int(task.beta_index),
        int(task.dataset_id),
        int(task.ref_id),
        int(task.radius_index),
        sid,
    )
    return {
        role: stable_seed(
            *common,
            role,
            namespace=settings.rng_stream_namespace,
            bits=63,
        )
        for role in ("wood_axial", "cuda_tangent", "decision")
    }


def _pcg64(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed)))


def _require_state_ce(state: _ResidentSplitState) -> tuple[torch.Tensor, np.ndarray]:
    if state.ce_gpu is None or state.ce is None:
        raise RuntimeError("resident split CE has not been initialized")
    return state.ce_gpu, state.ce


def run_two_splits_resident(
    theta_ref: np.ndarray,
    task: ShellTask,
    evaluator: ParticleEvaluator,
    settings: SMCSettings,
) -> list[dict[str, Any]]:
    """Run both independent diagnostic pools in one resident CUDA state machine."""

    if settings.engine_id != SAMPLER_ENGINE_ID:
        raise ValueError("resident two-split runner received a non-resident engine")
    if settings.rng_stream_scheme_id != RNG_STREAM_SCHEME_ID:
        raise ValueError("resident two-split runner received an unknown RNG stream scheme")
    if settings.mh_sweeps != 1:
        raise ValueError("synthetic resident C2 requires exactly one MH sweep per event")
    theta_ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    if theta_ref.shape != (PARAMETER_COUNT,) or not np.all(np.isfinite(theta_ref)):
        raise ValueError("resident shell reference has invalid geometry")
    radius = float(task.radius)
    l2 = shell_l2_components(theta_ref, radius, contract=CONTRACT)
    mu = torch.tensor(
        -theta_ref / float(l2["theta_ref_norm"]),
        device=evaluator.device,
        dtype=torch.float64,
    )
    kappa_r = float(l2["kappa"])
    particle_count = int(settings.particles_per_split)
    states: list[_ResidentSplitState] = []
    for split_id in (0, 1):
        stream_seeds = _split_rng_stream_seeds(task, split_id, settings)
        axial_rng = _pcg64(stream_seeds["wood_axial"])
        decision_rng = _pcg64(stream_seeds["decision"])
        tangent_generator = torch.Generator(device=evaluator.device)
        tangent_generator.manual_seed(stream_seeds["cuda_tangent"])
        directions = sample_vmf_resident(
            mu,
            kappa_r,
            particle_count,
            axial_rng=axial_rng,
            tangent_generator=tangent_generator,
        )
        states.append(
            _ResidentSplitState(
                split_id=split_id,
                axial_rng=axial_rng,
                decision_rng=decision_rng,
                tangent_generator=tangent_generator,
                directions=directions,
                log_weights=np.full(
                    particle_count,
                    -math.log(particle_count),
                    dtype=np.float64,
                ),
            )
        )

    initial_ce_parts = evaluator.ce_resident(
        theta_ref,
        [state.directions for state in states],
        radius,
    )
    initial_ce = torch.cat(initial_ce_parts, dim=0).cpu().numpy().astype(np.float64)
    for split_id, state in enumerate(states):
        start = split_id * particle_count
        state.ce_gpu = initial_ce_parts[split_id]
        state.ce = initial_ce[start : start + particle_count].copy()

    for _round_index in range(settings.maximum_temperature_events):
        pending: list[dict[str, Any]] = []
        for state in states:
            if state.temperature >= 1.0 - 1.0e-14:
                continue
            ce_gpu, ce = _require_state_ce(state)
            temperature_start = float(state.temperature)
            new_temperature, overlap, direct = choose_next_temperature(
                temperature_start,
                ce,
                state.log_weights,
                settings,
            )
            increment = -(
                float(new_temperature) - temperature_start
            ) * CONTRACT.gamma_ce * ce
            state.logz_ce += float(logsumexp(state.log_weights + increment))
            state.log_weights = normalise_log_weights(
                state.log_weights + increment
            )
            before_resample = pool_overlap(state.log_weights)
            resampled = before_resample < settings.pool_overlap_threshold
            if resampled:
                ancestors = systematic_resample(
                    state.log_weights,
                    state.decision_rng,
                )
                ancestor_tensor = torch.tensor(
                    ancestors,
                    device=evaluator.device,
                    dtype=torch.long,
                )
                state.directions = state.directions.index_select(
                    0,
                    ancestor_tensor,
                )
                state.ce_gpu = ce_gpu.index_select(0, ancestor_tensor)
                state.ce = ce[ancestors].copy()
                state.log_weights = np.full(
                    particle_count,
                    -math.log(particle_count),
                    dtype=np.float64,
                )
                state.resampling_count += 1
            proposal = sample_vmf_resident_batch(
                state.directions,
                settings.move_kappa,
                axial_rng=state.axial_rng,
                tangent_generator=state.tangent_generator,
            )
            pending.append(
                {
                    "state": state,
                    "temperature_start": temperature_start,
                    "temperature_end": float(new_temperature),
                    "step_overlap": float(overlap),
                    "direct_transition": bool(direct),
                    "pool_overlap_before_resample": float(before_resample),
                    "resampled": bool(resampled),
                    "proposal": proposal,
                }
            )
        if not pending:
            break

        proposed_ce_parts = evaluator.ce_resident(
            theta_ref,
            [row["proposal"] for row in pending],
            radius,
        )
        acceptance_tensors: list[torch.Tensor] = []
        accepted_ce_tensors: list[torch.Tensor] = []
        for row, proposed_ce in zip(pending, proposed_ce_parts):
            state = row["state"]
            ce_gpu, _ce = _require_state_ce(state)
            proposal = row["proposal"]
            current_projection = state.directions @ mu
            proposed_projection = proposal @ mu
            log_ratio = (
                kappa_r * (proposed_projection - current_projection)
                - float(row["temperature_end"])
                * CONTRACT.gamma_ce
                * (proposed_ce.to(torch.float64) - ce_gpu.to(torch.float64))
            )
            log_uniform = torch.tensor(
                np.log(state.decision_rng.random(particle_count)),
                device=evaluator.device,
                dtype=torch.float64,
            )
            accept = log_uniform <= torch.minimum(
                torch.zeros_like(log_ratio),
                log_ratio,
            )
            state.directions = torch.where(
                accept[:, None],
                proposal,
                state.directions,
            )
            state.ce_gpu = torch.where(accept, proposed_ce, ce_gpu)
            acceptance_tensors.append(torch.mean(accept.to(torch.float64)))
            accepted_ce_tensors.append(state.ce_gpu)

        accepted_ce = torch.cat(accepted_ce_tensors, dim=0).cpu().numpy().astype(
            np.float64
        )
        acceptance_rates = torch.stack(acceptance_tensors).cpu().numpy()
        cursor = 0
        for pending_index, row in enumerate(pending):
            state = row["state"]
            state.ce = accepted_ce[cursor : cursor + particle_count].copy()
            cursor += particle_count
            state.history.append(
                {
                    "event_index": len(state.history),
                    "temperature_start": row["temperature_start"],
                    "temperature_end": row["temperature_end"],
                    "step_overlap": row["step_overlap"],
                    "pool_overlap_before_resample": row[
                        "pool_overlap_before_resample"
                    ],
                    "resampled": row["resampled"],
                    "mh_acceptance": float(acceptance_rates[pending_index]),
                    "direct_transition": row["direct_transition"],
                }
            )
            state.direct_transition_count += int(row["direct_transition"])
            state.temperature = float(row["temperature_end"])

    for state in states:
        if state.temperature < 1.0 - 1.0e-12:
            raise RuntimeError(
                "resident SMC did not reach t=1 after "
                f"{settings.maximum_temperature_events} events; "
                f"split={state.split_id} t={state.temperature}"
            )

    direct_parts = evaluator.direct_scores_resident(
        theta_ref,
        [state.directions for state in states],
        radius,
    )
    direction_rows = torch.cat(
        [state.directions for state in states],
        dim=0,
    ).cpu().numpy()
    direction_parts = np.split(
        direction_rows,
        [particle_count],
        axis=0,
    )
    outputs: list[dict[str, Any]] = []
    for state, directions, direct in zip(states, direction_parts, direct_parts):
        ce_gpu, ce = _require_state_ce(state)
        del ce_gpu
        replay_ce, ce_scores, l2_scores, total_scores, accuracy = direct
        final_theta = (
            theta_ref[None, :]
            + math.sqrt(PARAMETER_COUNT) * radius * directions
        )
        l2_penalties = (
            CONTRACT.lambda_reg
            * np.sum(final_theta * final_theta, axis=1)
            / (2.0 * PARAMETER_COUNT)
        )
        maximum_replay_difference = float(np.max(np.abs(replay_ce - ce)))
        if maximum_replay_difference > settings.ce_replay_tolerance:
            raise RuntimeError(
                "resident direct derivative CE replay mismatch "
                f"{maximum_replay_difference} > {settings.ce_replay_tolerance}"
            )
        maximum_norm_error = float(
            np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0))
        )
        if maximum_norm_error > 3.0e-12:
            raise RuntimeError(
                f"resident vMF direction norm drifted by {maximum_norm_error}"
            )
        outputs.append(
            {
                "logZ_CE": float(state.logz_ce),
                "directions": directions,
                "ce": ce,
                "log_weights": normalise_log_weights(state.log_weights),
                "dlogZ_dr": weighted_mean(total_scores, state.log_weights),
                "weighted_ce_radial_score": weighted_mean(
                    ce_scores,
                    state.log_weights,
                ),
                "weighted_l2_radial_score": weighted_mean(
                    l2_scores,
                    state.log_weights,
                ),
                "weighted_ce": weighted_mean(ce, state.log_weights),
                "weighted_l2": weighted_mean(
                    l2_penalties,
                    state.log_weights,
                ),
                "event_count": len(state.history),
                "resampling_count": state.resampling_count,
                "direct_transition_count": state.direct_transition_count,
                "minimum_step_overlap": float(
                    min(row["step_overlap"] for row in state.history)
                ),
                "minimum_pool_overlap": float(
                    min(
                        row["pool_overlap_before_resample"]
                        for row in state.history
                    )
                ),
                "mean_mh_acceptance": float(
                    np.mean(
                        [row["mh_acceptance"] for row in state.history]
                    )
                ),
                "ce_replay_max_abs_difference": maximum_replay_difference,
                "history": state.history,
            }
        )
        if math.isclose(radius, 1.0, rel_tol=0, abs_tol=1e-12):
            outputs[-1]["weighted_training_accuracy"] = weighted_mean(accuracy, state.log_weights)
    return outputs


def _split_mixture_weights(split_logz: np.ndarray, split_counts: np.ndarray) -> np.ndarray:
    values = np.log(split_counts / np.sum(split_counts)) + split_logz
    return np.exp(values - logsumexp(values))


class ShellInputCache:
    def __init__(
        self,
        config: dict,
        synthetic_root: Path,
        settings: SMCSettings,
        runtime_fingerprints: dict[str, str],
    ):
        self.config = config
        self.synthetic_root = Path(synthetic_root)
        self.settings = settings
        self.runtime_fingerprints = dict(runtime_fingerprints)
        self.key: tuple[int, int] | None = None
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.dataset_meta: dict | None = None
        self.theta_final: np.ndarray | None = None
        self.reference_meta: dict | None = None
        self.reference_provenance: dict[str, Any] | None = None
        self.evaluator: ParticleEvaluator | None = None

    def load(self, task: ShellTask) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, dict, ParticleEvaluator]:
        key = (task.beta_index, task.dataset_id)
        if self.key != key:
            x, y, data_meta = _load_dataset(self.config, task, self.synthetic_root)
            pack_path = reference_pack_path(self.synthetic_root, task)
            meta_path = reference_metadata_path(self.synthetic_root, task)
            reference_meta = json.loads(meta_path.read_text())
            with np.load(pack_path, allow_pickle=False) as pack:
                theta = np.asarray(pack["theta_final"], dtype=np.float64)
            count = int(self.config["reference_search"]["references_per_dataset"])
            if theta.shape != (count, PARAMETER_COUNT) or not np.all(np.isfinite(theta)):
                raise ValueError(f"invalid reference pack: {pack_path}")
            self.x, self.y = x, y
            self.theta_final = theta
            self.dataset_meta = data_meta
            self.reference_meta = reference_meta
            self.reference_provenance = {"mode": "existing_filename"}
            self.evaluator = ParticleEvaluator(x, y, self.settings)
            self.key = key
        return self.x, self.y, self.dataset_meta, self.theta_final, self.reference_meta, self.evaluator


def run_unit(
    config: dict,
    task: ShellTask,
    settings: SMCSettings,
    cache: ShellInputCache,
    *,
    synthetic_root: Path = SYNTHETIC_ROOT,
) -> dict[str, Any]:
    _x, _y, dataset_meta, references, reference_meta, evaluator = cache.load(task)
    theta_ref = references[task.ref_id]
    splits = run_two_splits_resident(
        theta_ref,
        task,
        evaluator,
        settings,
    )
    split_logz_ce = np.asarray([split["logZ_CE"] for split in splits], dtype=np.float64)
    split_counts = np.asarray([settings.particles_per_split, settings.particles_per_split], dtype=np.float64)
    logz_ce = combine_split_log_normalizers(split_logz_ce, split_counts)
    mixture = _split_mixture_weights(split_logz_ce, split_counts)
    l2 = shell_l2_components(theta_ref, task.radius, contract=CONTRACT)
    log_m = log_sphere_mgf(PARAMETER_COUNT, float(l2["kappa"]))
    common_log_factor = (
        float(l2["reference_l2_log_weight"])
        + float(l2["radius_l2_log_weight"])
        + log_m
    )
    split_logz_angular = split_logz_ce + common_log_factor
    logz_angular = logz_ce + common_log_factor
    split_derivative = np.asarray([split["dlogZ_dr"] for split in splits], dtype=np.float64)
    dlogz_dr = float(np.sum(mixture * split_derivative))
    ce_radial = float(np.sum(mixture * np.asarray([split["weighted_ce_radial_score"] for split in splits])))
    l2_radial = float(np.sum(mixture * np.asarray([split["weighted_l2_radial_score"] for split in splits])))
    weighted_ce = float(np.sum(mixture * np.asarray([split["weighted_ce"] for split in splits])))
    weighted_l2 = float(np.sum(mixture * np.asarray([split["weighted_l2"] for split in splits])))
    result = {
        "record_type": "shell_unit",
        "task_index": task.task_index,
        "beta_index": task.beta_index,
        "data_beta": task.beta,
        "dataset_id": task.dataset_id,
        "ref_id": task.ref_id,
        "radius_index": task.radius_index,
        "radius": task.radius,
        "shell_pass": task.shell_pass,
        "pass_radius_position": task.pass_radius_position,
        "parameter_count": PARAMETER_COUNT,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": cache.runtime_fingerprints["protocol_sha256"],
        "source_sha256": cache.runtime_fingerprints["source_sha256"],
        "runtime_fingerprint": cache.runtime_fingerprints["environment_fingerprint"],
        "objective_id": OBJECTIVE_ID,
        "objective_fingerprint": OBJECTIVE_FINGERPRINT,
        "gamma_ce": CONTRACT.gamma_ce,
        "lambda_reg": CONTRACT.lambda_reg,
        "sampler_id": SAMPLER_ID,
        "sampler_engine_id": settings.engine_id,
        "rng_stream_scheme_id": settings.rng_stream_scheme_id,
        "rng_stream_namespace": settings.rng_stream_namespace,
        "derivative_methodology_id": DERIVATIVE_ID,
        "derivative_method": CONTRACT.first_radial_derivative["method"],
        "finite_difference_first_derivative_used": False,
        "dataset_payload_sha256": dataset_meta["payload_sha256"],
        "reference_pack_file_sha256": reference_meta["pack_file_sha256"],
        "reference_pack_payload_sha256": reference_meta["pack_payload_sha256"],
        "reference_theta_sha256": reference_meta["selected_endpoint_sha256"][task.ref_id],
        "reference_provenance": dict(cache.reference_provenance or {}),
        "split_seeds": list(task.split_seeds),
        "particles_total": 2 * settings.particles_per_split,
        "particles_per_split": settings.particles_per_split,
        "theta_ref_norm": float(l2["theta_ref_norm"]),
        "kappa_r": float(l2["kappa"]),
        "kappa_move": settings.move_kappa,
        "reference_l2_log_weight": float(l2["reference_l2_log_weight"]),
        "radius_l2_log_weight": float(l2["radius_l2_log_weight"]),
        "log_vmf_normalizer": log_m,
        "logZ_CE": logz_ce,
        "logZ_angular_total": logz_angular,
        "phi_energy_absolute": logz_angular / PARAMETER_COUNT,
        "split0_logZ_angular_total": float(split_logz_angular[0]),
        "split1_logZ_angular_total": float(split_logz_angular[1]),
        "split_logZ_per_parameter_difference": float(abs(split_logz_angular[0] - split_logz_angular[1]) / PARAMETER_COUNT),
        "split_normalizer_mixture_weight0": float(mixture[0]),
        "split_normalizer_mixture_weight1": float(mixture[1]),
        "dlogZ_angular_dr_direct": dlogz_dr,
        "dphi_energy_dr_direct": dlogz_dr / PARAMETER_COUNT,
        "dphi_shell_entropy_dr_analytic": (PARAMETER_COUNT - 2.0) / (PARAMETER_COUNT * task.radius),
        "dphi_shell_dr_direct": dlogz_dr / PARAMETER_COUNT
        + (PARAMETER_COUNT - 2.0) / (PARAMETER_COUNT * task.radius),
        "split0_dlogZ_angular_dr_direct": float(split_derivative[0]),
        "split1_dlogZ_angular_dr_direct": float(split_derivative[1]),
        "split_dlogZ_per_parameter_difference": float(abs(split_derivative[0] - split_derivative[1]) / PARAMETER_COUNT),
        "weighted_ce_radial_score": ce_radial,
        "weighted_l2_radial_score": l2_radial,
        "weighted_total_radial_score": dlogz_dr,
        "weighted_ce_mean": weighted_ce,
        "weighted_l2_per_parameter": weighted_l2,
        "weighted_total_loss": CONTRACT.gamma_ce * weighted_ce + weighted_l2,
        "smc_completed": True,
        "split_event_counts": [int(split["event_count"]) for split in splits],
        "split_resampling_counts": [int(split["resampling_count"]) for split in splits],
        "minimum_step_overlap": float(min(split["minimum_step_overlap"] for split in splits)),
        "minimum_pool_overlap": float(min(split["minimum_pool_overlap"] for split in splits)),
        "mean_mh_acceptance": float(np.mean([split["mean_mh_acceptance"] for split in splits])),
        "ce_replay_max_abs_difference": float(max(split["ce_replay_max_abs_difference"] for split in splits)),
        "evaluation_dtype": settings.dtype,
        "resident_direction_dtype": settings.direction_dtype,
        "combined_particle_batch_size": settings.combined_particle_batch_size,
        "single_forward_ce": True,
        "resident_two_split_execution": True,
        "scalar_storage_dtype": "float64",
    }
    if all("weighted_training_accuracy" in split for split in splits):
        result["weighted_training_accuracy"] = float(np.sum(mixture * np.asarray([split["weighted_training_accuracy"] for split in splits])))
    return result


def shard_path(
    synthetic_root: Path,
    dataset_job_index: int,
    shell_pass: str,
) -> Path:
    selected_pass = str(shell_pass)
    if selected_pass not in {"odd", "even"}:
        raise ValueError("shell_pass must be 'odd' or 'even'")
    job_index = int(dataset_job_index)
    if job_index < 0:
        raise ValueError("dataset job index cannot be negative")
    return (
        Path(synthetic_root)
        / "04_sampling"
        / "raw_outputs"
        / f"{selected_pass}_radius"
        / f"dataset_job_{job_index:04d}.jsonl.gz"
    )


def shell_sampler_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical sampler projection shared by writers and aggregators."""

    return {
        "particles": config["shell"]["particles"],
        "tempering": config["shell"]["tempering"],
        "mutation": config["shell"]["mutation"],
        "derivative": config["shell"]["derivative"],
        "engine": config["shell"]["engine"],
        "evaluation": config["shell"]["evaluation"],
    }


def shell_sampler_config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(shell_sampler_config_projection(config))
    ).hexdigest()


def shell_sampler_engine_provenance(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    engine = config["shell"]["engine"]
    evaluation = config["shell"]["evaluation"]
    return {
        "engine_id": engine["id"],
        "resident_directions": engine["resident_directions"],
        "direction_dtype": engine["direction_dtype"],
        "combine_independent_splits": engine["combine_independent_splits"],
        "combined_particle_batch_size": engine["combined_particle_batch_size"],
        "rng": dict(engine["rng"]),
        "single_forward_ce": evaluation["single_forward_ce"],
        "evaluation_dtype": evaluation["dtype"],
        "deterministic_algorithms": evaluation["deterministic_algorithms"],
        "cudnn_deterministic": evaluation["cudnn_deterministic"],
        "cudnn_benchmark": evaluation["cudnn_benchmark"],
        "cuda_matmul_allow_tf32": evaluation[
            "cuda_matmul_allow_tf32"
        ],
        "cudnn_allow_tf32": evaluation["cudnn_allow_tf32"],
        "float32_matmul_precision": evaluation["float32_matmul_precision"],
    }


def _sampler_backend_runtime_provenance(
    config: dict,
    device: str,
    *,
    assert_active_contract: bool,
) -> dict[str, Any]:
    """Return the device/backend contract committed by every shell shard.

    This intentionally lives in the shell source closure rather than the
    shared runtime signature: changing the shell engine must not relabel the
    retained dataset or reference artifacts.
    """

    requested = torch.device(str(device))
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "resident production shell backend provenance requires CUDA"
        )
    resolved_index = (
        int(requested.index)
        if requested.index is not None
        else int(torch.cuda.current_device())
    )
    if not 0 <= resolved_index < torch.cuda.device_count():
        raise ValueError(
            f"resolved CUDA device index {resolved_index} is unavailable"
        )
    evaluation = config["shell"]["evaluation"]
    expected_backend_flags = {
        "deterministic_algorithms": bool(
            evaluation["deterministic_algorithms"]
        ),
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": bool(evaluation["cudnn_deterministic"]),
        "cudnn_benchmark": bool(evaluation["cudnn_benchmark"]),
        "cuda_matmul_allow_tf32": bool(
            evaluation["cuda_matmul_allow_tf32"]
        ),
        "cudnn_allow_tf32": bool(evaluation["cudnn_allow_tf32"]),
        "float32_matmul_precision": str(
            evaluation["float32_matmul_precision"]
        ),
    }
    if assert_active_contract:
        active_backend_flags = _active_torch_backend_flags()
        if active_backend_flags != expected_backend_flags:
            raise RuntimeError(
                "active CUDA backend changed after evaluator "
                f"initialization: active={active_backend_flags!r} "
                f"expected={expected_backend_flags!r}"
            )
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace_config not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "resident shell backend provenance requires "
            "CUBLAS_WORKSPACE_CONFIG=':4096:8' or ':16:8'"
        )
    properties = torch.cuda.get_device_properties(resolved_index)
    compatibility = {
        "schema": "synthetic_shell_cuda_backend_runtime_v1",
        "cuda_device_name": str(properties.name),
        "cuda_compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "torch_version": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "torch_cuda_build_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": cublas_workspace_config,
        "backend_flags": expected_backend_flags,
    }
    execution_diagnostic = {
        "requested_device": str(device),
        "resolved_cuda_device_index": resolved_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return {
        "compatibility": compatibility,
        "execution_diagnostic": execution_diagnostic,
    }


def _shard_header(
    config: dict,
    dataset_job_index: int,
    shell_pass: str,
    settings: SMCSettings,
    cache: ShellInputCache,
    first_task: ShellTask,
    last_task: ShellTask,
) -> dict[str, Any]:
    radius_indices = shell_radius_indices(config, shell_pass)
    unit_count = shell_pass_unit_count(config, shell_pass)
    if (
        cache.dataset_meta is None
        or cache.reference_meta is None
        or cache.reference_provenance is None
    ):
        raise RuntimeError("shell inputs must be live-verified before constructing a shard header")
    fingerprints = strict_fingerprints(
        config,
        dataset_fingerprint=str(cache.dataset_meta["payload_sha256"]),
        reference_fingerprint=str(cache.reference_meta["pack_payload_sha256"]),
        task=shell_shard_task_projection(
            first_task,
            last_task,
            dataset_job_index=dataset_job_index,
            shell_pass=shell_pass,
            radius_indices=radius_indices,
            unit_count=unit_count,
        ),
        seed=shell_shard_seed_projection(
            config,
            first_task,
            last_task,
            dataset_job_index=dataset_job_index,
            shell_pass=shell_pass,
            radius_indices=radius_indices,
            unit_count=unit_count,
        ),
    )
    backend_runtime = _sampler_backend_runtime_provenance(
        config,
        settings.device,
        assert_active_contract=True,
    )
    return {
        "record_type": "shard_header",
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": protocol_fingerprint(config),
        "source_sha256": source_fingerprint(),
        "runtime_fingerprint": runtime_fingerprint(),
        "runtime_signature": runtime_signature(),
        "objective_id": OBJECTIVE_ID,
        "objective_fingerprint": OBJECTIVE_FINGERPRINT,
        "dataset_payload_sha256": cache.dataset_meta["payload_sha256"],
        "reference_pack_payload_sha256": cache.reference_meta["pack_payload_sha256"],
        "reference_provenance": dict(cache.reference_provenance),
        "fingerprints": fingerprints,
        "sampler_id": SAMPLER_ID,
        "sampler_engine": shell_sampler_engine_provenance(config),
        "sampler_backend_compatibility": backend_runtime["compatibility"],
        "sampler_backend_compatibility_sha256": hashlib.sha256(
            canonical_json_bytes(backend_runtime["compatibility"])
        ).hexdigest(),
        "sampler_backend_execution": backend_runtime[
            "execution_diagnostic"
        ],
        "sampler_config_sha256": shell_sampler_config_sha256(config),
        "dataset_job_index": int(dataset_job_index),
        "shell_pass": str(shell_pass),
        "radius_indices": list(radius_indices),
        "radii": [
            float(radii_from_config(config)[radius_index])
            for radius_index in radius_indices
        ],
        "references_per_dataset": int(config["reference_search"]["references_per_dataset"]),
        "canonical_radius_count": len(radii_from_config(config)),
        "canonical_task_index_bounds": [
            int(first_task.task_index),
            int(last_task.task_index),
        ],
        "unit_count": unit_count,
        "evaluation_dtype": settings.dtype,
    }


def run_shard(
    config: dict,
    *,
    dataset_job_index: int,
    shell_pass: str,
    device: str | None = None,
    synthetic_root: Path = SYNTHETIC_ROOT,
) -> tuple[Path, str]:
    selected_pass = str(shell_pass)
    output = shard_path(synthetic_root, int(dataset_job_index), selected_pass)
    if output.is_file():
        return output, "skipped_existing"
    if selected_pass not in shell_pass_order(config):
        raise ValueError(f"unknown shell pass {selected_pass!r}")
    job_index = int(dataset_job_index)
    if not 0 <= job_index < dataset_count(config):
        raise IndexError(
            f"shell dataset job {job_index} outside [0,{dataset_count(config)})"
        )
    maximum_size = int(config["shell"]["output"]["units_per_atomic_gzip_jsonl_shard"])
    expected_size = shell_pass_unit_count(config, selected_pass)
    if maximum_size != expected_size:
        raise ValueError(
            f"configured shell shard size {maximum_size} differs from pass size {expected_size}"
        )
    tasks = iter_shell_pass_tasks(config, job_index, selected_pass)
    first_task = next(tasks)
    last_task = None
    for last_task in tasks:
        pass
    if last_task is None:
        raise RuntimeError("shell pass iterator unexpectedly contains fewer than two tasks")
    if (
        (first_task.beta_index, first_task.dataset_id)
        != (last_task.beta_index, last_task.dataset_id)
        or first_task.shell_pass != selected_pass
        or last_task.shell_pass != selected_pass
    ):
        raise ValueError("production shell shard crossed its dataset/pass boundary")
    settings = SMCSettings.from_config(config, device=device)
    output = shard_path(synthetic_root, job_index, selected_pass)
    current_source_fingerprint = source_fingerprint()
    current_environment_fingerprint = runtime_fingerprint()
    cache = ShellInputCache(
        config,
        synthetic_root,
        settings,
        {
            "protocol_sha256": protocol_fingerprint(config),
            "source_sha256": current_source_fingerprint,
            "environment_fingerprint": current_environment_fingerprint,
        },
    )
    # Resume is intentionally not a metadata-only shortcut: the input dataset
    # and full reference pack are opened and rehashed before a shard is reused.
    cache.load(first_task)
    header = _shard_header(
        config,
        job_index,
        selected_pass,
        settings,
        cache,
        first_task,
        last_task,
    )
    def rows() -> Iterator[dict[str, Any]]:
        import hashlib

        content_digest = hashlib.sha256()
        yield header
        count = 0
        for task in iter_shell_pass_tasks(config, job_index, selected_pass):
            row = run_unit(config, task, settings, cache, synthetic_root=synthetic_root)
            content_digest.update(canonical_json_bytes(row))
            content_digest.update(b"\n")
            yield row
            count += 1
        yield {
            "record_type": "shard_trailer",
            "dataset_job_index": job_index,
            "shell_pass": selected_pass,
            "unit_count": count,
            "unit_content_sha256": content_digest.hexdigest(),
        }

    atomic_gzip_jsonl(output, rows())
    return output, "written"
