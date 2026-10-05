"""Exact-shell SMC with direct CE-autograd and exact-FP64 L2 radial scores."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .datasets import load_condition_dataset
from .fingerprints import source_fingerprint
from .io_utils import resolve_project_path
from .manifests import contiguous_sampling_shard, read_sampling_rows
from .modeling import P, torch_logits_batched
from .protocol import (
    condition_names,
    config_fingerprint,
    load_objective_contract,
    validate_config,
)
from .reference_training import (
    load_selected_reference,
    load_selected_reference_pack,
    resolve_device,
)

from .provenance import canonical_json_bytes, fingerprint_mapping, runtime_fingerprint, sha256_array, sha256_file, stable_seed
from .radial import (
    combine_split_log_normalizers,
    shell_l2_components,
)
from .shards import write_jsonl_gzip


SAMPLING_BACKEND_CONTRACT = {
    "backend_id": "cuda_resident_split_batched_c2_v1",
    "particle_geometry_dtype": "float64",
    "loss_forward_dtype": "float32",
    "vmf_axial_sampler": "numpy_wood_exact_float64",
    "vmf_tangent_sampler": "torch_cuda_normal_float64",
    "particle_location": "cuda_resident",
    "split_forward_batching": True,
    "persistent_dataset_tensors": True,
    "persistent_reference_across_radii": True,
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "rng_derivation": "shared.provenance.stable_seed.v1.sha256.63bit",
    "rng_namespace": "mnist-shell-cuda-c2-v1",
    "rng_domains": [
        "wood_axial",
        "cuda_tangent",
        "smc_decision",
    ],
}

SAMPLING_RESULT_SCHEMA_VERSION = "complexity-revised.mnist.shell-result.v2"
SAMPLING_SHARD_SCHEMA_VERSION = "complexity-revised.mnist.shell-shard.v2"
SAMPLING_SHARD_TRAILER_SCHEMA_VERSION = (
    "complexity-revised.mnist.shell-shard-trailer.v1"
)
DIRECT_SCORE_KEYS = (
    "dlogz_dr_direct",
    "dlogz_ce_dr_direct",
    "dlogz_l2_dr_direct",
    "dlogz_l2_dr_autograd_check",
)
L2_AUTOGRAD_CHECK_RELATIVE_SCALE_TOLERANCE = 2.0e-5
SAMPLING_OUTPUT_CONTRACT = {
    "radial_derivative": "direct_autograd",
    "ce_radial_derivative": "direct_autograd_float32",
    "l2_radial_derivative_authority": "exact_fp64_shell_algebra",
    "l2_autograd_role": "float32_diagnostic_check_only",
    "l2_autograd_check_relative_scale_tolerance": (
        L2_AUTOGRAD_CHECK_RELATIVE_SCALE_TOLERANCE
    ),
    "allow_finite_difference_first_derivative": False,
    "shard_integrity": (
        "terminal_ordered_canonical_sampling_result_sha256_v1"
    ),
}
def sampling_backend_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the single allowed production sampling backend contract."""

    backend = dict(config["sampling"].get("backend", {}))
    if backend != SAMPLING_BACKEND_CONTRACT:
        raise ValueError(
            "sampling.backend must exactly select the production CUDA-resident "
            "split-batched C2 contract"
        )
    sampling = config["sampling"]
    scientific_contract = {
        key: value
        for key, value in SAMPLING_OUTPUT_CONTRACT.items()
        if key != "shard_integrity"
    }
    if any(sampling.get(key) != value for key, value in scientific_contract.items()):
        raise ValueError(
            "sampling derivative/output integrity contract drift"
        )
    return backend


def sampling_backend_fingerprint(config: Mapping[str, Any]) -> str:
    return fingerprint_mapping(sampling_backend_contract(config))


def derived_rng_provenance(
    config: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive independent, audit-visible stochastic streams for one shell task."""

    backend = sampling_backend_contract(config)
    raw_split_seeds = [int(value) for value in task["split_seeds"]]
    expected_splits = int(config["sampling"]["independent_splits"])
    if len(raw_split_seeds) != expected_splits:
        raise ValueError("sampling task split-seed count disagrees with config")
    protocol = str(config["protocol"])
    unit_id = str(task["unit_id"])
    namespace = str(backend["rng_namespace"])
    domains = [str(value) for value in backend["rng_domains"]]
    split_rows: list[dict[str, int]] = []
    all_derived: list[int] = []
    for split_index, raw_seed in enumerate(raw_split_seeds):
        seeds = {
            f"{domain}_seed": stable_seed(
                protocol,
                unit_id,
                split_index,
                raw_seed,
                domain,
                namespace=namespace,
                bits=63,
            )
            for domain in domains
        }
        all_derived.extend(seeds.values())
        split_rows.append(
            {
                "split_index": split_index,
                "manifest_split_seed": raw_seed,
                **seeds,
            }
        )
    if len(set(all_derived)) != len(all_derived):
        raise ValueError("derived sampling RNG streams contain a collision")
    return {
        "schema_version": 1,
        "derivation": str(backend["rng_derivation"]),
        "namespace": namespace,
        "unit_id": unit_id,
        "splits": split_rows,
    }


def sampling_result_seed_payload(
    config: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = derived_rng_provenance(config, task)
    return {
        "unit_seed": int(task["unit_seed"]),
        "split_seeds": [int(value) for value in task["split_seeds"]],
        "derived_rng_streams": provenance["splits"],
        "rng_derivation": provenance["derivation"],
        "rng_namespace": provenance["namespace"],
    }


def sampling_shard_seed_payload(
    config: Mapping[str, Any],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "unit_seeds": [int(row["unit_seed"]) for row in tasks],
        "split_seeds": [
            [int(value) for value in row["split_seeds"]] for row in tasks
        ],
        "derived_rng_streams": [
            {
                "unit_id": str(row["unit_id"]),
                "splits": derived_rng_provenance(config, row)["splits"],
            }
            for row in tasks
        ],
        "rng_derivation": SAMPLING_BACKEND_CONTRACT["rng_derivation"],
        "rng_namespace": SAMPLING_BACKEND_CONTRACT["rng_namespace"],
    }


def validate_sampling_result_semantics(config: Mapping[str, Any], task: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep basic numerical QC; historical byte identity is not a criterion."""
    if len(result["splits"]) != int(config["sampling"]["independent_splits"]):
        raise ValueError("incomplete independent particle pools")
    for key in ("logz_angular", "dlogz_dr_direct", "dlogz_ce_dr_direct", "dlogz_l2_dr_direct"):
        if not math.isfinite(float(result[key])):
            raise ValueError(f"nonfinite {key} for {task['unit_id']}")
    if result["condition"] != task["condition"] or int(result["radius_index"]) != int(task["radius_index"]):
        raise ValueError("sampling result coordinate differs from the task")
    if "weighted_training_accuracy" in result:
        accuracy = float(result["weighted_training_accuracy"])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError("weighted_training_accuracy must lie in [0,1]")
        if not math.isclose(
            float(task["radius"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("weighted_training_accuracy is terminal-radius only")
    return dict(result)


def build_sampling_shard_trailer(header: Mapping[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != int(header["unit_count"]):
        raise ValueError("sampling shard has missing results")
    return {"row_type": "sampling_shard_trailer", "unit_count": len(results)}


def _logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.sum(np.exp(array - maximum))))


def _normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    return values - _logsumexp(values)


def weighted_means(
    value_sets: list[np.ndarray],
    weight_sets: list[np.ndarray],
) -> list[float]:
    """Return one normalized weighted mean per independent particle split."""

    if len(value_sets) != len(weight_sets) or not value_sets:
        raise ValueError("value and weight split counts must match and be non-empty")
    output: list[float] = []
    for values, weights in zip(value_sets, weight_sets, strict=True):
        value = np.asarray(values, dtype=np.float64).reshape(-1)
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if value.shape != weight.shape or not np.all(np.isfinite(value)) or not np.all(np.isfinite(weight)):
            raise ValueError("invalid values or weights for a split-weighted mean")
        total = float(np.sum(weight))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("split weights must have positive finite mass")
        output.append(float(np.sum((weight / total) * value)))
    return output


def combine_split_weighted_accuracy(
    split_accuracies: list[float],
    split_mixture_weights: np.ndarray,
) -> float:
    """Combine split accuracies with the same log-normalizer mixture as logZ."""

    accuracies = np.asarray(split_accuracies, dtype=np.float64).reshape(-1)
    mixture = np.asarray(split_mixture_weights, dtype=np.float64).reshape(-1)
    if accuracies.shape != mixture.shape or not np.all(np.isfinite(accuracies)) or not np.all(np.isfinite(mixture)):
        raise ValueError("invalid split accuracies or mixture weights")
    if not np.all((accuracies >= 0.0) & (accuracies <= 1.0)) or np.any(mixture < 0.0):
        raise ValueError("accuracy/mixture values are outside their domains")
    if not math.isclose(float(np.sum(mixture)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("split mixture weights must sum to one")
    return float(np.sum(mixture * accuracies))


def vmf_log_moment_uniform(kappa: float, dimension: int) -> float:
    """log E_uniform[exp(kappa*u_1)] via a stable 0F1 positive series."""

    value = abs(float(kappa))
    p = int(dimension)
    if p < 2 or not math.isfinite(value):
        raise ValueError("invalid vMF moment arguments")
    squared_quarter = value * value / 4.0
    total = 1.0
    term = 1.0
    for order in range(1, 10000):
        term *= squared_quarter / (order * (0.5 * p + order - 1.0))
        total_next = total + term
        if term <= np.finfo(np.float64).eps * total_next:
            total = total_next
            break
        total = total_next
    else:  # pragma: no cover - production kappa is tiny.
        raise RuntimeError("vMF normalizer series failed to converge")
    return math.log(total)


def _unit_gaussian(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    values = rng.normal(size=(int(count), int(dimension)))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise FloatingPointError("zero Gaussian norm while sampling sphere")
    return values / norms


def _draw_vmf_cosines(
    rng: np.random.Generator,
    *,
    count: int,
    dimension: int,
    kappa: float,
) -> np.ndarray:
    """Wood's exact rejection sampler for the axial cosine."""

    n = int(count)
    p = int(dimension)
    concentration = float(kappa)
    if concentration <= 0.0:
        return _unit_gaussian(rng, n, p)[:, 0]
    p_minus_one = float(p - 1)
    b = p_minus_one / (
        2.0 * concentration
        + math.sqrt(4.0 * concentration * concentration + p_minus_one**2)
    )
    x0 = (1.0 - b) / (1.0 + b)
    c = concentration * x0 + p_minus_one * math.log1p(-(x0 * x0))
    result = np.empty(n, dtype=np.float64)
    pending = np.arange(n)
    beta_shape = 0.5 * p_minus_one
    iterations = 0
    while pending.size:
        iterations += 1
        if iterations > 10000:  # pragma: no cover - defensive.
            raise RuntimeError("vMF rejection sampler failed to accept")
        z = rng.beta(beta_shape, beta_shape, size=pending.size)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        log_acceptance = (
            concentration * w
            + p_minus_one * np.log1p(-x0 * w)
            - c
        )
        accepted = np.log(rng.random(pending.size)) <= log_acceptance
        result[pending[accepted]] = w[accepted]
        pending = pending[~accepted]
    return result


def step_overlap(
    log_weights: np.ndarray,
    ce_energy: np.ndarray,
    delta_t: float,
) -> float:
    """Inverse-chi-squared overlap relative to the current weighted pool."""

    logw = _normalise_log_weights(log_weights)
    increment = -float(delta_t) * np.asarray(ce_energy, dtype=np.float64)
    log_first = _logsumexp(logw + increment)
    log_second = _logsumexp(logw + 2.0 * increment)
    return float(math.exp(2.0 * log_first - log_second))


def pool_overlap(log_weights: np.ndarray) -> float:
    """Inverse-chi-squared overlap relative to a uniform particle pool."""

    logw = _normalise_log_weights(log_weights)
    count = logw.size
    return float(math.exp(-_logsumexp(2.0 * logw)) / count)


def _next_delta_t(
    log_weights: np.ndarray,
    ce_energy: np.ndarray,
    remaining: float,
    config: Mapping[str, Any],
) -> tuple[float, float]:
    sampling = config["sampling"]
    threshold = float(sampling["step_overlap_threshold"])
    full_overlap = step_overlap(log_weights, ce_energy, remaining)
    if full_overlap >= threshold:
        return float(remaining), full_overlap
    low, high = 0.0, float(remaining)
    for _ in range(int(sampling["bisection_steps"])):
        middle = 0.5 * (low + high)
        if step_overlap(log_weights, ce_energy, middle) >= threshold:
            low = middle
        else:
            high = middle
    minimum = min(float(sampling["min_delta_t"]), float(remaining))
    minimum_overlap = step_overlap(log_weights, ce_energy, minimum)
    if minimum_overlap < threshold:
        raise RuntimeError(
            "no admissible tempering step: min_delta_t violates the configured "
            "current-weighted-pool overlap threshold"
        )
    delta = max(low, minimum)
    resolved_overlap = step_overlap(log_weights, ce_energy, delta)
    if resolved_overlap < threshold - 1.0e-10:
        raise RuntimeError("bisection produced an overlap-violating tempering step")
    return delta, resolved_overlap


def _systematic_resample(
    rng: np.random.Generator,
    log_weights: np.ndarray,
) -> np.ndarray:
    weights = np.exp(_normalise_log_weights(log_weights))
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(weights.size)) / weights.size
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _configure_deterministic_cuda(device: str):
    """Install and verify the fixed CUDA numerical backend contract."""

    import torch

    resolved = resolve_device(device)
    if resolved.type != "cuda":
        raise RuntimeError(
            "production C2 shell sampling requires an explicit CUDA device"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "deterministic CUDA shell sampling requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before importing Torch"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Torch deterministic algorithms were not enabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must remain disabled for production sampling")
    if torch.backends.cudnn.benchmark:
        raise RuntimeError("cuDNN benchmarking must remain disabled")
    if not torch.backends.cudnn.deterministic:
        raise RuntimeError("cuDNN deterministic execution must remain enabled")
    return resolved


def sampling_backend_runtime(device) -> dict[str, Any]:
    """Capture the CUDA/math runtime that materially defines C2 execution."""

    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("sampling backend runtime requires a CUDA device")
    device_index = (
        torch.cuda.current_device()
        if resolved.index is None
        else int(resolved.index)
    )
    runtime = {
        "schema_version": 1,
        "backend_id": SAMPLING_BACKEND_CONTRACT["backend_id"],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": (
            None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version())
        ),
        "cuda_device_name": str(torch.cuda.get_device_name(device_index)),
        "cuda_compute_capability": [
            int(value)
            for value in torch.cuda.get_device_capability(device_index)
        ],
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(
            torch.get_float32_matmul_precision()
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    }
    validate_sampling_backend_runtime(runtime)
    return runtime


def validate_sampling_backend_runtime(
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on deterministic flags while permitting recorded GPU identity."""

    value = dict(runtime)
    required = {
        "schema_version",
        "backend_id",
        "torch_version",
        "torch_cuda_version",
        "cudnn_version",
        "cuda_device_name",
        "cuda_compute_capability",
        "torch_deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "float32_matmul_precision",
        "cublas_workspace_config",
    }
    if set(value) != required:
        raise ValueError("sampling backend runtime fields are incomplete")
    if value["schema_version"] != 1:
        raise ValueError("sampling backend runtime schema must equal one")
    if value["backend_id"] != SAMPLING_BACKEND_CONTRACT["backend_id"]:
        raise ValueError("sampling backend runtime backend ID drift")
    if not isinstance(value["torch_version"], str) or not value["torch_version"]:
        raise ValueError("sampling backend runtime lacks Torch version")
    if not isinstance(value["torch_cuda_version"], str):
        raise ValueError("sampling backend runtime lacks Torch CUDA version")
    if (
        not isinstance(value["cudnn_version"], int)
        or value["cudnn_version"] <= 0
    ):
        raise ValueError("sampling backend runtime lacks cuDNN version")
    if (
        not isinstance(value["cuda_device_name"], str)
        or not value["cuda_device_name"]
    ):
        raise ValueError("sampling backend runtime lacks CUDA device name")
    capability = value["cuda_compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            not isinstance(component, int) or component < 0
            for component in capability
        )
    ):
        raise ValueError("invalid CUDA compute capability provenance")
    expected_flags = {
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if value["cublas_workspace_config"] not in {":4096:8", ":16:8"}:
        raise ValueError("sampling requires a deterministic cuBLAS workspace setting")
    for key, expected in expected_flags.items():
        if value[key] != expected:
            raise ValueError(
                f"sampling backend runtime {key} drift: "
                f"expected {expected!r}, found {value[key]!r}"
            )
    return value


class _CudaReferenceEvaluator:
    """Persistent CUDA tensors and split-batched loss evaluation for one reference."""

    def __init__(
        self,
        *,
        x,
        y,
        theta_ref: np.ndarray,
        device,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.x = x
        self.y = y
        self.theta_ref_np = np.asarray(theta_ref, dtype=np.float64).copy()
        self.ref32 = torch.as_tensor(
            self.theta_ref_np, dtype=torch.float32, device=device
        )
        self.ref64 = torch.as_tensor(
            self.theta_ref_np, dtype=torch.float64, device=device
        )
        ref_norm = float(np.linalg.norm(self.theta_ref_np))
        if not math.isfinite(ref_norm) or ref_norm <= 0.0:
            raise ValueError("reference norm must be finite and positive")
        self.ref_norm = ref_norm
        self.base_mean64 = -self.ref64 / ref_norm

    def ce_sets(
        self,
        direction_sets: list[Any],
        *,
        radius: float,
        gamma_ce: float,
    ) -> list[np.ndarray]:
        """Evaluate one or both split pools in a single ordered CUDA forward."""

        torch = self.torch
        import torch.nn.functional as functional

        if not direction_sets:
            return []
        counts = [int(value.shape[0]) for value in direction_sets]
        if any(value <= 0 for value in counts):
            raise ValueError("CE direction sets must be nonempty")
        with torch.no_grad():
            unit = torch.cat(direction_sets, dim=0).to(dtype=torch.float32)
            theta = self.ref32[None, :] + math.sqrt(P) * float(radius) * unit
            logits = torch_logits_batched(theta, self.x)
            ce = functional.softplus(-self.y[None, :] * logits).mean(dim=1)
            energy = float(gamma_ce) * ce
        values = energy.detach().cpu().numpy().astype(np.float64)
        if values.shape != (sum(counts),) or not np.all(np.isfinite(values)):
            raise FloatingPointError("non-finite split-batched CE energies")
        boundaries = np.cumsum(counts)[:-1]
        return [
            np.asarray(value, dtype=np.float64)
            for value in np.split(values, boundaries)
        ]

    def joint_direct_scores(
        self,
        direction_sets: list[Any],
        weight_sets: list[np.ndarray],
        *,
        radius: float,
        contract,
    ) -> list[dict[str, float]]:
        """Return direct CE autograd plus authoritative exact-FP64 L2 scores.

        The float32 L2 autograd path is evaluated only as a numerical diagnostic.
        The stored L2 contribution, and therefore the stored total derivative,
        use the analytic shell identity evaluated with float64 directions,
        weights, and reference parameters.
        """

        torch = self.torch
        import torch.nn.functional as functional

        if not direction_sets or len(direction_sets) != len(weight_sets):
            raise ValueError("direct-score split directions/weights disagree")
        split_count = len(direction_sets)
        counts = [int(value.shape[0]) for value in direction_sets]
        normalised_weights: list[np.ndarray] = []
        weight_tensors: list[Any] = []
        for count, raw_weights in zip(counts, weight_sets, strict=True):
            weights = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
            if weights.shape != (count,) or not np.all(np.isfinite(weights)):
                raise ValueError("invalid direct-score particle weights")
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError("direct-score particle weights have zero mass")
            weights = weights / total
            normalised_weights.append(weights)
            weight_tensors.append(
                torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            )

        radial = torch.full(
            (split_count,),
            float(radius),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        theta_sets = [
            self.ref32[None, :]
            + math.sqrt(P)
            * radial[index]
            * directions.to(dtype=torch.float32)
            for index, directions in enumerate(direction_sets)
        ]
        theta = torch.cat(theta_sets, dim=0)
        logits = torch_logits_batched(theta, self.x)
        ce_each = functional.softplus(-self.y[None, :] * logits).mean(dim=1)
        l2_each = (
            float(contract.lambda_reg)
            * torch.sum(theta * theta, dim=1)
            / (2.0 * P)
        )
        ce_losses: list[Any] = []
        l2_losses: list[Any] = []
        start = 0
        for count, weights in zip(counts, weight_tensors, strict=True):
            stop = start + count
            ce_losses.append(torch.sum(weights * ce_each[start:stop]))
            l2_losses.append(torch.sum(weights * l2_each[start:stop]))
            start = stop
        dce = (
            torch.autograd.grad(
                torch.stack(ce_losses).sum(),
                radial,
                retain_graph=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        dl2 = (
            torch.autograd.grad(torch.stack(l2_losses).sum(), radial)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        outputs: list[dict[str, float]] = []
        for index, (directions, weights) in enumerate(
            zip(direction_sets, normalised_weights, strict=True)
        ):
            weights64 = torch.as_tensor(
                weights, dtype=torch.float64, device=self.device
            )
            weighted_direction = torch.sum(
                weights64[:, None] * directions,
                dim=0,
            )
            directional_dot = float(
                torch.dot(weighted_direction, self.ref64).detach().cpu()
            )
            analytic_l2 = -float(contract.lambda_reg) * (
                float(radius) + directional_dot / math.sqrt(P)
            )
            ce_score = -float(contract.gamma_ce) * float(dce[index])
            l2_autograd_check = -float(dl2[index])
            tolerance = (
                L2_AUTOGRAD_CHECK_RELATIVE_SCALE_TOLERANCE
                * max(1.0, abs(analytic_l2))
            )
            if abs(l2_autograd_check - analytic_l2) > tolerance:
                raise FloatingPointError(
                    "CUDA joint direct-autograd L2 score disagrees with "
                    "exact shell algebra"
                )
            outputs.append(
                {
                    "dlogz_dr_direct": ce_score + analytic_l2,
                    "dlogz_ce_dr_direct": ce_score,
                    "dlogz_l2_dr_direct": analytic_l2,
                    "dlogz_l2_dr_autograd_check": l2_autograd_check,
                }
            )
        return outputs


class _CudaShardContext:
    """Reuse one dataset upload and each reference upload across a whole shard."""

    def __init__(
        self,
        config: Mapping[str, Any],
        dataset_cache: Mapping[str, Mapping[str, np.ndarray]],
        reference_cache: Mapping[
            tuple[str, int], tuple[np.ndarray, Mapping[str, Any]]
        ],
        *,
        device: str,
    ) -> None:
        import torch

        sampling_backend_contract(config)
        self.device = _configure_deterministic_cuda(device)
        device_index = (
            torch.cuda.current_device()
            if self.device.index is None
            else int(self.device.index)
        )
        self.execution_device = f"cuda:{device_index}"
        self.backend_runtime = sampling_backend_runtime(self.device)
        self.backend_runtime_fingerprint = fingerprint_mapping(
            self.backend_runtime
        )
        self._torch = torch
        self._dataset_cache = dataset_cache
        self._reference_cache = reference_cache
        self._gpu_datasets: dict[str, tuple[Any, Any]] = {}
        self._evaluators: dict[tuple[str, int], _CudaReferenceEvaluator] = {}

    def evaluator(
        self,
        condition: str,
        reference_index: int,
    ) -> _CudaReferenceEvaluator:
        key = (str(condition), int(reference_index))
        if key in self._evaluators:
            return self._evaluators[key]
        condition_name = key[0]
        if condition_name not in self._gpu_datasets:
            dataset = self._dataset_cache[condition_name]
            self._gpu_datasets[condition_name] = (
                self._torch.as_tensor(
                    dataset["x_train"],
                    dtype=self._torch.float32,
                    device=self.device,
                ),
                self._torch.as_tensor(
                    dataset["y_train"],
                    dtype=self._torch.float32,
                    device=self.device,
                ),
            )
        x, y = self._gpu_datasets[condition_name]
        theta_ref, _ = self._reference_cache[key]
        value = _CudaReferenceEvaluator(
            x=x,
            y=y,
            theta_ref=theta_ref,
            device=self.device,
        )
        self._evaluators[key] = value
        return value

    def close(self) -> None:
        self._evaluators.clear()
        self._gpu_datasets.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _sample_vmf_cuda_numpy_axial(
    axial_rng: np.random.Generator,
    tangent_generator,
    means,
    *,
    kappa: float,
):
    """Exact Wood axial draw with CUDA-float64 tangent geometry."""

    import torch

    centres = means
    if centres.ndim == 1:
        centres = centres[None, :]
    if centres.dtype != torch.float64 or centres.device.type != "cuda":
        raise ValueError("CUDA vMF centres must be resident float64 tensors")
    count, dimension = centres.shape
    concentration = float(kappa)
    if concentration <= 0.0:
        result = torch.randn(
            (count, dimension),
            dtype=torch.float64,
            device=centres.device,
            generator=tangent_generator,
        )
        norms = torch.linalg.vector_norm(result, dim=1, keepdim=True)
        if bool(torch.any(norms == 0.0)):
            raise FloatingPointError("zero CUDA Gaussian norm in sphere sampler")
        return result / norms
    cosines_np = _draw_vmf_cosines(
        axial_rng,
        count=int(count),
        dimension=int(dimension),
        kappa=concentration,
    )
    cosines = torch.as_tensor(
        cosines_np,
        dtype=torch.float64,
        device=centres.device,
    )
    tangent = torch.randn(
        (count, dimension),
        dtype=torch.float64,
        device=centres.device,
        generator=tangent_generator,
    )
    tangent = tangent - torch.sum(
        tangent * centres, dim=1, keepdim=True
    ) * centres
    tangent_norm = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    if bool(torch.any(tangent_norm == 0.0)):
        raise FloatingPointError("zero CUDA tangent norm in vMF sampler")
    tangent = tangent / tangent_norm
    result = (
        cosines[:, None] * centres
        + torch.sqrt(torch.clamp(1.0 - cosines * cosines, min=0.0))[:, None]
        * tangent
    )
    result_norm = torch.linalg.vector_norm(result, dim=1, keepdim=True)
    if bool(torch.any(result_norm == 0.0)):
        raise FloatingPointError("zero CUDA vMF result norm")
    return result / result_norm


def _run_two_splits_cuda(
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    theta_ref: np.ndarray,
    *,
    evaluator: _CudaReferenceEvaluator,
    sentinel: bool,
) -> list[dict[str, Any]]:
    """Run two independent SMC estimators with CUDA-resident directions."""

    import torch

    contract = load_objective_contract()
    sampling = config["sampling"]
    count = int(sampling["particles_per_split"])
    radius = float(task["radius"])
    components = shell_l2_components(theta_ref, radius, contract=contract)
    if not math.isclose(
        evaluator.ref_norm,
        float(components["theta_ref_norm"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("CUDA reference context disagrees with task reference")
    base_kappa = float(components["kappa"])
    move_kappa = float(sampling["move_kappa_factor_per_parameter"]) * P
    repeated_mean = (
        evaluator.base_mean64[None, :].expand(count, -1).contiguous()
    )
    rng_provenance = derived_rng_provenance(config, task)
    states: list[dict[str, Any]] = []
    for split_row in rng_provenance["splits"]:
        tangent_generator = torch.Generator(device=evaluator.device)
        tangent_generator.manual_seed(int(split_row["cuda_tangent_seed"]))
        state: dict[str, Any] = {
            "raw_seed": int(split_row["manifest_split_seed"]),
            "rng_stream_seeds": dict(split_row),
            "axial_rng": np.random.default_rng(
                int(split_row["wood_axial_seed"])
            ),
            "tangent_generator": tangent_generator,
            "decision_rng": np.random.default_rng(
                int(split_row["smc_decision_seed"])
            ),
            "log_weights": np.full(
                count, -math.log(count), dtype=np.float64
            ),
            "temperature": 0.0,
            "logz_ce": 0.0,
            "histories": [],
            "resample_count": 0,
            "acceptance_values": [],
        }
        state["directions"] = _sample_vmf_cuda_numpy_axial(
            state["axial_rng"],
            state["tangent_generator"],
            repeated_mean,
            kappa=base_kappa,
        )
        states.append(state)

    initial_energies = evaluator.ce_sets(
        [state["directions"] for state in states],
        radius=radius,
        gamma_ce=contract.gamma_ce,
    )
    for state, energy in zip(states, initial_energies, strict=True):
        state["ce_energy"] = energy

    for _global_event in range(int(sampling["max_tempering_steps"])):
        active = [
            state
            for state in states
            if float(state["temperature"]) < 1.0 - 1.0e-12
        ]
        if not active:
            break
        for state in active:
            remaining = 1.0 - float(state["temperature"])
            delta, overlap = _next_delta_t(
                state["log_weights"],
                state["ce_energy"],
                remaining,
                config,
            )
            increment = -delta * state["ce_energy"]
            increment_logz = _logsumexp(
                state["log_weights"] + increment
            )
            state["logz_ce"] += increment_logz
            state["log_weights"] = _normalise_log_weights(
                state["log_weights"] + increment - increment_logz
            )
            state["temperature"] = min(
                1.0, float(state["temperature"]) + delta
            )
            pool_pre = pool_overlap(state["log_weights"])
            pool_post = pool_pre
            resampled = False
            if pool_pre < float(sampling["pool_overlap_threshold"]):
                ancestors = _systematic_resample(
                    state["decision_rng"], state["log_weights"]
                )
                indices = torch.as_tensor(
                    ancestors,
                    dtype=torch.int64,
                    device=evaluator.device,
                )
                state["directions"] = state["directions"].index_select(
                    0, indices
                )
                state["ce_energy"] = state["ce_energy"][ancestors]
                state["log_weights"].fill(-math.log(count))
                pool_post = 1.0
                resampled = True
                state["resample_count"] += 1
            state["_event"] = {
                "delta_t": delta,
                "step_overlap": overlap,
                "pool_pre": pool_pre,
                "pool_post": pool_post,
                "resampled": resampled,
                "accepted": 0,
                "proposed": 0,
            }

        for _sweep in range(int(sampling["mh_sweeps"])):
            proposals = [
                _sample_vmf_cuda_numpy_axial(
                    state["axial_rng"],
                    state["tangent_generator"],
                    state["directions"],
                    kappa=move_kappa,
                )
                for state in active
            ]
            proposal_energies = evaluator.ce_sets(
                proposals,
                radius=radius,
                gamma_ce=contract.gamma_ce,
            )
            for state, proposal, proposal_energy in zip(
                active, proposals, proposal_energies, strict=True
            ):
                geometric = (
                    (proposal - state["directions"])
                    @ evaluator.base_mean64
                ).detach().cpu().numpy().astype(np.float64)
                log_ratio = (
                    base_kappa * geometric
                    - float(state["temperature"])
                    * (proposal_energy - state["ce_energy"])
                )
                accept = (
                    np.log(state["decision_rng"].random(count))
                    < np.minimum(0.0, log_ratio)
                )
                accept_gpu = torch.as_tensor(
                    accept,
                    dtype=torch.bool,
                    device=evaluator.device,
                )
                state["directions"] = torch.where(
                    accept_gpu[:, None],
                    proposal,
                    state["directions"],
                )
                state["ce_energy"][accept] = proposal_energy[accept]
                state["_event"]["accepted"] += int(np.count_nonzero(accept))
                state["_event"]["proposed"] += count

        for state in active:
            event = state.pop("_event")
            acceptance = float(
                event["accepted"] / max(event["proposed"], 1)
            )
            state["acceptance_values"].append(acceptance)
            state["histories"].append(
                {
                    "step": len(state["histories"]) + 1,
                    "temperature": float(state["temperature"]),
                    "delta_t": float(event["delta_t"]),
                    "step_overlap_current_weighted_pool": float(
                        event["step_overlap"]
                    ),
                    "pool_overlap_uniform_pool_pre_resample": float(
                        event["pool_pre"]
                    ),
                    "pool_overlap_uniform_pool_post_resample": float(
                        event["pool_post"]
                    ),
                    "resampled": bool(event["resampled"]),
                    "mh_acceptance": acceptance,
                }
            )

    if any(
        abs(float(state["temperature"]) - 1.0) > 1.0e-12
        for state in states
    ):
        raise RuntimeError(
            f"tempering did not reach 1 within "
            f"{sampling['max_tempering_steps']} steps"
        )

    weight_sets = [
        np.exp(_normalise_log_weights(state["log_weights"]))
        for state in states
    ]
    direct_sets = evaluator.joint_direct_scores(
        [state["directions"] for state in states],
        weight_sets,
        radius=radius,
        contract=contract,
    )
    split_training_accuracy: list[float] | None = None
    if math.isclose(
        radius,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        with torch.no_grad():
            directions = torch.cat(
                [state["directions"] for state in states], dim=0
            ).to(dtype=torch.float32)
            theta = (
                evaluator.ref32[None, :]
                + math.sqrt(P) * radius * directions
            )
            logits = torch_logits_batched(theta, evaluator.x)
            particle_accuracy = (
                evaluator.y[None, :] * logits > 0.0
            ).to(torch.float64).mean(dim=1)
        boundaries = np.cumsum([count for _ in states])[:-1]
        accuracy_sets = list(
            np.split(
                particle_accuracy.detach().cpu().numpy().astype(np.float64),
                boundaries,
            )
        )
        split_training_accuracy = weighted_means(accuracy_sets, weight_sets)
        del particle_accuracy, logits, theta, directions
    base_logz = (
        float(components["reference_l2_log_weight"])
        + float(components["radius_l2_log_weight"])
        + vmf_log_moment_uniform(base_kappa, P)
    )
    outputs: list[dict[str, Any]] = []
    for state, weights, direct in zip(
        states, weight_sets, direct_sets, strict=True
    ):
        histories = state["histories"]
        output: dict[str, Any] = {
            "seed": int(state["raw_seed"]),
            "rng_stream_seeds": dict(state["rng_stream_seeds"]),
            "particle_count": count,
            "logz_angular": base_logz + float(state["logz_ce"]),
            "base_l2_logz": base_logz,
            "ce_tempering_logz": float(state["logz_ce"]),
            **direct,
            "temperature_step_count": len(histories),
            "final_temperature": float(state["temperature"]),
            "resample_count": int(state["resample_count"]),
            "minimum_step_overlap": float(
                min(
                    row["step_overlap_current_weighted_pool"]
                    for row in histories
                )
            ),
            "minimum_pool_overlap_pre_resample": float(
                min(
                    row["pool_overlap_uniform_pool_pre_resample"]
                    for row in histories
                )
            ),
            "mean_mh_acceptance": float(
                np.mean(state["acceptance_values"])
            ),
        }
        if split_training_accuracy is not None:
            output["weighted_training_accuracy"] = float(
                split_training_accuracy[len(outputs)]
            )
        if sentinel:
            particle_count = int(
                sampling["sentinel_particle_count_per_split"]
            )
            coordinate_count = int(
                sampling["sentinel_coordinate_count"]
            )
            output["sentinel_payload"] = {
                "directions_head": (
                    state["directions"][
                        :particle_count, :coordinate_count
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                ),
                "ce_energy_head": state["ce_energy"][
                    :particle_count
                ].tolist(),
                "normalised_weights_head": weights[
                    :particle_count
                ].tolist(),
                "temperature_path": histories,
            }
        outputs.append(output)
    return outputs


def run_shell_task(
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    dataset: Mapping[str, np.ndarray],
    theta_ref: np.ndarray,
    reference_metadata: Mapping[str, Any],
    *,
    device: str,
    cuda_context: _CudaShardContext | None = None,
) -> dict[str, Any]:
    validate_config(config)
    sampling_backend_contract(config)
    contract = load_objective_contract()
    owns_context = cuda_context is None
    context = cuda_context
    if context is None:
        condition = str(task["condition"])
        reference_index = int(task["reference_index"])
        context = _CudaShardContext(
            config,
            {condition: dataset},
            {(condition, reference_index): (theta_ref, reference_metadata)},
            device=device,
        )
    try:
        evaluator = context.evaluator(
            str(task["condition"]),
            int(task["reference_index"]),
        )
        splits = _run_two_splits_cuda(
            config,
            task,
            theta_ref,
            evaluator=evaluator,
            sentinel=bool(task["sentinel"]),
        )
    finally:
        if owns_context:
            context.close()
    split_logz = np.asarray([row["logz_angular"] for row in splits], dtype=np.float64)
    split_counts = np.asarray([row["particle_count"] for row in splits], dtype=np.float64)
    combined_logz = combine_split_log_normalizers(split_logz, split_counts)
    split_mixture = (
        split_counts
        / float(np.sum(split_counts))
        * np.exp(split_logz - combined_logz)
    )
    combined_direct = {
        key: float(
            np.sum(
                split_mixture
                * np.asarray([row[key] for row in splits], dtype=np.float64)
            )
        )
        for key in DIRECT_SCORE_KEYS
    }
    split_outputs = [dict(row) for row in splits]
    rng_provenance = derived_rng_provenance(config, task)
    backend = sampling_backend_contract(config)
    backend_fingerprint = sampling_backend_fingerprint(config)
    result = {
        "row_type": "sampling_result",
        "schema_version": SAMPLING_RESULT_SCHEMA_VERSION,
        "protocol": str(config["protocol"]),
        "unit_id": str(task["unit_id"]),
        "manifest_index": int(task["manifest_index"]),
        "condition": str(task["condition"]),
        "reference_index": int(task["reference_index"]),
        "radius_index": int(task["radius_index"]),
        "radius": float(task["radius"]),
        "objective_id": contract.objective_id,
        "objective_fingerprint": contract.fingerprint,
        "config_fingerprint": config_fingerprint(config),
        "sampling_backend_id": str(backend["backend_id"]),
        "sampling_backend_fingerprint": backend_fingerprint,
        "sampling_backend_runtime_fingerprint": (
            context.backend_runtime_fingerprint
        ),
        "rng_provenance": rng_provenance,
        "rng_derivation_fingerprint": fingerprint_mapping(rng_provenance),
        "dataset_hash": str(np.asarray(dataset["dataset_hash"]).reshape(()).item()),
        "theta_ref_sha256": sha256_array(np.asarray(theta_ref, dtype=np.float64)),
        "reference_theta_sha256": str(reference_metadata["theta_array_sha256"]),
        "reference_provenance": dict(
            reference_metadata["reference_provenance"]
        ),
        "logz_angular": combined_logz,
        **combined_direct,
        "split_mixture_weights": split_mixture.tolist(),
        "splits": split_outputs,
        "sentinel": bool(task["sentinel"]),
        "particle_arrays_retained": bool(task["sentinel"]),
    }
    if all("weighted_training_accuracy" in row for row in splits):
        result["weighted_training_accuracy"] = combine_split_weighted_accuracy(
            [float(row["weighted_training_accuracy"]) for row in splits],
            split_mixture,
        )
    from .fingerprints import strict_fingerprints

    result["fingerprints"] = strict_fingerprints(
        config,
        dataset_fingerprint=result["dataset_hash"],
        reference_fingerprint=result["theta_ref_sha256"],
        task={
            "stage": "04_sampling",
            "protocol": str(config["protocol"]),
            "unit_id": str(task["unit_id"]),
            "condition": str(task["condition"]),
            "reference_index": int(task["reference_index"]),
            "radius_index": int(task["radius_index"]),
            "radius": float(task["radius"]),
        },
        seed=sampling_result_seed_payload(config, task),
    )
    validate_sampling_result_semantics(config, task, result)
    return result


def _semantic_shard_path(config: Mapping[str, Any], tasks: list[dict[str, Any]]) -> Path:
    if len(tasks) != int(config["sampling"]["units_per_shard"]):
        raise ValueError("incomplete sampling shard")
    conditions = {str(task["condition"]) for task in tasks}
    if len(conditions) != 1:
        raise ValueError("a shard must contain exactly one condition")
    condition = next(iter(conditions))
    if condition not in condition_names(config):
        raise ValueError(f"unknown condition: {condition}")
    return resolve_project_path(config["paths"]["sampling_shard_root"]) / f"{condition}.jsonl.gz"


def _shard_header(
    config: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
    dataset_fingerprint: str,
    reference_fingerprint: str,
    sampling_runtime: Mapping[str, Any],
    execution_device: str,
) -> dict[str, Any]:
    contract = load_objective_contract()
    backend = sampling_backend_contract(config)
    backend_fingerprint = sampling_backend_fingerprint(config)
    backend_runtime = validate_sampling_backend_runtime(sampling_runtime)
    backend_runtime_fingerprint = fingerprint_mapping(backend_runtime)
    execution_device_value = str(execution_device)
    if not execution_device_value.startswith("cuda:"):
        raise ValueError("execution_device must be a diagnostic CUDA ordinal")
    shard_seed_payload = sampling_shard_seed_payload(config, tasks)
    header = {
        "row_type": "sampling_shard_header",
        "schema_version": SAMPLING_SHARD_SCHEMA_VERSION,
        "protocol": str(config["protocol"]),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "unit_count": len(tasks),
        "unit_selection_fingerprint": fingerprint_mapping(
            {"unit_ids": [row["unit_id"] for row in tasks]}
        ),
        "config_fingerprint": config_fingerprint(config),
        "objective_fingerprint": contract.fingerprint,
        "sampling_backend": backend,
        "sampling_backend_fingerprint": backend_fingerprint,
        "sampling_backend_runtime": backend_runtime,
        "sampling_backend_runtime_fingerprint": (
            backend_runtime_fingerprint
        ),
        "execution_device": execution_device_value,
        "shard_rng_derivation_fingerprint": fingerprint_mapping(
            shard_seed_payload
        ),
        "source_fingerprint": source_fingerprint("sampling"),
        "runtime_environment_signature": runtime_fingerprint(),
    }
    from .fingerprints import strict_fingerprints

    header["fingerprints"] = strict_fingerprints(
        config,
        dataset_fingerprint=dataset_fingerprint,
        reference_fingerprint=reference_fingerprint,
        task={
            "stage": "04_sampling_shard",
            "protocol": str(config["protocol"]),
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "unit_ids": [str(row["unit_id"]) for row in tasks],
        },
        seed=shard_seed_payload,
    )
    return header


def _live_shard_context(
    config: Mapping[str, Any],
    tasks: list[dict[str, Any]],
) -> tuple[
    dict[str, Mapping[str, np.ndarray]],
    dict[tuple[str, int], tuple[np.ndarray, Mapping[str, Any]]],
    str,
    str,
]:
    dataset_cache: dict[str, Mapping[str, np.ndarray]] = {}
    reference_cache: dict[
        tuple[str, int], tuple[np.ndarray, Mapping[str, Any]]
    ] = {}
    for task in tasks:
        condition = str(task["condition"])
        if condition not in dataset_cache:
            dataset_cache[condition] = load_condition_dataset(config, condition)
    for condition in sorted(dataset_cache):
        pack = load_selected_reference_pack(config, condition)
        needed = {
            int(task["reference_index"])
            for task in tasks
            if str(task["condition"]) == condition
        }
        for reference_index in needed:
            reference_cache[(condition, reference_index)] = pack[reference_index]
    dataset_record = [
        {
            "condition": condition,
            "dataset_hash": str(
                np.asarray(dataset["dataset_hash"]).reshape(()).item()
            ),
        }
        for condition, dataset in sorted(dataset_cache.items())
    ]
    reference_record = [
        {
            "condition": condition,
            "reference_index": reference_index,
            "theta_hash": str(metadata["theta_array_sha256"]),
        }
        for (condition, reference_index), (_, metadata) in sorted(
            reference_cache.items()
        )
    ]
    return (
        dataset_cache,
        reference_cache,
        fingerprint_mapping({"datasets": dataset_record}),
        fingerprint_mapping({"references": reference_record}),
    )


def run_sampling_shard(
    config: Mapping[str, Any],
    *,
    shard_index: int,
    shard_count: int,
    device: str,
    resume: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    sampling_backend_contract(config)
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must lie in [0, shard_count)")
    if int(shard_count) <= 0:
        raise ValueError("shard_count must be positive")
    all_tasks = read_sampling_rows(config)
    units_per_shard = int(config["sampling"]["units_per_shard"])
    expected_shard_count = math.ceil(len(all_tasks) / units_per_shard)
    if int(shard_count) != expected_shard_count:
        raise ValueError(
            f"production shard_count must be {expected_shard_count} "
            f"({len(all_tasks)} units / {units_per_shard} units per shard)"
        )
    tasks = contiguous_sampling_shard(
        all_tasks,
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        units_per_shard=units_per_shard,
    )
    path = _semantic_shard_path(config, tasks)
    if path.is_file():
        return {"path": str(path), "reused": True, "status": "skipped_existing"}
    (
        dataset_cache,
        reference_cache,
        live_dataset_fingerprint,
        live_reference_fingerprint,
    ) = _live_shard_context(config, tasks)
    results: list[dict[str, Any]] = []
    with _CudaShardContext(
        config,
        dataset_cache,
        reference_cache,
        device=device,
    ) as cuda_context:
        header = _shard_header(
            config,
            tasks,
            shard_index=shard_index,
            shard_count=shard_count,
            dataset_fingerprint=live_dataset_fingerprint,
            reference_fingerprint=live_reference_fingerprint,
            sampling_runtime=cuda_context.backend_runtime,
            execution_device=cuda_context.execution_device,
        )
        for task in tasks:
            condition = str(task["condition"])
            if condition not in dataset_cache:
                dataset_cache[condition] = load_condition_dataset(
                    config, condition
                )
            dataset = dataset_cache[condition]
            reference_key = (condition, int(task["reference_index"]))
            if reference_key not in reference_cache:
                reference_cache[reference_key] = load_selected_reference(
                    config, *reference_key
                )
            theta_ref, reference_metadata = reference_cache[reference_key]
            results.append(
                run_shell_task(
                    config,
                    task,
                    dataset,
                    theta_ref,
                    reference_metadata,
                    device=device,
                    cuda_context=cuda_context,
                )
            )
    trailer = build_sampling_shard_trailer(header, results)
    shard_sha = write_jsonl_gzip(path, [header, *results, trailer])
    return {"path": str(path), "reused": False, "unit_count": len(results)}
