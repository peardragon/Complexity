"""Scientific settings for the standalone MNIST label-noise experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io_utils import (
    MNIST_ROOT,
    PROTOCOL_ROOT,
    PROTOCOLS,
    STAGE_ROOT,
    load_json,
)
from .modeling import ParameterLayout


SHARED_OBJECTIVE_PATH = STAGE_ROOT / "config" / "objective.json"
SHARED_RESOURCE_PATH = STAGE_ROOT / "config" / "resources.json"

EXPECTED_STAGES = (
    "01_dataset",
    "02_complexity_measure",
    "03_reference_search",
    "04_sampling",
    "05_proxy_local_entropy",
)

EXPECTED_ETAS = (0.0, 0.05, 0.15, 0.25, 0.5)
EXPECTED_RADII = np.linspace(0.01, 1.0, 100, dtype=np.float64)
EXPECTED_REFERENCE_ACCEPTANCE_POLICY = (
    "first_manifest_order_authoritative_float64_exact_replay_unique"
)


def _require_shared_imports():
    required = (
        SHARED_OBJECTIVE_PATH,
        SHARED_RESOURCE_PATH,
        Path(__file__).resolve().parent / "objective.py",
        Path(__file__).resolve().parent / "resources.py",
        Path(__file__).resolve().parent / "provenance.py",
        Path(__file__).resolve().parent / "radial.py",
        Path(__file__).resolve().parent / "shards.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "mandatory revised shared contract files are missing: " + ", ".join(missing)
        )
    from .config import config_fingerprint
    from .objective import ObjectiveContract
    from .resources import ResourcePolicy

    return ObjectiveContract, ResourcePolicy, config_fingerprint


def load_config(protocol: str) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    path = STAGE_ROOT / "config" / "default.json"
    config = load_json(path)
    if config.get("protocol") != protocol:
        raise ValueError(f"{path}: protocol field mismatch")
    return config


def load_objective_contract():
    ObjectiveContract, _, _ = _require_shared_imports()
    return ObjectiveContract.from_json(SHARED_OBJECTIVE_PATH)


def load_resource_policy():
    _, ResourcePolicy, _ = _require_shared_imports()
    return ResourcePolicy.from_json(SHARED_RESOURCE_PATH)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    _, _, shared_config_fingerprint = _require_shared_imports()
    return str(shared_config_fingerprint(config))


def condition_names(config: Mapping[str, Any]) -> list[str]:
    if config["protocol"] != "label_noise_sweep":
        raise ValueError(f"unsupported label-noise protocol {config['protocol']!r}")
    from .datasets import label_condition_name

    return [label_condition_name(float(eta)) for eta in config["dataset"]["etas"]]


def radius_grid(config: Mapping[str, Any]) -> np.ndarray:
    sampling = config["sampling"]
    values = np.linspace(
        float(sampling["radii_start"]),
        float(sampling["radii_stop"]),
        int(sampling["radii_count"]),
        dtype=np.float64,
    )
    if values.shape != EXPECTED_RADII.shape or not np.array_equal(
        values, EXPECTED_RADII
    ):
        raise ValueError("production radius grid must be exactly 0.01,...,1.00")
    return values


def baseline_index(config: Mapping[str, Any]) -> int:
    radii = radius_grid(config)
    matches = np.flatnonzero(
        np.isclose(
            radii,
            float(config["sampling"]["baseline_radius"]),
            rtol=0.0,
            atol=1.0e-15,
        )
    )
    if matches.size != 1:
        raise ValueError("baseline radius is absent or duplicated in radius grid")
    return int(matches[0])


def _expect(
    value: Any,
    expected: Any,
    *,
    context: str,
) -> None:
    if value != expected:
        raise ValueError(f"{context}: expected {expected!r}, found {value!r}")


def _assert_path_contract(config: Mapping[str, Any]) -> None:
    protocol = str(config["protocol"])
    stage_by_key = {
        "dataset_root": "01_dataset",
        "complexity_root": "02_complexity_measure",
        "reference_root": "03_reference_search",
        "manifest_root": "03_reference_search",
        "sampling_shard_root": "04_sampling",
        "summary_root": "05_proxy_local_entropy",
    }
    for key, stage in stage_by_key.items():
        relative = Path(str(config["paths"][key]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"paths.{key} must be a safe MNIST-relative path")
        expected_prefix = Path(PROTOCOL_ROOT.name) / stage
        if relative.parts[:2] != expected_prefix.parts:
            raise ValueError(
                f"paths.{key} must begin with {expected_prefix}, found {relative}"
            )
        resolved = (MNIST_ROOT / relative).resolve()
        resolved.relative_to(MNIST_ROOT)


def _assert_stage_layout(protocol: str) -> None:
    root = PROTOCOL_ROOT
    missing: list[str] = []
    for stage in EXPECTED_STAGES:
        stage_root = root / stage
        if not (stage_root / "config" / "default.json").is_file():
            missing.append(f"{stage}/config/default.json")
        if not (stage_root / "src").is_dir():
            missing.append(f"{stage}/src")
    if missing:
        raise ValueError(f"{protocol}: incomplete 01..05 stage skeleton: {missing}")


def validate_config(
    config: Mapping[str, Any],
    *,
    require_layout: bool = True,
) -> dict[str, Any]:
    protocol = str(config["protocol"])
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported production protocol {protocol!r}")
    _expect(
        config["schema_version"],
        "complexity-revised.mnist.phase1.v2",
        context="schema_version",
    )

    contract = load_objective_contract()
    objective = config["objective"]
    objective_path = (STAGE_ROOT / str(objective["contract_path"])).resolve()
    _expect(objective_path, SHARED_OBJECTIVE_PATH, context="objective.contract_path")
    projection = {
        "objective_id": contract.objective_id,
        "loss": contract.loss,
        "ce_reduction": contract.ce_reduction,
        "gamma_ce": contract.gamma_ce,
        "lambda_reg": contract.lambda_reg,
        "regularized_parameters": contract.regularized_parameters,
    }
    for key, expected in projection.items():
        _expect(objective[key], expected, context=f"objective.{key}")
    _expect(
        objective["forbid_n_train_multiplier"],
        True,
        context="objective.forbid_n_train_multiplier",
    )
    _expect(objective["parameter_count"], 2461, context="objective.parameter_count")

    layout = ParameterLayout.from_config(config["model"])
    _expect(layout.parameter_count, 2461, context="model parameter count")
    _expect(
        config["model"]["adam_compute_dtype"],
        "float32",
        context="model.adam_compute_dtype",
    )
    _expect(
        config["model"]["lbfgs_compute_dtype"],
        "float64",
        context="model.lbfgs_compute_dtype",
    )
    _expect(
        config["model"]["sampling_compute_dtype"],
        "float32",
        context="model.sampling_compute_dtype",
    )
    _expect(
        config["model"]["theta_storage_dtype"],
        "float64",
        context="model.theta_storage_dtype",
    )

    dataset = config["dataset"]
    replica_index = int(dataset["replica_index"])
    if not 0 <= replica_index <= 9:
        raise ValueError("dataset.replica_index must lie in [0,9]")
    _expect(
        dataset["split_seed"],
        20260610 + replica_index,
        context="dataset.split_seed",
    )
    _expect(dataset["n_train"], 512, context="dataset.n_train")
    _expect(dataset["n_test"], 2048, context="dataset.n_test")
    _expect(dataset["input_shape"], [1, 10, 10], context="dataset.input_shape")
    _expect(
        dataset["downscale"],
        "PIL.Image.resize((10, 10), Image.Resampling.BOX)",
        context="dataset.downscale",
    )
    _expect(
        dataset["standardization_std_floor"],
        1.0e-6,
        context="dataset.standardization_std_floor",
    )
    _expect(
        dataset["standardization_floor_replacement"],
        1.0,
        context="dataset.standardization_floor_replacement",
    )
    _expect(
        dataset["class_balance_train"], [256, 256], context="class_balance_train"
    )
    _expect(
        dataset["class_balance_test"], [1024, 1024], context="class_balance_test"
    )
    _expect(
        dataset["openml_cache_root"],
        f"{PROTOCOL_ROOT.name}/01_dataset/raw_outputs/source_cache",
        context="dataset.openml_cache_root",
    )

    _expect(
        tuple(float(value) for value in dataset["etas"]),
        EXPECTED_ETAS,
        context="dataset.etas",
    )
    _expect(
        dataset["flip_uniform_master_seed"],
        2026065000 + 1000 * replica_index,
        context="dataset.flip_uniform_master_seed",
    )
    _expect(
        dataset["flip_uniform_draw_order"],
        "train_then_test",
        context="dataset.flip_uniform_draw_order",
    )
    _expect(
        dataset["noise_applies_to"],
        "train_and_test",
        context="dataset.noise_applies_to",
    )
    _expect(
        dataset["standardization"],
        "clean_even_odd_train_feature_mean_std",
        context="dataset.standardization",
    )

    reference = config["reference_search"]
    _expect(
        reference["references_per_condition"],
        10,
        context="reference_search.references_per_condition",
    )
    _expect(
        reference["max_attempts_per_condition"],
        240,
        context="reference_search.max_attempts_per_condition",
    )
    _expect(
        reference["execution_mode"],
        "sequential_attempts_condition_parallel",
        context="reference_search.execution_mode",
    )
    _expect(
        reference["initialization"],
        "fresh_deterministic_fan_in_normal_zero_bias",
        context="reference_search.initialization",
    )
    _expect(
        reference["acceptance_policy"],
        EXPECTED_REFERENCE_ACCEPTANCE_POLICY,
        context="reference_search.acceptance_policy",
    )
    _expect(
        reference["first_authoritative_exact_is_endpoint"],
        True,
        context="reference_search.first_authoritative_exact_is_endpoint",
    )
    _expect(
        reference["acceptance_requires_optimizer_convergence"],
        False,
        context="reference_search.acceptance_requires_optimizer_convergence",
    )
    _expect(reference["allow_fallback"], False, context="allow_fallback")
    _expect(reference["allow_replacement"], False, context="allow_replacement")
    _expect(
        reference["convergence_diagnostics"]["role"],
        "diagnostic_only_not_acceptance_gate",
        context="reference_search.convergence_diagnostics.role",
    )
    if "convergence" in reference or "first_exact_is_diagnostic_only" in reference:
        raise ValueError("legacy convergence-gated reference keys are forbidden")
    _expect(
        reference["exact_replay"]["n_wrong"], 0, context="exact_replay.n_wrong"
    )
    _expect(
        reference["exact_replay"]["classification_margin_tolerance"],
        0.0,
        context="exact_replay.classification_margin_tolerance",
    )
    _expect(
        reference["exact_replay"]["min_margin_strictly_greater_than_tolerance"],
        True,
        context="exact_replay.min_margin_strictly_greater_than_tolerance",
    )
    _expect(
        reference["deduplication"]["metric"],
        "parameter_l2_distance",
        context="reference_search.deduplication.metric",
    )
    _expect(
        reference["deduplication"]["minimum_l2_distance"],
        1.0e-6,
        context="reference_search.deduplication.minimum_l2_distance",
    )
    expected_conditions = 5
    _expect(
        reference["expected_attempt_count"],
        240 * expected_conditions,
        context="reference_search.expected_attempt_count",
    )
    _expect(
        reference["expected_reference_count"],
        10 * expected_conditions,
        context="reference_search.expected_reference_count",
    )

    sampling = config["sampling"]
    exact_sampling = {
        "particles_total": 1024,
        "independent_splits": 2,
        "particles_per_split": 512,
        "step_overlap_threshold": 0.95,
        "step_overlap_reference": "current_weighted_pool_inverse_chi_squared_overlap",
        "pool_overlap_threshold": 0.50,
        "pool_overlap_reference": "uniform_pool_inverse_chi_squared_overlap",
        "max_tempering_steps": 180,
        "min_delta_t": 0.0001,
        "bisection_steps": 32,
        "mh_sweeps": 2,
        "move_kappa_factor_per_parameter": 80.0,
        "radii_start": 0.01,
        "radii_stop": 1.0,
        "radii_count": 100,
        "baseline_radius": 0.01,
        "radial_derivative": "direct_autograd",
        "allow_finite_difference_first_derivative": False,
        "jacobian_convention": "squared_distance_delta",
        "units_per_shard": 1000,
    }
    for key, expected in exact_sampling.items():
        _expect(sampling[key], expected, context=f"sampling.{key}")
    _expect(
        sampling["expected_unit_count"],
        expected_conditions * 10 * 100,
        context="sampling.expected_unit_count",
    )
    baseline_index(config)

    policy = load_resource_policy()
    _assert_path_contract(config)
    if require_layout:
        _assert_stage_layout(protocol)

    conditions = condition_names(config)
    if len(conditions) != expected_conditions or len(set(conditions)) != len(conditions):
        raise ValueError("condition list is incomplete or contains duplicates")
    return {
        "protocol": protocol,
        "condition_count": len(conditions),
        "reference_attempt_count": int(reference["expected_attempt_count"]),
        "selected_reference_count": int(reference["expected_reference_count"]),
        "sampling_unit_count": int(sampling["expected_unit_count"]),
        "parameter_count": layout.parameter_count,
        "baseline_index": baseline_index(config),
        "objective_id": contract.objective_id,
        "objective_fingerprint": contract.fingerprint,
        "worker_environment": dict(policy.worker_environment),
        "max_cpu_threads": policy.max_cpu_threads,
        "max_concurrent_gpus": policy.max_concurrent_gpus,
    }


def validate_protocol(protocol: str, *, require_layout: bool = True) -> dict[str, Any]:
    return validate_config(load_config(protocol), require_layout=require_layout)


def validate_all(*, require_layout: bool = True) -> list[dict[str, Any]]:
    return [
        validate_protocol(protocol, require_layout=require_layout)
        for protocol in PROTOCOLS
    ]
