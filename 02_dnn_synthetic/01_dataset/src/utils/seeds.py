from __future__ import annotations


SEED_BLOCK_SIZE = 100_000_000
GLOBAL_STRIDE = 1_000_000_000
STAGE_BLOCK = {
    "dataset": 1,
    "reference": 2,
    "shell": 3,
}


def _coordinate_dataset_count(config: dict) -> int:
    count = int(config["seeds"]["coordinate_universe_datasets_per_beta"])
    if count != 90:
        raise ValueError("Synthetic seed coordinates must retain the 90-slot universe")
    if int(config["dataset"]["datasets_per_beta"]) > count:
        raise ValueError("materialized dataset count exceeds the seed-coordinate universe")
    return count


def _seed_from_linear(config: dict, stage: str, linear_index: int, stage_cardinality: int) -> int:
    if config["seeds"]["scheme"] != "injective_stage_block_mixed_radix_uint63_v1":
        raise ValueError("unknown/non-injective production seed scheme")
    index = int(linear_index)
    cardinality = int(stage_cardinality)
    if not 0 <= index < cardinality:
        raise IndexError(f"{stage} seed index {index} outside [0,{cardinality})")
    if cardinality > SEED_BLOCK_SIZE:
        raise ValueError(f"{stage} cardinality exceeds its injective seed block")
    global_component = int(config["seeds"]["global_seed"]) * GLOBAL_STRIDE
    seed = global_component + STAGE_BLOCK[stage] * SEED_BLOCK_SIZE + index
    if not 0 < seed < (1 << 63):
        raise OverflowError("injective production seed lies outside positive uint63")
    return seed


def dataset_seed(config: dict, beta_index: int, dataset_id: int) -> int:
    beta_count = len(config["dataset"]["beta_values"])
    materialized_count = int(config["dataset"]["datasets_per_beta"])
    dataset_count = _coordinate_dataset_count(config)
    if not 0 <= int(beta_index) < beta_count or not 0 <= int(dataset_id) < materialized_count:
        raise IndexError("dataset seed coordinates outside configured axes")
    linear = int(beta_index) * dataset_count + int(dataset_id)
    return _seed_from_linear(config, "dataset", linear, beta_count * dataset_count)


def reference_seed(config: dict, beta_index: int, dataset_id: int, attempt_id: int) -> int:
    beta_count = len(config["dataset"]["beta_values"])
    materialized_count = int(config["dataset"]["datasets_per_beta"])
    dataset_count = _coordinate_dataset_count(config)
    attempt_count = int(config["reference_search"]["max_attempts_per_dataset"])
    if (
        not 0 <= int(beta_index) < beta_count
        or not 0 <= int(dataset_id) < materialized_count
        or not 0 <= int(attempt_id) < attempt_count
    ):
        raise IndexError("reference seed coordinates outside configured axes")
    linear_dataset = int(beta_index) * dataset_count + int(dataset_id)
    linear = linear_dataset * attempt_count + int(attempt_id)
    return _seed_from_linear(
        config,
        "reference",
        linear,
        beta_count * dataset_count * attempt_count,
    )


def shell_split_seed(
    config: dict,
    beta_index: int,
    dataset_id: int,
    ref_id: int,
    radius_index: int,
    split_id: int,
) -> int:
    beta_count = len(config["dataset"]["beta_values"])
    materialized_count = int(config["dataset"]["datasets_per_beta"])
    dataset_count = _coordinate_dataset_count(config)
    reference_count = int(config["reference_search"]["references_per_dataset"])
    radius_count = int(config["shell"]["radii"]["count"])
    if (
        not 0 <= int(beta_index) < beta_count
        or not 0 <= int(dataset_id) < materialized_count
        or not 0 <= int(ref_id) < reference_count
        or not 0 <= int(radius_index) < radius_count
        or not 0 <= int(split_id) < 2
    ):
        raise IndexError("shell seed coordinates outside configured axes")
    linear = (
        (
            (
                int(beta_index) * dataset_count
                + int(dataset_id)
            )
            * reference_count
            + int(ref_id)
        )
        * radius_count
        + int(radius_index)
    ) * 2 + int(split_id)
    cardinality = beta_count * dataset_count * reference_count * radius_count * 2
    return _seed_from_linear(config, "shell", linear, cardinality)


def seed_proof(config: dict) -> dict:
    dataset_cardinality = len(config["dataset"]["beta_values"]) * _coordinate_dataset_count(config)
    reference_cardinality = dataset_cardinality * int(config["reference_search"]["max_attempts_per_dataset"])
    shell_cardinality = (
        dataset_cardinality
        * int(config["reference_search"]["references_per_dataset"])
        * int(config["shell"]["radii"]["count"])
        * 2
    )
    cardinalities = {
        "dataset": dataset_cardinality,
        "reference": reference_cardinality,
        "shell": shell_cardinality,
    }
    intervals = {}
    for stage, cardinality in cardinalities.items():
        start = _seed_from_linear(config, stage, 0, cardinality)
        stop_inclusive = _seed_from_linear(config, stage, cardinality - 1, cardinality)
        intervals[stage] = {
            "cardinality": cardinality,
            "first_seed": start,
            "last_seed": stop_inclusive,
            "contiguous": True,
        }
    ordered = [intervals[name] for name in ("dataset", "reference", "shell")]
    if any(left["last_seed"] >= right["first_seed"] for left, right in zip(ordered, ordered[1:])):
        raise RuntimeError("stage seed intervals overlap")
    return {
        "scheme": config["seeds"]["scheme"],
        "proof": "injective mixed-radix coordinate linearization inside three disjoint contiguous stage blocks",
        "intervals": intervals,
        "shell_split_adjacency": "split seed 1 equals split seed 0 plus one for every shell unit",
        "all_seeds_positive_uint63": ordered[-1]["last_seed"] < (1 << 63),
    }
