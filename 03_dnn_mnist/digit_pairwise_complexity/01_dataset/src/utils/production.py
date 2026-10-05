"""Resolve one of the ten standalone production dataset replicas."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def resolve_replica_config(
    source: Mapping[str, Any],
    dataset_index: int,
) -> dict[str, Any]:
    """Return a stage-local config for one production dataset.

    All paths remain relative to ``03_dnn_mnist``.  The shell scale is applied
    exactly once through the effective objective already consumed by the
    reference-search and sampling kernels.
    """

    index = int(dataset_index)
    if not 0 <= index <= 9:
        raise ValueError("dataset_index must lie in [0, 9]")
    config = deepcopy(dict(source))
    protocol = str(config["protocol"])
    dataset = config["dataset"]
    dataset["replica_index"] = index
    dataset["split_seed"] = 20260610 + index
    if protocol == "label_noise_sweep":
        dataset["flip_uniform_master_seed"] = 2026065000 + 1000 * index
        reference_prefix = "mnist-label-reference-v1"
        sampling_prefix = "mnist-label-shell-v1"
    elif protocol == "digit_pairwise_sweep":
        reference_prefix = "mnist-pair-reference-v1"
        sampling_prefix = "mnist-pair-shell-v1"
    else:
        raise ValueError(f"unsupported protocol: {protocol}")
    # Replica 000 is the original production dataset and therefore owns the
    # unsuffixed seed namespace.  Replicas 001..009 were added later with
    # explicitly suffixed namespaces.  Keeping that historical/canonical
    # distinction is required to reproduce the retained exact references and
    # shell particles byte-for-byte.
    config["reference_search"]["seed_namespace"] = (
        reference_prefix
        if index == 0
        else f"{reference_prefix}-dataset-{index:02d}"
    )
    config["sampling"]["seed_namespace"] = (
        sampling_prefix
        if index == 0
        else f"{sampling_prefix}-dataset-{index:02d}"
    )
    directory = (
        "label_noise_sweep"
        if protocol == "label_noise_sweep"
        else "digit_pairwise_complexity"
    )
    dataset_name = f"dataset_{index:03d}"
    config["paths"] = {
        "dataset_root": (
            f"{directory}/01_dataset/raw_outputs/datasets/{dataset_name}"
        ),
        "complexity_root": (
            f"{directory}/02_complexity_measure/summarized_outputs"
        ),
        "reference_root": (
            f"{directory}/03_reference_search/raw_outputs/references/"
            f"{dataset_name}"
        ),
        "manifest_root": (
            f"{directory}/03_reference_search/raw_outputs/manifests/"
            f"{dataset_name}"
        ),
        "sampling_shard_root": (
            f"{directory}/04_sampling/raw_outputs/shell_beta_100/"
            f"{dataset_name}"
        ),
        "summary_root": (
            f"{directory}/05_proxy_local_entropy/summarized_outputs"
        ),
    }
    return config
