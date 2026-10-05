from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Mapping


STAGE_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_ROOT = Path(__file__).resolve().parents[3]
LOCAL_SOURCE_ROOT = STAGE_ROOT / "src" / "utils"
DEFAULT_PROTOCOL_PATH = STAGE_ROOT / "config" / "default.json"
EXPECTED_BETAS = tuple(round(0.05 + 0.02 * i, 2) for i in range(18))
EXPECTED_RADII = tuple(round(0.01 * i, 2) for i in range(1, 251))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def protocol_fingerprint(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(config))


@lru_cache(maxsize=4)
def source_fingerprint(source_root: Path | None = None) -> str:
    digest = hashlib.sha256()
    if source_root is None:
        local_roots = [STAGE_ROOT / "src"]
        labelled_paths = [
            (f"02_dnn_synthetic/{path.relative_to(SYNTHETIC_ROOT).as_posix()}", path)
            for root in local_roots
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        ]
    else:
        local_root = Path(source_root).resolve()
        labelled_paths = [
            (path.relative_to(local_root).as_posix(), path)
            for path in local_root.rglob("*.py")
            if "__pycache__" not in path.parts
        ]
    if not labelled_paths:
        raise FileNotFoundError("no production source files found")
    for label, path in sorted(labelled_paths):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _normalized_stage(stage: str) -> str:
    aliases = {
        "dataset": "dataset",
        "synthetic_dataset": "dataset",
        "reference": "reference",
        "synthetic_reference_search": "reference",
        "shell": "shell",
        "synthetic_shell_shard": "shell",
    }
    try:
        return aliases[str(stage)]
    except KeyError as exc:
        raise ValueError(f"unsupported synthetic provenance stage: {stage!r}") from exc


def stage_config_projection(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Return only the resolved configuration that can affect one stage.

    Dataset provenance intentionally excludes optimizer and shell settings so a
    reference-only revision does not relabel an unchanged fresh dataset.
    Downstream stages retain the upstream payload hash separately in their
    strict-eight record.
    """

    normalized = _normalized_stage(stage)
    common = {
        "schema_version": int(config["schema_version"]),
        "protocol_id": str(config["protocol_id"]),
        "stage": normalized,
        "seeds": {
            "scheme": str(config["seeds"]["scheme"]),
            "global_seed": int(config["seeds"]["global_seed"]),
        },
    }
    if normalized == "dataset":
        return {**common, "dataset": dict(config["dataset"])}
    if normalized == "reference":
        return {
            **common,
            "dataset": dict(config["dataset"]),
            "model": dict(config["model"]),
            "objective": dict(config["objective"]),
            "reference_search": dict(config["reference_search"]),
        }
    return {
        **common,
        "dataset_axes": {
            "beta_values": list(config["dataset"]["beta_values"]),
            "datasets_per_beta": int(config["dataset"]["datasets_per_beta"]),
        },
        "model": dict(config["model"]),
        "objective": dict(config["objective"]),
        "reference_search": dict(config["reference_search"]),
        "shell": dict(config["shell"]),
    }


def stage_config_fingerprint(config: Mapping[str, Any], stage: str) -> str:
    return sha256_bytes(canonical_json_bytes(stage_config_projection(config, stage)))


def stage_source_paths(stage: str) -> tuple[Path, ...]:
    """Return the code closure intentionally committed by a stage artifact."""

    normalized = _normalized_stage(stage)
    origin_stage = (
        SYNTHETIC_ROOT / "03_reference_search"
        if normalized == "reference"
        else STAGE_ROOT
    )
    origin_source = origin_stage / "src" / "utils"
    local = origin_source
    shared = origin_source
    names = {
        "dataset": (
            local / "config.py",
            local / "dataset.py",
            local / "fingerprints.py",
            local / "io.py",
            local / "manifests.py",
            local / "runtime.py",
            local / "seeds.py",
            shared / "provenance.py",
            shared / "resources.py",
        ),
        "reference": (
            local / "config.py",
            local / "fingerprints.py",
            local / "reference.py",
            local / "objective.py",
            local / "model.py",
            local / "io.py",
            local / "manifests.py",
            local / "runtime.py",
            local / "seeds.py",
            shared / "objective_contract.py",
            shared / "provenance.py",
            shared / "resources.py",
        ),
        "shell": (
            local / "config.py",
            local / "dataset.py",
            local / "fingerprints.py",
            local / "reference.py",
            local / "smc.py",
            local / "vmf.py",
            local / "objective.py",
            local / "model.py",
            local / "io.py",
            local / "manifests.py",
            local / "runtime.py",
            local / "seeds.py",
            shared / "objective_contract.py",
            shared / "provenance.py",
            shared / "radial.py",
            shared / "resources.py",
        ),
    }[normalized]
    missing = [path for path in names if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {normalized} stage source files: {missing}")
    return names


def stage_source_fingerprint(stage: str) -> str:
    digest = hashlib.sha256()
    domain_root = SYNTHETIC_ROOT.resolve()
    for path in sorted(stage_source_paths(stage)):
        resolved = path.resolve()
        label = resolved.relative_to(domain_root).as_posix()
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def radii_from_config(config: Mapping[str, Any]) -> tuple[float, ...]:
    spec = config["shell"]["radii"]
    count = int(spec["count"])
    start = float(spec["start"])
    step = float(spec["step"])
    radii = tuple(round(start + step * i, 12) for i in range(count))
    if abs(radii[-1] - float(spec["stop"])) > 1.0e-12:
        raise ValueError("radius start/step/count does not end at configured stop")
    return radii


def validate_protocol(config: Mapping[str, Any]) -> None:
    """Check the numerical dimensions used by the paper, without hash gates."""
    dataset, model, shell = config["dataset"], config["model"], config["shell"]
    if tuple(dataset["beta_values"]) != EXPECTED_BETAS:
        raise ValueError("expected the 18 paper beta conditions")
    if (dataset["datasets_per_beta"], dataset["n_train"]) != (60, 512):
        raise ValueError("expected 60 datasets of 512 samples per condition")
    if model["parameter_count"] != 2545 or model["hidden_widths"] != [48, 48]:
        raise ValueError("expected the paper's 2-48-48-1 model")
    if config["reference_search"]["references_per_dataset"] != 10:
        raise ValueError("expected ten references per dataset")
    if radii_from_config(config) != EXPECTED_RADII:
        raise ValueError("expected radii 0.01 through 2.50")
    particles = shell["particles"]
    if (particles["total"], particles["independent_splits"], particles["per_split"]) != (1024, 2, 512):
        raise ValueError("expected two pools of 512 particles")
    from utils.objective_contract import ObjectiveContract
    objective = ObjectiveContract.from_json(STAGE_ROOT / config["objective"]["contract_path"])
    if (objective.gamma_ce, objective.lambda_reg) != (100.0, 1.0):
        raise ValueError("the shell beta must be applied exactly once")


def load_protocol(path: Path | str | None = None) -> dict[str, Any]:
    resolved = Path(path or DEFAULT_PROTOCOL_PATH).resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"protocol must be a JSON object: {resolved}")
    validate_protocol(config)
    return config
