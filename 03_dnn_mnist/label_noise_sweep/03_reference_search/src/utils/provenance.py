"""Stable hashes, collision-resistant task seeds, and atomic metadata writes."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


REQUIRED_RESUME_FINGERPRINT_KEYS = (
    "objective_fingerprint",
    "resolved_config_fingerprint",
    "dataset_fingerprint",
    "reference_fingerprint",
    "code_fingerprint",
    "task_fingerprint",
    "seed_fingerprint",
    "environment_fingerprint",
)
RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "torch": "torch",
    "scikit-learn": "scikit-learn",
    "Pillow": "Pillow",
    "networkx": "networkx",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def fingerprint_mapping(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))


def runtime_signature(
    distributions: Mapping[str, str] = RUNTIME_DISTRIBUTIONS,
) -> dict[str, Any]:
    """Return a deterministic signature of the active production runtime.

    Distribution metadata is inspected without importing numerical packages,
    so this helper neither initializes CUDA nor creates package-specific
    caches.  Missing dependencies remain explicit ``null`` entries and are
    therefore fingerprinted rather than silently omitted.
    """

    package_versions: dict[str, str | None] = {}
    for public_name, distribution_name in sorted(distributions.items()):
        try:
            package_versions[str(public_name)] = importlib_metadata.version(
                str(distribution_name)
            )
        except importlib_metadata.PackageNotFoundError:
            package_versions[str(public_name)] = None
    from .resources import REQUIRED_WORKER_ENVIRONMENT

    worker_environment = {
        key: os.environ.get(key)
        for key in sorted(REQUIRED_WORKER_ENVIRONMENT)
    }
    return {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": package_versions,
        "worker_environment": worker_environment,
    }


def runtime_fingerprint(
    distributions: Mapping[str, str] = RUNTIME_DISTRIBUTIONS,
) -> str:
    """Fingerprint :func:`runtime_signature` with canonical JSON SHA-256."""

    return fingerprint_mapping(runtime_signature(distributions))


def stable_seed(
    *parts: Any,
    namespace: str,
    bits: int = 63,
) -> int:
    if not namespace:
        raise ValueError("a non-empty seed namespace is required")
    if bits < 32 or bits > 64:
        raise ValueError("bits must lie in [32, 64]")
    payload = {
        "namespace": namespace,
        "parts": list(parts),
        "version": 1,
    }
    raw = hashlib.sha256(canonical_json_bytes(payload)).digest()
    integer = int.from_bytes(raw[:8], byteorder="big", signed=False)
    mask = (1 << bits) - 1 if bits < 64 else (1 << 64) - 1
    seed = integer & mask
    return seed if seed != 0 else 1


def require_unique(values: Iterable[Any], *, label: str) -> None:
    seen: set[Any] = set()
    duplicates: list[Any] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
            if len(duplicates) >= 5:
                break
        seen.add(value)
    if duplicates:
        raise ValueError(f"{label} contains duplicates; examples={duplicates!r}")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    rendered = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    atomic_write_bytes(path, (rendered + "\n").encode("utf-8"))


def validate_fingerprint_record(
    value: Mapping[str, Any],
    *,
    required_keys: Iterable[str] = REQUIRED_RESUME_FINGERPRINT_KEYS,
) -> None:
    """Require a complete set of lowercase SHA-256 fingerprints.

    A concept that does not apply to one artifact must still be represented by
    the SHA-256 of a canonical ``{"state": "not_applicable"}`` marker.  Empty
    strings and ad-hoc sentinel text are not accepted.
    """

    errors: list[str] = []
    for key in required_keys:
        raw = value.get(key)
        if not isinstance(raw, str):
            errors.append(f"{key}: missing or not a string")
            continue
        if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
            errors.append(f"{key}: not a lowercase SHA-256 digest")
    if errors:
        raise ValueError(
            "incomplete artifact fingerprint record: " + "; ".join(errors)
        )
