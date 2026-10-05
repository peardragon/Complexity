from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = Path(__file__).resolve().parents[3]
MNIST_ROOT = Path(__file__).resolve().parents[4]
# Output paths are resolved against the MNIST domain so each stage can consume
# the preceding sibling stage.  Source, config, and contracts remain local to
# the current stage.
REVISED_ROOT = MNIST_ROOT
PROTOCOL_DIRECTORY_TO_ID = {
    "label_noise_sweep": "label_noise_sweep",
    "digit_pairwise_complexity": "digit_pairwise_sweep",
}
try:
    ACTIVE_PROTOCOL = PROTOCOL_DIRECTORY_TO_ID[PROTOCOL_ROOT.name]
except KeyError as exc:
    raise RuntimeError(f"unknown production protocol directory: {PROTOCOL_ROOT}") from exc
PROTOCOLS = (ACTIVE_PROTOCOL,)

from .provenance import (
    canonical_json_bytes as _shared_canonical_json_bytes,
    fingerprint_mapping,
    sha256_file,
    stable_seed as _shared_stable_seed,
)


def canonical_json_bytes(payload: Any) -> bytes:
    return _shared_canonical_json_bytes(payload)


def payload_sha256(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return fingerprint_mapping(payload)


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    del chunk_size
    return sha256_file(path)


def stable_seed(namespace: str, key: str) -> int:
    return _shared_stable_seed(key, namespace=namespace, bits=63)


def stable_id(namespace: str, key: str, *, length: int = 24) -> str:
    digest = hashlib.sha256(f"{namespace}\0{key}".encode("utf-8")).hexdigest()
    return digest[: int(length)]


def protocol_config_path(protocol: str) -> Path:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    return STAGE_ROOT / "config" / "default.json"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def load_config(protocol: str) -> dict[str, Any]:
    return load_json(protocol_config_path(protocol))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (MNIST_ROOT / path).resolve()
    try:
        resolved.relative_to(MNIST_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes revised MNIST root: {value}") from exc
    return resolved


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    content = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_replace_bytes(Path(path), content)


def csv_bytes(
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return stream.getvalue().encode("utf-8")


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    _atomic_replace_bytes(Path(path), csv_bytes(rows, fieldnames=fieldnames))


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _atomic_replace_bytes(Path(path), stream.getvalue())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def hash_rows(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    return hashlib.sha256(csv_bytes(rows, fieldnames=fieldnames)).hexdigest()


def require_exact_keys(payload: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")
