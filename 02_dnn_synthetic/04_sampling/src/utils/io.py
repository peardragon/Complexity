from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .config import canonical_json_bytes


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path | str, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_gzip_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                for row in rows:
                    compressed.write(canonical_json_bytes(row))
                    compressed.write(b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_gzip_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            yield value


def matching_header(path: Path | str, expected: Mapping[str, Any]) -> bool:
    target = Path(path)
    if not target.exists():
        return False
    try:
        header = next(read_gzip_jsonl(target))
    except (OSError, StopIteration, json.JSONDecodeError, TypeError):
        return False
    return header.get("record_type") == "shard_header" and all(header.get(key) == value for key, value in expected.items())
