"""Deterministic, gzip-compressed JSONL result shards."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .provenance import atomic_write_bytes, canonical_json_bytes, sha256_bytes


def render_jsonl_gzip(rows: Iterable[Mapping[str, Any]]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
        for row in rows:
            zipped.write(canonical_json_bytes(dict(row)))
            zipped.write(b"\n")
    return raw.getvalue()


def write_jsonl_gzip(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> str:
    encoded = render_jsonl_gzip(rows)
    atomic_write_bytes(path, encoded)
    return sha256_bytes(encoded)


def read_jsonl_gzip(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(Path(path), mode="rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: JSONL row must be an object"
                )
            yield value
