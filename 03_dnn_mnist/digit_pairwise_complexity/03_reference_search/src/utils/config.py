"""Strict configuration loading with no scientific defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .provenance import canonical_json_bytes, sha256_bytes


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def require_keys(
    value: Mapping[str, Any],
    keys: Iterable[str],
    *,
    context: str,
) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{context}: missing required keys {missing!r}")


def config_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))
