from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    tmp.replace(path)
