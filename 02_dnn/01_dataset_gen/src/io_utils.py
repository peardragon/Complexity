from __future__ import annotations

import builtins
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, TextIO


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        except json.JSONDecodeError:
            continue
    return default


def save_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class VerbosePrintCapture:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        ensure_dir(log_path.parent)
        self.handle: TextIO = open(log_path, "a", encoding="utf-8", buffering=1)
        self._original_print = builtins.print

        def redirected_print(*args: Any, **kwargs: Any) -> None:
            options = dict(kwargs)
            target = options.get("file")
            if target is None or target is sys.stdout or target is sys.stderr:
                options["file"] = self.handle
            flush_requested = bool(options.pop("flush", False))
            self._original_print(*args, **options)
            if options.get("file") is self.handle or flush_requested:
                self.handle.flush()

        self._redirected_print = redirected_print
        self._original_print(f"[{now_iso()}] verbose log started", file=self.handle, flush=True)
        builtins.print = self._redirected_print

    def close(self) -> None:
        if builtins.print is self._redirected_print:
            builtins.print = self._original_print
        self.handle.flush()
        self.handle.close()


def start_verbose_print_capture(summary_root: Path, *, enabled: bool, filename: str = "verbose.log") -> VerbosePrintCapture | None:
    if not enabled:
        return None
    return VerbosePrintCapture(summary_root / "logs" / filename)


__all__ = [
    "ensure_dir",
    "timestamp",
    "now_iso",
    "save_json",
    "load_json",
    "save_csv",
    "load_csv_rows",
    "VerbosePrintCapture",
    "start_verbose_print_capture",
]
