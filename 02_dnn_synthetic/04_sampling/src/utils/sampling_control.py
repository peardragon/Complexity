"""Prevent two sampling commands from writing shards at the same time."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator, MutableMapping


class SamplingProductionLockError(RuntimeError):
    """Another sampling command currently owns this experiment's lock."""


@contextmanager
def sampling_production_guard(
    project_root: str | Path,
    *,
    environment: MutableMapping[str, str] | None = None,
    lock_path: str | Path | None = None,
) -> Iterator[Path]:
    """Acquire a nonblocking advisory lock without changing the environment.

    Keep the lock file after release: unlinking it would let a later process
    lock a different inode while an earlier process still holds the old one.
    The environment argument remains accepted by the standalone CLI.
    """
    root = Path(project_root).resolve()
    target = (root / ".sampling_production.lock" if lock_path is None else Path(lock_path)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"sampling lock must stay within {root}: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SamplingProductionLockError(f"another sampling command owns {target}") from exc
        try:
            yield target
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
