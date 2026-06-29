#!/usr/bin/env python3
"""Keep refpool1024 shard CPU affinity inside a fixed logical CPU cap."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path


PROGRESS_RE = re.compile(
    r"shard=(?P<shard>\d+)/(?P<total>\d+) "
    r"unit=(?P<unit>\d+)/(?P<unit_total>\d+) "
    r"rule=(?P<rule>[^ ]+) ref=(?P<ref>\d+) r=(?P<radius>[0-9.]+)"
)


def load_pids(run_root: Path) -> dict[int, int]:
    pid_file = run_root / "RUNNING_PIDS.txt"
    out: dict[int, int] = {}
    if not pid_file.exists():
        return out
    for line in pid_file.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        match = re.search(r"pinned_(\d+)", parts[1])
        if match:
            out[int(match.group(1))] = pid
    return out


def load_shard_count(run_root: Path, fallback: int) -> int:
    status_path = run_root / "MANAGER_STATUS.txt"
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("shards="):
                try:
                    return int(line.split("=", 1)[1])
                except ValueError:
                    pass
    pid_file = run_root / "RUNNING_PIDS.txt"
    if pid_file.exists():
        counts: list[int] = []
        for line in pid_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"shard(\d+)_pinned_", line)
            if match:
                counts.append(int(match.group(1)))
        if counts:
            return max(counts)
    return int(fallback)


def latest_progress(run_root: Path, shard: int, shard_count: int) -> int:
    latest = -1
    exact_path = run_root / "logs" / f"shard{shard_count}_pinned_{shard}.log"
    paths = [exact_path] if exact_path.exists() else sorted((run_root / "logs").glob(f"shard*_pinned_{shard}.log"))
    if not paths:
        return latest
    path = paths[-1]
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[refpool1024]" not in line:
            continue
        match = PROGRESS_RE.search(line)
        if match:
            latest = int(match.group("unit"))
    return latest


def process_alive(pid: int) -> bool:
    return Path("/proc", str(pid)).exists()


def tids_for(pid: int) -> list[str]:
    task_dir = Path("/proc", str(pid), "task")
    if not task_dir.exists():
        return []
    return sorted(path.name for path in task_dir.iterdir())


def compute_assignments(
    pids: dict[int, int],
    run_root: Path,
    max_logical_cpus: int,
    shard_count: int | None = None,
) -> dict[int, str]:
    shards = sorted(pids)
    if not shards or max_logical_cpus <= 0:
        return {}
    if shard_count is None:
        shard_count = load_shard_count(run_root, len(shards))
    if max_logical_cpus < len(shards):
        return {shard: str(idx % max_logical_cpus) for idx, shard in enumerate(shards)}
    two_core_count = max(0, min(len(shards), max_logical_cpus - len(shards)))
    progress = [(latest_progress(run_root, shard, int(shard_count)), shard) for shard in shards]
    two_core_shards = {shard for _unit, shard in sorted(progress)[:two_core_count]}
    assignments: dict[int, str] = {}
    cursor = 0
    for shard in sorted(two_core_shards):
        assignments[shard] = f"{cursor},{cursor + 1}"
        cursor += 2
    for shard in shards:
        if shard in two_core_shards:
            continue
        assignments[shard] = str(cursor)
        cursor += 1
    return assignments


def apply_assignments(pids: dict[int, int], assignments: dict[int, str], dry_run: bool) -> list[str]:
    lines: list[str] = []
    for shard in sorted(assignments):
        pid = pids.get(shard)
        cpus = assignments[shard]
        if pid is None or not process_alive(pid):
            lines.append(f"shard={shard:02d} pid={pid} cpus={cpus} skipped=dead")
            continue
        tids = tids_for(pid)
        ok = 0
        fail = 0
        if not dry_run:
            for tid in tids:
                result = subprocess.run(["taskset", "-pc", cpus, tid], text=True, capture_output=True)
                if result.returncode == 0:
                    ok += 1
                else:
                    fail += 1
        lines.append(
            f"shard={shard:02d} pid={pid} tids={len(tids)} cpus={cpus} "
            f"dry_run={int(dry_run)} ok={ok} fail={fail}"
        )
    return lines


def rebalance_once(run_root: Path, max_logical_cpus: int, dry_run: bool) -> bool:
    pids = {shard: pid for shard, pid in load_pids(run_root).items() if process_alive(pid)}
    if not pids:
        print("no_alive_shards", flush=True)
        return False
    shard_count = load_shard_count(run_root, len(pids))
    assignments = compute_assignments(pids, run_root, max_logical_cpus, shard_count)
    print(f"checked_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print(f"alive_shards={len(pids)} shard_count={shard_count} max_logical_cpus={max_logical_cpus}", flush=True)
    for line in apply_assignments(pids, assignments, dry_run):
        print(line, flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--max-logical-cpus", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    while True:
        alive = rebalance_once(run_root, int(args.max_logical_cpus), bool(args.dry_run))
        if args.once or not alive:
            return 0
        time.sleep(float(args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
