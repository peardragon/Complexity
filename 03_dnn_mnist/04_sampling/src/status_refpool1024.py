#!/usr/bin/env python3
"""Print current status for the refpool1024 mechanical sampling run."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_RUN_ROOT = "/home/bjyong/Complexity/local_project/03_dnn_mnist/04_sampling/raw_outputs/refpool1024_all_radii_60ref"
RUN_ROOT = Path(os.environ.get("REFPOOL1024_RUN_ROOT", DEFAULT_RUN_ROOT))
TARGET_REFS = int(os.environ.get("REFPOOL1024_TARGET_REFS", "60"))
RADIUS_COUNT = int(os.environ.get("REFPOOL1024_RADIUS_COUNT", "25"))
EXPECTED_UNITS = int(os.environ.get("REFPOOL1024_EXPECTED_UNITS", str(4 * TARGET_REFS * RADIUS_COUNT)))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def unit_root() -> Path:
    return RUN_ROOT / "05_pool2_pm_sais_sampling" / "unit_summaries"


def file_paths(pattern: str) -> list[Path]:
    root = unit_root()
    return list(root.rglob(pattern)) if root.exists() else []


def count_files(pattern: str) -> int:
    return len(file_paths(pattern))


def rate_line(paths: list[Path], window_s: int) -> str:
    now = time.time()
    count = sum(1 for path in paths if now - path.stat().st_mtime <= window_s)
    rate = count / (window_s / 60.0)
    if rate <= 0:
        return f"last_{window_s // 60}min: {count} units, rate unavailable"
    remaining = max(0, EXPECTED_UNITS - len(paths))
    eta_minutes = remaining / rate
    eta_hours = int(eta_minutes // 60)
    eta_mins = int(round(eta_minutes % 60))
    return f"last_{window_s // 60}min: {count} units, {rate:.2f} units/min, eta ~{eta_hours}h {eta_mins}m"


def running_entries() -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for line in read_text(RUN_ROOT / "RUNNING_PIDS.txt").splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            entries.append((int(parts[0]), parts[1] if len(parts) > 1 else f"pid_{parts[0]}"))
    return entries


def ps_for(pids: list[int]) -> str:
    if not pids:
        return ""
    cmd = ["ps", "-p", ",".join(str(pid) for pid in pids), "-o", "pid,psr,stat,pcpu,pmem,etime,cmd"]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return proc.stdout.strip()


def gpu_status() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else "nvidia-smi unavailable"


def tail(path: Path, n: int = 1) -> str:
    lines = read_text(path).splitlines()
    return "\n".join(lines[-n:])


def main() -> int:
    unit_paths = file_paths("unit_summary.json")
    units = len(unit_paths)
    npz = count_files("samples.npz")
    entries = running_entries()
    pids = [pid for pid, _label in entries]
    alive = []
    for pid in pids:
        proc = subprocess.run(["kill", "-0", str(pid)], check=False, capture_output=True)
        if proc.returncode == 0:
            alive.append(pid)

    print(f"run_root: {RUN_ROOT}")
    print(f"target_refs: {TARGET_REFS}")
    print(f"radius_count: {RADIUS_COUNT}")
    print(f"units: {units}/{EXPECTED_UNITS} ({100.0 * units / EXPECTED_UNITS:.2f}%)")
    print(f"samples_npz: {npz}/{EXPECTED_UNITS} ({'match' if units == npz else 'mismatch'})")
    print(f"alive_shards: {len(alive)}/{len(pids)}")
    print("rates:")
    print(rate_line(unit_paths, 5 * 60))
    print(rate_line(unit_paths, 10 * 60))
    print()
    print("manager_status:")
    print(read_text(RUN_ROOT / "MANAGER_STATUS.txt").strip() or "(missing)")
    print()
    print("monitor_status:")
    print(read_text(RUN_ROOT / "MONITOR_STATUS.txt").strip() or "(missing)")
    print()
    print("sampling_status_json:")
    status_path = RUN_ROOT / "SAMPLING_STATUS.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print("(missing)")
    print()
    print("processes:")
    print(ps_for(pids) or "(none)")
    print()
    print("gpu_status:")
    print(gpu_status())
    print()
    print("recent_shard_lines:")
    for _pid, label in entries:
        path = RUN_ROOT / "logs" / f"{label}.log"
        print(f"{label}: {tail(path) or '(no log)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
