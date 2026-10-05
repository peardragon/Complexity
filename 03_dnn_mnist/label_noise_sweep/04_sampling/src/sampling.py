#!/usr/bin/env python3
"""Run one atomic exact-shell SMC sampling shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
DOMAIN_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "label_noise_sweep"
CONFIG_PATH = STAGE_ROOT / "config" / "default.json"
sys.path.insert(0, str(STAGE_ROOT / "src"))

from utils.resources import ResourcePolicy  # noqa: E402

RESOURCE_POLICY = ResourcePolicy.from_json(
    STAGE_ROOT / "config" / "resources.json"
)
RESOURCE_POLICY.install_worker_environment()

from utils.io_utils import load_json  # noqa: E402
from utils.production import resolve_replica_config  # noqa: E402
from utils.protocol import validate_config  # noqa: E402
from utils.sampling_control import sampling_production_guard  # noqa: E402
from utils.shell_smc import run_sampling_shard  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="production dataset replica in [0,9]",
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Compatibility option; existing files are always skipped.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--execute", "--execute-approved-full-run", dest="execute",
        action="store_true", help="Compute missing outputs; both spellings are equivalent.",
    )
    args = parser.parse_args()

    config = resolve_replica_config(
        load_json(CONFIG_PATH),
        args.dataset_index,
    )
    validation = validate_config(config)
    expected_shards = (
        int(config["sampling"]["expected_unit_count"])
        // int(config["sampling"]["units_per_shard"])
    )
    if args.shard_count != expected_shards:
        raise ValueError(f"--shard-count must be {expected_shards}")
    if not 0 <= args.shard_index < expected_shards:
        raise ValueError("--shard-index is outside the production range")
    plan = {
        "protocol": PROTOCOL,
        "dataset_index": args.dataset_index,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "device": args.device,
        "resume": bool(args.resume),
        "radius_grid": {"start": 0.01, "stop": 1.0, "count": 100, "delta": 0.01},
        "radial_derivative": config["sampling"]["radial_derivative"],
        "validation": validation,
    }
    if not args.execute:
        print(json.dumps({"status": "dry_run", "plan": plan}, indent=2))
        return 0

    environment = RESOURCE_POLICY.worker_env()
    RESOURCE_POLICY.assert_worker_environment(environment)
    with sampling_production_guard(PROTOCOL_ROOT, environment=environment):
        result = run_sampling_shard(
            config,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            resume=bool(args.resume),
        )
    print(json.dumps({"status": "completed", "plan": plan, "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
