#!/usr/bin/env python3
"""Run fresh-random-initialization exact-reference search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
DOMAIN_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    "label_noise_sweep"
    if PROTOCOL_ROOT.name == "label_noise_sweep"
    else "digit_pairwise_sweep"
)
CONFIG_PATH = STAGE_ROOT / "config" / "default.json"
sys.path.insert(0, str(STAGE_ROOT / "src"))

from utils.resources import ResourcePolicy  # noqa: E402

RESOURCE_POLICY = ResourcePolicy.from_json(
    STAGE_ROOT / "config" / "resources.json"
)
RESOURCE_POLICY.install_worker_environment()

from utils.io_utils import load_json  # noqa: E402
from utils.production import resolve_replica_config  # noqa: E402
from utils.protocol import condition_names, validate_config  # noqa: E402
from utils.reference_training import run_reference_search  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="production dataset replica in [0,9]",
    )
    parser.add_argument("--condition")
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
    selected = condition_names(config) if args.condition is None else [args.condition]
    unknown = sorted(set(selected) - set(condition_names(config)))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    plan = {
        "protocol": PROTOCOL,
        "dataset_index": args.dataset_index,
        "conditions": selected,
        "device": args.device,
        "resume": bool(args.resume),
        "initialization": config["reference_search"]["initialization"],
        "acceptance_policy": config["reference_search"]["acceptance_policy"],
        "validation": validation,
    }
    if not args.execute:
        print(json.dumps({"status": "dry_run", "plan": plan}, indent=2))
        return 0

    result = run_reference_search(
        config,
        condition=args.condition,
        device=args.device,
        resume=bool(args.resume),
    )
    print(json.dumps({"status": "completed", "plan": plan, "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
