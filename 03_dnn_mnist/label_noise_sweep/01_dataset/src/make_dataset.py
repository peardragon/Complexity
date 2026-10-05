#!/usr/bin/env python3
"""Generate or verify the protocol's canonical MNIST datasets."""

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

from utils.datasets import generate_label_noise  # noqa: E402
from utils.io_utils import load_json  # noqa: E402
from utils.production import resolve_replica_config  # noqa: E402
from utils.protocol import validate_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="production dataset replica in [0,9]",
    )
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
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "dataset_index": args.dataset_index,
                    "validation": validation,
                },
                indent=2,
            )
        )
        return 0

    RESOURCE_POLICY.assert_worker_environment()
    result = generate_label_noise(config, resume=bool(args.resume))
    print(json.dumps({"status": "completed", "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
