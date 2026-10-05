#!/usr/bin/env python3
"""Production entrypoint for schedule creation and one revised sampling shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[1]
UTILS = STAGE_ROOT / "src" / "utils"
sys.path.insert(0, str(UTILS))

from run_sampling_shard import check_only, execute_shard
from schedule import DEFAULT_CONFIG, project_path, write_schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=336)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = project_path(args.config)
    if args.prepare:
        result = write_schedule(config_path)
    else:
        if args.shard_index is None:
            parser.error("--shard-index is required unless --prepare is used")
        if args.execute:
            result = execute_shard(
                config_path=config_path,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                resume=args.resume,
            )
        else:
            result = check_only(
                config_path=config_path,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
