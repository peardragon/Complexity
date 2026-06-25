from __future__ import annotations

import argparse
from pathlib import Path

from two_pool_core import copy_summary_view, load_json, run_group_for_preset, run_id_from_config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run theory-local two-pool perceptron sampling validation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", choices=["smoke", "default"], default="smoke")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="Accepted for runner compatibility; execution is serial.")
    args = parser.parse_args()

    config_path = Path(args.config)
    manifest_path = run_pipeline(config_path, preset=args.preset, force=bool(args.force))
    config = load_json(config_path)
    run_group = run_group_for_preset(args.preset)
    run_id = run_id_from_config(config, args.preset)
    copy_summary_view(manifest_path.parents[0], run_group)
    print(manifest_path)
    print(f"run_group={run_group}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()

