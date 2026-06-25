#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weakens_benchmark.config import load_config  # noqa: E402
from weakens_benchmark.run import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the weakens proxy sampling benchmark.")
    parser.add_argument(
        "--config",
        default="01_dataset_proxy/config/default.json",
        help="Path to JSON config, relative to the weakens root unless absolute.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_experiment(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

