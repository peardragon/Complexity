#!/usr/bin/env python3
"""Run all deterministic Discussion summary adapters in stage order."""

import argparse
from pathlib import Path

from utils.discussion_pipeline import run_all, validate_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Mirror the four stage outputs below this directory.",
    )
    arguments = parser.parse_args()
    run_all(output_dir=arguments.output_dir, force=arguments.force)
    report = validate_all(output_dir=arguments.output_dir)
    print(
        "Discussion outputs available and validated: "
        f"{report['checked_stages']} stages, "
        f"{report['checked_output_files']} output files."
    )
