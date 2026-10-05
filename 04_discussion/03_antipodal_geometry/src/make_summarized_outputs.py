#!/usr/bin/env python3
"""Aggregate Appendix D NPZs, with compact published CSV fallback."""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.discussion_pipeline import run_stage_03  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument("--output-dir", type=Path, help="Mirror stage outputs below this root.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Read the two antipodal NPZs from this explicit directory.",
    )
    arguments = parser.parse_args()
    run_stage_03(
        output_dir=arguments.output_dir,
        force=arguments.force,
        raw_dir=arguments.raw_dir,
    )
