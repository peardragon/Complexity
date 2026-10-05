#!/usr/bin/env python3
"""Resolve label-noise radial sign topology and positive intervals."""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.discussion_pipeline import run_stage_04  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument("--output-dir", type=Path, help="Mirror stage outputs below this root.")
    arguments = parser.parse_args()
    run_stage_04(output_dir=arguments.output_dir, force=arguments.force)
