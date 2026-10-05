#!/usr/bin/env python3
"""Read-only validation of Discussion outputs and semantic invariants."""

import argparse
from pathlib import Path

from utils.discussion_pipeline import validate_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Validate a mirrored four-stage output tree.",
    )
    parser.add_argument(
        "--published-benchmarks",
        action="store_true",
        help="Also require the two Appendix D values quoted in arXiv v1.",
    )
    arguments = parser.parse_args()
    benchmark_check = True if arguments.published_benchmarks else None
    report = validate_all(
        output_dir=arguments.output_dir,
        check_published_benchmarks=benchmark_check,
    )
    print(
        "Discussion pipeline validation passed: "
        f"{report['checked_stages']} stages, "
        f"{report['checked_output_files']} output files."
    )
