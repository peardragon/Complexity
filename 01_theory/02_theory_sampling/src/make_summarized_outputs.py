#!/usr/bin/env python3
"""Verify the released compact profile or aggregate a newly generated run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


STAGE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = STAGE_ROOT / "summarized_outputs" / "phi_by_sampling.csv"


def verify_summary() -> dict[str, object]:
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dimensions = sorted({int(row["N"]) for row in rows})
    if len(rows) != 336 or dimensions != [40, 80, 160, 320]:
        raise RuntimeError("theory sampling summary coverage is incomplete")
    if any(
        row["first_derivative_method"] != "direct_particle_radial_score"
        for row in rows
    ):
        raise RuntimeError("theory sampling summary does not store direct derivatives")
    return {
        "status": "verified_compact_summary",
        "rows": len(rows),
        "dimensions": dimensions,
        "raw_sampling_artifacts_opened": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--config", type=Path, default=STAGE_ROOT / "config/default.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(STAGE_ROOT / "src/utils"))
    from aggregate import main as aggregate_main, validate_summary
    if args.aggregate:
        forwarded = ["--config", str(args.config)]
        if args.output_dir is not None: forwarded += ["--output-dir", str(args.output_dir)]
        if args.force: forwarded.append("--force")
        if args.check_only: forwarded.append("--check-only")
        aggregate_main(forwarded)
        return
    path = (args.output_dir / "phi_by_sampling.csv") if args.output_dir else SUMMARY
    validate_summary(path, json.loads(args.config.read_text()))
    print(json.dumps({"status": "validated", "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
