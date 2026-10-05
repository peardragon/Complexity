#!/usr/bin/env python3
"""Validate the fixed exact reference pool used by the revised theory run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src" / "utils"))

from input_manifest import DEFAULT_DATASET_ROOT, DEFAULT_REFERENCE_ROOT, validate_inputs


def main() -> None:
    report = validate_inputs(DEFAULT_DATASET_ROOT, DEFAULT_REFERENCE_ROOT)
    print(
        json.dumps(
            {
                "status": "validated_exact_theory_references",
                "reference_count": report["reference_count"],
                "unique_reference_vector_count": report[
                    "unique_reference_vector_count"
                ],
                "minimum_signed_margin": report["minimum_signed_margin"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
