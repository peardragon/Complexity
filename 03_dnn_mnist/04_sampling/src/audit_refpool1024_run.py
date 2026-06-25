#!/usr/bin/env python3
"""Audit mechanical refpool1024 sampling outputs for completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sample_refpool1024_all_radii as sampler  # noqa: E402


DEFAULT_RUN_ROOT = sampler.DEFAULT_RUN_ROOT
REQUIRED_NPZ_LENGTH_KEYS = {
    "ce",
    "direction_projection",
    "error",
    "h",
    "l2_penalty",
    "logw_target",
    "theta_norm_sq",
}


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rounded_radius(value: Any) -> float:
    return round(float(value), 6)


def audit_npz(path: Path, n_samples: int) -> list[str]:
    problems: list[str] = []
    if not path.exists():
        return [f"missing npz: {path}"]
    try:
        with np.load(path) as data:
            keys = set(data.files)
            missing = REQUIRED_NPZ_LENGTH_KEYS - keys
            if missing:
                problems.append(f"{path}: missing keys {sorted(missing)}")
            for key in sorted(REQUIRED_NPZ_LENGTH_KEYS & keys):
                if data[key].shape[:1] != (n_samples,):
                    problems.append(f"{path}: key {key} shape {data[key].shape}, expected first dim {n_samples}")
            if "split" not in keys:
                problems.append(f"{path}: missing key split")
    except Exception as exc:
        problems.append(f"{path}: failed to load npz: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit refpool1024 mechanical sampling output coverage.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--radius-grid", choices=["production", "advanced"], default="production")
    parser.add_argument("--radii", default="", help="Optional comma-separated custom radius grid; must include r0=0.1.")
    parser.add_argument("--target-refs", type=int, default=60)
    parser.add_argument("--samples-per-ref-radius", type=int, default=1024)
    parser.add_argument("--check-npz-arrays", action="store_true")
    parser.add_argument("--max-problems", type=int, default=80)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    sampler.activate_radii(str(args.radius_grid), str(args.radii))
    pool = sampler.load_reference_pool(Path(sampler.DEFAULT_EXTRA_REFERENCE_RUN_ROOT), int(args.target_refs))
    expected = {
        (str(row.rule), int(row.ref_id), float(radius))
        for row in pool.itertuples(index=False)
        for radius in sampler.RADII
    }

    summary_root = run_root / "05_pool2_pm_sais_sampling" / "unit_summaries"
    observed: dict[tuple[str, int, float], Path] = {}
    duplicate_keys: list[tuple[str, int, float]] = []
    problems: list[str] = []
    n_samples_seen: set[int] = set()
    fallback_names: set[str] = set()

    for path in sorted(summary_root.rglob("unit_summary.json")) if summary_root.exists() else []:
        try:
            payload = load_json(path)
        except Exception as exc:
            problems.append(f"{path}: failed to load unit_summary.json: {exc}")
            continue
        key = (str(payload.get("rule")), int(payload.get("ref_id")), rounded_radius(payload.get("radius")))
        if key in observed:
            duplicate_keys.append(key)
        observed[key] = path
        n_samples = int(payload.get("n_samples", payload.get("n_samples_total", -1)))
        n_samples_seen.add(n_samples)
        if n_samples != int(args.samples_per_ref_radius):
            problems.append(f"{path}: n_samples={n_samples}, expected {args.samples_per_ref_radius}")
        fallback_names.add(str(payload.get("fallback_policy_name", "")))
        if str(payload.get("fallback_policy_name", "")) != "baseline":
            problems.append(f"{path}: fallback_policy_name={payload.get('fallback_policy_name')!r}, expected baseline")
        if args.check_npz_arrays:
            problems.extend(audit_npz(path.with_name("samples.npz"), int(args.samples_per_ref_radius)))
        elif not path.with_name("samples.npz").exists():
            problems.append(f"{path}: missing samples.npz")

    observed_keys = set(observed)
    missing = sorted(expected - observed_keys)
    extra = sorted(observed_keys - expected)
    status_path = run_root / "SAMPLING_STATUS.json"
    status_payload = load_json(status_path) if status_path.exists() else {}
    expected_units = len(expected)
    passed = (
        not missing
        and not extra
        and not duplicate_keys
        and not problems
        and len(observed_keys) == expected_units
        and status_payload.get("status") == "complete"
        and int(status_payload.get("completed_units", -1)) == expected_units
        and int(status_payload.get("samples_per_ref_radius", -1)) == int(args.samples_per_ref_radius)
        and int(status_payload.get("target_refs_per_rule", -1)) == int(args.target_refs)
    )
    if status_payload.get("status") != "complete":
        problems.append(f"SAMPLING_STATUS status={status_payload.get('status')!r}, expected complete")
    if int(status_payload.get("completed_units", -1)) != expected_units:
        problems.append(f"SAMPLING_STATUS completed_units={status_payload.get('completed_units')!r}, expected {expected_units}")

    report = {
        "passed": bool(passed),
        "run_root": str(run_root),
        "target_refs": int(args.target_refs),
        "radii": sampler.RADII,
        "samples_per_ref_radius": int(args.samples_per_ref_radius),
        "expected_units": int(expected_units),
        "observed_units": int(len(observed_keys)),
        "missing_units": int(len(missing)),
        "extra_units": int(len(extra)),
        "duplicate_units": int(len(duplicate_keys)),
        "n_samples_seen": sorted(int(value) for value in n_samples_seen),
        "fallback_policy_names": sorted(fallback_names),
        "status": status_payload,
        "problem_count": int(len(problems) + len(missing) + len(extra) + len(duplicate_keys)),
        "examples": {
            "missing": missing[: int(args.max_problems)],
            "extra": extra[: int(args.max_problems)],
            "duplicates": duplicate_keys[: int(args.max_problems)],
            "problems": problems[: int(args.max_problems)],
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
