"""Run one paper SMC shard; resume by completed output filename."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))
from schedule import DEFAULT_CONFIG, build_rows, project_path, validate_config
from smc import run_unit
from input_generation import ensure_row_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def output_path(root: Path, shard_index: int, shard_count: int) -> Path:
    return root / f"shard_{shard_index:04d}_of_{shard_count:04d}.jsonl.gz"


def selected_rows(config, index, count):
    if count != int(config["artifacts"]["default_shard_count"]) or not 0 <= index < count:
        raise ValueError("shard index/count differ from configured partition")
    return [row for row in build_rows(config, "") if int(row["unit_id"]) % count == index]


def check_only(*, config_path: Path, shard_index: int, shard_count: int):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    rows = selected_rows(config, shard_index, shard_count)
    path = output_path(project_path(config["artifacts"]["shard_output_directory"]), shard_index, shard_count)
    return {"status": "existing" if path.is_file() else "planned", "output": str(path), "unit_count": len(rows), "heavy_sampling_executed": False}


def execute_shard(*, config_path: Path, shard_index: int, shard_count: int, resume: bool = False):
    config = json.loads(Path(config_path).read_text())
    destination = output_path(project_path(config["artifacts"]["shard_output_directory"]), shard_index, shard_count)
    if destination.is_file():
        return {"status": "skipped_existing", "output": str(destination)}
    validate_config(config)
    rows = selected_rows(config, shard_index, shard_count)
    ensure_row_inputs(config, rows, PROJECT_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            header = {"record_type": "shard_metadata", "schema_version": 3, "shard_index": shard_index, "shard_count": shard_count, "unit_count": len(rows)}
            handle.write(json.dumps(header, allow_nan=False) + "\n")
            for row in rows:
                scalar, _ = run_unit(row=row, config=config, project_root=PROJECT_ROOT)
                if not scalar["smc_completed"]:
                    raise ValueError(f"unfinished SMC unit {row['unit_id']}")
                scalar["record_type"] = "sampling_unit"
                handle.write(json.dumps(scalar, allow_nan=False) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"status": "written", "output": str(destination), "unit_count": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=336)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Compatibility option; existing outputs always skip.")
    args = parser.parse_args()
    call = execute_shard if args.execute else check_only
    kwargs = {"config_path": project_path(args.config), "shard_index": args.shard_index, "shard_count": args.shard_count}
    if args.execute: kwargs["resume"] = args.resume
    print(json.dumps(call(**kwargs), indent=2))


if __name__ == "__main__":
    main()
