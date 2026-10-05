from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


STAGE_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SOURCE_ROOT = STAGE_ROOT / "src"
for path in (LOCAL_SOURCE_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.resources import ResourcePolicy  # noqa: E402

RESOURCE_POLICY = ResourcePolicy.from_json(
    STAGE_ROOT / "config" / "resources.json"
)
RESOURCE_POLICY.install_worker_environment(os.environ)

from utils.config import load_protocol  # noqa: E402
from utils.manifests import dataset_count, shell_pass_unit_count  # noqa: E402
from utils.sampling_control import sampling_production_guard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one atomic dataset-local shell_beta_100 SMC shard on the "
            "odd- or even-radius pass."
        )
    )
    parser.add_argument("--dataset-job", type=int, required=True)
    parser.add_argument(
        "--shell-pass",
        choices=("odd", "even"),
        required=True,
        help="explicit full-resolution radius pass",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_protocol()
    if not 0 <= args.dataset_job < dataset_count(config):
        raise ValueError("shell dataset job lies outside the canonical manifest")
    unit_count = shell_pass_unit_count(config, args.shell_pass)
    if not args.execute:
        print(
            f"dry_run dataset_job={args.dataset_job} shell_pass={args.shell_pass} "
            f"unit_count={unit_count} device={args.device or 'config'}; add --execute"
        )
        return
    with sampling_production_guard(SYNTHETIC_ROOT):
        from utils.smc import run_shard

        print(
            run_shard(
                config,
                dataset_job_index=args.dataset_job,
                shell_pass=args.shell_pass,
                device=args.device,
                synthetic_root=SYNTHETIC_ROOT,
            )
        )


if __name__ == "__main__":
    main()
