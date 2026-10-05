from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


STAGE_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from utils.resources import ResourcePolicy  # noqa: E402

RESOURCE_POLICY = ResourcePolicy.from_json(
    STAGE_ROOT / "config" / "resources.json"
)
RESOURCE_POLICY.install_worker_environment(os.environ)

from utils.config import load_protocol  # noqa: E402
from utils.dataset import generate_range  # noqa: E402
from utils.manifests import dataset_count  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fresh balanced WS-Ising synthetic datasets.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--execute", action="store_true", help="Required guard against accidental heavy generation.")
    args = parser.parse_args()
    config = load_protocol()
    if not 0 <= args.start < args.stop <= dataset_count(config):
        raise ValueError("dataset range lies outside the canonical manifest")
    if not args.execute:
        print(f"dry_run dataset_tasks=[{args.start},{args.stop}); add --execute to generate")
        return
    RESOURCE_POLICY.assert_worker_environment()
    print(generate_range(config, start=args.start, stop=args.stop, synthetic_root=SYNTHETIC_ROOT))


if __name__ == "__main__":
    main()
