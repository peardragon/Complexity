from __future__ import annotations

import argparse

from utils.dataset_builder import build_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MNIST digit-pair dataset payloads from a stage-local original MNIST cache.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate pair dataset.npz payloads from raw_outputs/original_mnist.",
    )
    parser.add_argument(
        "--source-cache",
        default=None,
        help="Optional existing mnist_openml_uint8.npz to copy into raw_outputs/original_mnist before generation.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail if raw_outputs/original_mnist is absent and --source-cache is not provided.",
    )
    args = parser.parse_args()
    counts = build_datasets(
        overwrite=bool(args.overwrite),
        source_cache=args.source_cache,
        download=not bool(args.no_download),
    )
    print(
        "datasets "
        f"generated={counts['generated']} "
        f"validated={counts['validated']} "
        f"metadata_written={counts['metadata_written']} "
        f"missing={counts['missing']}"
    )


if __name__ == "__main__":
    main()
