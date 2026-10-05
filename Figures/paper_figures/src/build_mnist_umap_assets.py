#!/usr/bin/env python3
"""Bootstrap the four paper MNIST UMAP visual inputs when their PDFs are absent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "figure_inputs" / "static_assets"
)
OUTPUT_STEMS = {
    "label_eta_0": "label_noise_eta_0p0_composed",
    "label_eta_0p5": "label_noise_eta_0p5_composed",
    "pair_high": "digit_pair_highest_complexity_pair_4_9_composed",
    "pair_low": "digit_pair_lowest_complexity_pair_0_1_composed",
}


def output_paths(output_dir: Path) -> dict[str, tuple[Path, Path]]:
    return {
        key: (output_dir / f"{stem}.pdf", output_dir / f"{stem}.png")
        for key, stem in OUTPUT_STEMS.items()
    }


def prepare_inputs(
    project_root: Path | None = None,
    output_dir: Path | None = None,
    *,
    execute: bool = True,
    check_only: bool = False,
) -> dict[str, Any]:
    """Return immediately when PDFs exist; otherwise optionally build missing ones."""

    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    destination = (
        root / "Figures/paper_figures/figure_inputs/static_assets"
        if output_dir is None
        else Path(output_dir).resolve()
    )
    paths = output_paths(destination)
    missing = {key for key, (pdf, _) in paths.items() if not pdf.is_file()}
    if not missing:
        return {
            "status": "skipped_existing",
            "helper_imported": False,
            "outputs": {key: str(pdf) for key, (pdf, _) in paths.items()},
        }
    if check_only:
        raise FileNotFoundError(
            "missing required MNIST UMAP PDFs: "
            + ", ".join(str(paths[key][0]) for key in sorted(missing))
        )
    if not execute:
        return {
            "status": "planned",
            "helper_imported": False,
            "missing": sorted(missing),
            "output_dir": str(destination),
        }

    # Heavy numerical imports occur only after the complete filename-skip gate.
    from utils.mnist_umap_assets import build_missing_assets

    actions = build_missing_assets(
        project_root=root,
        output_paths=paths,
        missing_keys=missing,
    )
    remaining = [str(pdf) for pdf, _ in paths.values() if not pdf.is_file()]
    if remaining:
        raise RuntimeError("UMAP bootstrap left required PDFs missing: " + ", ".join(remaining))
    return {
        "status": "completed",
        "helper_imported": True,
        "actions": actions,
        "outputs": {key: str(pdf) for key, (pdf, _) in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: PROJECT_ROOT/Figures/paper_figures/figure_inputs/static_assets)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        result = prepare_inputs(
            args.project_root,
            args.output_dir,
            execute=bool(args.execute),
            check_only=bool(args.check_only),
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
