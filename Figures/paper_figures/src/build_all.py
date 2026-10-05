#!/usr/bin/env python3
"""Build and audit every figure referenced by the active paper authority."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pymupdf
from PIL import Image

from utils.paper_style import (
    FIGURE_BY_ID,
    MANIFEST,
    MANIFEST_PATH,
    PACKAGE_ROOT,
    STYLE,
    STYLE_PATH,
    all_declared_inputs,
    figure_size,
    output_path,
    sha256,
    width_inches,
    write_inventory,
)


def notebook_source_hash() -> str:
    notebook = json.loads((PACKAGE_ROOT / "releases/rebuild_all_paper_figures.ipynb").read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest()


def _validate_pdf(figure_id: str, path: Path) -> tuple[float, float]:
    spec = FIGURE_BY_ID[figure_id]
    expected_width_points = width_inches(spec["width_mode"]) * 72.0
    with pymupdf.open(path) as document:
        if document.page_count != 1:
            raise ValueError(f"{path}: expected one PDF page")
        page = document[0]
        width_points = float(page.rect.width)
        height_points = float(page.rect.height)
    if abs(width_points - expected_width_points) > 0.04:
        raise ValueError(
            f"{figure_id}: PDF width {width_points:.5f} pt; "
            f"expected {expected_width_points:.5f} pt"
        )
    if height_points <= 0:
        raise ValueError(f"{figure_id}: invalid PDF height")
    return width_points, height_points


def _validate_png(figure_id: str, path: Path) -> tuple[int, int]:
    expected = round(
        width_inches(FIGURE_BY_ID[figure_id]["width_mode"])
        * int(STYLE["export"]["png_dpi"])
    )
    with Image.open(path) as image:
        width, height = image.size
    if abs(width - expected) > 1:
        raise ValueError(f"{figure_id}: PNG width {width}px; expected {expected}px")
    if height <= 0:
        raise ValueError(f"{figure_id}: invalid PNG height")
    return width, height


def _assert_portable_text() -> None:
    checked_roots = (PACKAGE_ROOT / "src", PACKAGE_ROOT / "config")
    offenders: list[str] = []
    for root in checked_roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json", ".md"}:
                continue
            absolute_home_marker = "/" + "home" + "/"
            if absolute_home_marker in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    if offenders:
        raise ValueError(f"absolute home paths in portable files: {offenders}")


def audit(*, write=True) -> dict[str, object]:
    _assert_portable_text()
    missing_inputs = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in all_declared_inputs()
        if not path.is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"missing declared inputs: {missing_inputs}")

    inventory_rows: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    for spec in MANIFEST["figures"]:
        figure_id = spec["id"]
        paths = {
            suffix: output_path(figure_id, suffix)
            for suffix in STYLE["export"]["formats"]
        }
        for suffix, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{figure_id}: missing {suffix} output {path}")
        pdf_width, pdf_height = _validate_pdf(figure_id, paths["pdf"])
        png_width, png_height = _validate_png(figure_id, paths["png"])
        if "svg" in paths and "<svg" not in paths["svg"].read_text(
            encoding="utf-8", errors="ignore"
        )[:1000]:
            raise ValueError(f"{figure_id}: invalid SVG header")
        width, height = figure_size(figure_id)
        inventory_rows.append(
            {
                "figure_id": figure_id,
                "manuscript_location": spec["manuscript_location"],
                "width_mode": spec["width_mode"],
                "configured_width_inches": f"{width:.8f}",
                "configured_height_inches": f"{height:.8f}",
                "output_pdf": str(paths["pdf"].relative_to(PACKAGE_ROOT)),
                "output_pdf_sha256": sha256(paths["pdf"]),
                "renderer": spec["renderer"],
            }
        )
        outputs.append(
            {
                "figure_id": figure_id,
                "width_mode": spec["width_mode"],
                "pdf_width_points": pdf_width,
                "pdf_height_points": pdf_height,
                "png_width_pixels": png_width,
                "png_height_pixels": png_height,
                "files": {
                    suffix: {
                        "path": str(path.relative_to(PACKAGE_ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for suffix, path in paths.items()
                },
            }
        )
    inventory_path = write_inventory(inventory_rows) if write else PACKAGE_ROOT / "summarized_outputs/figure_inventory.csv"
    input_receipt = [
        {
            "path": str(path.relative_to(PACKAGE_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(set(all_declared_inputs()))
    ]
    receipt: dict[str, object] = {
        "schema_version": "complexity-paper-figures-current-build-receipt-v1",
        "status": "complete",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_page_profile": STYLE["active_page_profile"],
        "style_config": {
            "path": str(STYLE_PATH.relative_to(PACKAGE_ROOT.parent)),
            "sha256": sha256(STYLE_PATH),
        },
        "figure_manifest_sha256": sha256(MANIFEST_PATH),
        "notebook_source_sha256": notebook_source_hash(),
        "figure_count": len(outputs),
        "input_count": len(input_receipt),
        "inputs": input_receipt,
        "outputs": outputs,
        "inventory": {
            "path": str(inventory_path.relative_to(PACKAGE_ROOT)),
            "sha256": sha256(inventory_path) if inventory_path.is_file() else None,
        },
        "width_contract": {
            "one_column_inches": width_inches("one_column"),
            "two_column_inches": width_inches("two_column"),
            "pdf_tolerance_points": 0.04,
        },
    }
    receipt_path = PACKAGE_ROOT / "receipts" / "build_receipt.json"
    if write:
        if receipt_path.is_file():
            previous = json.loads(receipt_path.read_text())
            if {k:v for k,v in previous.items() if k!="built_at_utc"} == {k:v for k,v in receipt.items() if k!="built_at_utc"}:
                return previous
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Audit existing outputs without invoking the renderers.",
    )
    parser.add_argument(
        "--refresh-inputs",
        action="store_true",
        help="Explicitly refresh staged inputs from the mapped repository authorities.",
    )
    parser.add_argument("--force", action="store_true", help="Redraw all active figures with the notebook settings.")
    args = parser.parse_args()
    if args.check_only and (args.force or args.refresh_inputs): parser.error("--check-only is read-only")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from refresh_inputs import sync_inputs
    changed = set(sync_inputs(refresh=not args.check_only))
    previous_path = PACKAGE_ROOT / "receipts/build_receipt.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    changed.update(item["path"] for item in previous.get("inputs", []) if not (PACKAGE_ROOT / item["path"]).is_file() or sha256(PACKAGE_ROOT / item["path"]) != item["sha256"])
    settings_changed = (
        previous.get("style_config", {}).get("sha256", sha256(STYLE_PATH)) != sha256(STYLE_PATH)
        or previous.get("figure_manifest_sha256", sha256(MANIFEST_PATH)) != sha256(MANIFEST_PATH)
        or previous.get("notebook_source_sha256", notebook_source_hash()) != notebook_source_hash()
    )
    if args.check_only and (changed or settings_changed):
        raise ValueError("figure inputs or notebook settings changed; run the build to sync")
    if not args.check_only:
        with tempfile.TemporaryDirectory(prefix="complexity_paper_figures_mpl_") as cache:
            os.environ["MPLCONFIGDIR"] = cache
            from utils.notebook_runner import render_figures
            for spec in MANIFEST["figures"]:
                stale = settings_changed or bool(changed.intersection(spec["inputs"]))
                missing = any(not output_path(spec["id"], suffix).is_file() for suffix in STYLE["export"]["formats"])
                if not (args.force or stale or missing):
                    print(f"{spec['id']}: skipped_existing")
                    continue
                render_figures([spec["id"]], force=args.force or stale)
    receipt = audit(write=not args.check_only)
    print(
        f"paper_figures PASS: {receipt['figure_count']} figures; "
        f"profile={receipt['active_page_profile']}"
    )


if __name__ == "__main__":
    main()
