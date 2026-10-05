#!/usr/bin/env python3
"""Compose a deterministic contact sheet of all active paper figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from utils.paper_style import MANIFEST, PACKAGE_ROOT, output_path


OUTPUT = PACKAGE_ROOT / "receipts" / "visual_audit_contact_sheet.png"
RECEIPT = PACKAGE_ROOT / "receipts" / "visual_audit.json"
CANVAS_WIDTH = 2096
SIDE_MARGIN = 58
TOP_MARGIN = 46
BOTTOM_MARGIN = 52
LABEL_HEIGHT = 58
PANEL_GAP = 34
MAX_IMAGE_WIDTH = CANVAS_WIDTH - 2 * SIDE_MARGIN
MAX_IMAGE_HEIGHT = 720


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def main() -> None:
    entries: list[tuple[str, Image.Image]] = []
    for index, spec in enumerate(MANIFEST["figures"], start=1):
        source = output_path(spec["id"], "png")
        with Image.open(source) as image:
            rendered = image.convert("RGB")
        rendered.thumbnail(
            (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS
        )
        label = f"Fig. {index}  |  {spec['id']}  |  {spec['width_mode']}"
        entries.append((label, rendered))

    canvas_height = (
        TOP_MARGIN
        + BOTTOM_MARGIN
        + sum(LABEL_HEIGHT + image.height for _, image in entries)
        + PANEL_GAP * (len(entries) - 1)
    )
    canvas = Image.new("RGB", (CANVAS_WIDTH, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _font(29, bold=True)
    rule_color = (180, 180, 180)
    text_color = (25, 25, 25)

    y = TOP_MARGIN
    for label, rendered in entries:
        draw.text((SIDE_MARGIN, y), label, font=label_font, fill=text_color)
        rule_y = y + LABEL_HEIGHT - 12
        draw.line(
            (SIDE_MARGIN, rule_y, CANVAS_WIDTH - SIDE_MARGIN, rule_y),
            fill=rule_color,
            width=2,
        )
        y += LABEL_HEIGHT
        x = (CANVAS_WIDTH - rendered.width) // 2
        canvas.paste(rendered, (x, y))
        y += rendered.height + PANEL_GAP

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", optimize=False, compress_level=9)
    rendered_paths = [
        str(output_path(spec["id"], "png").relative_to(PACKAGE_ROOT))
        for spec in MANIFEST["figures"]
    ]
    payload = {
        "schema_version": "complexity-paper-figures-visual-audit-current-v1",
        "status": "contact_sheet_ready",
        "paper_authority": MANIFEST["paper_authority"],
        "figure_count": len(entries),
        "contact_sheet": {
            "path": str(OUTPUT.relative_to(PACKAGE_ROOT)),
            "width_pixels": canvas.width,
            "height_pixels": canvas.height,
            "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        },
        "active_pngs": rendered_paths,
        "checks": [
            "all active figures are present in manifest order",
            "Current Fig. 1, squared-distance Fig. 2, and composite Fig. 9 are active",
        ],
        "manual_review": "Legibility, clipping, and panel overlap require visual inspection.",
    }
    RECEIPT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT.relative_to(PACKAGE_ROOT)}: {canvas.width}x{canvas.height}")


if __name__ == "__main__":
    main()
