#!/usr/bin/env python3
"""Stage author-supplied static artwork at the configured physical width."""

from __future__ import annotations

from io import BytesIO

import pymupdf
from PIL import Image
from utils import paper_style

from utils.paper_style import (
    FIGURE_BY_ID,
    STYLE,
    figure_size,
    input_path,
    merge_settings,
    output_path,
    output_stem,
    width_inches,
)


STATIC_IDS = (
    "fig01_dataset_to_landscape",
)


def stage(figure_id: str, *, settings: dict[str, object] | None = None) -> None:
    spec = FIGURE_BY_ID[figure_id]
    config: dict[str, object] = {
        "canvas_inches": figure_size(figure_id),
        "preserve_source_aspect": True,
        "crop_transparent_padding": False,
        "png_dpi": int(STYLE["export"]["png_dpi"]),
    }
    config = merge_settings(config, settings)
    asset_kind = spec.get("asset_kind")
    if asset_kind not in {"author_supplied_ppt_emf_export", "author_supplied_png"}:
        raise ValueError(f"{figure_id}: missing static-asset provenance type")
    source = input_path(spec["inputs"][0])
    canvas_width_inches, canvas_height_inches = config["canvas_inches"]
    target_width_points = float(canvas_width_inches) * 72.0
    target_height_points = float(canvas_height_inches) * 72.0
    keep_proportion = bool(config["preserve_source_aspect"])
    output_document = pymupdf.open()
    if asset_kind == "author_supplied_png":
        image_stream: bytes | None = None
        if bool(config["crop_transparent_padding"]):
            with Image.open(source) as source_image:
                rgba = source_image.convert("RGBA")
                content_bbox = rgba.getchannel("A").getbbox()
                if content_bbox is None:
                    raise ValueError(f"{source}: empty alpha content")
                cropped = rgba.crop(content_bbox)
                stream = BytesIO()
                cropped.save(stream, format="PNG")
                image_stream = stream.getvalue()
        output_page = output_document.new_page(
            width=target_width_points, height=target_height_points
        )
        output_page.insert_image(
            output_page.rect,
            filename=str(source) if image_stream is None else None,
            stream=image_stream,
            keep_proportion=keep_proportion,
        )
    else:
        with pymupdf.open(source) as source_document:
            if source_document.page_count != 1:
                raise ValueError(f"{source}: expected a one-page composed panel")
            source_page = source_document[0]
            output_page = output_document.new_page(
                width=target_width_points, height=target_height_points
            )
            output_page.show_pdf_page(
                output_page.rect,
                source_document,
                0,
                keep_proportion=keep_proportion,
            )

    stem = output_stem(figure_id)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path(figure_id, "pdf")
    write_pdf = not (paper_style.PRESERVE_EXISTING_OUTPUTS and pdf_path.is_file())
    if write_pdf:
        output_document.save(pdf_path, garbage=4, deflate=True, clean=True, no_new_id=True, reproducible=True)
    output_document.close()

    # PyMuPDF writes a PDF-1.7 header even though this composed page uses no
    # post-1.5 feature.  RevTeX's pdfTeX backend accepts at most 1.5; changing
    # the same-length header avoids a spurious inclusion warning while leaving
    # every object and cross-reference offset unchanged.
    payload = pdf_path.read_bytes()
    if write_pdf and not payload.startswith(b"%PDF-1.7"):
        raise ValueError(f"{pdf_path}: unexpected PDF header")
    if write_pdf:
        pdf_path.write_bytes(b"%PDF-1.5" + payload[len(b"%PDF-1.7") :])

    with pymupdf.open(pdf_path) as rendered:
        page = rendered[0]
        dpi = int(config["png_dpi"])
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        png = output_path(figure_id, "png")
        svg = output_path(figure_id, "svg")
        if not (paper_style.PRESERVE_EXISTING_OUTPUTS and png.is_file()): pixmap.save(png)
        if not (paper_style.PRESERVE_EXISTING_OUTPUTS and svg.is_file()):
            svg.write_text(page.get_svg_image(text_as_path=False), encoding="utf-8")


def main() -> None:
    for figure_id in STATIC_IDS:
        stage(figure_id)


if __name__ == "__main__":
    main()
