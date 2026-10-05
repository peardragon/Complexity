#!/usr/bin/env python3
"""Central style, physical-width, export, and receipt helpers for paper figures."""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FIGURES_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = FIGURES_ROOT.parent
MANIFEST_PATH = PACKAGE_ROOT / "config" / "figure_manifest.json"
PRESERVE_EXISTING_OUTPUTS = False


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


MANIFEST = _read_json(MANIFEST_PATH)
STYLE_PATH = (PACKAGE_ROOT / "config" / MANIFEST["style_config"]).resolve()
STYLE = _read_json(STYLE_PATH)
_FIGURE_IDS = [str(item["id"]) for item in MANIFEST["figures"]]
_OUTPUT_STEMS = [str(item["output"]) for item in MANIFEST["figures"]]
if len(_FIGURE_IDS) != len(set(_FIGURE_IDS)):
    raise ValueError("figure manifest contains duplicate figure IDs")
if len(_OUTPUT_STEMS) != len(set(_OUTPUT_STEMS)):
    raise ValueError("figure manifest contains duplicate output stems")
FIGURE_BY_ID = {item["id"]: item for item in MANIFEST["figures"]}


def active_profile() -> dict[str, Any]:
    return STYLE["page_profiles"][STYLE["active_page_profile"]]


def width_inches(width_mode: str) -> float:
    profile = active_profile()
    points_per_inch = float(STYLE["units"]["tex_points_per_inch"])
    if width_mode == "one_column":
        tex_points = float(profile["one_column_width_tex_pt"])
    elif width_mode == "two_column":
        tex_points = float(profile["text_width_tex_pt"])
    else:
        raise ValueError(f"unknown width mode: {width_mode}")
    return tex_points / points_per_inch


def figure_size(figure_id: str) -> tuple[float, float]:
    spec = FIGURE_BY_ID[figure_id]
    width = width_inches(str(spec["width_mode"]))
    return width, width * float(spec["height_to_width"])


def merge_settings(
    defaults: dict[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recursively merge a compact notebook config into renderer defaults."""
    merged = deepcopy(defaults)
    if not overrides:
        return merged
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = merge_settings(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_axis_config(axis: mpl.axes.Axes, config: Mapping[str, Any] | None) -> None:
    """Apply the notebook's shared label/scale/limit/tick schema to one axis."""
    if not config:
        return
    for dimension in ("x", "y"):
        item = dict(config.get(dimension, {}))
        if not item:
            continue
        label = item.get("label")
        if label is not None:
            label_kwargs: dict[str, Any] = {}
            if item.get("label_pad") is not None:
                label_kwargs["labelpad"] = float(item["label_pad"])
            if item.get("label_size") is not None:
                label_kwargs["fontsize"] = float(item["label_size"])
            getattr(axis, f"set_{dimension}label")(str(label), **label_kwargs)
        scale = item.get("scale")
        if scale is not None:
            getattr(axis, f"set_{dimension}scale")(str(scale))
        limits = item.get("limits")
        if limits is not None:
            getattr(axis, f"set_{dimension}lim")(*tuple(limits))
        ticks = item.get("ticks")
        tick_labels = item.get("tick_labels")
        if ticks is not None:
            setter = getattr(axis, f"set_{dimension}ticks")
            if tick_labels is None:
                setter(tuple(ticks))
            else:
                setter(tuple(ticks), tuple(str(value) for value in tick_labels))
        elif tick_labels is not None:
            raise ValueError(f"{dimension}.tick_labels requires {dimension}.ticks")
        rotation = item.get("tick_rotation")
        if rotation is not None:
            axis.tick_params(axis=dimension, labelrotation=float(rotation))
        tick_ha = item.get("tick_ha")
        if tick_ha is not None:
            for tick_label in getattr(axis, f"get_{dimension}ticklabels")():
                tick_label.set_horizontalalignment(str(tick_ha))
        tick_size = item.get("tick_label_size")
        if tick_size is not None:
            axis.tick_params(axis=dimension, labelsize=float(tick_size))
        tick_params = dict(item.get("tick_params", {}))
        if tick_params:
            axis.tick_params(axis=dimension, **tick_params)
        if bool(item.get("minor_ticks", False)):
            axis.minorticks_on()


def scale_axis_box(
    axis: mpl.axes.Axes,
    config: Mapping[str, Any] | None,
) -> None:
    """Scale an axis box about its center inside the allocated panel cell."""
    if not config:
        return
    width_scale = float(config.get("width_scale", 1.0))
    height_scale = float(config.get("height_scale", 1.0))
    if width_scale <= 0.0 or height_scale <= 0.0:
        raise ValueError("axis element width/height scales must be positive")
    box = axis.get_position()
    width = box.width * width_scale
    height = box.height * height_scale
    axis.set_position(
        [
            box.x0 + 0.5 * (box.width - width) + float(config.get("shift_x", 0.0)),
            box.y0 + 0.5 * (box.height - height) + float(config.get("shift_y", 0.0)),
            width,
            height,
        ]
    )


def place_group_label(
    figure: plt.Figure,
    axes: Iterable[mpl.axes.Axes],
    config: Mapping[str, Any],
) -> None:
    """Place a label relative to the top-left corner of a logical panel group."""
    axis_list = list(axes)
    if not axis_list:
        raise ValueError("panel label requires at least one axis")
    boxes = [axis.get_position() for axis in axis_list]
    left = float(config.get("anchor_left", min(box.x0 for box in boxes)))
    top = max(box.y1 for box in boxes)
    figure.text(
        left + float(config.get("offset_left", 0.0)),
        top + float(config.get("offset_top", 0.0)),
        str(config["text"]),
        ha=str(config.get("ha", "left")),
        va=str(config.get("va", "bottom")),
        fontsize=float(config.get("fontsize", STYLE["typography"]["panel_label_size_pt"])),
    )


def place_aligned_panel_labels(
    figure: plt.Figure,
    panel_axes: Mapping[str, Iterable[mpl.axes.Axes]],
    labels: Mapping[str, Mapping[str, Any] | str],
    layout: Mapping[str, Any],
) -> dict[str, mpl.text.Text]:
    """Place all panel labels from one shared offset and alignment policy.

    ``vertical_stack`` shares one left anchor, ``horizontal_row`` shares one
    top anchor, and ``grid`` uses explicit row/column groups.  Thus scaling an
    individual panel cannot silently move only its label.
    """
    keys = tuple(panel_axes)
    if set(keys) != set(labels):
        raise ValueError("panel-label keys must exactly match panel-axis keys")
    boxes: dict[str, tuple[float, float]] = {}
    for key, axes in panel_axes.items():
        axis_list = list(axes)
        if not axis_list:
            raise ValueError(f"panel {key}: label requires at least one axis")
        positions = [axis.get_position() for axis in axis_list]
        boxes[key] = (
            min(box.x0 for box in positions),
            max(box.y1 for box in positions),
        )

    orientation = str(layout["orientation"])
    if orientation == "vertical_stack":
        column_groups = (keys,)
        row_groups = tuple((key,) for key in keys)
    elif orientation == "horizontal_row":
        column_groups = tuple((key,) for key in keys)
        row_groups = (keys,)
    elif orientation == "grid":
        column_groups = tuple(tuple(group) for group in layout["column_groups"])
        row_groups = tuple(tuple(group) for group in layout["row_groups"])
    else:
        raise ValueError(f"unknown panel-label orientation: {orientation}")

    def validate_groups(groups: tuple[tuple[str, ...], ...], name: str) -> None:
        flattened = [key for group in groups for key in group]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(keys):
            raise ValueError(f"panel-label {name} must cover each panel exactly once")

    validate_groups(column_groups, "column_groups")
    validate_groups(row_groups, "row_groups")
    left = {key: boxes[key][0] for key in keys}
    top = {key: boxes[key][1] for key in keys}
    for group in column_groups:
        shared_left = min(boxes[key][0] for key in group)
        left.update({key: shared_left for key in group})
    for group in row_groups:
        shared_top = max(boxes[key][1] for key in group)
        top.update({key: shared_top for key in group})

    offset_left = float(layout.get("offset_left", 0.0))
    offset_top = float(layout.get("offset_top", 0.0))
    style = dict(layout.get("style", {}))
    style.setdefault("ha", "left")
    style.setdefault("va", "bottom")
    style.setdefault("fontsize", STYLE["typography"]["panel_label_size_pt"])
    artists: dict[str, mpl.text.Text] = {}
    for key in keys:
        label = labels[key]
        text = str(label["text"] if isinstance(label, Mapping) else label)
        artist = figure.text(
            left[key] + offset_left,
            top[key] + offset_top,
            text,
            **style,
        )
        artist.set_gid(f"panel-label-{key}")
        artists[key] = artist

    tolerance = 1.0e-12
    for group in column_groups:
        xs = [artists[key].get_position()[0] for key in group]
        if max(xs) - min(xs) > tolerance:
            raise RuntimeError(f"panel-label column alignment failed: {group}")
    for group in row_groups:
        ys = [artists[key].get_position()[1] for key in group]
        if max(ys) - min(ys) > tolerance:
            raise RuntimeError(f"panel-label row alignment failed: {group}")
    return artists


def apply_style() -> None:
    typography = STYLE["typography"]
    strokes = STYLE["strokes"]
    export = STYLE["export"]
    mpl.rcParams.update(
        {
            "font.family": typography["font_family"],
            "font.size": typography["base_font_size_pt"],
            "axes.labelsize": typography["axes_label_size_pt"],
            "axes.titlesize": typography["axes_title_size_pt"],
            "xtick.labelsize": typography["tick_label_size_pt"],
            "ytick.labelsize": typography["tick_label_size_pt"],
            "legend.fontsize": typography["legend_size_pt"],
            "axes.linewidth": strokes["axes_line_width_pt"],
            "lines.linewidth": strokes["line_width_pt"],
            "mathtext.fontset": typography["mathtext_fontset"],
            "pdf.fonttype": export["pdf_fonttype"],
            "ps.fonttype": export["ps_fonttype"],
            "svg.fonttype": export["svg_fonttype"],
            "svg.hashsalt": "complexity-paper-figures-current",
            "savefig.transparent": export["transparent"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def output_stem(figure_id: str) -> Path:
    return PACKAGE_ROOT / str(FIGURE_BY_ID[figure_id]["output"])


def output_path(figure_id: str, suffix: str) -> Path:
    """Append an extension without treating dots inside a stem as suffixes."""
    normalized = suffix if suffix.startswith(".") else f".{suffix}"
    return Path(f"{output_stem(figure_id)}{normalized}")


def input_path(relative_path: str) -> Path:
    path = PACKAGE_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_figure(figure: plt.Figure, figure_id: str) -> list[Path]:
    """Save without tight bounding so the configured physical width is retained."""
    output_stem(figure_id).parent.mkdir(parents=True, exist_ok=True)
    export = STYLE["export"]
    outputs: list[Path] = []
    for suffix in export["formats"]:
        path = output_path(figure_id, suffix)
        if PRESERVE_EXISTING_OUTPUTS and path.is_file():
            outputs.append(path)
            continue
        kwargs: dict[str, Any] = {
            "transparent": bool(export["transparent"]),
            "metadata": {"CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else None,
        }
        if suffix == "png":
            kwargs["dpi"] = int(export["png_dpi"])
            kwargs["metadata"] = {"Software": "Complexity paper_figures"}
        if suffix == "svg":
            kwargs["metadata"] = {
                "Date": None,
                "Creator": "Complexity paper_figures",
            }
        if kwargs["metadata"] is None:
            kwargs.pop("metadata")
        figure.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def style_axes(axis: mpl.axes.Axes, *, grid_axis: str | None = "both") -> None:
    strokes = STYLE["strokes"]
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(
        direction="out",
        length=2.8,
        width=float(strokes["axes_line_width_pt"]),
    )
    if grid_axis:
        axis.grid(
            axis=grid_axis,
            color="#D9D9D9",
            linewidth=float(strokes["grid_line_width_pt"]),
            alpha=0.68,
        )
        axis.set_axisbelow(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_declared_inputs() -> Iterable[Path]:
    for spec in MANIFEST["figures"]:
        for item in spec.get("inputs", []):
            yield PACKAGE_ROOT / item


def write_inventory(rows: list[dict[str, Any]]) -> Path:
    path = PACKAGE_ROOT / "summarized_outputs" / "figure_inventory.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "figure_id",
        "manuscript_location",
        "width_mode",
        "configured_width_inches",
        "configured_height_inches",
        "output_pdf",
        "output_pdf_sha256",
        "renderer",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    if not path.is_file() or path.read_bytes() != payload:
        path.write_bytes(payload)
    return path
