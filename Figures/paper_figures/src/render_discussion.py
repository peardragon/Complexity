#!/usr/bin/env python3
"""Render arXiv v1 Figure 9 from the current Discussion summaries."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from utils.paper_style import (
    STYLE,
    apply_axis_config,
    apply_style,
    figure_size,
    input_path,
    merge_settings,
    place_aligned_panel_labels,
    save_figure,
    scale_axis_box,
    style_axes,
    width_inches,
)


TYPOGRAPHY = STYLE["typography"]
HIGH = "#D55E00"
LOW = "#0072B2"
TEXT = "#1A1A1A"
SCHEMATIC_BLUE = "#3F79B6"

CURRENT_STUDIES = (
    {
        "domain": "synthetic",
        "label": "Synthetic",
        "linestyle": "-",
        "marker": "o",
        "higher": "data_beta_0p05",
        "lower": "data_beta_0p39",
    },
    {
        "domain": "mnist_label_noise",
        "label": "MNIST label noise",
        "linestyle": (0, (4.0, 2.0)),
        "marker": "s",
        "higher": "noise_eta_0p50",
        "lower": "noise_eta_0p00",
    },
    {
        "domain": "mnist_digit_pair",
        "label": "MNIST digit pairs",
        "linestyle": (0, (1.1, 1.5)),
        "marker": "^",
        "higher": "pair_4_9",
        "lower": "pair_0_1",
    },
)


def _current_affine_rows() -> pd.DataFrame:
    authority = pd.read_csv(
        input_path("figure_inputs/discussion/normalized_radial_profiles.csv")
    )
    if len(authority) != 6200 or authority["condition"].nunique() != 35:
        raise ValueError("current normalized-radial authority drift")
    blocks: list[pd.DataFrame] = []
    for study in CURRENT_STUDIES:
        conditions = (str(study["higher"]), str(study["lower"]))
        block = authority.loc[
            authority["domain"].eq(study["domain"])
            & authority["condition"].isin(conditions)
            & authority["radius"].between(0.01, 0.50)
        ].copy()
        if len(block) != 100 or block.groupby("condition").size().nunique() != 1:
            raise ValueError(f"{study['domain']}: expected two 50-radius profiles")
        joint_min = float(block["h_mean"].min())
        joint_max = float(block["h_mean"].max())
        joint_range = joint_max - joint_min
        if not np.isfinite(joint_range) or joint_range <= 0.0:
            raise ValueError(f"{study['domain']}: invalid pooled affine range")
        block["h_affine"] = (block["h_mean"] - joint_min) / joint_range
        block["h_se_affine"] = block["h_se_across_datasets"] / joint_range
        blocks.append(block)
    rows = pd.concat(blocks, ignore_index=True)
    if len(rows) != 300:
        raise ValueError(f"current composite expected 300 rows, found {len(rows)}")
    return rows


def _draw_current_empirical_panel(
    axis: plt.Axes,
    rows: pd.DataFrame,
    *,
    settings: dict[str, object],
) -> None:
    for study in CURRENT_STUDIES:
        for group, color in (("lower", LOW), ("higher", HIGH)):
            block = rows.loc[
                rows["domain"].eq(study["domain"])
                & rows["condition"].eq(study[group])
            ].sort_values("radius")
            radius = block["radius"].to_numpy(float)
            mean = block["h_affine"].to_numpy(float)
            sem = block["h_se_affine"].to_numpy(float)
            axis.fill_between(
                radius,
                mean - sem,
                mean + sem,
                color=color,
                alpha=float(settings["band_alpha"]),
                linewidth=0.0,
                zorder=1,
            )
            axis.plot(
                radius,
                mean,
                color=color,
                linestyle=study["linestyle"],
                linewidth=float(settings["line_width"]),
                marker=str(study["marker"]),
                markersize=float(settings["marker_size"]),
                markevery=int(settings["marker_every"]),
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.55,
                zorder=3 if group == "higher" else 2,
            )
    axis.set_xlim(*tuple(settings["xlim"]))
    axis.set_ylim(*tuple(settings["ylim"]))
    axis.set_xticks(tuple(settings["xticks"]))
    axis.set_yticks(tuple(settings["yticks"]))
    axis.set_xlabel(str(settings["xlabel"]))
    axis.set_ylabel(str(settings["ylabel"]))
    axis.set_title(str(settings["title"]), **dict(settings["title_kwargs"]))
    axis.axhline(0.0, color="#666666", linewidth=0.55, zorder=0)
    style_axes(axis)
    apply_axis_config(axis, dict(settings.get("axes", {})))
    scale_axis_box(axis, dict(settings.get("element_size", {})))


def _add_current_empirical_legends(
    axis: plt.Axes,
    *,
    legend_labels: dict[str, object],
    settings: dict[str, object],
) -> None:
    """Add independently adjustable line and study-marker legends."""
    legend_settings = dict(settings)
    legend_styles = dict(legend_settings.pop("styles", {}))
    complexity_style = dict(legend_styles.get("complexity", {}))
    study_style = dict(legend_styles.get("studies", {}))
    complexity_handles = [
        Line2D(
            [0], [0], color=HIGH,
            linewidth=float(complexity_style.get("line_width", 1.45)),
            label=str(legend_labels["higher"]),
        ),
        Line2D(
            [0], [0], color=LOW,
            linewidth=float(complexity_style.get("line_width", 1.45)),
            label=str(legend_labels["lower"]),
        ),
    ]
    study_handles = [
        Line2D(
            [0], [0],
            color=str(study_style.get("color", "#555555")),
            linestyle="None",
            marker=study["marker"],
            markerfacecolor="white",
            markeredgewidth=float(study_style.get("marker_edge_width", 0.55)),
            markersize=float(study_style.get("marker_size", 3.0)),
            label=str(legend_labels[str(study["domain"])]),
        )
        for study in CURRENT_STUDIES
    ]
    complexity_legend = axis.legend(
        handles=complexity_handles,
        **dict(legend_settings["complexity"]),
    )
    axis.add_artist(complexity_legend)
    axis.legend(handles=study_handles, **dict(legend_settings["studies"]))





def _corridor_half_angle(radius, regime, *, geometry=None):
    """Angular width used by the paper's schematic corridor surface."""
    settings = {
        "lower": {"knots": (0.0, 0.24, 0.58, 1.18), "widths": (0.43, 0.40, 0.34, 0.27)},
        "higher": {"knots": (0.0, 0.16, 0.30, 0.46, 0.58, 0.84, 1.18), "widths": (0.46, 0.45, 0.41, 0.13, 0.080, 0.12, 0.14)},
    }
    if geometry:
        settings.update(geometry)
    radius = np.asarray(radius, dtype=float)
    corridor = dict(settings[regime])
    knots = np.asarray(corridor["knots"], dtype=float)
    widths = np.asarray(corridor["widths"], dtype=float)
    if regime == "lower":
        return np.interp(radius, knots, widths)
    clipped = np.clip(radius, knots[0], knots[-1])
    segment = np.clip(np.searchsorted(knots, clipped, side="right") - 1, 0, len(knots) - 2)
    fraction = (clipped - knots[segment]) / (knots[segment + 1] - knots[segment])
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return widths[segment] + (widths[segment + 1] - widths[segment]) * smooth


def _landscape_surface(
    axis: plt.Axes,
    *,
    regime: str,
    settings: dict[str, object],
) -> None:
    """Draw a qualitative 3-D analogue of a Fig. 9 corridor panel."""
    surface = dict(settings["surface"])
    xlim = tuple(float(value) for value in surface["xlim"])
    ylim = tuple(float(value) for value in surface["ylim"])
    points = int(surface["points"])
    x = np.linspace(*xlim, points)
    y = np.linspace(*ylim, points)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    radius = np.sqrt(xx * xx + yy * yy)
    angle = np.abs(np.arctan2(yy, xx))
    half_angle = _corridor_half_angle(
        np.clip(radius, 0.0, 1.18),
        regime,
        geometry=dict(surface.get("geometry", {})),
    )
    corridor = np.exp(
        -np.power(angle / np.maximum(0.72 * half_angle, 0.035), 4.0)
    )
    radial_gate = 1.0 - np.exp(-np.power(radius / 0.17, 2.0))
    bowl = float(surface["bowl_strength"]) * radius * radius
    walls = float(surface["wall_height"]) * radial_gate * (1.0 - corridor)
    undulation = float(surface["undulation"]) * (
        np.sin(4.2 * xx - 0.8) * np.cos(3.4 * yy + 0.3)
    )
    height = bowl + walls + undulation * radial_gate
    if regime == "higher":
        pinch_radius = float(surface.get("pinch_radius", 0.52))
        pinch_width = max(float(surface.get("pinch_width", 0.12)), 1.0e-6)
        pinch = np.exp(
            -np.power((radius - pinch_radius) / pinch_width, 2.0)
        )
        height += float(surface["pinch_height"]) * pinch * (1.0 - corridor)
        pre_bulge_height = float(surface.get("pre_bulge_height", 0.0))
        if pre_bulge_height != 0.0:
            pre_bulge_radius = float(surface.get("pre_bulge_radius", 0.36))
            pre_bulge_width = max(
                float(surface.get("pre_bulge_width", 0.10)), 1.0e-6
            )
            pre_bulge = np.exp(
                -np.power(
                    (radius - pre_bulge_radius) / pre_bulge_width,
                    2.0,
                )
            )
            height += pre_bulge_height * pre_bulge * (1.0 - corridor)

    floor = float(height.min() - float(surface["projection_drop"]))
    axis.plot_surface(
        xx,
        yy,
        height,
        cmap=str(surface["cmap"]),
        linewidth=0.0,
        antialiased=True,
        alpha=float(surface["alpha"]),
        rcount=int(surface["surface_count"]),
        ccount=int(surface["surface_count"]),
    )
    axis.contour(
        xx,
        yy,
        height,
        zdir="z",
        offset=floor,
        levels=int(surface["contour_levels"]),
        cmap=str(surface["cmap"]),
        linewidths=0.60,
    )
    center_index = np.unravel_index(np.argmin(radius), radius.shape)
    center_height = float(height[center_index])
    axis.scatter(
        [0.0],
        [0.0],
        [center_height],
        color=TEXT,
        s=float(surface["reference_marker_size"]),
        depthshade=False,
        zorder=22,
    )
    reference_line = dict(surface.get("reference_line", {}))
    if bool(reference_line.pop("enabled", True)):
        line_bottom = floor + float(reference_line.pop("bottom_offset", 0.0))
        line_top = center_height + float(reference_line.pop("top_offset", 0.0))
        axis.plot(
            [0.0, 0.0],
            [0.0, 0.0],
            [line_bottom, line_top],
            **reference_line,
        )
    label_position = dict(surface.get("reference_label_position", {}))
    label_z_fraction = float(label_position.pop("z_fraction", 0.62))
    label_z = floor + label_z_fraction * (center_height - floor)
    axis.text(
        float(label_position.pop("offset_x", 0.045)),
        float(label_position.pop("offset_y", 0.025)),
        label_z + float(surface.get("reference_label_z_offset", 0.0)),
        str(settings["reference_label"]),
        color=TEXT,
        fontsize=TYPOGRAPHY["annotation_size_pt"],
        zorder=20,
        **label_position,
    )
    view = dict(settings["view"])
    axis.set_proj_type(str(view.get("projection", "persp")))
    axis.view_init(
        elev=float(view["elevation"]),
        azim=float(view["azimuth"]),
        roll=float(view.get("roll", 0.0)),
    )
    axis.set(xlim=xlim, ylim=ylim, zlim=(floor, float(height.max()) + 0.05))
    axis.set_box_aspect(tuple(float(value) for value in surface["box_aspect"]))
    axis.set_axis_off()
    axis.set_title(
        str(settings["title"]),
        loc="left",
        pad=float(settings["title_pad"]),
        fontsize=TYPOGRAPHY["panel_label_size_pt"],
    )
    axis.text2D(
        0.02,
        0.91,
        str(settings["subtitle"]),
        transform=axis.transAxes,
        color="#666666",
        style="italic",
        fontsize=TYPOGRAPHY["annotation_size_pt"],
    )





def render_hardening_3d_composite(
    *, settings: dict[str, object] | None = None
) -> None:
    """Render the active empirical-plus-3-D Fig. 9 composite."""
    config: dict[str, object] = {
        "figsize": figure_size("fig09_hardening_3d_composite"),
        "grid": {
            "width_ratios": (1.10, 1.16, 1.16),
            "left": 0.065,
            "right": 0.995,
            "bottom": 0.135,
            "top": 0.955,
            "wspace": 0.015,
        },
        "empirical": {
            "xlim": (0.01, 0.50),
            "ylim": (-0.035, 1.175),
            "xticks": (0.10, 0.20, 0.30, 0.40, 0.50),
            "yticks": (0.0, 0.25, 0.50, 0.75, 1.0),
            "xlabel": r"$r$",
            "ylabel": r"$h^{\ast}(r)$",
            "title": "",
            "title_kwargs": {"loc": "left", "pad": 6.0},
            "band_alpha": 0.075,
            "line_width": 1.25,
            "marker_size": 2.5,
            "marker_every": 10,
            "axes": {
                "x": {
                    "label": r"$r$",
                    "limits": (0.01, 0.50),
                    "ticks": (0.10, 0.20, 0.30, 0.40, 0.50),
                    "minor_ticks": True,
                },
                "y": {
                    "label": r"$h^{\ast}(r)$",
                    "limits": (-0.035, 1.175),
                    "ticks": (0.0, 0.25, 0.50, 0.75, 1.0),
                    "minor_ticks": True,
                },
            },
            "element_size": {"width_scale": 1.0, "height_scale": 1.0},
        },
        "panels": (
            {
                "regime": "lower",
                "title": "",
                "subtitle": "",
                "title_pad": 0.0,
                "reference_label": r"$\widetilde{\boldsymbol{w}}$",
                "view": {
                    "elevation": 24.0,
                    "azimuth": -68.0,
                    "roll": 0.0,
                    "projection": "persp",
                },
            },
            {
                "regime": "higher",
                "title": "",
                "subtitle": "",
                "title_pad": 0.0,
                "reference_label": r"$\widetilde{\boldsymbol{w}}$",
                "view": {
                    "elevation": 24.0,
                    "azimuth": -68.0,
                    "roll": 0.0,
                    "projection": "persp",
                },
            },
        ),
        "surface": {
            "xlim": (-0.34, 1.25),
            "ylim": (-0.82, 0.82),
            "points": 180,
            "surface_count": 120,
            "cmap": "turbo",
            "alpha": 0.83,
            "contour_levels": 13,
            "projection_drop": 0.42,
            "bowl_strength": 0.12,
            "wall_height": 0.58,
            "pinch_height": 0.34,
            "pinch_radius": 0.56,
            "pinch_width": 0.085,
            "pre_bulge_height": 0.14,
            "pre_bulge_radius": 0.36,
            "pre_bulge_width": 0.10,
            "undulation": 0.035,
            "reference_marker_size": 16.0,
            "reference_label_z_offset": 0.0,
            "reference_line": {
                "enabled": True,
                "bottom_offset": 0.0,
                "top_offset": 0.0,
                "color": "#333333",
                "linewidth": 0.8,
                "linestyle": (0, (2.0, 2.0)),
                "zorder": 21,
            },
            "reference_label_position": {
                "offset_x": 0.045,
                "offset_y": 0.025,
                "z_fraction": 0.62,
            },
            "box_aspect": (1.45, 1.5, 0.72),
            "geometry": {},
        },
        "legend_labels": {
            "higher": "Higher complexity",
            "lower": "Lower complexity",
            "synthetic": "Synthetic",
            "mnist_label_noise": "MNIST label noise",
            "mnist_digit_pair": "MNIST digit pairs",
        },
        "legend": {
            "complexity": {
                "loc": "upper right",
                "bbox_to_anchor": (0.98, 0.98),
                "ncol": 1,
                "fontsize": 5.7,
                "labelspacing": 0.25,
                "handlelength": 1.9,
                "handletextpad": 0.70,
                "borderaxespad": 0.0,
                "frameon": False,
            },
            "studies": {
                "loc": "upper right",
                "bbox_to_anchor": (0.98, 0.85),
                "ncol": 1,
                "fontsize": 5.7,
                "labelspacing": 0.25,
                "handlelength": 0.0,
                "handletextpad": 0.70,
                "borderaxespad": 0.0,
                "frameon": False,
            },
            "styles": {
                "complexity": {"line_width": 1.45},
                "studies": {
                    "color": "#555555",
                    "marker_size": 3.0,
                    "marker_edge_width": 0.55,
                },
            },
        },
        "panel_labels": {
            "A": {"text": "(a)"},
            "B": {"text": "(b)  Lower complexity"},
            "C": {"text": "(c)  Higher complexity"},
        },
        "panel_label_layout": {
            "orientation": "horizontal_row",
            "offset_left": 0.0,
            "offset_top": 0.018,
        },
    }
    config = merge_settings(config, settings)
    figure = plt.figure(figsize=tuple(config["figsize"]))
    grid_settings = dict(config["grid"])
    width_ratios = tuple(grid_settings.pop("width_ratios"))
    grid = figure.add_gridspec(1, 3, width_ratios=width_ratios, **grid_settings)
    empirical_axis = figure.add_subplot(grid[0, 0])
    _draw_current_empirical_panel(
        empirical_axis,
        _current_affine_rows(),
        settings=dict(config["empirical"]),
    )
    _add_current_empirical_legends(
        empirical_axis,
        legend_labels=dict(config["legend_labels"]),
        settings=dict(config["legend"]),
    )
    panels = tuple(config["panels"])
    if len(panels) != 2:
        raise ValueError("3-D composite requires exactly panels B and C")
    panel_axes = [empirical_axis]
    for index, panel in enumerate(panels, start=1):
        axis = figure.add_subplot(grid[0, index], projection="3d")
        panel_axes.append(axis)
        panel_settings = dict(panel)
        panel_settings["surface"] = dict(config["surface"])
        _landscape_surface(
            axis,
            regime=str(panel_settings["regime"]),
            settings=panel_settings,
        )
        # A transparent 3-D axes background must not hide panel A's edge tick.
        axis.patch.set_alpha(0.0)
        scale_axis_box(axis, dict(panel_settings.get("element_size", {})))
    label_settings = dict(config["panel_labels"])
    label_artists = place_aligned_panel_labels(
        figure,
        {"A": (panel_axes[0],), "B": (panel_axes[1],), "C": (panel_axes[2],)},
        label_settings,
        dict(config["panel_label_layout"]),
    )
    # Fig. 9-only fine adjustment layered on the shared alignment baseline.
    # Zero offsets preserve exact alignment; changing one entry moves only that label.
    for key, artist in label_artists.items():
        label = dict(label_settings[key])
        x, y = artist.get_position()
        artist.set_position(
            (
                x + float(label.get("offset_x", 0.0)),
                y + float(label.get("offset_y", 0.0)),
            )
        )
        style = dict(label.get("style", {}))
        if style:
            artist.set(**style)
    save_figure(figure, "fig09_hardening_3d_composite")
    plt.close(figure)

def main() -> None:
    apply_style()
    render_hardening_3d_composite()


if __name__ == "__main__":
    main()
