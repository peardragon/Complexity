#!/usr/bin/env python3
"""Notebook-facing adapters for the current canonical paper figures.

The notebook exposes one consistent, detailed configuration vocabulary.  This
module translates that vocabulary into the smaller renderer-specific settings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import matplotlib as mpl
from IPython.display import Image as IPythonImage, display

import build_all
import make_visual_audit_sheet
from utils import paper_style
import render_discussion
import render_quantitative
import stage_static_assets


_NOTEBOOK_STYLE: dict[str, Any] | None = None


def configure(style: dict[str, Any]) -> None:
    """Store and immediately apply the editable notebook-wide style."""
    global _NOTEBOOK_STYLE
    _NOTEBOOK_STYLE = deepcopy(style)
    _apply_style()


def _apply_style() -> None:
    if _NOTEBOOK_STYLE is None:
        raise RuntimeError("Run the notebook Setup cell before a figure cell")
    typography = _NOTEBOOK_STYLE["typography"]
    math = _NOTEBOOK_STYLE["math"]
    paper_style.STYLE["typography"].update(
        {key: value for key, value in typography.items() if key != "font_fallbacks"}
    )
    paper_style.STYLE["strokes"].update(_NOTEBOOK_STYLE["strokes"])
    paper_style.STYLE["export"].update(_NOTEBOOK_STYLE["export"])
    paper_style.apply_style()
    mpl.rcParams.update(
        {
            "font.family": typography["font_family"],
            "font.serif": list(typography["font_fallbacks"]),
            "text.usetex": bool(math["text_usetex"]),
            "text.latex.preamble": math["latex_preamble"],
            "mathtext.fontset": typography["mathtext_fontset"],
            "mathtext.default": math["mathtext_default"],
            "axes.formatter.use_mathtext": bool(math["axes_formatter_use_mathtext"]),
        }
    )


def _figure_inches(conf: dict[str, Any]) -> tuple[float, float]:
    if _NOTEBOOK_STYLE is None:
        raise RuntimeError("Run the notebook Setup cell before a figure cell")
    size = dict(conf["figure_size"])
    width_key = {
        "one_column": "one_column_inches",
        "two_column": "two_column_inches",
    }
    width_mode = str(size["width"])
    if width_mode not in width_key:
        raise ValueError(f"unknown figure width keyword: {width_mode}")
    width = float(_NOTEBOOK_STYLE["page"][width_key[width_mode]])
    width *= float(size.get("width_scale", 1.0))
    height_ratio = float(size["height_ratio"])
    if width <= 0.0 or height_ratio <= 0.0:
        raise ValueError("figure width and height_ratio must be positive")
    return width, width * height_ratio


def _display_width(conf: dict[str, Any]) -> int:
    return int(dict(conf.get("display", {})).get("width_px", 1000))


def _axes(panel: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(panel.get("axes", {})))


def _elements(panel: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(panel.get("elements", {})))


def _label(panel: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(panel["label"]))


def _title_kwargs(panel: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(panel.get("label_style", {"loc": "left", "pad": 3.0})))


def _margins(layout: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(layout.get("margins", {})))


def _run(
    figure_id: str,
    renderer: Callable[[dict[str, Any]], None],
    settings: dict[str, Any],
    display_width: int,
) -> None:
    _apply_style()
    renderer(settings)
    png = paper_style.output_path(figure_id, "png")
    if not png.is_file():
        raise FileNotFoundError(png)
    print(f"wrote {png.relative_to(paper_style.PACKAGE_ROOT)}")
    display(IPythonImage(filename=str(png), width=display_width))


def fig01(conf: dict[str, Any]) -> None:
    asset = dict(conf.get("asset", {}))
    settings = {
        "canvas_inches": _figure_inches(conf),
        "preserve_source_aspect": bool(asset.get("preserve_source_aspect", True)),
        "crop_transparent_padding": bool(asset.get("crop_transparent_padding", False)),
    }
    _run(
        "fig01_dataset_to_landscape",
        lambda item: stage_static_assets.stage("fig01_dataset_to_landscape", settings=item),
        settings,
        _display_width(conf),
    )


def fig02(conf: dict[str, Any]) -> None:
    layout = dict(conf["layout"])
    panel = dict(dict(conf["panels"])["A"])
    axes = _axes(panel)
    title = dict(panel.get("title", {}))
    legend = deepcopy(dict(conf["legend"]))
    legend_labels = deepcopy(dict(legend.pop("labels")))
    settings: dict[str, Any] = {
        "figsize": _figure_inches(conf),
        "constrained_layout": bool(layout.get("constrained", True)),
        "subplot_adjust": layout.get("margins"),
        "xlabel": str(dict(axes["x"])["label"]),
        "ylabel": str(dict(axes["y"])["label"]),
        "title": str(title.get("text", "")),
        "title_kwargs": {key: value for key, value in title.items() if key != "text"},
        "axes": axes,
        "element_size": _elements(panel),
        "legend": legend,
        "legend_labels": legend_labels,
    }
    settings.update(deepcopy(dict(conf.get("curves", {}))))
    _run(
        "fig02_perceptron_full_shell",
        lambda item: render_quantitative.render_perceptron(settings=item),
        settings,
        _display_width(conf),
    )


def fig03(conf: dict[str, Any]) -> None:
    layout = dict(conf["layout"])
    spacing = dict(conf["panel_spacing"])
    panel_a = dict(dict(conf["panels"])["A"])
    panel_b = dict(dict(conf["panels"])["B"])
    inner = dict(panel_a["inner_grid"])
    axes_b = _axes(panel_b)
    settings = {
        "figsize": _figure_inches(conf),
        "outer_grid": {
            "height_ratios": (float(panel_a["height_weight"]), float(panel_b["height_weight"])),
            **_margins(layout),
            "hspace": float(spacing["vertical"]),
        },
        "montage_grid": {
            "rows": int(inner["rows"]),
            "cols": int(inner["columns"]),
            "wspace": float(inner["wspace"]),
            "hspace": float(inner["hspace"]),
        },
        "dataset_panel": {
            "title_template": str(panel_a["element_title_template"]),
            "title_kwargs": deepcopy(dict(panel_a["element_title_style"])),
            "element_size": _elements(panel_a),
        },
        "complexity": {
            "xlabel": str(dict(axes_b["x"])["label"]),
            "ylabel": str(dict(axes_b["y"])["label"]),
            "axes": axes_b,
            "element_size": _elements(panel_b),
        },
        "panel_labels": {"A": _label(panel_a), "B": _label(panel_b)},
        "panel_label_layout": deepcopy(dict(conf["panel_label_layout"])),
    }
    _run(
        "fig03_synthetic_data_complexity",
        lambda item: render_quantitative.render_synthetic_data_complexity(settings=item),
        settings,
        _display_width(conf),
    )


def _mnist_data_complexity(
    conf: dict[str, Any], *, figure_id: str, domain: str
) -> None:
    layout = dict(conf["layout"])
    spacing = dict(conf["panel_spacing"])
    panel_a = dict(dict(conf["panels"])["A"])
    panel_b = dict(dict(conf["panels"])["B"])
    inner = dict(panel_a["inner_grid"])
    axes_b = _axes(panel_b)
    titles = tuple(panel_a["element_titles"])
    visual_panels = tuple(
        {
            "title": str(title),
            "title_kwargs": deepcopy(dict(panel_a["element_title_style"])),
            "element_size": _elements(panel_a),
        }
        for title in titles
    )
    settings = {
        "figsize": _figure_inches(conf),
        "outer_grid": {
            "height_ratios": (float(panel_a["height_weight"]), float(panel_b["height_weight"])),
            **_margins(layout),
            "hspace": float(spacing["vertical"]),
        },
        "visual_grid": {
            "rows": int(inner["rows"]),
            "cols": int(inner["columns"]),
            "wspace": float(inner["wspace"]),
            "hspace": float(inner.get("hspace", 0.0)),
        },
        "visual_panels": visual_panels,
        "complexity": {
            "xlabel": str(dict(axes_b["x"])["label"]),
            "ylabel": str(dict(axes_b["y"])["label"]),
            "axes": axes_b,
            "element_size": _elements(panel_b),
        },
        "panel_labels": {"A": _label(panel_a), "B": _label(panel_b)},
        "panel_label_layout": deepcopy(dict(conf["panel_label_layout"])),
    }
    _run(
        figure_id,
        lambda item: render_quantitative.render_mnist_data_complexity(
            figure_id=figure_id, domain=domain, settings=item
        ),
        settings,
        _display_width(conf),
    )


def _landscape(conf: dict[str, Any], *, figure_id: str, domain: str) -> None:
    layout = dict(conf["layout"])
    spacing = dict(conf["panel_spacing"])
    panels = {key: dict(value) for key, value in dict(conf["panels"]).items()}
    panel_c = panels["C"]
    panel_d = panels["D"]

    def profile_settings(key: str) -> dict[str, Any]:
        panel = panels[key]
        axes = _axes(panel)
        result: dict[str, Any] = {
            "xlabel": str(dict(axes["x"])["label"]),
            "ylabel": str(dict(axes["y"])["label"]),
            "title": "",
            "title_kwargs": _title_kwargs(panel),
            "axes": axes,
            "element_size": _elements(panel),
        }
        result.update(deepcopy(dict(panel.get("annotations", {}))))
        return result

    settings = {
        "figsize": _figure_inches(conf),
        "grid": {
            "width_ratios": (
                float(panels["A"]["width_weight"]),
                float(panels["B"]["width_weight"]),
                float(panel_c["width_weight"]),
                float(dict(conf["colorbar"])["width_weight"]),
            ),
            "height_ratios": (float(panel_c["height_weight"]), float(panel_d["height_weight"])),
            **_margins(layout),
            "wspace": float(spacing["horizontal"]),
            "hspace": float(spacing["vertical"]),
        },
        "phi_panel": profile_settings("A"),
        "derivative_panel": profile_settings("B"),
        "accuracy_panel": profile_settings("C"),
        "atv_panel": profile_settings("D"),
        "offset_text": deepcopy(dict(conf.get("offset_text", {}))),
        "panel_labels": {key: _label(panels[key]) for key in ("A", "B", "C", "D")},
        "panel_label_layout": deepcopy(dict(conf["panel_label_layout"])),
        "colorbar": deepcopy(dict(conf["colorbar"])),
    }
    settings["colorbar"].pop("width_weight", None)
    _run(
        figure_id,
        lambda item: render_quantitative.render_landscape_summary(
            domain, figure_id, settings=item
        ),
        settings,
        _display_width(conf),
    )


def fig04(conf: dict[str, Any]) -> None:
    _landscape(conf, figure_id="fig04_synthetic_landscape", domain="synthetic")


def fig05(conf: dict[str, Any]) -> None:
    _mnist_data_complexity(
        conf,
        figure_id="fig05_label_noise_data_complexity",
        domain="mnist_label_noise",
    )


def fig06(conf: dict[str, Any]) -> None:
    _landscape(
        conf, figure_id="fig06_label_noise_landscape", domain="mnist_label_noise"
    )


def fig07(conf: dict[str, Any]) -> None:
    _mnist_data_complexity(
        conf,
        figure_id="fig07_digit_pair_data_complexity",
        domain="mnist_digit_pair",
    )


def fig08(conf: dict[str, Any]) -> None:
    _landscape(
        conf, figure_id="fig08_digit_pair_landscape", domain="mnist_digit_pair"
    )








def fig09(conf: dict[str, Any]) -> None:
    """Render the active Fig. 9(a) plus qualitative 3-D panels (b,c)."""
    layout = dict(conf["layout"])
    spacing = dict(conf["panel_spacing"])
    panels = {key: dict(value) for key, value in dict(conf["panels"]).items()}
    panel_a = panels["A"]
    axes_a = _axes(panel_a)
    empirical = {
        "xlabel": str(dict(axes_a["x"])["label"]),
        "ylabel": str(dict(axes_a["y"])["label"]),
        "title": "",
        "title_kwargs": _title_kwargs(panel_a),
        "axes": axes_a,
        "element_size": _elements(panel_a),
    }
    empirical.update(deepcopy(dict(panel_a.get("curves", {}))))
    panel_settings = []
    for key in ("B", "C"):
        panel = panels[key]
        panel_settings.append(
            {
                "regime": str(panel["regime"]),
                "title": "",
                "subtitle": str(panel["subtitle"]),
                "title_pad": float(panel.get("title_pad", 0.0)),
                "reference_label": str(panel["reference_label"]),
                "view": deepcopy(dict(panel["view"])),
                "element_size": _elements(panel),
            }
        )
    legend = deepcopy(dict(conf["legend"]))
    legend_labels = deepcopy(dict(legend.pop("labels")))
    settings = {
        "figsize": _figure_inches(conf),
        "grid": {
            "width_ratios": tuple(
                float(panels[key]["width_weight"]) for key in ("A", "B", "C")
            ),
            **_margins(layout),
            "wspace": float(spacing["horizontal"]),
        },
        "empirical": empirical,
        "panels": tuple(panel_settings),
        "surface": deepcopy(dict(conf["surface"])),
        "panel_labels": {
            key: _label(panels[key]) for key in ("A", "B", "C")
        },
        "panel_label_layout": deepcopy(dict(conf["panel_label_layout"])),
        "legend": legend,
        "legend_labels": legend_labels,
    }
    _run(
        "fig09_hardening_3d_composite",
        lambda item: render_discussion.render_hardening_3d_composite(settings=item),
        settings,
        _display_width(conf),
    )





def final_audit(display_width: int = 1000) -> dict[str, object]:
    receipt = build_all.audit()
    make_visual_audit_sheet.main()
    expected = len(paper_style.MANIFEST["figures"])
    assert receipt["status"] == "complete"
    assert receipt["figure_count"] == expected
    assert all(len(item["files"]) == 3 for item in receipt["outputs"])
    print(
        f"Notebook rebuild PASS: {expected} figures; "
        f"{sum(len(item['files']) for item in receipt['outputs'])} exported files"
    )
    display(
        IPythonImage(
            filename=str(make_visual_audit_sheet.OUTPUT), width=int(display_width)
        )
    )
    return receipt
