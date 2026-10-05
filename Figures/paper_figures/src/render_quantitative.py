#!/usr/bin/env python3
"""Render the active theory, synthetic, and MNIST paper figures."""

from __future__ import annotations

import json
import math

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pymupdf
from sklearn.neighbors import KNeighborsRegressor

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
)


VIRIDIS = mpl.colormaps["viridis"]
STROKES = STYLE["strokes"]
TYPOGRAPHY = STYLE["typography"]


def _read_csv(relative: str) -> pd.DataFrame:
    frame = pd.read_csv(input_path(relative))
    if frame.empty:
        raise ValueError(f"empty figure input: {relative}")
    return frame


def _mean_se(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    if array.size < 2:
        raise ValueError("a dataset-level standard error needs at least two values")
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def _aggregate_cms(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "condition_order", "dataset_index", "C_MS"}
    if not required.issubset(frame.columns):
        raise ValueError(f"C_MS table missing columns: {sorted(required - set(frame))}")
    records: list[dict[str, float | int | str]] = []
    for condition, rows in frame.groupby("condition", sort=False):
        mean, se = _mean_se(rows["C_MS"])
        orders = set(rows["condition_order"].astype(int))
        if len(orders) != 1 or rows["dataset_index"].duplicated().any():
            raise ValueError(f"inconsistent C_MS metadata: {condition}")
        record: dict[str, float | int | str] = {
            "condition": str(condition),
            "condition_order": orders.pop(),
            "mean": mean,
            "se": se,
            "dataset_count": len(rows),
        }
        for field in ("data_beta", "noise_eta", "digit_a", "digit_b", "pair_rank"):
            if field in rows:
                values = set(rows[field])
                if len(values) != 1:
                    raise ValueError(f"{condition}: inconsistent {field}")
                record[field] = values.pop()
        records.append(record)
    return pd.DataFrame(records).sort_values("condition_order").reset_index(drop=True)


def _profile_sources(domain: str) -> tuple[str, str, str]:
    root = {
        "synthetic": "synthetic",
        "mnist_label_noise": "mnist_label_noise",
        "mnist_digit_pair": "mnist_digit_pair",
    }[domain]
    return (
        f"figure_inputs/{root}/energetic_profiles.csv",
        f"figure_inputs/{root}/condition_metrics.csv",
        f"figure_inputs/{root}/cms_by_dataset.csv",
    )


def _load_profiles(domain: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile_path, metric_path, cms_path = _profile_sources(domain)
    profiles = _read_csv(profile_path)
    metrics = _read_csv(metric_path).sort_values("condition_order")
    cms = _aggregate_cms(_read_csv(cms_path))
    conditions = set(profiles["condition"])
    if conditions != set(metrics["condition"]) or conditions != set(cms["condition"]):
        raise ValueError(f"condition join mismatch for {domain}")
    if set(profiles["domain"]) != {domain}:
        raise ValueError(f"unexpected domain rows for {domain}")
    return profiles, metrics, cms


def render_perceptron(*, settings: dict[str, object] | None = None) -> None:
    config: dict[str, object] = {
        "figsize": figure_size("fig02_perceptron_full_shell"),
        "constrained_layout": True,
        "subplot_adjust": None,
        "xlabel": r"Shell radius $r$",
        "ylabel": r"Centered full-shell entropy $\Delta\Phi^{\mathrm{shell}}(r)$",
        "title": "",
        "title_kwargs": {"loc": "left", "pad": 3.0},
        "legend_labels": {
            "analytic": "RS prediction",
            "system_sizes": {
                40: r"SMC $N=40$",
                80: r"SMC $N=80$",
                160: r"SMC $N=160$",
                320: r"SMC $N=320$",
            },
        },
        "legend": {
            "analytic": {
                "loc": "lower left",
                "bbox_to_anchor": (0.18, 0.22),
                "ncol": 1,
                "handlelength": 1.55,
                "borderaxespad": 0.0,
            },
            "smc": {
                "loc": "lower left",
                "bbox_to_anchor": (0.18, 0.02),
                "ncol": 2,
                "handlelength": 1.55,
                "borderaxespad": 0.0,
            },
        },
        "analytic_linewidth": 1.8,
        "analytic_linestyle": (0, (5.0, 2.3)),
        "smc_linewidth": 1.25,
        "smc_marker": "o",
        "smc_marker_size": 2.0,
        "smc_marker_every": 4,
        "sem_alpha": 0.12,
        "axes": {},
        "element_size": {},
    }
    config = merge_settings(config, settings)
    legend_labels = dict(config["legend_labels"])
    size_labels = {
        int(system_size): str(label)
        for system_size, label in dict(legend_labels["system_sizes"]).items()
    }
    analytic = _read_csv("figure_inputs/theory/phi_by_analytic_solution_alpha0p1.csv")
    sampling = _read_csv("figure_inputs/theory/phi_by_sampling.csv")
    analytic = analytic.sort_values("r")
    if len(analytic) != 42 or not analytic["quadrature_convergence_passed"].all():
        raise ValueError("perceptron analytic authority failed its frozen contract")
    if not analytic["root_success"].all():
        raise ValueError("perceptron analytic authority contains an unsuccessful root")

    figure, axis = plt.subplots(
        figsize=config["figsize"],
        constrained_layout=bool(config["constrained_layout"]),
    )
    axis.plot(
        analytic["r"],
        analytic["phi_rel"],
        color="#202020",
        linestyle=config["analytic_linestyle"],
        linewidth=float(config["analytic_linewidth"]),
        label=str(legend_labels["analytic"]),
        zorder=6,
    )
    sizes = (40, 80, 160, 320)
    if set(size_labels) != set(sizes):
        raise ValueError(
            f"legend_labels.system_sizes must define exactly {sizes}"
        )
    colors = mpl.colormaps["viridis"](np.linspace(0.15, 0.85, len(sizes)))
    rs_profile = analytic["phi_rel"].to_numpy(float)
    rmse: list[float] = []
    for system_size, color in zip(sizes, colors, strict=True):
        rows = sampling.loc[
            (sampling["N"] == system_size)
            & (sampling["total_particles"] == 32768)
        ].sort_values("radius")
        if len(rows) != len(analytic):
            raise ValueError(f"N={system_size}: incomplete sampling radius grid")
        radius = rows["radius"].to_numpy(float)
        baseline_radius = float(rows["baseline_radius"].iloc[0])
        angular = rows["phi_energy_rel"].to_numpy(float)
        geometry = ((system_size - 2.0) / system_size) * np.log(
            radius / baseline_radius
        )
        mean = angular + geometry
        sem = rows["phi_energy_rel_dataset_sem"].to_numpy(float)
        rmse.append(float(np.sqrt(np.mean((mean - rs_profile) ** 2))))
        axis.fill_between(
            radius,
            mean - sem,
            mean + sem,
            color=color,
            alpha=float(config["sem_alpha"]),
        )
        axis.plot(
            radius,
            mean,
            color=color,
            linewidth=float(config["smc_linewidth"]),
            marker=str(config["smc_marker"]),
            markersize=float(config["smc_marker_size"]),
            markevery=int(config["smc_marker_every"]),
            label=size_labels[system_size],
        )
    if not np.all(np.diff(np.asarray(rmse)) < 0.0):
        raise ValueError("perceptron RS discrepancy does not decrease with N")
    axis.axhline(0.0, color="#666666", linewidth=0.6)
    axis.set_xlabel(str(config["xlabel"]))
    axis.set_ylabel(str(config["ylabel"]))
    if config["title"]:
        axis.set_title(str(config["title"]), **dict(config["title_kwargs"]))
    legend = dict(config["legend"])
    handles, labels = axis.get_legend_handles_labels()
    if set(legend) == {"analytic", "smc"}:
        analytic_legend = axis.legend(
            handles=[handles[0]],
            labels=[labels[0]],
            frameon=False,
            **dict(legend["analytic"]),
        )
        axis.add_artist(analytic_legend)
        axis.legend(
            handles=handles[1:],
            labels=labels[1:],
            frameon=False,
            **dict(legend["smc"]),
        )
    else:
        # Backward-compatible single-block legend for older configurations.
        item_order = legend.pop("item_order", None)
        if item_order is not None:
            indices = [int(value) - 1 for value in tuple(item_order)]
            if sorted(indices) != list(range(len(handles))):
                raise ValueError(
                    f"legend.item_order must be a 1-based permutation of 1..{len(handles)}"
                )
            handles = [handles[index] for index in indices]
            labels = [labels[index] for index in indices]
        axis.legend(handles, labels, frameon=False, **legend)
    style_axes(axis)
    apply_axis_config(axis, dict(config["axes"]))
    scale_axis_box(axis, dict(config["element_size"]))
    if config["subplot_adjust"]:
        figure.subplots_adjust(**dict(config["subplot_adjust"]))
    save_figure(figure, "fig02_perceptron_full_shell")
    plt.close(figure)


def _draw_synthetic_dataset(
    axis: plt.Axes,
    beta_slug: str,
    *,
    settings: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "title_template": r"$\beta_{\mathrm{data}}={beta}$",
        "title_kwargs": {
            "pad": 1.2,
            "fontsize": TYPOGRAPHY["annotation_size_pt"],
        },
        "xlim": (-1.05, 1.05),
        "ylim": (-1.05, 1.05),
        "negative_marker_size": 1.8,
        "positive_marker_size": 1.8,
        "element_size": {},
    }
    config = merge_settings(config, settings)
    path = input_path(
        f"figure_inputs/synthetic/dataset_000/data_beta_{beta_slug}.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        x = np.asarray(payload["X_raw"], dtype=float)
        y = np.asarray(payload["y"], dtype=float)
    if x.shape != (512, 2) or y.shape != (512,):
        raise ValueError(f"unexpected representative dataset shape: {path}")
    grid_axis = np.linspace(-1.05, 1.05, 180)
    xx, yy = np.meshgrid(grid_axis, grid_axis, indexing="xy")
    regressor = KNeighborsRegressor(n_neighbors=15, weights="distance")
    regressor.fit(x, y)
    score = regressor.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(
        xx.shape
    )

    axis.pcolormesh(
        grid_axis,
        grid_axis,
        (score >= 0.0).astype(np.int8),
        shading="nearest",
        cmap=ListedColormap(["#EFE6CC", "#7FA9C4"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axis.contour(xx, yy, score, levels=[0.0], colors="#262626", linewidths=0.5)
    positive = y > 0
    axis.scatter(
        x[~positive, 0],
        x[~positive, 1],
        s=float(config["negative_marker_size"]),
        facecolor="white",
        edgecolor="#6B5133",
        linewidth=0.22,
        alpha=0.76,
    )
    axis.scatter(
        x[positive, 0],
        x[positive, 1],
        s=float(config["positive_marker_size"]),
        color="#254E6C",
        linewidth=0.0,
        alpha=0.70,
    )
    beta = beta_slug.replace("p", ".")
    axis.set_title(
        str(config["title_template"]).replace("{beta}", beta),
        **dict(config["title_kwargs"]),
    )
    axis.set(
        xlim=tuple(config["xlim"]),
        ylim=tuple(config["ylim"]),
        xticks=[],
        yticks=[],
    )
    axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#555555")
    scale_axis_box(axis, dict(config["element_size"]))


def _draw_complexity_curve(
    axis: plt.Axes,
    aggregated: pd.DataFrame,
    *,
    x_field: str,
    xlabel: str,
    settings: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "xlabel": xlabel,
        "ylabel": r"dataset complexity $\mathcal{C}_{\mathrm{MS}}$",
        "line_color": "#666666",
        "line_width": 0.85,
        "marker_size": 4.0,
        "capsize": 1.5,
        "axes": {},
        "element_size": {},
    }
    config = merge_settings(config, settings)
    norm = mpl.colors.Normalize(aggregated["mean"].min(), aggregated["mean"].max())
    colors = VIRIDIS(norm(aggregated["mean"].to_numpy(float)))
    x = aggregated[x_field].to_numpy(float)
    axis.plot(
        x,
        aggregated["mean"],
        color=str(config["line_color"]),
        linewidth=float(config["line_width"]),
        zorder=0,
    )
    for index, row in aggregated.iterrows():
        axis.errorbar(
            float(row[x_field]),
            float(row["mean"]),
            yerr=float(row["se"]),
            fmt="o",
            markersize=float(config["marker_size"]),
            color=colors[index],
            ecolor=colors[index],
            elinewidth=float(STROKES["errorbar_line_width_pt"]),
            capsize=float(config["capsize"]),
        )
    axis.set_xlabel(str(config["xlabel"]))
    axis.set_ylabel(str(config["ylabel"]))
    style_axes(axis)
    apply_axis_config(axis, dict(config["axes"]))
    scale_axis_box(axis, dict(config["element_size"]))


def render_synthetic_data_complexity(
    *, settings: dict[str, object] | None = None
) -> None:
    config: dict[str, object] = {
        "figsize": figure_size("fig03_synthetic_data_complexity"),
        "outer_grid": {
            "height_ratios": (1.22, 0.78),
            "left": 0.16,
            "right": 0.97,
            "bottom": 0.11,
            "top": 0.88,
            "hspace": 0.30,
        },
        "montage_grid": {"rows": 2, "cols": 2, "wspace": 0.02, "hspace": 0.16},
        "beta_slugs": ("0p05", "0p15", "0p27", "0p39"),
        "dataset_panel": {},
        "panel_label": {
            "x": 0.075,
            "y": 0.975,
            "text": "(a)  Synthetic label configurations",
            "ha": "left",
            "va": "top",
            "fontsize": TYPOGRAPHY["panel_label_size_pt"],
        },
        "complexity": {
            "xlabel": r"generation parameter $\beta_{\mathrm{data}}$",
            "ylabel": r"dataset complexity $\mathcal{C}_{\mathrm{MS}}$",
        },
        "complexity_title": {
            "text": "(b)  Controlled complexity sweep",
            "loc": "left",
            "pad": 3.0,
        },
        "panel_labels": None,
        "panel_label_layout": None,
    }
    config = merge_settings(config, settings)
    figure = plt.figure(figsize=config["figsize"])
    outer = figure.add_gridspec(
        2,
        1,
        **dict(config["outer_grid"]),
    )
    montage_config = dict(config["montage_grid"])
    montage = outer[0].subgridspec(
        int(montage_config.pop("rows")),
        int(montage_config.pop("cols")),
        **montage_config,
    )
    beta_slugs = tuple(config["beta_slugs"])
    montage_axes = [figure.add_subplot(montage[index]) for index in range(4)]
    for axis, beta_slug in zip(montage_axes, beta_slugs, strict=True):
        _draw_synthetic_dataset(
            axis, str(beta_slug), settings=dict(config["dataset_panel"])
        )
    if config["panel_labels"] is None:
        panel_label = dict(config["panel_label"])
        figure.text(
            float(panel_label.pop("x")),
            float(panel_label.pop("y")),
            str(panel_label.pop("text")),
            **panel_label,
        )

    complexity_axis = figure.add_subplot(outer[1])
    aggregated = _aggregate_cms(
        _read_csv("figure_inputs/synthetic/cms_by_dataset.csv")
    )
    _draw_complexity_curve(
        complexity_axis,
        aggregated,
        x_field="data_beta",
        xlabel=str(dict(config["complexity"])["xlabel"]),
        settings=dict(config["complexity"]),
    )
    if config["panel_labels"] is None:
        complexity_title = dict(config["complexity_title"])
        complexity_axis.set_title(
            str(complexity_title.pop("text")), **complexity_title
        )
    else:
        panel_labels = dict(config["panel_labels"])
        place_aligned_panel_labels(
            figure,
            {"A": tuple(montage_axes), "B": (complexity_axis,)},
            panel_labels,
            dict(config["panel_label_layout"]),
        )
    save_figure(figure, "fig03_synthetic_data_complexity")
    plt.close(figure)


def _pdf_page_rgb(relative: str, *, dpi: int = 400) -> np.ndarray:
    path = input_path(relative)
    with pymupdf.open(path) as document:
        if document.page_count != 1:
            raise ValueError(f"{path}: expected one PDF page")
        pixmap = document[0].get_pixmap(
            matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0), alpha=False
        )
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    return array.reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3]


def _draw_static_visual(
    axis: plt.Axes,
    relative: str,
    title: str,
    *,
    settings: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "title": title,
        "title_kwargs": {"pad": 1.8},
        "interpolation": "lanczos",
        "element_size": {},
    }
    config = merge_settings(config, settings)
    axis.imshow(
        _pdf_page_rgb(relative), interpolation=str(config["interpolation"])
    )
    axis.set_title(str(config["title"]), **dict(config["title_kwargs"]))
    axis.axis("off")
    scale_axis_box(axis, dict(config["element_size"]))


def _validated_digit_complexity() -> pd.DataFrame:
    aggregated = _aggregate_cms(
        _read_csv("figure_inputs/mnist_digit_pair/cms_by_dataset.csv")
    )
    manifest = json.loads(
        input_path(
            "figure_inputs/mnist_digit_pair/frozen_pair_manifest.json"
        ).read_text(encoding="utf-8")
    )
    expected = [
        (item["pair_id"], int(item["rank"]), int(item["digit_a"]), int(item["digit_b"]))
        for item in manifest["expected_selected_pairs"]
    ]
    observed = [
        (
            str(row.condition),
            int(row.pair_rank),
            int(row.digit_a),
            int(row.digit_b),
        )
        for row in aggregated.itertuples()
    ]
    if observed != expected or observed[0][0] != "pair_4_9" or observed[-1][0] != "pair_0_1":
        raise ValueError("digit-pair C_MS rows do not match the Study23 authority")
    if not np.all(np.diff(aggregated["mean"].to_numpy(float)) < 0.0):
        raise ValueError("digit-pair mean C_MS is not decreasing in frozen rank order")
    return aggregated


def _draw_digit_complexity(
    axis: plt.Axes,
    aggregated: pd.DataFrame,
    *,
    settings: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "xlabel": "digit pair",
        "ylabel": r"dataset complexity $\mathcal{C}_{\mathrm{MS}}$",
        "tick_rotation": 45,
        "tick_ha": "right",
        "axes": {},
        "element_size": {},
    }
    config = merge_settings(config, settings)
    norm = mpl.colors.Normalize(aggregated["mean"].min(), aggregated["mean"].max())
    colors = VIRIDIS(norm(aggregated["mean"].to_numpy(float)))
    x = np.arange(len(aggregated))
    axis.bar(
        x,
        aggregated["mean"],
        yerr=aggregated["se"],
        color=colors,
        edgecolor="#303030",
        linewidth=0.32,
        error_kw={
            "elinewidth": float(STROKES["errorbar_line_width_pt"]),
            "capsize": 1.4,
            "capthick": float(STROKES["errorbar_line_width_pt"]),
        },
    )
    labels = [f"{int(row.digit_a)}/{int(row.digit_b)}" for row in aggregated.itertuples()]
    axis.set_xticks(
        x,
        labels,
        rotation=float(config["tick_rotation"]),
        ha=str(config["tick_ha"]),
    )
    axis.set_xlabel(str(config["xlabel"]))
    axis.set_ylabel(str(config["ylabel"]))
    style_axes(axis, grid_axis="y")
    apply_axis_config(axis, dict(config["axes"]))
    scale_axis_box(axis, dict(config["element_size"]))


def render_mnist_data_complexity(
    *,
    figure_id: str,
    domain: str,
    settings: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "figsize": figure_size(figure_id),
        "outer_grid": {
            "height_ratios": (1.08, 0.82),
            "left": 0.17,
            "right": 0.97,
            "bottom": 0.115,
            "top": 0.935,
            "hspace": 0.31,
        },
        "visual_grid": {"rows": 1, "cols": 2, "wspace": 0.08},
        "visual_panels": ({}, {}),
        "panel_label": {
            "x": 0.075,
            "y": 0.963,
            "text": "(a)  Dataset visualization",
            "ha": "left",
            "va": "top",
            "fontsize": TYPOGRAPHY["panel_label_size_pt"],
        },
        "complexity": {},
        "complexity_title": {
            "text": "(b)  Measured dataset complexity",
            "loc": "left",
            "pad": 3.0,
        },
        "panel_labels": None,
        "panel_label_layout": None,
    }
    config = merge_settings(config, settings)
    if domain == "mnist_label_noise":
        visuals = (
            (
                "figure_inputs/static_assets/label_noise_eta_0p0_composed.pdf",
                r"clean labels, $\eta=0$",
            ),
            (
                "figure_inputs/static_assets/label_noise_eta_0p5_composed.pdf",
                r"random labels, $\eta=0.5$",
            ),
        )
        aggregated = _aggregate_cms(
            _read_csv("figure_inputs/mnist_label_noise/cms_by_dataset.csv")
        )
    elif domain == "mnist_digit_pair":
        visuals = (
            (
                "figure_inputs/static_assets/digit_pair_highest_complexity_pair_4_9_composed.pdf",
                "higher complexity: 4/9",
            ),
            (
                "figure_inputs/static_assets/digit_pair_lowest_complexity_pair_0_1_composed.pdf",
                "lower complexity: 0/1",
            ),
        )
        aggregated = _validated_digit_complexity()
    else:
        raise ValueError(domain)

    figure = plt.figure(figsize=config["figsize"])
    outer = figure.add_gridspec(
        2,
        1,
        **dict(config["outer_grid"]),
    )
    visual_grid_config = dict(config["visual_grid"])
    visual_grid = outer[0].subgridspec(
        int(visual_grid_config.pop("rows")),
        int(visual_grid_config.pop("cols")),
        **visual_grid_config,
    )
    visual_axes = [figure.add_subplot(visual_grid[index]) for index in range(2)]
    visual_panels = tuple(config["visual_panels"])
    for index, (axis, (source, title)) in enumerate(
        zip(visual_axes, visuals, strict=True)
    ):
        panel_settings = dict(visual_panels[index]) if index < len(visual_panels) else {}
        _draw_static_visual(axis, source, title, settings=panel_settings)
    if config["panel_labels"] is None:
        panel_label = dict(config["panel_label"])
        figure.text(
            float(panel_label.pop("x")),
            float(panel_label.pop("y")),
            str(panel_label.pop("text")),
            **panel_label,
        )

    complexity_axis = figure.add_subplot(outer[1])
    if domain == "mnist_label_noise":
        _draw_complexity_curve(
            complexity_axis,
            aggregated,
            x_field="noise_eta",
            xlabel=r"label-noise probability $\eta$",
            settings=dict(config["complexity"]),
        )
    else:
        _draw_digit_complexity(
            complexity_axis, aggregated, settings=dict(config["complexity"])
        )
    if config["panel_labels"] is None:
        complexity_title = dict(config["complexity_title"])
        complexity_axis.set_title(
            str(complexity_title.pop("text")), **complexity_title
        )
    else:
        panel_labels = dict(config["panel_labels"])
        place_aligned_panel_labels(
            figure,
            {"A": tuple(visual_axes), "B": (complexity_axis,)},
            panel_labels,
            dict(config["panel_label_layout"]),
        )
    save_figure(figure, figure_id)
    plt.close(figure)


def _draw_profile_axis(
    axis: plt.Axes,
    profiles: pd.DataFrame,
    cms: pd.DataFrame,
    norm: mpl.colors.Normalize,
    *,
    derivative: bool,
    settings: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "xlabel": r"radius $r$",
        "ylabel": (
            r"$g_{\mathrm{E}}(r)=\partial_r\phi_{\mathrm{E}}(r)$"
            if derivative
            else r"$\Delta\phi_{\mathrm{E}}(r)$"
        ),
        "title": (
            "(b)  Radial derivative"
            if derivative
            else "(a)  Centered local entropy"
        ),
        "title_kwargs": {"loc": "left", "pad": 3.0},
        "xlim": None,
        "r1_label": {
            "x": 0.985,
            "y": 0.975,
            "text": r"$r=1$",
            "ha": "right",
            "va": "top",
            "fontsize": TYPOGRAPHY["annotation_size_pt"],
            "color": "#444444",
        },
        "axes": {},
        "element_size": {},
    }
    config = merge_settings(config, settings)
    cms_by_condition = dict(zip(cms["condition"], cms["mean"], strict=True))
    mean_field = "dphi_energetic_dr_direct_mean" if derivative else "phi_energetic_mean"
    se_field = (
        "dphi_energetic_dr_direct_se_across_datasets"
        if derivative
        else "phi_energetic_se_across_datasets"
    )
    for condition, rows in profiles.groupby("condition", sort=False):
        rows = rows.sort_values("radius")
        color = VIRIDIS(norm(cms_by_condition[str(condition)]))
        radius = rows["radius"].to_numpy(float)
        mean = rows[mean_field].to_numpy(float)
        se = rows[se_field].to_numpy(float)
        axis.fill_between(radius, mean - se, mean + se, color=color, alpha=0.065)
        axis.plot(radius, mean, color=color, linewidth=0.86, alpha=0.97)
    axis.set_xlabel(str(config["xlabel"]))
    if derivative:
        axis.axhline(0.0, color="#555555", linewidth=0.62)
        axis.axvline(1.0, color="#555555", linewidth=0.72, linestyle=":", zorder=7)
        r1_label = dict(config["r1_label"])
        axis.text(
            float(r1_label.pop("x")),
            float(r1_label.pop("y")),
            str(r1_label.pop("text")),
            transform=axis.get_xaxis_transform(),
            **r1_label,
        )
    else:
        axis.axhline(0.0, color="#777777", linewidth=0.52)
    axis.set_ylabel(str(config["ylabel"]))
    axis.set_title(str(config["title"]), **dict(config["title_kwargs"]))
    maximum_radius = float(profiles["radius"].max())
    if config["xlim"] is not None:
        axis.set_xlim(*tuple(config["xlim"]))
    elif maximum_radius <= 1.000001:
        axis.set_xlim(0.0, 1.055)
    style_axes(axis)
    apply_axis_config(axis, dict(config["axes"]))
    scale_axis_box(axis, dict(config["element_size"]))


def _metric_limits(values: np.ndarray, errors: np.ndarray, *, floor: float | None = None) -> tuple[float, float]:
    lower = float(np.min(values - errors))
    upper = float(np.max(values + errors))
    span = max(upper - lower, 1.0e-8)
    lower -= 0.16 * span
    upper += 0.16 * span
    if floor is not None:
        lower = max(floor, lower)
    return lower, upper


def _draw_metric_axis(
    axis: plt.Axes,
    merged: pd.DataFrame,
    norm: mpl.colors.Normalize,
    *,
    y_field: str,
    yerr_field: str,
    ylabel: str,
    title: str,
    connected: bool,
    settings: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "xlabel": r"$\mathcal{C}_{\mathrm{MS}}$",
        "ylabel": ylabel,
        "title": title,
        "title_kwargs": {"loc": "left", "pad": 3.0},
        "xlim": None,
        "ylim": None,
        "marker_size": 3.4,
        "axes": {},
        "element_size": {},
    }
    config = merge_settings(config, settings)
    x = merged["mean"].to_numpy(float)
    xerr = merged["se"].to_numpy(float)
    y = merged[y_field].to_numpy(float)
    yerr = merged[yerr_field].to_numpy(float)
    colors = VIRIDIS(norm(x))
    if connected:
        axis.plot(x, y, color="#707070", linewidth=0.75, zorder=0)
    for index in range(len(merged)):
        axis.errorbar(
            x[index],
            y[index],
            xerr=xerr[index],
            yerr=yerr[index],
            fmt="o",
            markersize=float(config["marker_size"]),
            color=colors[index],
            ecolor=colors[index],
            elinewidth=0.58,
            capsize=1.15,
            markeredgecolor="white",
            markeredgewidth=0.32,
            zorder=2,
        )
    axis.set_xlabel(str(config["xlabel"]))
    axis.set_ylabel(str(config["ylabel"]))
    axis.set_title(str(config["title"]), **dict(config["title_kwargs"]))
    axis.set_xlim(
        *(tuple(config["xlim"]) if config["xlim"] is not None else _metric_limits(x, xerr, floor=0.0))
    )
    axis.set_ylim(
        *(tuple(config["ylim"]) if config["ylim"] is not None else _metric_limits(y, yerr, floor=0.0))
    )
    style_axes(axis)
    axis.tick_params(labelsize=TYPOGRAPHY["annotation_size_pt"])
    apply_axis_config(axis, dict(config["axes"]))
    scale_axis_box(axis, dict(config["element_size"]))


def render_landscape_summary(
    domain: str,
    figure_id: str,
    *,
    settings: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "figsize": figure_size(figure_id),
        "grid": {
            "width_ratios": (1.08, 1.08, 0.80, 0.055),
            "height_ratios": (1.0, 1.0),
            "left": 0.078,
            "right": 0.982,
            "bottom": 0.145,
            "top": 0.915,
            "wspace": 0.58,
            "hspace": 0.60,
        },
        "phi_panel": {},
        "derivative_panel": {},
        "accuracy_panel": {
            "ylabel": r"accuracy at $r=1$",
            "title": "(c)  Retained accuracy",
        },
        "atv_panel": {
            "ylabel": r"$\mathcal{A}_{\mathrm{TV}}$",
            "title": "(d)  Radial variation",
        },
        "panel_labels": None,
        "panel_label_layout": None,
        "colorbar": {
            "orientation": "vertical",
            "tick_position": "left",
            "title": r"$\mathcal{C}_{\mathrm{MS}}$",
            "title_kwargs": {
                "fontsize": TYPOGRAPHY["annotation_size_pt"],
                "pad": 3.0,
            },
            "tick_params": {
                "labelsize": TYPOGRAPHY["annotation_size_pt"],
                "length": 2.0,
            },
            "elements": {},
        },
    }
    config = merge_settings(config, settings)
    profiles, metrics, cms = _load_profiles(domain)
    merged = metrics.merge(
        cms[["condition", "mean", "se"]], on="condition", validate="one_to_one"
    ).sort_values("condition_order")
    norm = mpl.colors.Normalize(cms["mean"].min(), cms["mean"].max())
    connected = domain != "mnist_digit_pair"

    figure = plt.figure(figsize=config["figsize"])
    grid = figure.add_gridspec(
        2,
        4,
        **dict(config["grid"]),
    )
    phi_axis = figure.add_subplot(grid[:, 0])
    derivative_axis = figure.add_subplot(grid[:, 1])
    accuracy_axis = figure.add_subplot(grid[0, 2])
    atv_axis = figure.add_subplot(grid[1, 2])
    colorbar_axis = figure.add_subplot(grid[:, 3])

    _draw_profile_axis(
        phi_axis,
        profiles,
        cms,
        norm,
        derivative=False,
        settings=dict(config["phi_panel"]),
    )
    _draw_profile_axis(
        derivative_axis,
        profiles,
        cms,
        norm,
        derivative=True,
        settings=dict(config["derivative_panel"]),
    )
    _draw_metric_axis(
        accuracy_axis,
        merged,
        norm,
        y_field="r1_weighted_accuracy_mean",
        yerr_field="r1_weighted_accuracy_se_across_datasets",
        ylabel=r"accuracy at $r=1$",
        title="(c)  Retained accuracy",
        connected=connected,
        settings=dict(config["accuracy_panel"]),
    )
    _draw_metric_axis(
        atv_axis,
        merged,
        norm,
        y_field="A_TV_mean",
        yerr_field="A_TV_se_across_datasets",
        ylabel=r"$\mathcal{A}_{\rm TV}$",
        title="(d)  Radial variation",
        connected=connected,
        settings=dict(config["atv_panel"]),
    )
    atv_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    atv_axis.yaxis.set_offset_position("right")
    shared_offset = dict(config.get("offset_text", {}))
    atv_axes = dict(dict(config["atv_panel"]).get("axes", {}))
    atv_y_axis = dict(atv_axes.get("y", {}))
    offset_config = merge_settings(
        shared_offset,
        dict(atv_y_axis.get("offset_text", {})),
    )
    offset_text = atv_axis.yaxis.get_offset_text()
    offset_text.set_fontsize(
        float(offset_config.get("fontsize", TYPOGRAPHY["annotation_size_pt"]))
    )
    offset_text.set_horizontalalignment(str(offset_config.get("ha", "right")))
    offset_text.set_verticalalignment(str(offset_config.get("va", "bottom")))
    offset_text.set_visible(bool(offset_config.get("visible", True)))
    if offset_config.get("x") is not None:
        offset_text.set_x(float(offset_config["x"]))
    if offset_config.get("y") is not None:
        offset_text.set_y(float(offset_config["y"]))

    colorbar_config = dict(config["colorbar"])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=VIRIDIS),
        cax=colorbar_axis,
        orientation=str(colorbar_config["orientation"]),
    )
    colorbar.ax.yaxis.set_ticks_position(str(colorbar_config["tick_position"]))
    colorbar.ax.set_title(
        str(colorbar_config["title"]),
        **dict(colorbar_config["title_kwargs"]),
    )
    colorbar.ax.tick_params(**dict(colorbar_config["tick_params"]))
    scale_axis_box(colorbar.ax, dict(colorbar_config["elements"]))
    if config["panel_labels"] is not None:
        place_aligned_panel_labels(
            figure,
            {
                "A": (phi_axis,),
                "B": (derivative_axis,),
                "C": (accuracy_axis,),
                "D": (atv_axis,),
            },
            dict(config["panel_labels"]),
            dict(config["panel_label_layout"]),
        )
    save_figure(figure, figure_id)
    plt.close(figure)


def main() -> None:
    apply_style()
    render_perceptron()
    render_synthetic_data_complexity()
    render_landscape_summary("synthetic", "fig04_synthetic_landscape")
    render_mnist_data_complexity(
        figure_id="fig05_label_noise_data_complexity",
        domain="mnist_label_noise",
    )
    render_landscape_summary("mnist_label_noise", "fig06_label_noise_landscape")
    render_mnist_data_complexity(
        figure_id="fig07_digit_pair_data_complexity",
        domain="mnist_digit_pair",
    )
    render_landscape_summary("mnist_digit_pair", "fig08_digit_pair_landscape")


if __name__ == "__main__":
    main()
