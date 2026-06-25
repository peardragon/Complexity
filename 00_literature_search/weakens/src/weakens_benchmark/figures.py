from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, Patch

from .importance import ImportanceResult, normalized_weights
from .landscape import ProxyLandscape
from .samplers import SampleResult


METHOD_COLORS = {
    "truth": "#222222",
    "random_walk_mcmc": "#C44E52",
    "hmc": "#4C72B0",
    "pseudo_langevin": "#DD8452",
    "vmf_l2_final": "#55A868",
}

SCHEMATIC_COLORS = {
    "basin": "#FFF3B0",
    "basin_edge": "#202020",
    "miss": "#C44E52",
    "local": "#6F7D8C",
    "vmf": "#2E8B57",
    "vmf_light": "#BFE8CC",
    "barrier": "#A8A8A8",
    "panel_bg": "#FBFAF7",
}

REGION_CODES = {
    "solution_core": "R1",
    "near_same_valley": "R2",
    "across_barrier": "R3",
    "remote_needle": "R4",
}

REGION_KEY = "R1 core   R2 same valley   R3 across barrier   R4 remote needle"

REGION_COLORS = {
    "solution_core": "#4C72B0",
    "near_same_valley": "#CCB974",
    "across_barrier": "#8172B3",
    "remote_needle": "#64B5CD",
}


def _plot_region_ellipse(ax: plt.Axes, landscape: ProxyLandscape) -> None:
    t = np.linspace(0, 2.0 * np.pi, 240)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    for basin in landscape.basins:
        pts = circle * (basin.axes[None, :] * basin.region_radius)
        pts = pts @ basin.rotation.T + basin.center[None, :]
        ax.plot(pts[:, 0], pts[:, 1], color="white", lw=1.3, alpha=0.95)
        ax.text(
            basin.center[0],
            basin.center[1],
            basin.name.replace("_", "\n"),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            weight="bold",
        )


def make_final_figure(
    landscape: ProxyLandscape,
    grid: dict[str, np.ndarray],
    baseline_results: dict[str, SampleResult],
    importance_result: ImportanceResult,
    region_mass_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()

    energy = grid["energy"]
    cap = np.nanquantile(energy[np.isfinite(energy)], 0.96)
    im = ax0.contourf(grid["xx"], grid["yy"], np.minimum(energy, cap), levels=48, cmap="magma_r")
    ax0.contour(grid["xx"], grid["yy"], energy, levels=16, colors="black", linewidths=0.25, alpha=0.35)
    _draw_energy_schematic_overlay(ax0, landscape, mode="regions")
    _region_key(ax0)
    ax0.set_title("A. Proxy landscape + schematic regions", loc="left", fontsize=12, weight="bold")
    ax0.set_xlabel("collective coordinate 1")
    ax0.set_ylabel("collective coordinate 2")
    ax0.set_xlim(landscape.xlim)
    ax0.set_ylim(landscape.ylim)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02, label="proxy energy")

    ax1.contourf(grid["xx"], grid["yy"], np.minimum(energy, cap), levels=28, cmap="Greys", alpha=0.28)
    _draw_energy_schematic_overlay(ax1, landscape, mode="success")
    _draw_final_retained_trajectory_panel(ax1, landscape, baseline_results, importance_result, qc_df)
    ax1.set_title("B. Retained trajectories and vMF+L2 footprint", loc="left", fontsize=12, weight="bold")
    ax1.set_xlabel("collective coordinate 1")
    ax1.set_ylabel("collective coordinate 2")
    ax1.set_xlim(landscape.xlim)
    ax1.set_ylim(landscape.ylim)

    regions = landscape.region_names()
    methods = ["truth", "random_walk_mcmc", "hmc", "pseudo_langevin", "vmf_l2_final"]
    x = np.arange(len(regions), dtype=np.float64)
    width = 0.15
    offsets = np.linspace(-2.0 * width, 2.0 * width, len(methods))
    truth = region_mass_df.drop_duplicates("region").set_index("region")["truth_mass"]
    for offset, method in zip(offsets, methods):
        if method == "truth":
            vals = [float(truth.loc[r]) for r in regions]
        else:
            vals = [
                float(
                    region_mass_df[
                        (region_mass_df["method"] == method) & (region_mass_df["region"] == region)
                    ]["estimated_mass"].iloc[0]
                )
                for region in regions
            ]
        ax2.bar(x + offset, vals, width=width, color=METHOD_COLORS[method], label=method)
    ax2.set_title("C. Region mass estimates", loc="left", fontsize=12, weight="bold")
    ax2.set_ylabel("target mass / estimated mass")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r.replace("_", "\n") for r in regions], fontsize=8)
    mass_max = max(
        float(region_mass_df["truth_mass"].max()),
        float(region_mass_df["estimated_mass"].max()),
    )
    ax2.set_ylim(0.0, max(0.55, mass_max * 1.12))
    ax2.legend(fontsize=8, ncols=2)

    qc = qc_df.set_index("method").loc[["random_walk_mcmc", "hmc", "pseudo_langevin", "vmf_l2_final"]]
    metrics = ["covered_important_regions", "region_l1_error", "ess_fraction"]
    display = qc[metrics].astype(float).copy()
    display["covered_important_regions"] /= np.maximum(display["covered_important_regions"].max(), 1.0)
    display["region_l1_error"] = 1.0 - np.minimum(display["region_l1_error"], 1.0)
    display["ess_fraction"] = np.minimum(display["ess_fraction"] / max(float(display["ess_fraction"].max()), 1.0e-12), 1.0)
    matrix = display.to_numpy()
    ax3.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax3.set_title("D. QC summary (brighter is better)", loc="left", fontsize=12, weight="bold")
    ax3.set_xticks(np.arange(len(metrics)))
    ax3.set_xticklabels(["coverage", "L1 err", "ESS frac"], fontsize=9)
    ax3.set_yticks(np.arange(qc.shape[0]))
    ax3.set_yticklabels(qc.index.tolist(), fontsize=9)
    for i, method in enumerate(qc.index):
        raw = qc.loc[method]
        labels = [
            f"{int(raw['covered_important_regions'])}/{int(raw['important_regions'])}",
            f"{float(raw['region_l1_error']):.2f}",
            f"{float(raw['ess_fraction']):.3f}",
        ]
        for j, label in enumerate(labels):
            ax3.text(j, i, label, ha="center", va="center", color="white", fontsize=9, weight="bold")
    for spine in ax3.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Complex proxy landscape: existing local samplers miss regions; final vMF+L2 recovers weighted mass",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_ab_clarified_figure(
    landscape: ProxyLandscape,
    grid: dict[str, np.ndarray],
    baseline_results: dict[str, SampleResult],
    importance_result: ImportanceResult,
    qc_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    """Make an A/B-only figure with outside titles and explicit legends."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.80, bottom=0.30, wspace=0.18)

    energy = grid["energy"]
    cap = np.nanquantile(energy[np.isfinite(energy)], 0.96)

    ax0, ax1 = axes
    im = ax0.contourf(grid["xx"], grid["yy"], np.minimum(energy, cap), levels=44, cmap="magma_r")
    ax0.contour(grid["xx"], grid["yy"], energy, levels=12, colors="black", linewidths=0.25, alpha=0.25)
    _draw_ab_region_overlay(ax0, landscape, success=False)
    ax0.set_xlim(landscape.xlim)
    ax0.set_ylim(landscape.ylim)
    ax0.set_xlabel("collective coordinate 1")
    ax0.set_ylabel("collective coordinate 2")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.047, pad=0.018)
    cbar.set_label("proxy energy")

    ax1.contourf(grid["xx"], grid["yy"], np.minimum(energy, cap), levels=24, cmap="Greys", alpha=0.22)
    ax1.contour(grid["xx"], grid["yy"], energy, levels=10, colors="black", linewidths=0.18, alpha=0.12)
    _draw_ab_region_overlay(ax1, landscape, success=True)
    _draw_ab_retained_trajectories(ax1, landscape, baseline_results, importance_result)
    ax1.set_xlim(landscape.xlim)
    ax1.set_ylim(landscape.ylim)
    ax1.set_xlabel("collective coordinate 1")
    ax1.set_ylabel("collective coordinate 2")

    fig.text(0.07, 0.885, "A. Proxy landscape with registered regions", fontsize=13, weight="bold", ha="left")
    fig.text(0.55, 0.885, "B. Retained trajectories and vMF+L2 weighted footprint", fontsize=13, weight="bold", ha="left")
    fig.suptitle("Sampling failure and vMF+L2 recovery on a softened proxy landscape", fontsize=14.5, weight="bold", y=0.97)

    fig.legend(
        handles=_ab_region_legend_handles(landscape),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.165),
        ncols=4,
        frameon=False,
        fontsize=9,
        title="Registered regions",
        title_fontsize=9.5,
    )
    fig.legend(
        handles=_ab_method_legend_handles(qc_df),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.035),
        ncols=5,
        frameon=False,
        fontsize=8.6,
        title="Marks: shaded bands and retained samples",
        title_fontsize=9.5,
    )
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def make_schematic_figure(
    landscape: ProxyLandscape,
    baseline_results: dict[str, SampleResult],
    importance_result: ImportanceResult,
    qc_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    """Make a simplified problem-statement schematic for papers/slides."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.8, 8.6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.92]},
    )
    for ax in axes.ravel():
        _setup_schematic_axis(ax, landscape)

    _draw_schematic_landscape(axes[0, 0], landscape, mode="regions")
    axes[0, 0].set_title("A. Registered hard regions", loc="left", fontsize=13, weight="bold")
    _panel_note(axes[0, 0], "thin valleys + distant low-loss mass")
    _region_key(axes[0, 0])

    _draw_schematic_landscape(axes[0, 1], landscape, mode="miss")
    _draw_local_sampler_failure(axes[0, 1], landscape, baseline_results, qc_df)
    axes[0, 1].set_title("B. HMC / pL / MCMC stay local", loc="left", fontsize=13, weight="bold")

    _draw_schematic_landscape(axes[0, 2], landscape, mode="success")
    _draw_vmf_recovery(axes[0, 2], landscape, importance_result, qc_df)
    axes[0, 2].set_title("C. vMF+L2 proposes directions, then reweights", loc="left", fontsize=13, weight="bold")

    _draw_schematic_landscape(axes[1, 0], landscape, mode="miss")
    _draw_actual_local_trajectories(axes[1, 0], landscape, baseline_results, qc_df)
    axes[1, 0].set_title("D. Actual local-chain traces", loc="left", fontsize=13, weight="bold")

    _draw_schematic_landscape(axes[1, 1], landscape, mode="success")
    _draw_actual_vmf_process(axes[1, 1], landscape, importance_result, qc_df)
    axes[1, 1].set_title("E. Actual vMF+L2 proposal footprint", loc="left", fontsize=13, weight="bold")

    _setup_process_axis(axes[1, 2])
    _draw_coverage_accumulation(axes[1, 2], landscape, baseline_results, importance_result, qc_df)
    axes[1, 2].set_title("F. Coverage over samples", loc="left", fontsize=13, weight="bold")

    fig.suptitle(
        "Complex landscape breaks local samplers; vMF+L2 restores coverage",
        fontsize=14.5,
        weight="bold",
    )
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _setup_schematic_axis(ax: plt.Axes, landscape: ProxyLandscape) -> None:
    ax.set_facecolor(SCHEMATIC_COLORS["panel_bg"])
    ax.set_xlim(landscape.xlim[0] - 0.1, landscape.xlim[1] + 0.1)
    ax.set_ylim(landscape.ylim[0] - 0.1, landscape.ylim[1] + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_energy_schematic_overlay(ax: plt.Axes, landscape: ProxyLandscape, mode: str) -> None:
    ax.plot(
        [-0.2, 0.55, 1.05],
        [-0.03, 0.08, 0.17],
        color="#F2D887",
        lw=16,
        alpha=0.22,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        [1.35, 1.48, 1.62, 1.80],
        [-0.55, 0.15, 0.82, 1.45],
        color="#666666",
        lw=7.0,
        alpha=0.32,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        [-1.15, -1.55, -1.95, -2.35],
        [-0.65, -1.10, -1.47, -1.82],
        color="#666666",
        lw=6.2,
        alpha=0.24,
        solid_capstyle="round",
        zorder=4,
    )
    for basin in landscape.basins:
        face = SCHEMATIC_COLORS["basin"] if mode != "success" else SCHEMATIC_COLORS["vmf_light"]
        edge = "#1F1F1F" if mode != "success" else SCHEMATIC_COLORS["vmf"]
        patch = Ellipse(
            xy=basin.center,
            width=float(2.0 * basin.axes[0] * basin.region_radius * 0.88),
            height=float(2.0 * basin.axes[1] * basin.region_radius * 0.88),
            angle=float(np.degrees(basin.angle)),
            facecolor=face,
            edgecolor=edge,
            lw=1.7,
            alpha=0.54,
            zorder=5,
        )
        ax.add_patch(patch)
        ax.text(
            basin.center[0],
            basin.center[1],
            REGION_CODES.get(basin.name, ""),
            fontsize=9.5,
            ha="center",
            va="center",
            weight="bold",
            color="#202020",
            zorder=6,
        )


def _draw_ab_region_overlay(ax: plt.Axes, landscape: ProxyLandscape, success: bool) -> None:
    ax.plot(
        [-0.2, 0.55, 1.05],
        [-0.03, 0.08, 0.17],
        color="#F2D887",
        lw=13,
        alpha=0.24,
        solid_capstyle="round",
        zorder=4,
    )
    for xs, ys, lw, alpha in [
        ([1.35, 1.48, 1.62, 1.80], [-0.55, 0.15, 0.82, 1.45], 6.0, 0.28),
        ([-1.15, -1.55, -1.95, -2.35], [-0.65, -1.10, -1.47, -1.82], 5.2, 0.20),
    ]:
        ax.plot(xs, ys, color="#606060", lw=lw, alpha=alpha, solid_capstyle="round", zorder=4)

    for basin in landscape.basins:
        color = REGION_COLORS.get(basin.name, "#333333")
        patch = Ellipse(
            xy=basin.center,
            width=float(2.0 * basin.axes[0] * basin.region_radius * 0.88),
            height=float(2.0 * basin.axes[1] * basin.region_radius * 0.88),
            angle=float(np.degrees(basin.angle)),
            facecolor=color,
            edgecolor=color if not success else SCHEMATIC_COLORS["vmf"],
            lw=2.0 if not success else 2.2,
            alpha=0.23 if not success else 0.20,
            zorder=5,
        )
        ax.add_patch(patch)


def _draw_ab_retained_trajectories(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    baseline_results: dict[str, SampleResult],
    importance_result: ImportanceResult,
) -> None:
    specs = [
        ("random_walk_mcmc", "MCMC", METHOD_COLORS["random_walk_mcmc"]),
        ("hmc", "HMC", METHOD_COLORS["hmc"]),
        ("pseudo_langevin", "pL", METHOD_COLORS["pseudo_langevin"]),
    ]
    for method, _, color in specs:
        path = _coarse_path(baseline_results[method].samples, max_points=120)
        if path.shape[0] < 2:
            continue
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.6, alpha=0.58, zorder=7)
        ax.scatter(path[:1, 0], path[:1, 1], s=22, color=color, edgecolors="white", linewidths=0.5, zorder=8)
        end = path[-1]
        ax.plot([end[0] - 0.07, end[0] + 0.07], [end[1] - 0.07, end[1] + 0.07], color=color, lw=1.9, alpha=0.76, zorder=8)
        ax.plot([end[0] - 0.07, end[0] + 0.07], [end[1] + 0.07, end[1] - 0.07], color=color, lw=1.9, alpha=0.76, zorder=8)

    _draw_vmf_anchor_rays(ax, landscape, importance_result, max_rays=44, alpha=0.13, zorder=6)
    _draw_vmf_region_footprint(ax, landscape, importance_result, max_per_region=620, alpha=0.20, zorder=9)
    _draw_found_disks(ax, landscape)


def _ab_region_legend_handles(landscape: ProxyLandscape) -> list[Patch]:
    labels = {
        "solution_core": "R1 solution core",
        "near_same_valley": "R2 same-valley basin",
        "across_barrier": "R3 across barrier",
        "remote_needle": "R4 remote needle",
    }
    return [
        Patch(
            facecolor=REGION_COLORS.get(basin.name, "#333333"),
            edgecolor=REGION_COLORS.get(basin.name, "#333333"),
            alpha=0.35,
            label=labels.get(basin.name, basin.name),
        )
        for basin in landscape.basins
    ]


def _ab_method_legend_handles(qc_df: pd.DataFrame) -> list[object]:
    qc = qc_df.set_index("method")
    covered = int(qc.loc["vmf_l2_final", "covered_important_regions"])
    total = int(qc.loc["vmf_l2_final", "important_regions"])
    return [
        Patch(
            facecolor="#F2D887",
            edgecolor="#D8BE72",
            alpha=0.42,
            label="yellow shaded band: same-valley low-loss corridor / valley floor",
        ),
        Patch(
            facecolor="#606060",
            edgecolor="#606060",
            alpha=0.30,
            label="gray shaded bands: softened proxy ridges or barriers to cross",
        ),
        Line2D([0], [0], color=METHOD_COLORS["random_walk_mcmc"], lw=1.8, alpha=0.75, label="MCMC retained trajectory"),
        Line2D([0], [0], color=METHOD_COLORS["hmc"], lw=1.8, alpha=0.75, label="HMC retained trajectory"),
        Line2D([0], [0], color=METHOD_COLORS["pseudo_langevin"], lw=1.8, alpha=0.75, label="pL retained trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=SCHEMATIC_COLORS["vmf"],
            lw=1.8,
            alpha=0.9,
            markersize=6,
            label=f"vMF+L2 weighted samples and found-region rims ({covered}/{total})",
        ),
    ]


def _draw_schematic_landscape(ax: plt.Axes, landscape: ProxyLandscape, mode: str) -> None:
    # Soft low-energy band connecting the easy local region.
    ax.plot(
        [-0.2, 0.55, 1.05],
        [-0.03, 0.08, 0.17],
        color="#D8BE72",
        lw=17,
        alpha=0.18,
        solid_capstyle="round",
        zorder=0,
    )
    # Deliberately overdraw the two barriers; this is schematic, not a contour plot.
    ax.plot(
        [1.35, 1.48, 1.62, 1.80],
        [-0.55, 0.15, 0.82, 1.45],
        color=SCHEMATIC_COLORS["barrier"],
        lw=8.0,
        alpha=0.46,
        solid_capstyle="round",
        zorder=1,
    )
    ax.plot(
        [-1.15, -1.55, -1.95, -2.35],
        [-0.65, -1.10, -1.47, -1.82],
        color=SCHEMATIC_COLORS["barrier"],
        lw=7.0,
        alpha=0.35,
        solid_capstyle="round",
        zorder=1,
    )
    for basin in landscape.basins:
        alpha = 0.92
        face = SCHEMATIC_COLORS["basin"]
        edge = SCHEMATIC_COLORS["basin_edge"]
        if mode == "miss" and basin.name in {"across_barrier", "remote_needle"}:
            alpha = 0.34
            edge = SCHEMATIC_COLORS["miss"]
        if mode == "success":
            face = SCHEMATIC_COLORS["vmf_light"]
            edge = SCHEMATIC_COLORS["vmf"]
        width = float(2.0 * basin.axes[0] * basin.region_radius * 0.88)
        height = float(2.0 * basin.axes[1] * basin.region_radius * 0.88)
        patch = Ellipse(
            xy=basin.center,
            width=width,
            height=height,
            angle=float(np.degrees(basin.angle)),
            facecolor=face,
            edgecolor=edge,
            lw=2.0,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(patch)
        label = REGION_CODES.get(basin.name, "")
        ax.text(
            basin.center[0],
            basin.center[1],
            label,
            fontsize=10.5,
            ha="center",
            va="center",
            weight="bold",
            color="#202020",
            zorder=4,
        )


def _draw_local_sampler_failure(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    baseline_results: dict[str, SampleResult],
    qc_df: pd.DataFrame,
) -> None:
    core = _basin_center(landscape, "solution_core")
    same = _basin_center(landscape, "near_same_valley")
    envelope_center = 0.58 * core + 0.42 * same
    envelope = Ellipse(
        xy=envelope_center,
        width=1.95,
        height=0.70,
        angle=8,
        facecolor="#DCE2E8",
        edgecolor=SCHEMATIC_COLORS["local"],
        lw=1.5,
        alpha=0.55,
        zorder=4,
    )
    ax.add_patch(envelope)

    _draw_schematic_trace(
        ax,
        np.asarray([[-0.10, 0.18], [0.25, 0.38], [0.62, 0.48], [0.92, 0.58]]),
        METHOD_COLORS["random_walk_mcmc"],
        "",
        label_offset=(0.10, 0.05),
    )
    _draw_schematic_trace(
        ax,
        np.asarray([[-0.04, -0.02], [0.45, 0.05], [0.95, 0.23], [1.28, 0.43]]),
        METHOD_COLORS["hmc"],
        "",
        label_offset=(0.08, 0.08),
    )
    _draw_schematic_trace(
        ax,
        np.asarray([[-0.25, -0.20], [-0.05, -0.34], [0.22, -0.27], [0.38, -0.12]]),
        METHOD_COLORS["pseudo_langevin"],
        "",
        label_offset=(0.08, -0.18),
    )

    for name in ["across_barrier", "remote_needle"]:
        target = _basin_center(landscape, name)
        arrow = FancyArrowPatch(
            posA=core,
            posB=target,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=2.1,
            linestyle=(0, (4, 4)),
            color=SCHEMATIC_COLORS["miss"],
            alpha=0.86,
            connectionstyle="arc3,rad=0.13",
            zorder=2,
        )
        ax.add_patch(arrow)
        ax.text(target[0], target[1] + 0.58, "MISS", fontsize=10, color=SCHEMATIC_COLORS["miss"], ha="center", weight="bold")

    coverage = _coverage_range(qc_df, ["random_walk_mcmc", "hmc", "pseudo_langevin"])
    _badge(
        ax,
        0.50,
        0.93,
        f"coverage {coverage}/4",
        face="#F7DFDF",
        edge=SCHEMATIC_COLORS["miss"],
        text_color="#7F1D1D",
    )
    _panel_note(ax, "finite-budget chains miss distant mass")


def _draw_vmf_recovery(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    importance_result: ImportanceResult,
    qc_df: pd.DataFrame,
) -> None:
    core = _basin_center(landscape, "solution_core")
    _draw_l2_shell_arcs(ax, radii=[1.05, 2.45, 3.15])
    for basin in landscape.basins:
        target = basin.center
        if basin.name != "solution_core":
            arrow = FancyArrowPatch(
                posA=core,
                posB=target,
                arrowstyle="-|>",
                mutation_scale=15,
                lw=2.6,
                color=SCHEMATIC_COLORS["vmf"],
                alpha=0.88,
                connectionstyle="arc3,rad=-0.10",
                zorder=2,
            )
            ax.add_patch(arrow)
    _draw_found_disks(ax, landscape)

    # Show a sparse set of high-weight samples only as anchors, not as a cloud.
    weights = normalized_weights(importance_result.log_weights)
    if weights.size:
        take = np.argsort(weights)[-160:]
        pts = importance_result.samples[take]
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=SCHEMATIC_COLORS["vmf"], alpha=0.25, edgecolors="none", zorder=5)

    vmf_row = qc_df.set_index("method").loc["vmf_l2_final"]
    covered = int(vmf_row["covered_important_regions"])
    total = int(vmf_row["important_regions"])
    l1_error = float(vmf_row["region_l1_error"])
    _badge(
        ax,
        0.50,
        0.93,
        f"vMF+L2  {covered}/{total}",
        face="#DFF3E5",
        edge=SCHEMATIC_COLORS["vmf"],
        text_color="#14532D",
    )
    _coverage_strip(ax, [True, True, True, True], SCHEMATIC_COLORS["vmf"], y=0.82)
    _badge(
        ax,
        0.50,
        0.72,
        f"mass error {l1_error:.3f}",
        face="#EFFAF2",
        edge="#75C58A",
        text_color="#14532D",
    )
    _panel_note(ax, "directional proposal + L2 weighting")


def _draw_actual_local_trajectories(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    baseline_results: dict[str, SampleResult],
    qc_df: pd.DataFrame,
) -> None:
    coverage = _coverage_range(qc_df, ["random_walk_mcmc", "hmc", "pseudo_langevin"])
    _badge(
        ax,
        0.50,
        0.93,
        f"observed hits {coverage}/4",
        face="#F7DFDF",
        edge=SCHEMATIC_COLORS["miss"],
        text_color="#7F1D1D",
    )
    specs = [
        ("random_walk_mcmc", "MCMC", METHOD_COLORS["random_walk_mcmc"]),
        ("hmc", "HMC", METHOD_COLORS["hmc"]),
        ("pseudo_langevin", "pL", METHOD_COLORS["pseudo_langevin"]),
    ]
    for method, label, color in specs:
        path = _coarse_path(baseline_results[method].samples, max_points=110)
        if path.shape[0] < 2:
            continue
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.8, alpha=0.72, zorder=7)
        ax.scatter(path[:1, 0], path[:1, 1], s=24, color=color, alpha=0.95, zorder=8)
        end = path[-1]
        ax.plot([end[0] - 0.07, end[0] + 0.07], [end[1] - 0.07, end[1] + 0.07], color=color, lw=2.0, zorder=8)
        ax.plot([end[0] - 0.07, end[0] + 0.07], [end[1] + 0.07, end[1] - 0.07], color=color, lw=2.0, zorder=8)
    _mini_method_key(ax, specs, anchor=(0.06, 0.76))
    _panel_note(ax, "coarse retained-sample paths")


def _draw_final_retained_trajectory_panel(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    baseline_results: dict[str, SampleResult],
    importance_result: ImportanceResult,
    qc_df: pd.DataFrame,
) -> None:
    specs = [
        ("random_walk_mcmc", "MCMC", METHOD_COLORS["random_walk_mcmc"]),
        ("hmc", "HMC", METHOD_COLORS["hmc"]),
        ("pseudo_langevin", "pL", METHOD_COLORS["pseudo_langevin"]),
    ]
    for method, _, color in specs:
        path = _coarse_path(baseline_results[method].samples, max_points=95)
        if path.shape[0] >= 2:
            ax.plot(path[:, 0], path[:, 1], color=color, lw=1.45, alpha=0.55, zorder=8)
            ax.scatter(path[:1, 0], path[:1, 1], s=20, color=color, alpha=0.8, zorder=9)
            end = path[-1]
            ax.plot([end[0] - 0.06, end[0] + 0.06], [end[1] - 0.06, end[1] + 0.06], color=color, lw=1.8, alpha=0.75, zorder=9)
            ax.plot([end[0] - 0.06, end[0] + 0.06], [end[1] + 0.06, end[1] - 0.06], color=color, lw=1.8, alpha=0.75, zorder=9)
    _draw_vmf_region_footprint(ax, landscape, importance_result, max_per_region=420, alpha=0.20, zorder=10)
    _draw_vmf_anchor_rays(ax, landscape, importance_result, max_rays=36, alpha=0.13, zorder=7)
    _draw_found_disks(ax, landscape)
    qc = qc_df.set_index("method")
    covered = int(qc.loc["vmf_l2_final", "covered_important_regions"])
    total = int(qc.loc["vmf_l2_final", "important_regions"])
    _badge(
        ax,
        0.50,
        0.93,
        f"vMF+L2 weighted hits {covered}/{total}",
        face="#DFF3E5",
        edge=SCHEMATIC_COLORS["vmf"],
        text_color="#14532D",
    )
    _mini_method_key(ax, specs, anchor=(0.055, 0.13))


def _draw_actual_vmf_process(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    importance_result: ImportanceResult,
    qc_df: pd.DataFrame,
) -> None:
    _draw_l2_shell_arcs(ax, radii=[1.05, 2.45, 3.15])
    weights = normalized_weights(importance_result.log_weights)
    if weights.size:
        _draw_vmf_region_footprint(ax, landscape, importance_result, max_per_region=260, alpha=0.18, zorder=5)
        _draw_vmf_anchor_rays(ax, landscape, importance_result, max_rays=40, alpha=0.13, zorder=2)
    _draw_found_disks(ax, landscape)

    vmf_row = qc_df.set_index("method").loc["vmf_l2_final"]
    covered = int(vmf_row["covered_important_regions"])
    total = int(vmf_row["important_regions"])
    _badge(
        ax,
        0.50,
        0.93,
        f"weighted hits {covered}/{total}",
        face="#DFF3E5",
        edge=SCHEMATIC_COLORS["vmf"],
        text_color="#14532D",
    )
    _coverage_strip(ax, [True, True, True, True], SCHEMATIC_COLORS["vmf"], y=0.82)
    _panel_note(ax, "high-weight samples form all halos")


def _draw_vmf_region_footprint(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    importance_result: ImportanceResult,
    max_per_region: int,
    alpha: float,
    zorder: int,
) -> None:
    masks = landscape.region_mask(importance_result.samples)
    rng = np.random.default_rng(20260622)
    for idx, _ in enumerate(landscape.basins):
        region_idx = np.flatnonzero(masks[:, idx])
        if not region_idx.size:
            continue
        take = rng.choice(region_idx, size=min(max_per_region, region_idx.size), replace=False)
        pts = importance_result.samples[take]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=10,
            color=SCHEMATIC_COLORS["vmf"],
            alpha=alpha,
            edgecolors="none",
            zorder=zorder,
        )


def _draw_vmf_anchor_rays(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    importance_result: ImportanceResult,
    max_rays: int,
    alpha: float,
    zorder: int,
) -> None:
    weights = normalized_weights(importance_result.log_weights)
    if weights.size == 0:
        return
    ranked = np.argsort(weights)
    rays = importance_result.samples[ranked[-max(1, max_rays * 3) :]]
    origin = _basin_center(landscape, "solution_core")
    stride = max(1, rays.shape[0] // max_rays)
    for pt in rays[::stride][:max_rays]:
        ax.plot(
            [origin[0], pt[0]],
            [origin[1], pt[1]],
            color=SCHEMATIC_COLORS["vmf"],
            lw=0.9,
            alpha=alpha,
            zorder=zorder,
        )


def _draw_coverage_accumulation(
    ax: plt.Axes,
    landscape: ProxyLandscape,
    baseline_results: dict[str, SampleResult],
    importance_result: ImportanceResult,
    qc_df: pd.DataFrame,
) -> None:
    specs = [
        ("random_walk_mcmc", "MCMC", METHOD_COLORS["random_walk_mcmc"], baseline_results["random_walk_mcmc"].samples, -0.06),
        ("hmc", "HMC", METHOD_COLORS["hmc"], baseline_results["hmc"].samples, 0.06),
        ("pseudo_langevin", "pL", METHOD_COLORS["pseudo_langevin"], baseline_results["pseudo_langevin"].samples, 0.00),
        ("vmf_l2_final", "vMF+L2", SCHEMATIC_COLORS["vmf"], importance_result.samples, 0.00),
    ]
    for _, label, color, samples, yoff in specs:
        x, y, hit_labels = _first_hit_step_curve(landscape, samples)
        ax.step(x, y + yoff, where="post", lw=2.8 if label == "vMF+L2" else 1.8, color=color, alpha=0.95)
        _draw_endpoint_label(ax, x[-1], y[-1] + yoff, label, color)
        if label == "vMF+L2":
            for hx, hy, region in hit_labels:
                ax.scatter([hx], [hy], s=42, color=color, edgecolors="white", linewidths=0.9, zorder=5)
                ax.text(hx, hy + 0.18, region, color=color, fontsize=8.5, ha="center", va="bottom", weight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 4.25)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=8, length=0)
    ax.grid(axis="y", color="#D9D3C7", lw=0.8, alpha=0.75)
    ax.set_xlabel("first-hit order", fontsize=9)
    ax.set_ylabel("regions reached", fontsize=9)
    qc = qc_df.set_index("method")
    covered = int(qc.loc["vmf_l2_final", "covered_important_regions"])
    _badge(
        ax,
        0.50,
        0.92,
        f"vMF+L2 reaches R1-R4 ({covered}/4)",
        face="#DFF3E5",
        edge=SCHEMATIC_COLORS["vmf"],
        text_color="#14532D",
    )


def _coverage_range(qc_df: pd.DataFrame, methods: list[str]) -> str:
    qc = qc_df.set_index("method")
    vals = [int(qc.loc[m, "covered_important_regions"]) for m in methods if m in qc.index]
    if not vals:
        return "?-?"
    low = min(vals)
    high = max(vals)
    return str(low) if low == high else f"{low}-{high}"


def _draw_found_disks(ax: plt.Axes, landscape: ProxyLandscape) -> None:
    offsets = {
        "solution_core": np.asarray([-0.42, 0.38], dtype=np.float64),
        "near_same_valley": np.asarray([0.48, 0.30], dtype=np.float64),
        "across_barrier": np.asarray([0.42, 0.30], dtype=np.float64),
        "remote_needle": np.asarray([-0.35, 0.30], dtype=np.float64),
    }
    for basin in landscape.basins:
        offset = offsets.get(basin.name, np.asarray([0.28, 0.25], dtype=np.float64))
        pos = basin.center + offset
        ax.scatter(
            [pos[0]],
            [pos[1]],
            s=116,
            marker="o",
            facecolor=SCHEMATIC_COLORS["vmf"],
            edgecolor="white",
            linewidth=1.2,
            zorder=9,
        )


def _coverage_strip(ax: plt.Axes, found: list[bool], color: str, y: float) -> None:
    xs = np.linspace(0.39, 0.61, 4)
    for idx, x in enumerate(xs):
        face = color if found[idx] else "#FFFFFF"
        edge = color if found[idx] else "#999999"
        text_color = "white" if found[idx] else "#555555"
        ax.scatter(
            [x],
            [y],
            transform=ax.transAxes,
            s=118,
            marker="s",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=13,
            clip_on=False,
        )
        ax.text(
            x,
            y,
            f"R{idx + 1}",
            transform=ax.transAxes,
            fontsize=6.6,
            color=text_color,
            ha="center",
            va="center",
            weight="bold",
            zorder=14,
            clip_on=False,
        )


def _draw_endpoint_label(ax: plt.Axes, x: float, y: float, label: str, color: str) -> None:
    ax.text(
        min(0.985, x + 0.012),
        y,
        label,
        fontsize=8.2,
        color=color,
        ha="left",
        va="center",
        weight="bold" if label == "vMF+L2" else "normal",
        clip_on=False,
    )


def _first_hit_step_curve(
    landscape: ProxyLandscape,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, str]]]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]), []
    masks = landscape.region_mask(samples).astype(bool)
    events: list[tuple[int, int]] = []
    for region_idx in range(masks.shape[1]):
        hits = np.flatnonzero(masks[:, region_idx])
        if hits.size:
            events.append((int(hits[0]), region_idx))
    events.sort(key=lambda item: item[0])
    if not events:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]), []
    xs = np.linspace(0.08, 0.72, len(events))
    x = np.concatenate([[0.0], xs, [1.0]])
    y = np.concatenate([[0.0], np.arange(1, len(events) + 1, dtype=np.float64), [float(len(events))]])
    hit_labels = [
        (float(xs[idx]), float(idx + 1), REGION_CODES.get(landscape.basins[region_idx].name, f"R{region_idx + 1}"))
        for idx, (_, region_idx) in enumerate(events)
    ]
    return x, y, hit_labels


def _panel_note(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.50,
        0.045,
        text,
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="center",
        color="#4A4A4A",
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.05",
            "facecolor": SCHEMATIC_COLORS["panel_bg"],
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=12,
    )


def _region_key(ax: plt.Axes) -> None:
    ax.text(
        0.50,
        0.91,
        REGION_KEY,
        transform=ax.transAxes,
        fontsize=8.6,
        ha="center",
        va="center",
        color="#303030",
        bbox={
            "boxstyle": "round,pad=0.24,rounding_size=0.05",
            "facecolor": "#FFFFFF",
            "edgecolor": "#D7D1C4",
            "linewidth": 0.8,
            "alpha": 0.95,
        },
        zorder=12,
    )


def _setup_process_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SCHEMATIC_COLORS["panel_bg"])
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _coarse_path(samples: np.ndarray, max_points: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if samples.shape[0] > max_points:
        idx = np.linspace(0, samples.shape[0] - 1, max_points).astype(int)
        samples = samples[idx]
    if samples.shape[0] >= 7:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        x = np.convolve(samples[:, 0], kernel, mode="same")
        y = np.convolve(samples[:, 1], kernel, mode="same")
        samples = np.column_stack([x, y])
    return samples


def _mini_method_key(
    ax: plt.Axes,
    specs: list[tuple[str, str, str]],
    anchor: tuple[float, float],
) -> None:
    x0, y0 = anchor
    for idx, (_, label, color) in enumerate(specs):
        y = y0 + 0.055 * idx
        ax.plot([x0, x0 + 0.055], [y, y], transform=ax.transAxes, color=color, lw=2.0, clip_on=False, zorder=12)
        ax.text(
            x0 + 0.065,
            y,
            label,
            transform=ax.transAxes,
            fontsize=8.2,
            ha="left",
            va="center",
            color=color,
            weight="bold",
            zorder=12,
        )


def _coverage_curve(landscape: ProxyLandscape, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0])
    masks = landscape.region_mask(samples).astype(bool)
    cumulative = np.maximum.accumulate(masks.astype(np.int32), axis=0)
    reached = np.sum(cumulative, axis=1).astype(float)
    if reached.size > 260:
        idx = np.unique(np.linspace(0, reached.size - 1, 260).astype(int))
        reached = reached[idx]
        x = idx.astype(np.float64) / max(1, samples.shape[0] - 1)
    else:
        x = np.arange(reached.size, dtype=np.float64) / max(1, reached.size - 1)
    return x, reached


def _draw_schematic_trace(
    ax: plt.Axes,
    pts: np.ndarray,
    color: str,
    label: str,
    label_offset: tuple[float, float],
) -> None:
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.7, alpha=0.95, zorder=6)
    end = pts[-1]
    ax.plot([end[0] - 0.08, end[0] + 0.08], [end[1] - 0.08, end[1] + 0.08], color=color, lw=2.3, zorder=7)
    ax.plot([end[0] - 0.08, end[0] + 0.08], [end[1] + 0.08, end[1] - 0.08], color=color, lw=2.3, zorder=7)
    if label:
        ax.text(
            end[0] + label_offset[0],
            end[1] + label_offset[1],
            label,
            fontsize=8.5,
            color=color,
            weight="bold",
            zorder=8,
        )


def _draw_l2_shell_arcs(ax: plt.Axes, radii: list[float]) -> None:
    for radius in radii:
        arc = Arc(
            xy=(0.0, 0.0),
            width=2.0 * radius,
            height=2.0 * radius,
            theta1=205,
            theta2=55,
            color="#8CCFA0",
            lw=1.4,
            alpha=0.38,
            linestyle=(0, (3, 4)),
            zorder=1,
        )
        ax.add_patch(arc)


def _badge(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    face: str,
    edge: str,
    text_color: str,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=text_color,
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.08",
            "facecolor": face,
            "edgecolor": edge,
            "linewidth": 1.5,
        },
        zorder=10,
    )


def _basin_center(landscape: ProxyLandscape, name: str) -> np.ndarray:
    for basin in landscape.basins:
        if basin.name == name:
            return basin.center
    raise KeyError(name)


def _short_region_label(name: str) -> str:
    labels = {
        "solution_core": "solution\ncore",
        "near_same_valley": "same\nvalley",
        "across_barrier": "across\nbarrier",
        "remote_needle": "remote\nneedle",
    }
    return labels.get(name, name.replace("_", "\n"))
