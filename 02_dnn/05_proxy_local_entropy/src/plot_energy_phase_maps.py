from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from io_utils import ensure_dir, read_csv, repo_relative


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _metric_matrix(rows: Sequence[dict[str, Any]], value_key: str) -> tuple[np.ndarray, list[float], list[float]]:
    betas = sorted({round(float(row["beta"]), 8) for row in rows})
    radii = sorted({round(float(row["radius"]), 8) for row in rows})
    beta_index = {value: index for index, value in enumerate(betas)}
    radius_index = {value: index for index, value in enumerate(radii)}
    matrix = np.full((len(betas), len(radii)), np.nan, dtype=np.float64)
    for row in rows:
        beta = round(float(row["beta"]), 8)
        radius = round(float(row["radius"]), 8)
        value = _finite_float(row.get(value_key))
        if np.isfinite(value):
            matrix[beta_index[beta], radius_index[radius]] = value
    return matrix, betas, radii


def _matrix_on_grid(rows: Sequence[dict[str, Any]], value_key: str, betas: Sequence[float], radii: Sequence[float]) -> np.ndarray:
    beta_index = {round(value, 8): index for index, value in enumerate(betas)}
    radius_index = {round(value, 8): index for index, value in enumerate(radii)}
    matrix = np.full((len(betas), len(radii)), np.nan, dtype=np.float64)
    for row in rows:
        try:
            beta = round(float(row["beta"]), 8)
            radius = round(float(row["radius"]), 8)
        except (KeyError, TypeError, ValueError):
            continue
        if beta not in beta_index or radius not in radius_index:
            continue
        value = _finite_float(row.get(value_key))
        if np.isfinite(value):
            matrix[beta_index[beta], radius_index[radius]] = value
    return matrix


def _linear_edges(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(sorted(values), dtype=np.float64)
    if arr.size == 1:
        width = max(abs(arr[0]) * 0.1, 0.5)
        return np.asarray([arr[0] - width, arr[0] + width], dtype=np.float64)
    mids = (arr[:-1] + arr[1:]) / 2.0
    first = arr[0] - (mids[0] - arr[0])
    last = arr[-1] + (arr[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _log_edges(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(sorted(values), dtype=np.float64)
    if np.any(arr <= 0.0):
        raise ValueError("log-radius plots require positive radius values")
    if arr.size == 1:
        factor = 1.1
        return np.asarray([arr[0] / factor, arr[0] * factor], dtype=np.float64)
    mids = np.sqrt(arr[:-1] * arr[1:])
    first = arr[0] * arr[0] / mids[0]
    last = arr[-1] * arr[-1] / mids[-1]
    return np.concatenate([[first], mids, [last]])


def _tick_values(radii: Sequence[float], *, log_radius: bool) -> list[float]:
    lo = min(radii)
    hi = max(radii)
    candidates = (
        (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 2.50)
        if log_radius
        else (0.01, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00, 2.50)
    )
    return [value for value in candidates if lo <= value <= hi]


def _format_tick(value: float) -> str:
    if value < 0.1:
        return f"{value:.2f}"
    if value < 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:g}"


def _norm_for_matrix(matrix: np.ndarray, *, zero_reference: bool) -> Normalize | TwoSlopeNorm:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    if zero_reference:
        vmin = float(np.nanpercentile(finite, 2.0))
        vmax = float(np.nanpercentile(finite, 98.0))
        if float(np.nanmin(finite)) < 0.0 < float(np.nanmax(finite)):
            return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        if float(np.nanmax(finite)) <= 0.0:
            vmax = 0.0
        elif float(np.nanmin(finite)) >= 0.0:
            vmin = 0.0
    else:
        vmin = float(np.nanpercentile(finite, 2.0))
        vmax = float(np.nanpercentile(finite, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1.0e-12:
        center = float(np.nanmean(finite)) if finite.size else 0.0
        pad = max(abs(center) * 0.05, 1.0e-6)
        vmin, vmax = center - pad, center + pad
    return Normalize(vmin=vmin, vmax=vmax)


def _add_zero_contour(ax: plt.Axes, matrix: np.ndarray, betas: Sequence[float], radii: Sequence[float]) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0 or not (float(np.nanmin(finite)) < 0.0 < float(np.nanmax(finite))):
        return
    contour = ax.contour(
        np.asarray(radii, dtype=np.float64),
        np.asarray(betas, dtype=np.float64),
        matrix,
        levels=[0.0],
        colors="black",
        linewidths=1.0,
    )
    ax.clabel(contour, fmt={0.0: "0"}, inline=True, fontsize=8)


def _beta_groups(betas: Sequence[float]) -> list[tuple[str, str, list[float]]]:
    values = list(betas)
    midpoint = len(values) // 2
    return [
        ("all_beta", "all beta", values),
        ("low_beta", "low beta", values[:midpoint]),
        ("high_beta", "high beta", values[midpoint:]),
    ]


def _slice_beta_matrix(matrix: np.ndarray, betas: Sequence[float], selected_betas: Sequence[float]) -> tuple[np.ndarray, list[float]]:
    beta_index = {round(value, 8): index for index, value in enumerate(betas)}
    indices = [beta_index[round(value, 8)] for value in selected_betas]
    return matrix[indices, :], [float(value) for value in selected_betas]


def _y_limits(matrix: np.ndarray, *, include_zero: bool) -> tuple[float, float]:
    return _y_limits_from_matrices([matrix], include_zero=include_zero)


def _y_limits_from_matrices(matrices: Sequence[np.ndarray | None], *, include_zero: bool) -> tuple[float, float]:
    finite_parts = [matrix[np.isfinite(matrix)] for matrix in matrices if matrix is not None and np.any(np.isfinite(matrix))]
    finite = np.concatenate(finite_parts) if finite_parts else np.asarray([], dtype=np.float64)
    if finite.size == 0:
        return (-1.0, 1.0)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    pad = max((hi - lo) * 0.05, max(abs(lo), abs(hi), 1.0) * 1.0e-4)
    return lo - pad, hi + pad


def _scale_label(log_radius: bool) -> str:
    return "log radius" if log_radius else "linear radius"


def _phase_output_path(figures_dir: Path, slug: str, group_slug: str, scale_slug: str) -> Path:
    if group_slug == "all_beta":
        return figures_dir / f"{slug}_phase_heatmap_{scale_slug}.png"
    return figures_dir / f"{group_slug}_{slug}_phase_heatmap_{scale_slug}.png"


def _plot_phase_heatmap(
    matrix: np.ndarray,
    betas: Sequence[float],
    radii: Sequence[float],
    path: Path,
    *,
    title: str,
    cbar_label: str,
    cmap: str,
    norm: Normalize | TwoSlopeNorm,
    log_radius: bool,
    baseline_radius: float | None = None,
) -> Path:
    ensure_dir(path.parent)
    x_edges = _log_edges(radii) if log_radius else _linear_edges(radii)
    y_edges = _linear_edges(betas)
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="none",
    )
    if log_radius:
        ax.set_xscale("log")
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    if baseline_radius is not None:
        ax.axvline(baseline_radius, color="black", linewidth=0.8, alpha=0.55)
    _add_zero_contour(ax, matrix, betas, radii)
    ax.set_xlabel(r"distance threshold $d$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title)
    ax.set_xticks(_tick_values(radii, log_radius=log_radius))
    ax.set_xticklabels([_format_tick(value) for value in _tick_values(radii, log_radius=log_radius)], rotation=35, ha="right")
    ax.set_yticks(list(betas))
    ax.set_yticklabels([f"{value:g}" for value in betas])
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def _plot_line_plot(
    matrix: np.ndarray,
    betas: Sequence[float],
    radii: Sequence[float],
    path: Path,
    *,
    ci_low_matrix: np.ndarray | None = None,
    ci_high_matrix: np.ndarray | None = None,
    title: str,
    ylabel: str,
    log_radius: bool,
    y_limits: tuple[float, float],
    baseline_radius: float | None = None,
    zero_reference: bool = False,
) -> Path:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, beta in enumerate(betas):
        y = np.asarray(matrix[index, :], dtype=np.float64)
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        color = cmap(index / max(len(betas) - 1, 1))
        if ci_low_matrix is not None and ci_high_matrix is not None:
            low = np.asarray(ci_low_matrix[index, :], dtype=np.float64)
            high = np.asarray(ci_high_matrix[index, :], dtype=np.float64)
            ci_mask = mask & np.isfinite(low) & np.isfinite(high)
            if np.any(ci_mask):
                ax.fill_between(
                    np.asarray(radii, dtype=np.float64)[ci_mask],
                    low[ci_mask],
                    high[ci_mask],
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                )
        ax.plot(
            np.asarray(radii, dtype=np.float64)[mask],
            y[mask],
            linewidth=1.55,
            color=color,
            label=fr"$\beta={beta:g}$",
        )
    if log_radius:
        x_edges = _log_edges(radii)
        ax.set_xscale("log")
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    if baseline_radius is not None:
        ax.axvline(baseline_radius, color="black", linewidth=0.8, alpha=0.55)
    if zero_reference:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlabel(r"distance threshold $d$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(_tick_values(radii, log_radius=log_radius))
    ax.set_xticklabels([_format_tick(value) for value in _tick_values(radii, log_radius=log_radius)], rotation=35, ha="right")
    ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def _delta_from_min_radius(energy_matrix: np.ndarray, radii: Sequence[float]) -> tuple[np.ndarray, float]:
    baseline_radius = min(radii)
    radius_index = list(radii).index(baseline_radius)
    baseline = energy_matrix[:, [radius_index]]
    return energy_matrix - baseline, baseline_radius


def _delta_bounds_from_min_radius(
    delta_matrix: np.ndarray,
    ci_half_width_matrix: np.ndarray,
    radii: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    baseline_radius = min(radii)
    radius_index = list(radii).index(baseline_radius)
    baseline_half_width = ci_half_width_matrix[:, [radius_index]]
    half_width = np.sqrt(np.square(ci_half_width_matrix) + np.square(baseline_half_width))
    half_width[:, radius_index] = 0.0
    return delta_matrix - half_width, delta_matrix + half_width


def _stats_lines(name: str, matrix: np.ndarray) -> list[str]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return [f"## {name}", "", "No finite data.", ""]
    lines = [f"## {name}", ""]
    for label, value in (
        ("min", float(np.nanmin(finite))),
        ("p02", float(np.nanpercentile(finite, 2.0))),
        ("median", float(np.nanmedian(finite))),
        ("p98", float(np.nanpercentile(finite, 98.0))),
        ("max", float(np.nanmax(finite))),
    ):
        lines.append(f"- {label}: `{value:.12g}`")
    lines.extend(
        [
            f"- negative cells: `{int(np.sum(finite < 0.0))}`",
            f"- positive cells: `{int(np.sum(finite > 0.0))}`",
            f"- zero cells: `{int(np.sum(finite == 0.0))}`",
            "",
        ]
    )
    return lines


def plot_energy_phase_maps(summary_root: Path, figures_dir: Path) -> list[Path]:
    abs_rows = read_csv(summary_root / "absolute_phi_by_beta_radius.csv")
    dphi_rows = read_csv(summary_root / "dphi_dr_by_beta_radius.csv")

    energy_matrix, betas, radii = _metric_matrix(abs_rows, "phi_energy")
    denergy_dr_matrix, denergy_betas, denergy_radii = _metric_matrix(dphi_rows, "dphi_energy_dr")
    if betas != denergy_betas or radii != denergy_radii:
        raise ValueError("energy and dE/dd tables do not share the same beta/radius grid")
    delta_energy_matrix, baseline_radius = _delta_from_min_radius(energy_matrix, radii)
    energy_ci_low = _matrix_on_grid(abs_rows, "phi_energy_ci95_low", betas, radii)
    energy_ci_high = _matrix_on_grid(abs_rows, "phi_energy_ci95_high", betas, radii)
    energy_ci_half_width = _matrix_on_grid(abs_rows, "phi_energy_ci95_half_width", betas, radii)
    delta_energy_ci_low, delta_energy_ci_high = _delta_bounds_from_min_radius(delta_energy_matrix, energy_ci_half_width, radii)
    denergy_dr_ci_low = _matrix_on_grid(dphi_rows, "dphi_energy_dr_ci95_low", betas, radii)
    denergy_dr_ci_high = _matrix_on_grid(dphi_rows, "dphi_energy_dr_ci95_high", betas, radii)
    energy_dataset_std = _matrix_on_grid(abs_rows, "phi_energy_dataset_mean_std", betas, radii)
    energy_dataset_std_low = energy_matrix - energy_dataset_std
    energy_dataset_std_high = energy_matrix + energy_dataset_std
    delta_energy_dataset_std_low, delta_energy_dataset_std_high = _delta_bounds_from_min_radius(
        delta_energy_matrix,
        energy_dataset_std,
        radii,
    )
    denergy_dr_dataset_std = _matrix_on_grid(dphi_rows, "dphi_energy_dr_dataset_mean_std", betas, radii)
    denergy_dr_dataset_std_low = denergy_dr_matrix - denergy_dr_dataset_std
    denergy_dr_dataset_std_high = denergy_dr_matrix + denergy_dr_dataset_std
    energy_dataset_sem = _matrix_on_grid(abs_rows, "phi_energy_sem", betas, radii)
    energy_dataset_sem_low = energy_matrix - energy_dataset_sem
    energy_dataset_sem_high = energy_matrix + energy_dataset_sem
    delta_energy_dataset_sem_low, delta_energy_dataset_sem_high = _delta_bounds_from_min_radius(
        delta_energy_matrix,
        energy_dataset_sem,
        radii,
    )
    denergy_dr_dataset_sem = _matrix_on_grid(dphi_rows, "dphi_energy_dr_sem", betas, radii)
    denergy_dr_dataset_sem_low = denergy_dr_matrix - denergy_dr_dataset_sem
    denergy_dr_dataset_sem_high = denergy_dr_matrix + denergy_dr_dataset_sem

    specs = [
        (
            "delta_energy",
            delta_energy_matrix,
            delta_energy_ci_low,
            delta_energy_ci_high,
            delta_energy_dataset_std_low,
            delta_energy_dataset_std_high,
            delta_energy_dataset_sem_low,
            delta_energy_dataset_sem_high,
            r"$\Delta E_\beta(d;d_{\min})$ phase map",
            r"$\Delta E_\beta(d;d_{\min})$",
            "Blues_r",
            True,
            baseline_radius,
        ),
        (
            "energy",
            energy_matrix,
            energy_ci_low,
            energy_ci_high,
            energy_dataset_std_low,
            energy_dataset_std_high,
            energy_dataset_sem_low,
            energy_dataset_sem_high,
            r"$E_\beta(d)$ phase map",
            r"$E_\beta(d)$",
            "viridis",
            False,
            None,
        ),
        (
            "denergy_dr",
            denergy_dr_matrix,
            denergy_dr_ci_low,
            denergy_dr_ci_high,
            denergy_dr_dataset_std_low,
            denergy_dr_dataset_std_high,
            denergy_dr_dataset_sem_low,
            denergy_dr_dataset_sem_high,
            r"$dE_\beta/dd$ phase map",
            r"$dE_\beta/dd$",
            "viridis",
            False,
            None,
        ),
    ]

    outputs: list[Path] = []
    report_lines = [
        "# Energy phase heatmaps report",
        "",
        f"- summary root: `{repo_relative(summary_root)}`",
        f"- output root: `{repo_relative(figures_dir)}`",
        f"- beta count: `{len(betas)}`",
        f"- radius count: `{len(radii)}`",
        f"- missing grid cells: `{len(betas) * len(radii) - int(np.sum(np.isfinite(energy_matrix)))}`",
        f"- delta baseline radius: `{baseline_radius:g}`",
        "- log-radius plots use geometric cell edges and a log-scaled x-axis.",
        "- low/high split is sorted beta values divided 9/9.",
        "- phase maps use shared per-metric color scales so low/high panels remain comparable.",
        "- line plots use per-panel y-limits so low/high beta curves remain readable.",
        "- line plot shaded bands are 95% confidence intervals from summary-table CI columns.",
        "- delta-energy bands use propagated absolute-energy CI half widths relative to `d_min`; this is approximate because only summary tables are available.",
        "- dataset-SD line plots use mean +/- one standard deviation of the 30 dataset-level means.",
        "- delta-energy dataset-SD bands use propagated absolute-energy dataset SD relative to `d_min`; this is approximate because only summary tables are available.",
        "- dataset-SEM line plots use mean +/- one standard error of the 30 dataset-level means.",
        "- delta-energy dataset-SEM bands use propagated absolute-energy dataset SEM relative to `d_min`; this is approximate because only summary tables are available.",
        "",
    ]
    for group_slug, group_label, group_betas in _beta_groups(betas):
        report_lines.append(f"- {group_label}: `{', '.join(f'{value:g}' for value in group_betas)}`")
    report_lines.append("")

    for (
        slug,
        matrix,
        ci_low_matrix,
        ci_high_matrix,
        dataset_std_low_matrix,
        dataset_std_high_matrix,
        dataset_sem_low_matrix,
        dataset_sem_high_matrix,
        title,
        cbar_label,
        cmap,
        zero_reference,
        baseline,
    ) in specs:
        norm = _norm_for_matrix(matrix, zero_reference=zero_reference)
        for group_slug, group_label, group_betas in _beta_groups(betas):
            group_matrix, sliced_betas = _slice_beta_matrix(matrix, betas, group_betas)
            group_ci_low = _slice_beta_matrix(ci_low_matrix, betas, group_betas)[0]
            group_ci_high = _slice_beta_matrix(ci_high_matrix, betas, group_betas)[0]
            group_dataset_std_low = _slice_beta_matrix(dataset_std_low_matrix, betas, group_betas)[0]
            group_dataset_std_high = _slice_beta_matrix(dataset_std_high_matrix, betas, group_betas)[0]
            group_dataset_sem_low = _slice_beta_matrix(dataset_sem_low_matrix, betas, group_betas)[0]
            group_dataset_sem_high = _slice_beta_matrix(dataset_sem_high_matrix, betas, group_betas)[0]
            line_y_limits = _y_limits_from_matrices([group_matrix, group_ci_low, group_ci_high], include_zero=zero_reference)
            dataset_std_y_limits = _y_limits_from_matrices(
                [group_matrix, group_dataset_std_low, group_dataset_std_high],
                include_zero=zero_reference,
            )
            dataset_sem_y_limits = _y_limits_from_matrices(
                [group_matrix, group_dataset_sem_low, group_dataset_sem_high],
                include_zero=zero_reference,
            )
            for scale_slug, log_radius in (("linear", False), ("log_radius", True)):
                outputs.append(
                    _plot_phase_heatmap(
                        group_matrix,
                        sliced_betas,
                        radii,
                        _phase_output_path(figures_dir, slug, group_slug, scale_slug),
                        title=f"{title}, {group_label}, {_scale_label(log_radius)}",
                        cbar_label=cbar_label,
                        cmap=cmap,
                        norm=norm,
                        log_radius=log_radius,
                        baseline_radius=baseline,
                    )
                )
                outputs.append(
                    _plot_line_plot(
                        group_matrix,
                        sliced_betas,
                        radii,
                        figures_dir / f"{group_slug}_{slug}_line_{scale_slug}.png",
                        ci_low_matrix=group_ci_low,
                        ci_high_matrix=group_ci_high,
                        title=f"{cbar_label} by distance, {group_label}, {_scale_label(log_radius)}",
                        ylabel=cbar_label,
                        log_radius=log_radius,
                        y_limits=line_y_limits,
                        baseline_radius=baseline,
                        zero_reference=zero_reference,
                    )
                )
                outputs.append(
                    _plot_line_plot(
                        group_matrix,
                        sliced_betas,
                        radii,
                        figures_dir / f"{group_slug}_{slug}_dataset_std_line_{scale_slug}.png",
                        ci_low_matrix=group_dataset_std_low,
                        ci_high_matrix=group_dataset_std_high,
                        title=f"{cbar_label} by distance, {group_label}, dataset SD, {_scale_label(log_radius)}",
                        ylabel=cbar_label,
                        log_radius=log_radius,
                        y_limits=dataset_std_y_limits,
                        baseline_radius=baseline,
                        zero_reference=zero_reference,
                    )
                )
                outputs.append(
                    _plot_line_plot(
                        group_matrix,
                        sliced_betas,
                        radii,
                        figures_dir / f"{group_slug}_{slug}_dataset_sem_line_{scale_slug}.png",
                        ci_low_matrix=group_dataset_sem_low,
                        ci_high_matrix=group_dataset_sem_high,
                        title=f"{cbar_label} by distance, {group_label}, dataset SEM, {_scale_label(log_radius)}",
                        ylabel=cbar_label,
                        log_radius=log_radius,
                        y_limits=dataset_sem_y_limits,
                        baseline_radius=baseline,
                        zero_reference=zero_reference,
                    )
                )
        report_lines.extend(_stats_lines(slug, matrix))

    report_path = figures_dir / "energy_phase_heatmaps_report.md"
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    outputs.append(report_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot energy phase heatmaps from 05 proxy summary tables.")
    parser.add_argument("--summary-root", type=Path, required=True, help="Directory containing proxy summary CSV tables.")
    parser.add_argument("--figures-dir", type=Path, required=True, help="Output figure directory.")
    args = parser.parse_args()

    outputs = plot_energy_phase_maps(args.summary_root, args.figures_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
