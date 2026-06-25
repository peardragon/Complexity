from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from compute_phi import DEFAULT_LAMBDA_REG, P, enrich_unit_rows_for_full_phi
from io_utils import REPO_ROOT, ensure_dir, save_csv
from plot_summary import (
    DEFAULT_Q_VALUES,
    _clear_png_outputs,
    _global_center_value,
    _grid_edges,
    _line_groups,
    _logmeanexp,
    _q_label,
    _q_slug,
    _style_axes,
    hq_phase_rows,
    plot_delta_phi,
    plot_energy_term,
    plot_entropic_term,
    plot_hq_phase_maps,
    plot_phi,
    plot_reference_prior_correction,
)


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def absolute_phi_rows(
    unit_rows: Sequence[dict[str, Any]],
    *,
    param_count: int = P,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float], list[dict[str, float]]] = {}
    enriched_rows = enrich_unit_rows_for_full_phi([dict(row) for row in unit_rows], lambda_reg=lambda_reg, param_count=param_count)
    for row in enriched_rows:
        beta = round(float(row["beta"]), 8)
        radius = round(float(row["radius"]), 8)
        stripped = _finite_float(row.get("logZ_inf_stripped", row.get("logZ_inf")))
        full = _finite_float(row.get("logZ_inf_full"))
        correction = _finite_float(row.get("reference_prior_log_weight"))
        norm_sq = _finite_float(row.get("theta_ref_norm_sq"))
        dlogz_dr = _finite_float(row.get("dlogZ_inf_full_dr", row.get("dlogZ_inf_dr")))
        if not np.isfinite(stripped) or not np.isfinite(full) or radius <= 0:
            continue
        groups.setdefault((beta, radius), []).append(
            {
                "stripped": stripped,
                "full": full,
                "correction": correction,
                "norm_sq": norm_sq,
                "dlogz_dr": dlogz_dr,
            }
        )

    out: list[dict[str, Any]] = []
    for (beta, radius), values in sorted(groups.items()):
        stripped_arr = np.asarray([row["stripped"] for row in values], dtype=np.float64)
        full_arr = np.asarray([row["full"] for row in values], dtype=np.float64)
        correction_arr = np.asarray([row["correction"] for row in values], dtype=np.float64)
        norm_sq_arr = np.asarray([row["norm_sq"] for row in values], dtype=np.float64)
        dlogz_dr_arr = np.asarray([row["dlogz_dr"] for row in values], dtype=np.float64)
        finite_dlogz_dr = dlogz_dr_arr[np.isfinite(dlogz_dr_arr)]
        energy_stripped = float(np.mean(stripped_arr) / param_count)
        energy_full = float(np.mean(full_arr) / param_count)
        annealed_full_energy = float(_logmeanexp(full_arr) / param_count)
        annealed_stripped_energy = float(_logmeanexp(stripped_arr) / param_count)
        correction_per_p = float(np.mean(correction_arr[np.isfinite(correction_arr)]) / param_count) if np.any(np.isfinite(correction_arr)) else float("nan")
        mean_norm_sq = float(np.mean(norm_sq_arr[np.isfinite(norm_sq_arr)])) if np.any(np.isfinite(norm_sq_arr)) else float("nan")
        area = float((param_count - 1) / param_count * math.log(radius))
        dphi_energy_dr = float(np.mean(finite_dlogz_dr) / param_count) if finite_dlogz_dr.size else float("nan")
        dphi_entropic_dr = float((param_count - 1) / (param_count * radius)) if radius > 0 else float("nan")
        dphi_full_dr = dphi_entropic_dr + dphi_energy_dr if np.isfinite(dphi_entropic_dr) and np.isfinite(dphi_energy_dr) else float("nan")
        out.append(
            {
                "beta": beta,
                "radius": radius,
                "ref_count": int(full_arr.size),
                "phi_full": area + energy_full,
                "phi_energy": energy_full,
                "phi_full_quenched": area + energy_full,
                "phi_energy_full_quenched": energy_full,
                "phi_stripped_proxy": area + energy_stripped,
                "phi_energy_stripped_proxy": energy_stripped,
                "phi_full_annealed": area + annealed_full_energy,
                "phi_energy_full_annealed": annealed_full_energy,
                "phi_stripped_annealed": area + annealed_stripped_energy,
                "reference_prior_correction_per_P": correction_per_p,
                "theta_ref_norm_sq_mean": mean_norm_sq,
                "area_term": area,
                "derivative_ref_count": int(finite_dlogz_dr.size),
                "mean_dlogZ_inf_full_dr": float(np.mean(finite_dlogz_dr)) if finite_dlogz_dr.size else float("nan"),
                "sd_dlogZ_inf_full_dr": float(np.std(finite_dlogz_dr, ddof=1)) if finite_dlogz_dr.size > 1 else 0.0,
                "dphi_full_dr": dphi_full_dr,
                "dphi_energy_dr": dphi_energy_dr,
                "dphi_entropic_dr": dphi_entropic_dr,
                "claim": "pass",
            }
        )
    return out


def derivative_phi_rows(abs_phi_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "beta": row.get("beta"),
            "radius": row.get("radius"),
            "ref_count": row.get("derivative_ref_count", row.get("ref_count")),
            "dphi_full_dr": row.get("dphi_full_dr"),
            "dphi_energy_dr": row.get("dphi_energy_dr"),
            "dphi_entropic_dr": row.get("dphi_entropic_dr"),
            "mean_dlogZ_inf_full_dr": row.get("mean_dlogZ_inf_full_dr"),
            "sd_dlogZ_inf_full_dr": row.get("sd_dlogZ_inf_full_dr"),
            "claim": row.get("claim", ""),
        }
        for row in abs_phi_rows
    ]


def _resolve_samples_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _weighted_accuracy_cutoff(error: np.ndarray, logw: np.ndarray, q_value: float) -> tuple[float, float, float]:
    accuracy = 1.0 - np.asarray(error, dtype=np.float64)
    logw = np.asarray(logw, dtype=np.float64)
    mask = np.isfinite(accuracy) & np.isfinite(logw)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")
    accuracy = np.clip(accuracy[mask], 0.0, 1.0)
    logw = logw[mask]
    peak = float(np.max(logw))
    weights = np.exp(logw - peak)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan"), float("nan"), float("nan")
    weights = weights / weight_sum
    order = np.argsort(-accuracy, kind="mergesort")
    sorted_accuracy = accuracy[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, float(q_value) - 1.0e-12, side="left"))
    index = min(max(index, 0), sorted_accuracy.size - 1)
    cutoff = float(sorted_accuracy[index])
    accepted = accuracy >= cutoff - 1.0e-12
    weighted_ratio = float(np.sum(weights[accepted]))
    unweighted_ratio = float(np.mean(accepted))
    return cutoff, weighted_ratio, unweighted_ratio


def _claim_map_from_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple[float, float], str]:
    out: dict[tuple[float, float], str] = {}
    for row in rows:
        try:
            key = (round(float(row["beta"]), 8), round(float(row["radius"]), 8))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = str(row.get("claim", out.get(key, "")))
    return out


def _write_accuracy_status(progress_path: Path | None, payload: dict[str, Any]) -> None:
    if progress_path is None:
        return
    ensure_dir(progress_path.parent)
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def accuracy_q_phase_rows(
    unit_rows: Sequence[dict[str, Any]],
    claim_rows: Sequence[dict[str, Any]] = (),
    *,
    q_values: Sequence[float] = DEFAULT_Q_VALUES,
    progress_path: Path | None = None,
) -> list[dict[str, Any]]:
    claim_map = _claim_map_from_rows(claim_rows)
    grouped: dict[tuple[float, float, float], list[tuple[float, float, float]]] = {}
    started = time.time()
    total = len(unit_rows)
    status_every = max(1, total // 100)
    _write_accuracy_status(
        progress_path,
        {
            "event": "started",
            "completed": 0,
            "total": total,
            "elapsed_seconds": 0.0,
        },
    )
    for index, row in enumerate(unit_rows, start=1):
        try:
            beta = round(float(row["beta"]), 8)
            radius = round(float(row["radius"]), 8)
            samples_path = _resolve_samples_path(row["samples_path"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            with np.load(samples_path) as samples:
                error = np.asarray(samples["error"], dtype=np.float64)
                logw = np.asarray(samples["logw_ce"], dtype=np.float64)
        except (OSError, KeyError, ValueError):
            continue
        for q_value in q_values:
            cutoff, weighted_ratio, unweighted_ratio = _weighted_accuracy_cutoff(error, logw, float(q_value))
            grouped.setdefault((beta, radius, float(q_value)), []).append((cutoff, weighted_ratio, unweighted_ratio))
        if progress_path is not None and (index == total or index % status_every == 0):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed > 0.0 else 0.0
            eta = (total - index) / rate if rate > 0.0 else float("nan")
            _write_accuracy_status(
                progress_path,
                {
                    "event": "unit_completed",
                    "completed": index,
                    "total": total,
                    "percent": 100.0 * index / max(1, total),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "rate_units_per_s": rate,
                    "last_event": {
                        "beta": beta,
                        "radius": radius,
                        "samples_path": str(samples_path),
                    },
                },
            )

    out: list[dict[str, Any]] = []
    for (beta, radius, q_value), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        finite_cutoff = arr[np.isfinite(arr[:, 0]), 0]
        finite_weighted = arr[np.isfinite(arr[:, 1]), 1]
        finite_unweighted = arr[np.isfinite(arr[:, 2]), 2]
        mean_accuracy = float(np.mean(finite_cutoff)) if finite_cutoff.size else float("nan")
        out.append(
            {
                "q": _q_label(q_value),
                "beta": beta,
                "radius": radius,
                "accuracy_q": mean_accuracy,
                "accuracy_q_percent": 100.0 * mean_accuracy if np.isfinite(mean_accuracy) else float("nan"),
                "error_q": 1.0 - mean_accuracy if np.isfinite(mean_accuracy) else float("nan"),
                "error_q_percent": 100.0 * (1.0 - mean_accuracy) if np.isfinite(mean_accuracy) else float("nan"),
                "mean_weighted_solution_ratio_at_accuracy_q": float(np.mean(finite_weighted)) if finite_weighted.size else float("nan"),
                "mean_unweighted_solution_ratio_at_accuracy_q": float(np.mean(finite_unweighted)) if finite_unweighted.size else float("nan"),
                "ref_count": int(arr.shape[0]),
                "claim": claim_map.get((beta, radius), ""),
            }
        )
    _write_accuracy_status(
        progress_path,
        {
            "event": "completed",
            "completed": total,
            "total": total,
            "percent": 100.0,
            "elapsed_seconds": time.time() - started,
            "row_count": len(out),
        },
    )
    return out


def _accuracy_phase_matrix(rows_for_q: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[float], list[float]]:
    betas = sorted({round(float(row["beta"]), 8) for row in rows_for_q})
    radii = sorted({round(float(row["radius"]), 8) for row in rows_for_q})
    beta_index = {value: index for index, value in enumerate(betas)}
    radius_index = {value: index for index, value in enumerate(radii)}
    matrix = np.full((len(betas), len(radii)), np.nan, dtype=np.float64)
    for row in rows_for_q:
        beta = round(float(row["beta"]), 8)
        radius = round(float(row["radius"]), 8)
        value = _finite_float(row.get("accuracy_q_percent"))
        if np.isfinite(value):
            matrix[beta_index[beta], radius_index[radius]] = value
    return matrix, betas, radii


def _draw_accuracy_phase_panel(ax: plt.Axes, rows: Sequence[dict[str, Any]], *, q_value: float) -> Any:
    matrix, betas, radii = _accuracy_phase_matrix(rows)
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        vmin = float(np.percentile(finite, 2.0))
        vmax = float(np.percentile(finite, 98.0))
        if abs(vmax - vmin) < 1.0e-9:
            vmin = max(0.0, float(np.min(finite)) - 0.5)
            vmax = min(100.0, float(np.max(finite)) + 0.5)
    else:
        vmin, vmax = 0.0, 100.0
    mesh = ax.pcolormesh(
        _grid_edges(radii),
        _grid_edges(betas),
        np.ma.masked_invalid(matrix),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors="white",
        linewidth=0.12,
    )
    ax.set_title(fr"$A_q(\beta,d)$, $q={_q_label(q_value)}$")
    ax.set_xlabel(r"distance threshold $d$")
    ax.set_xticks(radii[:: max(1, len(radii) // 10)])
    ax.set_yticks(betas)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(False)
    return mesh


def plot_accuracy_phase_maps(
    accuracy_rows_all: Sequence[dict[str, Any]],
    q_values: Sequence[float],
    combined_path: Path,
    per_q_dir: Path,
) -> list[Path]:
    ensure_dir(combined_path.parent)
    ensure_dir(per_q_dir)
    by_q: dict[str, list[dict[str, Any]]] = {}
    for row in accuracy_rows_all:
        by_q.setdefault(str(row["q"]), []).append(row)

    fig, axes = plt.subplots(1, len(q_values), figsize=(5.2 * len(q_values), 4.4), sharey=True, constrained_layout=True)
    if len(q_values) == 1:
        axes = np.asarray([axes])
    mesh = None
    for ax, q_value in zip(axes, q_values):
        mesh = _draw_accuracy_phase_panel(ax, by_q.get(_q_label(q_value), []), q_value=q_value)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label(r"accuracy cutoff $A_q$ (%)")
    axes[0].set_ylabel(r"$\beta$")
    fig.savefig(combined_path, dpi=220)
    plt.close(fig)

    outputs = [combined_path]
    for q_value in q_values:
        rows = by_q.get(_q_label(q_value), [])
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        mesh = _draw_accuracy_phase_panel(ax, rows, q_value=q_value)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(r"accuracy cutoff $A_q$ (%)")
        ax.set_ylabel(r"$\beta$")
        out_path = per_q_dir / f"accuracy_phase_map_{_q_slug(q_value)}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def _rows_for_q(rows: Sequence[dict[str, Any]], q_value: float) -> list[dict[str, Any]]:
    label = _q_label(float(q_value))
    return [row for row in rows if str(row.get("q")) == label]


def _metric_matrix(rows: Sequence[dict[str, Any]], value_key: str) -> tuple[np.ndarray, list[float], list[float]]:
    betas = sorted({round(float(row["beta"]), 8) for row in rows})
    radii = sorted({round(float(row["radius"]), 8) for row in rows})
    matrix = np.full((len(betas), len(radii)), np.nan, dtype=np.float64)
    beta_index = {value: index for index, value in enumerate(betas)}
    radius_index = {value: index for index, value in enumerate(radii)}
    for row in rows:
        beta = round(float(row["beta"]), 8)
        radius = round(float(row["radius"]), 8)
        value = _finite_float(row.get(value_key))
        if not np.isfinite(value) and value_key == "H_q_numeric":
            text = str(row.get("H_q", "")).strip().lower()
            if text in {"inf", "+inf", "infinity", "+infinity"}:
                value = 9.0
        if np.isfinite(value):
            matrix[beta_index[beta], radius_index[radius]] = value
    return matrix, betas, radii


def _claim_matrix(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[float], list[float]]:
    betas = sorted({round(float(row["beta"]), 8) for row in rows})
    radii = sorted({round(float(row["radius"]), 8) for row in rows})
    matrix = np.full((len(betas), len(radii)), np.nan, dtype=np.float64)
    beta_index = {value: index for index, value in enumerate(betas)}
    radius_index = {value: index for index, value in enumerate(radii)}
    for row in rows:
        beta = round(float(row["beta"]), 8)
        radius = round(float(row["radius"]), 8)
        claim = str(row.get("claim", "")).strip().lower()
        matrix[beta_index[beta], radius_index[radius]] = 1.0 if claim == "pass" else 0.0
    return matrix, betas, radii


def _blank_figure(path: Path, title: str) -> Path:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_metric_heatmap(rows: Sequence[dict[str, Any]], path: Path, *, value_key: str, title: str, cbar_label: str) -> Path:
    matrix, betas, radii = _metric_matrix(rows, value_key)
    if matrix.size == 0 or not np.any(np.isfinite(matrix)):
        return _blank_figure(path, title)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    finite = matrix[np.isfinite(matrix)]
    mesh = ax.pcolormesh(
        _grid_edges(radii),
        _grid_edges(betas),
        np.ma.masked_invalid(matrix),
        cmap="viridis",
        vmin=float(np.nanpercentile(finite, 2.0)),
        vmax=float(np.nanpercentile(finite, 98.0)),
        shading="flat",
        edgecolors="white",
        linewidth=0.12,
    )
    ax.set_xlabel(r"distance threshold $d$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title)
    ax.set_xticks(radii[:: max(1, len(radii) // 10)])
    ax.set_yticks(betas)
    ax.tick_params(axis="x", labelrotation=45)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_claim_heatmap(rows: Sequence[dict[str, Any]], path: Path, *, title: str) -> Path:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    matrix, betas, radii = _claim_matrix(rows)
    if matrix.size == 0:
        return _blank_figure(path, title)
    cmap = ListedColormap(["#ef4444", "#16a34a"])
    cmap.set_bad("#f1f5f9")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        _grid_edges(radii),
        _grid_edges(betas),
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.12,
    )
    ax.set_xlabel(r"distance threshold $d$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title)
    ax.set_xticks(radii[:: max(1, len(radii) // 10)])
    ax.set_yticks(betas)
    ax.tick_params(axis="x", labelrotation=45)
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1], pad=0.02)
    cbar.ax.set_yticklabels(["no_claim", "pass"])
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_metric_contour(rows: Sequence[dict[str, Any]], path: Path, *, value_key: str, title: str, cbar_label: str) -> Path:
    matrix, betas, radii = _metric_matrix(rows, value_key)
    if matrix.size == 0 or len(betas) < 2 or len(radii) < 2 or not np.any(np.isfinite(matrix)):
        return _plot_metric_heatmap(rows, path, value_key=value_key, title=title, cbar_label=cbar_label)
    x, y = np.meshgrid(np.asarray(radii, dtype=np.float64), np.asarray(betas, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    levels = min(12, max(4, int(np.sum(np.isfinite(matrix)))))
    contour = ax.contourf(x, y, np.ma.masked_invalid(matrix), levels=levels, cmap="viridis")
    ax.contour(x, y, np.ma.masked_invalid(matrix), levels=levels, colors="white", linewidths=0.35, alpha=0.7)
    ax.set_xlabel(r"distance threshold $d$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title)
    cbar = fig.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _phase_alias_figures(
    *,
    accuracy_rows_all: Sequence[dict[str, Any]],
    hq_rows_all: Sequence[dict[str, Any]],
    figs_dir: Path,
    q_value: float = 0.90,
) -> list[Path]:
    acc_rows = _rows_for_q(accuracy_rows_all, q_value)
    hq_rows = _rows_for_q(hq_rows_all, q_value)
    return [
        _plot_metric_heatmap(
            acc_rows,
            figs_dir / "fig_acc_phase_map.png",
            value_key="accuracy_q_percent",
            title=fr"Accuracy phase map, $q={_q_label(q_value)}$",
            cbar_label=r"accuracy cutoff $A_q$ (%)",
        ),
        _plot_claim_heatmap(acc_rows, figs_dir / "fig_acc_phase_claim_map.png", title=fr"Accuracy phase claim map, $q={_q_label(q_value)}$"),
        _plot_metric_contour(
            acc_rows,
            figs_dir / "fig_acc_phase_contour.png",
            value_key="accuracy_q_percent",
            title=fr"Accuracy phase contour, $q={_q_label(q_value)}$",
            cbar_label=r"accuracy cutoff $A_q$ (%)",
        ),
        _plot_metric_heatmap(
            hq_rows,
            figs_dir / "fig_hq_phase_map.png",
            value_key="H_q_numeric",
            title=fr"$H_q$ phase map, $q={_q_label(q_value)}$",
            cbar_label=r"$H_q$",
        ),
        _plot_claim_heatmap(hq_rows, figs_dir / "fig_hq_claim_map.png", title=fr"$H_q$ claim map, $q={_q_label(q_value)}$"),
        _plot_metric_contour(
            hq_rows,
            figs_dir / "fig_hq_phase_contour.png",
            value_key="H_q_numeric",
            title=fr"$H_q$ phase contour, $q={_q_label(q_value)}$",
            cbar_label=r"$H_q$",
        ),
    ]


def plot_entropic_derivative(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    by_radius: dict[float, list[float]] = {}
    for row in rows:
        radius = _finite_float(row.get("radius"))
        value = _finite_float(row.get("dphi_entropic_dr"))
        if np.isfinite(radius) and np.isfinite(value):
            by_radius.setdefault(round(radius, 8), []).append(value)
    x = np.asarray(sorted(by_radius), dtype=np.float64)
    y = np.asarray([float(np.mean(by_radius[float(radius)])) for radius in x], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(x, y, marker="o", markersize=3.4, linewidth=1.8, color="#3f3f46")
    _style_axes(
        ax,
        xlabel=r"distance threshold $d$",
        ylabel=r"$dA/dd$",
        title=r"Radial derivative of geometric shell term $A(d)$",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def render_proxy_figures(
    *,
    summary_root: Path,
    unit_rows: Sequence[dict[str, Any]],
    phi_rows: Sequence[dict[str, Any]],
    rli_rows: Sequence[dict[str, Any]],
    q_values: Sequence[float] = DEFAULT_Q_VALUES,
) -> dict[str, list[Path]]:
    figs_dir = summary_root / "figs"
    tables: list[Path] = []
    figs: list[Path] = []
    audit_figs: list[Path] = []
    _clear_png_outputs(figs_dir)

    abs_phi_rows = absolute_phi_rows(unit_rows)
    abs_phi_path = summary_root / "absolute_phi_by_beta_radius.csv"
    save_csv(abs_phi_path, abs_phi_rows, list(abs_phi_rows[0].keys()) if abs_phi_rows else ["beta", "radius", "phi_full", "claim"])
    tables.append(abs_phi_path)

    derivative_rows = derivative_phi_rows(abs_phi_rows)
    derivative_path = summary_root / "dphi_dr_by_beta_radius.csv"
    save_csv(derivative_path, derivative_rows, list(derivative_rows[0].keys()) if derivative_rows else ["beta", "radius", "dphi_full_dr", "claim"])
    tables.append(derivative_path)

    hq_rows_all = hq_phase_rows(rli_rows, q_values=q_values)
    hq_path = summary_root / "hq_by_beta_radius.csv"
    save_csv(hq_path, hq_rows_all, list(hq_rows_all[0].keys()) if hq_rows_all else ["q", "beta", "radius", "H_q"])
    tables.append(hq_path)

    accuracy_rows_all = accuracy_q_phase_rows(
        unit_rows,
        phi_rows,
        q_values=q_values,
        progress_path=summary_root / "logs" / "accuracy_phase_status.json",
    )
    accuracy_path = summary_root / "accuracy_q_by_beta_radius.csv"
    save_csv(
        accuracy_path,
        accuracy_rows_all,
        list(accuracy_rows_all[0].keys()) if accuracy_rows_all else ["q", "beta", "radius", "accuracy_q", "claim"],
    )
    tables.append(accuracy_path)

    figs.append(
        plot_phi(
            abs_phi_rows,
            figs_dir / "phi_by_distance.png",
            value_key="phi_full",
            ylabel=r"$\phi^{\mathrm{full}}_\beta(d)$",
            title=r"Raw full Gibbs $\phi^{\mathrm{full}}_\beta(d)$",
        )
    )
    figs.append(
        plot_energy_term(
            abs_phi_rows,
            figs_dir / "phi_energy_term_by_distance.png",
            value_key="phi_energy",
            ylabel=r"$E_\beta(d)$",
            title=r"Raw volume-removed Gibbs contribution $E_\beta(d)$",
        )
    )
    figs.append(
        plot_entropic_term(
            abs_phi_rows,
            figs_dir / "phi_entropic_term_by_distance.png",
            ylabel=r"$A(d)$",
            title=r"Raw geometric shell term $A(d)$",
        )
    )

    figs.append(plot_delta_phi(phi_rows, figs_dir / "delta_phi_by_distance.png"))
    figs.append(
        plot_energy_term(
            phi_rows,
            figs_dir / "delta_phi_energy_term_by_distance.png",
            value_key="delta_phi_energy",
            ylabel=r"$\Delta E_\beta(d;0.1)$",
            title=r"Relative energy/angular shift $\Delta E_\beta(d;0.1)$",
        )
    )
    figs.append(
        plot_entropic_term(
            phi_rows,
            figs_dir / "delta_phi_entropic_term_by_distance.png",
            ylabel=r"$\Delta A(d;0.1)$",
            title=r"Relative geometric shell shift $\Delta A(d;0.1)$",
        )
    )

    figs.append(
        plot_energy_term(
            derivative_rows,
            figs_dir / "dphi_full_dr_by_distance.png",
            value_key="dphi_full_dr",
            ylabel=r"$d\phi^{\mathrm{full}}_\beta/dd$",
            title=r"Radial derivative of full Gibbs $\phi_\beta(d)$",
        )
    )
    figs.append(
        plot_energy_term(
            derivative_rows,
            figs_dir / "dphi_energy_dr_by_distance.png",
            value_key="dphi_energy_dr",
            ylabel=r"$dE_\beta/dd$",
            title=r"Radial derivative of volume-removed Gibbs contribution",
        )
    )
    figs.append(plot_entropic_derivative(derivative_rows, figs_dir / "dphi_entropic_dr_by_distance.png"))

    audit_figs.append(
        plot_phi(
            abs_phi_rows,
            figs_dir / "audit_phi_stripped_proxy_by_distance.png",
            value_key="phi_stripped_proxy",
            ylabel=r"$\phi_{\mathrm{stripped}}(d)-C_{\mathrm{stripped}}$",
            title=r"Audit: stripped PM-SAIS proxy $\phi(d)$",
            center_value=_global_center_value(abs_phi_rows, "phi_stripped_proxy"),
        )
    )
    audit_figs.append(
        plot_energy_term(
            abs_phi_rows,
            figs_dir / "audit_phi_stripped_proxy_energy_term_by_distance.png",
            value_key="phi_energy_stripped_proxy",
            ylabel=r"$E_{\mathrm{stripped}}(d)$",
            title=r"Audit: stripped proxy energy/angular contribution",
        )
    )
    audit_figs.append(plot_reference_prior_correction(abs_phi_rows, figs_dir / "audit_reference_prior_correction_by_beta.png"))
    audit_figs.extend(plot_hq_phase_maps(hq_rows_all, list(q_values), figs_dir / "hq_phase_maps_q050_q090_q099.png", figs_dir))
    audit_figs.extend(
        plot_accuracy_phase_maps(
            accuracy_rows_all,
            list(q_values),
            figs_dir / "accuracy_phase_maps_q050_q090_q099.png",
            figs_dir,
        )
    )
    figs.extend(_phase_alias_figures(accuracy_rows_all=accuracy_rows_all, hq_rows_all=hq_rows_all, figs_dir=figs_dir))
    return {"tables": tables, "figs": figs, "audit_figs": audit_figs}
