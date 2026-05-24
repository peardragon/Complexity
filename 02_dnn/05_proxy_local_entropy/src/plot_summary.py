from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from compute_phi import DEFAULT_LAMBDA_REG, P, enrich_unit_rows_for_full_phi
from io_utils import ensure_dir, save_csv


DEFAULT_Q_VALUES = (0.5, 0.9, 0.99)
H_LEVELS = (1.0, 2.0, 4.0, 6.0, 8.0, math.inf)
H_LABELS = ("1", "2", "4", "6", "8", "inf")
DEFAULT_CENTER_BETA = 0.06
DEFAULT_CENTER_RADIUS = 0.10


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _logmeanexp(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    peak = float(np.max(finite))
    return float(peak + math.log(float(np.mean(np.exp(finite - peak)))))


def _h_float(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    return float(value)


def _h_label(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if float(value).is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def _q_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _q_slug(value: float) -> str:
    return f"q{int(round(value * 100)):03d}"


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
        if not np.isfinite(stripped) or not np.isfinite(full) or radius <= 0:
            continue
        groups.setdefault((beta, radius), []).append(
            {
                "stripped": stripped,
                "full": full,
                "correction": correction,
                "norm_sq": norm_sq,
            }
        )

    out: list[dict[str, Any]] = []
    for (beta, radius), values in sorted(groups.items()):
        stripped_arr = np.asarray([row["stripped"] for row in values], dtype=np.float64)
        full_arr = np.asarray([row["full"] for row in values], dtype=np.float64)
        correction_arr = np.asarray([row["correction"] for row in values], dtype=np.float64)
        norm_sq_arr = np.asarray([row["norm_sq"] for row in values], dtype=np.float64)
        energy_stripped = float(np.mean(stripped_arr) / param_count)
        energy_full = float(np.mean(full_arr) / param_count)
        annealed_full_energy = float(_logmeanexp(full_arr) / param_count)
        annealed_stripped_energy = float(_logmeanexp(stripped_arr) / param_count)
        correction_per_p = float(np.mean(correction_arr[np.isfinite(correction_arr)]) / param_count) if np.any(np.isfinite(correction_arr)) else float("nan")
        mean_norm_sq = float(np.mean(norm_sq_arr[np.isfinite(norm_sq_arr)])) if np.any(np.isfinite(norm_sq_arr)) else float("nan")
        area = float((param_count - 1) / param_count * math.log(radius))
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
                "claim": "pass",
            }
        )
    return out


def hq_phase_rows(rli_rows: Sequence[dict[str, Any]], *, q_values: Iterable[float] = DEFAULT_Q_VALUES) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[tuple[float, float]]] = {}
    claims: dict[tuple[float, float], str] = {}
    ref_counts: dict[tuple[float, float], int] = {}
    for row in rli_rows:
        beta = round(float(row["beta"]), 8)
        radius = round(float(row["radius"]), 8)
        key = (beta, radius)
        h_value = _h_float(row["H"])
        mean_r = _finite_float(row.get("mean_R_H", row.get("R_H")))
        grouped.setdefault(key, []).append((h_value, mean_r))
        claims[key] = str(row.get("claim", claims.get(key, "")))
        try:
            ref_counts[key] = max(ref_counts.get(key, 0), int(row.get("ref_count", 0)))
        except (TypeError, ValueError):
            ref_counts[key] = ref_counts.get(key, 0)

    out: list[dict[str, Any]] = []
    for key, entries in sorted(grouped.items()):
        beta, radius = key
        sorted_entries = sorted(entries, key=lambda pair: pair[0])
        for q_value in q_values:
            h_q = float("nan")
            r_at_h = float("nan")
            for h_value, mean_r in sorted_entries:
                if np.isfinite(mean_r) and mean_r >= float(q_value):
                    h_q = h_value
                    r_at_h = mean_r
                    break
            out.append(
                {
                    "q": _q_label(float(q_value)),
                    "beta": beta,
                    "radius": radius,
                    "H_q": _h_label(h_q) if not np.isnan(h_q) else "nan",
                    "H_q_numeric": h_q,
                    "mean_R_H_at_H_q": r_at_h,
                    "ref_count": ref_counts.get(key, 0),
                    "claim": claims.get(key, ""),
                }
            )
    return out


def _line_groups(rows: Sequence[dict[str, Any]], value_key: str) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        x = _finite_float(row.get("radius"))
        y = _finite_float(row.get(value_key))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        grouped.setdefault(round(float(row["beta"]), 8), []).append((x, y))
    return {
        beta: (
            np.asarray([item[0] for item in sorted(values)], dtype=np.float64),
            np.asarray([item[1] for item in sorted(values)], dtype=np.float64),
        )
        for beta, values in sorted(grouped.items())
    }


def _global_center_value(
    rows: Sequence[dict[str, Any]],
    value_key: str,
    *,
    beta: float = DEFAULT_CENTER_BETA,
    radius: float = DEFAULT_CENTER_RADIUS,
) -> float:
    for row in rows:
        row_beta = _finite_float(row.get("beta"))
        row_radius = _finite_float(row.get("radius"))
        value = _finite_float(row.get(value_key))
        if np.isfinite(value) and abs(row_beta - beta) < 1e-8 and abs(row_radius - radius) < 1e-8:
            return value

    candidates: list[tuple[float, float, float]] = []
    for row in rows:
        row_beta = _finite_float(row.get("beta"))
        row_radius = _finite_float(row.get("radius"))
        value = _finite_float(row.get(value_key))
        if np.isfinite(row_beta) and np.isfinite(row_radius) and np.isfinite(value):
            candidates.append((row_radius, row_beta, value))
    if not candidates:
        return 0.0
    _row_radius, _row_beta, value = sorted(candidates)[0]
    return value


def _area_center_value(rows: Sequence[dict[str, Any]], *, radius: float = DEFAULT_CENTER_RADIUS) -> float:
    for row in rows:
        row_radius = _finite_float(row.get("radius"))
        area = _finite_float(row.get("area_term"))
        if np.isfinite(area) and abs(row_radius - radius) < 1e-8:
            return area
    areas = [
        (_finite_float(row.get("radius")), _finite_float(row.get("area_term")))
        for row in rows
        if np.isfinite(_finite_float(row.get("radius"))) and np.isfinite(_finite_float(row.get("area_term")))
    ]
    if not areas:
        return 0.0
    return sorted(areas)[0][1]


def _style_axes(ax: plt.Axes, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.7)


def plot_delta_phi(phi_rows: Sequence[dict[str, Any]], path: Path) -> Path:
    groups = _line_groups(phi_rows, "delta_phi_full")
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    betas = list(groups)
    for index, beta in enumerate(betas):
        x, y = groups[beta]
        color = cmap(index / max(len(betas) - 1, 1))
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.6, color=color, label=fr"$\beta={beta:g}$")
    _style_axes(
        ax,
        xlabel=r"distance threshold $d$",
        ylabel=r"$\Delta\phi^{\mathrm{full}}_\beta(d;0.1)$",
        title=r"Relative full Gibbs shift $\Delta\phi^{\mathrm{full}}_\beta(d;0.1)$",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_phi(
    phi_rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    value_key: str = "phi_full",
    ylabel: str = r"$\phi(d)$",
    title: str = r"Full Gibbs local entropy $\phi(d)$",
    center_value: float | None = None,
) -> Path:
    groups = _line_groups(phi_rows, value_key)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    betas = list(groups)
    for index, beta in enumerate(betas):
        x, y = groups[beta]
        if center_value is not None:
            y = y - float(center_value)
        color = cmap(index / max(len(betas) - 1, 1))
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.6, color=color, label=fr"$\beta={beta:g}$")
    _style_axes(ax, xlabel=r"distance threshold $d$", ylabel=ylabel, title=title)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_energy_term(rows: Sequence[dict[str, Any]], path: Path, *, value_key: str, ylabel: str, title: str) -> Path:
    groups = _line_groups(rows, value_key)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    betas = list(groups)
    for index, beta in enumerate(betas):
        x, y = groups[beta]
        color = cmap(index / max(len(betas) - 1, 1))
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.6, color=color, label=fr"$\beta={beta:g}$")
    _style_axes(ax, xlabel=r"distance threshold $d$", ylabel=ylabel, title=title)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_centered_energy_term(rows: Sequence[dict[str, Any]], path: Path, *, value_key: str, ylabel: str, title: str, center_value: float) -> Path:
    groups = _line_groups(rows, value_key)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    betas = list(groups)
    for index, beta in enumerate(betas):
        x, y = groups[beta]
        color = cmap(index / max(len(betas) - 1, 1))
        ax.plot(x, y - float(center_value), marker="o", markersize=3.2, linewidth=1.6, color=color, label=fr"$\beta={beta:g}$")
    _style_axes(ax, xlabel=r"distance threshold $d$", ylabel=ylabel, title=title)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_entropic_term(rows: Sequence[dict[str, Any]], path: Path, *, ylabel: str, title: str, center_value: float = 0.0) -> Path:
    by_radius: dict[float, list[float]] = {}
    for row in rows:
        radius = _finite_float(row.get("radius"))
        area = _finite_float(row.get("area_term"))
        if np.isfinite(radius) and np.isfinite(area):
            by_radius.setdefault(round(radius, 8), []).append(area)
    x = np.asarray(sorted(by_radius), dtype=np.float64)
    y = np.asarray([float(np.mean(by_radius[float(radius)])) for radius in x], dtype=np.float64) - float(center_value)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(x, y, marker="o", markersize=3.4, linewidth=1.8, color="#3f3f46")
    _style_axes(ax, xlabel=r"distance threshold $d$", ylabel=ylabel, title=title)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_reference_prior_correction(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    by_beta: dict[float, list[float]] = {}
    for row in rows:
        beta = _finite_float(row.get("beta"))
        correction = _finite_float(row.get("reference_prior_correction_per_P"))
        if np.isfinite(beta) and np.isfinite(correction):
            by_beta.setdefault(round(beta, 8), []).append(correction)
    x = np.asarray(sorted(by_beta), dtype=np.float64)
    y = np.asarray([float(np.mean(by_beta[float(beta)])) for beta in x], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.6, color="#7c2d12")
    _style_axes(
        ax,
        xlabel=r"$\beta$",
        ylabel=r"$\langle\log w_{\mathrm{ref}}\rangle/P$",
        title=r"Audit: reference prior correction by $\beta$",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _clear_png_outputs(figs_dir: Path) -> None:
    if not figs_dir.exists():
        return
    for path in sorted(figs_dir.rglob("*.png")):
        path.unlink()


def _grid_edges(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(sorted(values), dtype=np.float64)
    if arr.size == 1:
        width = max(abs(arr[0]) * 0.1, 0.5)
        return np.asarray([arr[0] - width, arr[0] + width], dtype=np.float64)
    mids = (arr[:-1] + arr[1:]) / 2.0
    first = arr[0] - (mids[0] - arr[0])
    last = arr[-1] + (arr[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _h_index(h_value: object) -> float:
    try:
        h_float = _h_float(h_value)
    except (TypeError, ValueError):
        return float("nan")
    for index, level in enumerate(H_LEVELS):
        if math.isinf(level) and math.isinf(h_float):
            return float(index)
        if np.isfinite(level) and np.isfinite(h_float) and abs(h_float - level) < 1e-8:
            return float(index)
    return float("nan")


def _phase_matrix(hq_rows_for_q: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[float], list[float]]:
    betas = sorted({round(float(row["beta"]), 8) for row in hq_rows_for_q})
    radii = sorted({round(float(row["radius"]), 8) for row in hq_rows_for_q})
    beta_index = {value: index for index, value in enumerate(betas)}
    radius_index = {value: index for index, value in enumerate(radii)}
    matrix = np.full((len(betas), len(radii)), np.nan, dtype=np.float64)
    for row in hq_rows_for_q:
        beta = round(float(row["beta"]), 8)
        radius = round(float(row["radius"]), 8)
        matrix[beta_index[beta], radius_index[radius]] = _h_index(row["H_q"])
    return matrix, betas, radii


def _phase_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    colors = ("#2c7bb6", "#74add1", "#ffffbf", "#fdae61", "#d7191c", "#3f3f46")
    cmap = ListedColormap(colors)
    cmap.set_bad("#f1f5f9")
    norm = BoundaryNorm(np.arange(-0.5, len(H_LABELS) + 0.5, 1.0), cmap.N)
    return cmap, norm


def _draw_phase_panel(ax: plt.Axes, rows: Sequence[dict[str, Any]], *, q_value: float) -> Any:
    matrix, betas, radii = _phase_matrix(rows)
    cmap, norm = _phase_cmap()
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
    ax.set_title(fr"$H_q(\beta,d)$, $q={_q_label(q_value)}$")
    ax.set_xlabel(r"distance threshold $d$")
    ax.set_xticks(radii[:: max(1, len(radii) // 10)])
    ax.set_yticks(betas)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(False)
    return mesh


def plot_hq_phase_maps(hq_rows_all: Sequence[dict[str, Any]], q_values: Sequence[float], combined_path: Path, per_q_dir: Path) -> list[Path]:
    ensure_dir(combined_path.parent)
    ensure_dir(per_q_dir)
    by_q: dict[str, list[dict[str, Any]]] = {}
    for row in hq_rows_all:
        by_q.setdefault(str(row["q"]), []).append(row)

    fig, axes = plt.subplots(1, len(q_values), figsize=(5.2 * len(q_values), 4.4), sharey=True, constrained_layout=True)
    if len(q_values) == 1:
        axes = np.asarray([axes])
    mesh = None
    for ax, q_value in zip(axes, q_values):
        mesh = _draw_phase_panel(ax, by_q.get(_q_label(q_value), []), q_value=q_value)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), ticks=np.arange(len(H_LABELS)), pad=0.02)
        cbar.ax.set_yticklabels(H_LABELS)
        cbar.set_label(r"minimum accepted loss threshold $H_q$")
    axes[0].set_ylabel(r"$\beta$")
    fig.savefig(combined_path, dpi=220)
    plt.close(fig)

    outputs = [combined_path]
    for q_value in q_values:
        rows = by_q.get(_q_label(q_value), [])
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        mesh = _draw_phase_panel(ax, rows, q_value=q_value)
        cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(len(H_LABELS)), pad=0.02)
        cbar.ax.set_yticklabels(H_LABELS)
        cbar.set_label(r"minimum accepted loss threshold $H_q$")
        ax.set_ylabel(r"$\beta$")
        out_path = per_q_dir / f"hq_phase_map_{_q_slug(q_value)}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


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

    hq_rows_all = hq_phase_rows(rli_rows, q_values=q_values)
    hq_path = summary_root / "hq_by_beta_radius.csv"
    save_csv(hq_path, hq_rows_all, list(hq_rows_all[0].keys()) if hq_rows_all else ["q", "beta", "radius", "H_q"])
    tables.append(hq_path)

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

    full_center = _global_center_value(abs_phi_rows, "phi_full")
    energy_center = _global_center_value(abs_phi_rows, "phi_energy")
    area_center = _area_center_value(abs_phi_rows)
    figs.append(
        plot_phi(
            abs_phi_rows,
            figs_dir / "phi_by_distance.png",
            value_key="phi_full",
            ylabel=r"$\phi^{\mathrm{full}}_\beta(d)-C_{\mathrm{full}}$",
            title=r"Absolute full Gibbs $\phi^{\mathrm{full}}_\beta(d)$",
            center_value=full_center,
        )
    )
    figs.append(
        plot_centered_energy_term(
            abs_phi_rows,
            figs_dir / "phi_energy_term_by_distance.png",
            value_key="phi_energy",
            ylabel=r"$E_\beta(d)-C_E$",
            title=r"Absolute volume-removed Gibbs contribution $E_\beta(d)$",
            center_value=energy_center,
        )
    )
    figs.append(
        plot_entropic_term(
            abs_phi_rows,
            figs_dir / "phi_entropic_term_by_distance.png",
            ylabel=r"$A(d)-A(0.1)$",
            title=r"Absolute geometric shell term $A(d)$",
            center_value=area_center,
        )
    )

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
        plot_centered_energy_term(
            abs_phi_rows,
            figs_dir / "audit_phi_stripped_proxy_energy_term_by_distance.png",
            value_key="phi_energy_stripped_proxy",
            ylabel=r"$E_{\mathrm{stripped}}(d)-C_{\mathrm{stripped},E}$",
            title=r"Audit: stripped proxy energy/angular contribution",
            center_value=_global_center_value(abs_phi_rows, "phi_energy_stripped_proxy"),
        )
    )
    audit_figs.append(plot_reference_prior_correction(abs_phi_rows, figs_dir / "audit_reference_prior_correction_by_beta.png"))
    audit_figs.extend(plot_hq_phase_maps(hq_rows_all, list(q_values), figs_dir / "hq_phase_maps_q050_q090_q099.png", figs_dir))
    return {"tables": tables, "figs": figs, "audit_figs": audit_figs}
