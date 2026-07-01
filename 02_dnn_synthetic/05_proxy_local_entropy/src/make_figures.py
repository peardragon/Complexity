from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DNN_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = DNN_ROOT / "05_proxy_local_entropy"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
FIGURE_ROOT = STAGE_ROOT / "figures"

CURVE_FIGURES = (
    (
        "phi_d_curve",
        "phi_full_mean",
        "phi_full_sem",
        r"$\phi(d)$",
        r"Synthetic $\phi(d)$ by distance",
        "phi_d_curve.png",
    ),
    (
        "phi_energetic_d_curve",
        "phi_energy_mean",
        "phi_energy_sem",
        r"energetic $\phi(d)$",
        r"Synthetic energetic $\phi(d)$ by distance",
        "phi_energetic_d_curve.png",
    ),
    (
        "derivative_phi_d_curve",
        "dphi_full_dr_mean",
        "dphi_full_dr_sem",
        r"$d\phi/dd$",
        r"Synthetic derivative of $\phi(d)$",
        "derivative_phi_d_curve.png",
    ),
    (
        "derivative_phi_energetic_d_curve",
        "dphi_energy_dr_mean",
        "dphi_energy_dr_sem",
        r"energetic $d\phi/dd$",
        r"Synthetic energetic derivative of $\phi(d)$",
        "derivative_phi_energetic_d_curve.png",
    ),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _group_curves(
    rows: Iterable[dict[str, str]],
    value_key: str,
    sem_key: str,
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    grouped: dict[float, list[tuple[float, float, float]]] = {}
    for row in rows:
        beta = round(_float(row.get("beta")), 8)
        radius = _float(row.get("radius"))
        value = _float(row.get(value_key))
        sem = _float(row.get(sem_key))
        if np.isfinite(beta) and np.isfinite(radius) and np.isfinite(value):
            grouped.setdefault(beta, []).append((radius, value, sem))

    out: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for beta, values in sorted(grouped.items()):
        ordered = sorted(values)
        radius = np.asarray([x for x, _y, _sem in ordered], dtype=np.float64)
        value = np.asarray([y for _x, y, _sem in ordered], dtype=np.float64)
        sem_values = np.asarray([sem for _x, _y, sem in ordered], dtype=np.float64)
        sem_out = sem_values if np.isfinite(sem_values).any() else None
        out[beta] = (radius, value, sem_out)
    return out


def _plot_curve(
    rows: list[dict[str, str]],
    value_key: str,
    sem_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    curves = _group_curves(rows, value_key, sem_key)
    if not curves:
        raise ValueError(f"no finite rows to plot for {path}")

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(curves), max(curves))
    has_band = False

    for beta, (radius, value, sem) in curves.items():
        color = cmap(norm(beta))
        ax.plot(radius, value, linewidth=1.35, alpha=0.9, color=color)
        if sem is not None:
            err = np.nan_to_num(sem, nan=0.0)
            ax.fill_between(radius, value - err, value + err, color=color, alpha=0.10, linewidth=0)
            has_band = True

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\beta$")
    ax.set_xlabel("distance d (linear r sampling)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    if has_band:
        ax.text(
            0.01,
            0.02,
            "band: mean +/- standard error",
            transform=ax.transAxes,
            fontsize=7.5,
            color="0.35",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_phase(name: str, x_key: str, x_label: str, title: str, output_name: str) -> None:
    root = FIGURE_INPUT_ROOT / name
    phase = _read_csv(root / f"{name}.csv")
    curves = _read_csv(root / "phase_derivative_curves.csv")

    curve_groups = _group_curves(curves, "dphi_dr_smooth_mean", "dphi_dr_smooth_sem")
    if not curve_groups:
        raise ValueError(f"no finite derivative rows to plot for {name}")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(curve_groups), max(curve_groups))
    has_band = False
    for beta, (radius, value, sem) in curve_groups.items():
        color = cmap(norm(beta))
        ax_left.plot(radius, value, linewidth=1.1, alpha=0.85, color=color)
        if sem is not None:
            err = np.nan_to_num(sem, nan=0.0)
            ax_left.fill_between(radius, value - err, value + err, color=color, alpha=0.09, linewidth=0)
            has_band = True

    ax_left.set_xlabel("distance d (linear r sampling)")
    ax_left.set_ylabel(r"energetic $d\phi/dd$")
    ax_left.set_title("Energetic derivative")
    ax_left.grid(True, alpha=0.22)
    if has_band:
        ax_left.text(
            0.01,
            0.02,
            "band: mean +/- standard error",
            transform=ax_left.transAxes,
            fontsize=7.5,
            color="0.35",
        )

    x = np.asarray([_float(row.get(x_key)) for row in phase], dtype=np.float64)
    y = np.asarray([_float(row.get("A_transition_total_variation_mean")) for row in phase], dtype=np.float64)
    yerr = np.asarray([_float(row.get("A_transition_total_variation_sem")) for row in phase], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    yerr = yerr[mask]
    order = np.argsort(x)
    if np.isfinite(yerr).any():
        ax_right.errorbar(
            x[order],
            y[order],
            yerr=np.nan_to_num(yerr[order], nan=0.0),
            marker="o",
            linewidth=1.5,
            capsize=2.5,
            color="#7a3e9d",
        )
    else:
        ax_right.plot(x[order], y[order], "o-", color="#7a3e9d", linewidth=1.4, markersize=4.5)
    ax_right.set_xlabel(x_label)
    ax_right.set_ylabel("A measure")
    ax_right.set_title(title)
    ax_right.grid(True, alpha=0.25)

    out_path = FIGURE_ROOT / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def build() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for name, value_key, sem_key, ylabel, title, output_name in CURVE_FIGURES:
        rows = _read_csv(FIGURE_INPUT_ROOT / name / f"{name}.csv")
        _plot_curve(rows, value_key, sem_key, ylabel, title, FIGURE_ROOT / output_name)

    _plot_phase(
        "phase_like_A_by_beta",
        "beta",
        r"$\beta$",
        "A measure by beta",
        "phase_like_A_by_beta.png",
    )
    _plot_phase(
        "phase_like_A_by_complexity",
        "complexity_mean",
        "3-NN complexity",
        "A measure by complexity",
        "phase_like_A_by_complexity.png",
    )


def main() -> None:
    build()


if __name__ == "__main__":
    main()
