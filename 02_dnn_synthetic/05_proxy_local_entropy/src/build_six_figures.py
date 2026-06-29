from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "02_dnn_synthetic" / "05_proxy_local_entropy"
RUN_ROOT = STAGE_ROOT / "summarized_outputs" / "18_beta_cell_90_dataset_30_reference" / "d_0.01_to_2.50_dense"
SUMMARY_ROOT = RUN_ROOT / "summary_tables"
COMPLEXITY_SUMMARY = (
    REPO_ROOT
    / "02_dnn_synthetic"
    / "02_complexity_measure"
    / "summarized_outputs"
    / "18_beta_cell_90_dataset"
    / "beta_complexity_summary.csv"
)
FIGURE_ROOT = STAGE_ROOT / "figures"


FIGURE_SPECS = (
    ("phi_d_curve.png", "absolute_phi_by_beta_radius.csv", "phi_full", r"$\phi(d)$", r"$\phi(d)$ by distance"),
    ("phi_energetic_d_curve.png", "absolute_phi_by_beta_radius.csv", "phi_energy", r"energetic $\phi(d)$", r"Energetic $\phi(d)$ by distance"),
    ("derivative_phi_d_curve.png", "dphi_dr_by_beta_radius.csv", "dphi_full_dr", r"$d\phi/dd$", r"Derivative of $\phi(d)$"),
    (
        "derivative_phi_energetic_d_curve.png",
        "dphi_dr_by_beta_radius.csv",
        "dphi_energy_dr",
        r"energetic $d\phi/dd$",
        r"Energetic derivative of $\phi(d)$",
    ),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _group_curves(rows: Iterable[dict[str, str]], value_key: str) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        beta = round(_float(row.get("beta")), 8)
        radius = _float(row.get("radius"))
        value = _float(row.get(value_key))
        if np.isfinite(beta) and np.isfinite(radius) and np.isfinite(value):
            grouped.setdefault(beta, []).append((radius, value))
    return {
        beta: (
            np.asarray([x for x, _y in sorted(values)], dtype=np.float64),
            np.asarray([y for _x, y in sorted(values)], dtype=np.float64),
        )
        for beta, values in sorted(grouped.items())
    }


def _plot_curves(rows: list[dict[str, str]], value_key: str, ylabel: str, title: str, path: Path) -> None:
    curves = _group_curves(rows, value_key)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, (beta, (radius, value)) in enumerate(curves.items()):
        color = cmap(index / max(len(curves) - 1, 1))
        ax.plot(radius, value, linewidth=1.35, alpha=0.9, color=color)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(curves), max(curves)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\beta$")
    ax.set_xscale("log")
    ax.set_xlabel(r"distance $d$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _a_measure_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    curves = _group_curves(rows, "dphi_energy_dr")
    out: list[dict[str, object]] = []
    for beta, (radius, value) in curves.items():
        mask = (radius > 0) & np.isfinite(value)
        radius = radius[mask]
        value = value[mask]
        if radius.size < 3:
            continue
        log_radius = np.log(radius)
        slope = np.gradient(value, log_radius)
        total_variation = float(np.trapz(np.abs(slope), log_radius))
        signed_variation = float(np.trapz(slope, log_radius))
        out.append(
            {
                "beta": beta,
                "A_transition_total_variation": total_variation,
                "A_transition_signed_variation": signed_variation,
                "radius_count": int(radius.size),
            }
        )
    return out


def _complexity_by_beta() -> dict[float, float]:
    rows = _read_csv(COMPLEXITY_SUMMARY)
    return {round(float(row["beta"]), 8): float(row["complexity_mean"]) for row in rows}


def _plot_phase_panel(
    dphi_rows: list[dict[str, str]],
    a_rows: list[dict[str, object]],
    path: Path,
    *,
    x_key: str,
    x_label: str,
    title: str,
) -> None:
    curves = _group_curves(dphi_rows, "dphi_energy_dr")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, (beta, (radius, value)) in enumerate(curves.items()):
        ax_left.plot(radius, value, linewidth=1.1, alpha=0.85, color=cmap(index / max(len(curves) - 1, 1)))
    ax_left.set_xscale("log")
    ax_left.set_xlabel(r"distance $d$")
    ax_left.set_ylabel(r"energetic $d\phi/dd$")
    ax_left.set_title("Energetic derivative")
    ax_left.grid(True, alpha=0.22)

    x = np.asarray([float(row[x_key]) for row in a_rows], dtype=np.float64)
    y = np.asarray([float(row["A_transition_total_variation"]) for row in a_rows], dtype=np.float64)
    order = np.argsort(x)
    ax_right.plot(x[order], y[order], "o-", color="#7b3f8c", linewidth=1.4, markersize=4.5)
    ax_right.set_xlabel(x_label)
    ax_right.set_ylabel("A measure")
    ax_right.set_title(title)
    ax_right.grid(True, alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    for output_name, source_name, value_key, ylabel, title in FIGURE_SPECS:
        rows = _read_csv(SUMMARY_ROOT / source_name)
        _plot_curves(rows, value_key, ylabel, title, FIGURE_ROOT / output_name)

    dphi_rows = _read_csv(SUMMARY_ROOT / "dphi_dr_by_beta_radius.csv")
    a_rows = _a_measure_rows(dphi_rows)
    complexity_lookup = _complexity_by_beta()
    enriched_a_rows: list[dict[str, object]] = []
    for row in a_rows:
        beta = round(float(row["beta"]), 8)
        enriched = dict(row)
        enriched["complexity_mean"] = complexity_lookup.get(beta, float("nan"))
        enriched_a_rows.append(enriched)
    _write_csv(SUMMARY_ROOT / "phase_like_A_measure.csv", enriched_a_rows)
    _plot_phase_panel(
        dphi_rows,
        enriched_a_rows,
        FIGURE_ROOT / "phase_like_A_by_beta.png",
        x_key="beta",
        x_label=r"$\beta$",
        title="A measure by beta",
    )
    _plot_phase_panel(
        dphi_rows,
        enriched_a_rows,
        FIGURE_ROOT / "phase_like_A_by_complexity.png",
        x_key="complexity_mean",
        x_label="3-NN complexity",
        title="A measure by complexity",
    )


if __name__ == "__main__":
    main()
