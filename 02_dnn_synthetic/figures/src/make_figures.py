#!/usr/bin/env python3
"""Rebuild every figure available from the 02_dnn_synthetic stage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DNN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = DNN_ROOT.parent
FIGURE_ROOT = DNN_ROOT / "figures"

DATASET_SUMMARY_ROOT = DNN_ROOT / "01_dataset" / "summarized_outputs" / "figure_inputs"
COMPLEXITY_SUMMARY = DNN_ROOT / "02_complexity_measure" / "summarized_outputs" / "beta_complexity_summary.csv"
SAMPLING_INPUT_ROOT = DNN_ROOT / "04_sampling" / "summarized_outputs" / "figure_inputs" / "logZ_split"
PLE_INPUT_ROOT = DNN_ROOT / "05_proxy_local_entropy" / "summarized_outputs" / "figure_inputs"

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


@dataclass(frozen=True)
class FigureRecord:
    section: str
    title: str
    path: Path
    inputs: tuple[Path, ...]


def rel_to_dnn(path: Path) -> str:
    return str(path.resolve().relative_to(DNN_ROOT))


def rel_to_figure(path: Path) -> str:
    return str(path.resolve().relative_to(FIGURE_ROOT))


def clear_outputs(root: Path, *patterns: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()


def clear_figure_root() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in FIGURE_ROOT.iterdir():
        if path.name == "src":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def beta_from_name(path: Path) -> float:
    match = re.search(r"(\d+)p(\d+)", path.stem)
    if match is None:
        raise ValueError(f"cannot parse beta from {path.name}")
    return float(f"{match.group(1)}.{match.group(2)}")


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def group_curves(
    frame: pd.DataFrame,
    value_key: str,
    sem_key: str | None,
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    required = {"beta", "radius", value_key}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns for curve plot: {', '.join(missing)}")

    work = frame.copy()
    for col in ("beta", "radius", value_key):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if sem_key is not None and sem_key in work.columns:
        work[sem_key] = pd.to_numeric(work[sem_key], errors="coerce")

    grouped: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for beta, group in work.dropna(subset=["beta", "radius", value_key]).groupby("beta", sort=True):
        group = group.sort_values("radius")
        radius = group["radius"].to_numpy(dtype=float)
        value = group[value_key].to_numpy(dtype=float)
        sem: np.ndarray | None = None
        if sem_key is not None and sem_key in group.columns:
            sem_values = group[sem_key].to_numpy(dtype=float)
            if np.isfinite(sem_values).any():
                sem = sem_values
        grouped[round(float(beta), 8)] = (radius, value, sem)
    return grouped


def plot_curve_frame(
    frame: pd.DataFrame,
    value_key: str,
    sem_key: str | None,
    ylabel: str,
    title: str,
    path: Path,
    *,
    figsize: tuple[float, float] = (7.4, 4.8),
    dpi: int = 240,
    xscale: str = "linear",
    colorbar: bool = True,
    legend: bool = False,
) -> None:
    curves = group_curves(frame, value_key, sem_key)
    if not curves:
        raise ValueError(f"no finite rows to plot for {path}")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(curves), max(curves))
    has_band = False

    for beta, (radius, value, sem) in curves.items():
        color = cmap(norm(beta))
        label = rf"$\beta={beta:.2f}$"
        ax.plot(radius, value, linewidth=1.35, alpha=0.9, color=color, label=label)
        if sem is not None:
            err = np.nan_to_num(sem, nan=0.0)
            ax.fill_between(radius, value - err, value + err, color=color, alpha=0.10, linewidth=0)
            has_band = True

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$\beta$")
    if legend:
        ax.legend(frameon=False, fontsize=7.0, ncol=2)

    ax.set_xscale(xscale)
    ax.set_xlabel("distance d")
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
    fig.savefig(path)
    plt.close(fig)


def build_dataset_figures() -> list[FigureRecord]:
    out_root = FIGURE_ROOT / "01_dataset"
    clear_outputs(out_root, "*.png")
    records: list[FigureRecord] = []

    sample_csv = require_file(DATASET_SUMMARY_ROOT / "sample_figures" / "selected_sample_indices.csv")
    sample_frame = pd.read_csv(sample_csv)
    if sample_frame.empty:
        raise ValueError(f"{sample_csv} is empty")

    sample_path = out_root / "sample_figure.png"
    ncols = 6
    nrows = int(math.ceil(len(sample_frame) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.45 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, row in sample_frame.iterrows():
        image_path = require_file(REPO_ROOT / str(row["source_image_path"]))
        ax = axes[int(idx) // ncols, int(idx) % ncols]
        ax.imshow(mpimg.imread(image_path))
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.05)
    fig.savefig(sample_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    records.append(FigureRecord("01_dataset", "sample figure", sample_path, (sample_csv,)))

    spin_csv = require_file(DATASET_SUMMARY_ROOT / "spin_dynamics" / "spin_alignment_by_beta.csv")
    spin_frame = pd.read_csv(spin_csv)
    beta = pd.to_numeric(spin_frame["beta_ising"], errors="coerce").to_numpy(dtype=float)
    mean = pd.to_numeric(spin_frame["mean_edge_alignment"], errors="coerce").to_numpy(dtype=float)
    sem = pd.to_numeric(spin_frame["sem_edge_alignment"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(beta) & np.isfinite(mean) & np.isfinite(sem)
    beta = beta[mask]
    mean = mean[mask]
    sem = sem[mask]
    order = np.argsort(beta)

    spin_path = out_root / "spin_dynamics_phase_transition.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(beta[order], mean[order], color="#252525", linewidth=2.2, marker="o", markersize=4.8)
    ax.fill_between(
        beta[order],
        mean[order] - sem[order],
        mean[order] + sem[order],
        color="#5b8db8",
        alpha=0.22,
        linewidth=0.0,
    )
    ax.set_xlabel("inverse temperature beta (lower T to the right)")
    ax.set_ylabel("mean edge spin alignment <s_i s_j>")
    ax.set_title("Spin-dynamics snapshots show temperature-driven ordering")
    ax.set_ylim(0.0, 0.96)
    ax.set_xlim(float(beta.min()) - 0.01, float(beta.max()) + 0.01)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax.text(
        0.03,
        0.93,
        "90 final snapshots per beta\n2000 Kawasaki sweeps per snapshot",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ticks = np.asarray([0.05, 0.10, 0.20, 0.30, 0.39], dtype=float)
    top_ax.set_xticks(top_ticks)
    top_ax.set_xticklabels([f"{1.0 / tick:.1f}" for tick in top_ticks])
    top_ax.set_xlabel("temperature T = 1 / beta")
    fig.savefig(spin_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    records.append(FigureRecord("01_dataset", "spin dynamics phase transition", spin_path, (spin_csv,)))

    return records


def build_complexity_figure() -> list[FigureRecord]:
    out_root = FIGURE_ROOT / "02_complexity_measure"
    clear_outputs(out_root, "*.png")
    summary_path = require_file(COMPLEXITY_SUMMARY)
    frame = pd.read_csv(summary_path)
    beta = pd.to_numeric(frame["beta"], errors="coerce").to_numpy(dtype=float)
    mean = pd.to_numeric(frame["complexity_mean"], errors="coerce").to_numpy(dtype=float)
    se = pd.to_numeric(frame["complexity_se"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(beta) & np.isfinite(mean) & np.isfinite(se)
    beta = beta[mask]
    mean = mean[mask]
    se = se[mask]
    order = np.argsort(beta)
    r = pearson_r(beta[order], mean[order])

    out_path = out_root / "beta_complexity_figure.png"
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.errorbar(beta[order], mean[order], yerr=se[order], fmt="o-", color="#284f8f", ecolor="#8aa7d6", capsize=3)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("3-NN label-disagreement complexity")
    ax.set_title(f"Beta vs complexity (Pearson r={r:.3f})")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

    return [FigureRecord("02_complexity_measure", "beta complexity figure", out_path, (summary_path,))]


def radius_label(value: float) -> str:
    return f"{value:g}"


def xtick_labels(radii: list[float]) -> list[str]:
    if len(radii) <= 24:
        return [radius_label(value) for value in radii]
    step = max(1, int(np.ceil(len(radii) / 14)))
    return [radius_label(value) if idx % step == 0 or idx == len(radii) - 1 else "" for idx, value in enumerate(radii)]


def plot_logz_split_distribution(input_csv: Path, output_png: Path, *, max_scatter_per_radius: int) -> None:
    frame = pd.read_csv(input_csv)
    frame["r"] = pd.to_numeric(frame["r"], errors="coerce")
    frame["signed_split_logZ_per_P_diff"] = pd.to_numeric(frame["signed_split_logZ_per_P_diff"], errors="coerce")
    frame = frame.dropna(subset=["r", "signed_split_logZ_per_P_diff"]).sort_values("r")
    if frame.empty:
        raise ValueError(f"no finite logZ split rows in {input_csv}")

    beta = float(frame["beta"].iloc[0])
    radii = [float(value) for value in sorted(frame["r"].unique())]
    values = [frame.loc[np.isclose(frame["r"], radius), "signed_split_logZ_per_P_diff"].to_numpy(float) for radius in radii]
    positions = np.arange(len(radii), dtype=float)
    width = max(10.5, min(18.0, 0.34 * len(radii)))
    fig, ax = plt.subplots(figsize=(width, 5.8))

    violin_positions = positions + 0.16
    parts = ax.violinplot(
        values,
        positions=violin_positions,
        widths=0.34,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for center, body in zip(violin_positions, parts["bodies"]):
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.maximum(vertices[:, 0], center)
        body.set_facecolor("#6f6f6f")
        body.set_edgecolor("none")
        body.set_alpha(0.82)

    rng = np.random.default_rng(1729 + int(round(beta * 1000)))
    color_extent = max(float(np.nanquantile(np.abs(frame["signed_split_logZ_per_P_diff"]), 0.98)), 1.0e-12)
    for idx, radius in enumerate(radii):
        group = frame.loc[np.isclose(frame["r"], radius), "signed_split_logZ_per_P_diff"].to_numpy(float)
        if len(group) > max_scatter_per_radius:
            group = rng.choice(group, size=max_scatter_per_radius, replace=False)
        x = rng.normal(loc=positions[idx] - 0.12, scale=0.045, size=len(group))
        x = np.clip(x, positions[idx] - 0.27, positions[idx] + 0.04)
        ax.scatter(
            x,
            group,
            c=group,
            cmap="coolwarm",
            vmin=-color_extent,
            vmax=color_extent,
            s=18,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    y_extent = max(float(np.nanquantile(np.abs(frame["signed_split_logZ_per_P_diff"]), 0.995)), 1.0e-12) * 1.12
    ax.set_ylim(-y_extent, y_extent)
    ax.set_xlim(-0.55, len(radii) - 0.45)
    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_labels(radii), rotation=90, fontsize=7)
    ax.set_xlabel("d")
    ax.set_ylabel("signed split logZ diff per P")
    ax.set_title(f"logZ split distributions, beta={beta:.2f}")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def build_sampling_figures(max_scatter_per_radius: int) -> list[FigureRecord]:
    out_root = FIGURE_ROOT / "04_sampling" / "logZ_split_distributions"
    clear_outputs(out_root, "*.png", "manifest.csv")
    files = sorted(SAMPLING_INPUT_ROOT.glob("beta_*.csv"), key=beta_from_name)
    if not files:
        raise FileNotFoundError(f"no beta_*.csv files found under {SAMPLING_INPUT_ROOT}")

    records: list[FigureRecord] = []
    manifest_rows: list[dict[str, object]] = []
    for input_csv in files:
        beta = beta_from_name(input_csv)
        out_path = out_root / f"{input_csv.stem}.png"
        plot_logz_split_distribution(input_csv, out_path, max_scatter_per_radius=max_scatter_per_radius)
        records.append(FigureRecord("04_sampling", f"logZ split beta={beta:.2f}", out_path, (input_csv,)))
        manifest_rows.append(
            {
                "condition_name": "beta",
                "condition_value": beta,
                "input_csv": rel_to_dnn(input_csv),
                "figure_path": rel_to_dnn(out_path),
                "max_scatter_per_radius": max_scatter_per_radius,
            }
        )

    pd.DataFrame(manifest_rows).sort_values("condition_value").to_csv(out_root / "manifest.csv", index=False)
    return records


def plot_phase_panel(
    phase_frame: pd.DataFrame,
    derivative_frame: pd.DataFrame,
    x_key: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    curve_groups = group_curves(derivative_frame, "dphi_dr_smooth_mean", "dphi_dr_smooth_sem")
    if not curve_groups:
        raise ValueError(f"no finite derivative rows to plot for {output_path}")

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

    ax_left.set_xlabel("distance d")
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

    x = pd.to_numeric(phase_frame[x_key], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(phase_frame["A_transition_total_variation_mean"], errors="coerce").to_numpy(dtype=float)
    yerr = pd.to_numeric(phase_frame.get("A_transition_total_variation_sem"), errors="coerce").to_numpy(dtype=float)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def build_proxy_local_entropy_figures() -> list[FigureRecord]:
    out_root = FIGURE_ROOT / "05_proxy_local_entropy"
    clear_outputs(out_root, "*.png")
    records: list[FigureRecord] = []

    for name, value_key, sem_key, ylabel, title, output_name in CURVE_FIGURES:
        input_csv = require_file(PLE_INPUT_ROOT / name / f"{name}.csv")
        out_path = out_root / output_name
        plot_curve_frame(pd.read_csv(input_csv), value_key, sem_key, ylabel, title, out_path)
        records.append(FigureRecord("05_proxy_local_entropy", name, out_path, (input_csv,)))

    for name, x_key, x_label, title, output_name in (
        ("phase_like_A_by_beta", "beta", r"$\beta$", "A measure by beta", "phase_like_A_by_beta.png"),
        (
            "phase_like_A_by_complexity",
            "complexity_mean",
            "3-NN complexity",
            "A measure by complexity",
            "phase_like_A_by_complexity.png",
        ),
    ):
        phase_csv = require_file(PLE_INPUT_ROOT / name / f"{name}.csv")
        derivative_csv = require_file(PLE_INPUT_ROOT / name / "phase_derivative_curves.csv")
        out_path = out_root / output_name
        plot_phase_panel(pd.read_csv(phase_csv), pd.read_csv(derivative_csv), x_key, x_label, title, out_path)
        records.append(FigureRecord("05_proxy_local_entropy", name, out_path, (phase_csv, derivative_csv)))

    return records


def record_to_dict(record: FigureRecord) -> dict[str, Any]:
    return {
        "section": record.section,
        "title": record.title,
        "path": rel_to_dnn(record.path),
        "inputs": [rel_to_dnn(path) for path in record.inputs],
    }


def write_manifest(records: list[FigureRecord]) -> None:
    payload = {
        "root": rel_to_dnn(FIGURE_ROOT),
        "figure_count": len(records),
        "figures": [record_to_dict(record) for record in records],
        "policy": "All images under figures/ are redrawn from current summarized outputs or figure-input CSVs.",
    }
    (FIGURE_ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def section_title(section: str) -> str:
    return {
        "01_dataset": "01 Dataset",
        "02_complexity_measure": "02 Complexity Measure",
        "04_sampling": "04 Sampling",
        "05_proxy_local_entropy": "05 Proxy Local Entropy",
    }.get(section, section)


def grouped_records(records: Iterable[FigureRecord]) -> dict[str, list[FigureRecord]]:
    grouped: dict[str, list[FigureRecord]] = {}
    for record in records:
        grouped.setdefault(record.section, []).append(record)
    return grouped


def write_markdown_index(records: list[FigureRecord]) -> None:
    lines = [
        "# 02_dnn_synthetic figures",
        "",
        "Generated by `figures/src/make_figures.py` from the current summarized outputs.",
        "",
    ]
    for section, section_records in grouped_records(records).items():
        lines.extend([f"## {section_title(section)}", ""])
        for record in section_records:
            image_path = rel_to_figure(record.path)
            lines.extend([f"### {record.title}", "", f"![{record.title}]({image_path})", ""])
    (FIGURE_ROOT / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_html_index(records: list[FigureRecord]) -> None:
    sections = grouped_records(records)
    blocks = []
    for section, section_records in sections.items():
        cards = []
        for record in section_records:
            cards.append(
                f"""
                <article class="card">
                  <h3>{escape(record.title)}</h3>
                  <a href="{escape(rel_to_figure(record.path))}">
                    <img src="{escape(rel_to_figure(record.path))}" alt="{escape(record.title)}">
                  </a>
                </article>
                """
            )
        blocks.append(
            f"""
            <section>
              <h2>{escape(section_title(section))}</h2>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>02_dnn_synthetic figures</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #202124;
      background: #f5f6f7;
    }}
    header, section {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 22px;
    }}
    header {{
      padding-top: 38px;
      padding-bottom: 12px;
    }}
    h1, h2, h3 {{
      letter-spacing: 0;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    h2 {{
      margin: 4px 0 18px;
      font-size: 21px;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 650;
    }}
    p {{
      margin: 0;
      color: #5f6368;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #dedede;
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 1px 2px rgb(0 0 0 / 5%);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #ececec;
      background: white;
    }}
  </style>
</head>
<body>
  <header>
    <h1>02_dnn_synthetic figures</h1>
    <p>Generated by <code>figures/src/make_figures.py</code> from current summarized outputs.</p>
  </header>
  {''.join(blocks)}
</body>
</html>
"""
    (FIGURE_ROOT / "index.html").write_text(html, encoding="utf-8")


def build_all(max_scatter_per_radius: int) -> list[FigureRecord]:
    clear_figure_root()
    records: list[FigureRecord] = []
    records.extend(build_dataset_figures())
    records.extend(build_complexity_figure())
    records.extend(build_sampling_figures(max_scatter_per_radius=max_scatter_per_radius))
    records.extend(build_proxy_local_entropy_figures())
    write_manifest(records)
    write_markdown_index(records)
    write_html_index(records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all figures available from 02_dnn_synthetic.")
    parser.add_argument("--max-scatter-per-radius", type=int, default=120)
    args = parser.parse_args()

    records = build_all(max_scatter_per_radius=int(args.max_scatter_per_radius))
    print(f"figure_count={len(records)}")
    print(f"figure_root={FIGURE_ROOT}")
    print(f"index_md={FIGURE_ROOT / 'index.md'}")
    print(f"index_html={FIGURE_ROOT / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
