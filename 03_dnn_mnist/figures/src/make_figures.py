#!/usr/bin/env python3
"""Build the 03_dnn_mnist top-level figure gallery from stage figures."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


DNN_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = DNN_ROOT / "figures"
MERGED_PLE_ROOT = FIGURE_ROOT / "05_proxy_local_entropy" / "merged"

STAGE_TITLES = {
    "01_dataset": "01 Dataset",
    "02_complexity_measure": "02 Complexity Measure",
    "03_reference_search": "03 Reference Search",
    "04_sampling": "04 Sampling",
    "05_proxy_local_entropy": "05 Proxy Local Entropy",
}
SERIES_TITLES = {
    "label_noise_sweep": "label noise sweep",
    "manual_rules": "manual rules",
    "merged": "merged endpoints + eta sweep",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
COMPANION_SUFFIXES = {".csv", ".json", ".md", ".txt"}
PLE_INPUT_ROOTS = {
    "label_noise_sweep": DNN_ROOT / "label_noise_sweep" / "05_proxy_local_entropy" / "summarized_outputs" / "figure_inputs",
    "manual_rules": DNN_ROOT / "manual_rules" / "05_proxy_local_entropy" / "summarized_outputs" / "figure_inputs",
}
ETA_COMPLEXITY_PATH = (
    DNN_ROOT
    / "label_noise_sweep"
    / "02_complexity_measure"
    / "summarized_outputs"
    / "eta_complexity_summary.csv"
)
MANUAL_COMPLEXITY_PATH = (
    DNN_ROOT
    / "manual_rules"
    / "02_complexity_measure"
    / "summarized_outputs"
    / "manual_rule_complexity_summary.csv"
)
ENDPOINT_ETA = {
    "real_even_odd": 0.0,
    "random_label": 0.5,
}
ENDPOINT_LABEL = {
    "real_even_odd": "even_odd (eta 0.00)",
    "random_label": "random (eta 0.50)",
}
CURVE_SPECS = (
    (
        "phi_d_curve",
        "delta_phi_energy_mean",
        "delta_phi_energy_sem",
        "delta_phi_energy_unit_mean",
        "delta_phi_energy_unit_sem",
        r"$\phi(d)-\phi(d_0)$",
        "Merged MNIST phi(d): endpoints + eta sweep",
    ),
    (
        "phi_energetic_d_curve",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        r"energetic $\phi(d)$",
        "Merged MNIST energetic phi(d): endpoints + eta sweep",
    ),
    (
        "derivative_phi_d_curve",
        "d_delta_phi_energy_dd",
        "d_delta_phi_energy_dd_sem",
        "d_delta_phi_energy_direct_dd_unit_mean",
        "d_delta_phi_energy_direct_dd_unit_sem",
        r"$d\phi/dd$",
        "Merged MNIST derivative of phi(d): endpoints + eta sweep",
    ),
    (
        "derivative_phi_energetic_d_curve",
        "d_phi_energy_direct_dd",
        "d_phi_energy_direct_dd_sem",
        "d_phi_energy_direct_dd_unit_mean",
        "d_phi_energy_direct_dd_unit_sem",
        r"energetic $d\phi/dd$",
        "Merged MNIST energetic derivative: endpoints + eta sweep",
    ),
)


@dataclass(frozen=True)
class FigureRecord:
    section: str
    series: str
    title: str
    path: Path
    inputs: tuple[Path, ...]


@dataclass(frozen=True)
class CopiedAsset:
    source: Path
    destination: Path
    is_figure: bool


def rel_to_dnn(path: Path) -> str:
    return str(path.resolve().relative_to(DNN_ROOT))


def rel_to_figure(path: Path) -> str:
    return str(path.resolve().relative_to(FIGURE_ROOT))


def clear_figure_root() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in FIGURE_ROOT.iterdir():
        if path.name == "src":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def stage_sort_key(stage: str) -> int:
    try:
        return list(STAGE_TITLES).index(stage)
    except ValueError:
        return len(STAGE_TITLES)


def source_series() -> tuple[tuple[str, Path], ...]:
    return (
        ("label_noise_sweep", DNN_ROOT / "label_noise_sweep"),
        ("manual_rules", DNN_ROOT / "manual_rules"),
    )


def should_copy(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES.union(COMPANION_SUFFIXES)


def is_figure(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def humanize_part(text: str) -> str:
    return text.replace("_", " ").replace("-", " ")


def title_for(series: str, relative_path: Path) -> str:
    parts = [SERIES_TITLES.get(series, series)]
    parts.extend(humanize_part(part) for part in relative_path.with_suffix("").parts)
    return " ".join(parts)


def copy_stage_assets() -> tuple[list[FigureRecord], list[CopiedAsset]]:
    records: list[FigureRecord] = []
    copied: list[CopiedAsset] = []

    for series, series_root in source_series():
        for stage in STAGE_TITLES:
            source_root = series_root / stage / "figures"
            if not source_root.exists():
                continue

            for source_path in sorted(path for path in source_root.rglob("*") if path.is_file() and should_copy(path)):
                relative_path = source_path.relative_to(source_root)
                destination = FIGURE_ROOT / stage / series / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)

                figure = is_figure(source_path)
                copied.append(CopiedAsset(source_path, destination, figure))
                if figure:
                    records.append(
                        FigureRecord(
                            section=stage,
                            series=series,
                            title=title_for(series, relative_path),
                            path=destination,
                            inputs=(source_path,),
                        )
                    )

    records.sort(key=lambda record: (stage_sort_key(record.section), record.series, rel_to_figure(record.path)))
    copied.sort(key=lambda asset: rel_to_figure(asset.destination))
    return records, copied


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def finite_or_zero(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def eta_complexity_map() -> dict[float, float]:
    frame = pd.read_csv(require_file(ETA_COMPLEXITY_PATH))
    return {
        round(float(row["eta"]), 10): float(row["complexity_mean"])
        for _, row in frame.dropna(subset=["eta", "complexity_mean"]).iterrows()
    }


def manual_complexity_map() -> dict[str, float]:
    frame = pd.read_csv(require_file(MANUAL_COMPLEXITY_PATH))
    return {
        str(row["rule_name"]): float(row["complexity_mean"])
        for _, row in frame.dropna(subset=["rule_name", "complexity_mean"]).iterrows()
    }


def map_eta_complexity(values: pd.Series) -> pd.Series:
    lookup = eta_complexity_map()
    eta = pd.to_numeric(values, errors="coerce")
    missing = sorted({float(value) for value in eta.dropna() if round(float(value), 10) not in lookup})
    if missing:
        raise ValueError(f"{ETA_COMPLEXITY_PATH} is missing eta values required by PLE phase inputs: {missing}")
    return eta.map(lambda value: lookup[round(float(value), 10)] if pd.notna(value) else np.nan)


def map_manual_complexity(values: pd.Series) -> pd.Series:
    lookup = manual_complexity_map()
    rules = values.astype(str)
    missing = sorted(set(rules.dropna()).difference(lookup))
    if missing:
        raise ValueError(f"{MANUAL_COMPLEXITY_PATH} is missing rules required by PLE phase inputs: {missing}")
    return rules.map(lookup)


def label_noise_curve_frame(name: str, value_col: str, sem_col: str) -> tuple[pd.DataFrame, Path]:
    path = require_file(PLE_INPUT_ROOTS["label_noise_sweep"] / name / f"{name}.csv")
    frame = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "eta": pd.to_numeric(frame["eta"], errors="coerce"),
            "condition": frame["eta"].map(lambda eta: f"eta {float(eta):.2f}"),
            "source": "eta_sweep",
            "radius": pd.to_numeric(frame["radius"], errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
            "sem": pd.to_numeric(frame[sem_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["eta", "radius", "value"]), path


def manual_endpoint_curve_frame(name: str, value_col: str, sem_col: str) -> tuple[pd.DataFrame, Path]:
    path = require_file(PLE_INPUT_ROOTS["manual_rules"] / name / f"{name}.csv")
    frame = pd.read_csv(path)
    frame = frame[frame["rule_name"].isin(ENDPOINT_ETA)].copy()
    out = pd.DataFrame(
        {
            "eta": frame["rule_name"].map(ENDPOINT_ETA),
            "condition": frame["rule_name"].map(ENDPOINT_LABEL),
            "source": "manual_endpoint",
            "radius": pd.to_numeric(frame["radius"], errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
            "sem": pd.to_numeric(frame[sem_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["eta", "radius", "value"]), path


def merged_curve_frame(
    name: str,
    label_value_col: str,
    label_sem_col: str,
    manual_value_col: str,
    manual_sem_col: str,
) -> tuple[pd.DataFrame, tuple[Path, ...]]:
    label_frame, label_path = label_noise_curve_frame(name, label_value_col, label_sem_col)
    manual_frame, manual_path = manual_endpoint_curve_frame(name, manual_value_col, manual_sem_col)
    frame = pd.concat([manual_frame, label_frame], ignore_index=True).sort_values(["eta", "radius"])
    return frame, (label_path, manual_path)


def plot_merged_curves(frame: pd.DataFrame, ylabel: str, title: str, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"no finite rows to plot for {path}")

    fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0.0, 0.5)
    handles = []

    for eta, group in frame.groupby("eta", sort=True):
        group = group.sort_values("radius")
        color = cmap(norm(float(eta)))
        source = str(group["source"].iloc[0])
        label = str(group["condition"].iloc[0])
        linestyle = "-" if source == "manual_endpoint" else "--"
        linewidth = 2.1 if source == "manual_endpoint" else 1.65
        x = group["radius"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)
        line = ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)[0]
        handles.append(line)
        if "sem" in group.columns:
            err = finite_or_zero(group["sem"])
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.11, linewidth=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"aligned $\eta$")
    ax.set_xlabel("distance d")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(handles=handles, frameon=False, fontsize=8.0, ncol=2)
    ax.text(
        0.01,
        0.02,
        "solid: manual endpoints; dashed: label-noise sweep; band: mean +/- SE",
        transform=ax.transAxes,
        fontsize=7.2,
        color="0.35",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def label_phase_frames(name: str) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Path, ...]]:
    phase_path = require_file(PLE_INPUT_ROOTS["label_noise_sweep"] / name / f"{name}.csv")
    curves_path = require_file(PLE_INPUT_ROOTS["label_noise_sweep"] / name / "phase_derivative_curves.csv")
    phase = pd.read_csv(phase_path)
    curves = pd.read_csv(curves_path)

    phase_out = pd.DataFrame(
        {
            "eta": pd.to_numeric(phase["eta"], errors="coerce"),
            "complexity": map_eta_complexity(phase["eta"]),
            "condition": phase["eta"].map(lambda eta: f"eta {float(eta):.2f}"),
            "source": "eta_sweep",
            "A_kappa_mean": pd.to_numeric(phase["A_kappa_mean"], errors="coerce"),
            "A_kappa_sem": pd.to_numeric(phase["A_kappa_sem"], errors="coerce"),
        }
    )
    curves_out = pd.DataFrame(
        {
            "eta": pd.to_numeric(curves["eta"], errors="coerce"),
            "condition": curves["eta"].map(lambda eta: f"eta {float(eta):.2f}"),
            "source": "eta_sweep",
            "radius": pd.to_numeric(curves["radius"], errors="coerce"),
            "value": pd.to_numeric(curves["dphi_dr_smooth_mean"], errors="coerce"),
            "sem": pd.to_numeric(curves["dphi_dr_smooth_sem"], errors="coerce"),
        }
    )
    return (
        phase_out.dropna(subset=["eta", "A_kappa_mean"]),
        curves_out.dropna(subset=["eta", "radius", "value"]),
        (phase_path, curves_path, ETA_COMPLEXITY_PATH),
    )


def manual_endpoint_phase_frames(name: str) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Path, ...]]:
    phase_path = require_file(PLE_INPUT_ROOTS["manual_rules"] / name / f"{name}.csv")
    curves_path = require_file(PLE_INPUT_ROOTS["manual_rules"] / name / "phase_derivative_curves.csv")
    phase = pd.read_csv(phase_path)
    curves = pd.read_csv(curves_path)
    phase = phase[phase["rule_name"].isin(ENDPOINT_ETA)].copy()
    curves = curves[curves["rule_name"].isin(ENDPOINT_ETA)].copy()

    phase_out = pd.DataFrame(
        {
            "eta": phase["rule_name"].map(ENDPOINT_ETA),
            "complexity": map_manual_complexity(phase["rule_name"]),
            "condition": phase["rule_name"].map(ENDPOINT_LABEL),
            "source": "manual_endpoint",
            "A_kappa_mean": pd.to_numeric(phase["A_kappa_mean"], errors="coerce"),
            "A_kappa_sem": pd.to_numeric(phase["A_kappa_sem"], errors="coerce"),
        }
    )
    curves_out = pd.DataFrame(
        {
            "eta": curves["rule_name"].map(ENDPOINT_ETA),
            "condition": curves["rule_name"].map(ENDPOINT_LABEL),
            "source": "manual_endpoint",
            "radius": pd.to_numeric(curves["radius"], errors="coerce"),
            "value": pd.to_numeric(curves["dphi_dr_smooth_mean"], errors="coerce"),
            "sem": pd.to_numeric(curves["dphi_dr_smooth_sem"], errors="coerce"),
        }
    )
    return (
        phase_out.dropna(subset=["eta", "A_kappa_mean"]),
        curves_out.dropna(subset=["eta", "radius", "value"]),
        (phase_path, curves_path, MANUAL_COMPLEXITY_PATH),
    )


def plot_merged_phase(
    phase: pd.DataFrame,
    curves: pd.DataFrame,
    path: Path,
    *,
    x_col: str,
    x_label: str,
    right_title: str,
    eta_xlim: bool = False,
) -> None:
    if phase.empty or curves.empty:
        raise ValueError(f"no finite rows to plot for {path}")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0.0, 0.5)
    handles = []

    for eta, group in curves.groupby("eta", sort=True):
        group = group.sort_values("radius")
        color = cmap(norm(float(eta)))
        source = str(group["source"].iloc[0])
        label = str(group["condition"].iloc[0])
        linestyle = "-" if source == "manual_endpoint" else "--"
        linewidth = 2.1 if source == "manual_endpoint" else 1.55
        x = group["radius"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)
        line = ax_left.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)[0]
        handles.append(line)
        err = finite_or_zero(group["sem"])
        ax_left.fill_between(x, y - err, y + err, color=color, alpha=0.10, linewidth=0)

    ax_left.set_xlabel("distance d")
    ax_left.set_ylabel(r"energetic $d\phi/dd$")
    ax_left.set_title("Energetic derivative")
    ax_left.grid(True, alpha=0.24)
    ax_left.legend(handles=handles, frameon=False, fontsize=8.0, ncol=1)
    ax_left.text(
        0.01,
        0.02,
        "solid endpoints; dashed eta sweep",
        transform=ax_left.transAxes,
        fontsize=7.2,
        color="0.35",
    )

    phase = phase.sort_values(x_col)
    colors = [cmap(norm(float(eta))) for eta in phase["eta"]]
    ax_right.errorbar(
        phase[x_col],
        phase["A_kappa_mean"],
        yerr=finite_or_zero(phase["A_kappa_sem"]),
        fmt="none",
        ecolor="0.35",
        elinewidth=1.0,
        capsize=2.6,
        zorder=1,
    )
    ax_right.scatter(phase[x_col], phase["A_kappa_mean"], c=colors, s=42, zorder=2)
    ax_right.plot(phase[x_col], phase["A_kappa_mean"], color="0.45", linewidth=1.1, alpha=0.7, zorder=0)
    for _, row in phase.iterrows():
        ax_right.annotate(
            str(row["condition"]),
            (row[x_col], row["A_kappa_mean"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    if eta_xlim:
        ax_right.set_xlim(-0.03, 0.53)
    ax_right.set_xlabel(x_label)
    ax_right.set_ylabel("A measure")
    ax_right.set_title(right_title)
    ax_right.grid(True, alpha=0.24)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def build_merged_proxy_local_entropy_figures() -> list[FigureRecord]:
    records: list[FigureRecord] = []

    for name, label_value, label_sem, manual_value, manual_sem, ylabel, title in CURVE_SPECS:
        frame, inputs = merged_curve_frame(name, label_value, label_sem, manual_value, manual_sem)
        output_path = MERGED_PLE_ROOT / f"{name}.png"
        plot_merged_curves(frame, ylabel, title, output_path)
        records.append(
            FigureRecord(
                section="05_proxy_local_entropy",
                series="merged",
                title=f"merged {humanize_part(name)}",
                path=output_path,
                inputs=inputs,
            )
        )

    for output_name, label_input_name, manual_input_name, x_col, x_label, right_title, eta_xlim in (
        (
            "phase_like_A_by_eta",
            "phase_like_A_by_eta",
            "phase_like_A_by_rule",
            "eta",
            r"aligned $\eta$",
            "A_kappa by aligned eta",
            True,
        ),
        (
            "phase_like_A_by_complexity",
            "phase_like_A_by_complexity",
            "phase_like_A_by_complexity",
            "complexity",
            "3-NN MNIST complexity",
            "A_kappa by complexity",
            False,
        ),
    ):
        label_phase, label_curves, label_inputs = label_phase_frames(label_input_name)
        manual_phase, manual_curves, manual_inputs = manual_endpoint_phase_frames(manual_input_name)
        phase = pd.concat([manual_phase, label_phase], ignore_index=True).sort_values(x_col)
        curves = pd.concat([manual_curves, label_curves], ignore_index=True).sort_values(["eta", "radius"])
        output_path = MERGED_PLE_ROOT / f"{output_name}.png"
        plot_merged_phase(
            phase,
            curves,
            output_path,
            x_col=x_col,
            x_label=x_label,
            right_title=right_title,
            eta_xlim=eta_xlim,
        )
        records.append(
            FigureRecord(
                section="05_proxy_local_entropy",
                series="merged",
                title=f"merged {humanize_part(output_name)}",
                path=output_path,
                inputs=label_inputs + manual_inputs,
            )
        )
    return records


def record_to_dict(record: FigureRecord) -> dict[str, Any]:
    return {
        "section": record.section,
        "series": record.series,
        "title": record.title,
        "path": rel_to_dnn(record.path),
        "inputs": [rel_to_dnn(path) for path in record.inputs],
    }


def write_manifest(records: list[FigureRecord], copied: list[CopiedAsset], generated_count: int) -> None:
    payload = {
        "root": rel_to_dnn(FIGURE_ROOT),
        "figure_count": len(records),
        "copied_asset_count": len(copied),
        "generated_figure_count": generated_count,
        "figures": [record_to_dict(record) for record in records],
        "sources": [
            {
                "series": series,
                "root": rel_to_dnn(root),
            }
            for series, root in source_series()
            if root.exists()
        ],
        "policy": "Top-level figures are copied from current stage figure outputs; merged proxy-local-entropy figures are regenerated from summarized outputs.",
    }
    (FIGURE_ROOT / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def grouped_records(records: Iterable[FigureRecord]) -> dict[str, list[FigureRecord]]:
    grouped: dict[str, list[FigureRecord]] = {}
    for record in records:
        grouped.setdefault(record.section, []).append(record)
    return dict(sorted(grouped.items(), key=lambda item: stage_sort_key(item[0])))


def write_markdown_index(records: list[FigureRecord]) -> None:
    lines = [
        "# 03_dnn_mnist figures",
        "",
        "Generated by `figures/src/make_figures.py` from the current stage figure outputs.",
        "",
    ]
    for section, section_records in grouped_records(records).items():
        lines.extend([f"## {STAGE_TITLES.get(section, section)}", ""])
        for record in section_records:
            image_path = rel_to_figure(record.path)
            lines.extend([f"### {record.title}", "", f"![{record.title}]({image_path})", ""])
    (FIGURE_ROOT / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_html_index(records: list[FigureRecord]) -> None:
    blocks = []
    for section, section_records in grouped_records(records).items():
        cards = []
        for record in section_records:
            image_path = rel_to_figure(record.path)
            cards.append(
                f"""
                <article class="card">
                  <h3>{escape(record.title)}</h3>
                  <a href="{escape(image_path)}">
                    <img src="{escape(image_path)}" alt="{escape(record.title)}">
                  </a>
                </article>
                """
            )
        blocks.append(
            f"""
            <section>
              <h2>{escape(STAGE_TITLES.get(section, section))}</h2>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>03_dnn_mnist figures</title>
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
    <h1>03_dnn_mnist figures</h1>
    <p>Generated by <code>figures/src/make_figures.py</code> from current stage figure outputs.</p>
  </header>
  {''.join(blocks)}
</body>
</html>
"""
    (FIGURE_ROOT / "index.html").write_text(html, encoding="utf-8")


def build_all() -> tuple[list[FigureRecord], list[CopiedAsset]]:
    clear_figure_root()
    records, copied = copy_stage_assets()
    merged_records = build_merged_proxy_local_entropy_figures()
    records.extend(merged_records)
    if not records:
        raise RuntimeError("no stage figures found to collect")
    write_manifest(records, copied, generated_count=len(merged_records))
    write_markdown_index(records)
    write_html_index(records)
    return records, copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 03_dnn_mnist top-level figure gallery.")
    parser.parse_args()

    records, copied = build_all()
    print(f"figure_count={len(records)}")
    print(f"copied_asset_count={len(copied)}")
    print(f"generated_figure_count={len(records) - len(copied)}")
    print(f"figure_root={FIGURE_ROOT}")
    print(f"index_md={FIGURE_ROOT / 'index.md'}")
    print(f"index_html={FIGURE_ROOT / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
