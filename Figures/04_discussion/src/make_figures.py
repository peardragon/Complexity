#!/usr/bin/env python3
"""Draw Fig. 9 and companion plots of the published Discussion evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_ROOT = Path(__file__).resolve().parents[1]
ROOT = FIGURE_ROOT.parents[1]
DISCUSSION = ROOT / "04_discussion"
PAPER_SOURCE = ROOT / "Figures/paper_figures/src"
sys.path.insert(0, str(PAPER_SOURCE))
from utils.paper_style import apply_style, output_path, style_axes
from utils.notebook_runner import render_figures

CONFIG = json.loads((FIGURE_ROOT / "config/default.json").read_text())


def paths(name: str) -> list[Path]:
    return [FIGURE_ROOT / name / f"{name}.{suffix}" for suffix in CONFIG["formats"]]


def save(figure, name: str, *, force: bool = False) -> None:
    for path in paths(name):
        if path.is_file() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(figure)


def antipodal(*, force: bool = False) -> None:
    frame = pd.read_csv(DISCUSSION / "03_antipodal_geometry/summarized_outputs/matched_antipodal_profiles.csv")
    figure, axis = plt.subplots(figsize=(3.4, 2.5), layout="constrained")
    for condition, label, color in (("noise_eta_0p00", "Clean labels", "#0072B2"), ("noise_eta_0p50", "Random labels", "#D55E00")):
        rows = frame.loc[frame.condition.eq(condition)].sort_values("radius")
        x, y, se = (rows[key].to_numpy(float) for key in ("radius", "mean_K", "se_K"))
        axis.plot(x, y, "o-", label=label, color=color, markersize=3)
        axis.fill_between(x, y-se, y+se, alpha=0.18, color=color, linewidth=0)
    axis.set(xlabel=r"$r$", ylabel=r"$K_r$", xlim=CONFIG["antipodal_radius_limits"])
    axis.legend(frameon=False)
    style_axes(axis)
    save(figure, "appendix_d_antipodal", force=force)


def reentrance(*, force: bool = False) -> None:
    frame = pd.read_csv(DISCUSSION / "02_normalized_radial_geometry/summarized_outputs/normalized_radial_profiles.csv")
    frame = frame.loc[frame.domain.eq("mnist_label_noise")]
    figure, axis = plt.subplots(figsize=(3.4, 2.5), layout="constrained")
    for color, (condition, rows) in zip(plt.cm.viridis(np.linspace(0.1, 0.9, 5)), frame.groupby("condition", sort=True)):
        rows = rows.sort_values("radius")
        eta = float(condition.removeprefix("noise_eta_").replace("p", "."))
        x, y, se = (rows[key].to_numpy(float) for key in ("radius", "g_mean", "g_se_across_datasets"))
        axis.plot(x, y, label=rf"$\eta={eta:g}$", color=color)
        axis.fill_between(x, y-se, y+se, alpha=0.10, color=color, linewidth=0)
    axis.axhline(0, color="0.4", linewidth=0.6)
    axis.set(xlabel=r"$r$", ylabel=r"$g_{\mathrm{E}}(r)$", xlim=CONFIG["reentrance_radius_limits"])
    axis.legend(frameon=False, fontsize=6)
    style_axes(axis)
    save(figure, "random_label_reentrance", force=force)


def paper_figure(*, force: bool = False) -> None:
    name = "fig09_hardening_3d_composite"
    subprocess.run([sys.executable, str(PAPER_SOURCE / "build_all.py")], cwd=ROOT, check=True)
    if force:
        render_figures([name], force=True)
        from build_all import audit
        audit()
    for path in paths(name):
        source = output_path(name, path.suffix)
        if path.is_file() and path.read_bytes() == source.read_bytes():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Redraw existing outputs.")
    args = parser.parse_args()
    apply_style()
    paper_figure(force=args.force)
    print("fig09_hardening_3d_composite: synced_with_paper_notebook")
    for name, render in (("appendix_d_antipodal", antipodal), ("random_label_reentrance", reentrance)):
        if not args.force and all(path.is_file() for path in paths(name)):
            print(f"{name}: skipped_existing")
            continue
        render(force=args.force)
        print(f"{name}: written")


if __name__ == "__main__":
    main()
