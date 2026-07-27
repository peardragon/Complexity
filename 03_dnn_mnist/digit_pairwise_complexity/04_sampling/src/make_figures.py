from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .make_summarized_outputs import DEFAULT_FIGURE_INPUT_ROOT, clear_outputs, pair_sort_key, pair_summary_files
except ImportError:
    from make_summarized_outputs import DEFAULT_FIGURE_INPUT_ROOT, clear_outputs, pair_sort_key, pair_summary_files


SAMPLING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_ROOT = SAMPLING_ROOT / "figures" / "logZ_split_distributions"


def _radius_label(value: float) -> str:
    return f"{value:g}"


def _xtick_labels(radii: list[float]) -> list[str]:
    if len(radii) <= 24:
        return [_radius_label(value) for value in radii]
    step = max(1, int(np.ceil(len(radii) / 14)))
    return [_radius_label(value) if idx % step == 0 or idx == len(radii) - 1 else "" for idx, value in enumerate(radii)]


def plot_logz_split_distribution(input_csv: Path, output_png: Path, *, max_scatter_per_radius: int) -> None:
    frame = pd.read_csv(input_csv)
    frame["r"] = pd.to_numeric(frame["r"], errors="coerce")
    frame["signed_split_logZ_per_P_diff"] = pd.to_numeric(frame["signed_split_logZ_per_P_diff"], errors="coerce")
    frame = frame.dropna(subset=["r", "signed_split_logZ_per_P_diff"]).sort_values("r")
    if frame.empty:
        raise ValueError(f"no finite logZ split rows in {input_csv}")

    pair_id = str(frame["pair_id"].iloc[0])
    pair = str(frame["pair"].iloc[0])
    radii = [float(value) for value in sorted(frame["r"].unique())]
    values = [frame.loc[np.isclose(frame["r"], radius), "signed_split_logZ_per_P_diff"].to_numpy(float) for radius in radii]
    positions = np.arange(len(radii), dtype=float)
    width = max(10.5, min(18.0, 0.34 * len(radii)))
    fig, ax = plt.subplots(figsize=(width, 5.8))

    violin_positions = positions + 0.16
    parts = ax.violinplot(values, positions=violin_positions, widths=0.34, showmeans=False, showmedians=False, showextrema=False)
    for center, body in zip(violin_positions, parts["bodies"]):
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.maximum(vertices[:, 0], center)
        body.set_facecolor("#6f6f6f")
        body.set_edgecolor("none")
        body.set_alpha(0.82)

    rng = np.random.default_rng(1729 + 100 * pair_sort_key(pair_id)[0] + pair_sort_key(pair_id)[1])
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
    ax.set_xticklabels(_xtick_labels(radii), rotation=90, fontsize=7)
    ax.set_xlabel("d")
    ax.set_ylabel("signed split logZ diff per P")
    ax.set_title(f"logZ split distributions, {pair_id}: {pair}")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def render_figures(input_root: Path, output_root: Path, *, max_scatter_per_radius: int) -> list[Path]:
    files = pair_summary_files(input_root)
    if not files:
        raise FileNotFoundError(f"no pair_*.csv files found under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "pair_*.png")
    outputs: list[Path] = []
    for path in files:
        out = output_root / f"{path.stem}.png"
        plot_logz_split_distribution(path, out, max_scatter_per_radius=max_scatter_per_radius)
        outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build digit-pair sampling figures from figure-input CSVs.")
    parser.add_argument("--figure-input-root", "--input-root", dest="figure_input_root", type=Path, default=DEFAULT_FIGURE_INPUT_ROOT)
    parser.add_argument("--figure-root", "--output-root", dest="figure_root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--max-scatter-per-radius", type=int, default=120)
    args = parser.parse_args()
    figure_outputs = render_figures(
        args.figure_input_root.resolve(),
        args.figure_root.resolve(),
        max_scatter_per_radius=int(args.max_scatter_per_radius),
    )
    print(f"figure_files={len(figure_outputs)} root={args.figure_root.resolve()}")


if __name__ == "__main__":
    main()
