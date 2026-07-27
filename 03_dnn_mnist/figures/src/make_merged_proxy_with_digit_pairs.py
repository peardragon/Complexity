#!/usr/bin/env python3
"""Regenerate merged proxy-local-entropy figures with digit-pair curves included."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd


DNN_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = DNN_ROOT / "figures" / "05_proxy_local_entropy" / "merged"
LABEL_INPUT_ROOT = DNN_ROOT / "label_noise_sweep" / "05_proxy_local_entropy" / "summarized_outputs" / "figure_inputs"
MANUAL_INPUT_ROOT = DNN_ROOT / "manual_rules" / "05_proxy_local_entropy" / "summarized_outputs" / "figure_inputs"
DIGIT_INPUT_ROOT = DNN_ROOT / "digit_pairwise_complexity_dense" / "05_proxy_local_entropy" / "summarized_outputs" / "figure_inputs"

ENDPOINTS = {
    "real_even_odd": {
        "eta": 0.0,
        "label": "even/odd",
        "order": 0.0,
        "color": "#202124",
    },
    "random_label": {
        "eta": 0.5,
        "label": "random",
        "order": 4.0,
        "color": "#777777",
    },
}
DIGIT_COLORS = {
    "pair_7_9": "#0072B2",
    "pair_3_8": "#56B4E9",
    "pair_8_9": "#009E73",
    "pair_4_5": "#E69F00",
    "pair_3_9": "#D55E00",
    "pair_0_2": "#CC79A7",
    "pair_1_4": "#A6761D",
    "pair_5_7": "#666666",
    "pair_1_5": "#1B9E77",
    "pair_3_6": "#7570B3",
    "pair_0_4": "#E7298A",
    "pair_6_7": "#6F4E7C",
}
ETA_CMAP = plt.get_cmap("viridis")
ETA_NORM = Normalize(0.0, 0.5)

CURVE_SPECS = (
    (
        "phi_d_curve",
        "delta_phi_energy_mean",
        "delta_phi_energy_sem",
        "delta_phi_energy_unit_mean",
        "delta_phi_energy_unit_sem",
        r"$\phi(d)-\phi(d_0)$",
        "Merged MNIST phi(d): eta, endpoints, digit pairs",
    ),
    (
        "phi_energetic_d_curve",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        r"energetic $\phi(d)$",
        "Merged MNIST energetic phi(d): eta, endpoints, digit pairs",
    ),
    (
        "derivative_phi_d_curve",
        "d_delta_phi_energy_dd",
        "d_delta_phi_energy_dd_sem",
        "d_delta_phi_energy_direct_dd_unit_mean",
        "d_delta_phi_energy_direct_dd_unit_sem",
        r"$d\phi/dd$",
        "Merged MNIST derivative of phi(d): eta, endpoints, digit pairs",
    ),
    (
        "derivative_phi_energetic_d_curve",
        "d_phi_energy_direct_dd",
        "d_phi_energy_direct_dd_sem",
        "d_phi_energy_direct_dd_unit_mean",
        "d_phi_energy_direct_dd_unit_sem",
        r"energetic $d\phi/dd$",
        "Merged MNIST energetic derivative: eta, endpoints, digit pairs",
    ),
)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def finite_or_zero(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def eta_color(eta: float) -> str:
    return ETA_CMAP(ETA_NORM(float(eta)))


def label_noise_curve(name: str, value_col: str, sem_col: str) -> pd.DataFrame:
    frame = pd.read_csv(require(LABEL_INPUT_ROOT / name / f"{name}.csv"))
    eta = pd.to_numeric(frame["eta"], errors="coerce")
    return pd.DataFrame(
        {
            "line_id": eta.map(lambda value: f"eta_{float(value):.2f}"),
            "label": eta.map(lambda value: f"eta {float(value):.2f}"),
            "family": "eta_sweep",
            "order": eta.map(lambda value: 1.0 + float(value)),
            "radius": pd.to_numeric(frame["radius"], errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
            "sem": pd.to_numeric(frame[sem_col], errors="coerce"),
            "eta": eta,
            "complexity": pd.to_numeric(frame.get("nmstv"), errors="coerce") if "nmstv" in frame else np.nan,
        }
    ).dropna(subset=["line_id", "radius", "value"])


def manual_endpoint_curve(name: str, value_col: str, sem_col: str) -> pd.DataFrame:
    frame = pd.read_csv(require(MANUAL_INPUT_ROOT / name / f"{name}.csv"))
    frame = frame[frame["rule_name"].isin(ENDPOINTS)].copy()
    return pd.DataFrame(
        {
            "line_id": frame["rule_name"],
            "label": frame["rule_name"].map(lambda value: ENDPOINTS[str(value)]["label"]),
            "family": "manual_endpoint",
            "order": frame["rule_name"].map(lambda value: ENDPOINTS[str(value)]["order"]),
            "radius": pd.to_numeric(frame["radius"], errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
            "sem": pd.to_numeric(frame[sem_col], errors="coerce"),
            "eta": frame["rule_name"].map(lambda value: ENDPOINTS[str(value)]["eta"]),
            "complexity": pd.to_numeric(frame.get("nmstv_mean"), errors="coerce"),
        }
    ).dropna(subset=["line_id", "radius", "value"])


def digit_pair_curve(name: str, value_col: str, sem_col: str) -> pd.DataFrame:
    frame = pd.read_csv(require(DIGIT_INPUT_ROOT / name / f"{name}.csv"))
    pair_order = pd.to_numeric(frame["pair_order"], errors="coerce")
    return pd.DataFrame(
        {
            "line_id": frame["pair_id"],
            "label": frame["pair_label"].map(lambda value: f"digit {value}"),
            "family": "digit_pair",
            "order": 5.0 + pair_order / 10.0,
            "radius": pd.to_numeric(frame["radius"], errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
            "sem": pd.to_numeric(frame[sem_col], errors="coerce"),
            "eta": np.nan,
            "complexity": pd.to_numeric(frame["complexity_mean"], errors="coerce"),
        }
    ).dropna(subset=["line_id", "radius", "value"])


def merged_curve(name: str, label_value: str, label_sem: str, shared_value: str, shared_sem: str) -> pd.DataFrame:
    frame = pd.concat(
        [
            manual_endpoint_curve(name, shared_value, shared_sem),
            label_noise_curve(name, label_value, label_sem),
            digit_pair_curve(name, shared_value, shared_sem),
        ],
        ignore_index=True,
    )
    return frame.sort_values(["order", "radius"]).reset_index(drop=True)


def style_for(line_id: str, family: str, eta: float | None) -> dict[str, Any]:
    if family == "manual_endpoint":
        return {
            "color": ENDPOINTS[line_id]["color"],
            "linestyle": "-",
            "linewidth": 2.25,
            "alpha": 1.0,
        }
    if family == "eta_sweep":
        return {
            "color": eta_color(float(eta)),
            "linestyle": "--",
            "linewidth": 1.8,
            "alpha": 0.96,
        }
    return {
        "color": DIGIT_COLORS.get(line_id, "#4C78A8"),
        "linestyle": "-",
        "linewidth": 2.0,
        "alpha": 0.98,
    }


def plot_curve(frame: pd.DataFrame, ylabel: str, title: str, path: Path, ax: Any | None = None) -> Any:
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    else:
        fig = ax.figure
    handles = []
    labels = []
    for line_id, group in frame.groupby("line_id", sort=False):
        group = group.sort_values("radius")
        family = str(group["family"].iloc[0])
        eta = group["eta"].iloc[0] if "eta" in group else np.nan
        style = style_for(str(line_id), family, float(eta) if pd.notna(eta) else None)
        x = group["radius"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)
        line = ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            label=str(group["label"].iloc[0]),
        )[0]
        handles.append(line)
        labels.append(str(group["label"].iloc[0]))
        err = finite_or_zero(group["sem"])
        ax.fill_between(x, y - err, y + err, color=style["color"], alpha=0.055, linewidth=0)
    ax.set_xlabel("distance d")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    if owns_fig:
        ax.legend(handles, labels, frameon=False, fontsize=8.0, ncol=2)
        ax.text(
            0.01,
            0.02,
            "black/gray: endpoints; dashed: eta sweep; colored solid: digit pairs; band: mean +/- SE",
            transform=ax.transAxes,
            fontsize=7.2,
            color="0.35",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=240)
        plt.close(fig)
    return handles, labels


def load_phase_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    label_phase = pd.read_csv(require(LABEL_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_like_A_by_complexity.csv"))
    label_curves = pd.read_csv(require(LABEL_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_derivative_curves.csv"))
    manual_phase = pd.read_csv(require(MANUAL_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_like_A_by_complexity.csv"))
    manual_curves = pd.read_csv(require(MANUAL_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_derivative_curves.csv"))
    digit_phase = pd.read_csv(require(DIGIT_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_like_A_by_complexity.csv"))
    digit_curves = pd.read_csv(require(DIGIT_INPUT_ROOT / "phase_like_A_by_complexity" / "phase_derivative_curves.csv"))

    manual_phase = manual_phase[manual_phase["rule_name"].isin(ENDPOINTS)].copy()
    manual_curves = manual_curves[manual_curves["rule_name"].isin(ENDPOINTS)].copy()

    phase = pd.concat(
        [
            pd.DataFrame(
                {
                    "line_id": manual_phase["rule_name"],
                    "label": manual_phase["rule_name"].map(lambda value: ENDPOINTS[str(value)]["label"]),
                    "family": "manual_endpoint",
                    "order": manual_phase["rule_name"].map(lambda value: ENDPOINTS[str(value)]["order"]),
                    "complexity": pd.to_numeric(manual_phase["nmstv_mean"], errors="coerce"),
                    "A_kappa_mean": pd.to_numeric(manual_phase["A_kappa_mean"], errors="coerce"),
                    "A_kappa_sem": pd.to_numeric(manual_phase["A_kappa_sem"], errors="coerce"),
                    "eta": manual_phase["rule_name"].map(lambda value: ENDPOINTS[str(value)]["eta"]),
                }
            ),
            pd.DataFrame(
                {
                    "line_id": pd.to_numeric(label_phase["eta"], errors="coerce").map(lambda value: f"eta_{float(value):.2f}"),
                    "label": pd.to_numeric(label_phase["eta"], errors="coerce").map(lambda value: f"eta {float(value):.2f}"),
                    "family": "eta_sweep",
                    "order": pd.to_numeric(label_phase["eta"], errors="coerce").map(lambda value: 1.0 + float(value)),
                    "complexity": pd.to_numeric(label_phase["nmstv"], errors="coerce"),
                    "A_kappa_mean": pd.to_numeric(label_phase["A_kappa_mean"], errors="coerce"),
                    "A_kappa_sem": pd.to_numeric(label_phase["A_kappa_sem"], errors="coerce"),
                    "eta": pd.to_numeric(label_phase["eta"], errors="coerce"),
                }
            ),
            pd.DataFrame(
                {
                    "line_id": digit_phase["pair_id"],
                    "label": digit_phase["pair_label"].map(lambda value: f"digit {value}"),
                    "family": "digit_pair",
                    "order": 5.0 + pd.to_numeric(digit_phase["pair_order"], errors="coerce") / 10.0,
                    "complexity": pd.to_numeric(digit_phase["complexity_mean"], errors="coerce"),
                    "A_kappa_mean": pd.to_numeric(digit_phase["A_kappa_mean"], errors="coerce"),
                    "A_kappa_sem": pd.to_numeric(digit_phase["A_kappa_sem"], errors="coerce"),
                    "eta": np.nan,
                }
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["line_id", "complexity", "A_kappa_mean"])

    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "line_id": manual_curves["rule_name"],
                    "label": manual_curves["rule_name"].map(lambda value: ENDPOINTS[str(value)]["label"]),
                    "family": "manual_endpoint",
                    "order": manual_curves["rule_name"].map(lambda value: ENDPOINTS[str(value)]["order"]),
                    "radius": pd.to_numeric(manual_curves["radius"], errors="coerce"),
                    "value": pd.to_numeric(manual_curves["dphi_dr_smooth_mean"], errors="coerce"),
                    "sem": pd.to_numeric(manual_curves["dphi_dr_smooth_sem"], errors="coerce"),
                    "eta": manual_curves["rule_name"].map(lambda value: ENDPOINTS[str(value)]["eta"]),
                }
            ),
            pd.DataFrame(
                {
                    "line_id": pd.to_numeric(label_curves["eta"], errors="coerce").map(lambda value: f"eta_{float(value):.2f}"),
                    "label": pd.to_numeric(label_curves["eta"], errors="coerce").map(lambda value: f"eta {float(value):.2f}"),
                    "family": "eta_sweep",
                    "order": pd.to_numeric(label_curves["eta"], errors="coerce").map(lambda value: 1.0 + float(value)),
                    "radius": pd.to_numeric(label_curves["radius"], errors="coerce"),
                    "value": pd.to_numeric(label_curves["dphi_dr_smooth_mean"], errors="coerce"),
                    "sem": pd.to_numeric(label_curves["dphi_dr_smooth_sem"], errors="coerce"),
                    "eta": pd.to_numeric(label_curves["eta"], errors="coerce"),
                }
            ),
            pd.DataFrame(
                {
                    "line_id": digit_curves["pair_id"],
                    "label": digit_curves["pair_label"].map(lambda value: f"digit {value}"),
                    "family": "digit_pair",
                    "order": 5.0 + pd.to_numeric(digit_curves["pair_order"], errors="coerce") / 10.0,
                    "radius": pd.to_numeric(digit_curves["radius"], errors="coerce"),
                    "value": pd.to_numeric(digit_curves["dphi_dr_smooth_mean"], errors="coerce"),
                    "sem": pd.to_numeric(digit_curves["dphi_dr_smooth_sem"], errors="coerce"),
                    "eta": np.nan,
                }
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["line_id", "radius", "value"])

    return phase.sort_values("order"), curves.sort_values(["order", "radius"])


def digit_label_offset(line_id: str) -> tuple[int, int]:
    offsets = {
        "pair_6_7": (5, -10),
        "pair_0_4": (5, 2),
        "pair_3_6": (5, 12),
        "pair_1_5": (5, 21),
        "pair_5_7": (18, 1),
        "pair_1_4": (8, -14),
        "pair_0_2": (13, 10),
        "pair_3_9": (8, -12),
        "pair_4_5": (8, 5),
        "pair_8_9": (8, 6),
        "pair_3_8": (-36, -3),
        "pair_7_9": (-34, 4),
    }
    return offsets.get(line_id, (6, 4))


def plot_phase_points(
    ax: Any,
    phase: pd.DataFrame,
    *,
    annotate_digits: bool,
    annotate_non_digits: bool,
    digit_offsets: bool = False,
    marker_size: float = 4.8,
    annotation_size: float = 7.2,
    connect: bool = True,
) -> None:
    phase = phase.sort_values("complexity")
    if connect:
        ax.plot(phase["complexity"], phase["A_kappa_mean"], color="0.58", linewidth=1.0, alpha=0.55, zorder=0)
    for _, row in phase.iterrows():
        is_digit = str(row["family"]) == "digit_pair"
        style = style_for(str(row["line_id"]), str(row["family"]), float(row["eta"]) if pd.notna(row["eta"]) else None)
        ax.errorbar(
            [row["complexity"]],
            [row["A_kappa_mean"]],
            yerr=[0.0 if pd.isna(row["A_kappa_sem"]) else float(row["A_kappa_sem"])],
            marker="o",
            markersize=marker_size,
            color=style["color"],
            ecolor=style["color"],
            capsize=2.4,
            linestyle="none",
        )
        if (is_digit and not annotate_digits) or ((not is_digit) and not annotate_non_digits):
            continue
        xytext = digit_label_offset(str(row["line_id"])) if is_digit and digit_offsets else (5, 4)
        ax.annotate(
            str(row["label"]),
            (row["complexity"], row["A_kappa_mean"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=annotation_size,
            arrowprops={"arrowstyle": "-", "linewidth": 0.45, "color": "0.45", "alpha": 0.70} if is_digit and digit_offsets else None,
        )


def add_digit_complexity_inset(ax_right: Any, phase: pd.DataFrame) -> None:
    digit = phase[phase["family"] == "digit_pair"].sort_values("complexity").copy()
    if digit.empty:
        return
    x = pd.to_numeric(digit["complexity"], errors="coerce")
    y = pd.to_numeric(digit["A_kappa_mean"], errors="coerce")
    x_pad = max(0.006, 0.08 * float(x.max() - x.min()))
    y_pad = max(0.010, 0.10 * float(y.max() - y.min()))
    axins = inset_axes(ax_right, width="50%", height="48%", loc="lower right", borderpad=1.05)
    plot_phase_points(
        axins,
        digit,
        annotate_digits=True,
        annotate_non_digits=False,
        digit_offsets=True,
        marker_size=4.0,
        annotation_size=6.4,
        connect=True,
    )
    axins.set_xlim(max(0.0, float(x.min()) - x_pad), float(x.max()) + x_pad)
    axins.set_ylim(max(0.0, float(y.min()) - y_pad), float(y.max()) + y_pad)
    axins.set_title("digit-pair zoom", fontsize=8.0)
    axins.set_xlabel("complexity", fontsize=7.0)
    axins.set_ylabel("A", fontsize=7.0)
    axins.grid(True, alpha=0.20)
    axins.tick_params(axis="both", labelsize=6.6, pad=1.0)
    mark_inset(ax_right, axins, loc1=2, loc2=4, fc="none", ec="0.48", linewidth=0.8, alpha=0.75)


def plot_phase_complexity_panel(ax_right: Any, phase: pd.DataFrame, *, add_inset: bool) -> None:
    plot_phase_points(ax_right, phase, annotate_digits=False, annotate_non_digits=True)
    ax_right.set_xlabel("3-NN MNIST complexity")
    ax_right.set_ylabel("A measure")
    ax_right.set_title("A-measure by complexity")
    ax_right.grid(True, alpha=0.24)
    digit = phase[phase["family"] == "digit_pair"]
    if not digit.empty:
        ax_right.axvspan(
            0.0,
            float(pd.to_numeric(digit["complexity"], errors="coerce").max()) * 1.08,
            color="0.2",
            alpha=0.045,
            linewidth=0,
            zorder=-1,
        )
    if add_inset:
        add_digit_complexity_inset(ax_right, phase)


def plot_phase_by_complexity(path: Path) -> None:
    phase, curves = load_phase_frames()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    handles, labels = plot_curve(curves, r"energetic $d\phi/dd$", "Energetic derivative", path, ax=ax_left)
    ax_left.legend(handles, labels, frameon=False, fontsize=7.5, ncol=2)
    plot_phase_complexity_panel(ax_right, phase, add_inset=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_phase_complexity_standalone(path: Path) -> None:
    phase, _curves = load_phase_frames()
    fig, ax = plt.subplots(figsize=(8.4, 5.9), constrained_layout=True)
    plot_phase_complexity_panel(ax, phase, add_inset=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_four_panel(frames: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    legend_handles = None
    legend_labels = None
    for ax, (name, _label_value, _label_sem, _shared_value, _shared_sem, ylabel, title) in zip(axes.ravel(), CURVE_SPECS):
        handles, labels = plot_curve(frames[name], ylabel, title.replace("Merged MNIST ", ""), path, ax=ax)
        legend_handles, legend_labels = handles, labels
    if legend_handles is not None and legend_labels is not None:
        fig.legend(legend_handles, legend_labels, loc="outside lower center", ncol=5, frameon=False, fontsize=8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for name, label_value, label_sem, shared_value, shared_sem, ylabel, title in CURVE_SPECS:
        frame = merged_curve(name, label_value, label_sem, shared_value, shared_sem)
        line_count = frame["line_id"].nunique()
        digit_count = digit_pair_curve(name, shared_value, shared_sem)["line_id"].nunique()
        eta_count = label_noise_curve(name, label_value, label_sem)["line_id"].nunique()
        expected_lines = len(ENDPOINTS) + eta_count + digit_count
        if line_count != expected_lines:
            raise RuntimeError(f"{name} expected {expected_lines} merged lines, got {line_count}")
        frames[name] = frame
        plot_curve(frame, ylabel, title, OUTPUT_ROOT / f"{name}.png")
    representative_count = next(iter(frames.values()))["line_id"].nunique()
    plot_four_panel(frames, OUTPUT_ROOT / f"proxy_curves_dense12_{representative_count}line.png")
    plot_phase_by_complexity(OUTPUT_ROOT / "phase_like_A_by_complexity.png")
    plot_phase_complexity_standalone(OUTPUT_ROOT / "phase_like_A_by_complexity_inset.png")
    print(f"output_root={OUTPUT_ROOT}")
    for name, frame in frames.items():
        print(f"{name}: rows={len(frame)} lines={frame['line_id'].nunique()}")
    print("phase_like_A_by_complexity: regenerated with digit-pair lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
