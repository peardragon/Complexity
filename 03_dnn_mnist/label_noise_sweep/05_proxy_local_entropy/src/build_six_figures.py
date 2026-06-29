from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "05_proxy_local_entropy"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_ROOT = STAGE_ROOT / "figures"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
RUN_TAG = "direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
ETA_RUN = (
    REPO_ROOT
    / "03_dnn_mnist"
    / "label_noise_sweep"
    / "04_sampling"
    / "summarized_outputs"
    / RUN_TAG
    / "02_eta_flip_sampling"
)
ETA_RESULTS = ETA_RUN / "06_results_figures"
DIRECT_RUN = SUMMARY_ROOT / RUN_TAG
DERIVATIVE_SOURCE = "sampling_time_direct_radial_score_derivative"


def _write_input(name: str, frame: pd.DataFrame) -> None:
    out_dir = FIGURE_INPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{name}.csv", index=False)


def _plot_eta_curves(frame: pd.DataFrame, value_key: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    etas = np.array(sorted(frame["eta"].dropna().unique()), dtype=float)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(etas.min()), float(etas.max()))
    for eta, sub in frame.groupby("eta", sort=True):
        sub = sub.sort_values("radius")
        ax.plot(sub["radius"], sub[value_key], linewidth=1.7, color=cmap(norm(float(eta))), label=f"eta={eta:.2f}")
    ax.set_xlabel("distance d")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_phase(curves: pd.DataFrame, phase: pd.DataFrame, x_key: str, x_label: str, path: Path) -> None:
    eta_curves = curves[curves["source"].eq("flip")].copy()
    phase = phase[phase["source"].eq("flip")].copy()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    etas = np.array(sorted(eta_curves["eta"].dropna().unique()), dtype=float)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(etas.min()), float(etas.max()))
    for eta, sub in eta_curves.groupby("eta", sort=True):
        sub = sub.sort_values("radius")
        ax_left.plot(sub["radius"], sub["dphi_dr_smooth_mean"], linewidth=1.2, color=cmap(norm(float(eta))))
    ax_left.set_xlabel("distance d")
    ax_left.set_ylabel("energetic dphi/dd")
    ax_left.set_title("Energetic derivative")
    ax_left.grid(True, alpha=0.24)

    phase = phase.sort_values(x_key)
    ax_right.errorbar(
        phase[x_key],
        phase["A_kappa_mean"],
        yerr=phase["A_kappa_sem"],
        marker="o",
        linewidth=1.5,
        capsize=2.5,
        color="#7a3e9d",
    )
    ax_right.set_xlabel(x_label)
    ax_right.set_ylabel("A measure")
    ax_right.set_title("A-measure phase-like plot")
    ax_right.grid(True, alpha=0.24)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    phi = pd.read_csv(ETA_RESULTS / "eta_reference_phi_by_eta_radius.csv")
    dphi = pd.read_csv(ETA_RESULTS / "eta_reference_dphi_dd_by_eta_radius.csv")
    phase = pd.read_csv(DIRECT_RUN / "combined_direct_curvature_metrics_by_group.csv")
    curves = pd.read_csv(DIRECT_RUN / "combined_direct_curvature_curve_by_group_radius.csv")

    phi_input = phi[["eta", "rule", "radius", "n_units", "delta_phi_energy_mean", "phi_energy_raw_mean"]].copy()
    phase_input = phase[phase["source"].eq("flip")][
        ["source", "group", "label", "nmstv", "n_refs", "positive_curvature_mass_mean", "positive_curvature_mass_sem"]
    ].copy()
    phase_input["eta"] = phase_input["group"].str.removeprefix("eta_").str.replace("p", ".", regex=False).astype(float)
    phase_input = phase_input.rename(
        columns={
            "group": "case_id",
            "label": "case_label",
            "positive_curvature_mass_mean": "A_kappa_mean",
            "positive_curvature_mass_sem": "A_kappa_sem",
        }
    )
    phase_input["derivative_source"] = DERIVATIVE_SOURCE
    phase_input = phase_input[
        [
            "source",
            "case_id",
            "case_label",
            "eta",
            "nmstv",
            "n_refs",
            "A_kappa_mean",
            "A_kappa_sem",
            "derivative_source",
        ]
    ].copy()
    curve_input = curves[curves["source"].eq("flip")][
        ["source", "group", "label", "nmstv", "radius", "n_refs", "d_phi_energy_direct_dd_mean", "d_phi_energy_direct_dd_sem"]
    ].copy()
    curve_input["eta"] = curve_input["group"].str.removeprefix("eta_").str.replace("p", ".", regex=False).astype(float)
    curve_input = curve_input.rename(
        columns={
            "group": "case_id",
            "label": "case_label",
            "d_phi_energy_direct_dd_mean": "dphi_dr_smooth_mean",
            "d_phi_energy_direct_dd_sem": "dphi_dr_smooth_sem",
        }
    )
    curve_input["derivative_source"] = DERIVATIVE_SOURCE
    curve_input = curve_input[
        [
            "source",
            "case_id",
            "case_label",
            "eta",
            "nmstv",
            "radius",
            "n_refs",
            "dphi_dr_smooth_mean",
            "dphi_dr_smooth_sem",
            "derivative_source",
        ]
    ].copy()
    dphi_input = dphi[
        [
            "eta",
            "rule",
            "radius",
            "n_units",
            "d_phi_energy_direct_dd_unit_mean",
            "d_phi_energy_direct_dd_unit_sem",
        ]
    ].copy()
    dphi_input["d_delta_phi_energy_dd"] = dphi_input["d_phi_energy_direct_dd_unit_mean"]
    dphi_input["d_delta_phi_energy_dd_sem"] = dphi_input["d_phi_energy_direct_dd_unit_sem"]
    dphi_input["d_phi_energy_direct_dd"] = dphi_input["d_phi_energy_direct_dd_unit_mean"]
    dphi_input["d_phi_energy_direct_dd_sem"] = dphi_input["d_phi_energy_direct_dd_unit_sem"]
    dphi_input["derivative_source"] = DERIVATIVE_SOURCE
    dphi_input = dphi_input[
        [
            "eta",
            "rule",
            "radius",
            "n_units",
            "d_delta_phi_energy_dd",
            "d_delta_phi_energy_dd_sem",
            "d_phi_energy_direct_dd",
            "d_phi_energy_direct_dd_sem",
            "derivative_source",
        ]
    ].copy()

    _write_input("phi_d_curve", phi_input)
    _write_input("phi_energetic_d_curve", phi_input)
    _write_input("derivative_phi_d_curve", dphi_input)
    _write_input("derivative_phi_energetic_d_curve", dphi_input)
    _write_input("phase_like_A_by_eta", phase_input)
    _write_input("phase_like_A_by_complexity", phase_input)
    for name in ["phase_like_A_by_eta", "phase_like_A_by_complexity"]:
        out_dir = FIGURE_INPUT_ROOT / name
        curve_input.to_csv(out_dir / "phase_derivative_curves.csv", index=False)

    _plot_eta_curves(
        phi_input,
        "delta_phi_energy_mean",
        "phi(d) - phi(d0)",
        "MNIST label-noise phi(d)",
        FIGURE_ROOT / "phi_d_curve.png",
    )
    _plot_eta_curves(
        phi_input,
        "phi_energy_raw_mean",
        "energetic phi(d)",
        "MNIST label-noise energetic phi(d)",
        FIGURE_ROOT / "phi_energetic_d_curve.png",
    )
    _plot_eta_curves(
        dphi_input,
        "d_delta_phi_energy_dd",
        "d phi / dd",
        "MNIST label-noise derivative of phi(d)",
        FIGURE_ROOT / "derivative_phi_d_curve.png",
    )
    _plot_eta_curves(
        dphi_input,
        "d_phi_energy_direct_dd",
        "energetic d phi / dd",
        "MNIST label-noise direct energetic derivative",
        FIGURE_ROOT / "derivative_phi_energetic_d_curve.png",
    )
    _plot_phase(curve_input, phase_input, "eta", "label noise eta", FIGURE_ROOT / "phase_like_A_by_eta.png")
    _plot_phase(curve_input, phase_input, "nmstv", "3-NN MNIST complexity", FIGURE_ROOT / "phase_like_A_by_complexity.png")


if __name__ == "__main__":
    main()
