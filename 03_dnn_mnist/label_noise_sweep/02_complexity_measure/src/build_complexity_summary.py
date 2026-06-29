from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
STAGE_ROOT = REPO_ROOT / "03_dnn_mnist" / "label_noise_sweep" / "02_complexity_measure"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs" / "complexity_axis_spin_mnist_30ref_eta0p02_0p05_0p15_0p25"
FIGURE_ROOT = STAGE_ROOT / "figures"
METRICS_CSV = SUMMARY_ROOT / "mnist_complexity_axis_metrics.csv"


def _eta_from_group(group: str) -> float | None:
    if not group.startswith("eta_"):
        return None
    return float(group.removeprefix("eta_").replace("_", "."))


def main() -> None:
    metrics = pd.read_csv(METRICS_CSV)
    eta_rows = metrics[metrics["source"].astype(str).eq("flip")].copy()
    eta_rows["eta"] = eta_rows["group"].astype(str).map(_eta_from_group)
    eta_rows = eta_rows.dropna(subset=["eta"]).sort_values("eta")
    out = eta_rows[
        [
            "eta",
            "group",
            "label",
            "nmstv",
            "complexity_proxy",
            "complexity_norm",
            "A_kappa_savgol21_group_mean",
            "min_dphi_dr_savgol21",
            "min_dphi_dr_radius_savgol21",
            "phi_energy_at_d1",
        ]
    ].rename(columns={"group": "noise_eta"})
    out["noise_eta"] = out["noise_eta"].str.replace("eta_", "noise_eta_", regex=False)
    out.to_csv(SUMMARY_ROOT / "label_noise_complexity_by_eta.csv", index=False)

    manifest = {
        "stage": "02_complexity_measure",
        "complexity_metric": "nmstv / complexity_proxy",
        "figure": "figures/label_noise_complexity_by_eta.png",
        "summary": "summarized_outputs/complexity_axis_spin_mnist_30ref_eta0p02_0p05_0p15_0p25/label_noise_complexity_by_eta.csv",
    }
    (SUMMARY_ROOT / "label_noise_complexity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.plot(out["eta"], out["complexity_proxy"], "o-", color="#2364aa", linewidth=1.5, markersize=5)
    ax.set_xlabel("label noise eta")
    ax.set_ylabel("3-NN MNIST complexity proxy")
    ax.set_title("MNIST label-noise complexity")
    ax.grid(True, alpha=0.25)
    fig.savefig(FIGURE_ROOT / "label_noise_complexity_by_eta.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
