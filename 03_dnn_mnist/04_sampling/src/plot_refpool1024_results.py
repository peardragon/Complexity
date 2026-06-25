from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "raw_outputs"
RUNS = {
    "60ref": RAW_ROOT / "refpool1024_all_radii_60ref",
    "90ref": RAW_ROOT / "refpool1024_all_radii_90ref",
}
FINAL_TAG = "90ref"
RULE_ORDER = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
RULE_LABELS = {
    "low_tv_spectral_teacher": "low tv spectral teacher",
    "real_even_odd": "real even/odd",
    "teacher_nn": "teacher nn",
    "random_label": "random label",
}
COLORS = {
    "low_tv_spectral_teacher": "#0072B2",
    "real_even_odd": "#009E73",
    "teacher_nn": "#D55E00",
    "random_label": "#CC79A7",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def sorted_rules(df: pd.DataFrame) -> list[str]:
    available = set(str(rule) for rule in df["rule"].unique())
    ordered = [rule for rule in RULE_ORDER if rule in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def load_run(tag: str) -> dict[str, pd.DataFrame | Path]:
    run_root = RUNS[tag]
    results_dir = run_root / "06_results_figures"
    sampling_dir = run_root / "05_pool2_pm_sais_sampling"
    phi = read_csv_required(results_dir / "phi_by_rule_radius.csv")
    summary = read_csv_required(sampling_dir / "shell_summary_by_rule_radius.csv")
    qc = read_csv_required(sampling_dir / "qc_diagnostics_by_rule_radius.csv")
    unit_path = sampling_dir / "shell_summary_by_unit_with_phi.csv"
    if not unit_path.exists():
        unit_path = sampling_dir / "shell_summary_by_unit.csv"
    unit = read_csv_required(unit_path)

    summary_cols = [
        "rule",
        "radius",
        "target_ref_count",
        "observed_ref_count",
        "finite_unit_fraction",
        "q05_ess_fraction",
        "max_split_logZ_per_P_diff",
        "bootstrap_sd_delta_phi_energy",
        "qc_diagnostic_pass",
        "sampling_status",
    ]
    merged = phi.drop(columns=[c for c in ["qc_diagnostic_pass", "sampling_status"] if c in phi.columns]).merge(
        summary[summary_cols],
        on=["rule", "radius"],
        how="left",
    )
    merged["run"] = tag
    qc["run"] = tag
    unit["run"] = tag
    return {
        "root": run_root,
        "results_dir": results_dir,
        "sampling_dir": sampling_dir,
        "phi": merged,
        "qc": qc,
        "unit": unit,
    }


def plot_main_curves(phi: pd.DataFrame, out: Path, field: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for rule in sorted_rules(phi):
        sub = phi[phi["rule"] == rule].sort_values("radius")
        color = COLORS.get(rule)
        ax.plot(
            sub["radius"],
            sub[field],
            color=color,
            linewidth=2.0,
            label=RULE_LABELS.get(rule, rule),
        )
        if field == "delta_phi_energy" and "bootstrap_sd_delta_phi_energy" in sub:
            sd = sub["bootstrap_sd_delta_phi_energy"].to_numpy(dtype=float)
            y = sub[field].to_numpy(dtype=float)
            ax.fill_between(sub["radius"], y - sd, y + sd, color=color, alpha=0.13, linewidth=0)
        pass_sub = sub[sub["qc_diagnostic_pass"].astype(bool)]
        fail_sub = sub[~sub["qc_diagnostic_pass"].astype(bool)]
        if not pass_sub.empty:
            ax.scatter(pass_sub["radius"], pass_sub[field], color=color, s=34, marker="o", zorder=3)
        if not fail_sub.empty:
            ax.scatter(
                fail_sub["radius"],
                fail_sub[field],
                color=color,
                s=30,
                marker="x",
                linewidth=1.0,
                alpha=0.72,
                zorder=3,
            )
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
    ax.set_xlabel("radius d")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.45, alpha=0.28)
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)


def plot_rule_panels(phi: pd.DataFrame, out: Path) -> None:
    rules = sorted_rules(phi)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.3), sharex=True)
    for ax, rule in zip(axes.ravel(), rules):
        sub = phi[phi["rule"] == rule].sort_values("radius")
        color = COLORS.get(rule)
        y = sub["delta_phi_energy"].to_numpy(dtype=float)
        sd = sub["bootstrap_sd_delta_phi_energy"].to_numpy(dtype=float)
        ax.plot(sub["radius"], y, color=color, linewidth=2.0)
        ax.fill_between(sub["radius"], y - sd, y + sd, color=color, alpha=0.16, linewidth=0)
        pass_sub = sub[sub["qc_diagnostic_pass"].astype(bool)]
        fail_sub = sub[~sub["qc_diagnostic_pass"].astype(bool)]
        ax.scatter(pass_sub["radius"], pass_sub["delta_phi_energy"], color=color, s=32, marker="o", zorder=3)
        ax.scatter(
            fail_sub["radius"],
            fail_sub["delta_phi_energy"],
            color=color,
            s=28,
            marker="x",
            linewidth=1.0,
            alpha=0.72,
            zorder=3,
        )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.32)
        ax.set_title(RULE_LABELS.get(rule, rule))
        ax.set_xlabel("radius d")
        ax.set_ylabel("delta phi energy")
        ax.grid(True, linewidth=0.4, alpha=0.25)
    fig.suptitle("90ref delta phi energy by rule; circles=QC pass, x=QC diagnostic fail", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)


def plot_qc_heatmap(qc: pd.DataFrame, out: Path) -> None:
    pivot = (
        qc.pivot(index="rule", columns="radius", values="qc_diagnostic_pass")
        .reindex(sorted_rules(qc))
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(11.5, 3.3))
    cmap = ListedColormap(["#F2F2F2", "#009E73"])
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    xticks = np.arange(len(pivot.columns))
    keep = [idx for idx in xticks if idx % 2 == 0 or idx == len(xticks) - 1]
    ax.set_xticks(keep)
    ax.set_xticklabels([f"{float(pivot.columns[idx]):.1f}" for idx in keep], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([RULE_LABELS.get(str(rule), str(rule)) for rule in pivot.index])
    ax.set_xlabel("radius d")
    ax.set_title("90ref QC diagnostic pass map")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], fraction=0.045, pad=0.02)
    cbar.ax.set_yticklabels(["fail", "pass"])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_qc_metrics(qc: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("q05_ess_fraction", "q05 ESS fraction"),
        ("max_split_logZ_per_P_diff", "max split logZ/P diff"),
        ("bootstrap_sd_delta_phi_energy", "bootstrap sd delta phi energy"),
        ("finite_unit_fraction", "finite unit fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), sharex=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for rule in sorted_rules(qc):
            sub = qc[qc["rule"] == rule].sort_values("radius")
            ax.plot(
                sub["radius"],
                sub[metric],
                color=COLORS.get(rule),
                linewidth=1.75,
                label=RULE_LABELS.get(rule, rule),
            )
        ax.set_xlabel("radius d")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.4, alpha=0.25)
    axes.ravel()[0].legend(fontsize=8, ncol=2)
    fig.suptitle("90ref sampling QC diagnostics", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=185)
    plt.close(fig)


def plot_reference_spaghetti(unit: pd.DataFrame, phi: pd.DataFrame, out: Path) -> None:
    if "delta_phi_energy_unit" not in unit.columns:
        raise ValueError("shell_summary_by_unit_with_phi.csv is required for reference spaghetti plot")
    rules = sorted_rules(unit)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    for ax, rule in zip(axes.ravel(), rules):
        sub = unit[unit["rule"] == rule].sort_values(["ref_id", "radius"])
        color = COLORS.get(rule)
        for _, ref_sub in sub.groupby("ref_id", sort=False):
            ax.plot(
                ref_sub["radius"],
                ref_sub["delta_phi_energy_unit"],
                color=color,
                linewidth=0.55,
                alpha=0.11,
            )
        mean_sub = phi[phi["rule"] == rule].sort_values("radius")
        ax.plot(mean_sub["radius"], mean_sub["delta_phi_energy"], color="black", linewidth=2.15, label="mean")
        pass_sub = mean_sub[mean_sub["qc_diagnostic_pass"].astype(bool)]
        ax.scatter(pass_sub["radius"], pass_sub["delta_phi_energy"], color="black", s=24, marker="o", zorder=3)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.28)
        ax.set_title(RULE_LABELS.get(rule, rule))
        ax.set_xlabel("radius d")
        ax.set_ylabel("per-reference delta phi energy")
        ax.grid(True, linewidth=0.35, alpha=0.22)
    fig.suptitle("90ref per-reference delta phi energy curves; black=reference mean", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=185)
    plt.close(fig)


def derivative_df(phi: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for rule, sub in phi.sort_values("radius").groupby("rule", sort=False):
        x = sub["radius"].to_numpy(dtype=float)
        y = sub["delta_phi_energy"].to_numpy(dtype=float)
        dy = np.gradient(y, x)
        for idx, row in enumerate(sub.to_dict("records")):
            rows.append(
                {
                    "rule": str(rule),
                    "radius": float(row["radius"]),
                    "d_delta_phi_energy_dd": float(dy[idx]),
                    "qc_diagnostic_pass": bool(row["qc_diagnostic_pass"]),
                }
            )
    return pd.DataFrame(rows)


def plot_derivative(deriv: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for rule in sorted_rules(deriv):
        sub = deriv[deriv["rule"] == rule].sort_values("radius")
        ax.plot(
            sub["radius"],
            sub["d_delta_phi_energy_dd"],
            color=COLORS.get(rule),
            linewidth=2.0,
            label=RULE_LABELS.get(rule, rule),
        )
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
    ax.set_xlabel("radius d")
    ax.set_ylabel("d(delta phi energy) / dd")
    ax.set_title("90ref numerical derivative of delta phi energy")
    ax.grid(True, linewidth=0.45, alpha=0.28)
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)


def plot_60_vs_90(phi60: pd.DataFrame, phi90: pd.DataFrame, out: Path) -> pd.DataFrame:
    cmp = pd.concat([phi60, phi90], ignore_index=True)
    rules = sorted_rules(cmp)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.3), sharex=True)
    style = {"60ref": ("--", 1.55), "90ref": ("-", 2.15)}
    for ax, rule in zip(axes.ravel(), rules):
        for run in ["60ref", "90ref"]:
            sub = cmp[(cmp["rule"] == rule) & (cmp["run"] == run)].sort_values("radius")
            linestyle, linewidth = style[run]
            ax.plot(
                sub["radius"],
                sub["delta_phi_energy"],
                color=COLORS.get(rule),
                linestyle=linestyle,
                linewidth=linewidth,
                label=run,
            )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.32)
        ax.set_title(RULE_LABELS.get(rule, rule))
        ax.set_xlabel("radius d")
        ax.set_ylabel("delta phi energy")
        ax.grid(True, linewidth=0.4, alpha=0.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("60ref vs 90ref delta phi energy", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return cmp


def write_report(out_dir: Path, figure_paths: list[Path], derived_paths: list[Path], phi: pd.DataFrame) -> None:
    qc_counts = (
        phi.groupby("rule")["qc_diagnostic_pass"]
        .agg(["sum", "count"])
        .reset_index()
        .assign(rule=lambda df: df["rule"].map(lambda r: RULE_LABELS.get(str(r), str(r))))
    )
    lines = [
        "# Refpool1024 result figures",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Source run: refpool1024_all_radii_90ref",
        "",
        "## Figures",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- figures/{path.name}")
    lines.extend(["", "## Derived tables", ""])
    for path in derived_paths:
        lines.append(f"- derived/{path.name}")
    lines.extend(["", "## QC pass counts", "", "| rule | pass | total |", "| --- | ---: | ---: |"])
    for row in qc_counts.to_dict("records"):
        lines.append(f"| {row['rule']} | {int(row['sum'])} | {int(row['count'])} |")
    lines.append("")
    (out_dir / "FIGURE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    loaded = {tag: load_run(tag) for tag in RUNS}
    final = loaded[FINAL_TAG]
    results_dir = final["results_dir"]  # type: ignore[assignment]
    out_dir = Path(results_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    derived_dir = ensure_dir(out_dir / "derived")
    phi90 = final["phi"]  # type: ignore[assignment]
    qc90 = final["qc"]  # type: ignore[assignment]
    unit90 = final["unit"]  # type: ignore[assignment]
    phi60 = loaded["60ref"]["phi"]  # type: ignore[index]

    assert isinstance(phi90, pd.DataFrame)
    assert isinstance(qc90, pd.DataFrame)
    assert isinstance(unit90, pd.DataFrame)
    assert isinstance(phi60, pd.DataFrame)

    deriv90 = derivative_df(phi90)
    comparison = plot_60_vs_90(phi60, phi90, fig_dir / "fig06_60ref_vs_90ref_delta_phi_energy.png")

    derived_paths = [
        derived_dir / "d_delta_phi_energy_dd_90ref.csv",
        derived_dir / "phi_energy_60ref_vs_90ref.csv",
    ]
    deriv90.to_csv(derived_paths[0], index=False)
    comparison.to_csv(derived_paths[1], index=False)

    figure_paths = [
        fig_dir / "fig01_90ref_delta_phi_energy_raw_qc.png",
        fig_dir / "fig02_90ref_delta_phi_full_raw_qc.png",
        fig_dir / "fig03_90ref_rule_panels_delta_phi_energy.png",
        fig_dir / "fig04_90ref_qc_diagnostic_heatmap.png",
        fig_dir / "fig05_90ref_reference_delta_phi_spaghetti.png",
        fig_dir / "fig06_60ref_vs_90ref_delta_phi_energy.png",
        fig_dir / "fig07_90ref_qc_metrics.png",
        fig_dir / "fig08_90ref_d_delta_phi_energy_dd.png",
    ]
    plot_main_curves(
        phi90,
        figure_paths[0],
        "delta_phi_energy",
        "delta phi energy",
        "90ref raw delta phi energy; circles=QC pass, x=QC diagnostic fail",
    )
    plot_main_curves(
        phi90,
        figure_paths[1],
        "delta_phi_full",
        "delta phi full",
        "90ref raw delta phi full; circles=QC pass, x=QC diagnostic fail",
    )
    plot_rule_panels(phi90, figure_paths[2])
    plot_qc_heatmap(qc90, figure_paths[3])
    plot_reference_spaghetti(unit90, phi90, figure_paths[4])
    plot_qc_metrics(qc90, figure_paths[6])
    plot_derivative(deriv90, figure_paths[7])
    write_report(out_dir, figure_paths, derived_paths, phi90)

    print(f"wrote {len(figure_paths)} figures to {fig_dir}")
    for path in figure_paths:
        print(path)
    print(f"wrote report to {out_dir / 'FIGURE_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
