#!/usr/bin/env python3
"""Build legacy-style active-rule figures for the completed advanced run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_active_rule_dataset_examples as dataset_old
import plot_nmstv_raw_phi_spaghetti as phi_old


LOCAL_ROOT = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist")
RUN_ROOT = LOCAL_ROOT / "04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref"

DATASET_RAW_ROOT = (
    LOCAL_ROOT
    / "01_dataset_gen/raw_outputs/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/legacy_style"
)
DATASET_FIG_ROOT = (
    LOCAL_ROOT
    / "01_dataset_gen/figures/active_rule_dataset_representations_very_low_refpool1024_advanced_90ref/legacy_style"
)
PROXY_RAW_ROOT = LOCAL_ROOT / "05_proxy_local_entropy/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref/legacy_style"
PROXY_FIG_ROOT = LOCAL_ROOT / "05_proxy_local_entropy/figures/very_low_tv_spectral_teacher_refpool1024_advanced_90ref/legacy_style"

RULE_ORDER = [
    "very_low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
DEPRECATED_RULES = ["low_tv_spectral_teacher"]
LABELS = {
    "very_low_tv_spectral_teacher": "very low tv",
    "real_even_odd": "even odd",
    "teacher_nn": "teacher nn",
    "random_label": "random",
}


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_advanced_ref_phi() -> pd.DataFrame:
    unit_path = RUN_ROOT / "05_pool2_pm_sais_sampling/shell_summary_by_unit_with_phi_derivatives.csv"
    unit = pd.read_csv(unit_path)
    ref = unit[["rule", "ref_id", "radius", "phi_energy_raw"]].copy()
    ref = ref[ref["rule"].isin(RULE_ORDER)].copy()
    ref["rule"] = pd.Categorical(ref["rule"], RULE_ORDER, ordered=True)
    ref = ref.sort_values(["rule", "ref_id", "radius"]).reset_index(drop=True)
    ref["source_run"] = "advanced_90ref"
    return ref


def load_complexity_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    comp_all = phi_old.load_complexity()
    comp = comp_all[comp_all["rule"].astype(str).isin(RULE_ORDER)].copy()
    deprecated = comp_all[comp_all["rule"].astype(str).isin(DEPRECATED_RULES)].copy()
    comp["label"] = comp["rule"].map({**LABELS, **{"low_tv_spectral_teacher": "low tv"}})
    return comp.reset_index(drop=True), deprecated.reset_index(drop=True)


def draw_advanced_spaghetti(ref: pd.DataFrame, comp: pd.DataFrame, out_path: Path) -> None:
    merged = ref.merge(comp[["rule", "nmstv_mean", "tv_mean"]], on="rule", how="left")
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(comp["nmstv_mean"].min()), float(comp["nmstv_mean"].max()))

    fig = plt.figure(figsize=(13.2, 7.2))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[24, 2.8], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    inline_labels: list[dict[str, object]] = []
    for rule in RULE_ORDER:
        sub = merged[merged["rule"].astype(str).eq(rule)].copy()
        if sub.empty:
            continue
        color = cmap(norm(float(sub["nmstv_mean"].iloc[0])))
        for _ref_id, rsub in sub.groupby("ref_id", sort=False):
            rsub = rsub.sort_values("radius")
            ax.plot(rsub["radius"], rsub["phi_energy_raw"], color=color, linewidth=0.55, alpha=0.13)
        mean = sub.groupby("radius", as_index=False)["phi_energy_raw"].mean().sort_values("radius")
        ax.plot(mean["radius"], mean["phi_energy_raw"], color=color, linewidth=2.4, alpha=0.98)
        inline_labels.append(
            {
                "x": float(mean["radius"].max()),
                "y": float(mean["phi_energy_raw"].iloc[-1]),
                "label": LABELS.get(rule, rule),
                "color": color,
            }
        )

    if inline_labels:
        ordered = sorted(inline_labels, key=lambda item: float(item["y"]))
        min_gap = 0.007
        placed = []
        for item in ordered:
            y = float(item["y"])
            y_label = y if not placed else max(y, float(placed[-1]["y_label"]) + min_gap)
            placed.append({**item, "y_label": y_label})
        top_limit = max(float(item["y"]) for item in inline_labels) + 0.006
        overflow = float(placed[-1]["y_label"]) - top_limit
        if overflow > 0:
            for item in placed:
                item["y_label"] = float(item["y_label"]) - overflow
        for item in placed:
            x = float(item["x"])
            y = float(item["y"])
            y_label = float(item["y_label"])
            color = item["color"]
            if abs(y_label - y) > 1.0e-4:
                ax.plot([x - 0.02, x + 0.02], [y, y_label], color=color, linewidth=0.65, alpha=0.55, clip_on=False)
            ax.text(x + 0.025, y_label, str(item["label"]), color=color, fontsize=9, va="center", clip_on=False)

    ax.set_xlabel("radius d")
    ax.set_ylabel("phi(d) energy = logZ_inf_full / P")
    ax.set_title("Per-reference raw phi(d) energy curves; color encodes dataset NMSTV")
    ax.grid(True, linewidth=0.35, alpha=0.23)
    ax.set_xlim(0.08, 2.78)

    gradient = np.linspace(float(comp["nmstv_mean"].min()), float(comp["nmstv_mean"].max()), 256)[:, None]
    cax.imshow(gradient, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    cax.set_xticks([])
    ticks = np.linspace(0, 255, 5)
    tick_values = np.linspace(float(comp["nmstv_mean"].min()), float(comp["nmstv_mean"].max()), 5)
    cax.set_yticks(ticks)
    cax.set_yticklabels([f"{value:.3f}" for value in tick_values])
    cax.set_title("NMSTV", fontsize=10)
    for _, row in comp.iterrows():
        value = float(row["nmstv_mean"])
        y = (value - norm.vmin) / max(norm.vmax - norm.vmin, 1.0e-12) * 255.0
        cax.plot([0.08, 0.92], [y, y], color="white", linewidth=2.0, solid_capstyle="round")
        cax.plot([0.08, 0.92], [y, y], color="black", linewidth=0.55, solid_capstyle="round")
        cax.text(1.12, y, f"{row['label']} {value:.3f}", va="center", fontsize=8.5, transform=cax.transData)
    cax.set_ylim(0, 255)

    fig.suptitle("Advanced raw phi(d) energy vs rule complexity; low-TV rule deprecated", y=0.985)
    ensure(out_path.parent)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_proxy_legacy_figures() -> list[Path]:
    ensure(PROXY_FIG_ROOT)
    table_dir = ensure(PROXY_RAW_ROOT / "tables")
    ref = load_advanced_ref_phi()
    comp, deprecated = load_complexity_tables()
    ref_out = ref.merge(comp[["rule", "nmstv_mean", "tv_mean"]], on="rule", how="left")
    write_csv(table_dir / "nmstv_raw_phi_energy_by_ref_radius.csv", ref_out)
    write_csv(table_dir / "nmstv_values_for_raw_phi_plot.csv", comp)
    write_csv(table_dir / "deprecated_rules_excluded_from_figures.csv", deprecated)

    phi_rule = pd.read_csv(RUN_ROOT / "06_results_figures/phi_by_rule_radius.csv")
    phi_raw = pd.read_csv(RUN_ROOT / "06_results_figures/phi_raw_by_rule_radius.csv")
    write_csv(PROXY_RAW_ROOT / "phi_energy_by_rule_radius.csv", phi_rule)
    write_csv(PROXY_RAW_ROOT / "phi_raw_by_rule_radius.csv", phi_raw)
    config_src = RUN_ROOT / "06_results_figures/run_config_resolved.json"
    if config_src.exists():
        shutil.copy2(config_src, PROXY_RAW_ROOT / "run_config_resolved.json")

    fig_main = PROXY_FIG_ROOT / "fig_nmstv_raw_phi_energy_spaghetti.png"
    draw_advanced_spaghetti(ref, comp, fig_main)
    fig_alias = PROXY_FIG_ROOT / "fig01_active_rules_raw_phi_energy_spaghetti_deprecated_low.png"
    shutil.copy2(fig_main, fig_alias)
    return [fig_alias, fig_main]


def draw_legacy_tsne(all_data: dict[str, dict[str, object]], nmstv: dict[str, float]) -> list[Path]:
    ensure(DATASET_FIG_ROOT)
    table_dir = ensure(DATASET_RAW_ROOT / "tables")
    first = all_data[RULE_ORDER[0]]
    emb = dataset_old.compute_tsne(np.asarray(first["X_train"], dtype=np.float64), table_dir / "mnist10_train_tsne_embedding.npz")
    emb_table = pd.DataFrame(
        {
            "train_row": np.arange(emb.shape[0]),
            "tsne_1": emb[:, 0],
            "tsne_2": emb[:, 1],
            "digit": np.asarray(first["digit_train"], dtype=np.int16),
        }
    )
    write_csv(table_dir / "mnist10_train_tsne_embedding.csv", emb_table)

    paths: list[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), sharex=True, sharey=True)
    for ax, rule in zip(axes.ravel(), RULE_ORDER):
        dataset_old.draw_tsne_panel(ax, emb, all_data[rule], rule, nmstv.get(rule))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="binary label", loc="center right", frameon=False)
    fig.suptitle("t-SNE view of the shared MNIST10 train images under active label rules", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 0.91, 0.96))
    combined = DATASET_FIG_ROOT / "fig03_active_rule_tsne_label_embedding.png"
    fig.savefig(combined, dpi=190)
    plt.close(fig)
    paths.append(combined)

    for rule in RULE_ORDER:
        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        dataset_old.draw_tsne_panel(ax, emb, all_data[rule], rule, nmstv.get(rule))
        ax.legend(title="binary label", frameon=False, loc="best")
        fig.tight_layout()
        path = DATASET_FIG_ROOT / f"fig_tsne_label_embedding_{rule}.png"
        fig.savefig(path, dpi=190)
        plt.close(fig)
        paths.append(path)
    return paths


def build_dataset_legacy_figures() -> list[Path]:
    ensure(DATASET_FIG_ROOT)
    table_dir = ensure(DATASET_RAW_ROOT / "tables")
    comp, deprecated = load_complexity_tables()
    write_csv(table_dir / "active_rules_for_result_figures.csv", pd.DataFrame({"rule": RULE_ORDER, "status": "active"}))
    write_csv(table_dir / "deprecated_rules_for_result_figures.csv", pd.DataFrame({"rule": DEPRECATED_RULES, "status": "deprecated"}))
    write_csv(table_dir / "nmstv_values_for_raw_phi_plot.csv", comp)
    write_csv(table_dir / "deprecated_rules_excluded_from_figures.csv", deprecated)

    nmstv = {str(row.rule): float(row.nmstv_mean) for row in comp.itertuples(index=False)}
    all_data = {rule: dataset_old.load_rule(rule) for rule in RULE_ORDER}

    paths: list[Path] = []
    for rule, data in all_data.items():
        path = DATASET_FIG_ROOT / f"fig_dataset_label_examples_{rule}.png"
        dataset_old.draw_rule_grid(rule, data, nmstv.get(rule), path)
        paths.append(path)
    combined = DATASET_FIG_ROOT / "fig02_active_rule_dataset_label_examples.png"
    dataset_old.draw_combined_grids(all_data, nmstv, combined)
    paths.append(combined)
    paths.extend(draw_legacy_tsne(all_data, nmstv))
    return paths


def main() -> int:
    proxy_figures = build_proxy_legacy_figures()
    dataset_figures = build_dataset_legacy_figures()
    status = {
        "run_root": str(RUN_ROOT),
        "radius_grid": "advanced_0.05",
        "active_rules": RULE_ORDER,
        "deprecated_rules": DEPRECATED_RULES,
        "proxy_raw_root": str(PROXY_RAW_ROOT),
        "proxy_fig_root": str(PROXY_FIG_ROOT),
        "dataset_raw_root": str(DATASET_RAW_ROOT),
        "dataset_fig_root": str(DATASET_FIG_ROOT),
        "proxy_figures": [str(path) for path in proxy_figures],
        "dataset_figures": [str(path) for path in dataset_figures],
    }
    write_json(PROXY_RAW_ROOT / "LEGACY_STYLE_STATUS.json", status)
    write_json(DATASET_RAW_ROOT / "LEGACY_STYLE_STATUS.json", status)
    for path in proxy_figures + dataset_figures:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
