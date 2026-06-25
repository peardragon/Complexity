from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RUN_ROOT = Path(
    "/home/bjyong/Complexity/local_project/03_dnn_mnist/04_sampling/raw_outputs/refpool1024_all_radii_90ref"
)
UNIT_PATH = RUN_ROOT / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit_with_phi.csv"
OUT_ROOT = RUN_ROOT / "06_results_figures" / "within_rule_phi_curve_clustering"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"

RULES = [
    "low_tv_spectral_teacher",
    "real_even_odd",
    "teacher_nn",
    "random_label",
]
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_k(x: np.ndarray, seed: int) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, float | int | bool]] = []
    max_k = min(8, x.shape[0] - 1)
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=50, random_state=seed)
        labels = model.fit_predict(x)
        score = float(silhouette_score(x, labels)) if len(set(labels)) > 1 else float("nan")
        rows.append({"k": int(k), "inertia": float(model.inertia_), "silhouette": score, "chosen": False})
    scores = pd.DataFrame(rows)
    if scores.empty:
        return 1, scores
    chosen_idx = int(scores["silhouette"].astype(float).idxmax())
    scores.loc[chosen_idx, "chosen"] = True
    return int(scores.loc[chosen_idx, "k"]), scores


def feature_table(unit: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    radii = sorted(float(r) for r in unit["radius"].unique())
    pivot = (
        unit.pivot_table(index=["rule", "ref_id"], columns="radius", values="delta_phi_energy_unit", aggfunc="mean")
        .reindex(columns=radii)
        .sort_index()
    )
    if pivot.isna().any().any():
        raise ValueError("missing phi curve values in rule/ref/radius grid")
    cols = [f"delta_phi_r_{r:.1f}" for r in radii]
    features = pivot.reset_index()
    features.columns = ["rule", "ref_id", *cols]
    return features, radii


def plot_rule_curves(unit: pd.DataFrame, assignments: pd.DataFrame, radii: list[float], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    for ax, rule in zip(axes.ravel(), RULES):
        sub_assign = assignments[assignments["rule"].eq(rule)]
        sub_unit = unit[unit["rule"].eq(rule)]
        for cluster, cluster_rows in sub_assign.groupby("phi_curve_cluster"):
            color = COLORS[int(cluster) % len(COLORS)]
            ref_ids = set(cluster_rows["ref_id"].astype(int).tolist())
            cluster_unit = sub_unit[sub_unit["ref_id"].astype(int).isin(ref_ids)]
            for _, ref_sub in cluster_unit.groupby("ref_id"):
                ref_sub = ref_sub.sort_values("radius")
                ax.plot(ref_sub["radius"], ref_sub["delta_phi_energy_unit"], color=color, alpha=0.09, linewidth=0.65)
            mean_curve = (
                cluster_unit.groupby("radius")["delta_phi_energy_unit"]
                .mean()
                .reindex(radii)
                .to_numpy(dtype=float)
            )
            ax.plot(radii, mean_curve, color=color, linewidth=2.4, label=f"cluster {int(cluster)} n={len(ref_ids)}")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.3)
        ax.set_title(rule)
        ax.set_xlabel("radius d")
        ax.set_ylabel("delta phi energy")
        ax.grid(True, linewidth=0.35, alpha=0.24)
        ax.legend(fontsize=8)
    fig.suptitle("Within-rule reference clustering using only phi(d) curves", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=185)
    plt.close(fig)


def plot_rule_pca(assignments: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    for ax, rule in zip(axes.ravel(), RULES):
        sub = assignments[assignments["rule"].eq(rule)]
        for cluster, cluster_rows in sub.groupby("phi_curve_cluster"):
            color = COLORS[int(cluster) % len(COLORS)]
            ax.scatter(
                cluster_rows["pc1"],
                cluster_rows["pc2"],
                s=34,
                alpha=0.82,
                color=color,
                label=f"cluster {int(cluster)} n={len(cluster_rows)}",
            )
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.25)
        ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.25)
        ax.set_title(rule)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linewidth=0.35, alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("Phi-curve-only clustering PCA view", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=185)
    plt.close(fig)


def main() -> int:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)
    unit = pd.read_csv(UNIT_PATH)
    features, radii = feature_table(unit)
    feature_cols = [col for col in features.columns if col.startswith("delta_phi_r_")]
    assignment_rows: list[pd.DataFrame] = []
    score_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for rule_idx, rule in enumerate(RULES):
        sub = features[features["rule"].eq(rule)].copy()
        x_raw = sub[feature_cols].to_numpy(dtype=float)
        x = StandardScaler().fit_transform(x_raw)
        k, scores = choose_k(x, seed=33100 + rule_idx)
        if k <= 1:
            labels = np.zeros(x.shape[0], dtype=int)
        else:
            labels = KMeans(n_clusters=k, n_init=100, random_state=44100 + rule_idx).fit_predict(x)
        pca = PCA(n_components=3, random_state=55100 + rule_idx)
        pcs = pca.fit_transform(x)
        out = sub[["rule", "ref_id"]].copy()
        out["phi_curve_cluster"] = labels.astype(int)
        out["pc1"] = pcs[:, 0]
        out["pc2"] = pcs[:, 1]
        out["pc3"] = pcs[:, 2]
        out["mean_delta_phi"] = x_raw.mean(axis=1)
        out["final_delta_phi"] = x_raw[:, -1]
        assignment_rows.append(out)
        scores["rule"] = rule
        score_rows.append(scores)
        for cluster, cluster_rows in out.groupby("phi_curve_cluster"):
            raw_rows = x_raw[np.isin(out["ref_id"], cluster_rows["ref_id"])]
            summary_rows.append(
                {
                    "rule": rule,
                    "phi_curve_cluster": int(cluster),
                    "ref_count": int(len(cluster_rows)),
                    "mean_delta_phi": float(np.mean(raw_rows)),
                    "mean_final_delta_phi": float(np.mean(raw_rows[:, -1])),
                    "sd_final_delta_phi": float(np.std(raw_rows[:, -1], ddof=1)) if len(cluster_rows) > 1 else 0.0,
                    "ref_ids": ",".join(f"{int(ref_id):03d}" for ref_id in sorted(cluster_rows["ref_id"].tolist())),
                }
            )

    assignments = pd.concat(assignment_rows, ignore_index=True)
    scores = pd.concat(score_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["rule", "phi_curve_cluster"])
    features.to_csv(TABLE_DIR / "phi_curve_features_by_ref.csv", index=False)
    assignments.to_csv(TABLE_DIR / "within_rule_phi_curve_cluster_assignments.csv", index=False)
    scores.to_csv(TABLE_DIR / "within_rule_phi_curve_cluster_scores.csv", index=False)
    summary.to_csv(TABLE_DIR / "within_rule_phi_curve_cluster_summary.csv", index=False)
    plot_rule_curves(unit, assignments, radii, FIG_DIR / "fig01_within_rule_phi_curve_clusters.png")
    plot_rule_pca(assignments, FIG_DIR / "fig02_within_rule_phi_curve_cluster_pca.png")

    lines = [
        "# Within-Rule Phi Curve Clustering",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Features: reference-level `delta_phi_energy_unit(d)` at all 25 radii. Numerical derivative features are not used in this run.",
        "",
        "## Selected k",
        "",
        "| rule | selected k | silhouette |",
        "| --- | ---: | ---: |",
    ]
    chosen = scores[scores["chosen"]].sort_values("rule")
    for row in chosen.to_dict("records"):
        lines.append(f"| {row['rule']} | {int(row['k'])} | {float(row['silhouette']):.4f} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `tables/phi_curve_features_by_ref.csv`",
            "- `tables/within_rule_phi_curve_cluster_assignments.csv`",
            "- `tables/within_rule_phi_curve_cluster_scores.csv`",
            "- `tables/within_rule_phi_curve_cluster_summary.csv`",
            "- `figures/fig01_within_rule_phi_curve_clusters.png`",
            "- `figures/fig02_within_rule_phi_curve_cluster_pca.png`",
            "",
        ]
    )
    (OUT_ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT_ROOT / "REPORT.md")
    print(chosen[["rule", "k", "silhouette"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
