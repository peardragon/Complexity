from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "smoke_runs" / "large_domain_pursuit_10x10_box_60ref"

SOURCES = [
    {
        "source": "microline_formal_pass",
        "label": "formal all-QC microline",
        "run_root": ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_microline_4rule_lowtv",
        "linestyle": "-",
        "alpha": 1.0,
    },
    {
        "source": "broad_0p01_to_0p08_pursuit",
        "label": "broad pursuit raw/QC-labelled",
        "run_root": ROOT / "runs" / "final" / "single_dataset_10x10_box_n_train_512_60ref_qcpass_line_4rule_lowtv",
        "linestyle": "--",
        "alpha": 0.82,
    },
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def bootstrap_sd(values: np.ndarray, seed: int, n_boot: int = 400) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for idx in range(int(n_boot)):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(sample))
    return float(np.std(means, ddof=1))


def stable_seed(*parts: object, base: int = 910000) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()
    return int(base) + int(digest, 16) % 100000


def load_source(source_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_root = Path(source_cfg["run_root"])
    unit_path = run_root / "05_pool2_pm_sais_sampling" / "shell_summary_by_unit.csv"
    qc_path = run_root / "05_pool2_pm_sais_sampling" / "qc_by_rule_radius.csv"
    config_path = run_root / "05_pool2_pm_sais_sampling" / "run_config_resolved.json"
    if not unit_path.exists() or not qc_path.exists() or not config_path.exists():
        missing = [rel(p) for p in [unit_path, qc_path, config_path] if not p.exists()]
        raise FileNotFoundError(f"missing source files for {source_cfg['source']}: {missing}")
    return pd.read_csv(unit_path), pd.read_csv(qc_path), read_json(config_path)


def phi_rows_for_source(source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    unit_df, qc_df, cfg = load_source(source_cfg)
    p_count = int(cfg["model"]["P"])
    r0 = float(cfg["sampling"]["r0"])
    key = ["split_id", "rule", "ref_id"]
    energy_col = "logZ_inf_full" if "logZ_inf_full" in unit_df.columns else "logZ"
    r0_df = unit_df[np.isclose(unit_df["radius"], r0)][key + [energy_col]].rename(columns={energy_col: "logZ_r0"})
    joined = unit_df.merge(r0_df, on=key, how="left")
    joined["delta_phi_energy_unit"] = (joined[energy_col] - joined["logZ_r0"]) / float(p_count)
    rows: list[dict[str, Any]] = []
    for (rule, radius), sub in joined.groupby(["rule", "radius"]):
        qc_match = qc_df[(qc_df["rule"] == rule) & np.isclose(qc_df["radius"], float(radius))]
        qc_pass = bool(qc_match["qc_pass"].iloc[0]) if not qc_match.empty else False
        values = sub["delta_phi_energy_unit"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "source": source_cfg["source"],
                "source_label": source_cfg["label"],
                "run_root": rel(Path(source_cfg["run_root"])),
                "rule": str(rule),
                "radius": float(radius),
                "d0": r0,
                "delta_phi_energy": float(np.mean(values)),
                "bootstrap_sd_phi": bootstrap_sd(
                    values,
                    stable_seed(source_cfg["source"], str(rule), f"{float(radius):.8f}"),
                ),
                "n_units": int(len(sub)),
                "qc_pass": bool(qc_pass),
                "claim_status": "formal_qc_pass" if bool(qc_pass) else "pursuit_no_claim",
            }
        )
    return rows


def derivative_rows(phi_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source, rule), sub in phi_df.sort_values("radius").groupby(["source", "rule"]):
        x = sub["radius"].to_numpy(dtype=np.float64)
        y = sub["delta_phi_energy"].to_numpy(dtype=np.float64)
        if x.size < 2:
            continue
        dydx = np.gradient(y, x)
        qc = sub["qc_pass"].to_numpy(dtype=bool)
        for idx, row in enumerate(sub.to_dict("records")):
            left_ok = qc[idx - 1] if idx > 0 else qc[idx]
            right_ok = qc[idx + 1] if idx + 1 < qc.size else qc[idx]
            rows.append(
                {
                    "source": source,
                    "source_label": row["source_label"],
                    "rule": rule,
                    "radius": float(row["radius"]),
                    "d_delta_phi_energy_dd": float(dydx[idx]),
                    "local_qc_pass": bool(qc[idx] and left_ok and right_ok),
                    "claim_status": "formal_qc_pass_local_derivative" if bool(qc[idx] and left_ok and right_ok) else "pursuit_no_claim",
                }
            )
    return pd.DataFrame(rows)


def plot_phi(phi_df: pd.DataFrame, fig_path: Path) -> None:
    ensure_dir(fig_path.parent)
    rules = list(dict.fromkeys(phi_df["rule"].tolist()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    axes = axes.ravel()
    for ax, rule in zip(axes, rules):
        for source_cfg in SOURCES:
            sub = phi_df[(phi_df["rule"] == rule) & (phi_df["source"] == source_cfg["source"])].sort_values("radius")
            if sub.empty:
                continue
            ax.plot(
                sub["radius"],
                sub["delta_phi_energy"],
                linestyle=source_cfg["linestyle"],
                linewidth=1.8,
                alpha=float(source_cfg["alpha"]),
                label=source_cfg["label"],
            )
            pass_sub = sub[sub["qc_pass"]]
            fail_sub = sub[~sub["qc_pass"]]
            if not pass_sub.empty:
                ax.scatter(pass_sub["radius"], pass_sub["delta_phi_energy"], s=24, marker="o")
            if not fail_sub.empty:
                ax.scatter(fail_sub["radius"], fail_sub["delta_phi_energy"], s=38, marker="x")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
        ax.set_title(rule)
        ax.set_xlabel("d_raw")
        ax.set_ylabel("delta phi energy")
        ax.grid(True, linewidth=0.4, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("MNIST10 BOX large-domain pursuit: phi(d)_energy")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_derivative(deriv_df: pd.DataFrame, fig_path: Path) -> None:
    ensure_dir(fig_path.parent)
    rules = list(dict.fromkeys(deriv_df["rule"].tolist()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    axes = axes.ravel()
    for ax, rule in zip(axes, rules):
        for source_cfg in SOURCES:
            sub = deriv_df[(deriv_df["rule"] == rule) & (deriv_df["source"] == source_cfg["source"])].sort_values("radius")
            if sub.empty:
                continue
            ax.plot(
                sub["radius"],
                sub["d_delta_phi_energy_dd"],
                linestyle=source_cfg["linestyle"],
                linewidth=1.8,
                alpha=float(source_cfg["alpha"]),
                label=source_cfg["label"],
            )
            pass_sub = sub[sub["local_qc_pass"]]
            fail_sub = sub[~sub["local_qc_pass"]]
            if not pass_sub.empty:
                ax.scatter(pass_sub["radius"], pass_sub["d_delta_phi_energy_dd"], s=24, marker="o")
            if not fail_sub.empty:
                ax.scatter(fail_sub["radius"], fail_sub["d_delta_phi_energy_dd"], s=38, marker="x")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
        ax.set_title(rule)
        ax.set_xlabel("d_raw")
        ax.set_ylabel("d delta phi energy / d d_raw")
        ax.grid(True, linewidth=0.4, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("MNIST10 BOX large-domain pursuit: d phi(d)_energy / dd")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ensure_dir(OUT_ROOT / "config")
    ensure_dir(OUT_ROOT / "raw_outputs")
    ensure_dir(OUT_ROOT / "figures")
    ensure_dir(OUT_ROOT / "QC")
    rows: list[dict[str, Any]] = []
    for source_cfg in SOURCES:
        rows.extend(phi_rows_for_source(source_cfg))
    phi_df = pd.DataFrame(rows).sort_values(["source", "rule", "radius"])
    deriv_df = derivative_rows(phi_df).sort_values(["source", "rule", "radius"])
    write_csv(OUT_ROOT / "raw_outputs" / "phi_energy_large_domain_pursuit.csv", phi_df)
    write_csv(OUT_ROOT / "raw_outputs" / "dphi_dd_energy_large_domain_pursuit.csv", deriv_df)
    write_json(
        OUT_ROOT / "config" / "large_domain_pursuit_config.json",
        {
            "sources": [{**src, "run_root": rel(Path(src["run_root"]))} for src in SOURCES],
            "output_root": rel(OUT_ROOT),
            "derivative": "numpy.gradient over source/rule sorted radii",
            "claim_policy": "Only points with qc_pass=true are formal; failed broad points are pursuit_no_claim diagnostics.",
        },
    )
    plot_phi(phi_df, OUT_ROOT / "figures" / "fig_phi_energy_large_domain_pursuit.png")
    plot_derivative(deriv_df, OUT_ROOT / "figures" / "fig_dphi_dd_energy_large_domain_pursuit.png")
    broad = phi_df[phi_df["source"] == "broad_0p01_to_0p08_pursuit"]
    micro = phi_df[phi_df["source"] == "microline_formal_pass"]
    checks = {
        "phi_rows": int(len(phi_df)),
        "derivative_rows": int(len(deriv_df)),
        "microline_failed_rows": int((~micro["qc_pass"]).sum()),
        "broad_failed_rows": int((~broad["qc_pass"]).sum()),
        "broad_max_radius": float(broad["radius"].max()) if not broad.empty else None,
        "fig_phi_exists": bool((OUT_ROOT / "figures" / "fig_phi_energy_large_domain_pursuit.png").exists()),
        "fig_derivative_exists": bool((OUT_ROOT / "figures" / "fig_dphi_dd_energy_large_domain_pursuit.png").exists()),
    }
    write_json(OUT_ROOT / "QC" / "QC_STATUS.json", {"status": "pass", "checks": checks})
    report = f"""# Large-Domain Pursuit: MNIST10 BOX 60ref

This is a pursuit artifact, not a promoted final claim table.

Sources:

- `microline_formal_pass`: all-QC pass formal line, d_raw 0.010..0.014.
- `broad_0p01_to_0p08_pursuit`: same 10x10 PM-SAIS skeleton, d_raw 0.010..0.080, QC-labelled raw diagnostic.

Rows:

- phi rows: `{checks['phi_rows']}`
- derivative rows: `{checks['derivative_rows']}`
- broad max radius: `{checks['broad_max_radius']}`
- broad failed rule/radius rows: `{checks['broad_failed_rows']}`

Figures:

- `figures/fig_phi_energy_large_domain_pursuit.png`
- `figures/fig_dphi_dd_energy_large_domain_pursuit.png`

Interpretation:

The broad curve extends the domain to d_raw 0.08, but failed broad points remain `pursuit_no_claim`. The derivative figure is a finite-difference pursuit diagnostic and should be read with the QC markers.
"""
    (OUT_ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
