from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "summarized_outputs"
DEFAULT_PHI_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "phi_by_sampling"
DEFAULT_PHI_OUTPUT_PNG = (
    PROJECT_ROOT
    / "01_theory"
    / "02_theory_sampling"
    / "figures"
    / "phi_by_sampling"
    / "phi_by_sampling.png"
)
DEFAULT_LOGZ_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "logZ_split"
DEFAULT_LOGZ_SOURCE_CSV = SUMMARY_ROOT / "sample_unit_summary.csv"
DEFAULT_LOGZ_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "01_theory"
    / "02_theory_sampling"
    / "figures"
    / "logZ_split_distributions"
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def clear_csvs(path: Path, pattern: str = "N_*.csv") -> None:
    if not path.exists():
        return
    for csv_path in path.glob(pattern):
        csv_path.unlink()


def clear_pngs(path: Path, pattern: str = "N_*.png") -> None:
    if not path.exists():
        return
    for png_path in path.glob(pattern):
        png_path.unlink()


def _series_value(frame: pd.DataFrame) -> pd.Series:
    if "phi_emp_rel" in frame.columns:
        return frame["phi_emp_rel"]
    base = frame.sort_values("r")["phi_emp"].iloc[0]
    return frame["phi_emp"] - base


def load_phi_inputs(input_path: Path) -> pd.DataFrame:
    if input_path.is_file():
        return pd.read_csv(input_path)
    files = sorted(input_path.glob("N_*.csv"))
    if not files:
        raise FileNotFoundError(f"no N_*.csv files found under {input_path}")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True, sort=False)


def make_phi_figure(input_path: Path, output_png: Path) -> None:
    df = load_phi_inputs(input_path).sort_values(["N", "r"])
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.4, 4.8))
    for n_value, group in df.groupby("N", sort=True):
        group = group.sort_values("r")
        plt.plot(group["r"], _series_value(group), marker="o", linewidth=1.7, label=f"N={int(n_value)}")
    plt.xlabel("d")
    plt.ylabel("empirical phi(d) - phi(d0)")
    plt.title("Two-pool shell sampling, alpha=0.1")
    plt.grid(True, alpha=0.28)
    plt.legend(title="system size", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def resolve_source_csv(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(path)


def normalize_logz_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    if "scale_value" in frame.columns:
        out["N"] = pd.to_numeric(frame["scale_value"], errors="coerce")
    elif "N" in frame.columns:
        out["N"] = pd.to_numeric(frame["N"], errors="coerce")
    elif "condition_value" in frame.columns:
        out["N"] = pd.to_numeric(frame["condition_value"], errors="coerce")
    else:
        raise KeyError("expected one of scale_value, N, or condition_value")

    radius_col = "radius" if "radius" in frame.columns else "r"
    if radius_col not in frame.columns:
        raise KeyError("expected radius or r")
    out["r"] = pd.to_numeric(frame[radius_col], errors="coerce")

    if "split_logZ_per_scale_diff" in frame.columns:
        out["split_logZ_per_N_diff"] = pd.to_numeric(frame["split_logZ_per_scale_diff"], errors="coerce")
    elif "split_logZ_per_N_diff" in frame.columns:
        out["split_logZ_per_N_diff"] = pd.to_numeric(frame["split_logZ_per_N_diff"], errors="coerce")
    else:
        raise KeyError("expected split_logZ_per_scale_diff or split_logZ_per_N_diff")

    optional_columns = {
        "dataset_id": "dataset_id",
        "ref_id": "ref_id",
        "split0_logZ": "split0_logZ",
        "split1_logZ": "split1_logZ",
        "split0_logZ_shell": "split0_logZ",
        "split1_logZ_shell": "split1_logZ",
        "signed_split_logZ_per_scale": "signed_split_logZ_per_N_diff",
        "signed_split_logZ_per_N_diff": "signed_split_logZ_per_N_diff",
        "ess_fraction": "ess_fraction",
        "ess_frac": "ess_fraction",
        "smc_min_cess_fraction": "smc_min_cess_fraction",
        "smc_completed": "smc_completed",
        "sampler_method": "sampler_method",
        "n_particles": "n_particles",
        "sample_payload_path": "sample_payload_path",
        "payload_split": "payload_split",
        "far_split_start_r": "far_split_start_r",
        "source_path": "source_path",
    }
    for source, target in optional_columns.items():
        if source in frame.columns and target not in out.columns:
            out[target] = frame[source]

    out = out.dropna(subset=["N", "r", "split_logZ_per_N_diff"]).copy()
    out["N"] = out["N"].astype(int)
    out["r"] = out["r"].astype(float)
    out["split_logZ_per_N_diff"] = out["split_logZ_per_N_diff"].astype(float)
    if "signed_split_logZ_per_N_diff" not in out.columns and {"split0_logZ", "split1_logZ"}.issubset(out.columns):
        split0 = pd.to_numeric(out["split0_logZ"], errors="coerce")
        split1 = pd.to_numeric(out["split1_logZ"], errors="coerce")
        out["signed_split_logZ_per_N_diff"] = (split0 - split1) / out["N"]
    if "payload_split" not in out.columns and "sample_payload_path" in out.columns:
        payload_paths = out["sample_payload_path"].fillna("").astype(str)
        far_from_path = payload_paths.str.contains("far_split", regex=False)
        far_from_particles = (
            pd.to_numeric(out.get("n_particles", pd.Series(np.nan, index=out.index)), errors="coerce") >= 32768
        )
        out["payload_split"] = np.where(far_from_path | far_from_particles, "far_split", "near_split")
    if "far_split_start_r" not in out.columns and "payload_split" in out.columns:
        far_start = out.loc[out["payload_split"].eq("far_split")].groupby("N", sort=True)["r"].min()
        out["far_split_start_r"] = out["N"].map(far_start)
    sort_cols = [col for col in ["N", "r", "dataset_id", "ref_id"] if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _split_metadata(split_source_csv: Path | None) -> pd.DataFrame:
    if split_source_csv is None or not split_source_csv.exists():
        return pd.DataFrame()
    usecols = ["N", "dataset_id", "ref_id", "radius", "n_particles", "sample_payload_path"]
    meta = pd.read_csv(split_source_csv, usecols=usecols)
    meta = meta.rename(columns={"radius": "r"})
    meta["N"] = pd.to_numeric(meta["N"], errors="coerce")
    meta["r"] = pd.to_numeric(meta["r"], errors="coerce")
    meta["dataset_id"] = pd.to_numeric(meta["dataset_id"], errors="coerce")
    meta["ref_id"] = pd.to_numeric(meta["ref_id"], errors="coerce")
    meta["n_particles"] = pd.to_numeric(meta["n_particles"], errors="coerce")
    meta = meta.dropna(subset=["N", "r", "dataset_id", "ref_id"]).copy()
    meta["N"] = meta["N"].astype(int)
    meta["dataset_id"] = meta["dataset_id"].astype(int)
    meta["ref_id"] = meta["ref_id"].astype(int)
    meta["payload_split"] = np.where(
        meta["sample_payload_path"].astype(str).str.contains("/far_split/"),
        "far_split",
        "near_split",
    )
    return meta


def enrich_split_metadata(frame: pd.DataFrame, split_source_csv: Path | None = None) -> pd.DataFrame:
    if "payload_split" in frame.columns and "far_split_start_r" in frame.columns:
        return frame
    meta = _split_metadata(split_source_csv)
    if meta.empty or not {"dataset_id", "ref_id"}.issubset(frame.columns):
        return frame
    out = frame.merge(
        meta,
        on=["N", "r", "dataset_id", "ref_id"],
        how="left",
        validate="one_to_one",
    )
    far_start = (
        out.loc[out["payload_split"].eq("far_split")]
        .groupby("N", sort=True)["r"]
        .min()
        .rename("far_split_start_r")
    )
    out = out.merge(far_start, on="N", how="left")
    return out


def write_logz_split_inputs(
    source_csv: Path,
    output_root: Path,
    split_source_csv: Path | None = None,
) -> list[Path]:
    source_csv = resolve_source_csv(source_csv)
    output_root.mkdir(parents=True, exist_ok=True)
    frame = enrich_split_metadata(normalize_logz_frame(pd.read_csv(source_csv)), split_source_csv)
    if "source_path" not in frame.columns:
        frame["source_path"] = str(source_csv.resolve())
    clear_csvs(output_root)
    outputs: list[Path] = []
    for n_value, group in frame.groupby("N", sort=True):
        out = output_root / f"N_{int(n_value)}.csv"
        group.to_csv(out, index=False)
        outputs.append(out)
    return outputs


def _radius_label(value: float) -> str:
    return f"{value:g}"


def _n_from_path(path: Path) -> int:
    return int(path.stem.removeprefix("N_"))


def _far_split_start(frame: pd.DataFrame) -> float | None:
    if "far_split_start_r" in frame.columns:
        values = pd.to_numeric(frame["far_split_start_r"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    if "payload_split" in frame.columns:
        far = frame.loc[frame["payload_split"].eq("far_split"), "r"]
        if not far.empty:
            return float(far.min())
    return None


def _xtick_labels(radii: list[float]) -> list[str]:
    if len(radii) <= 24:
        return [_radius_label(value) for value in radii]
    step = max(1, int(np.ceil(len(radii) / 14)))
    labels = []
    for idx, value in enumerate(radii):
        labels.append(_radius_label(value) if idx % step == 0 or idx == len(radii) - 1 else "")
    return labels


def _plot_value_column(frame: pd.DataFrame) -> tuple[str, str, bool]:
    if "signed_split_logZ_per_N_diff" in frame.columns:
        return "signed_split_logZ_per_N_diff", "signed split logZ diff per N", True
    return "split_logZ_per_N_diff", "absolute split logZ diff per N", False


def plot_logz_split_distribution(input_csv: Path, output_png: Path, *, split_threshold: float = 0.006) -> None:
    frame = pd.read_csv(input_csv)
    frame["r"] = pd.to_numeric(frame["r"], errors="coerce")
    y_col, y_label, signed = _plot_value_column(frame)
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    frame = frame.dropna(subset=["r", y_col]).sort_values(["r"])
    if frame.empty:
        raise ValueError(f"no finite logZ split rows in {input_csv}")

    n_value = int(frame["N"].iloc[0]) if "N" in frame.columns else int(input_csv.stem.removeprefix("N_"))
    radii = [float(value) for value in sorted(frame["r"].unique())]
    values = [frame.loc[np.isclose(frame["r"], radius), y_col].to_numpy(float) for radius in radii]
    positions = np.arange(len(radii), dtype=float)
    width = max(10.5, min(18.0, 0.34 * len(radii)))
    fig, ax = plt.subplots(figsize=(width, 5.8))

    violin_positions = positions + 0.16
    parts = ax.violinplot(
        values,
        positions=violin_positions,
        widths=0.34,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for center, body in zip(violin_positions, parts["bodies"]):
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.maximum(vertices[:, 0], center)
        body.set_facecolor("#6f6f6f")
        body.set_edgecolor("none")
        body.set_alpha(0.82)

    rng = np.random.default_rng(1729 + n_value)
    if signed:
        color_extent = max(float(np.nanquantile(np.abs(frame[y_col]), 0.98)), 1.0e-12)
        cmap = "coolwarm"
        vmin, vmax = -color_extent, color_extent
    else:
        vmax = max(float(frame[y_col].quantile(0.98)), 1.0e-12)
        cmap = "magma"
        vmin = 0.0
    for idx, radius in enumerate(radii):
        group = frame.loc[np.isclose(frame["r"], radius), y_col].to_numpy(float)
        x = rng.normal(loc=positions[idx] - 0.12, scale=0.045, size=len(group))
        x = np.clip(x, positions[idx] - 0.27, positions[idx] + 0.04)
        ax.scatter(
            x,
            group,
            c=group,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=18,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    far_start = _far_split_start(frame)
    if far_start is not None:
        far_idx = next((idx for idx, radius in enumerate(radii) if np.isclose(radius, far_start)), None)
        if far_idx is not None:
            ax.axvline(far_idx - 0.5, color="#2f4b7c", linestyle="--", linewidth=1.2, alpha=0.8)
    if signed:
        y_extent = max(float(np.nanquantile(np.abs(frame[y_col]), 0.995)), 1.0e-12) * 1.12
        ax.set_ylim(-y_extent, y_extent)
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_xlim(-0.55, len(radii) - 0.45)
    ax.set_xticks(positions)
    ax.set_xticklabels(_xtick_labels(radii), rotation=90, fontsize=7)
    ax.set_xlabel("d")
    ax.set_ylabel(y_label)
    ax.set_title(f"logZ split distributions, N={n_value}")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def render_logz_split_figures(input_root: Path, output_root: Path, *, split_threshold: float = 0.006) -> list[Path]:
    files = sorted(input_root.glob("N_*.csv"), key=_n_from_path)
    if not files:
        raise FileNotFoundError(f"no N_*.csv files found under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    clear_pngs(output_root)
    outputs: list[Path] = []
    for path in files:
        out = output_root / f"{path.stem}.png"
        plot_logz_split_distribution(path, out, split_threshold=split_threshold)
        outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build theory figures from N-wise figure-input CSVs.")
    parser.add_argument("--which", choices=["all", "phi", "logz"], default="all")
    parser.add_argument("--phi-input", "--input", dest="phi_input", type=Path, default=DEFAULT_PHI_INPUT_ROOT)
    parser.add_argument("--phi-input-csv", "--input-csv", dest="phi_input_csv", type=Path, default=None)
    parser.add_argument("--phi-output-png", "--output-png", dest="phi_output_png", type=Path, default=DEFAULT_PHI_OUTPUT_PNG)
    parser.add_argument("--logz-source-csv", "--source-csv", dest="logz_source_csv", type=Path, default=None)
    parser.add_argument("--split-source-csv", type=Path, default=None)
    parser.add_argument("--logz-input-root", "--input-root", dest="logz_input_root", type=Path, default=DEFAULT_LOGZ_INPUT_ROOT)
    parser.add_argument("--logz-output-root", "--output-root", dest="logz_output_root", type=Path, default=DEFAULT_LOGZ_OUTPUT_ROOT)
    parser.add_argument("--split-threshold", type=float, default=0.006)
    parser.add_argument("--refresh-logz-inputs", action="store_true")
    parser.add_argument("--skip-logz-input-write", "--skip-input-write", dest="skip_logz_input_write", action="store_true")
    args = parser.parse_args()

    outputs: list[Path] = []
    if args.which in {"all", "phi"}:
        phi_input = args.phi_input_csv if args.phi_input_csv is not None else args.phi_input
        phi_output = project_path(args.phi_output_png)
        make_phi_figure(project_path(phi_input), phi_output)
        outputs.append(phi_output)

    if args.which in {"all", "logz"}:
        input_root = project_path(args.logz_input_root)
        refresh_logz_inputs = args.refresh_logz_inputs or args.logz_source_csv is not None
        if refresh_logz_inputs and not args.skip_logz_input_write:
            split_source_csv = project_path(args.split_source_csv) if args.split_source_csv is not None else None
            source_csv = project_path(args.logz_source_csv) if args.logz_source_csv is not None else DEFAULT_LOGZ_SOURCE_CSV
            write_logz_split_inputs(source_csv, input_root, split_source_csv)
        outputs.extend(
            render_logz_split_figures(
                input_root,
                project_path(args.logz_output_root),
                split_threshold=float(args.split_threshold),
            )
        )

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
