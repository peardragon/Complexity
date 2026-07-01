from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLING_ROOT = Path(__file__).resolve().parents[1]
DNN_ROOT = SAMPLING_ROOT.parent
SUMMARY_ROOT = SAMPLING_ROOT / "summarized_outputs"
UNIT_SUMMARY_ROOT = SUMMARY_ROOT / "unit_summary"
DEFAULT_BETA_SUMMARY_ROOT = UNIT_SUMMARY_ROOT
DEFAULT_FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "logZ_split"
DEFAULT_RAW_ROOT = SAMPLING_ROOT / "raw_outputs" / "shell_pool"
P_DIM_DEFAULT = 2545.0

UNIT_SUMMARY_FIELDS = [
    "stage",
    "block",
    "condition_name",
    "condition_value",
    "dataset_id",
    "ref_id",
    "radius",
    "source_type",
    "logZ_main",
    "logZ_CE",
    "logZ_stripped",
    "logZ_full",
    "reference_prior_log_weight",
    "log_prefactor",
    "dlogZ_dr",
    "split0_logZ",
    "split1_logZ",
    "signed_split_logZ_per_scale",
    "split_logZ_per_scale_diff",
    "dlogZ_dr_split0",
    "dlogZ_dr_split1",
    "split_dlogZ_dr_per_scale_diff",
    "scale_name",
    "scale_value",
    "ess_fraction",
    "smc_min_cess_fraction",
    "smc_completed",
    "sampler_method",
    "source_path",
]


def beta_slug(beta: float) -> str:
    return f"{beta:.2f}".replace(".", "p")


def beta_from_name(path: Path) -> float:
    match = re.search(r"beta_(\d+p\d+)", path.stem)
    if not match:
        raise ValueError(f"cannot parse beta from {path.name}")
    return float(match.group(1).replace("p", "."))


def beta_part_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("beta_"):
            return part
        if part.startswith("cell_beta_"):
            return "beta_" + part.removeprefix("cell_beta_")
    raise ValueError(f"cannot parse beta directory from {path}")


def beta_summary_files(root: Path) -> list[Path]:
    files = {path.resolve(): path for path in root.glob("beta_*.csv")}
    for path in root.glob("cell_beta_*.csv"):
        files.setdefault(path.resolve(), path)
    return sorted(files.values(), key=beta_from_name)


def tagged_float(name: str, prefix: str) -> float:
    return float(name.removeprefix(prefix).replace("p", "."))


def tagged_int(name: str, prefix: str) -> int:
    return int(name.removeprefix(prefix))


def radius_sort_key(path: Path) -> float:
    return tagged_float(path.name, "r_")


def clear_outputs(path: Path, pattern: str) -> None:
    if not path.exists():
        return
    for item in path.glob(pattern):
        item.unlink()


def relative_to_dnn(path: Path) -> str:
    try:
        return path.resolve().relative_to(DNN_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_source_beta_summary_root(source_root: Path) -> Path:
    return source_root.resolve()


def unit_summary_row(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    parts = path.parts
    beta = float(payload.get("beta", tagged_float(beta_part_from_path(path), "beta_")))
    dataset_part = next(part for part in parts if part.startswith("dataset_"))
    ref_part = next(part for part in parts if part.startswith("ref_"))
    radius_part = next(part for part in parts if part.startswith("r_"))
    dataset_id = tagged_int(dataset_part, "dataset_")
    ref_id = tagged_int(ref_part, "ref_")
    radius = float(payload.get("radius", tagged_float(radius_part, "r_")))
    scale_value = float(payload.get("P_params", payload.get("P", P_DIM_DEFAULT)))

    split0 = float(payload.get("split0_logZ_inf", np.nan))
    split1 = float(payload.get("split1_logZ_inf", np.nan))
    signed_split = (split0 - split1) / scale_value if np.isfinite(split0) and np.isfinite(split1) else np.nan
    dsplit = float(payload.get("split_dlogZ_dr_per_P_diff", np.nan))
    if not np.isfinite(dsplit):
        d0 = float(payload.get("dlogZ_dr_split0", np.nan))
        d1 = float(payload.get("dlogZ_dr_split1", np.nan))
        dsplit = abs(d0 - d1) / scale_value if np.isfinite(d0) and np.isfinite(d1) else np.nan

    return {
        "stage": "02_dnn_synthetic",
        "block": "synthetic_main",
        "condition_name": "beta",
        "condition_value": beta,
        "dataset_id": dataset_id,
        "ref_id": ref_id,
        "radius": radius,
        "source_type": "unit_summary_json",
        "logZ_main": payload.get("logZ_inf_full", ""),
        "logZ_CE": payload.get("logZ_CE", ""),
        "logZ_stripped": payload.get("logZ_inf_stripped", ""),
        "logZ_full": payload.get("logZ_inf_full", ""),
        "reference_prior_log_weight": payload.get("reference_prior_log_weight", ""),
        "log_prefactor": payload.get("log_prefactor", ""),
        "dlogZ_dr": payload.get("dlogZ_inf_full_dr", payload.get("dlogZ_inf_dr", "")),
        "split0_logZ": payload.get("split0_logZ_inf", ""),
        "split1_logZ": payload.get("split1_logZ_inf", ""),
        "signed_split_logZ_per_scale": f"{signed_split:.6g}" if np.isfinite(signed_split) else "",
        "split_logZ_per_scale_diff": abs(signed_split) if np.isfinite(signed_split) else "",
        "dlogZ_dr_split0": payload.get("dlogZ_dr_split0", ""),
        "dlogZ_dr_split1": payload.get("dlogZ_dr_split1", ""),
        "split_dlogZ_dr_per_scale_diff": dsplit if np.isfinite(dsplit) else "",
        "scale_name": "P",
        "scale_value": scale_value,
        "ess_fraction": payload.get("ess_frac", payload.get("ess_fraction", "")),
        "smc_min_cess_fraction": payload.get("smc_min_cess_fraction", ""),
        "smc_completed": str(bool(payload.get("smc_completed", False))).lower(),
        "sampler_method": payload.get("sampler_method", ""),
        "source_path": relative_to_dnn(path),
    }


def build_unit_summaries_from_raw(raw_root: Path, output_root: Path) -> list[Path]:
    beta_dirs = sorted(raw_root.glob("beta_*"), key=lambda path: tagged_float(path.name, "beta_"))
    beta_dirs.extend(path for path in sorted(raw_root.glob("cell_beta_*"), key=lambda path: tagged_float(path.name, "cell_beta_")) if path not in beta_dirs)
    if not beta_dirs:
        raise FileNotFoundError(f"no unit_summary.json files found under {raw_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "cell_beta_*.csv")
    clear_outputs(output_root, "beta_*.csv")

    outputs: list[Path] = []
    for beta_dir in beta_dirs:
        beta = tagged_float(beta_part_from_path(beta_dir), "beta_")
        out = output_root / f"beta_{beta_slug(beta)}.csv"
        wrote_any = False
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=UNIT_SUMMARY_FIELDS)
            writer.writeheader()
            for dataset_dir in sorted(beta_dir.glob("dataset_*"), key=lambda path: tagged_int(path.name, "dataset_")):
                for ref_dir in sorted(dataset_dir.glob("ref_*"), key=lambda path: tagged_int(path.name, "ref_")):
                    for radius_dir in sorted(ref_dir.glob("r_*"), key=radius_sort_key):
                        path = radius_dir / "unit_summary.json"
                        if path.exists():
                            writer.writerow(unit_summary_row(path))
                            wrote_any = True
        if wrote_any:
            outputs.append(out)
        else:
            out.unlink(missing_ok=True)
    if not outputs:
        raise FileNotFoundError(f"no unit_summary.json files found under {raw_root}")
    return outputs


def probe_scale_value(summary_root: Path) -> int:
    first = next(iter(beta_summary_files(summary_root)), None)
    if first is None:
        raise FileNotFoundError(f"no beta_*.csv files found under {summary_root}")
    values = pd.read_csv(first, usecols=["scale_value"], nrows=1)["scale_value"]
    return int(round(float(values.iloc[0])))


def link_beta_summaries(source_root: Path, output_root: Path) -> list[Path]:
    source_files = beta_summary_files(source_root)
    if not source_files:
        raise FileNotFoundError(f"no beta_*.csv files found under {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    same_root = source_root.resolve() == output_root.resolve()
    if not same_root:
        clear_outputs(output_root, "cell_beta_*.csv")
        clear_outputs(output_root, "beta_*.csv")

    outputs: list[Path] = []
    for source in source_files:
        target = output_root / f"beta_{beta_slug(beta_from_name(source))}.csv"
        if same_root:
            if source.name != target.name:
                if target.exists():
                    source.unlink()
                else:
                    source.replace(target)
            outputs.append(target)
        else:
            try:
                os.link(source, target)
            except OSError:
                target.symlink_to(source)
            outputs.append(target)
    return outputs


def selected_radius_mask(radius: pd.Series) -> pd.Series:
    scaled = np.rint(pd.to_numeric(radius, errors="coerce") * 100).astype("Int64")
    return scaled.mod(5).eq(0)


def write_figure_inputs_from_selected_points(
    selected_points_csv: Path,
    output_root: Path,
    *,
    scale_value: int,
) -> list[Path]:
    if not selected_points_csv.exists():
        raise FileNotFoundError(selected_points_csv)
    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "cell_beta_*.csv")
    clear_outputs(output_root, "beta_*.csv")

    frame = pd.read_csv(selected_points_csv)
    required = {"beta", "r", "signed_split_logZ_per_P_diff"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{selected_points_csv} missing columns: {sorted(missing)}")

    frame["beta"] = pd.to_numeric(frame["beta"], errors="coerce")
    frame["P"] = scale_value
    frame["r"] = pd.to_numeric(frame["r"], errors="coerce")
    frame["signed_split_logZ_per_P_diff"] = pd.to_numeric(frame["signed_split_logZ_per_P_diff"], errors="coerce")
    frame["split_logZ_per_P_diff"] = frame["signed_split_logZ_per_P_diff"].abs()
    frame = frame.dropna(subset=["beta", "r", "signed_split_logZ_per_P_diff"])
    frame = frame.sort_values(["beta", "r"]).reset_index(drop=True)

    outputs: list[Path] = []
    for beta, group in frame.groupby("beta", sort=True):
        out = output_root / f"beta_{beta_slug(float(beta))}.csv"
        group[["beta", "P", "r", "split_logZ_per_P_diff", "signed_split_logZ_per_P_diff"]].to_csv(out, index=False)
        outputs.append(out)
    return outputs


def write_figure_inputs_from_beta_summaries(
    summary_root: Path,
    output_root: Path,
    *,
    scale_value: int,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "cell_beta_*.csv")
    clear_outputs(output_root, "beta_*.csv")
    outputs: list[Path] = []
    usecols = ["condition_value", "radius", "signed_split_logZ_per_scale", "split_logZ_per_scale_diff"]
    for source in beta_summary_files(summary_root):
        beta = beta_from_name(source)
        frame = pd.read_csv(source, usecols=usecols)
        frame = frame.loc[selected_radius_mask(frame["radius"])].copy()
        frame = frame.rename(
            columns={
                "condition_value": "beta",
                "radius": "r",
                "split_logZ_per_scale_diff": "split_logZ_per_P_diff",
                "signed_split_logZ_per_scale": "signed_split_logZ_per_P_diff",
            }
        )
        frame["P"] = scale_value
        frame = frame[["beta", "P", "r", "split_logZ_per_P_diff", "signed_split_logZ_per_P_diff"]].sort_values(
            ["beta", "r"]
        )
        out = output_root / f"beta_{beta_slug(beta)}.csv"
        frame.to_csv(out, index=False)
        outputs.append(out)
    return outputs


def build_summarized_outputs(
    source_beta_summary_root: Path,
    beta_summary_root: Path,
    figure_input_root: Path,
    selected_points_csv: Path | None,
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    from_raw: bool = False,
) -> tuple[list[Path], list[Path]]:
    source_beta_summary_root = resolve_source_beta_summary_root(source_beta_summary_root)
    if from_raw or not beta_summary_files(source_beta_summary_root):
        summary_outputs = build_unit_summaries_from_raw(raw_root, beta_summary_root)
        source_beta_summary_root = beta_summary_root
    else:
        summary_outputs = link_beta_summaries(source_beta_summary_root, beta_summary_root)
    scale_value = probe_scale_value(source_beta_summary_root)
    if selected_points_csv is not None and selected_points_csv.exists():
        input_outputs = write_figure_inputs_from_selected_points(
            selected_points_csv,
            figure_input_root,
            scale_value=scale_value,
        )
    else:
        input_outputs = write_figure_inputs_from_beta_summaries(
            source_beta_summary_root,
            figure_input_root,
            scale_value=scale_value,
        )
    return summary_outputs, input_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DNN sampling summarized outputs and figure-input CSVs.")
    parser.add_argument("--source-beta-summary-root", type=Path, default=DEFAULT_BETA_SUMMARY_ROOT)
    parser.add_argument("--beta-summary-root", type=Path, default=DEFAULT_BETA_SUMMARY_ROOT)
    parser.add_argument("--figure-input-root", type=Path, default=DEFAULT_FIGURE_INPUT_ROOT)
    parser.add_argument("--selected-points-csv", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--from-raw", action="store_true", help="Rebuild unit_summary/beta_*.csv from raw unit_summary.json files.")
    args = parser.parse_args()

    summary_outputs, input_outputs = build_summarized_outputs(
        args.source_beta_summary_root.resolve(),
        args.beta_summary_root.resolve(),
        args.figure_input_root.resolve(),
        args.selected_points_csv.resolve() if args.selected_points_csv is not None else None,
        raw_root=args.raw_root.resolve(),
        from_raw=bool(args.from_raw),
    )
    print(f"beta_summary_files={len(summary_outputs)} root={args.beta_summary_root.resolve()}")
    print(f"figure_input_files={len(input_outputs)} root={args.figure_input_root.resolve()}")


if __name__ == "__main__":
    main()
