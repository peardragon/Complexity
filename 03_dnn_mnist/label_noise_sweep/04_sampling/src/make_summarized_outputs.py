from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .utils.dnn_model import P
except ImportError:
    from utils.dnn_model import P


SAMPLING_ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = SAMPLING_ROOT.parent
CONFIG_ROOT = SAMPLING_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default.json"
SUMMARY_ROOT = SAMPLING_ROOT / "summarized_outputs"
UNIT_SUMMARY_ROOT = SUMMARY_ROOT / "unit_summary"
DEFAULT_ETA_SUMMARY_ROOT = UNIT_SUMMARY_ROOT
DEFAULT_FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "logZ_split"
DEFAULT_RAW_ROOT = SAMPLING_ROOT / "raw_outputs" / "shell_pool"
SHELL_POOL_ROOT = DEFAULT_RAW_ROOT


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


def eta_slug(eta: float) -> str:
    return f"{float(eta):.2f}".replace(".", "p")


def eta_from_name(path: Path) -> float:
    match = re.search(r"eta_(\d+p\d+)", path.stem)
    if not match:
        raise ValueError(f"cannot parse eta from {path.name}")
    return float(match.group(1).replace("p", "."))


def eta_from_token(token: str) -> float:
    token = str(token)
    if token.startswith("noise_eta_"):
        token = token.removeprefix("noise_eta_")
    elif token.startswith("eta_"):
        token = token.removeprefix("eta_")
    return float(token.replace("p", "."))


def tagged_int(name: str, prefix: str) -> int:
    return int(str(name).removeprefix(prefix))


def relative_to_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(LABEL_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def eta_summary_files(root: Path) -> list[Path]:
    return sorted(root.glob("eta_*.csv"), key=eta_from_name)


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def config_etas(config: dict[str, Any]) -> list[float]:
    return [float(value) for value in (config.get("ensemble") or {}).get("condition_values", [])]


def config_ref_count(config: dict[str, Any]) -> int:
    return int((config.get("ensemble") or {}).get("references_per_condition", 30))


def config_radii(config: dict[str, Any]) -> list[float]:
    sampling = config.get("sampling") or {}
    if sampling.get("radii"):
        return [float(value) for value in sampling["radii"]]
    start = float(sampling.get("radius_start", 0.01))
    stop = float(sampling.get("radius_stop", 1.0))
    step = float(sampling.get("radius_step", 0.01))
    count = int(round((stop - start) / step)) + 1
    return [round(start + idx * step, 10) for idx in range(count)]


def radius_from_token(token: str) -> float:
    return float(str(token).removeprefix("r_").replace("p", "."))


def clear_outputs(path: Path, pattern: str) -> None:
    if not path.exists():
        return
    for item in path.glob(pattern):
        item.unlink()


def clear_generated_summary_root(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "unit_summary").mkdir(parents=True, exist_ok=True)
    (output_root / "figure_inputs" / "logZ_split").mkdir(parents=True, exist_ok=True)


def unit_summary_row(path: Path, raw_root: Path = SHELL_POOL_ROOT) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rel = path.relative_to(raw_root)
    noise_eta = rel.parts[0]
    ref_dir = rel.parts[1]
    radius_dir = rel.parts[2]
    eta = float(payload.get("eta", eta_from_token(noise_eta)))
    ref_id = int(payload.get("ref_id", tagged_int(ref_dir, "ref_")))
    radius = float(payload.get("radius", radius_from_token(radius_dir)))
    scale_value = float(payload.get("P_params", payload.get("P", P)))

    split0 = float(payload.get("split0_logZ", payload.get("split0_logZ_inf", np.nan)))
    split1 = float(payload.get("split1_logZ", payload.get("split1_logZ_inf", np.nan)))
    signed_split = (split0 - split1) / scale_value if np.isfinite(split0) and np.isfinite(split1) else np.nan
    split_diff = abs(signed_split) if np.isfinite(signed_split) else payload.get("split_logZ_per_P_diff", "")
    dsplit = float(payload.get("split_dlogZ_dr_per_P_diff", np.nan))
    if not np.isfinite(dsplit):
        d0 = float(payload.get("dlogZ_dr_split0", np.nan))
        d1 = float(payload.get("dlogZ_dr_split1", np.nan))
        dsplit = abs(d0 - d1) / scale_value if np.isfinite(d0) and np.isfinite(d1) else np.nan

    return {
        "stage": "03_dnn_mnist",
        "block": "label_noise_sweep",
        "condition_name": "eta",
        "condition_value": eta,
        "dataset_id": int(payload.get("dataset_id", 0)),
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
        "split0_logZ": payload.get("split0_logZ", payload.get("split0_logZ_inf", "")),
        "split1_logZ": payload.get("split1_logZ", payload.get("split1_logZ_inf", "")),
        "signed_split_logZ_per_scale": f"{signed_split:.6g}" if np.isfinite(signed_split) else "",
        "split_logZ_per_scale_diff": split_diff,
        "dlogZ_dr_split0": payload.get("dlogZ_dr_split0", ""),
        "dlogZ_dr_split1": payload.get("dlogZ_dr_split1", ""),
        "split_dlogZ_dr_per_scale_diff": dsplit if np.isfinite(dsplit) else "",
        "scale_name": "P",
        "scale_value": scale_value,
        "ess_fraction": payload.get("ess_frac", payload.get("ess_fraction", "")),
        "smc_min_cess_fraction": payload.get("smc_min_cess_fraction", ""),
        "smc_completed": str(bool(payload.get("smc_completed", False))).lower(),
        "sampler_method": payload.get("sampler_method", ""),
        "source_path": relative_to_label(path),
    }


def build_unit_summaries_from_raw(raw_root: Path, output_root: Path) -> list[Path]:
    eta_dirs = sorted(raw_root.glob("noise_eta_*"), key=lambda path: eta_from_token(path.name))
    if not eta_dirs:
        raise FileNotFoundError(f"no noise_eta_* directories found under {raw_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "eta_*.csv")

    outputs: list[Path] = []
    for eta_dir in eta_dirs:
        eta = eta_from_token(eta_dir.name)
        out = output_root / f"eta_{eta_slug(eta)}.csv"
        wrote_any = False
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=UNIT_SUMMARY_FIELDS)
            writer.writeheader()
            for ref_dir in sorted(eta_dir.glob("ref_*"), key=lambda path: tagged_int(path.name, "ref_")):
                for radius_dir in sorted(ref_dir.glob("r_*"), key=lambda path: radius_from_token(path.name)):
                    path = radius_dir / "unit_summary.json"
                    if path.exists():
                        writer.writerow(unit_summary_row(path, raw_root=raw_root))
                        wrote_any = True
        if wrote_any:
            outputs.append(out)
        else:
            out.unlink(missing_ok=True)
    if not outputs:
        raise FileNotFoundError(f"no unit_summary.json files found under {raw_root}")
    return outputs


def probe_scale_value(summary_root: Path) -> int:
    first = next(iter(eta_summary_files(summary_root)), None)
    if first is None:
        raise FileNotFoundError(f"no eta_*.csv files found under {summary_root}")
    values = pd.read_csv(first, usecols=["scale_value"], nrows=1)["scale_value"]
    return int(round(float(values.iloc[0])))


def link_eta_summaries(source_root: Path, output_root: Path) -> list[Path]:
    source_files = eta_summary_files(source_root)
    if not source_files:
        raise FileNotFoundError(f"no eta_*.csv files found under {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    same_root = source_root.resolve() == output_root.resolve()
    if not same_root:
        clear_outputs(output_root, "eta_*.csv")

    outputs: list[Path] = []
    for source in source_files:
        target = output_root / f"eta_{eta_slug(eta_from_name(source))}.csv"
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


def write_figure_inputs_from_eta_summaries(summary_root: Path, output_root: Path, *, scale_value: int) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "eta_*.csv")
    outputs: list[Path] = []
    usecols = ["condition_value", "radius", "signed_split_logZ_per_scale", "split_logZ_per_scale_diff"]
    for source in eta_summary_files(summary_root):
        eta = eta_from_name(source)
        frame = pd.read_csv(source, usecols=usecols)
        frame = frame.loc[selected_radius_mask(frame["radius"])].copy()
        frame = frame.rename(
            columns={
                "condition_value": "eta",
                "radius": "r",
                "split_logZ_per_scale_diff": "split_logZ_per_P_diff",
                "signed_split_logZ_per_scale": "signed_split_logZ_per_P_diff",
            }
        )
        frame["P"] = scale_value
        frame = frame[["eta", "P", "r", "split_logZ_per_P_diff", "signed_split_logZ_per_P_diff"]].sort_values(["eta", "r"])
        out = output_root / f"eta_{eta_slug(eta)}.csv"
        frame.to_csv(out, index=False)
        outputs.append(out)
    return outputs


def write_figure_inputs_from_selected_points(
    selected_points_csv: Path,
    output_root: Path,
    *,
    scale_value: int,
) -> list[Path]:
    if not selected_points_csv.exists():
        raise FileNotFoundError(selected_points_csv)
    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "eta_*.csv")

    frame = pd.read_csv(selected_points_csv)
    required = {"eta", "r", "signed_split_logZ_per_P_diff"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{selected_points_csv} missing columns: {sorted(missing)}")

    frame["eta"] = pd.to_numeric(frame["eta"], errors="coerce")
    frame["P"] = scale_value
    frame["r"] = pd.to_numeric(frame["r"], errors="coerce")
    frame["signed_split_logZ_per_P_diff"] = pd.to_numeric(frame["signed_split_logZ_per_P_diff"], errors="coerce")
    frame["split_logZ_per_P_diff"] = frame["signed_split_logZ_per_P_diff"].abs()
    frame = frame.dropna(subset=["eta", "r", "signed_split_logZ_per_P_diff"])
    frame = frame.sort_values(["eta", "r"]).reset_index(drop=True)

    outputs: list[Path] = []
    for eta, group in frame.groupby("eta", sort=True):
        out = output_root / f"eta_{eta_slug(float(eta))}.csv"
        group[["eta", "P", "r", "split_logZ_per_P_diff", "signed_split_logZ_per_P_diff"]].to_csv(out, index=False)
        outputs.append(out)
    return outputs


def build_summarized_outputs(
    source_eta_summary_root: Path,
    eta_summary_root: Path,
    figure_input_root: Path,
    selected_points_csv: Path | None,
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    from_raw: bool = False,
) -> tuple[list[Path], list[Path]]:
    source_eta_summary_root = source_eta_summary_root.resolve()
    if from_raw or not eta_summary_files(source_eta_summary_root):
        summary_outputs = build_unit_summaries_from_raw(raw_root, eta_summary_root)
        source_eta_summary_root = eta_summary_root
    else:
        summary_outputs = link_eta_summaries(source_eta_summary_root, eta_summary_root)
    scale_value = probe_scale_value(source_eta_summary_root)
    if selected_points_csv is not None and selected_points_csv.exists():
        input_outputs = write_figure_inputs_from_selected_points(
            selected_points_csv,
            figure_input_root,
            scale_value=scale_value,
        )
    else:
        input_outputs = write_figure_inputs_from_eta_summaries(
            source_eta_summary_root,
            figure_input_root,
            scale_value=scale_value,
        )
    return summary_outputs, input_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-noise sampling summarized outputs and figure-input CSVs.")
    parser.add_argument("--source-eta-summary-root", type=Path, default=DEFAULT_ETA_SUMMARY_ROOT)
    parser.add_argument("--eta-summary-root", type=Path, default=DEFAULT_ETA_SUMMARY_ROOT)
    parser.add_argument("--figure-input-root", type=Path, default=DEFAULT_FIGURE_INPUT_ROOT)
    parser.add_argument("--selected-points-csv", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--from-raw", action="store_true", help="Rebuild unit_summary/eta_*.csv from raw unit_summary.json files.")
    args = parser.parse_args()

    summary_outputs, input_outputs = build_summarized_outputs(
        args.source_eta_summary_root.resolve(),
        args.eta_summary_root.resolve(),
        args.figure_input_root.resolve(),
        args.selected_points_csv.resolve() if args.selected_points_csv is not None else None,
        raw_root=args.raw_root.resolve(),
        from_raw=bool(args.from_raw),
    )
    print(f"eta_summary_files={len(summary_outputs)} root={args.eta_summary_root.resolve()}")
    print(f"figure_input_files={len(input_outputs)} root={args.figure_input_root.resolve()}")


if __name__ == "__main__":
    main()
