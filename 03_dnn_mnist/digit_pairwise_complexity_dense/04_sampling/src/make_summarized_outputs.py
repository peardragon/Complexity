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
PAIRWISE_ROOT = SAMPLING_ROOT.parent
SUMMARY_ROOT = SAMPLING_ROOT / "summarized_outputs"
UNIT_SUMMARY_ROOT = SUMMARY_ROOT / "unit_summary"
DEFAULT_PAIR_SUMMARY_ROOT = UNIT_SUMMARY_ROOT
DEFAULT_FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "logZ_split"
DEFAULT_RAW_ROOT = SAMPLING_ROOT / "raw_outputs" / "shell_pool"
SHELL_POOL_ROOT = DEFAULT_RAW_ROOT
AGGREGATE_UNIT_TABLE = UNIT_SUMMARY_ROOT / "shell_summary_by_unit_with_phi_derivatives.csv"


UNIT_SUMMARY_FIELDS = [
    "stage",
    "block",
    "condition_name",
    "condition_value",
    "condition_label",
    "pair_id",
    "pair_label",
    "digit_a",
    "digit_b",
    "pair_order",
    "pair_rank_complexity_desc",
    "complexity_mean",
    "dataset_id",
    "ref_id",
    "ref_path_id",
    "radius",
    "radius_path_id",
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
    "theta_path",
    "dataset_path",
    "samples_path",
    "unit_summary_path",
    "source_path",
]


def pair_sort_key(text: str) -> tuple[int, int]:
    match = re.search(r"pair_(\d+)_(\d+)", text)
    if not match:
        raise ValueError(f"cannot parse pair id from {text}")
    return int(match.group(1)), int(match.group(2))


def tagged_int(name: str, prefix: str) -> int:
    return int(str(name).removeprefix(prefix))


def radius_from_token(token: str) -> float:
    return float(str(token).removeprefix("r_").replace("p", "."))


def relative_to_pairwise(path: Path) -> str:
    try:
        return path.resolve().relative_to(PAIRWISE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def pair_summary_files(root: Path) -> list[Path]:
    return sorted(root.glob("pair_*.csv"), key=lambda path: pair_sort_key(path.stem))


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
    pair_id = str(payload.get("pair_id", rel.parts[0]))
    ref_dir = rel.parts[1]
    radius_dir = rel.parts[2]
    pair_label = str(payload.get("pair_label", pair_id.removeprefix("pair_").replace("_", "/")))
    ref_id = int(payload.get("ref_id", tagged_int(ref_dir, "ref_") - 1))
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
        "block": "digit_pairwise_complexity_dense",
        "condition_name": "digit_pair",
        "condition_value": pair_id,
        "condition_label": pair_label,
        "pair_id": pair_id,
        "pair_label": pair_label,
        "digit_a": int(payload.get("digit_a", pair_sort_key(pair_id)[0])),
        "digit_b": int(payload.get("digit_b", pair_sort_key(pair_id)[1])),
        "pair_order": int(payload.get("pair_order", 0)),
        "pair_rank_complexity_desc": int(payload.get("pair_rank_complexity_desc", 0)),
        "complexity_mean": payload.get("complexity_mean", ""),
        "dataset_id": int(payload.get("dataset_id", 0)),
        "ref_id": ref_id,
        "ref_path_id": str(payload.get("ref_path_id", ref_dir)),
        "radius": radius,
        "radius_path_id": str(payload.get("radius_path_id", radius_dir)),
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
        "sampler_method": payload.get("sampler_method", "exact_shell_l2_vmf_adaptive_ce_tempered_smc"),
        "theta_path": payload.get("theta_path", ""),
        "dataset_path": payload.get("dataset_path", ""),
        "samples_path": payload.get("samples_path", ""),
        "unit_summary_path": payload.get("unit_summary_path", relative_to_pairwise(path)),
        "source_path": relative_to_pairwise(path),
    }


def build_unit_summaries_from_raw(raw_root: Path, output_root: Path) -> list[Path]:
    pair_dirs = sorted(raw_root.glob("pair_*"), key=lambda path: pair_sort_key(path.name))
    if not pair_dirs:
        raise FileNotFoundError(f"no pair_* directories found under {raw_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "pair_*.csv")

    outputs: list[Path] = []
    aggregate_rows: list[dict[str, Any]] = []
    for pair_dir in pair_dirs:
        out = output_root / f"{pair_dir.name}.csv"
        wrote_any = False
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNIT_SUMMARY_FIELDS)
            writer.writeheader()
            for ref_dir in sorted(pair_dir.glob("ref_*"), key=lambda path: tagged_int(path.name, "ref_")):
                for radius_dir in sorted(ref_dir.glob("r_*"), key=lambda path: radius_from_token(path.name)):
                    path = radius_dir / "unit_summary.json"
                    if path.exists():
                        row = unit_summary_row(path, raw_root=raw_root)
                        writer.writerow(row)
                        aggregate_rows.append(row)
                        wrote_any = True
        if wrote_any:
            outputs.append(out)
        else:
            out.unlink(missing_ok=True)
    if not outputs:
        raise FileNotFoundError(f"no unit_summary.json files found under {raw_root}")
    pd.DataFrame(aggregate_rows).to_csv(AGGREGATE_UNIT_TABLE, index=False)
    return outputs


def probe_scale_value(summary_root: Path) -> int:
    first = next(iter(pair_summary_files(summary_root)), None)
    if first is None:
        raise FileNotFoundError(f"no pair_*.csv files found under {summary_root}")
    values = pd.read_csv(first, usecols=["scale_value"], nrows=1)["scale_value"]
    return int(round(float(values.iloc[0])))


def link_pair_summaries(source_root: Path, output_root: Path) -> list[Path]:
    source_files = pair_summary_files(source_root)
    if not source_files:
        raise FileNotFoundError(f"no pair_*.csv files found under {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    same_root = source_root.resolve() == output_root.resolve()
    if not same_root:
        clear_outputs(output_root, "pair_*.csv")
    outputs: list[Path] = []
    aggregate_frames: list[pd.DataFrame] = []
    for source in source_files:
        target = output_root / source.name
        if same_root:
            outputs.append(target)
        else:
            try:
                os.link(source, target)
            except OSError:
                target.symlink_to(source)
            outputs.append(target)
        aggregate_frames.append(pd.read_csv(target))
    if aggregate_frames:
        pd.concat(aggregate_frames, ignore_index=True).to_csv(AGGREGATE_UNIT_TABLE, index=False)
    return outputs


def selected_radius_mask(radius: pd.Series) -> pd.Series:
    scaled = np.rint(pd.to_numeric(radius, errors="coerce") * 100).astype("Int64")
    return scaled.mod(5).eq(0)


def write_figure_inputs_from_pair_summaries(summary_root: Path, output_root: Path, *, scale_value: int) -> list[Path]:
    del scale_value
    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "pair_*.csv")
    outputs: list[Path] = []
    usecols = [
        "pair_id",
        "pair_label",
        "radius",
        "signed_split_logZ_per_scale",
        "split_logZ_per_scale_diff",
    ]
    for source in pair_summary_files(summary_root):
        frame = pd.read_csv(source, usecols=usecols)
        frame = frame.loc[selected_radius_mask(frame["radius"])].copy()
        frame = frame.rename(
            columns={
                "pair_label": "pair",
                "radius": "r",
                "split_logZ_per_scale_diff": "split_logZ_per_P_diff",
                "signed_split_logZ_per_scale": "signed_split_logZ_per_P_diff",
            }
        )
        frame["P"] = P
        frame = frame[
            [
                "pair_id",
                "pair",
                "P",
                "r",
                "split_logZ_per_P_diff",
                "signed_split_logZ_per_P_diff",
            ]
        ].sort_values(["pair_id", "r"])
        out = output_root / source.name
        frame.to_csv(out, index=False)
        outputs.append(out)
    return outputs


def build_summarized_outputs(
    source_pair_summary_root: Path,
    pair_summary_root: Path,
    figure_input_root: Path,
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    from_raw: bool = False,
) -> tuple[list[Path], list[Path]]:
    source_pair_summary_root = source_pair_summary_root.resolve()
    if from_raw or not pair_summary_files(source_pair_summary_root):
        summary_outputs = build_unit_summaries_from_raw(raw_root, pair_summary_root)
        source_pair_summary_root = pair_summary_root
    else:
        summary_outputs = link_pair_summaries(source_pair_summary_root, pair_summary_root)
    scale_value = probe_scale_value(source_pair_summary_root)
    input_outputs = write_figure_inputs_from_pair_summaries(
        source_pair_summary_root,
        figure_input_root,
        scale_value=scale_value,
    )
    return summary_outputs, input_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build digit-pair sampling summarized outputs and figure-input CSVs.")
    parser.add_argument("--source-pair-summary-root", type=Path, default=DEFAULT_PAIR_SUMMARY_ROOT)
    parser.add_argument("--pair-summary-root", type=Path, default=DEFAULT_PAIR_SUMMARY_ROOT)
    parser.add_argument("--figure-input-root", type=Path, default=DEFAULT_FIGURE_INPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--from-raw", action="store_true", help="Rebuild unit_summary/pair_*.csv from raw unit_summary.json files.")
    args = parser.parse_args()
    summary_outputs, input_outputs = build_summarized_outputs(
        args.source_pair_summary_root.resolve(),
        args.pair_summary_root.resolve(),
        args.figure_input_root.resolve(),
        raw_root=args.raw_root.resolve(),
        from_raw=bool(args.from_raw),
    )
    print(f"pair_summary_files={len(summary_outputs)} root={args.pair_summary_root.resolve()}")
    print(f"figure_input_files={len(input_outputs)} root={args.figure_input_root.resolve()}")
    print(f"aggregate_unit_table={AGGREGATE_UNIT_TABLE.resolve()}")


if __name__ == "__main__":
    main()
