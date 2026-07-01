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
MANUAL_ROOT = SAMPLING_ROOT.parent
CONFIG_ROOT = SAMPLING_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default.json"
SUMMARY_ROOT = SAMPLING_ROOT / "summarized_outputs"
UNIT_SUMMARY_ROOT = SUMMARY_ROOT / "unit_summary"
DEFAULT_RULE_SUMMARY_ROOT = UNIT_SUMMARY_ROOT
DEFAULT_FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs" / "logZ_split"
DEFAULT_RAW_ROOT = SAMPLING_ROOT / "raw_outputs" / "shell_pool"
SHELL_POOL_ROOT = DEFAULT_RAW_ROOT


UNIT_SUMMARY_FIELDS = [
    "stage",
    "block",
    "condition_name",
    "condition_value",
    "condition_label",
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


RULE_LABELS = {
    "rule_001": "very_low_tv_spectral_teacher",
    "rule_002": "real_even_odd",
    "rule_003": "teacher_nn",
    "rule_004": "random_label",
}


def rule_from_name(path: Path) -> str:
    match = re.search(r"(rule_\d{3})", path.stem)
    if not match:
        raise ValueError(f"cannot parse rule from {path.name}")
    return match.group(1)


def tagged_int(name: str, prefix: str) -> int:
    return int(str(name).removeprefix(prefix))


def radius_from_token(token: str) -> float:
    return float(str(token).removeprefix("r_").replace("p", "."))


def relative_to_manual(path: Path) -> str:
    try:
        return path.resolve().relative_to(MANUAL_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def rule_summary_files(root: Path) -> list[Path]:
    return sorted(root.glob("rule_*.csv"), key=rule_from_name)


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def config_rules(config: dict[str, Any]) -> list[str]:
    return [str(value) for value in (config.get("ensemble") or {}).get("condition_values", [])]


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
    rule_id = rel.parts[0]
    ref_dir = rel.parts[1]
    radius_dir = rel.parts[2]
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
        "block": "manual_rules",
        "condition_name": "rule",
        "condition_value": rule_id,
        "condition_label": RULE_LABELS.get(rule_id, payload.get("rule", rule_id)),
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
        "source_path": relative_to_manual(path),
    }


def build_unit_summaries_from_raw(raw_root: Path, output_root: Path) -> list[Path]:
    rule_dirs = sorted(raw_root.glob("rule_*"), key=lambda path: path.name)
    if not rule_dirs:
        raise FileNotFoundError(f"no rule_* directories found under {raw_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "rule_*.csv")

    outputs: list[Path] = []
    for rule_dir in rule_dirs:
        out = output_root / f"{rule_dir.name}.csv"
        wrote_any = False
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=UNIT_SUMMARY_FIELDS)
            writer.writeheader()
            for ref_dir in sorted(rule_dir.glob("ref_*"), key=lambda path: tagged_int(path.name, "ref_")):
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
    first = next(iter(rule_summary_files(summary_root)), None)
    if first is None:
        raise FileNotFoundError(f"no rule_*.csv files found under {summary_root}")
    values = pd.read_csv(first, usecols=["scale_value"], nrows=1)["scale_value"]
    return int(round(float(values.iloc[0])))


def link_rule_summaries(source_root: Path, output_root: Path) -> list[Path]:
    source_files = rule_summary_files(source_root)
    if not source_files:
        raise FileNotFoundError(f"no rule_*.csv files found under {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    same_root = source_root.resolve() == output_root.resolve()
    if not same_root:
        clear_outputs(output_root, "rule_*.csv")

    outputs: list[Path] = []
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
    return outputs


def selected_radius_mask(radius: pd.Series) -> pd.Series:
    scaled = np.rint(pd.to_numeric(radius, errors="coerce") * 100).astype("Int64")
    return scaled.mod(5).eq(0)


def write_figure_inputs_from_rule_summaries(summary_root: Path, output_root: Path, *, scale_value: int) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_root, "rule_*.csv")
    outputs: list[Path] = []
    usecols = ["condition_value", "condition_label", "radius", "signed_split_logZ_per_scale", "split_logZ_per_scale_diff"]
    for source in rule_summary_files(summary_root):
        frame = pd.read_csv(source, usecols=usecols)
        frame = frame.loc[selected_radius_mask(frame["radius"])].copy()
        frame = frame.rename(
            columns={
                "condition_value": "rule_id",
                "condition_label": "rule",
                "radius": "r",
                "split_logZ_per_scale_diff": "split_logZ_per_P_diff",
                "signed_split_logZ_per_scale": "signed_split_logZ_per_P_diff",
            }
        )
        frame["P"] = scale_value
        frame = frame[["rule_id", "rule", "P", "r", "split_logZ_per_P_diff", "signed_split_logZ_per_P_diff"]].sort_values(["rule_id", "r"])
        out = output_root / source.name
        frame.to_csv(out, index=False)
        outputs.append(out)
    return outputs


def build_summarized_outputs(
    source_rule_summary_root: Path,
    rule_summary_root: Path,
    figure_input_root: Path,
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    from_raw: bool = False,
) -> tuple[list[Path], list[Path]]:
    source_rule_summary_root = source_rule_summary_root.resolve()
    if from_raw or not rule_summary_files(source_rule_summary_root):
        summary_outputs = build_unit_summaries_from_raw(raw_root, rule_summary_root)
        source_rule_summary_root = rule_summary_root
    else:
        summary_outputs = link_rule_summaries(source_rule_summary_root, rule_summary_root)
    scale_value = probe_scale_value(source_rule_summary_root)
    input_outputs = write_figure_inputs_from_rule_summaries(
        source_rule_summary_root,
        figure_input_root,
        scale_value=scale_value,
    )
    return summary_outputs, input_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual-rule sampling summarized outputs and figure-input CSVs.")
    parser.add_argument("--source-rule-summary-root", type=Path, default=DEFAULT_RULE_SUMMARY_ROOT)
    parser.add_argument("--rule-summary-root", type=Path, default=DEFAULT_RULE_SUMMARY_ROOT)
    parser.add_argument("--figure-input-root", type=Path, default=DEFAULT_FIGURE_INPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--from-raw", action="store_true", help="Rebuild unit_summary/rule_*.csv from raw unit_summary.json files.")
    args = parser.parse_args()

    summary_outputs, input_outputs = build_summarized_outputs(
        args.source_rule_summary_root.resolve(),
        args.rule_summary_root.resolve(),
        args.figure_input_root.resolve(),
        raw_root=args.raw_root.resolve(),
        from_raw=bool(args.from_raw),
    )
    print(f"rule_summary_files={len(summary_outputs)} root={args.rule_summary_root.resolve()}")
    print(f"figure_input_files={len(input_outputs)} root={args.figure_input_root.resolve()}")


if __name__ == "__main__":
    main()
