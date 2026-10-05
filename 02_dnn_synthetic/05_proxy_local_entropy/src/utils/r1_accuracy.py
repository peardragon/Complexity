"""Rebuild paper r=1 accuracy from saved reference-level capture results.

No training, sampling, or model evaluation is performed. The original capture
already contains each split's SMC-weighted accuracy and log-normalizer mixture.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

FIELDS = (
    "condition", "dataset_index", "reference_index", "radius",
    "weighted_train_accuracy", "split0_weighted_train_accuracy",
    "split1_weighted_train_accuracy", "split0_logz_mixture_weight",
    "split1_logz_mixture_weight", "replay_abs_difference",
)


def dimensions(config, stage):
    synthetic = config.get("domain") == "synthetic"
    if not math.isclose(float(config["shell_inv_temp_beta"]), 100.0):
        raise ValueError("saved captures use shell beta=100")
    if synthetic:
        conditions = [f"data_beta_{float(beta):.2f}".replace(".", "p") for beta in config["beta_values"]]
        datasets = list(range(int(config["selection"]["datasets_per_condition"])))
        references = int(config["selection"]["references_per_dataset"])
    else:
        if config["protocol"] == "label_noise_sweep":
            conditions = [row["condition"] for row in config["conditions"]]
        elif config["protocol"] == "digit_pairwise_sweep":
            path = Path(config["pair_selection"]["frozen_manifest"])
            manifest = json.loads((path if path.is_absolute() else stage / path).read_text())
            conditions = [row["pair_id"] for row in manifest["expected_selected_pairs"]]
        else:
            raise ValueError("unsupported MNIST accuracy protocol")
        datasets = [int(value) for value in config["dataset_indices"]]
        references = int(config["references_per_dataset"])
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("accuracy conditions must be unique and nonempty")
    if not datasets or len(set(datasets)) != len(datasets) or min(datasets) < 0 or references <= 0:
        raise ValueError("invalid dataset/reference coordinates")
    return synthetic, conditions, datasets, references


def capture_path(config, stage):
    path = Path(config.get("accuracy_capture_input", "frozen_inputs/r1_accuracy_per_reference.csv"))
    return path if path.is_absolute() else stage / path


def read_capture(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_original_replays(source_dir, config, stage):
    synthetic, conditions, _, _ = dimensions(config, stage)
    study = "label_noise" if config.get("protocol") == "label_noise_sweep" else "digit_pair"
    rows = []
    for path in sorted(Path(source_dir).rglob("*.json")):
        payload = json.loads(path.read_text())
        if "reference_results" not in payload:
            continue
        if not synthetic and payload.get("study") != study:
            continue
        if float(payload.get("gamma_ce", -1)) != 100.0 or float(payload.get("lambda_reg", -1)) != 1.0:
            raise ValueError(f"capture loss scale differs from the paper: {path}")
        condition = (f"data_beta_{float(payload['data_beta']):.2f}".replace(".", "p")
                     if synthetic else str(payload["condition"]))
        if condition not in conditions:
            raise ValueError(f"capture is outside the selected paper conditions: {path}")
        if payload.get("status") != "complete" or float(payload.get("radius", -1)) != 1.0:
            raise ValueError(f"incomplete/non-r1 capture: {path}")
        for row in payload["reference_results"]:
            if int(row["dataset_index"]) != int(payload["dataset_index"]):
                raise ValueError(f"capture dataset coordinate differs: {path}")
            if not synthetic and row["condition"] != condition:
                raise ValueError(f"capture condition differs: {path}")
            if synthetic and float(row["data_beta"]) != float(payload["data_beta"]):
                raise ValueError(f"capture beta differs: {path}")
            rows.append({
                "condition": condition,
                "dataset_index": int(row["dataset_index"]),
                "reference_index": int(row["reference_index"]),
                "radius": float(row["radius"]),
                "weighted_train_accuracy": float(row["weighted_train_accuracy"]),
                "split0_weighted_train_accuracy": float(row["split0_weighted_train_accuracy"]),
                "split1_weighted_train_accuracy": float(row["split1_weighted_train_accuracy"]),
                "split0_logz_mixture_weight": float(row["split0_logz_mixture_weight"]),
                "split1_logz_mixture_weight": float(row["split1_logz_mixture_weight"]),
                "replay_abs_difference": float(row["weighted_ce_replay_abs_difference"]
                                              if synthetic else row["production_replay_max_abs_difference"]),
            })
    return rows


def validate_rows(rows, config, stage):
    _, conditions, datasets, references = dimensions(config, stage)
    expected = {(condition, dataset, reference) for condition in conditions
                for dataset in datasets for reference in range(references)}
    indexed = {}
    for raw in rows:
        row = {key: raw[key] for key in FIELDS}
        row["condition"] = str(row["condition"])
        for key in ("dataset_index", "reference_index"):
            row[key] = int(row[key])
        for key in FIELDS[3:]:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"non-finite capture field: {key}")
        coordinate = (row["condition"], row["dataset_index"], row["reference_index"])
        if coordinate in indexed or coordinate not in expected:
            raise ValueError(f"duplicate/unexpected accuracy coordinate: {coordinate}")
        if row["radius"] != 1.0:
            raise ValueError("accuracy capture is not at r=1")
        if any(not 0 <= row[key] <= 1 for key in FIELDS[4:9]):
            raise ValueError("accuracy or mixture weight is outside [0,1]")
        w0, w1 = row["split0_logz_mixture_weight"], row["split1_logz_mixture_weight"]
        mixed = w0 * row["split0_weighted_train_accuracy"] + w1 * row["split1_weighted_train_accuracy"]
        if not math.isclose(w0+w1, 1.0, rel_tol=0, abs_tol=5e-12):
            raise ValueError("split mixture weights do not sum to one")
        if not math.isclose(mixed, row["weighted_train_accuracy"], rel_tol=0, abs_tol=5e-12):
            raise ValueError("capture accuracy differs from its split-logZ mixture")
        # The original r=1 replay used this numerical agreement check.
        if not 0 <= row["replay_abs_difference"] <= 5e-5:
            raise ValueError("saved replay did not match its production observable")
        indexed[coordinate] = row
    if set(indexed) != expected:
        raise ValueError(f"missing accuracy coordinates: {len(expected-set(indexed))}")
    return [indexed[(condition, dataset, reference)] for condition in conditions
            for dataset in datasets for reference in range(references)]


def aggregate(rows, config, stage):
    synthetic, conditions, datasets, references = dimensions(config, stage)
    rows = validate_rows(rows, config, stage)
    indexed = {(row["condition"], row["dataset_index"], row["reference_index"]):
               row["weighted_train_accuracy"] for row in rows}
    result = {}
    for condition in conditions:
        means = []
        for dataset in datasets:
            values = [indexed[(condition, dataset, reference)] for reference in range(references)]
            means.append(float(np.mean(values)) if synthetic else statistics.fmean(values))
        mean = float(np.mean(means)) if synthetic else statistics.fmean(means)
        sd = (float(np.std(means, ddof=1)) if synthetic else statistics.stdev(means)) if len(means)>1 else 0.0
        result[condition] = {"mean": mean, "se_across_datasets": sd / math.sqrt(len(means))}
    if synthetic:
        return {
            "schema_version": 1, "artifact_id": "synthetic_r1_weighted_accuracy_shell_beta_100_v1",
            "method": "SMC-particle weighted training accuracy at r=1; split log-normalizer mixture, reference mean within dataset, equal-weight mean and sample-SE across datasets",
            "radius": 1.0, "dataset_count": len(datasets), "references_per_dataset": references,
            "rows": [{"data_beta": float(beta), **result[condition]}
                     for beta,condition in zip(config["beta_values"], conditions, strict=True)],
        }
    return {
        "schema_version": "complexity-revised.mnist.r1-weighted-accuracy-seal.v1",
        "scale_id": config["scale_id"], "radius": 1.0,
        "dataset_count": len(datasets), "references_per_dataset": references,
        "aggregation_order": "SMC-particle-weighted, split-logZ-mixture, reference-mean-within-dataset, equal-dataset-mean-and-SE",
        "conditions": result,
    }


def from_capture(config, stage):
    return aggregate(read_capture(capture_path(config, stage)), config, stage)


def compare(actual, expected):
    for key in ("radius", "dataset_count", "references_per_dataset"):
        if actual[key] != expected[key]:
            raise ValueError(f"accuracy authority coordinate differs: {key}")
    def values(payload):
        if "rows" in payload:
            return {f"data_beta_{float(row['data_beta']):.2f}".replace(".", "p"): row for row in payload["rows"]}
        return payload["conditions"]
    left, right = values(actual), values(expected)
    if set(left) != set(right):
        raise ValueError("accuracy authority conditions differ")
    difference = 0.0
    for condition in left:
        for key in ("mean", "se_across_datasets"):
            a,b = float(left[condition][key]),float(right[condition][key])
            if not math.isfinite(b) or not math.isclose(a,b,rel_tol=0,abs_tol=1e-12):
                raise ValueError(f"accuracy authority differs: {condition}/{key}")
            difference = max(difference,abs(a-b))
    return difference


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(stage):
    parser = argparse.ArgumentParser(description="Rebuild the paper r=1 accuracy JSON from saved captures; no GPU or SMC.")
    parser.add_argument("--config", type=Path, default=stage / "config/default.json")
    parser.add_argument("--source-dir", type=Path, help="Original saved r=1 replay JSON directory.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only and (args.execute or args.force):
        parser.error("--check-only is read-only")
    config = json.loads(args.config.read_text())
    destination = (args.output_dir or stage / "summarized_outputs") / "r1_weighted_accuracy.json"
    source = capture_path(config, stage)
    capture_destination = ((args.output_dir / source.name) if args.output_dir else source) if args.source_dir else None
    outputs = [destination] + ([capture_destination] if capture_destination else [])
    if (args.execute or args.force) and not args.force and all(path.is_file() for path in outputs):
        print(json.dumps({"status": "skipped_existing", "output": str(destination)}))
        return
    rows = (read_original_replays(args.source_dir, config, stage) if args.source_dir else read_capture(source))
    rows = validate_rows(rows, config, stage)
    payload = aggregate(rows, config, stage)
    if args.check_only:
        difference = compare(payload, json.loads(destination.read_text()))
        status = "validated"
    elif args.execute or args.force:
        if args.force or not destination.is_file():
            atomic_write(destination, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
        if capture_destination and (args.force or not capture_destination.is_file()):
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream,fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
            atomic_write(capture_destination, stream.getvalue())
        difference = None
        status = "written"
    else:
        difference = None
        status = "planned"
    print(json.dumps({"status": status, "references": len(rows), "output": str(destination),
                      "max_numerical_difference": difference, "sampling_executed": False}, indent=2))
