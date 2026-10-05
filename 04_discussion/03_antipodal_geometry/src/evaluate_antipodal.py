#!/usr/bin/env python3
"""Re-evaluate the published Appendix D symmetric response on retained references.

This command evaluates the loss only; it does not train references or run SMC.
The default invocation prints the plan. Existing condition files are skipped.
Run make_summarized_outputs.py afterward to build the four canonical tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

STAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = STAGE_ROOT.parents[1]
MNIST = ROOT / "03_dnn_mnist/label_noise_sweep"
sys.path.insert(0, str(MNIST / "04_sampling/src"))
from utils.modeling import P, torch_logits_batched
from utils.io_utils import load_json
from utils.production import resolve_replica_config
from utils.reference_training import load_selected_reference_pack

CONFIG = json.loads((STAGE_ROOT / "config/default.json").read_text())


def objective_each(theta, x, y):
    logits = torch_logits_batched(theta, x)
    ce = torch.nn.functional.softplus(-y[None, :] * logits).mean(dim=1)
    settings = CONFIG["evaluation"]
    return (
        float(settings["shell_inverse_temperature"]) * ce
        + float(settings["regularization_coefficient_after_shell_scaling"])
        * torch.sum(theta * theta, dim=1)
        / (2.0 * P)
    )


def _raw_root(output_dir: Path | None) -> Path:
    if output_dir is None:
        return STAGE_ROOT / "raw_outputs"
    return output_dir / STAGE_ROOT.name / "raw_outputs"


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def evaluate(condition: str, device: str, output_dir: Path | None, force: bool) -> Path:
    path = _raw_root(output_dir) / f"antipodal_{condition}.npz"
    if path.is_file() and not force:
        print(f"{path.name}: skipped_existing")
        return path
    settings = CONFIG["evaluation"]
    design = CONFIG["design"]
    datasets = int(design["datasets_per_condition"])
    references = int(design["references_per_dataset"])
    radii = np.asarray(settings["radii"], dtype=np.float64)
    directions = int(settings["directions"])
    chunk = int(settings["chunk_size"])
    configured_directions = int(design["matched_antipodal_directions_per_reference"])
    shell_beta = float(settings["shell_inverse_temperature"])
    regularization = float(settings["regularization_coefficient_after_shell_scaling"])
    if references <= 0:
        raise ValueError("reference count must be positive")
    if not 2 <= datasets <= 10:
        raise ValueError("dataset count must be between 2 and 10 inclusive")
    if radii.ndim != 1 or len(radii) < 2 or not np.all(np.isfinite(radii)):
        raise ValueError("radii must be a finite one-dimensional grid")
    if np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be positive and strictly increasing")
    if directions <= 0 or directions != configured_directions:
        raise ValueError("design/evaluation direction counts must agree and be positive")
    if chunk <= 0:
        raise ValueError("chunk size must be positive")
    if not math.isfinite(shell_beta) or shell_beta <= 0.0:
        raise ValueError("shell inverse temperature must be finite and positive")
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization coefficient must be finite and nonnegative")
    if int(settings["dataset_seed_stride"]) < references:
        raise ValueError("dataset seed stride must be at least the reference count")
    if int(settings["parameter_count"]) != P:
        raise ValueError(f"configured parameter count {settings['parameter_count']} != model P={P}")
    curvature = np.empty((datasets, references, len(radii), directions), dtype=np.float64)
    plus_increment = np.empty_like(curvature)
    minus_increment = np.empty_like(curvature)
    center_objective = np.empty((datasets, references), dtype=np.float64)
    minimum_margin = np.empty((datasets, references), dtype=np.float64)
    direction_hash = np.empty((datasets, references), dtype="U64")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source_config = load_json(MNIST / "04_sampling/config/default.json")
    for dataset in range(datasets):
        config = resolve_replica_config(source_config, dataset)
        data_path = (
            MNIST
            / "01_dataset/raw_outputs/datasets"
            / f"dataset_{dataset:03d}"
            / condition
            / "dataset.npz"
        )
        with np.load(data_path, allow_pickle=False) as data:
            x = torch.as_tensor(np.asarray(data["x_train"], dtype=np.float64), device=device)
            y = torch.as_tensor(np.asarray(data["y_train"], dtype=np.float64), device=device)
        selected = list(load_selected_reference_pack(config, condition))
        if len(selected) != references:
            raise ValueError(
                f"expected {references} references for {condition}/{dataset}, found {len(selected)}"
            )
        for reference, (value, _) in enumerate(selected):
            theta = torch.as_tensor(value, dtype=torch.float64, device=device)
            rng = np.random.default_rng(
                int(settings["base_seed"])
                + int(settings["dataset_seed_stride"]) * dataset
                + reference
            )
            unit_np = rng.normal(size=(directions, P))
            unit_np /= np.linalg.norm(unit_np, axis=1, keepdims=True)
            direction_hash[dataset, reference] = _sha256_array(unit_np)
            unit = torch.as_tensor(unit_np, dtype=torch.float64, device=device)
            with torch.no_grad():
                center = float(objective_each(theta[None, :], x, y)[0].cpu())
                margin = float((y * torch_logits_batched(theta[None, :], x)[0]).min().cpu())
                if margin <= 0:
                    raise ValueError(
                        "reference has nonzero training error: "
                        f"{condition}/{dataset}/{reference}"
                    )
                center_objective[dataset, reference] = center
                minimum_margin[dataset, reference] = margin
                for index, radius in enumerate(radii):
                    for start in range(0, directions, chunk):
                        delta = math.sqrt(P) * radius * unit[start:start+chunk]
                        values = objective_each(
                            torch.cat((theta[None, :] + delta, theta[None, :] - delta)),
                            x,
                            y,
                        ).cpu().numpy()
                        count = len(delta)
                        plus = values[:count] - center
                        minus = values[count:] - center
                        plus_increment[dataset, reference, index, start:start+count] = plus
                        minus_increment[dataset, reference, index, start:start+count] = minus
                        curvature[dataset, reference, index, start : start + count] = (
                            plus + minus
                        ) / (P * radius * radius)
        print(f"{condition}: dataset {dataset + 1}/{datasets}", flush=True)
    if not np.all(np.isfinite(curvature)):
        raise ValueError("nonfinite symmetric response")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        np.savez_compressed(
            stream,
            curvature=curvature,
            plus_increment=plus_increment,
            minus_increment=minus_increment,
            center_objective=center_objective,
            minimum_margin=minimum_margin,
            direction_hash=direction_hash,
            radii=radii,
            condition=np.asarray(condition),
            dataset_indices=np.arange(datasets, dtype=np.int64),
            reference_indices=np.arange(references, dtype=np.int64),
            direction_count=np.asarray(directions, dtype=np.int64),
            base_seed=np.asarray(int(settings["base_seed"]), dtype=np.int64),
        )
        temporary = Path(stream.name)
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONFIG["design"]["conditions"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace this condition NPZ.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Mirror the Stage-03 raw output below this root.",
    )
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {"condition": args.condition, "device": args.device, **CONFIG["evaluation"]},
                indent=2,
            )
        )
        return
    path = evaluate(args.condition, args.device, args.output_dir, args.force)
    print(f"raw Appendix D evaluation available: {path}")
    print(
        "after both conditions are available, run python "
        "04_discussion/03_antipodal_geometry/src/make_summarized_outputs.py --force"
        + (
            f" --output-dir {args.output_dir} --raw-dir {path.parent}"
            if args.output_dir
            else ""
        )
    )


if __name__ == "__main__":
    main()
