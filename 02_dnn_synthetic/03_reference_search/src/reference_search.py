from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2].resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LOCAL_UTILS_PACKAGE = "_reference_search_utils"


def _install_local_utils_package() -> None:
    if LOCAL_UTILS_PACKAGE in sys.modules:
        return
    utils_init = SCRIPT_DIR / "utils" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        LOCAL_UTILS_PACKAGE,
        utils_init,
        submodule_search_locations=[str(utils_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local reference-search utils package from {utils_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[LOCAL_UTILS_PACKAGE] = module
    spec.loader.exec_module(module)


_install_local_utils_package()

from _reference_search_utils.defaults import DEFAULT_CONFIG
from _reference_search_utils.io_utils import ensure_dir, load_json, now_iso, save_csv, save_json, start_verbose_print_capture
from _reference_search_utils.rescue import summarize_and_select_reference_candidates


EXCLUDED_SUMMARY_OUTPUT_NAMES = {
    "candidate_refs.json",
    "valid_refs_manifest.json",
    "invalid_for_sampling.json",
    "invalid_refs_manifest.json",
}


def _hash_config(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def _repo_root(part_root: Path) -> Path:
    return Path(part_root).resolve().parents[1]


def _resolve_repo_path(part_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _repo_root(part_root) / path


def _repo_relative(part_root: Path, path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(_repo_root(part_root)).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _resolve_recorded_path(part_root: Path, path: str | Path) -> Path:
    return _resolve_repo_path(part_root, path)


def _beta_slug(value: str | float) -> str:
    text = str(value).strip()
    for prefix in ("cell_beta_", "beta_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return f"{float(text.replace('p', '.')):.2f}".replace(".", "p")


def _reference_cell_id(row_or_value: dict[str, Any] | str | float) -> str:
    if isinstance(row_or_value, dict):
        value = row_or_value.get("cell_id", row_or_value.get("beta_ising"))
    else:
        value = row_or_value
    if value is None:
        raise KeyError("reference-search rows must include cell_id or beta_ising")
    return f"beta_{_beta_slug(value)}"


def merged_config(config_path: Path | None = None, upstream_manifest: Path | None = None, force: bool = False) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    file_config: dict[str, Any] = {}
    if config_path is not None and Path(config_path).exists():
        file_config = load_json(Path(config_path), {}) or {}
        config.update(file_config)
    if "references_per_dataset" in file_config and "selected_refs_per_dataset" not in file_config:
        config["selected_refs_per_dataset"] = int(file_config["references_per_dataset"])
    if upstream_manifest is not None:
        config["upstream_manifest"] = str(upstream_manifest)
    config["force"] = bool(force or config.get("force", False))
    return config


def discover_upstream_manifest(part_root: Path, config: dict[str, Any]) -> Path:
    configured = config.get("upstream_manifest")
    if configured:
        candidate = _resolve_repo_path(part_root, configured)
        if candidate.exists():
            return candidate
    dataset_root = Path(part_root).resolve().parent / "01_dataset"
    candidates = [
        *sorted((dataset_root / "raw_outputs").glob("*/dataset_index.csv")),
        dataset_root / "raw_outputs" / "dataset_index.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not discover an upstream dataset_index.csv. Pass --upstream-manifest "
        "pointing to a dataset_index.csv or a manifest that lists one."
    )


def _read_dataset_index(part_root: Path, upstream_manifest: Path) -> list[dict[str, str]]:
    upstream_path = _resolve_repo_path(part_root, upstream_manifest)
    manifest = load_json(upstream_path, None)
    candidates: list[Path] = []
    if isinstance(manifest, dict):
        for value in manifest.get("raw_outputs", []):
            path = _resolve_repo_path(part_root, value)
            if path.name == "dataset_index.csv":
                candidates.append(path)
        raw_root = manifest.get("raw_output_root")
        if raw_root:
            root = _resolve_repo_path(part_root, raw_root)
            candidates.append(root.parent / "dataset_index.csv")
    candidates.append(upstream_path)
    for path in candidates:
        if path.exists() and path.name == "dataset_index.csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
    raise FileNotFoundError(f"Could not locate dataset_index.csv from {upstream_manifest}")


def _filter_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, str]]:
    target_series = set(str(x) for x in config.get("target_series", ["beta"]))
    beta_values = [round(float(x), 8) for x in config.get("target_beta_values", [])]
    start = int(config.get("start_dataset_index", 0))
    max_datasets = int(config.get("max_datasets", 0))
    out: list[dict[str, str]] = []
    by_cell_count: dict[str, int] = {}
    for row in rows:
        if str(row.get("series", "")) not in target_series:
            continue
        beta = round(float(row.get("beta_ising", 0.0)), 8)
        if beta_values and beta not in beta_values:
            continue
        dataset_id = int(row.get("dataset_id", 0))
        if dataset_id < start:
            continue
        cell_id = _reference_cell_id(row)
        count = by_cell_count.get(cell_id, 0)
        if max_datasets and count >= max_datasets:
            continue
        by_cell_count[cell_id] = count + 1
        out.append(row)
    return out


def _load_dataset(part_root: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path = _resolve_repo_path(part_root, row["dataset_raw_path"])
    data = np.load(path)
    x = np.asarray(data["X_train"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    if set(float(v) for v in np.unique(y)).issubset({0.0, 1.0}):
        y = 2.0 * y - 1.0
    return x, y


def _attempt_seed(config: dict[str, Any], row: dict[str, str], attempt_id: int) -> int:
    return int(config.get("seed", 0)) + int(row.get("dataset_id", 0)) * 10000 + int(attempt_id)


def _attempt_fields() -> list[str]:
    return [
        "attempt_id",
        "optimizer_chain",
        "final_train_loss",
        "final_cls_err",
        "n_wrong",
        "final_train_accuracy",
        "final_train_accuracy_percent",
        "min_signed_margin",
        "is_exact_solution",
        "sampler_eligible",
        "P_params",
        "adam_epochs_completed",
        "adam_early_stop_epoch",
        "adam_early_stop_reached",
        "lbfgs_iters_completed",
        "lbfgs_early_stop_epoch",
        "lbfgs_early_stop_reached",
        "early_stop_phase",
        "early_stop_epoch",
        "theta_norm",
        "theta_init_norm",
        "theta_final_path",
        "theta_init_path",
        "summary_path",
    ]


def _write_attempt_rows(part_root: Path, width_dir: Path, outputs: list[dict[str, Any]]) -> None:
    with (width_dir / "attempt_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_attempt_fields())
        writer.writeheader()
        for output in outputs:
            attempt_id = int(output["attempt_id"])
            summary = output["summary"]
            attempt_dir = width_dir / f"attempt_{attempt_id:03d}"
            ensure_dir(attempt_dir)
            theta_final = attempt_dir / "theta_final.npy"
            theta_init = attempt_dir / "theta_init.npy"
            summary_path = attempt_dir / "train_summary.json"
            np.save(theta_final, np.asarray(output["theta"], dtype=np.float64))
            np.save(theta_init, np.asarray(output["theta_init"], dtype=np.float64))
            save_json(summary_path, summary)
            writer.writerow(
                {
                    "attempt_id": attempt_id,
                    "optimizer_chain": summary.get("optimizer_chain", ""),
                    "final_train_loss": summary.get("final_train_loss", ""),
                    "final_cls_err": summary.get("final_cls_err", ""),
                    "n_wrong": summary.get("n_wrong", ""),
                    "final_train_accuracy": summary.get("final_train_accuracy", ""),
                    "final_train_accuracy_percent": summary.get("final_train_accuracy_percent", ""),
                    "min_signed_margin": summary.get("min_signed_margin", ""),
                    "is_exact_solution": summary.get("is_exact_solution", ""),
                    "sampler_eligible": summary.get("sampler_eligible", ""),
                    "P_params": summary.get("P_params", ""),
                    "adam_epochs_completed": summary.get("adam_epochs_completed", ""),
                    "adam_early_stop_epoch": summary.get("adam_early_stop_epoch", ""),
                    "adam_early_stop_reached": summary.get("adam_early_stop_reached", ""),
                    "lbfgs_iters_completed": summary.get("lbfgs_iters_completed", ""),
                    "lbfgs_early_stop_epoch": summary.get("lbfgs_early_stop_epoch", ""),
                    "lbfgs_early_stop_reached": summary.get("lbfgs_early_stop_reached", ""),
                    "early_stop_phase": summary.get("early_stop_phase", ""),
                    "early_stop_epoch": summary.get("early_stop_epoch", ""),
                    "theta_norm": summary.get("theta_norm", ""),
                    "theta_init_norm": summary.get("theta_init_norm", ""),
                    "theta_final_path": _repo_relative(part_root, theta_final),
                    "theta_init_path": _repo_relative(part_root, theta_init),
                    "summary_path": _repo_relative(part_root, summary_path),
                }
            )


def _load_attempt_records(part_root: Path, width_dir: Path) -> list[dict[str, Any]]:
    attempt_csv = width_dir / "attempt_results.csv"
    if not attempt_csv.exists():
        return []
    rows = list(csv.DictReader(attempt_csv.read_text(encoding="utf-8").splitlines()))
    records: list[dict[str, Any]] = []
    for row in rows:
        theta_path = _resolve_recorded_path(part_root, row["theta_final_path"])
        theta_init_path = _resolve_recorded_path(part_root, row["theta_init_path"])
        summary_path = _resolve_recorded_path(part_root, row["summary_path"])
        if not theta_path.exists() or not theta_init_path.exists() or not summary_path.exists():
            continue
        summary = load_json(summary_path, {}) or {}
        if "final_train_accuracy" not in summary:
            cls_err = float(row["final_cls_err"])
            summary["final_cls_err"] = cls_err
            summary["final_train_accuracy"] = float(row["final_train_accuracy"])
            summary["final_train_loss"] = float(row["final_train_loss"])
            summary["is_exact_solution"] = bool(str(row["is_exact_solution"]).lower() == "true")
        records.append(
            {
                "attempt_id": int(row["attempt_id"]),
                "theta": np.load(theta_path).astype(np.float64),
                "theta_init": np.load(theta_init_path).astype(np.float64),
                "theta_path": _repo_relative(part_root, theta_path),
                "theta_init_path": _repo_relative(part_root, theta_init_path),
                "summary_path": _repo_relative(part_root, summary_path),
                "summary": summary,
                "is_rescue": False,
            }
        )
    return records


def _postprocess_selection(part_root: Path, run_root: Path, config: dict[str, Any]) -> None:
    target_count = int(config.get("selected_refs_per_dataset", 10))
    require_exact = bool(config.get("require_exact_selected_refs", False))
    fail_on_insufficient = bool(config.get("fail_on_insufficient_selected_refs", False))
    min_train_accuracy = float(config.get("fallback_reference_min_train_accuracy", 0.95))
    topk = int(config.get("selection_topk", 8))
    dedup_scale = float(config.get("selection_dedup_scale", 0.25))
    coverage_rows: list[dict[str, Any]] = []
    insufficient_rows: list[dict[str, Any]] = []
    for width_dir in sorted(path for path in run_root.glob("*/*/width_*") if path.parent.parent.name.startswith(("cell_beta_", "beta_"))):
        attempt_records = _load_attempt_records(part_root, width_dir)
        if not attempt_records:
            continue
        cell_id = _reference_cell_id(width_dir.parent.parent.name)
        dataset_tag = width_dir.parent.name
        width = int(width_dir.name.replace("width_", ""))
        selection = summarize_and_select_reference_candidates(
            attempt_records,
            cell_id=cell_id,
            dataset_tag=dataset_tag,
            width=width,
            min_train_accuracy=min_train_accuracy,
            target_valid_count=target_count,
            max_selected_count=target_count,
            topk=topk,
            dedup_scale=dedup_scale,
            require_exact=require_exact,
            rescue_enabled=False,
            rescue_policy_name="none",
        )
        retained_rows: list[dict[str, Any]] = []
        payload_root = width_dir / "selected_ref_payloads"
        for selected_row in selection["selected_rows"]:
            retained_row = dict(selected_row)
            ref_id = int(retained_row["ref_id"])
            ref_dir = payload_root / f"ref_{ref_id:03d}"
            ref_dir.mkdir(parents=True, exist_ok=True)
            theta_path = ref_dir / "theta.npy"
            theta_init_path = ref_dir / "theta_init.npy"
            summary_path = ref_dir / "train_summary.json"
            source_theta = _resolve_recorded_path(part_root, retained_row["theta_path"])
            source_theta_init = _resolve_recorded_path(part_root, retained_row["theta_init_path"])
            source_summary = _resolve_recorded_path(part_root, retained_row["summary_path"])
            if not theta_path.exists():
                np.save(theta_path, np.load(source_theta).astype(np.float64))
            if not theta_init_path.exists():
                np.save(theta_init_path, np.load(source_theta_init).astype(np.float64))
            if not summary_path.exists():
                save_json(summary_path, load_json(source_summary, {}) or {})
            retained_row["theta_path"] = _repo_relative(part_root, theta_path)
            retained_row["theta_init_path"] = _repo_relative(part_root, theta_init_path)
            retained_row["summary_path"] = _repo_relative(part_root, summary_path)
            retained_rows.append(retained_row)
        save_json(
            width_dir / "selected_refs.json",
            {"cell_id": cell_id, "dataset_tag": dataset_tag, "width": width, "selected_refs": retained_rows},
        )
        invalid_payload = selection.get("invalid_payload")
        if invalid_payload is not None:
            insufficient_rows.append(
                {
                    "cell_id": cell_id,
                    "dataset_tag": dataset_tag,
                    "width": int(width),
                    "required_selected_refs": int(target_count),
                    "selected_ref_count": int(len(retained_rows)),
                    "exact_count": int(selection["manifest_payload"]["exact_count"]),
                    "require_exact_selected_refs": bool(require_exact),
                    "reason": str(invalid_payload.get("reason", "insufficient_distinct_sampling_eligible_references")),
                }
            )
        for diagnostic_name in EXCLUDED_SUMMARY_OUTPUT_NAMES:
            diagnostic_path = width_dir / diagnostic_name
            if diagnostic_path.exists():
                diagnostic_path.unlink()
        coverage_rows.append(
            {
                "cell_id": cell_id,
                "dataset_tag": dataset_tag,
                "width": width,
                "selected_ref_count": int(len(retained_rows)),
                "required_selected_refs": int(target_count),
                "require_exact_selected_refs": bool(require_exact),
                "exact_count": int(selection["manifest_payload"]["exact_count"]),
                "relaxed_count": int(selection["manifest_payload"]["relaxed_count"]),
                "fake_count": int(selection["manifest_payload"]["fake_count"]),
                "sampling_eligible_count": int(selection["manifest_payload"]["sampling_eligible_count"]),
                "dedup_threshold": float(selection["manifest_payload"]["dedup_threshold"]),
            }
        )
    if coverage_rows:
        summary_tables = run_root / "summary_tables"
        summary_tables.mkdir(parents=True, exist_ok=True)
        save_csv(
            summary_tables / "selected_ref_coverage.csv",
            coverage_rows,
            [
                "cell_id",
                "dataset_tag",
                "width",
                "selected_ref_count",
                "required_selected_refs",
                "require_exact_selected_refs",
                "exact_count",
                "relaxed_count",
                "fake_count",
                "sampling_eligible_count",
                "dedup_threshold",
            ],
        )
        if insufficient_rows:
            save_csv(
                summary_tables / "insufficient_selected_refs.csv",
                insufficient_rows,
                [
                    "cell_id",
                    "dataset_tag",
                    "width",
                    "required_selected_refs",
                    "selected_ref_count",
                    "exact_count",
                    "require_exact_selected_refs",
                    "reason",
                ],
            )
    manifest_path = run_root / "manifest.json"
    manifest = load_json(manifest_path, {}) or {}
    raw_outputs = {
        _repo_relative(part_root, _resolve_recorded_path(part_root, str(path)))
        for path in manifest.get("raw_outputs", [])
    }
    for path in run_root.rglob("*"):
        if path.is_file() and path.name not in {"manifest.json", "run_config.json", *EXCLUDED_SUMMARY_OUTPUT_NAMES}:
            raw_outputs.add(_repo_relative(part_root, path))
    manifest["raw_outputs"] = sorted(
        path for path in raw_outputs if _resolve_recorded_path(part_root, str(path)).name not in EXCLUDED_SUMMARY_OUTPUT_NAMES
    )
    if insufficient_rows and fail_on_insufficient:
        manifest["status"] = "failed"
        manifest["failure_reason"] = "insufficient_selected_references"
        manifest["failed_selected_ref_units"] = int(len(insufficient_rows))
        save_json(manifest_path, manifest)
        raise RuntimeError(
            "reference search produced insufficient selected references for "
            f"{len(insufficient_rows)} dataset-width units; see {run_root / 'summary_tables' / 'insufficient_selected_refs.csv'}"
        )
    save_json(manifest_path, manifest)


def _run_training(
    *,
    part_root: Path,
    config_path: Path,
    upstream_manifest: Path,
    config: dict[str, Any],
    verbose: bool = False,
) -> Path:
    from _reference_search_utils.model_types import DNNArch, TrainConfig
    from _reference_search_utils.training import train_reference_solutions_simple_batched

    started_at = now_iso()
    run_name = str(config.get("run_name", f"reference_search_{_hash_config(config)}"))
    run_root = Path(part_root) / "raw_outputs" / run_name
    ensure_dir(run_root)
    log_capture = start_verbose_print_capture(run_root, enabled=verbose)
    completed = 0
    try:
        rows = _filter_rows(_read_dataset_index(part_root, upstream_manifest), config)
        widths = [int(x) for x in config.get("widths", [48])]
        attempt_count = int(config.get("attempts_per_dataset", 200))
        arch_by_width = {width: DNNArch(input_dim=2, width1=width, width2=width) for width in widths}
        for row in rows:
            x, y = _load_dataset(part_root, row)
            cell_id = _reference_cell_id(row)
            dataset_tag = Path(row["dataset_raw_path"]).parent.name
            for width in widths:
                width_dir = run_root / cell_id / dataset_tag / f"width_{width:03d}"
                attempt_csv = width_dir / "attempt_results.csv"
                if attempt_csv.exists() and not bool(config.get("force", False)):
                    completed += 1
                    continue
                ensure_dir(width_dir)
                cfgs = [
                    TrainConfig(
                        lr=float(config.get("train_lr", 0.03)),
                        weight_decay=float(config.get("train_weight_decay", 1.0e-4)),
                        momentum=float(config.get("train_momentum", 0.9)),
                        epochs=int(config.get("adam_epochs_ref", 4000)),
                        seed=_attempt_seed(config, row, attempt_id),
                        optimizer_name="adam",
                        lbfgs_max_iter=int(config.get("lbfgs_max_iter", 4000)),
                        init_scale_multiplier=float(config.get("init_scale_multiplier", 1.0)),
                        activation=str(config.get("activation", "tanh")),
                        loss=str(config.get("ref_loss_name", "logistic")),
                        margin=float(config.get("margin", 1.0)),
                    )
                    for attempt_id in range(attempt_count)
                ]
                outputs = train_reference_solutions_simple_batched(
                    x,
                    y,
                    arch_by_width[width],
                    cfgs,
                    device=str(config.get("sampling_device", "cuda")),
                    verbose=verbose,
                    progress_label=f"{cell_id}/{dataset_tag}/width_{width:03d}",
                )
                _write_attempt_rows(part_root, width_dir, outputs)
                completed += 1
        save_json(run_root / "run_config.json", config)
        save_json(
            run_root / "manifest.json",
            {
                "pipeline_id": str(config.get("pipeline_id", "simple_reference_search")),
                "methodology_id": str(config.get("methodology_id", "batched_adam_lbfgsb_simple_v1")),
                "status": "success",
                "started_at": started_at,
                "finished_at": now_iso(),
                "config_path": _repo_relative(part_root, config_path),
                "upstream_manifest": _repo_relative(part_root, upstream_manifest),
                "unit_count": completed,
                "raw_outputs": [_repo_relative(part_root, run_root / "run_config.json")],
            },
        )
    finally:
        if log_capture is not None:
            log_capture.close()
    return run_root / "manifest.json"


def run_pipeline(
    *,
    part_root: Path = STAGE_ROOT,
    config_path: Path | None = None,
    upstream_manifest: Path | None = None,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    part_root = Path(part_root).resolve()
    config_path = Path(config_path) if config_path is not None else part_root / "config" / "default.json"
    config = merged_config(config_path, upstream_manifest, force)
    upstream = Path(config["upstream_manifest"]) if config.get("upstream_manifest") else discover_upstream_manifest(part_root, config)
    if not upstream.is_absolute():
        upstream = _resolve_repo_path(part_root, upstream)
    config["upstream_manifest"] = _repo_relative(part_root, upstream)
    manifest_path = _run_training(
        part_root=part_root,
        config_path=config_path,
        upstream_manifest=upstream,
        config=config,
        verbose=verbose,
    )
    _postprocess_selection(part_root, Path(manifest_path).parent, config)
    return Path(manifest_path)


def check_layout(part_root: Path = STAGE_ROOT) -> dict[str, Any]:
    part_root = Path(part_root).resolve()
    top_level_files = sorted((part_root / "src").glob("*.py"))
    active_source_files = sorted(
        path.relative_to(part_root).as_posix()
        for path in top_level_files
    )
    utility_files = sorted(path.relative_to(part_root).as_posix() for path in (part_root / "src" / "utils").glob("*.py"))
    raw_root = part_root / "raw_outputs"
    cell_dirs = sorted(raw_root.glob("beta_*")) if raw_root.exists() else []
    legacy_cell_dirs = sorted(raw_root.glob("cell_beta_*")) if raw_root.exists() else []
    flat_dataset_dirs = sorted(raw_root.glob("dataset_*")) if raw_root.exists() else []
    cell_dataset_counts = {path.name: len(list(path.glob("dataset_*"))) for path in [*cell_dirs, *legacy_cell_dirs]}
    return {
        "stage_root": str(part_root),
        "entrypoint": "src/reference_search.py",
        "active_source_files": active_source_files,
        "utility_files": utility_files,
        "canonical_raw_layout": "raw_outputs/beta_*/dataset_*/ref_*",
        "raw_outputs_exists": bool((part_root / "raw_outputs").exists()),
        "raw_cell_count": len(cell_dirs),
        "raw_dataset_count": sum(cell_dataset_counts.values()),
        "raw_ref_count": sum(1 for _ in raw_root.glob("beta_*/dataset_*/ref_*")) if raw_root.exists() else 0,
        "legacy_raw_cell_count": len(legacy_cell_dirs),
        "legacy_raw_ref_count": sum(1 for _ in raw_root.glob("cell_beta_*/dataset_*/ref_*")) if raw_root.exists() else 0,
        "flat_dataset_dir_count": len(flat_dataset_dirs),
        "cell_dataset_counts": cell_dataset_counts,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 03_reference_search stage.")
    parser.add_argument(
        "--mode",
        choices=["run", "check"],
        default="run",
        help="run trains references from an upstream dataset_index; check reports the source layout.",
    )
    parser.add_argument("--part-root", type=Path, default=STAGE_ROOT, help="Path to 03_reference_search.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path. Defaults to <part-root>/config/default.json.")
    parser.add_argument("--upstream-manifest", type=Path, default=None, help="Dataset manifest or dataset_index.csv for training mode.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing attempt results in training mode.")
    parser.add_argument("--verbose", action="store_true", help="Capture verbose training logs under the run root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.mode == "check":
        print(json.dumps(check_layout(args.part_root), indent=2, sort_keys=True))
        return 0
    manifest_path = run_pipeline(
        part_root=args.part_root,
        config_path=args.config,
        upstream_manifest=args.upstream_manifest,
        force=args.force,
        verbose=args.verbose,
    )
    print(_repo_relative(Path(args.part_root).resolve(), manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "check_layout",
    "discover_upstream_manifest",
    "main",
    "merged_config",
    "run_pipeline",
]
