from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ROOT = SCRIPT_DIR.parent
MANUAL_ROOT = STAGE_ROOT.parent
DNN_ROOT = MANUAL_ROOT.parent
REPO_ROOT = STAGE_ROOT.parents[2]
PROJECT_ROOT = REPO_ROOT.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.defaults import DEFAULT_CONFIG
from utils.io_utils import ensure_dir, load_json, save_csv, save_json
from utils.mnist10_model import ARCH, P, ce_and_error_np, margin_stats_np
from utils.training import select_reference, train_attempt_batch


RAW_ROOT = STAGE_ROOT / "raw_outputs"
RULE_MAPPING = MANUAL_ROOT / "config" / "rule_mapping.csv"

POOL_FIELDS = [
    "dataset_id",
    "split_id",
    "rule",
    "ref_id",
    "theta_path",
    "dataset_path",
    "attempt_seed",
    "optimizer_chain",
    "P",
    "train_error",
    "test_error",
    "CE_mean_train",
    "CE_sum_train",
    "CE_mean_test",
    "theta_norm",
    "min_margin",
    "q05_margin",
    "median_margin",
    "mean_margin",
    "extra_reference_search",
    "pool_rank",
    "resample_seed_offset",
    "rule_id",
    "ref_path_id",
]

def _repo_relative(path: Path) -> str:
    path = path.resolve()
    for root in (PROJECT_ROOT.resolve(), REPO_ROOT.resolve(), DNN_ROOT.resolve()):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(path)


def _resolve_path(part_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        (part_root / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
        (REPO_ROOT / path).resolve(),
        (DNN_ROOT / path).resolve(),
        (MANUAL_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _coalesce(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return int(default)
    return int(float(value))


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return float(default)
    return float(value)


def merged_config(config_path: Path | None = None, force: bool = False) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path is not None and config_path.exists():
        file_config = load_json(config_path, {}) or {}
        config.update(file_config)
    config["force"] = bool(force or config.get("force", False))
    return config


def _manual_root(part_root: Path = STAGE_ROOT) -> Path:
    return Path(part_root).resolve().parent


def _stage_raw_root(part_root: Path = STAGE_ROOT) -> Path:
    return Path(part_root).resolve() / "raw_outputs"


def load_rule_mapping(rule_mapping: Path = RULE_MAPPING) -> list[dict[str, str]]:
    rows = _read_csv(rule_mapping)
    if not rows:
        raise FileNotFoundError(f"empty rule mapping: {rule_mapping}")
    return rows


def _rule_lookup(rule_mapping: Path = RULE_MAPPING) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_id: dict[str, dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}
    for row in load_rule_mapping(rule_mapping):
        by_id[str(row["rule_id"])] = row
        by_name[str(row["rule_name"])] = row
    return by_id, by_name


def discover_dataset_rows(part_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    manual_root = _manual_root(part_root)
    by_id, _ = _rule_lookup(manual_root / "config" / "rule_mapping.csv")
    dataset_rows = [
        {
            "rule_id": rule_id,
            "rule_name": rule["rule_name"],
            "label": rule["label"],
            "dataset_path": _repo_relative(manual_root / "01_dataset" / "raw_outputs" / rule_id / "dataset.npz"),
        }
        for rule_id, rule in by_id.items()
    ]
    requested = set(str(rule) for rule in config.get("rules", []))
    out: list[dict[str, Any]] = []
    for row in dataset_rows:
        if requested and str(row.get("rule_name")) not in requested and str(row.get("rule_id")) not in requested:
            continue
        out.append(
            {
                "rule_id": str(row["rule_id"]),
                "rule_name": str(row["rule_name"]),
                "label": str(row.get("label", "")),
                "rule": str(row["rule_name"]),
                "dataset_path": str(row["dataset_path"]),
            }
        )
    return out


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def _attempt_seed_start(config: dict[str, Any], rule_id: str, rule_name: str) -> int:
    starts = dict(config.get("attempt_seed_starts", {}))
    if rule_id in starts:
        return int(starts[rule_id])
    if rule_name in starts:
        return int(starts[rule_name])
    return int(config.get("base_seed", 2700000))


def _resample_seed_offset(config: dict[str, Any], rule_id: str) -> int:
    return int(dict(config.get("resample_seed_offsets", {})).get(rule_id, 2026061800))


def _write_report(run_root: Path, status: dict[str, Any], reference_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Manual Rules Reference Search",
        "",
        f"- Status: `{status['status']}`",
        f"- References: `{status['reference_rows']}` / `{status['expected_reference_rows']}`",
        f"- Attempts: `{status['attempt_rows']}`",
        f"- Architecture: `input_dim={ARCH.input_dim}, hidden_width={ARCH.hidden_width}, P={P}`",
        "",
        "This stage trains exact references for the 10x10 MNIST manual-rule datasets.",
        "It is the standalone reference-search source for `manual_rules/03_reference_search`.",
        "",
        "Primary files:",
        "",
        "- `04_exact_reference_search/reference_index.csv`",
        "- `04_exact_reference_search/selected_reference_pool/`",
        "- `04_exact_reference_search/attempt_logs/attempts.csv`",
    ]
    for row in reference_rows[:10]:
        lines.append(
            f"- `{row['rule']}` ref `{int(row['ref_id']):03d}` "
            f"seed `{int(row['attempt_seed'])}` train_error `{float(row['train_error']):.6g}`"
        )
    if len(reference_rows) > 10:
        lines.append(f"- ... {len(reference_rows) - 10} more")
    (run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    *,
    part_root: Path = STAGE_ROOT,
    config_path: Path | None = None,
    force: bool = False,
    materialize_canonical: bool = True,
) -> Path:
    part_root = Path(part_root).resolve()
    config_path = config_path or part_root / "config" / "default.json"
    config = merged_config(config_path, force=force)
    run_root = part_root / "raw_outputs" / str(config["run_name"])
    exact_root = ensure_dir(run_root / "04_exact_reference_search")
    reference_index = exact_root / "reference_index.csv"
    if reference_index.exists() and not bool(config.get("force", False)):
        if materialize_canonical:
            materialize_canonical_layout(part_root=part_root, config=config, reference_index_path=reference_index)
        return reference_index

    dataset_rows = discover_dataset_rows(part_root, config)
    reference_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    started = time.time()
    target_refs = int(config.get("selected_refs_per_rule", 30))
    max_attempts = int(config.get("max_attempts_per_rule", 240))
    batch_size = int(config.get("batch_size", 10))
    expected = len(dataset_rows) * target_refs

    save_json(run_root / "run_config_resolved.json", config)
    for dataset_id, row in enumerate(dataset_rows):
        rule_id = str(row["rule_id"])
        rule_name = str(row["rule_name"])
        ds = load_dataset(_resolve_path(part_root, row["dataset_path"]))
        selected: list[dict[str, Any]] = []
        attempts_used = 0
        seed_start = _attempt_seed_start(config, rule_id, rule_name)
        print(f"[manual-rule-reference] {rule_id}/{rule_name} selected=0/{target_refs}", flush=True)
        while attempts_used < max_attempts and len(selected) < target_refs:
            n_batch = min(batch_size, max_attempts - attempts_used)
            seeds = [seed_start + attempts_used + offset for offset in range(n_batch)]
            batch = train_attempt_batch(
                ds["X_train"],
                ds["y_train"],
                seeds,
                max_epochs=int(config.get("max_epochs", 4200)),
                lr=float(config.get("lr", 0.022)),
                device=str(config.get("device", "auto")),
            )
            attempts_used += n_batch
            for result in batch:
                theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
                ce_train, err_train = ce_and_error_np(theta, ds["X_train"], ds["y_train"])
                ce_test, err_test = ce_and_error_np(theta, ds["X_test"], ds["y_test"])
                selected_flag = False
                if err_train == 0.0 and theta.size == P:
                    selected_flag = select_reference(
                        selected,
                        {
                            "theta": theta,
                            "attempt_seed": int(result["seed"]),
                            "phase": str(result["phase"]),
                            "train_error": 0.0,
                        },
                    )
                attempt_rows.append(
                    {
                        "dataset_id": int(dataset_id),
                        "split_id": 0,
                        "rule": rule_name,
                        "rule_id": rule_id,
                        "attempt_seed": int(result["seed"]),
                        "phase": str(result["phase"]),
                        "epoch": int(result["epoch"]),
                        "train_error": float(err_train),
                        "test_error": float(err_test),
                        "CE_mean_train": float(ce_train),
                        "CE_mean_test": float(ce_test),
                        "theta_norm": float(np.linalg.norm(theta)),
                        "selected": bool(selected_flag),
                    }
                )
            save_csv(exact_root / "attempt_logs" / "attempts.csv", attempt_rows)
            print(
                f"[manual-rule-reference] {rule_id} attempts={attempts_used} selected={len(selected)}/{target_refs}",
                flush=True,
            )
        if len(selected) < target_refs:
            raise RuntimeError(
                f"insufficient exact references for {rule_id}: selected={len(selected)} target={target_refs}"
            )
        for source_ref_id, result in enumerate(selected[:target_refs]):
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ref_dir = ensure_dir(
                exact_root / "selected_reference_pool" / "split_000" / rule_name / f"ref_{source_ref_id:03d}"
            )
            theta_path = ref_dir / "theta.npy"
            np.save(theta_path, theta)
            ce_train, err_train = ce_and_error_np(theta, ds["X_train"], ds["y_train"])
            ce_test, err_test = ce_and_error_np(theta, ds["X_test"], ds["y_test"])
            summary = {
                "dataset_id": int(dataset_id),
                "split_id": 0,
                "rule": rule_name,
                "ref_id": int(source_ref_id),
                "theta_path": _repo_relative(theta_path),
                "dataset_path": row["dataset_path"],
                "attempt_seed": int(result["attempt_seed"]),
                "optimizer_chain": str(result["phase"]),
                "P": int(theta.size),
                "train_error": float(err_train),
                "test_error": float(err_test),
                "CE_mean_train": float(ce_train),
                "CE_sum_train": float(ce_train * ds["X_train"].shape[0]),
                "CE_mean_test": float(ce_test),
                "theta_norm": float(np.linalg.norm(theta)),
                **margin_stats_np(theta, ds["X_train"], ds["y_train"]),
                "extra_reference_search": "",
                "pool_rank": int(source_ref_id + 1),
                "resample_seed_offset": _resample_seed_offset(config, rule_id),
                "rule_id": rule_id,
                "ref_path_id": f"ref_{source_ref_id + int(config.get('canonical_ref_offset', 1)):03d}",
            }
            save_json(ref_dir / "ref_summary.json", summary)
            reference_rows.append(summary)

    save_csv(reference_index, reference_rows, POOL_FIELDS)
    status = {
        "status": "complete" if len(reference_rows) >= expected else "partial",
        "rules": [str(row["rule_id"]) for row in dataset_rows],
        "reference_rows": int(len(reference_rows)),
        "expected_reference_rows": int(expected),
        "attempt_rows": int(len(attempt_rows)),
        "all_selected_exact": bool(reference_rows and all(float(row["train_error"]) == 0.0 for row in reference_rows)),
        "theta_length_all_P": bool(reference_rows and all(int(row["P"]) == P for row in reference_rows)),
        "elapsed_s": float(time.time() - started),
    }
    save_json(run_root / "REFERENCE_SEARCH_STATUS.json", status)
    save_json(exact_root / "REFERENCE_SEARCH_STATUS.json", status)
    _write_report(run_root, status, reference_rows)
    if materialize_canonical:
        materialize_canonical_layout(part_root=part_root, config=config, reference_index_path=reference_index)
    return reference_index


def _find_reference_pool_index(part_root: Path, config: dict[str, Any]) -> Path:
    run_name = str(config.get("run_name", ""))
    candidates = [
        _stage_raw_root(part_root) / run_name / "04_exact_reference_search" / "reference_index.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate reference_index.csv under 03_reference_search/raw_outputs")


def _rule_info(row: dict[str, Any], part_root: Path = STAGE_ROOT) -> dict[str, str]:
    by_id, by_name = _rule_lookup(_manual_root(part_root) / "config" / "rule_mapping.csv")
    rule_id = str(_coalesce(row, "rule_id", default=""))
    rule_name = str(_coalesce(row, "rule", "rule_name", default=""))
    if rule_id and rule_id in by_id:
        info = by_id[rule_id]
    elif rule_name and rule_name in by_name:
        info = by_name[rule_name]
    else:
        raise ValueError(f"could not map rule row: {row}")
    return {
        "rule_id": str(info["rule_id"]),
        "rule_name": str(info["rule_name"]),
        "label": str(info.get("label", "")),
    }


def provenance_alignment(part_root: Path = STAGE_ROOT, config: dict[str, Any] | None = None) -> dict[str, Any]:
    part_root = Path(part_root).resolve()
    config = config or merged_config(part_root / "config" / "default.json")
    raw_root = _stage_raw_root(part_root)
    metadata_paths = sorted(raw_root.glob("rule_*/ref_*/reference_metadata.json"))
    if not metadata_paths:
        return {"available": False, "error": f"no reference metadata under {raw_root / 'rule_*' / 'ref_*'}"}
    offset = int(config.get("canonical_ref_offset", 1))
    seed_starts = {str(key): int(value) for key, value in dict(config.get("attempt_seed_starts", {})).items()}
    theta_exists = 0
    metadata_matches_layout = 0
    dataset_exists = 0
    seed_matches = 0
    p_matches = 0
    exact_matches = 0
    optimizers: set[str] = set()
    rule_counts: dict[str, int] = {}
    for metadata_path in metadata_paths:
        metadata = load_json(metadata_path)
        rule_id = str(metadata.get("rule_id", metadata_path.parent.parent.name))
        rule_name = str(metadata.get("rule_name", metadata.get("rule", rule_id)))
        rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        source_ref_id = _as_int(
            metadata.get("source_ref_id"),
            _as_int(metadata_path.parent.name.removeprefix("ref_")) - offset,
        )
        theta_path = metadata_path.parent / "theta.npy"
        if theta_path.exists():
            theta_exists += 1
        if metadata_path.parent.parent.name == rule_id and metadata_path.parent.name.startswith("ref_"):
            metadata_matches_layout += 1
        dataset_path = _resolve_path(
            part_root,
            metadata.get("dataset_payload_path", metadata.get("source_dataset_path", "")),
        )
        if dataset_path.exists():
            dataset_exists += 1
        expected_start = seed_starts.get(rule_id, seed_starts.get(rule_name))
        if expected_start is not None and _as_int(metadata.get("attempt_seed")) == int(expected_start) + source_ref_id:
            seed_matches += 1
        if _as_int(metadata.get("P")) == P:
            p_matches += 1
        if _as_float(metadata.get("train_error")) == 0.0:
            exact_matches += 1
        optimizers.add(str(metadata.get("optimizer_chain", "")))
    expected_count = int(len(config.get("rules", []))) * int(config.get("selected_refs_per_rule", 30))
    expected_rule_count = int(config.get("selected_refs_per_rule", 30))
    return {
        "available": True,
        "reference_rows": int(len(metadata_paths)),
        "expected_reference_rows": int(expected_count),
        "rule_counts": dict(sorted(rule_counts.items())),
        "theta_paths_exist": int(theta_exists),
        "metadata_paths_match_raw_layout": int(metadata_matches_layout),
        "dataset_paths_exist": int(dataset_exists),
        "attempt_seeds_match_config": int(seed_matches),
        "P_matches_mnist10_architecture": int(p_matches),
        "train_error_zero_rows": int(exact_matches),
        "optimizer_chains": sorted(optimizers),
        "fully_aligned": bool(
            metadata_paths
            and len(metadata_paths) == expected_count
            and all(count == expected_rule_count for count in rule_counts.values())
            and theta_exists == len(metadata_paths)
            and metadata_matches_layout == len(metadata_paths)
            and dataset_exists == len(metadata_paths)
            and seed_matches == len(metadata_paths)
            and p_matches == len(metadata_paths)
            and exact_matches == len(metadata_paths)
            and optimizers == {"adam"}
        ),
    }


def _dataset_path_for_rule(dataset_rows: list[dict[str, Any]], rule_id: str) -> str:
    for row in dataset_rows:
        if str(row["rule_id"]) == rule_id:
            return str(row["dataset_path"])
    return _repo_relative(MANUAL_ROOT / "01_dataset" / "raw_outputs" / rule_id / "dataset.npz")


def materialize_canonical_layout(
    *,
    part_root: Path = STAGE_ROOT,
    config: dict[str, Any] | None = None,
    reference_index_path: Path | None = None,
) -> Path:
    part_root = Path(part_root).resolve()
    config = config or merged_config(part_root / "config" / "default.json")
    reference_index_path = reference_index_path or _find_reference_pool_index(part_root, config)
    rows = _read_csv(reference_index_path)
    dataset_rows = discover_dataset_rows(part_root, config)
    offset = int(config.get("canonical_ref_offset", 1))
    for row in rows:
        info = _rule_info(row, part_root)
        rule_id = info["rule_id"]
        rule_name = info["rule_name"]
        label = info["label"]
        source_ref_id = _as_int(row.get("ref_id"))
        ref_path_id = str(_coalesce(row, "ref_path_id", default=f"ref_{source_ref_id + offset:03d}"))
        ref_dir = ensure_dir(_stage_raw_root(part_root) / rule_id / ref_path_id)
        local_theta = ref_dir / "theta.npy"
        source_theta = _resolve_path(part_root, row["theta_path"])
        if source_theta.exists() and source_theta.resolve() != local_theta.resolve():
            if bool(config.get("force", False)) or not local_theta.exists():
                shutil.copy2(source_theta, local_theta)
        dataset_path_value = _dataset_path_for_rule(dataset_rows, rule_id)
        dataset_path = _resolve_path(part_root, dataset_path_value)
        theta_shape: list[int] = []
        theta_norm = _as_float(row.get("theta_norm"))
        if local_theta.exists():
            theta = np.load(local_theta)
            theta_shape = list(theta.shape)
            theta_norm = float(np.linalg.norm(theta))
        metadata = {
            "rule_id": rule_id,
            "rule_name": rule_name,
            "label": label,
            "rule": rule_name,
            "ref_id": ref_path_id,
            "source_ref_id": source_ref_id,
            "attempt_seed": _as_int(row.get("attempt_seed")),
            "optimizer_chain": str(_coalesce(row, "optimizer_chain", default="")),
            "P": _as_int(row.get("P"), P),
            "train_error": _as_float(row.get("train_error")),
            "test_error": _as_float(row.get("test_error")),
            "CE_mean_train": _as_float(row.get("CE_mean_train")),
            "CE_sum_train": _as_float(row.get("CE_sum_train")),
            "CE_mean_test": _as_float(row.get("CE_mean_test")),
            "theta_norm": theta_norm,
            "theta_shape": theta_shape,
            "min_margin": _as_float(row.get("min_margin")),
            "q05_margin": _as_float(row.get("q05_margin")),
            "median_margin": _as_float(row.get("median_margin")),
            "mean_margin": _as_float(row.get("mean_margin")),
            "pool_rank": _as_int(row.get("pool_rank"), source_ref_id + 1),
            "resample_seed_offset": _as_int(row.get("resample_seed_offset"), _resample_seed_offset(config, rule_id)),
            "theta_payload_path": _repo_relative(local_theta),
            "theta_payload_exists": local_theta.exists(),
            "dataset_payload_path": _repo_relative(dataset_path),
            "dataset_payload_exists": dataset_path.exists(),
            "source_theta_path": str(row["theta_path"]),
            "source_theta_exists": source_theta.exists(),
            "source_dataset_path": str(_coalesce(row, "dataset_path", default=dataset_path_value)),
            "source_dataset_exists": _resolve_path(part_root, _coalesce(row, "dataset_path", default=dataset_path_value)).exists(),
        }
        metadata_path = ref_dir / "reference_metadata.json"
        save_json(metadata_path, metadata)
    return _stage_raw_root(part_root)


def summarize_existing_reference_pool(
    *,
    part_root: Path = STAGE_ROOT,
    config_path: Path | None = None,
    force: bool = False,
) -> Path:
    part_root = Path(part_root).resolve()
    config_path = config_path or part_root / "config" / "default.json"
    config = merged_config(config_path, force=force)
    try:
        reference_index = _find_reference_pool_index(part_root, config)
    except FileNotFoundError:
        raw_root = _stage_raw_root(part_root)
        if sorted(raw_root.glob("rule_*/ref_*/reference_metadata.json")):
            return raw_root
        raise
    return materialize_canonical_layout(part_root=part_root, config=config, reference_index_path=reference_index)


def check_layout(part_root: Path = STAGE_ROOT) -> dict[str, Any]:
    part_root = Path(part_root).resolve()
    config = merged_config(part_root / "config" / "default.json")
    source_files = sorted(path.relative_to(part_root).as_posix() for path in (part_root / "src").glob("*.py"))
    utility_files = sorted(path.relative_to(part_root).as_posix() for path in (part_root / "src" / "utils").glob("*.py"))
    theta_files = sorted((part_root / "raw_outputs").glob("rule_*/ref_*/theta.npy"))
    metadata_files = sorted((part_root / "raw_outputs").glob("rule_*/ref_*/reference_metadata.json"))
    rule_dirs = sorted((part_root / "raw_outputs").glob("rule_*"))
    return {
        "stage_root": str(part_root),
        "entrypoint": "src/reference_search.py",
        "run_modes": ["run", "summarize-existing", "check"],
        "active_source_files": source_files,
        "utility_files": utility_files,
        "architecture": {
            "input_dim": int(ARCH.input_dim),
            "hidden_width": int(ARCH.hidden_width),
            "P": int(P),
        },
        "raw_theta_count": int(len(theta_files)),
        "raw_metadata_count": int(len(metadata_files)),
        "raw_rule_dirs": int(len(rule_dirs)),
        "config_exists": (part_root / "config" / "default.json").exists(),
        "provenance_alignment": provenance_alignment(part_root, config),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or summarize MNIST manual-rules reference search.")
    parser.add_argument("--mode", choices=["run", "summarize-existing", "check"], default="run")
    parser.add_argument("--part-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-materialize-canonical", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.mode == "check":
        import json

        print(json.dumps(check_layout(args.part_root), indent=2, sort_keys=True))
        return 0
    if args.mode == "summarize-existing":
        path = summarize_existing_reference_pool(part_root=args.part_root, config_path=args.config, force=args.force)
    else:
        path = run_pipeline(
            part_root=args.part_root,
            config_path=args.config,
            force=args.force,
            materialize_canonical=not args.no_materialize_canonical,
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
