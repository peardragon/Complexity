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
PAIRWISE_ROOT = STAGE_ROOT.parent
DNN_ROOT = PAIRWISE_ROOT.parent
PROJECT_ROOT = DNN_ROOT.parents[1]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.io_utils import ensure_dir, load_json, save_csv, save_json
from utils.mnist10_model import ARCH, P, ce_and_error_np, margin_stats_np
from utils.training import select_reference, train_attempt_batch


RAW_ROOT = STAGE_ROOT / "raw_outputs"
PAIR_MAPPING = PAIRWISE_ROOT / "config" / "pair_mapping.csv"
COMPLEXITY_SUMMARY = (
    PAIRWISE_ROOT / "02_complexity_measure" / "summarized_outputs" / "digit_pairwise_complexity_summary.csv"
)

POOL_FIELDS = [
    "dataset_id",
    "split_id",
    "pair_id",
    "pair_label",
    "digit_a",
    "digit_b",
    "pair_rank_complexity_desc",
    "pair_order",
    "complexity_mean",
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
    "ref_path_id",
]


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    for root in (PROJECT_ROOT.resolve(), DNN_ROOT.resolve()):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(path)


def _resolve_path(value: str | Path, part_root: Path = STAGE_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        (part_root / path).resolve(),
        (PAIRWISE_ROOT / path).resolve(),
        (DNN_ROOT / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return int(default)
    return int(float(value))


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return float(default)
    return float(value)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or STAGE_ROOT / "config" / "default.json"
    config = load_json(config_path, {}) or {}
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a JSON object")
    return config


def _pair_mapping_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for order, row in enumerate(_read_csv(PAIR_MAPPING), start=1):
        pair_id = str(row["pair_id"])
        rows[pair_id] = {
            "pair_id": pair_id,
            "digit_a": int(row["digit_a"]),
            "digit_b": int(row["digit_b"]),
            "pair_label": str(row["label"]),
            "label": str(row["label"]),
            "pair_order": order,
        }
    return rows


def _complexity_rows() -> dict[str, dict[str, Any]]:
    if not COMPLEXITY_SUMMARY.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_csv(COMPLEXITY_SUMMARY):
        pair_id = str(row["pair_id"])
        rows[pair_id] = {
            "pair_rank_complexity_desc": int(row["rank_complexity_desc"]),
            "complexity_mean": float(row["complexity_mean"]),
            "dataset_path": str(row["dataset_path"]),
        }
    return rows


def discover_dataset_rows(part_root: Path = STAGE_ROOT, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    del part_root
    config = config or load_config()
    selected = [str(pair_id) for pair_id in config.get("selected_pairs", [])]
    if not selected:
        raise ValueError("config selected_pairs must list pair IDs to run")
    mapping = _pair_mapping_rows()
    complexity = _complexity_rows()
    rows: list[dict[str, Any]] = []
    for pair_order, pair_id in enumerate(selected, start=1):
        if pair_id not in mapping:
            raise KeyError(f"{PAIR_MAPPING} is missing {pair_id}")
        row = dict(mapping[pair_id])
        row.update(complexity.get(pair_id, {}))
        row.setdefault("pair_rank_complexity_desc", pair_order)
        row.setdefault("complexity_mean", float("nan"))
        row.setdefault(
            "dataset_path",
            _repo_relative(PAIRWISE_ROOT / "01_dataset" / "raw_outputs" / pair_id / "dataset.npz"),
        )
        row["pair_order"] = pair_order
        rows.append(row)
    return rows


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _attempt_seed_start(config: dict[str, Any], pair_id: str) -> int:
    starts = dict(config.get("attempt_seed_starts", {}))
    if pair_id in starts:
        return int(starts[pair_id])
    return int(config.get("base_seed", 2710000))


def _resample_seed_offset(config: dict[str, Any], pair_id: str) -> int:
    offsets = dict(config.get("resample_seed_offsets", {}))
    if pair_id in offsets:
        return int(offsets[pair_id])
    return int(config.get("resample_seed_offset", 2026061800))


def _find_reference_index(part_root: Path, *, require_exists: bool = True) -> Path:
    candidate = Path(part_root).resolve() / "raw_outputs" / "reference_index.csv"
    if candidate.exists() or not require_exists:
        return candidate
    raise FileNotFoundError(candidate)


def _metadata_to_index_row(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    metadata = load_json(path)
    pair_id = str(metadata["pair_id"])
    ref_path_id = path.parent.name
    theta_path = path.parent / "theta.npy"
    return {
        "dataset_id": _as_int(metadata.get("dataset_id")),
        "split_id": _as_int(metadata.get("split_id")),
        "pair_id": pair_id,
        "pair_label": str(metadata.get("pair_label", metadata.get("label", pair_id))),
        "digit_a": _as_int(metadata.get("digit_a")),
        "digit_b": _as_int(metadata.get("digit_b")),
        "pair_rank_complexity_desc": _as_int(metadata.get("pair_rank_complexity_desc")),
        "pair_order": _as_int(metadata.get("pair_order")),
        "complexity_mean": _as_float(metadata.get("complexity_mean")),
        "rule": str(metadata.get("rule", metadata.get("pair_label", pair_id))),
        "ref_id": _as_int(metadata.get("source_ref_id")),
        "theta_path": _repo_relative(theta_path),
        "dataset_path": str(metadata.get("dataset_payload_path", metadata.get("source_dataset_path", ""))),
        "attempt_seed": _as_int(metadata.get("attempt_seed")),
        "optimizer_chain": str(metadata.get("optimizer_chain", "")),
        "P": _as_int(metadata.get("P"), P),
        "train_error": _as_float(metadata.get("train_error")),
        "test_error": _as_float(metadata.get("test_error")),
        "CE_mean_train": _as_float(metadata.get("CE_mean_train")),
        "CE_sum_train": _as_float(metadata.get("CE_sum_train")),
        "CE_mean_test": _as_float(metadata.get("CE_mean_test")),
        "theta_norm": _as_float(metadata.get("theta_norm")),
        "min_margin": _as_float(metadata.get("min_margin")),
        "q05_margin": _as_float(metadata.get("q05_margin")),
        "median_margin": _as_float(metadata.get("median_margin")),
        "mean_margin": _as_float(metadata.get("mean_margin")),
        "extra_reference_search": "",
        "pool_rank": _as_int(metadata.get("pool_rank")),
        "resample_seed_offset": _as_int(metadata.get("resample_seed_offset"), _resample_seed_offset(config, pair_id)),
        "ref_path_id": ref_path_id,
    }


def summarize_existing_reference_pool(
    *,
    part_root: Path = STAGE_ROOT,
    config_path: Path | None = None,
    force: bool = False,
) -> Path:
    del force
    part_root = Path(part_root).resolve()
    config = load_config(config_path or part_root / "config" / "default.json")
    rows = [_metadata_to_index_row(path, config) for path in sorted((part_root / "raw_outputs").glob("pair_*/ref_*/reference_metadata.json"))]
    if not rows:
        raise FileNotFoundError(f"no reference_metadata.json files under {part_root / 'raw_outputs'}")
    reference_index = part_root / "raw_outputs" / "reference_index.csv"
    save_csv(reference_index, rows, POOL_FIELDS)
    return reference_index


def _write_report(run_root: Path, status: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Digit Pairwise Reference Search",
        "",
        f"- Status: `{status['status']}`",
        f"- References: `{status['reference_rows']}` / `{status['expected_reference_rows']}`",
        f"- Attempts: `{status['attempt_rows']}`",
        f"- Architecture: `{ARCH.input_dim}-{ARCH.hidden_width}-{ARCH.hidden_width}-1-{ARCH.activation}`, P=`{P}`",
        "",
        "Selected digit pairs:",
    ]
    for pair_id in status["selected_pairs"]:
        lines.append(f"- `{pair_id}`")
    lines.extend(["", "First reference rows:"])
    for row in rows[:10]:
        lines.append(
            f"- `{row['pair_id']}` ref `{int(row['ref_id']):03d}` "
            f"seed `{int(row['attempt_seed'])}` train_error `{float(row['train_error']):.6g}`"
        )
    (run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    *,
    part_root: Path = STAGE_ROOT,
    config_path: Path | None = None,
    force: bool = False,
    materialize_canonical: bool = True,
) -> Path:
    del materialize_canonical
    part_root = Path(part_root).resolve()
    config_path = config_path or part_root / "config" / "default.json"
    config = load_config(config_path)
    config["force"] = bool(force or config.get("force", False))
    run_root = ensure_dir(part_root / "raw_outputs")
    reference_index = _find_reference_index(part_root, require_exists=False)
    if reference_index.exists() and not bool(config.get("force", False)):
        return reference_index

    dataset_rows = discover_dataset_rows(part_root, config)
    target_refs = int(config.get("selected_refs_per_pair", 30))
    max_attempts = int(config.get("max_attempts_per_pair", 240))
    batch_size = int(config.get("batch_size", 10))
    expected = len(dataset_rows) * target_refs
    reference_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    started = time.time()
    save_json(run_root / "run_config_resolved.json", config)
    attempt_log = run_root / "attempt_logs" / "attempts.csv"

    for dataset_id, row in enumerate(dataset_rows):
        pair_id = str(row["pair_id"])
        pair_label = str(row["pair_label"])
        dataset_path = _resolve_path(row["dataset_path"], part_root)
        ds = load_dataset(dataset_path)
        selected: list[dict[str, Any]] = []
        attempts_used = 0
        seed_start = _attempt_seed_start(config, pair_id)
        print(f"[digit-pair-reference] {pair_id}/{pair_label} selected=0/{target_refs}", flush=True)
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
                        "pair_id": pair_id,
                        "pair_label": pair_label,
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
            save_csv(attempt_log, attempt_rows)
            print(
                f"[digit-pair-reference] {pair_id} attempts={attempts_used} selected={len(selected)}/{target_refs}",
                flush=True,
            )
        if len(selected) < target_refs:
            raise RuntimeError(f"insufficient exact references for {pair_id}: selected={len(selected)} target={target_refs}")

        for source_ref_id, result in enumerate(selected[:target_refs]):
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ref_path_id = f"ref_{source_ref_id + 1:03d}"
            ref_dir = ensure_dir(run_root / pair_id / ref_path_id)
            theta_path = ref_dir / "theta.npy"
            np.save(theta_path, theta)
            ce_train, err_train = ce_and_error_np(theta, ds["X_train"], ds["y_train"])
            ce_test, err_test = ce_and_error_np(theta, ds["X_test"], ds["y_test"])
            summary = {
                "dataset_id": int(dataset_id),
                "split_id": 0,
                "pair_id": pair_id,
                "pair_label": pair_label,
                "digit_a": int(row["digit_a"]),
                "digit_b": int(row["digit_b"]),
                "pair_rank_complexity_desc": int(row["pair_rank_complexity_desc"]),
                "pair_order": int(row["pair_order"]),
                "complexity_mean": float(row["complexity_mean"]),
                "rule": pair_label,
                "ref_id": int(source_ref_id),
                "theta_path": _repo_relative(theta_path),
                "dataset_path": _repo_relative(dataset_path),
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
                "resample_seed_offset": _resample_seed_offset(config, pair_id),
                "ref_path_id": ref_path_id,
            }
            save_json(ref_dir / "ref_summary.json", summary)
            metadata = {
                **summary,
                "label": pair_label,
                "source_ref_id": int(source_ref_id),
                "theta_payload_path": _repo_relative(theta_path),
                "theta_payload_exists": theta_path.exists(),
                "dataset_payload_path": _repo_relative(dataset_path),
                "dataset_payload_exists": dataset_path.exists(),
                "source_dataset_path": _repo_relative(dataset_path),
                "source_dataset_exists": dataset_path.exists(),
            }
            save_json(ref_dir / "reference_metadata.json", metadata)
            reference_rows.append(summary)

    save_csv(reference_index, reference_rows, POOL_FIELDS)
    status = {
        "status": "complete" if len(reference_rows) >= expected else "partial",
        "selected_pairs": [str(row["pair_id"]) for row in dataset_rows],
        "reference_rows": int(len(reference_rows)),
        "expected_reference_rows": int(expected),
        "attempt_rows": int(len(attempt_rows)),
        "all_selected_exact": bool(reference_rows and all(float(row["train_error"]) == 0.0 for row in reference_rows)),
        "theta_length_all_P": bool(reference_rows and all(int(row["P"]) == P for row in reference_rows)),
        "elapsed_s": float(time.time() - started),
    }
    save_json(run_root / "REFERENCE_SEARCH_STATUS.json", status)
    _write_report(run_root, status, reference_rows)
    return reference_index


def check_layout(part_root: Path = STAGE_ROOT) -> dict[str, Any]:
    part_root = Path(part_root).resolve()
    config = load_config(part_root / "config" / "default.json")
    dataset_rows = discover_dataset_rows(part_root, config)
    metadata_files = sorted((part_root / "raw_outputs").glob("pair_*/ref_*/reference_metadata.json"))
    theta_files = sorted((part_root / "raw_outputs").glob("pair_*/ref_*/theta.npy"))
    reference_index = part_root / "raw_outputs" / "reference_index.csv"
    return {
        "stage_root": str(part_root),
        "entrypoint": "src/reference_search.py",
        "run_modes": ["run", "summarize-existing", "check"],
        "selected_pairs": [str(row["pair_id"]) for row in dataset_rows],
        "available_pair_dataset_count": len(list((PAIRWISE_ROOT / "01_dataset" / "raw_outputs").glob("pair_*/dataset.npz"))),
        "selected_dataset_paths_exist": all(_resolve_path(row["dataset_path"], part_root).exists() for row in dataset_rows),
        "architecture": {
            "input_dim": int(ARCH.input_dim),
            "hidden_width": int(ARCH.hidden_width),
            "hidden_layers": int(ARCH.hidden_layers),
            "activation": ARCH.activation,
            "P": int(P),
        },
        "target_reference_rows": len(dataset_rows) * int(config.get("selected_refs_per_pair", 30)),
        "reference_index_exists": reference_index.exists(),
        "raw_theta_count": len(theta_files),
        "raw_metadata_count": len(metadata_files),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or summarize MNIST digit-pair reference search.")
    parser.add_argument("--mode", choices=["run", "summarize-existing", "check"], default="run")
    parser.add_argument("--part-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
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
        path = run_pipeline(part_root=args.part_root, config_path=args.config, force=args.force)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
