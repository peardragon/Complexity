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
LABEL_ROOT = STAGE_ROOT.parent
REPO_ROOT = STAGE_ROOT.parents[2]
PROJECT_ROOT = REPO_ROOT.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.defaults import DEFAULT_CONFIG
from utils.io_utils import ensure_dir, load_json, save_csv, save_json
from utils.mnist10_model import ARCH, P, ce_and_error_np, margin_stats_np
from utils.training import select_reference, train_attempt_batch


RAW_ROOT = STAGE_ROOT / "raw_outputs"
REFERENCE_FIELDS = [
    "dataset_id",
    "split_id",
    "rule",
    "eta",
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
]


def _eta_key(eta: float) -> str:
    return f"{float(eta):.2f}"


def eta_token(eta: float) -> str:
    return f"eta_{float(eta):.2f}".replace(".", "p")


def noise_eta_token(eta: float) -> str:
    return f"noise_eta_{float(eta):.2f}".replace(".", "p")


def eta_from_rule(rule: str) -> float:
    text = str(rule)
    if text.startswith("noise_eta_"):
        return float(text.removeprefix("noise_eta_").replace("p", "."))
    if text.startswith("eta_"):
        return float(text.removeprefix("eta_").replace("p", "."))
    return float(text)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            return str(path)


def _resolve_path(part_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        (part_root / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
        (REPO_ROOT / path).resolve(),
        (LABEL_ROOT / path).resolve(),
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


def merged_config(config_path: Path | None = None, force: bool = False) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path is not None and config_path.exists():
        file_config = load_json(config_path, {}) or {}
        config.update(file_config)
    config["force"] = bool(force or config.get("force", False))
    return config


def _stage_raw_root(part_root: Path = STAGE_ROOT) -> Path:
    return Path(part_root).resolve() / "raw_outputs"


def discover_dataset_rows(part_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_root = Path(part_root).resolve().parent
    dataset_root = label_root / "01_dataset" / "raw_outputs"
    for dataset_path in sorted(dataset_root.glob("noise_eta_*/dataset.npz")):
        eta = eta_from_rule(dataset_path.parent.name)
        rows.append(
            {
                "eta": eta,
                "rule": eta_token(eta),
                "noise_eta": dataset_path.parent.name,
                "dataset_path": _repo_relative(dataset_path),
            }
        )
    return rows


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def _attempt_seed_start(config: dict[str, Any], eta: float) -> int:
    starts = dict(config.get("attempt_seed_starts", {}))
    key = _eta_key(eta)
    if key in starts:
        return int(starts[key])
    return int(config.get("base_seed", 0)) + 100000 + int(round(float(eta) * 10000)) * 10


def _write_report(run_root: Path, status: dict[str, Any], reference_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# MNIST Eta Reference Search",
        "",
        f"- Status: `{status['status']}`",
        f"- References: `{status['reference_rows']}` / `{status['expected_reference_rows']}`",
        f"- Attempts: `{status['attempt_rows']}`",
        f"- Architecture: `input_dim={ARCH.input_dim}, hidden_width={ARCH.hidden_width}, P={P}`",
        "",
        "This stage trains eta-specific exact references for 10x10 MNIST even/odd label-noise datasets.",
        "It is the standalone reference-search source for `label_noise_sweep/03_reference_search`.",
        "",
        "Primary files:",
        "",
        "- `04_exact_reference_search/reference_index.csv`",
        "- `04_exact_reference_search/selected_reference_pool/`",
        "- `04_exact_reference_search/attempt_logs/attempts.csv`",
    ]
    if reference_rows:
        lines.extend(["", "Selected references:", ""])
        preview = reference_rows[:10]
        for row in preview:
            lines.append(
                f"- `{row['rule']}` ref `{int(row['ref_id']):03d}` "
                f"seed `{int(row['attempt_seed'])}` train_error `{float(row['train_error']):.6g}`"
            )
        if len(reference_rows) > len(preview):
            lines.append(f"- ... {len(reference_rows) - len(preview)} more")
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

    rows_by_eta = {
        _eta_key(float(row["eta"])): row
        for row in discover_dataset_rows(part_root, config)
    }
    reference_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    started = time.time()
    target_refs = int(config.get("selected_refs_per_eta", 30))
    max_attempts = int(config.get("max_attempts_per_eta", 240))
    batch_size = int(config.get("batch_size", 10))
    expected = len(config.get("etas", [])) * target_refs

    save_json(run_root / "run_config_resolved.json", config)
    for dataset_id, eta_value in enumerate(config.get("etas", [])):
        eta = float(eta_value)
        row = rows_by_eta.get(_eta_key(eta))
        if row is None:
            raise FileNotFoundError(f"dataset row not found for eta={eta}")
        ds = load_dataset(_resolve_path(part_root, row["dataset_path"]))
        selected: list[dict[str, Any]] = []
        attempts_used = 0
        seed_start = _attempt_seed_start(config, eta)
        print(f"[mnist-eta-reference] eta={eta:.2f} selected=0/{target_refs}", flush=True)
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
                    candidate = {
                        "theta": theta,
                        "attempt_seed": int(result["seed"]),
                        "phase": str(result["phase"]),
                        "train_error": 0.0,
                    }
                    selected_flag = select_reference(selected, candidate)
                attempt_rows.append(
                    {
                        "dataset_id": int(dataset_id),
                        "split_id": 0,
                        "rule": eta_token(eta),
                        "eta": float(eta),
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
                f"[mnist-eta-reference] eta={eta:.2f} attempts={attempts_used} selected={len(selected)}/{target_refs}",
                flush=True,
            )
        if len(selected) < target_refs:
            raise RuntimeError(
                f"insufficient exact references for eta={eta:.2f}: selected={len(selected)} target={target_refs}"
            )
        for ref_id, result in enumerate(selected[:target_refs]):
            theta = np.asarray(result["theta"], dtype=np.float64).reshape(-1)
            ref_dir = ensure_dir(exact_root / "selected_reference_pool" / "split_000" / eta_token(eta) / f"ref_{ref_id:03d}")
            theta_path = ref_dir / "theta.npy"
            np.save(theta_path, theta)
            ce_train, err_train = ce_and_error_np(theta, ds["X_train"], ds["y_train"])
            ce_test, err_test = ce_and_error_np(theta, ds["X_test"], ds["y_test"])
            summary = {
                "dataset_id": int(dataset_id),
                "split_id": 0,
                "rule": eta_token(eta),
                "eta": float(eta),
                "ref_id": int(ref_id),
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
            }
            save_json(ref_dir / "ref_summary.json", summary)
            reference_rows.append(summary)

    save_csv(reference_index, reference_rows, REFERENCE_FIELDS)
    status = {
        "status": "complete" if len(reference_rows) >= expected else "partial",
        "etas": [float(eta) for eta in config.get("etas", [])],
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


def _find_reference_index(part_root: Path, config: dict[str, Any]) -> Path:
    run_name = str(config.get("run_name", ""))
    candidates = [
        _stage_raw_root(part_root) / run_name / "04_exact_reference_search" / "reference_index.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate reference_index.csv under 03_reference_search/raw_outputs")


def provenance_alignment(part_root: Path = STAGE_ROOT, config: dict[str, Any] | None = None) -> dict[str, Any]:
    part_root = Path(part_root).resolve()
    config = config or merged_config(part_root / "config" / "default.json")
    raw_root = _stage_raw_root(part_root)
    metadata_paths = sorted(raw_root.glob("noise_eta_*/ref_*/reference_metadata.json"))
    if not metadata_paths:
        return {"available": False, "error": f"no reference metadata under {raw_root / 'noise_eta_*' / 'ref_*'}"}
    offset = int(config.get("canonical_ref_offset", 1))
    seed_starts = {str(key): int(value) for key, value in dict(config.get("attempt_seed_starts", {})).items()}
    theta_exists = 0
    metadata_matches_layout = 0
    dataset_exists = 0
    seed_matches = 0
    p_matches = 0
    exact_matches = 0
    optimizers: set[str] = set()
    eta_counts: dict[str, int] = {}
    for metadata_path in metadata_paths:
        metadata = load_json(metadata_path)
        eta = float(metadata["eta"])
        noise_eta = str(metadata.get("noise_eta", noise_eta_token(eta)))
        eta_counts[noise_eta] = eta_counts.get(noise_eta, 0) + 1
        source_ref_id = int(metadata.get("source_ref_id", int(metadata_path.parent.name.removeprefix("ref_")) - offset))
        theta_path = metadata_path.parent / "theta.npy"
        if theta_path.exists():
            theta_exists += 1
        if metadata_path.parent.parent.name == noise_eta and metadata_path.parent.name.startswith("ref_"):
            metadata_matches_layout += 1
        dataset_path = _resolve_path(part_root, metadata.get("dataset_payload_path", metadata.get("source_dataset_path", "")))
        if dataset_path.exists():
            dataset_exists += 1
        eta_key = _eta_key(eta)
        if eta_key in seed_starts and int(float(metadata.get("attempt_seed", -1))) == seed_starts[eta_key] + source_ref_id:
            seed_matches += 1
        if int(float(metadata.get("P", 0))) == P:
            p_matches += 1
        if float(metadata.get("train_error", "nan")) == 0.0:
            exact_matches += 1
        optimizers.add(str(metadata.get("optimizer_chain", "")))
    return {
        "available": True,
        "reference_rows": int(len(metadata_paths)),
        "eta_counts": dict(sorted(eta_counts.items())),
        "theta_paths_exist": int(theta_exists),
        "metadata_paths_match_raw_layout": int(metadata_matches_layout),
        "dataset_paths_exist": int(dataset_exists),
        "attempt_seeds_match_config": int(seed_matches),
        "P_matches_mnist10_architecture": int(p_matches),
        "train_error_zero_rows": int(exact_matches),
        "optimizer_chains": sorted(optimizers),
        "fully_aligned": bool(
            metadata_paths
            and theta_exists == len(metadata_paths)
            and metadata_matches_layout == len(metadata_paths)
            and dataset_exists == len(metadata_paths)
            and seed_matches == len(metadata_paths)
            and p_matches == len(metadata_paths)
            and exact_matches == len(metadata_paths)
            and optimizers == {"adam"}
        ),
    }


def materialize_canonical_layout(
    *,
    part_root: Path = STAGE_ROOT,
    config: dict[str, Any] | None = None,
    reference_index_path: Path | None = None,
) -> Path:
    part_root = Path(part_root).resolve()
    config = config or merged_config(part_root / "config" / "default.json")
    reference_index_path = reference_index_path or _find_reference_index(part_root, config)
    rows = _read_csv(reference_index_path)
    offset = int(config.get("canonical_ref_offset", 1))
    dataset_rows = {
        noise_eta_token(float(row["eta"])): row
        for row in discover_dataset_rows(part_root, config)
    }
    for row in rows:
        eta = float(_coalesce(row, "eta", default=eta_from_rule(str(row["rule"]))))
        noise_eta = noise_eta_token(eta)
        source_ref_id = int(row["ref_id"])
        ref = f"ref_{source_ref_id + offset:03d}"
        ref_dir = ensure_dir(_stage_raw_root(part_root) / noise_eta / ref)
        local_theta = ref_dir / "theta.npy"
        source_theta = _resolve_path(part_root, row["theta_path"])
        if source_theta.exists() and source_theta.resolve() != local_theta.resolve():
            if bool(config.get("force", False)) or not local_theta.exists():
                shutil.copy2(source_theta, local_theta)
        dataset_path = _resolve_path(part_root, dataset_rows.get(noise_eta, {}).get("dataset_path", row["dataset_path"]))
        metadata = {
            "noise_eta": noise_eta,
            "eta": float(eta),
            "ref": ref,
            "source_ref_id": int(source_ref_id),
            "attempt_seed": int(float(_coalesce(row, "attempt_seed", default=0))),
            "optimizer_chain": str(_coalesce(row, "optimizer_chain", "phase", default="")),
            "P": int(float(_coalesce(row, "P", default=P))),
            "train_error": float(_coalesce(row, "train_error", default=float("nan"))),
            "test_error": float(_coalesce(row, "test_error", default=float("nan"))),
            "CE_mean_train": float(_coalesce(row, "CE_mean_train", "ce_mean_train", default=float("nan"))),
            "theta_norm": float(_coalesce(row, "theta_norm", default=float("nan"))),
            "theta_payload_path": _repo_relative(local_theta),
            "theta_payload_exists": local_theta.exists(),
            "theta_init_payload_path": _repo_relative(ref_dir / "theta_init.npy"),
            "theta_init_payload_exists": (ref_dir / "theta_init.npy").exists(),
            "dataset_payload_path": _repo_relative(dataset_path),
            "dataset_payload_exists": dataset_path.exists(),
            "source_theta_path": str(row["theta_path"]),
            "source_theta_exists": source_theta.exists(),
            "source_dataset_path": str(row["dataset_path"]),
            "source_dataset_exists": _resolve_path(part_root, row["dataset_path"]).exists(),
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
        reference_index = _find_reference_index(part_root, config)
    except FileNotFoundError:
        raw_root = _stage_raw_root(part_root)
        if sorted(raw_root.glob("noise_eta_*/ref_*/reference_metadata.json")):
            return raw_root
        raise
    return materialize_canonical_layout(part_root=part_root, config=config, reference_index_path=reference_index)


def check_layout(part_root: Path = STAGE_ROOT) -> dict[str, Any]:
    part_root = Path(part_root).resolve()
    config = merged_config(part_root / "config" / "default.json")
    source_files = sorted(path.relative_to(part_root).as_posix() for path in (part_root / "src").glob("*.py"))
    utility_files = sorted(path.relative_to(part_root).as_posix() for path in (part_root / "src" / "utils").glob("*.py"))
    theta_files = sorted((part_root / "raw_outputs").glob("noise_eta_*/ref_*/theta.npy"))
    metadata_files = sorted((part_root / "raw_outputs").glob("noise_eta_*/ref_*/reference_metadata.json"))
    noise_eta_dirs = sorted((part_root / "raw_outputs").glob("noise_eta_*"))
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
        "raw_noise_eta_dirs": int(len(noise_eta_dirs)),
        "config_exists": (part_root / "config" / "default.json").exists(),
        "provenance_alignment": provenance_alignment(part_root, config),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or summarize MNIST label-noise reference search.")
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
