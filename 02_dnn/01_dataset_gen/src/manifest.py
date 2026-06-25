from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from io_utils import ensure_dir, now_iso, save_json


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _run_name(config: dict[str, Any], cfg_hash: str) -> str:
    for key in ("run_name", "name"):
        value = config.get(key)
        if value:
            return str(value)
    run = config.get("run")
    if isinstance(run, dict) and run.get("name"):
        return str(run["name"])
    beta_series = config.get("beta_series") or []
    datasets_per_cell = config.get("datasets_per_cell")
    if beta_series and datasets_per_cell:
        return f"{len(beta_series)}_beta_cell_{int(datasets_per_cell)}_dataset"
    return f"dataset_run_{cfg_hash}"


def build_roots(
    part_root: Path,
    config: dict[str, Any],
    *,
    run_group: str = "runs",
    upstream_manifest: Path | None = None,
) -> tuple[str, Path, Path]:
    cfg_hash = _config_hash(config)
    run_name = _run_name(config, cfg_hash)
    if run_group == "smoke_runs":
        summary_root = Path(part_root) / "smoke_runs" / run_name
        raw_root = summary_root / "raw_outputs" / "raw_datasets"
    else:
        summary_root = Path(part_root) / "raw_outputs" / run_name
        raw_root = summary_root / "raw_datasets"
    ensure_dir(raw_root)
    ensure_dir(summary_root)
    return cfg_hash, raw_root, summary_root


def write_manifest(
    summary_root: Path,
    *,
    pipeline_id: str,
    methodology_id: str,
    config_hash_value: str,
    config_path: str,
    raw_output_root: Path,
    upstream_refs: list[str] | None = None,
    summary_outputs: list[str] | None = None,
    dimension: int | None = None,
    status: str = "success",
    run_group: str = "runs",
    started_at: str | None = None,
) -> Path:
    manifest_path = Path(summary_root) / "manifest.json"
    save_json(
        manifest_path,
        {
            "pipeline_id": pipeline_id,
            "methodology_id": methodology_id,
            "config_hash": config_hash_value,
            "config_path": config_path,
            "raw_output_root": str(Path(raw_output_root)),
            "upstream_refs": upstream_refs or [],
            "summary_outputs": summary_outputs or [],
            "dimension": dimension,
            "status": status,
            "run_group": run_group,
            "started_at": started_at,
            "finished_at": now_iso(),
        },
    )
    return manifest_path
