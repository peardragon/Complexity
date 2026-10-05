#!/usr/bin/env python3
"""Validate the arXiv-v1 processed-data release rooted at directories 01--04.

This validator intentionally does not read ignored raw experiment payloads.
It checks the exact compact authorities used by the paper, the fixed digit-pair
panel, the four-stage Discussion closure, source syntax, and path isolation.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "04_discussion"
    / "provenance"
    / "ARXIV_V1_RELEASE_SCOPE.json"
)
RELEASE_ROOT_NAMES = (
    "01_theory",
    "02_dnn_synthetic",
    "03_dnn_mnist",
    "04_discussion",
)
IGNORED_WALK_DIRS = {
    "raw_outputs",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
TEXT_SUFFIXES_FOR_PATH_AUDIT = {".py", ".json", ".md", ".txt"}
EXPECTED_DIGIT_PAIRS = (
    "pair_4_9",
    "pair_3_8",
    "pair_5_9",
    "pair_2_7",
    "pair_4_5",
    "pair_0_2",
    "pair_2_9",
    "pair_7_8",
    "pair_4_6",
    "pair_1_6",
    "pair_0_9",
    "pair_0_1",
)
EXPECTED_FROZEN_PAIR_MANIFEST_SHA256 = (
    "59fa715666b897ecb8357fb9373dce5c2f973fbfa2c30f566bb1f7fc50f8c5e6"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV lacks a header: {path}") from error
        return sum(1 for _ in reader)


def iter_release_files() -> list[Path]:
    files: list[Path] = []
    for root_name in RELEASE_ROOT_NAMES:
        root = REPOSITORY_ROOT / root_name
        if not root.is_dir():
            raise FileNotFoundError(f"missing release root: {root}")
        for current, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = [
                name for name in directories if name not in IGNORED_WALK_DIRS
            ]
            base = Path(current)
            for filename in filenames:
                files.append(base / filename)
    return files


def validate_authorities(manifest: dict[str, Any]) -> int:
    authorities = manifest.get("authorities")
    if not isinstance(authorities, list) or not authorities:
        raise ValueError("release manifest has no authority list")
    observed_paths: set[str] = set()
    for record in authorities:
        if not isinstance(record, dict):
            raise ValueError("authority record is not an object")
        relative = str(record["path"])
        if relative in observed_paths:
            raise ValueError(f"duplicate authority path: {relative}")
        observed_paths.add(relative)
        if not relative.startswith(tuple(f"{name}/" for name in RELEASE_ROOT_NAMES)):
            raise ValueError(f"authority escapes release roots: {relative}")
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(
                f"paper authority hash mismatch: {relative}\n"
                f"expected={record['sha256']}\nactual={actual_hash}"
            )
        if path.suffix.lower() == ".csv":
            actual_rows = csv_row_count(path)
            if actual_rows != int(record["rows"]):
                raise ValueError(
                    f"paper authority row-count mismatch: {relative}: "
                    f"{actual_rows} != {record['rows']}"
                )
    return len(authorities)


def validate_digit_pair_panel() -> int:
    protocol = REPOSITORY_ROOT / "03_dnn_mnist" / "digit_pairwise_complexity"
    manifest_paths = [
        protocol / stage / "config" / "frozen_pair_manifest.json"
        for stage in (
            "01_dataset",
            "02_complexity_measure",
            "03_reference_search",
            "04_sampling",
            "05_proxy_local_entropy",
        )
    ]
    hashes = {sha256(path) for path in manifest_paths}
    if hashes != {EXPECTED_FROZEN_PAIR_MANIFEST_SHA256}:
        raise ValueError(f"digit-pair frozen manifests differ: {sorted(hashes)}")
    payload = load_object(manifest_paths[0])
    selected = payload.get("expected_selected_pairs")
    if not isinstance(selected, list):
        raise ValueError("digit-pair frozen manifest lacks selected pairs")
    observed = tuple(str(item["pair_id"]) for item in selected)
    if observed != EXPECTED_DIGIT_PAIRS:
        raise ValueError(f"digit-pair panel differs from arXiv v1: {observed}")
    return len(manifest_paths)


def validate_discussion_closure() -> None:
    config = load_object(REPOSITORY_ROOT / "04_discussion" / "config" / "default.json")
    expected_stages = [
        "01_frozen_inputs",
        "02_normalized_radial_geometry",
        "03_antipodal_geometry",
        "04_random_label_reentrance",
    ]
    if config.get("stages") != expected_stages:
        raise ValueError(f"Discussion stage list differs: {config.get('stages')}")

    frozen = load_object(
        REPOSITORY_ROOT
        / "04_discussion"
        / "01_frozen_inputs"
        / "config"
        / "default.json"
    )
    allowed_prefixes = tuple(f"{name}/" for name in RELEASE_ROOT_NAMES)
    for source in frozen["repository_relative_sources"]:
        relative = str(source["path"])
        if not relative.startswith(allowed_prefixes):
            raise ValueError(f"Discussion input escapes release roots: {relative}")
        if not (REPOSITORY_ROOT / relative).is_file():
            raise FileNotFoundError(REPOSITORY_ROOT / relative)


def validate_known_exclusions(manifest: dict[str, Any]) -> int:
    exclusions = manifest.get("known_exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("release manifest exclusions are not a list")
    for relative in exclusions:
        path = REPOSITORY_ROOT / str(relative)
        if path.exists():
            raise ValueError(f"known non-paper artifact remains active: {relative}")
    return len(exclusions)


def validate_sources_and_paths(files: list[Path]) -> tuple[int, int, int]:
    python_count = 0
    json_count = 0
    path_audit_count = 0
    validator_path = Path(__file__).resolve()
    forbidden_tokens = (
        "misc" + "ellaneous/",
        "99_" + "paper/",
        "Fig" + "ures/",
        "back" + "ups/",
        "/" + "home/",
    )

    for path in files:
        if path.is_symlink():
            resolved = path.resolve()
            try:
                resolved.relative_to(REPOSITORY_ROOT.resolve())
            except ValueError as error:
                raise ValueError(f"release symlink escapes project: {path}") from error
        suffix = path.suffix.lower()
        if suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            python_count += 1
        elif suffix == ".json":
            with path.open(encoding="utf-8") as stream:
                json.load(stream)
            json_count += 1

        if suffix in TEXT_SUFFIXES_FOR_PATH_AUDIT and path.resolve() != validator_path:
            text = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                if token in text:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    raise ValueError(
                        f"non-release path token {token!r} remains in {relative}"
                    )
            path_audit_count += 1

    cache_paths: list[str] = []
    for root_name in RELEASE_ROOT_NAMES:
        root = REPOSITORY_ROOT / root_name
        for current, directories, filenames in os.walk(root, followlinks=False):
            for directory in directories:
                if directory in IGNORED_WALK_DIRS - {"raw_outputs"}:
                    cache_paths.append(str(Path(current) / directory))
            for filename in filenames:
                if filename.endswith((".pyc", ".pyo")):
                    cache_paths.append(str(Path(current) / filename))
    if cache_paths:
        raise ValueError(f"runtime caches remain in release roots: {cache_paths[:10]}")
    return python_count, json_count, path_audit_count


def main() -> None:
    manifest = load_object(MANIFEST_PATH)
    declared_roots = tuple(manifest.get("release_roots", []))
    if declared_roots != RELEASE_ROOT_NAMES:
        raise ValueError(f"release-root declaration differs: {declared_roots}")
    if manifest.get("paper", {}).get("figure_numbers") != list(range(1, 10)):
        raise ValueError("arXiv-v1 figure range must be exactly 1--9")
    if manifest.get("rendered_manuscript_assets_in_scope") is not False:
        raise ValueError("processed release must not claim rendered-figure scope")

    authority_count = validate_authorities(manifest)
    pair_manifest_count = validate_digit_pair_panel()
    validate_discussion_closure()
    exclusion_count = validate_known_exclusions(manifest)
    files = iter_release_files()
    python_count, json_count, path_audit_count = validate_sources_and_paths(files)

    result = {
        "status": "pass",
        "paper": "arXiv:2608.22361v1",
        "scope": "01--04 code and processed numerical authorities",
        "authority_hashes_checked": authority_count,
        "digit_pair_frozen_manifests_checked": pair_manifest_count,
        "known_nonpaper_artifacts_absent": exclusion_count,
        "python_sources_parsed": python_count,
        "json_files_loaded": json_count,
        "text_files_path_audited": path_audit_count,
        "discussion_stage_count": 4,
        "external_runtime_paths_found": 0,
        "runtime_caches_found": 0,
        "raw_payloads_opened": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
