#!/usr/bin/env python3
"""Synchronize compact figure inputs with the included numerical authorities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from utils.paper_style import PACKAGE_ROOT, REPO_ROOT

MAP_PATH = PACKAGE_ROOT / "config/input_sources.json"


def sync_inputs(*, refresh=False):
    changed = []
    for item in json.loads(MAP_PATH.read_text())["sources"]:
        source, staged = REPO_ROOT / item["source"], PACKAGE_ROOT / item["staged"]
        if not source.is_file():
            # Representative dataset snapshots are shipped; large raw trees are not.
            if "/raw_outputs/" in item["source"] and staged.is_file(): continue
            raise FileNotFoundError(source)
        if source.resolve() == staged.resolve(): continue
        differs = not staged.is_file() or source.read_bytes() != staged.read_bytes()
        if differs:
            if not refresh: raise ValueError(f"staged input differs from source: {item['staged']}; run build normally to sync")
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            changed.append(item["staged"])
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    changed = sync_inputs(refresh=args.refresh)
    print(f"figure input sync PASS: {len(changed)} updated")


if __name__ == "__main__": main()
