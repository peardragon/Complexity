#!/usr/bin/env python3
"""Copy and validate only the public paper files, without raw data or Git history."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from validate_release import ROOT, ROOTS, validate

EXCLUDED = {"raw_outputs","raw_datasets","sample_payloads","transfer_logs","source_cache","openml_cache","run_state","execution_state","shards","sampling_shards","raw","__pycache__",".ipynb_checkpoints",".pytest_cache",".mypy_cache",".ruff_cache",".git","backups",".venv","venv","env","build","dist",".idea",".vscode"}
BINARY_DATA = {".npz",".npy",".pt",".pth",".ckpt",".pkl",".pickle",".h5",".hdf5",".joblib",".parquet",".feather",".arrow",".onnx"}
SNAPSHOTS = {f"Figures/paper_figures/figure_inputs/synthetic/dataset_000/data_beta_{beta}.npz" for beta in ("0p05","0p15","0p27","0p39")}
TOP = ("README.md","requirements.txt","CITATION.cff",".gitignore",".gitattributes")


def public_files():
    files=[]
    for name in ROOTS:
        for path in (ROOT/name).rglob("*"):
            if not path.is_file():continue
            relative=path.relative_to(ROOT)
            if any(part in EXCLUDED for part in relative.parts):continue
            if path.suffix in BINARY_DATA and relative.as_posix() not in SNAPSHOTS:continue
            if path.name.endswith((".tmp",".lock",".log",".pid",".sock",".pyc",".pyo",".pyd",".jsonl.gz",".zip",".tar",".tar.gz",".tgz",".tar.bz2",".tar.xz",".7z")):continue
            if path.name==".DS_Store" or (path.name.startswith(".env") and path.name!=".env.example"):continue
            if path.is_symlink():raise ValueError(f"symlink is not a release file: {relative}")
            files.append(path)
    files.extend(ROOT/name for name in TOP)
    return sorted(set(files))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    source_report=validate()
    destination=args.destination.resolve()
    if destination==ROOT or ROOT in destination.parents:raise ValueError("release destination must be outside the source tree")
    if destination.exists():raise FileExistsError(f"refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    inventory=[]
    for source in public_files():
        relative=source.relative_to(ROOT)
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        inventory.append({"path":relative.as_posix(),"bytes":target.stat().st_size,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    packages=("numpy","scipy","numba","torch","scikit-learn","umap-learn","pynndescent","pandas","matplotlib","pillow","pymupdf","nbformat","nbconvert","ipykernel")
    manifest={"paper":"arXiv:2608.22361v1","release_roots":list(ROOTS),"raw_excluded":True,"small_visual_snapshots":sorted(SNAPSHOTS),"file_count":len(inventory),"total_bytes":sum(x["bytes"] for x in inventory),"python":sys.version.split()[0],"tested_packages":{p:importlib.metadata.version(p) for p in packages},"files":inventory,"source_validation":source_report}
    (destination/"RELEASE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    subprocess.run([sys.executable,str(destination/"Figures/paper_figures/src/validate_release.py")],cwd=destination,check=True)
    print(json.dumps({"status":"ready","destination":str(destination),"file_count":len(inventory),"total_MiB":manifest["total_bytes"]/2**20},indent=2))


if __name__=="__main__":main()
