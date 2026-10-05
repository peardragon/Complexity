#!/usr/bin/env python3
"""Read-only validation of the five-root, raw-excluded paper release."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
ROOTS = ("01_theory", "02_dnn_synthetic", "03_dnn_mnist", "04_discussion", "Figures")
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(newline="") as stream: return list(csv.DictReader(stream))


def validate(*, check_figures=True):
    scope = json.loads((ROOT / "04_discussion/provenance/ARXIV_V1_RELEASE_SCOPE.json").read_text())
    for record in scope["authorities"]:
        path = ROOT / record["path"]
        if digest(path) != record["sha256"]: raise ValueError(f"numerical authority changed: {record['path']}")
        if path.suffix == ".csv" and len(read_rows(path)) != int(record["rows"]): raise ValueError(f"authority row count differs: {path}")
    python_count = 0
    for name in ROOTS:
        for path in (ROOT/name).rglob("*.py"):
            if "raw_outputs" in path.parts: continue
            ast.parse(path.read_text(),filename=str(path)); python_count += 1
    theory_stage=ROOT/"01_theory/02_theory_sampling"
    theory_config=json.loads((theory_stage/"config/default.json").read_text())
    dataset_count=len(theory_config["problem"]["dimensions"])*int(theory_config["problem"]["datasets_per_dimension"])
    for script,expected in (("make_datasets.py",dataset_count),("make_references.py",dataset_count*int(theory_config["problem"]["references_per_dataset"]))):
        probe=subprocess.run([sys.executable,str(theory_stage/"src"/script)],cwd=ROOT,check=True,capture_output=True,text=True)
        report=json.loads(probe.stdout)
        if report["written"]!=0 or report["missing"]+report["skipped_existing"]!=expected:raise ValueError(f"invalid read-only input generation plan: {script}")
    subprocess.run([sys.executable,str(ROOT/"Figures/paper_figures/src/build_mnist_umap_assets.py"),"--check-only"],cwd=ROOT,check=True,capture_output=True,text=True)
    for name in ("02_dnn_synthetic", "03_dnn_mnist/label_noise_sweep", "03_dnn_mnist/digit_pairwise_complexity"):
        subprocess.run([sys.executable,str(ROOT/name/"05_proxy_local_entropy/src/make_r1_accuracy.py"),"--check-only"],cwd=ROOT,check=True,capture_output=True,text=True)
        for stage in ("02_complexity_measure", "04_sampling", "05_proxy_local_entropy"):
            subprocess.run([sys.executable,str(ROOT/name/stage/"src/make_summarized_outputs.py"),"--check-only"],cwd=ROOT,check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(ROOT/"01_theory/02_theory_sampling/src/make_summarized_outputs.py"),"--check-only"],cwd=ROOT,check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,"-c","import sys;sys.path.insert(0,sys.argv[1]);from utils.discussion_pipeline import validate_all;validate_all(write_report=False)",str(ROOT/"04_discussion/src")],cwd=ROOT,check=True,capture_output=True,text=True)
    package=ROOT/"Figures/paper_figures"
    sources=json.loads((package/"config/input_sources.json").read_text())["sources"]
    for item in sources:
        original,staged=ROOT/item["source"],package/item["staged"]
        if not staged.is_file():raise FileNotFoundError(staged)
        if original.is_file() and original.read_bytes()!=staged.read_bytes():raise ValueError(f"source/staged input differs: {item['staged']}")
        if not original.is_file() and "/raw_outputs/" not in item["source"]:raise FileNotFoundError(original)
    notebook=json.loads((package/"releases/rebuild_all_paper_figures.ipynb").read_text())
    cells=[c for c in notebook["cells"] if c["cell_type"]=="code"]
    if any(c.get("execution_count") is None for c in cells):raise ValueError("paper notebook is not executed")
    if any(o.get("output_type")=="error" for c in cells for o in c.get("outputs",[])):raise ValueError("paper notebook has error outputs")
    if check_figures:
        subprocess.run([sys.executable,str(package/"src/build_all.py"),"--check-only"],cwd=ROOT,check=True,capture_output=True,text=True)
        figures=json.loads((package/"config/figure_manifest.json").read_text())["figures"]
        if len(figures)!=9:raise ValueError("paper main figure count must be nine")
        receipt=json.loads((package/"receipts/build_receipt.json").read_text())
        notebook_sources = ["".join(cell["source"]) for cell in cells]
        if receipt.get("notebook_source_sha256") != hashlib.sha256(json.dumps(notebook_sources,ensure_ascii=False).encode()).hexdigest():raise ValueError("figure receipt has stale notebook settings")
        if receipt["figure_manifest_sha256"]!=digest(package/"config/figure_manifest.json"):raise ValueError("figure receipt has stale manifest")
        if receipt["style_config"]["sha256"]!=digest(package.parent/receipt["style_config"]["path"]):raise ValueError("figure receipt has stale common style")
        if receipt["inventory"]["sha256"]!=digest(package/receipt["inventory"]["path"]):raise ValueError("figure inventory changed")
        for item in receipt["inputs"]:
            if digest(package/item["path"])!=item["sha256"]:raise ValueError(f"figure receipt has stale input: {item['path']}")
        for figure in receipt["outputs"]:
            for item in figure["files"].values():
                if digest(package/item["path"])!=item["sha256"]:raise ValueError(f"figure receipt has stale output: {item['path']}")
        for index,figure in enumerate(figures,1):
            matching=[cell for cell in cells if any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id==f"fig{index:02d}" for node in ast.walk(ast.parse("".join(cell["source"]))))]
            if len(matching)!=1:raise ValueError(f"notebook needs one fig{index:02d} cell")
            images=[output["data"]["image/png"] for output in matching[0].get("outputs",[]) if "image/png" in output.get("data",{})]
            if len(images)!=1 or base64.b64decode(images[0])!=(package/figure["output"]).with_suffix(".png").read_bytes():raise ValueError(f"notebook/output differs: {figure['id']}")
        companion=ROOT/"Figures/04_discussion"
        formats=json.loads((companion/"config/default.json").read_text())["formats"]
        for name in ("fig09_hardening_3d_composite","appendix_d_antipodal","random_label_reentrance"):
            for suffix in formats:
                path=companion/name/f"{name}.{suffix}"
                if not path.is_file() or path.stat().st_size==0:raise FileNotFoundError(path)
                if name=="fig09_hardening_3d_composite" and path.read_bytes()!=(package/figures[-1]["output"]).with_suffix(f".{suffix}").read_bytes():raise ValueError(f"Discussion Fig. 9 copy differs: {suffix}")
    return {"status":"pass","paper":"arXiv:2608.22361v1","release_roots":list(ROOTS),"numerical_authorities":len(scope["authorities"]),"figure_inputs":len(sources),"main_figures":9,"python_sources":python_count,"executed_notebook_cells":len(cells),"r1_accuracy_reconstruction":"three original capture tables validated","missing_input_entrypoints":"theory planning and four MNIST visual inputs checked read-only","raw_required_for_checks":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-figures",action="store_true")
    args=parser.parse_args()
    print(json.dumps(validate(check_figures=not args.skip_figures),indent=2))


if __name__=="__main__":main()
