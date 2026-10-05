"""Render selected paper figures using the preserved editable notebook settings."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys


def render_figures(figure_ids, *, force=False):
    package = Path(__file__).resolve().parents[2]
    notebook = json.loads((package / "releases/rebuild_all_paper_figures.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    manifest = json.loads((package / "config/figure_manifest.json").read_text())
    functions = {spec["id"]: f"fig{index:02d}" for index,spec in enumerate(manifest["figures"],1)}
    namespace = {"__name__": "paper_notebook_build"}
    original_cwd = Path.cwd()
    try:
        os.chdir(package)
        exec(compile(cells[0], "paper_notebook_setup", "exec"), namespace)
        style = sys.modules["utils.paper_style"]
        original_flag = style.PRESERVE_EXISTING_OUTPUTS
        style.PRESERVE_EXISTING_OUTPUTS = not force
        try:
            for figure_id in figure_ids:
                name = functions[figure_id]
                matching = [cell for cell in cells[1:] if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == name for n in ast.walk(ast.parse(cell)))]
                if len(matching) != 1: raise ValueError(f"notebook needs one {name} cell")
                exec(compile(matching[0],name,"exec"),namespace)
        finally:
            style.PRESERVE_EXISTING_OUTPUTS = original_flag
    finally:
        os.chdir(original_cwd)
