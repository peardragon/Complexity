from __future__ import annotations

import argparse
import ast
import csv
import json
import runpy
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FIGURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURES_DIR.parent
MANIFEST_PATH = FIGURES_DIR / "manifest.csv"
README_PATH = FIGURES_DIR / "README.md"
NOTEBOOK_PATH = FIGURES_DIR / "rebuild_all_figures_from_summaries.ipynb"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}
PRESERVED_ROOT_FILES = {"rebuild_figures.py", NOTEBOOK_PATH.name}


@dataclass(frozen=True)
class GallerySpec:
    group: str
    target_prefix: str
    source_root: Path
    description: str


STAGE_BUILD_SCRIPTS = (
    PROJECT_ROOT / "01_theory" / "figures" / "src" / "make_figures.py",
    PROJECT_ROOT / "02_dnn_synthetic" / "figures" / "src" / "make_figures.py",
    PROJECT_ROOT / "03_dnn_mnist" / "figures" / "src" / "make_figures.py",
)

GALLERIES = (
    GallerySpec(
        "theory",
        "theory/01_theory_analytic",
        PROJECT_ROOT / "01_theory" / "01_theory_analytic" / "figures",
        "analytic theory figures",
    ),
    GallerySpec(
        "theory",
        "theory/02_theory_sampling",
        PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "figures",
        "theory sampling figures",
    ),
    GallerySpec(
        "theory",
        "theory/summary",
        PROJECT_ROOT / "01_theory" / "figures",
        "combined theory figures",
    ),
    GallerySpec(
        "dnn_synthetic",
        "dnn_synthetic",
        PROJECT_ROOT / "02_dnn_synthetic" / "figures",
        "synthetic DNN stage gallery",
    ),
    GallerySpec(
        "dnn_mnist",
        "dnn_mnist",
        PROJECT_ROOT / "03_dnn_mnist" / "figures",
        "MNIST stage gallery collected from label-noise and manual-rule outputs",
    ),
)

def _relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _title_from_path(path: Path) -> str:
    title = path.stem.replace("_", " ").replace("-", " ")
    return " ".join(title.split())


def _figure_input_paths(category_path: str) -> list[str]:
    path = Path(category_path)
    parts = path.parts
    stem = path.stem

    if category_path == "theory/01_theory_analytic/phi_by_analytic_solution_alpha0p1.png":
        return ["01_theory/01_theory_analytic/summarized_outputs/phi_by_analytic_solution_alpha0p1.csv"]
    if category_path == "theory/02_theory_sampling/phi_by_sampling/phi_by_sampling.png":
        return ["01_theory/02_theory_sampling/summarized_outputs/figure_inputs/phi_by_sampling"]
    if category_path.startswith("theory/02_theory_sampling/logZ_split_distributions/"):
        return [f"01_theory/02_theory_sampling/summarized_outputs/figure_inputs/logZ_split/{stem}.csv"]
    if category_path.startswith("theory/summary/"):
        return [
            "01_theory/01_theory_analytic/summarized_outputs/phi_by_analytic_solution_alpha0p1.csv",
            "01_theory/02_theory_sampling/summarized_outputs/figure_inputs/phi_by_sampling",
        ]

    if category_path == "dnn_synthetic/01_dataset/sample_figure.png":
        return ["02_dnn_synthetic/01_dataset/summarized_outputs/figure_inputs/sample_figures/selected_sample_indices.csv"]
    if category_path == "dnn_synthetic/01_dataset/spin_dynamics_phase_transition.png":
        return ["02_dnn_synthetic/01_dataset/summarized_outputs/figure_inputs/spin_dynamics/spin_alignment_by_beta.csv"]
    if category_path == "dnn_synthetic/02_complexity_measure/beta_complexity_figure.png":
        return ["02_dnn_synthetic/02_complexity_measure/summarized_outputs/beta_complexity_summary.csv"]
    if category_path.startswith("dnn_synthetic/04_sampling/logZ_split_distributions/"):
        return [f"02_dnn_synthetic/04_sampling/summarized_outputs/figure_inputs/logZ_split/{stem}.csv"]
    if category_path.startswith("dnn_synthetic/05_proxy_local_entropy/phase_like_A_by_"):
        return [
            f"02_dnn_synthetic/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv",
            f"02_dnn_synthetic/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/phase_derivative_curves.csv",
        ]
    if category_path.startswith("dnn_synthetic/05_proxy_local_entropy/"):
        return [f"02_dnn_synthetic/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv"]

    if category_path.startswith("dnn_mnist/01_dataset/"):
        series = parts[2]
        return [f"03_dnn_mnist/{series}/01_dataset/summarized_outputs/figure_inputs/sample_figure/selected_sample_indices.csv"]
    if category_path.startswith("dnn_mnist/02_complexity_measure/label_noise_sweep/"):
        return ["03_dnn_mnist/label_noise_sweep/02_complexity_measure/summarized_outputs/eta_complexity_summary.csv"]
    if category_path.startswith("dnn_mnist/02_complexity_measure/manual_rules/"):
        return ["03_dnn_mnist/manual_rules/02_complexity_measure/summarized_outputs/manual_rule_complexity_summary.csv"]
    if category_path.startswith("dnn_mnist/03_reference_search/label_noise_sweep/"):
        return [
            "03_dnn_mnist/label_noise_sweep/03_reference_search/summarized_outputs/figure_inputs/reference_quality/reference_quality_by_eta.csv",
            "03_dnn_mnist/label_noise_sweep/03_reference_search/summarized_outputs/figure_inputs/reference_quality/reference_quality_by_ref.csv",
        ]
    if category_path.startswith("dnn_mnist/03_reference_search/manual_rules/"):
        return [
            "03_dnn_mnist/manual_rules/03_reference_search/summarized_outputs/figure_inputs/reference_quality/reference_quality_by_rule.csv",
            "03_dnn_mnist/manual_rules/03_reference_search/summarized_outputs/figure_inputs/reference_quality/reference_quality_by_ref.csv",
        ]
    if category_path.startswith("dnn_mnist/04_sampling/"):
        series = parts[2]
        return [f"03_dnn_mnist/{series}/04_sampling/summarized_outputs/figure_inputs/logZ_split/{stem}.csv"]
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/label_noise_sweep/phase_like_A_by_"):
        return [
            f"03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv",
            f"03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/phase_derivative_curves.csv",
        ]
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/label_noise_sweep/"):
        return [f"03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv"]
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/manual_rules/phase_like_A_by_"):
        return [
            f"03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv",
            f"03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/phase_derivative_curves.csv",
        ]
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/manual_rules/"):
        return [f"03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv"]
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/merged/phase_like_A_by_complexity"):
        return [
            "03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/phase_like_A_by_complexity/phase_like_A_by_complexity.csv",
            "03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/phase_like_A_by_complexity/phase_derivative_curves.csv",
            "03_dnn_mnist/label_noise_sweep/02_complexity_measure/summarized_outputs/eta_complexity_summary.csv",
            "03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/phase_like_A_by_complexity/phase_like_A_by_complexity.csv",
            "03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/phase_like_A_by_complexity/phase_derivative_curves.csv",
            "03_dnn_mnist/manual_rules/02_complexity_measure/summarized_outputs/manual_rule_complexity_summary.csv",
        ]
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/merged/phase_like_A_by_eta"):
        return [
            "03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/phase_like_A_by_eta/phase_like_A_by_eta.csv",
            "03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/phase_like_A_by_eta/phase_derivative_curves.csv",
            "03_dnn_mnist/label_noise_sweep/02_complexity_measure/summarized_outputs/eta_complexity_summary.csv",
            "03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/phase_like_A_by_rule/phase_like_A_by_rule.csv",
            "03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/phase_like_A_by_rule/phase_derivative_curves.csv",
            "03_dnn_mnist/manual_rules/02_complexity_measure/summarized_outputs/manual_rule_complexity_summary.csv",
        ]
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/merged/"):
        return [
            f"03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv",
            f"03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv",
        ]

    return []


def _clear_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIGURES_DIR.iterdir():
        if path.name in PRESERVED_ROOT_FILES:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _run_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    old_argv = sys.argv[:]
    sys.argv = [str(path)]
    try:
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv


def _load_manifest_metadata(root: Path) -> dict[Path, dict[str, str]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    module_root = root.parent
    lookup: dict[Path, dict[str, str]] = {}
    for item in payload.get("figures", []):
        source = (module_root / str(item["path"])).resolve()
        input_paths = []
        for raw_input in item.get("inputs", []):
            input_paths.append((module_root / str(raw_input)).resolve())
        lookup[source] = {
            "title": str(item.get("title") or _title_from_path(source)),
            "input_paths": ";".join(_relative_to_project(path) for path in input_paths if path.exists()),
        }
    return lookup


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _is_internal_path(path: Path) -> bool:
    return any(part in {"src", "__pycache__"} for part in path.parts)


def _copy_gallery(spec: GallerySpec, manifest_rows: list[dict[str, str]]) -> list[Path]:
    if not spec.source_root.exists():
        return []
    metadata = _load_manifest_metadata(spec.source_root)
    copied: list[Path] = []
    for source in sorted(spec.source_root.rglob("*")):
        if not _is_image(source):
            continue
        relative = source.relative_to(spec.source_root)
        if _is_internal_path(relative):
            continue
        target = FIGURES_DIR / spec.target_prefix / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)

        meta = metadata.get(source.resolve(), {})
        category_path = target.relative_to(FIGURES_DIR).as_posix()
        explicit_inputs = _figure_input_paths(category_path)
        manifest_rows.append(
            {
                "group": spec.group,
                "section": spec.target_prefix,
                "title": meta.get("title") or _title_from_path(source),
                "figure": target.name,
                "category_path": category_path,
                "source_path": _relative_to_project(source),
                "input_paths": ";".join(explicit_inputs) if explicit_inputs else meta.get("input_paths", ""),
            }
        )
    return copied


def _write_manifest(rows: list[dict[str, str]]) -> None:
    fields = ["group", "section", "title", "figure", "category_path", "source_path", "input_paths"]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(rows: list[dict[str, str]]) -> None:
    group_counts = {group: sum(1 for row in rows if row["group"] == group) for group in sorted({row["group"] for row in rows})}
    section_counts = {}
    for row in rows:
        section_counts[row["section"]] = section_counts.get(row["section"], 0) + 1

    lines = [
        "# Complexity Figures",
        "",
        "This folder is a central, reproducible figure bundle. It is rebuilt from the current stage-level figure galleries and summarized-output figure builders.",
        "",
        "Top-level PNG duplicates and stale legacy groups are intentionally removed by `rebuild_figures.py`.",
        "",
        "## Groups",
        "",
    ]
    for group, count in group_counts.items():
        lines.append(f"- `{group}/`: {count} figures.")
    lines.extend(["", "## Sections", ""])
    for section, count in sorted(section_counts.items()):
        lines.append(f"- `{section}/`: {count} figures.")
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "Run `python Figures/rebuild_figures.py` from the project root.",
            "",
            "The script first refreshes the stage galleries under `01_theory/figures`, `02_dnn_synthetic/figures`, and `03_dnn_mnist/figures`, then recopies current PNG outputs into grouped central folders.",
            "",
            "`manifest.csv` records the grouped output, source image, and source summarized inputs when a stage manifest exposes them.",
            "",
            "`rebuild_all_figures_from_summaries.ipynb` is an executable one-PNG-per-cell notebook. Each figure cell contains the concrete CSV/image loading and plotting code for one PNG, saves directly into `Figures/`, and displays that image inline.",
            "",
        ]
    )
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def _notebook_code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _notebook_markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _split_notebook_inputs(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    return [item for item in text.split(";") if item]


def _source_blocks(script_rel: str, names: tuple[str, ...]) -> str:
    source_path = PROJECT_ROOT / script_rel
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    chunks: list[str] = []
    for name in names:
        node = nodes.get(name)
        if node is None or node.end_lineno is None:
            raise KeyError(f"{name} not found in {script_rel}")
        chunks.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n\n".join(chunks)


def _cell_preamble() -> str:
    return (
        "from __future__ import annotations\n\n"
        "from collections import defaultdict\n"
        "from pathlib import Path\n"
        "import csv\n"
        "import math\n"
        "import re\n\n"
        "from IPython.display import Image, display\n"
        "from matplotlib.colors import Normalize\n"
        "import matplotlib.image as mpimg\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from scipy.stats import pearsonr\n\n"
        "FIGURES_DIR = Path.cwd()\n"
        "if FIGURES_DIR.name != 'Figures':\n"
        "    candidates = [Path.cwd() / 'Figures', Path.cwd().parent / 'Figures']\n"
        "    for candidate in candidates:\n"
        "        if (candidate / 'manifest.csv').exists():\n"
        "            FIGURES_DIR = candidate.resolve()\n"
        "            break\n"
        "    else:\n"
        "        raise RuntimeError('Run this notebook from the project root or the Figures directory.')\n"
        "ROOT = FIGURES_DIR.parent\n\n"
        "plt.rcParams.update({\n"
        "    'figure.dpi': 120,\n"
        "    'savefig.dpi': 220,\n"
        "    'axes.grid': True,\n"
        "    'grid.alpha': 0.25,\n"
        "    'axes.spines.top': False,\n"
        "    'axes.spines.right': False,\n"
        "})\n"
    )


def _path_assignments(category_path: str, input_paths: list[str]) -> str:
    return "\n".join(
        [
            f"output_png = FIGURES_DIR / {category_path!r}",
            "input_paths = [" + ", ".join(f"ROOT / {path!r}" for path in input_paths) + "]",
        ]
    )


def _display_tail() -> str:
    return (
        "for path in input_paths:\n"
        "    print('input:', path.relative_to(ROOT).as_posix())\n"
        "print('output:', output_png.relative_to(ROOT).as_posix())\n"
        "display(Image(filename=str(output_png)))"
    )


def _join_cell(*parts: str) -> str:
    return "\n\n".join(part.strip("\n") for part in parts if part and part.strip()) + "\n"


def _theory_analytic_cell(category_path: str) -> str:
    input_csv = "01_theory/01_theory_analytic/summarized_outputs/phi_by_analytic_solution_alpha0p1.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        (
            "input_csv = input_paths[0]\n"
            "df = pd.read_csv(input_csv).sort_values('r')\n"
            "y_col = 'phi_rel' if 'phi_rel' in df.columns else 'phi'\n\n"
            "output_png.parent.mkdir(parents=True, exist_ok=True)\n"
            "plt.figure(figsize=(7.0, 4.4))\n"
            "plt.plot(df['r'], df[y_col], marker='o', linewidth=2.0, color='#2457a7')\n"
            "plt.xlabel('d')\n"
            "plt.ylabel('phi(d) - phi(d0)' if y_col == 'phi_rel' else 'phi(d)')\n"
            "plt.title('Analytic full-RS solution, alpha=0.1')\n"
            "plt.grid(True, alpha=0.28)\n"
            "plt.tight_layout()\n"
            "plt.savefig(output_png, dpi=180)\n"
            "plt.show()\n"
            "plt.close()"
        ),
        _display_tail(),
    )


def _theory_sampling_phi_cell(category_path: str) -> str:
    input_root = "01_theory/02_theory_sampling/summarized_outputs/figure_inputs/phi_by_sampling"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_root]),
        (
            "input_root = input_paths[0]\n"
            "files = sorted(input_root.glob('N_*.csv'))\n"
            "if not files:\n"
            "    raise FileNotFoundError(f'no N_*.csv files found under {input_root}')\n"
            "df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True, sort=False).sort_values(['N', 'r'])\n\n"
            "def series_value(frame: pd.DataFrame) -> pd.Series:\n"
            "    if 'phi_emp_rel' in frame.columns:\n"
            "        return frame['phi_emp_rel']\n"
            "    base = frame.sort_values('r')['phi_emp'].iloc[0]\n"
            "    return frame['phi_emp'] - base\n\n"
            "output_png.parent.mkdir(parents=True, exist_ok=True)\n"
            "plt.figure(figsize=(7.4, 4.8))\n"
            "for n_value, group in df.groupby('N', sort=True):\n"
            "    group = group.sort_values('r')\n"
            "    plt.plot(group['r'], series_value(group), marker='o', linewidth=1.7, label=f'N={int(n_value)}')\n"
            "plt.xlabel('d')\n"
            "plt.ylabel('empirical phi(d) - phi(d0)')\n"
            "plt.title('Two-pool shell sampling, alpha=0.1')\n"
            "plt.grid(True, alpha=0.28)\n"
            "plt.legend(title='system size', fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.savefig(output_png, dpi=180)\n"
            "plt.show()\n"
            "plt.close()"
        ),
        _display_tail(),
    )


def _theory_logz_cell(category_path: str) -> str:
    stem = Path(category_path).stem
    input_csv = f"01_theory/02_theory_sampling/summarized_outputs/figure_inputs/logZ_split/{stem}.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        _source_blocks(
            "01_theory/02_theory_sampling/src/make_figures.py",
            ("_radius_label", "_far_split_start", "_xtick_labels", "_plot_value_column", "plot_logz_split_distribution"),
        ),
        "input_csv = input_paths[0]\nplot_logz_split_distribution(input_csv, output_png)",
        _display_tail(),
    )


def _theory_combined_cell(category_path: str) -> str:
    inputs = [
        "01_theory/01_theory_analytic/summarized_outputs/phi_by_analytic_solution_alpha0p1.csv",
        "01_theory/02_theory_sampling/summarized_outputs/figure_inputs/phi_by_sampling",
    ]
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, inputs),
        (
            "analytic_csv, sampling_root = input_paths\n"
            "analytic = pd.read_csv(analytic_csv).sort_values('r')\n"
            "sampling_files = sorted(sampling_root.glob('N_*.csv'))\n"
            "if not sampling_files:\n"
            "    raise FileNotFoundError(f'no N_*.csv files found under {sampling_root}')\n"
            "sampling = pd.concat((pd.read_csv(path) for path in sampling_files), ignore_index=True, sort=False).sort_values(['N', 'r'])\n\n"
            "def analytic_relative_phi(frame: pd.DataFrame) -> pd.Series:\n"
            "    if 'phi_rel' in frame.columns:\n"
            "        return frame['phi_rel']\n"
            "    ordered = frame.sort_values('r')\n"
            "    return ordered['phi'] - ordered['phi'].iloc[0]\n\n"
            "def sampling_relative_phi(frame: pd.DataFrame) -> pd.Series:\n"
            "    if 'phi_emp_rel' in frame.columns:\n"
            "        return frame['phi_emp_rel']\n"
            "    ordered = frame.sort_values('r')\n"
            "    return ordered['phi_emp'] - ordered['phi_emp'].iloc[0]\n\n"
            "plt.figure(figsize=(8.5, 5.1))\n"
            "plt.plot(analytic['r'], analytic_relative_phi(analytic), color='black', linewidth=2.4, label='analytic full-RS')\n"
            "for n_value, group in sampling.groupby('N', sort=True):\n"
            "    group = group.sort_values('r')\n"
            "    plt.plot(group['r'], sampling_relative_phi(group), marker='o', markersize=3.2, linewidth=1.55, label=f'N={int(n_value)} sampling')\n"
            "plt.xlabel('d')\n"
            "plt.ylabel('phi(d) - phi(d0)')\n"
            "plt.title('Analytic vs shell sampling, alpha=0.1')\n"
            "plt.grid(True, alpha=0.28)\n"
            "plt.legend(fontsize=8)\n"
            "plt.tight_layout()\n"
            "output_png.parent.mkdir(parents=True, exist_ok=True)\n"
            "plt.savefig(output_png, dpi=300)\n"
            "plt.show()\n"
            "plt.close()"
        ),
        _display_tail(),
    )


def _synthetic_sample_cell(category_path: str) -> str:
    input_csv = "02_dnn_synthetic/01_dataset/summarized_outputs/figure_inputs/sample_figures/selected_sample_indices.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        (
            "sample_csv = input_paths[0]\n"
            "sample_frame = pd.read_csv(sample_csv)\n"
            "if sample_frame.empty:\n"
            "    raise ValueError(f'{sample_csv} is empty')\n\n"
            "ncols = 6\n"
            "nrows = int(math.ceil(len(sample_frame) / ncols))\n"
            "fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.45 * nrows), squeeze=False)\n"
            "for ax in axes.ravel():\n"
            "    ax.axis('off')\n\n"
            "for idx, row in sample_frame.iterrows():\n"
            "    image_path = ROOT / str(row['source_image_path'])\n"
            "    if not image_path.exists():\n"
            "        raise FileNotFoundError(image_path)\n"
            "    ax = axes[int(idx) // ncols, int(idx) % ncols]\n"
            "    ax.imshow(mpimg.imread(image_path))\n"
            "    ax.axis('off')\n\n"
            "fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.05)\n"
            "output_png.parent.mkdir(parents=True, exist_ok=True)\n"
            "fig.savefig(output_png, dpi=220, bbox_inches='tight', pad_inches=0.04)\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        _display_tail(),
    )


def _synthetic_spin_cell(category_path: str) -> str:
    input_csv = "02_dnn_synthetic/01_dataset/summarized_outputs/figure_inputs/spin_dynamics/spin_alignment_by_beta.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        (
            "spin_frame = pd.read_csv(input_paths[0])\n"
            "beta = pd.to_numeric(spin_frame['beta_ising'], errors='coerce').to_numpy(dtype=float)\n"
            "mean = pd.to_numeric(spin_frame['mean_edge_alignment'], errors='coerce').to_numpy(dtype=float)\n"
            "sem = pd.to_numeric(spin_frame['sem_edge_alignment'], errors='coerce').to_numpy(dtype=float)\n"
            "mask = np.isfinite(beta) & np.isfinite(mean) & np.isfinite(sem)\n"
            "beta = beta[mask]\n"
            "mean = mean[mask]\n"
            "sem = sem[mask]\n"
            "order = np.argsort(beta)\n\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)\n"
            "ax.plot(beta[order], mean[order], color='#252525', linewidth=2.2, marker='o', markersize=4.8)\n"
            "ax.fill_between(beta[order], mean[order] - sem[order], mean[order] + sem[order], color='#5b8db8', alpha=0.22, linewidth=0.0)\n"
            "ax.set_xlabel('inverse temperature beta (lower T to the right)')\n"
            "ax.set_ylabel('mean edge spin alignment <s_i s_j>')\n"
            "ax.set_title('Spin-dynamics snapshots show temperature-driven ordering')\n"
            "ax.set_ylim(0.0, 0.96)\n"
            "ax.set_xlim(float(beta.min()) - 0.01, float(beta.max()) + 0.01)\n"
            "ax.grid(True, color='#d9d9d9', linewidth=0.8, alpha=0.75)\n"
            "ax.text(0.03, 0.93, '90 final snapshots per beta\\n2000 Kawasaki sweeps per snapshot', transform=ax.transAxes, ha='left', va='top', fontsize=9, bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#bbbbbb', 'alpha': 0.92})\n"
            "top_ax = ax.twiny()\n"
            "top_ax.set_xlim(ax.get_xlim())\n"
            "top_ticks = np.asarray([0.05, 0.10, 0.20, 0.30, 0.39], dtype=float)\n"
            "top_ax.set_xticks(top_ticks)\n"
            "top_ax.set_xticklabels([f'{1.0 / tick:.1f}' for tick in top_ticks])\n"
            "top_ax.set_xlabel('temperature T = 1 / beta')\n"
            "output_png.parent.mkdir(parents=True, exist_ok=True)\n"
            "fig.savefig(output_png, dpi=220, bbox_inches='tight')\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        _display_tail(),
    )


def _synthetic_complexity_cell(category_path: str) -> str:
    input_csv = "02_dnn_synthetic/02_complexity_measure/summarized_outputs/beta_complexity_summary.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        (
            "frame = pd.read_csv(input_paths[0])\n"
            "beta = pd.to_numeric(frame['beta'], errors='coerce').to_numpy(dtype=float)\n"
            "mean = pd.to_numeric(frame['complexity_mean'], errors='coerce').to_numpy(dtype=float)\n"
            "se = pd.to_numeric(frame['complexity_se'], errors='coerce').to_numpy(dtype=float)\n"
            "mask = np.isfinite(beta) & np.isfinite(mean) & np.isfinite(se)\n"
            "beta = beta[mask]\n"
            "mean = mean[mask]\n"
            "se = se[mask]\n"
            "order = np.argsort(beta)\n"
            "r = float(np.corrcoef(beta[order], mean[order])[0, 1]) if len(beta) > 1 else float('nan')\n\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)\n"
            "ax.errorbar(beta[order], mean[order], yerr=se[order], fmt='o-', color='#284f8f', ecolor='#8aa7d6', capsize=3)\n"
            "ax.set_xlabel(r'$\\beta$')\n"
            "ax.set_ylabel('3-NN label-disagreement complexity')\n"
            "ax.set_title(f'Beta vs complexity (Pearson r={r:.3f})')\n"
            "ax.grid(True, alpha=0.25)\n"
            "output_png.parent.mkdir(parents=True, exist_ok=True)\n"
            "fig.savefig(output_png, dpi=240)\n"
            "plt.show()\n"
            "plt.close(fig)"
        ),
        _display_tail(),
    )


def _synthetic_logz_cell(category_path: str) -> str:
    stem = Path(category_path).stem
    input_csv = f"02_dnn_synthetic/04_sampling/summarized_outputs/figure_inputs/logZ_split/{stem}.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        _source_blocks(
            "02_dnn_synthetic/figures/src/make_figures.py",
            ("radius_label", "xtick_labels", "plot_logz_split_distribution"),
        ),
        "plot_logz_split_distribution(input_paths[0], output_png, max_scatter_per_radius=120)",
        _display_tail(),
    )


SYNTHETIC_CURVE_SPECS = {
    "phi_d_curve.png": ("phi_d_curve", "phi_full_mean", "phi_full_sem", r"$\phi(d)$", r"Synthetic $\phi(d)$ by distance"),
    "phi_energetic_d_curve.png": (
        "phi_energetic_d_curve",
        "phi_energy_mean",
        "phi_energy_sem",
        r"energetic $\phi(d)$",
        r"Synthetic energetic $\phi(d)$ by distance",
    ),
    "derivative_phi_d_curve.png": (
        "derivative_phi_d_curve",
        "dphi_full_dr_mean",
        "dphi_full_dr_sem",
        r"$d\phi/dd$",
        r"Synthetic derivative of $\phi(d)$",
    ),
    "derivative_phi_energetic_d_curve.png": (
        "derivative_phi_energetic_d_curve",
        "dphi_energy_dr_mean",
        "dphi_energy_dr_sem",
        r"energetic $d\phi/dd$",
        r"Synthetic energetic derivative of $\phi(d)$",
    ),
}


def _synthetic_curve_cell(category_path: str) -> str:
    name, value_key, sem_key, ylabel, title = SYNTHETIC_CURVE_SPECS[Path(category_path).name]
    input_csv = f"02_dnn_synthetic/05_proxy_local_entropy/summarized_outputs/figure_inputs/{name}/{name}.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        _source_blocks("02_dnn_synthetic/figures/src/make_figures.py", ("group_curves", "plot_curve_frame")),
        (
            f"value_key = {value_key!r}\n"
            f"sem_key = {sem_key!r}\n"
            f"ylabel = {ylabel!r}\n"
            f"title = {title!r}\n"
            "frame = pd.read_csv(input_paths[0])\n"
            "plot_curve_frame(frame, value_key, sem_key, ylabel, title, output_png)"
        ),
        _display_tail(),
    )


def _synthetic_phase_cell(category_path: str) -> str:
    stem = Path(category_path).stem
    if stem == "phase_like_A_by_beta":
        x_key, x_label, title = "beta", r"$\beta$", "A measure by beta"
    else:
        x_key, x_label, title = "complexity_mean", "3-NN complexity", "A measure by complexity"
    inputs = [
        f"02_dnn_synthetic/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv",
        f"02_dnn_synthetic/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/phase_derivative_curves.csv",
    ]
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, inputs),
        _source_blocks("02_dnn_synthetic/figures/src/make_figures.py", ("group_curves", "plot_phase_panel")),
        (
            f"x_key = {x_key!r}\n"
            f"x_label = {x_label!r}\n"
            f"title = {title!r}\n"
            "phase_frame = pd.read_csv(input_paths[0])\n"
            "derivative_frame = pd.read_csv(input_paths[1])\n"
            "plot_phase_panel(phase_frame, derivative_frame, x_key, x_label, title, output_png)"
        ),
        _display_tail(),
    )


def _mnist_dataset_cell(category_path: str, series: str) -> str:
    script = f"03_dnn_mnist/{series}/01_dataset/src/utils/figure_builders.py"
    input_csv = f"03_dnn_mnist/{series}/01_dataset/summarized_outputs/figure_inputs/sample_figure/selected_sample_indices.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        (
            f"DNN_ROOT = ROOT / '03_dnn_mnist'\n"
            "FIGURE_ROOT = output_png.parent\n"
            f"SAMPLE_FIGURE_INPUT_ROOT = ROOT / '03_dnn_mnist/{series}/01_dataset/summarized_outputs/figure_inputs/sample_figure'\n"
            "SAMPLE_FIGURE_PATH = output_png\n\n"
            "def repo_path(relative_path: str) -> Path:\n"
            "    path = Path(relative_path)\n"
            "    if path.is_absolute():\n"
            "        return path\n"
            "    candidates = [DNN_ROOT / path]\n"
            "    text = path.as_posix()\n"
            "    marker = '03_dnn_mnist/'\n"
            "    if marker in text:\n"
            "        candidates.append(DNN_ROOT / text.split(marker, 1)[1])\n"
            "    for candidate in candidates:\n"
            "        if candidate.exists():\n"
            "            return candidate\n"
            "    return candidates[0]\n\n"
            "def load_csv_rows(path: Path) -> list[dict[str, str]]:\n"
            "    if not path.exists():\n"
            "        raise FileNotFoundError(path)\n"
            "    with path.open('r', encoding='utf-8', newline='') as handle:\n"
            "        return list(csv.DictReader(handle))"
        ),
        _source_blocks(script, ("_resolve_source", "build_sample_figure")),
        "build_sample_figure()",
        _display_tail(),
    )


def _mnist_complexity_cell(category_path: str, series: str) -> str:
    if series == "label_noise_sweep":
        script = "03_dnn_mnist/label_noise_sweep/02_complexity_measure/src/make_figures.py"
        input_csv = "03_dnn_mnist/label_noise_sweep/02_complexity_measure/summarized_outputs/eta_complexity_summary.csv"
    else:
        script = "03_dnn_mnist/manual_rules/02_complexity_measure/src/make_figures.py"
        input_csv = "03_dnn_mnist/manual_rules/02_complexity_measure/summarized_outputs/manual_rule_complexity_summary.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        "SUMMARY_PATH = input_paths[0]\nFIGURE_PATH = output_png",
        _source_blocks(script, ("_read_summary", "_plot")),
        "summary_rows = _read_summary(SUMMARY_PATH)\n_plot(summary_rows)",
        _display_tail(),
    )


def _mnist_reference_cell(category_path: str, series: str) -> str:
    if series == "label_noise_sweep":
        script = "03_dnn_mnist/label_noise_sweep/03_reference_search/src/make_figures.py"
        inputs = [
            "03_dnn_mnist/label_noise_sweep/03_reference_search/summarized_outputs/figure_inputs/reference_quality/reference_quality_by_eta.csv",
            "03_dnn_mnist/label_noise_sweep/03_reference_search/summarized_outputs/figure_inputs/reference_quality/reference_quality_by_ref.csv",
        ]
        functions = ("_read_inputs", "build_figures")
        globals_code = (
            "FIGURE_SUMMARY_PATH = input_paths[0]\n"
            "FIGURE_PER_REF_PATH = input_paths[1]\n"
            "FIGURE_ROOT = output_png.parent\n"
            "FIGURE_PATH = output_png"
        )
    else:
        script = "03_dnn_mnist/manual_rules/03_reference_search/src/make_figures.py"
        inputs = [
            "03_dnn_mnist/manual_rules/03_reference_search/summarized_outputs/figure_inputs/reference_quality/reference_quality_by_rule.csv",
            "03_dnn_mnist/manual_rules/03_reference_search/summarized_outputs/figure_inputs/reference_quality/reference_quality_by_ref.csv",
        ]
        functions = ("_rule_label", "_read_inputs", "build_figures")
        globals_code = (
            "SUMMARY_PATH = input_paths[0]\n"
            "PER_REF_PATH = input_paths[1]\n"
            "FIGURE_ROOT = output_png.parent\n"
            "FIGURE_PATH = output_png"
        )
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, inputs),
        globals_code,
        _source_blocks(script, functions),
        "build_figures()",
        _display_tail(),
    )


def _mnist_sampling_cell(category_path: str, series: str) -> str:
    stem = Path(category_path).stem
    if series == "label_noise_sweep":
        script = "03_dnn_mnist/label_noise_sweep/04_sampling/src/make_figures.py"
        input_csv = f"03_dnn_mnist/label_noise_sweep/04_sampling/summarized_outputs/figure_inputs/logZ_split/{stem}.csv"
    else:
        script = "03_dnn_mnist/manual_rules/04_sampling/src/make_figures.py"
        input_csv = f"03_dnn_mnist/manual_rules/04_sampling/summarized_outputs/figure_inputs/logZ_split/{stem}.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        _source_blocks(script, ("_radius_label", "_xtick_labels", "plot_logz_split_distribution")),
        "plot_logz_split_distribution(input_paths[0], output_png, max_scatter_per_radius=120)",
        _display_tail(),
    )


LABEL_PLE_CURVE_SPECS = {
    "phi_d_curve.png": ("phi_d_curve", "delta_phi_energy_mean", "delta_phi_energy_sem", r"$\phi(d)-\phi(d_0)$", r"MNIST label-noise $\phi(d)$"),
    "phi_energetic_d_curve.png": (
        "phi_energetic_d_curve",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        r"energetic $\phi(d)$",
        r"MNIST label-noise energetic $\phi(d)$",
    ),
    "derivative_phi_d_curve.png": (
        "derivative_phi_d_curve",
        "d_delta_phi_energy_dd",
        "d_delta_phi_energy_dd_sem",
        r"$d\phi/dd$",
        r"MNIST label-noise derivative of $\phi(d)$",
    ),
    "derivative_phi_energetic_d_curve.png": (
        "derivative_phi_energetic_d_curve",
        "d_phi_energy_direct_dd",
        "d_phi_energy_direct_dd_sem",
        r"energetic $d\phi/dd$",
        r"MNIST label-noise direct energetic derivative",
    ),
}


def _mnist_label_ple_curve_cell(category_path: str) -> str:
    name, value_key, sem_key, ylabel, title = LABEL_PLE_CURVE_SPECS[Path(category_path).name]
    input_csv = f"03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/{name}/{name}.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        (
            "FIGURE_INPUT_ROOT = ROOT / '03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs'\n"
            "FIGURE_ROOT = output_png.parent\n"
            "COMPLEXITY_SUMMARY_PATH = ROOT / '03_dnn_mnist/label_noise_sweep/02_complexity_measure/summarized_outputs/eta_complexity_summary.csv'"
        ),
        _source_blocks(
            "03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/src/make_figures.py",
            ("_read_csv", "_float", "_group_curves", "_plot_curve"),
        ),
        (
            f"value_key = {value_key!r}\n"
            f"sem_key = {sem_key!r}\n"
            f"ylabel = {ylabel!r}\n"
            f"title = {title!r}\n"
            "rows = _read_csv(input_paths[0])\n"
            "_plot_curve(rows, value_key, sem_key, ylabel, title, output_png)"
        ),
        _display_tail(),
    )


def _mnist_label_ple_phase_cell(category_path: str) -> str:
    stem = Path(category_path).stem
    if stem == "phase_like_A_by_eta":
        x_key, x_label, title = "eta", r"$\eta$", "A measure by eta"
    else:
        x_key, x_label, title = "nmstv", "3-NN MNIST complexity", "A measure by complexity"
    inputs = [
        f"03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv",
        f"03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/phase_derivative_curves.csv",
    ]
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, inputs),
        (
            "FIGURE_INPUT_ROOT = ROOT / '03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs'\n"
            "FIGURE_ROOT = output_png.parent\n"
            "COMPLEXITY_SUMMARY_PATH = ROOT / '03_dnn_mnist/label_noise_sweep/02_complexity_measure/summarized_outputs/eta_complexity_summary.csv'"
        ),
        _source_blocks(
            "03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/src/make_figures.py",
            ("_read_csv", "_float", "_eta_complexity_lookup", "_group_curves", "_plot_phase"),
        ),
        f"_plot_phase({stem!r}, {x_key!r}, {x_label!r}, {title!r}, output_png.name)",
        _display_tail(),
    )


MANUAL_PLE_CURVE_SPECS = {
    "phi_d_curve.png": (
        "phi_d_curve",
        "delta_phi_energy_unit_mean",
        "delta_phi_energy_unit_sem",
        "phi(d) - phi(d0)",
        "MNIST manual-rule phi(d)",
    ),
    "phi_energetic_d_curve.png": (
        "phi_d_curve",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        "energetic phi(d)",
        "MNIST manual-rule energetic phi(d)",
    ),
    "derivative_phi_d_curve.png": (
        "derivative_phi_d_curve",
        "d_delta_phi_energy_direct_dd_unit_mean",
        "d_delta_phi_energy_direct_dd_unit_sem",
        "d phi / dd",
        "MNIST manual-rule direct derivative of phi(d)",
    ),
    "derivative_phi_energetic_d_curve.png": (
        "derivative_phi_d_curve",
        "d_phi_energy_direct_dd_unit_mean",
        "d_phi_energy_direct_dd_unit_sem",
        "energetic d phi / dd",
        "MNIST manual-rule direct energetic derivative",
    ),
}


def _manual_ple_globals() -> str:
    return (
        "FIGURE_INPUT_ROOT = ROOT / '03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs'\n"
        "FIGURE_ROOT = output_png.parent\n"
        "COLORS = {\n"
        "    'rule_001': '#0072B2',\n"
        "    'rule_002': '#009E73',\n"
        "    'rule_003': '#D55E00',\n"
        "    'rule_004': '#CC79A7',\n"
        "}"
    )


def _mnist_manual_ple_curve_cell(category_path: str) -> str:
    name, value_key, sem_key, ylabel, title = MANUAL_PLE_CURVE_SPECS[Path(category_path).name]
    input_csv = f"03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/{name}/{name}.csv"
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, [input_csv]),
        _manual_ple_globals(),
        _source_blocks("03_dnn_mnist/manual_rules/05_proxy_local_entropy/src/make_figures.py", ("_plot_curves",)),
        (
            f"value_key = {value_key!r}\n"
            f"sem_key = {sem_key!r}\n"
            f"ylabel = {ylabel!r}\n"
            f"title = {title!r}\n"
            "frame = pd.read_csv(input_paths[0])\n"
            "_plot_curves(frame, value_key, sem_key, ylabel, title, output_png)"
        ),
        _display_tail(),
    )


def _mnist_manual_ple_phase_cell(category_path: str) -> str:
    stem = Path(category_path).stem
    if stem == "phase_like_A_by_rule":
        x_key, x_label = "rule_order", "manual-rule order"
    else:
        x_key, x_label = "nmstv_mean", "3-NN MNIST complexity"
    inputs = [
        f"03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/{stem}.csv",
        f"03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/{stem}/phase_derivative_curves.csv",
    ]
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, inputs),
        _manual_ple_globals(),
        _source_blocks("03_dnn_mnist/manual_rules/05_proxy_local_entropy/src/make_figures.py", ("_plot_phase",)),
        f"_plot_phase({x_key!r}, {x_label!r}, {stem!r})",
        _display_tail(),
    )


MERGED_CURVE_SPECS = {
    "phi_d_curve.png": (
        "phi_d_curve",
        "delta_phi_energy_mean",
        "delta_phi_energy_sem",
        "delta_phi_energy_unit_mean",
        "delta_phi_energy_unit_sem",
        r"$\phi(d)-\phi(d_0)$",
        "Merged MNIST phi(d): endpoints + eta sweep",
    ),
    "phi_energetic_d_curve.png": (
        "phi_energetic_d_curve",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        "phi_energy_raw_mean",
        "phi_energy_raw_sem",
        r"energetic $\phi(d)$",
        "Merged MNIST energetic phi(d): endpoints + eta sweep",
    ),
    "derivative_phi_d_curve.png": (
        "derivative_phi_d_curve",
        "d_delta_phi_energy_dd",
        "d_delta_phi_energy_dd_sem",
        "d_delta_phi_energy_direct_dd_unit_mean",
        "d_delta_phi_energy_direct_dd_unit_sem",
        r"$d\phi/dd$",
        "Merged MNIST derivative of phi(d): endpoints + eta sweep",
    ),
    "derivative_phi_energetic_d_curve.png": (
        "derivative_phi_energetic_d_curve",
        "d_phi_energy_direct_dd",
        "d_phi_energy_direct_dd_sem",
        "d_phi_energy_direct_dd_unit_mean",
        "d_phi_energy_direct_dd_unit_sem",
        r"energetic $d\phi/dd$",
        "Merged MNIST energetic derivative: endpoints + eta sweep",
    ),
}


def _merged_globals() -> str:
    return (
        "DNN_ROOT = ROOT / '03_dnn_mnist'\n"
        "PLE_INPUT_ROOTS = {\n"
        "    'label_noise_sweep': DNN_ROOT / 'label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs',\n"
        "    'manual_rules': DNN_ROOT / 'manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs',\n"
        "}\n"
        "ETA_COMPLEXITY_PATH = DNN_ROOT / 'label_noise_sweep/02_complexity_measure/summarized_outputs/eta_complexity_summary.csv'\n"
        "MANUAL_COMPLEXITY_PATH = DNN_ROOT / 'manual_rules/02_complexity_measure/summarized_outputs/manual_rule_complexity_summary.csv'\n"
        "ENDPOINT_ETA = {'real_even_odd': 0.0, 'random_label': 0.5}\n"
        "ENDPOINT_LABEL = {'real_even_odd': 'even_odd (eta 0.00)', 'random_label': 'random (eta 0.50)'}"
    )


def _mnist_merged_curve_cell(category_path: str) -> str:
    spec = MERGED_CURVE_SPECS[Path(category_path).name]
    name, label_value, label_sem, manual_value, manual_sem, ylabel, title = spec
    inputs = [
        f"03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/summarized_outputs/figure_inputs/{name}/{name}.csv",
        f"03_dnn_mnist/manual_rules/05_proxy_local_entropy/summarized_outputs/figure_inputs/{name}/{name}.csv",
    ]
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, inputs),
        _merged_globals(),
        _source_blocks(
            "03_dnn_mnist/figures/src/make_figures.py",
            (
                "require_file",
                "finite_or_zero",
                "label_noise_curve_frame",
                "manual_endpoint_curve_frame",
                "merged_curve_frame",
                "plot_merged_curves",
            ),
        ),
        (
            f"name = {name!r}\n"
            f"label_value_col = {label_value!r}\n"
            f"label_sem_col = {label_sem!r}\n"
            f"manual_value_col = {manual_value!r}\n"
            f"manual_sem_col = {manual_sem!r}\n"
            f"ylabel = {ylabel!r}\n"
            f"title = {title!r}\n"
            "frame, source_inputs = merged_curve_frame(name, label_value_col, label_sem_col, manual_value_col, manual_sem_col)\n"
            "plot_merged_curves(frame, ylabel, title, output_png)"
        ),
        _display_tail(),
    )


def _mnist_merged_phase_cell(category_path: str) -> str:
    stem = Path(category_path).stem
    if stem == "phase_like_A_by_eta":
        label_input_name = "phase_like_A_by_eta"
        manual_input_name = "phase_like_A_by_rule"
        x_col, x_label, right_title, eta_xlim = "eta", r"aligned $\eta$", "A_kappa by aligned eta", True
    else:
        label_input_name = "phase_like_A_by_complexity"
        manual_input_name = "phase_like_A_by_complexity"
        x_col, x_label, right_title, eta_xlim = "complexity", "3-NN MNIST complexity", "A_kappa by complexity", False
    inputs = _figure_input_paths(category_path)
    return _join_cell(
        _cell_preamble(),
        _path_assignments(category_path, inputs),
        _merged_globals(),
        _source_blocks(
            "03_dnn_mnist/figures/src/make_figures.py",
            (
                "require_file",
                "finite_or_zero",
                "eta_complexity_map",
                "manual_complexity_map",
                "map_eta_complexity",
                "map_manual_complexity",
                "label_phase_frames",
                "manual_endpoint_phase_frames",
                "plot_merged_phase",
            ),
        ),
        (
            f"label_input_name = {label_input_name!r}\n"
            f"manual_input_name = {manual_input_name!r}\n"
            f"x_col = {x_col!r}\n"
            f"x_label = {x_label!r}\n"
            f"right_title = {right_title!r}\n"
            f"eta_xlim = {eta_xlim!r}\n"
            "label_phase, label_curves, label_inputs = label_phase_frames(label_input_name)\n"
            "manual_phase, manual_curves, manual_inputs = manual_endpoint_phase_frames(manual_input_name)\n"
            "phase = pd.concat([manual_phase, label_phase], ignore_index=True).sort_values(x_col)\n"
            "curves = pd.concat([manual_curves, label_curves], ignore_index=True).sort_values(['eta', 'radius'])\n"
            "plot_merged_phase(phase, curves, output_png, x_col=x_col, x_label=x_label, right_title=right_title, eta_xlim=eta_xlim)"
        ),
        _display_tail(),
    )


def _figure_code(row: dict[str, str]) -> str:
    category_path = row["category_path"]
    path = Path(category_path)
    parts = path.parts

    if category_path == "theory/01_theory_analytic/phi_by_analytic_solution_alpha0p1.png":
        return _theory_analytic_cell(category_path)
    if category_path == "theory/02_theory_sampling/phi_by_sampling/phi_by_sampling.png":
        return _theory_sampling_phi_cell(category_path)
    if category_path.startswith("theory/02_theory_sampling/logZ_split_distributions/"):
        return _theory_logz_cell(category_path)
    if category_path.startswith("theory/summary/"):
        return _theory_combined_cell(category_path)

    if category_path == "dnn_synthetic/01_dataset/sample_figure.png":
        return _synthetic_sample_cell(category_path)
    if category_path == "dnn_synthetic/01_dataset/spin_dynamics_phase_transition.png":
        return _synthetic_spin_cell(category_path)
    if category_path == "dnn_synthetic/02_complexity_measure/beta_complexity_figure.png":
        return _synthetic_complexity_cell(category_path)
    if category_path.startswith("dnn_synthetic/04_sampling/logZ_split_distributions/"):
        return _synthetic_logz_cell(category_path)
    if category_path.startswith("dnn_synthetic/05_proxy_local_entropy/phase_like_A_by_"):
        return _synthetic_phase_cell(category_path)
    if category_path.startswith("dnn_synthetic/05_proxy_local_entropy/"):
        return _synthetic_curve_cell(category_path)

    if category_path.startswith("dnn_mnist/01_dataset/"):
        return _mnist_dataset_cell(category_path, parts[2])
    if category_path.startswith("dnn_mnist/02_complexity_measure/"):
        return _mnist_complexity_cell(category_path, parts[2])
    if category_path.startswith("dnn_mnist/03_reference_search/"):
        return _mnist_reference_cell(category_path, parts[2])
    if category_path.startswith("dnn_mnist/04_sampling/"):
        return _mnist_sampling_cell(category_path, parts[2])
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/label_noise_sweep/phase_like_A_by_"):
        return _mnist_label_ple_phase_cell(category_path)
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/label_noise_sweep/"):
        return _mnist_label_ple_curve_cell(category_path)
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/manual_rules/phase_like_A_by_"):
        return _mnist_manual_ple_phase_cell(category_path)
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/manual_rules/"):
        return _mnist_manual_ple_curve_cell(category_path)
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/merged/phase_like_A_by_"):
        return _mnist_merged_phase_cell(category_path)
    if category_path.startswith("dnn_mnist/05_proxy_local_entropy/merged/"):
        return _mnist_merged_curve_cell(category_path)

    raise NotImplementedError(category_path)


def _write_notebook(rows: list[dict[str, str]]) -> None:
    cells: list[dict[str, object]] = [
        _notebook_markdown_cell(
            "# Rebuild All Figures One PNG At A Time\n\n"
            "Generated by `Figures/rebuild_figures.py`. Each markdown/code pair below corresponds to exactly one grouped PNG in `Figures/manifest.csv`. The code cell is self-contained for that PNG: it loads the relevant summarized CSVs or image inputs, defines the copied plotting block from the corresponding `make_figures.py` path when needed, saves directly into `Figures/`, and displays the result inline."
        )
    ]

    for idx, row in enumerate(rows, start=1):
        inputs = _figure_input_paths(row["category_path"]) or _split_notebook_inputs(row.get("input_paths"))
        input_lines = "\n".join(f"- `{path}`" for path in inputs) if inputs else "- inferred from figure name"
        heading = (
            f"## {idx:02d}. {row['category_path']}\n\n"
            f"Source image: `{row['source_path']}`\n\n"
            f"Inputs:\n{input_lines}"
        )
        cells.append(_notebook_markdown_cell(heading))
        cells.append(_notebook_code_cell(_figure_code(row)))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def rebuild_figures(*, skip_stage_build: bool = False, write_notebook: bool = True) -> list[Path]:
    if not skip_stage_build:
        for script in STAGE_BUILD_SCRIPTS:
            _run_script(script)

    _clear_figures_dir()
    rows: list[dict[str, str]] = []
    outputs: list[Path] = []
    for spec in GALLERIES:
        outputs.extend(_copy_gallery(spec, rows))

    _write_manifest(rows)
    _write_readme(rows)
    if write_notebook:
        _write_notebook(rows)
    written = [MANIFEST_PATH, README_PATH]
    if write_notebook:
        written.append(NOTEBOOK_PATH)
    return outputs + written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the central Complexity/Figures bundle from current stage-level galleries."
    )
    parser.add_argument("--skip-stage-build", action="store_true", help="Only recopy existing stage figure outputs.")
    parser.add_argument("--skip-notebook", action="store_true", help="Do not rewrite the generated notebook.")
    args = parser.parse_args()

    for output in rebuild_figures(
        skip_stage_build=bool(args.skip_stage_build),
        write_notebook=not bool(args.skip_notebook),
    ):
        print(output)


if __name__ == "__main__":
    main()
