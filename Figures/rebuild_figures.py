from __future__ import annotations

import argparse
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

MAKE_FIGURE_CELLS = (
    {
        "title": "01 theory analytic",
        "script": "01_theory/01_theory_analytic/src/make_figures.py",
        "outputs": ["01_theory/01_theory_analytic/figures"],
    },
    {
        "title": "01 theory sampling",
        "script": "01_theory/02_theory_sampling/src/make_figures.py",
        "outputs": ["01_theory/02_theory_sampling/figures"],
    },
    {
        "title": "01 theory combined",
        "script": "01_theory/figures/src/make_figures.py",
        "outputs": ["01_theory/figures"],
    },
    {
        "title": "02 dnn synthetic dataset",
        "script": "02_dnn_synthetic/01_dataset/src/make_figures.py",
        "outputs": ["02_dnn_synthetic/01_dataset/figures"],
    },
    {
        "title": "02 dnn synthetic complexity",
        "script": "02_dnn_synthetic/02_complexity_measure/src/make_figures.py",
        "outputs": ["02_dnn_synthetic/02_complexity_measure/figures"],
    },
    {
        "title": "02 dnn synthetic sampling",
        "script": "02_dnn_synthetic/04_sampling/src/make_figures.py",
        "outputs": ["02_dnn_synthetic/04_sampling/figures"],
    },
    {
        "title": "02 dnn synthetic proxy local entropy",
        "script": "02_dnn_synthetic/05_proxy_local_entropy/src/make_figures.py",
        "outputs": ["02_dnn_synthetic/05_proxy_local_entropy/figures"],
    },
    {
        "title": "02 dnn synthetic top-level gallery",
        "script": "02_dnn_synthetic/figures/src/make_figures.py",
        "outputs": ["02_dnn_synthetic/figures"],
    },
    {
        "title": "03 dnn mnist label-noise dataset",
        "script": "03_dnn_mnist/label_noise_sweep/01_dataset/src/make_figures.py",
        "outputs": ["03_dnn_mnist/label_noise_sweep/01_dataset/figures"],
    },
    {
        "title": "03 dnn mnist label-noise complexity",
        "script": "03_dnn_mnist/label_noise_sweep/02_complexity_measure/src/make_figures.py",
        "outputs": ["03_dnn_mnist/label_noise_sweep/02_complexity_measure/figures"],
    },
    {
        "title": "03 dnn mnist label-noise reference search",
        "script": "03_dnn_mnist/label_noise_sweep/03_reference_search/src/make_figures.py",
        "outputs": ["03_dnn_mnist/label_noise_sweep/03_reference_search/figures"],
    },
    {
        "title": "03 dnn mnist label-noise sampling",
        "script": "03_dnn_mnist/label_noise_sweep/04_sampling/src/make_figures.py",
        "outputs": ["03_dnn_mnist/label_noise_sweep/04_sampling/figures"],
    },
    {
        "title": "03 dnn mnist label-noise proxy local entropy",
        "script": "03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/src/make_figures.py",
        "outputs": ["03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/figures"],
    },
    {
        "title": "03 dnn mnist manual-rules dataset",
        "script": "03_dnn_mnist/manual_rules/01_dataset/src/make_figures.py",
        "outputs": ["03_dnn_mnist/manual_rules/01_dataset/figures"],
    },
    {
        "title": "03 dnn mnist manual-rules complexity",
        "script": "03_dnn_mnist/manual_rules/02_complexity_measure/src/make_figures.py",
        "outputs": ["03_dnn_mnist/manual_rules/02_complexity_measure/figures"],
    },
    {
        "title": "03 dnn mnist manual-rules reference search",
        "script": "03_dnn_mnist/manual_rules/03_reference_search/src/make_figures.py",
        "outputs": ["03_dnn_mnist/manual_rules/03_reference_search/figures"],
    },
    {
        "title": "03 dnn mnist manual-rules sampling",
        "script": "03_dnn_mnist/manual_rules/04_sampling/src/make_figures.py",
        "outputs": ["03_dnn_mnist/manual_rules/04_sampling/figures"],
    },
    {
        "title": "03 dnn mnist manual-rules proxy local entropy",
        "script": "03_dnn_mnist/manual_rules/05_proxy_local_entropy/src/make_figures.py",
        "outputs": ["03_dnn_mnist/manual_rules/05_proxy_local_entropy/figures"],
    },
    {
        "title": "03 dnn mnist top-level gallery",
        "script": "03_dnn_mnist/figures/src/make_figures.py",
        "outputs": ["03_dnn_mnist/figures"],
    },
)


def _relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _title_from_path(path: Path) -> str:
    title = path.stem.replace("_", " ").replace("-", " ")
    return " ".join(title.split())


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
        manifest_rows.append(
            {
                "group": spec.group,
                "section": spec.target_prefix,
                "title": meta.get("title") or _title_from_path(source),
                "figure": target.name,
                "category_path": target.relative_to(FIGURES_DIR).as_posix(),
                "source_path": _relative_to_project(source),
                "input_paths": meta.get("input_paths", ""),
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
            "`rebuild_all_figures_from_summaries.ipynb` is an executable one-PNG-per-cell notebook. Each figure cell refreshes one source PNG, copies that single image into `Figures/`, prints the source/output paths, and displays that one image inline.",
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


def _write_notebook(rows: list[dict[str, str]]) -> None:
    cells: list[dict[str, object]] = [
        _notebook_markdown_cell(
            "# Rebuild All Figures One PNG At A Time\n\n"
            "Generated by `Figures/rebuild_figures.py`. Each code cell below corresponds to exactly one grouped PNG in `Figures/manifest.csv`: it refreshes that figure's source artifact, copies that one PNG into `Figures/`, and displays that one image inline."
        ),
        _notebook_code_cell(
            "import importlib.util\n"
            "import math\n"
            "import shutil\n"
            "import sys\n"
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "FIGURES_DIR = Path.cwd()\n"
            "if FIGURES_DIR.name != 'Figures':\n"
            f"    FIGURES_DIR = Path({str(FIGURES_DIR)!r})\n"
            "ROOT = FIGURES_DIR.parent\n"
            "IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}\n"
            "MANIFEST = pd.read_csv(FIGURES_DIR / 'manifest.csv')\n\n"
            "plt.rcParams.update({\n"
            "    'figure.dpi': 120,\n"
            "    'savefig.dpi': 220,\n"
            "    'axes.grid': True,\n"
            "    'grid.alpha': 0.25,\n"
            "    'axes.spines.top': False,\n"
            "    'axes.spines.right': False,\n"
            "})\n\n"
            "def _clear_local_import_cache():\n"
            "    for name in list(sys.modules):\n"
            "        if name == 'utils' or name.startswith('utils.') or name == 'make_summarized_outputs':\n"
            "            sys.modules.pop(name, None)\n\n"
            "def _load_script(script_rel):\n"
            "    script = ROOT / script_rel\n"
            "    if not script.exists():\n"
            "        raise FileNotFoundError(script)\n"
            "    module_name = '_single_fig_' + ''.join(ch if ch.isalnum() else '_' for ch in script_rel)\n"
            "    spec = importlib.util.spec_from_file_location(module_name, script)\n"
            "    module = importlib.util.module_from_spec(spec)\n"
            "    old_sys_path = sys.path[:]\n"
            "    sys.path.insert(0, str(script.parent))\n"
            "    _clear_local_import_cache()\n"
            "    try:\n"
            "        assert spec.loader is not None\n"
            "        sys.modules[module_name] = module\n"
            "        spec.loader.exec_module(module)\n"
            "    finally:\n"
            "        _clear_local_import_cache()\n"
            "        sys.path[:] = old_sys_path\n"
            "    return module\n\n"
            "def _manifest_row(category_path):\n"
            "    rows = MANIFEST.loc[MANIFEST['category_path'].eq(category_path)]\n"
            "    if rows.empty:\n"
            "        raise KeyError(category_path)\n"
            "    return rows.iloc[0]\n\n"
            "def _split_inputs(value):\n"
            "    if value is None or (isinstance(value, float) and math.isnan(value)):\n"
            "        return []\n"
            "    text = str(value)\n"
            "    if not text or text.lower() == 'nan':\n"
            "        return []\n"
            "    return [item for item in text.split(';') if item]\n\n"
            "def _copy_display(source, target):\n"
            "    if not source.exists():\n"
            "        raise FileNotFoundError(source)\n"
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.copy2(source, target)\n"
            "    print(f'source: {source.relative_to(ROOT).as_posix()}')\n"
            "    print(f'output: {target.relative_to(ROOT).as_posix()}')\n"
            "    display(Image(filename=str(target)))\n"
            "    return target\n\n"
            "def _generate_theory_source(source):\n"
            "    rel = source.relative_to(ROOT).as_posix()\n"
            "    if rel.startswith('01_theory/01_theory_analytic/figures/'):\n"
            "        mod = _load_script('01_theory/01_theory_analytic/src/make_figures.py')\n"
            "        mod.make_figure(mod.DEFAULT_INPUT_CSV, source)\n"
            "        return\n"
            "    if rel.endswith('01_theory/figures/fig01_sampling_vs_analytic_phi_by_distance_alpha0p1.png'):\n"
            "        mod = _load_script('01_theory/figures/src/make_figures.py')\n"
            "        mod.write_combined_figure(mod.read_required_csv(mod.DEFAULT_ANALYTIC_CSV), mod.read_sampling_input(mod.DEFAULT_SAMPLING_INPUT), source)\n"
            "        return\n"
            "    if rel.endswith('01_theory/02_theory_sampling/figures/phi_by_sampling/phi_by_sampling.png'):\n"
            "        mod = _load_script('01_theory/02_theory_sampling/src/make_figures.py')\n"
            "        mod.make_phi_figure(mod.DEFAULT_PHI_INPUT_ROOT, source)\n"
            "        return\n"
            "    if '01_theory/02_theory_sampling/figures/logZ_split_distributions/' in rel:\n"
            "        mod = _load_script('01_theory/02_theory_sampling/src/make_figures.py')\n"
            "        mod.plot_logz_split_distribution(mod.DEFAULT_LOGZ_INPUT_ROOT / f'{source.stem}.csv', source)\n"
            "        return\n"
            "    raise NotImplementedError(rel)\n\n"
            "def _generate_synthetic_source(source):\n"
            "    rel = source.relative_to(ROOT / '02_dnn_synthetic' / 'figures').as_posix()\n"
            "    mod = _load_script('02_dnn_synthetic/figures/src/make_figures.py')\n"
            "    if rel in {'01_dataset/sample_figure.png', '01_dataset/spin_dynamics_phase_transition.png'}:\n"
            "        mod.build_dataset_figures()\n"
            "        return\n"
            "    if rel == '02_complexity_measure/beta_complexity_figure.png':\n"
            "        mod.build_complexity_figure()\n"
            "        return\n"
            "    if rel.startswith('04_sampling/logZ_split_distributions/'):\n"
            "        input_csv = mod.SAMPLING_INPUT_ROOT / f'{source.stem}.csv'\n"
            "        mod.plot_logz_split_distribution(input_csv, source, max_scatter_per_radius=120)\n"
            "        return\n"
            "    if rel.startswith('05_proxy_local_entropy/'):\n"
            "        for name, value_key, sem_key, ylabel, title, output_name in mod.CURVE_FIGURES:\n"
            "            if source.name == output_name:\n"
            "                input_csv = mod.PLE_INPUT_ROOT / name / f'{name}.csv'\n"
            "                mod.plot_curve_frame(pd.read_csv(input_csv), value_key, sem_key, ylabel, title, source)\n"
            "                return\n"
            "        if source.name == 'phase_like_A_by_beta.png':\n"
            "            phase_csv = mod.PLE_INPUT_ROOT / 'phase_like_A_by_beta' / 'phase_like_A_by_beta.csv'\n"
            "            curve_csv = mod.PLE_INPUT_ROOT / 'phase_like_A_by_beta' / 'phase_derivative_curves.csv'\n"
            "            mod.plot_phase_panel(pd.read_csv(phase_csv), pd.read_csv(curve_csv), 'beta', r'$\\beta$', 'A measure by beta', source)\n"
            "            return\n"
            "        if source.name == 'phase_like_A_by_complexity.png':\n"
            "            phase_csv = mod.PLE_INPUT_ROOT / 'phase_like_A_by_complexity' / 'phase_like_A_by_complexity.csv'\n"
            "            curve_csv = mod.PLE_INPUT_ROOT / 'phase_like_A_by_complexity' / 'phase_derivative_curves.csv'\n"
            "            mod.plot_phase_panel(pd.read_csv(phase_csv), pd.read_csv(curve_csv), 'complexity_mean', '3-NN complexity', 'A measure by complexity', source)\n"
            "            return\n"
            "    raise NotImplementedError(rel)\n\n"
            "def _generate_mnist_stage_source(source):\n"
            "    rel = source.relative_to(ROOT).as_posix()\n"
            "    if rel.startswith('03_dnn_mnist/label_noise_sweep/01_dataset/figures/'):\n"
            "        _load_script('03_dnn_mnist/label_noise_sweep/01_dataset/src/make_figures.py').main(); return\n"
            "    if rel.startswith('03_dnn_mnist/manual_rules/01_dataset/figures/'):\n"
            "        _load_script('03_dnn_mnist/manual_rules/01_dataset/src/make_figures.py').main(); return\n"
            "    if rel.startswith('03_dnn_mnist/label_noise_sweep/02_complexity_measure/figures/'):\n"
            "        mod = _load_script('03_dnn_mnist/label_noise_sweep/02_complexity_measure/src/make_figures.py'); mod._plot(mod._read_summary(mod.SUMMARY_PATH)); return\n"
            "    if rel.startswith('03_dnn_mnist/manual_rules/02_complexity_measure/figures/'):\n"
            "        mod = _load_script('03_dnn_mnist/manual_rules/02_complexity_measure/src/make_figures.py'); mod._plot(mod._read_summary(mod.SUMMARY_PATH)); return\n"
            "    if rel.startswith('03_dnn_mnist/label_noise_sweep/03_reference_search/figures/'):\n"
            "        _load_script('03_dnn_mnist/label_noise_sweep/03_reference_search/src/make_figures.py').build_figures(); return\n"
            "    if rel.startswith('03_dnn_mnist/manual_rules/03_reference_search/figures/'):\n"
            "        _load_script('03_dnn_mnist/manual_rules/03_reference_search/src/make_figures.py').build_figures(); return\n"
            "    if rel.startswith('03_dnn_mnist/label_noise_sweep/04_sampling/figures/'):\n"
            "        mod = _load_script('03_dnn_mnist/label_noise_sweep/04_sampling/src/make_figures.py')\n"
            "        if source.name in {'eta_reference_phi_energy.png', 'eta_reference_direct_derivative.png'}:\n"
            "            summary = pd.read_csv(mod.DEFAULT_UNIT_SUMMARY_ROOT / 'eta_reference_phi_by_eta_radius.csv')\n"
            "            mod.plot_phi(summary, source.parent)\n"
            "            return\n"
            "        if 'logZ_split_distributions' in rel:\n"
            "            mod.plot_logz_inputs(mod.DEFAULT_LOGZ_INPUT_ROOT, source.parents[1], max_scatter_per_radius=30)\n"
            "            return\n"
            "    if rel.startswith('03_dnn_mnist/manual_rules/04_sampling/figures/fresh108_validation/'):\n"
            "        if source.exists():\n"
            "            return\n"
            "    if rel.startswith('03_dnn_mnist/manual_rules/04_sampling/figures/'):\n"
            "        mod = _load_script('03_dnn_mnist/manual_rules/04_sampling/src/make_figures.py')\n"
            "        phi = pd.read_csv(mod.SUMMARY_ROOT / 'figure_inputs' / 'phi_by_rule_radius.csv')\n"
            "        qc = pd.read_csv(mod.SUMMARY_ROOT / 'figure_inputs' / 'qc_diagnostics_by_rule_radius.csv')\n"
            "        if source.name == 'manual_rule_phi_energy.png': mod.plot_phi_curves(phi, source); return\n"
            "        if source.name == 'manual_rule_direct_derivative.png': mod.plot_direct_derivative(phi, source); return\n"
            "        if source.name == 'manual_rule_logZ_split_heatmap.png': mod.plot_qc_heatmap(qc, source); return\n"
            "        if source.name == 'manual_rule_ess_q05.png': mod.plot_ess_floor(qc, source); return\n"
            "        if 'logZ_split_distributions' in rel:\n"
            "            mod.plot_logz_split_distributions(mod.SUMMARY_ROOT / 'figure_inputs' / 'logZ_split', source.parent)\n"
            "            return\n"
            "    if rel.startswith('03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/figures/'):\n"
            "        mod = _load_script('03_dnn_mnist/label_noise_sweep/05_proxy_local_entropy/src/make_figures.py')\n"
            "        for name, value_key, sem_key, ylabel, title, output_name in mod.CURVE_FIGURES:\n"
            "            if source.name == output_name:\n"
            "                mod._plot_curve(mod._read_csv(mod.FIGURE_INPUT_ROOT / name / f'{name}.csv'), value_key, sem_key, ylabel, title, source)\n"
            "                return\n"
            "        if source.name == 'phase_like_A_by_eta.png': mod._plot_phase('phase_like_A_by_eta', 'eta', 'label noise eta', source.name); return\n"
            "        if source.name == 'phase_like_A_by_complexity.png': mod._plot_phase('phase_like_A_by_complexity', 'nmstv', '3-NN MNIST complexity', source.name); return\n"
            "        if source.name == 'logZ_split_qc_results.png': mod._plot_logz_qc(mod._read_csv(mod.FIGURE_INPUT_ROOT / 'logZ_split_qc_results' / 'logZ_split_qc_results.csv'), source); return\n"
            "        if source.name == 'reference_variability_results.png': mod._plot_reference_variability(mod._read_csv(mod.FIGURE_INPUT_ROOT / 'reference_variability_results' / 'reference_variability_results.csv'), source); return\n"
            "    if rel.startswith('03_dnn_mnist/manual_rules/05_proxy_local_entropy/figures/'):\n"
            "        mod = _load_script('03_dnn_mnist/manual_rules/05_proxy_local_entropy/src/make_figures.py')\n"
            "        mod.build_summarized_outputs()\n"
            "        phi = pd.read_csv(mod.FIGURE_INPUT_ROOT / 'phi_d_curve' / 'phi_d_curve.csv')\n"
            "        dphi = pd.read_csv(mod.FIGURE_INPUT_ROOT / 'derivative_phi_d_curve' / 'derivative_phi_d_curve.csv')\n"
            "        if source.name == 'phi_d_curve.png': mod._plot_curves(phi, 'delta_phi_energy_unit_mean', 'delta_phi_energy_unit_sem', 'phi(d) - phi(d0)', 'MNIST manual-rule phi(d)', source); return\n"
            "        if source.name == 'phi_energetic_d_curve.png': mod._plot_curves(phi, 'phi_energy_raw_mean', 'phi_energy_raw_sem', 'energetic phi(d)', 'MNIST manual-rule energetic phi(d)', source); return\n"
            "        if source.name == 'derivative_phi_d_curve.png': mod._plot_curves(dphi, 'd_delta_phi_energy_direct_dd_unit_mean', 'd_delta_phi_energy_direct_dd_unit_sem', 'd phi / dd', 'MNIST manual-rule direct derivative of phi(d)', source); return\n"
            "        if source.name == 'derivative_phi_energetic_d_curve.png': mod._plot_curves(dphi, 'd_phi_energy_direct_dd_unit_mean', 'd_phi_energy_direct_dd_unit_sem', 'energetic d phi / dd', 'MNIST manual-rule direct energetic derivative', source); return\n"
            "        if source.name == 'phase_like_A_by_rule.png': mod._plot_phase('rule_order', 'manual-rule order', 'phase_like_A_by_rule'); return\n"
            "        if source.name == 'phase_like_A_by_complexity.png': mod._plot_phase('nmstv_mean', '3-NN MNIST complexity', 'phase_like_A_by_complexity'); return\n"
            "        if source.name == 'logZ_split_qc_results.png': mod._plot_logz(source); return\n"
            "        if source.name == 'reference_variability_results.png': mod._plot_reference_variability(source); return\n"
            "    if source.exists():\n"
            "        print(f'using existing source: {rel}')\n"
            "        return\n"
            "    raise NotImplementedError(rel)\n\n"
            "def _generate_source(source, input_paths):\n"
            "    rel = source.relative_to(ROOT).as_posix()\n"
            "    if rel.startswith('01_theory/'):\n"
            "        _generate_theory_source(source)\n"
            "        return source\n"
            "    if rel.startswith('02_dnn_synthetic/figures/'):\n"
            "        _generate_synthetic_source(source)\n"
            "        return source\n"
            "    if rel.startswith('03_dnn_mnist/figures/'):\n"
            "        stage_inputs = [ROOT / item for item in input_paths if item.lower().endswith(tuple(IMAGE_SUFFIXES))]\n"
            "        if stage_inputs:\n"
            "            stage_source = stage_inputs[0]\n"
            "            _generate_mnist_stage_source(stage_source)\n"
            "            source.parent.mkdir(parents=True, exist_ok=True)\n"
            "            shutil.copy2(stage_source, source)\n"
            "            return source\n"
            "    if rel.startswith('03_dnn_mnist/'):\n"
            "        _generate_mnist_stage_source(source)\n"
            "        return source\n"
            "    if source.exists():\n"
            "        return source\n"
            "    raise NotImplementedError(rel)\n\n"
            "def build_single_figure(category_path):\n"
            "    row = _manifest_row(category_path)\n"
            "    source = ROOT / str(row['source_path'])\n"
            "    target = FIGURES_DIR / str(row['category_path'])\n"
            "    input_paths = _split_inputs(row.get('input_paths'))\n"
            "    print(f'figure: {category_path}')\n"
            "    generated = _generate_source(source, input_paths)\n"
            "    return _copy_display(generated, target)\n\n"
            "FIGURES_DIR\n"
        ),
    ]

    for idx, row in enumerate(rows, start=1):
        heading = f"## {idx:02d}. {row['category_path']}\n\nSource: `{row['source_path']}`"
        cells.append(_notebook_markdown_cell(heading))
        cells.append(_notebook_code_cell(f"build_single_figure({row['category_path']!r})\n"))

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
