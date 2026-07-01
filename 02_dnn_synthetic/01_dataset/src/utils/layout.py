from __future__ import annotations

import re
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = STAGE_ROOT.parents[1]

RAW_ROOT = STAGE_ROOT / "raw_outputs"
FIGURE_ROOT = STAGE_ROOT / "figures"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
SAMPLE_SUMMARY_ROOT = FIGURE_INPUT_ROOT / "sample_figures"
SPIN_SUMMARY_ROOT = FIGURE_INPUT_ROOT / "spin_dynamics"

SAMPLE_FIGURE_PATH = FIGURE_ROOT / "sample_figure.png"
SPIN_FIGURE_PATH = FIGURE_ROOT / "spin_dynamics_phase_transition.png"

K_GRAPH = 10
DATASET_ID_FOR_SAMPLE = 0
BETA_DIR_RE = re.compile(r"^beta_(?P<beta>\d+p\d+)$")
DATASET_RE = re.compile(r"^dataset_(?P<dataset_id>\d+)$")


def relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def repo_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def beta_from_dir_name(beta_dir_name: str) -> float:
    match = BETA_DIR_RE.match(beta_dir_name)
    if match is None:
        raise ValueError(f"Unexpected beta directory: {beta_dir_name}")
    return float(match.group("beta").replace("p", "."))


def dataset_id_from_dir(dataset_dir: Path) -> int:
    match = DATASET_RE.match(dataset_dir.name)
    if match is None:
        raise ValueError(f"Unexpected dataset directory: {dataset_dir.name}")
    return int(match.group("dataset_id"))


def dataset_label(dataset_id: int) -> str:
    return f"dataset_{int(dataset_id):03d}"


def beta_dirs() -> list[Path]:
    dirs = [path for path in RAW_ROOT.iterdir() if path.is_dir() and BETA_DIR_RE.match(path.name)]
    return sorted(dirs, key=lambda path: beta_from_dir_name(path.name))


def dataset_dirs(beta_dir: Path) -> list[Path]:
    dirs = [path for path in beta_dir.iterdir() if path.is_dir() and DATASET_RE.match(path.name)]
    return sorted(dirs, key=dataset_id_from_dir)


def find_dataset_dir(beta_dir: Path, dataset_id: int) -> Path:
    dataset_dir = beta_dir / dataset_label(dataset_id)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    return dataset_dir


def source_dataset_path(beta_dir: Path, dataset_id: int) -> str:
    return relative_to_repo(beta_dir / dataset_label(dataset_id))


def source_image_path(beta_dir: Path, dataset_id: int) -> str:
    return relative_to_repo(beta_dir / dataset_label(dataset_id) / "region_fill_d2.png")


def panel_image_path(dataset_dir: Path) -> Path:
    for filename in ("region_fill_d2.png", "dataset_view.png", "scatter_d2.png"):
        path = dataset_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(dataset_dir / "region_fill_d2.png")
