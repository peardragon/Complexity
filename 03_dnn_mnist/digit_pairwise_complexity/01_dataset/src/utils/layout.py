from __future__ import annotations

import re
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[2]
PAIRWISE_ROOT = STAGE_ROOT.parent
DNN_ROOT = STAGE_ROOT.parents[1]

CONFIG_PATH = STAGE_ROOT / "config" / "default.json"
PAIR_MAPPING_PATH = PAIRWISE_ROOT / "config" / "pair_mapping.csv"
RAW_ROOT = STAGE_ROOT / "raw_outputs"
FIGURE_ROOT = STAGE_ROOT / "figures"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
SAMPLE_FIGURE_INPUT_ROOT = FIGURE_INPUT_ROOT / "sample_figure"

SAMPLE_FIGURE_PATH = FIGURE_ROOT / "sample_figure.png"
DATASET_FILENAME = "dataset.npz"
PAIR_DIR_RE = re.compile(r"^pair_(?P<a>\d+)_(?P<b>\d+)$")


def pair_id(digit_a: int, digit_b: int) -> str:
    return f"pair_{int(digit_a)}_{int(digit_b)}"


def pair_label(digit_a: int, digit_b: int) -> str:
    return f"{int(digit_a)}/{int(digit_b)}"


def relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(DNN_ROOT))


def pair_sort_key(pair_dir_name: str) -> tuple[int, int]:
    match = PAIR_DIR_RE.match(pair_dir_name)
    if match is None:
        raise ValueError(f"Unexpected pair directory: {pair_dir_name}")
    return int(match.group("a")), int(match.group("b"))


def pair_dirs() -> list[Path]:
    if not RAW_ROOT.exists():
        return []
    dirs = [path for path in RAW_ROOT.iterdir() if path.is_dir() and PAIR_DIR_RE.match(path.name)]
    return sorted(dirs, key=lambda path: pair_sort_key(path.name))


def dataset_path(pair_dir: Path) -> Path:
    return pair_dir / DATASET_FILENAME


def source_dataset_path(pair_dir: Path) -> str:
    return relative_to_repo(dataset_path(pair_dir))
