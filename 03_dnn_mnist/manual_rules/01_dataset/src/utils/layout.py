from __future__ import annotations

import re
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[2]
DNN_ROOT = STAGE_ROOT.parents[1]

CONFIG_PATH = STAGE_ROOT / "config" / "default.json"
RAW_ROOT = STAGE_ROOT / "raw_outputs"
FIGURE_ROOT = STAGE_ROOT / "figures"
SUMMARY_ROOT = STAGE_ROOT / "summarized_outputs"
FIGURE_INPUT_ROOT = SUMMARY_ROOT / "figure_inputs"
SAMPLE_FIGURE_INPUT_ROOT = FIGURE_INPUT_ROOT / "sample_figure"

SAMPLE_FIGURE_PATH = FIGURE_ROOT / "sample_figure.png"
DATASET_FILENAME = "dataset.npz"
RULE_DIR_RE = re.compile(r"^rule_(?P<rule_num>\d+)$")


def relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(DNN_ROOT))


def repo_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    candidates = [DNN_ROOT / path]
    text = path.as_posix()
    marker = "03_dnn_mnist/"
    if marker in text:
        candidates.append(DNN_ROOT / text.split(marker, 1)[1])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def rule_number(rule_dir_name: str) -> int:
    match = RULE_DIR_RE.match(rule_dir_name)
    if match is None:
        raise ValueError(f"Unexpected rule directory: {rule_dir_name}")
    return int(match.group("rule_num"))


def rule_dirs() -> list[Path]:
    if not RAW_ROOT.exists():
        return []
    dirs = [path for path in RAW_ROOT.iterdir() if path.is_dir() and RULE_DIR_RE.match(path.name)]
    return sorted(dirs, key=lambda path: rule_number(path.name))


def dataset_path(rule_dir: Path) -> Path:
    return rule_dir / DATASET_FILENAME


def source_dataset_path(rule_dir: Path) -> str:
    return relative_to_repo(dataset_path(rule_dir))
