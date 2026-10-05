"""Numerical and rendering helpers for the four retained MNIST UMAP panels."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from contourpy import contour_generator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


UMAP_PARAMETERS: dict[str, Any] = {
    "n_components": 2,
    "n_neighbors": 30,
    "min_dist": 0.12,
    "metric": "euclidean",
    "init": "spectral",
    "random_state": 20260804,
    "transform_seed": 20260804,
    "n_jobs": 1,
    "low_memory": True,
}
BOUNDARY_GRID_SIZE = 180
BOUNDARY_NEIGHBORS = 61
BOUNDARY_BANDWIDTH_NEIGHBOR = 31
BOUNDARY_SMOOTHING_SIGMA = 3.0
CLOSE_MIN_SUPPORT = 0.50
FAR_MIN_SUPPORT = 0.55
PANEL_SIZE_INCHES = 3.153
PANEL_PNG_DPI = 600
DIGIT_ZOOM = 4.5

LABEL_DATASETS = {
    "noise_eta_0p00": Path(
        "03_dnn_mnist/label_noise_sweep/01_dataset/raw_outputs/datasets/"
        "dataset_000/noise_eta_0p00/dataset.npz"
    ),
    "noise_eta_0p50": Path(
        "03_dnn_mnist/label_noise_sweep/01_dataset/raw_outputs/datasets/"
        "dataset_000/noise_eta_0p50/dataset.npz"
    ),
}
PAIR_DATASETS = {
    "pair_4_9": Path(
        "03_dnn_mnist/digit_pairwise_complexity/01_dataset/raw_outputs/"
        "datasets/dataset_000/pair_4_9/dataset.npz"
    ),
    "pair_0_1": Path(
        "03_dnn_mnist/digit_pairwise_complexity/01_dataset/raw_outputs/"
        "datasets/dataset_000/pair_0_1/dataset.npz"
    ),
}
DATASET_ENTRYPOINTS = {
    "label": Path(
        "03_dnn_mnist/label_noise_sweep/01_dataset/src/make_dataset.py"
    ),
    "pair": Path(
        "03_dnn_mnist/digit_pairwise_complexity/01_dataset/src/make_dataset.py"
    ),
}


def _source_key(files: list[str], *aliases: str) -> str:
    key = next((value for value in aliases if value in files), None)
    if key is None:
        raise KeyError(f"dataset is missing all aliases {aliases}")
    return key


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load the raw 10x10 train/test representation used by the paper UMAP."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        files = list(payload.files)
        x_train = np.asarray(
            payload[_source_key(files, "x_train_raw10", "X_train_raw10")]
        )
        x_test = np.asarray(
            payload[_source_key(files, "x_test_raw10", "X_test_raw10")]
        )
        y_train = np.asarray(
            payload[_source_key(files, "y_train", "Y_train")]
        ).reshape(-1)
        y_test = np.asarray(
            payload[_source_key(files, "y_test", "Y_test")]
        ).reshape(-1)
        digit_train = np.asarray(payload["digit_train"]).reshape(-1)
        digit_test = np.asarray(payload["digit_test"]).reshape(-1)
    images = np.concatenate([x_train, x_test], axis=0)
    labels = np.concatenate([y_train, y_test], axis=0).astype(np.int8)
    digits = np.concatenate([digit_train, digit_test], axis=0).astype(np.int16)
    if images.shape[0] != labels.size or labels.size != digits.size:
        raise ValueError(f"{path}: raw image/label/digit count mismatch")
    if images.reshape(images.shape[0], -1).shape[1] != 100:
        raise ValueError(f"{path}: expected flattened 10x10 raw images")
    if not np.all(np.isfinite(images)) or not np.all(np.isin(labels, (-1, 1))):
        raise ValueError(f"{path}: non-finite images or non-binary labels")
    return {
        "images": images.reshape(images.shape[0], 100),
        "labels": labels,
        "digits": digits,
    }


def fit_umap(images: np.ndarray, seed: int) -> np.ndarray:
    """Fit the preserved single-thread UMAP contract to raw pixels / 255."""

    import umap

    matrix = np.asarray(images, dtype=np.float32).reshape(len(images), -1) / 255.0
    parameters = dict(UMAP_PARAMETERS)
    parameters["random_state"] = int(seed)
    parameters["transform_seed"] = int(seed)
    coordinates = np.asarray(umap.UMAP(**parameters).fit_transform(matrix), dtype=np.float64)
    if coordinates.shape != (len(matrix), 2) or not np.all(np.isfinite(coordinates)):
        raise RuntimeError("UMAP returned invalid coordinates")
    return coordinates


def _adaptive_probability(
    tree: cKDTree,
    query: np.ndarray,
    labels: np.ndarray,
    *,
    leave_one_out: bool,
) -> np.ndarray:
    count = min(BOUNDARY_NEIGHBORS + int(leave_one_out), len(labels))
    distances, indices = tree.query(
        np.asarray(query, dtype=np.float64), k=count, workers=1
    )
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    if leave_one_out:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    column = min(BOUNDARY_BANDWIDTH_NEIGHBOR - 1, distances.shape[1] - 1)
    bandwidth = np.maximum(distances[:, column : column + 1], 1.0e-6)
    weights = np.exp(-0.5 * np.square(distances / bandwidth))
    positive = (labels[indices] > 0).astype(np.float64)
    return np.sum(weights * positive, axis=1) / np.maximum(
        np.sum(weights, axis=1), 1.0e-12
    )


def boundary_geometry(
    coordinates: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the preserved adaptive-kNN boundary and per-point diagnostics."""

    xy = np.asarray(coordinates, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    minima = np.min(xy, axis=0)
    maxima = np.max(xy, axis=0)
    center = 0.5 * (minima + maxima)
    span = max(float(np.max(maxima - minima)), 0.4)
    half_span = 0.5 * span + max(0.045 * span, 0.2)
    x_axis = np.linspace(center[0] - half_span, center[0] + half_span, BOUNDARY_GRID_SIZE)
    y_axis = np.linspace(center[1] - half_span, center[1] + half_span, BOUNDARY_GRID_SIZE)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing="xy")
    tree = cKDTree(xy)
    probability = _adaptive_probability(
        tree,
        np.column_stack([grid_x.ravel(), grid_y.ravel()]),
        labels,
        leave_one_out=False,
    ).reshape(BOUNDARY_GRID_SIZE, BOUNDARY_GRID_SIZE)
    probability = gaussian_filter(
        probability, sigma=BOUNDARY_SMOOTHING_SIGMA, mode="nearest"
    )
    components = [
        np.asarray(component, dtype=np.float64)
        for component in contour_generator(x=x_axis, y=y_axis, z=probability).lines(0.5)
        if len(component) >= 8
    ]
    if not components:
        raise RuntimeError("UMAP embedding has no smoothed 0.5 label boundary")
    distance, _ = cKDTree(np.vstack(components)).query(xy, k=1, workers=1)
    sample_probability = _adaptive_probability(
        tree, xy, labels, leave_one_out=True
    )
    support = np.where(labels > 0, sample_probability, 1.0 - sample_probability)
    return grid_x, grid_y, probability, np.asarray(distance), np.asarray(support)


def choose_exemplars(
    coordinates: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    digits: np.ndarray,
) -> list[dict[str, Any]]:
    _, _, _, distance, support = boundary_geometry(coordinates, labels)
    output: list[dict[str, Any]] = []
    for prefix, kind, threshold in (
        ("N", "close", CLOSE_MIN_SUPPORT),
        ("F", "far", FAR_MIN_SUPPORT),
    ):
        for order, label in enumerate((1, -1), start=1):
            candidates = np.flatnonzero((labels == label) & (support >= threshold))
            if candidates.size == 0:
                raise RuntimeError(f"no {kind} exemplar for label {label}")
            values = distance[candidates]
            selected = int(
                candidates[np.argmin(values) if kind == "close" else np.argmax(values)]
            )
            image = np.rint(
                np.clip(np.asarray(images[selected], dtype=np.float64), 0.0, 255.0)
            ).astype(np.uint8)
            output.append(
                {
                    "selection_id": f"{prefix}{order}",
                    "umap_1": float(coordinates[selected, 0]),
                    "umap_2": float(coordinates[selected, 1]),
                    "image_uint8": image.reshape(-1),
                    "digit": int(digits[selected]),
                }
            )
    return output


def _atomic_figure(figure: plt.Figure, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=f"{path.suffix}.tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format=path.suffix.lstrip("."),
            dpi=PANEL_PNG_DPI if path.suffix == ".png" else None,
            bbox_inches=None,
            facecolor="white",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def render_panel(
    *,
    coordinates: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    digits: np.ndarray,
    pdf_path: Path,
    png_path: Path,
    layout: str,
) -> None:
    """Render one missing composed panel without replacing either output."""

    if pdf_path.is_file():
        return
    grid_x, grid_y, probability, _, _ = boundary_geometry(coordinates, labels)
    examples = choose_exemplars(coordinates, images, labels, digits)
    figure, axis = plt.subplots(figsize=(PANEL_SIZE_INCHES, PANEL_SIZE_INCHES))
    colors = {1: "#356A9A", -1: "#C07A3D"}
    for label in (1, -1):
        mask = labels == label
        axis.scatter(
            coordinates[mask, 0], coordinates[mask, 1],
            s=8.5, c=colors[label], alpha=0.58,
            edgecolors="none", rasterized=True,
        )
    if float(np.min(probability)) <= 0.5 <= float(np.max(probability)):
        axis.contour(
            grid_x, grid_y, probability, levels=[0.5],
            colors=["#242424"], linewidths=1.05,
        )
    if layout == "high":
        positions = {
            "N1": (0.22, 0.78), "N2": (0.78, 0.78),
            "F1": (0.22, 0.22), "F2": (0.78, 0.22),
        }
    elif layout == "low":
        positions = {
            "N1": (0.18, 0.82), "N2": (0.82, 0.82),
            "F1": (0.18, 0.18), "F2": (0.82, 0.18),
        }
    elif layout == "label":
        positions = {
            "N1": (0.20, 0.80), "N2": (0.80, 0.80),
            "F1": (0.20, 0.20), "F2": (0.80, 0.20),
        }
    else:
        raise ValueError(f"unknown annotation layout: {layout}")
    for example in examples:
        point = (example["umap_1"], example["umap_2"])
        axis.add_artist(
            AnnotationBbox(
                OffsetImage(
                    np.asarray(example["image_uint8"]).reshape(10, 10),
                    cmap="gray", zoom=DIGIT_ZOOM, interpolation="nearest",
                ),
                point,
                xybox=positions[str(example["selection_id"])],
                xycoords="data", boxcoords="axes fraction", frameon=True,
                bboxprops={"edgecolor": "#202226", "linewidth": 0.8, "facecolor": "white"},
                arrowprops={"arrowstyle": "-", "color": "#202226", "linewidth": 0.7},
                pad=0.12, zorder=20,
            )
        )
        axis.scatter(
            [point[0]], [point[1]], s=42, facecolors="none",
            edgecolors="#202226", linewidths=0.9, zorder=25,
        )
    axis.set_axis_off()
    axis.set_aspect("equal", adjustable="box")
    axis.margins(0.06)
    figure.tight_layout(pad=0.1)
    _atomic_figure(figure, pdf_path)
    _atomic_figure(figure, png_path)
    plt.close(figure)


def _bootstrap_dataset(project_root: Path, family: str) -> None:
    entrypoint = project_root / DATASET_ENTRYPOINTS[family]
    if not entrypoint.is_file():
        raise FileNotFoundError(
            f"required dataset is absent and its public entrypoint is missing: {entrypoint}"
        )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, str(entrypoint), "--dataset-index", "0", "--execute"],
        cwd=project_root,
        env=environment,
        check=True,
    )


def ensure_datasets(
    project_root: Path, *, need_label: bool, needed_pairs: set[str]
) -> None:
    if need_label and any(not (project_root / path).is_file() for path in LABEL_DATASETS.values()):
        _bootstrap_dataset(project_root, "label")
    if needed_pairs and any(
        not (project_root / PAIR_DATASETS[pair]).is_file() for pair in needed_pairs
    ):
        _bootstrap_dataset(project_root, "pair")
    required = (
        ([project_root / path for path in LABEL_DATASETS.values()] if need_label else [])
        + [project_root / PAIR_DATASETS[pair] for pair in sorted(needed_pairs)]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("dataset bootstrap did not create: " + ", ".join(missing))


def build_missing_assets(
    *,
    project_root: Path,
    output_paths: Mapping[str, tuple[Path, Path]],
    missing_keys: set[str],
) -> dict[str, str]:
    """Build only missing panels; callers perform filename skip first."""

    need_label = bool({"label_eta_0", "label_eta_0p5"} & missing_keys)
    needed_pairs = {
        pair
        for key, pair in (("pair_high", "pair_4_9"), ("pair_low", "pair_0_1"))
        if key in missing_keys
    }
    ensure_datasets(project_root, need_label=need_label, needed_pairs=needed_pairs)
    actions: dict[str, str] = {}
    if need_label:
        clean = load_dataset(project_root / LABEL_DATASETS["noise_eta_0p00"])
        noisy = load_dataset(project_root / LABEL_DATASETS["noise_eta_0p50"])
        if not np.array_equal(clean["images"], noisy["images"]) or not np.array_equal(clean["digits"], noisy["digits"]):
            raise ValueError("eta=0 and eta=.5 must share raw images and digits")
        coordinates = fit_umap(clean["images"], int(UMAP_PARAMETERS["random_state"]))
        for key, payload in (("label_eta_0", clean), ("label_eta_0p5", noisy)):
            if key not in missing_keys:
                actions[key] = "skipped_existing"
                continue
            pdf, png = output_paths[key]
            render_panel(
                coordinates=coordinates, images=clean["images"],
                labels=payload["labels"], digits=clean["digits"],
                pdf_path=pdf, png_path=png, layout="label",
            )
            actions[key] = "generated"
    for offset, (key, pair) in enumerate(
        (("pair_high", "pair_4_9"), ("pair_low", "pair_0_1")), start=1
    ):
        if key not in missing_keys:
            actions[key] = "skipped_existing"
            continue
        payload = load_dataset(project_root / PAIR_DATASETS[pair])
        expected_digits = {4, 9} if pair == "pair_4_9" else {0, 1}
        if set(map(int, np.unique(payload["digits"]))) != expected_digits:
            raise ValueError(f"{pair}: digit identities disagree with endpoint contract")
        coordinates = fit_umap(
            payload["images"], int(UMAP_PARAMETERS["random_state"]) + 100 * offset
        )
        pdf, png = output_paths[key]
        render_panel(
            coordinates=coordinates, images=payload["images"],
            labels=payload["labels"], digits=payload["digits"],
            pdf_path=pdf,
            png_path=png,
            layout="high" if pair == "pair_4_9" else "low",
        )
        actions[key] = "generated"
    return actions
