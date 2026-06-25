from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class Basin:
    name: str
    center: np.ndarray
    axes: np.ndarray
    angle: float
    offset: float
    region_radius: float

    @property
    def rotation(self) -> np.ndarray:
        c = float(np.cos(self.angle))
        s = float(np.sin(self.angle))
        return np.asarray([[c, -s], [s, c]], dtype=np.float64)


@dataclass(frozen=True)
class Ridge:
    center: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    width: float
    length: float
    amplitude: float


def _as_points(z: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(z, dtype=np.float64)
    scalar = arr.ndim == 1
    if scalar:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected points with shape (*, 2), got {arr.shape}")
    return arr, scalar


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("zero or non-finite vector")
    return v / norm


class ProxyLandscape:
    """Smooth low-loss proxy with dataset-induced ridges and rough terms."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.beta = float(config["beta"])
        domain = config["domain"]
        self.xlim = tuple(float(x) for x in domain["xlim"])
        self.ylim = tuple(float(y) for y in domain["ylim"])
        landscape = config["landscape"]
        self.softmin_tau = float(landscape["softmin_tau"])
        self.fixed_ridge_scale = float(landscape.get("fixed_ridge_scale", 1.0))
        self.rough_scale = float(landscape.get("rough_scale", 0.0))
        self.l2_scale = float(landscape.get("l2_scale", 0.0))
        self.basins = [
            Basin(
                name=str(item["name"]),
                center=np.asarray(item["center"], dtype=np.float64),
                axes=np.asarray(item["axes"], dtype=np.float64),
                angle=float(item["angle"]),
                offset=float(item["offset"]),
                region_radius=float(item["region_radius"]),
            )
            for item in landscape["basins"]
        ]
        self.ridges = self._build_fixed_ridges()
        self.rough = self._build_rough_terms(
            int(landscape.get("n_rough_terms", 0)),
            int(config.get("seed", 0)) + 7919,
        )

    def _build_fixed_ridges(self) -> list[Ridge]:
        specs = [
            ([1.55, 0.70], [1.0, -0.22], 0.120, 2.55, 2.25),
            ([1.84, 1.18], [0.78, -1.0], 0.110, 1.65, 2.00),
            ([-1.30, -0.92], [0.92, 0.55], 0.130, 2.45, 1.75),
            ([-2.00, -1.38], [0.32, 1.0], 0.100, 1.55, 1.65),
            ([0.55, -1.05], [-0.40, 1.0], 0.120, 1.95, 1.20),
        ]
        ridges: list[Ridge] = []
        for center, normal, width, length, amplitude in specs:
            n = _unit(np.asarray(normal, dtype=np.float64))
            t = np.asarray([-n[1], n[0]], dtype=np.float64)
            ridges.append(
                Ridge(
                    center=np.asarray(center, dtype=np.float64),
                    normal=n,
                    tangent=t,
                    width=float(width),
                    length=float(length),
                    amplitude=float(amplitude) * self.fixed_ridge_scale,
                )
            )
        return ridges

    def _build_rough_terms(self, n_terms: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        if n_terms <= 0:
            return {
                "center": np.empty((0, 2), dtype=np.float64),
                "normal": np.empty((0, 2), dtype=np.float64),
                "tangent": np.empty((0, 2), dtype=np.float64),
                "width": np.empty(0, dtype=np.float64),
                "length": np.empty(0, dtype=np.float64),
                "amplitude": np.empty(0, dtype=np.float64),
            }
        centers = np.column_stack(
            [
                rng.uniform(self.xlim[0], self.xlim[1], size=n_terms),
                rng.uniform(self.ylim[0], self.ylim[1], size=n_terms),
            ]
        )
        angles = rng.uniform(0.0, 2.0 * np.pi, size=n_terms)
        normal = np.column_stack([np.cos(angles), np.sin(angles)])
        tangent = np.column_stack([-np.sin(angles), np.cos(angles)])
        width = rng.uniform(0.045, 0.16, size=n_terms)
        length = rng.uniform(0.25, 1.10, size=n_terms)
        amplitude = rng.lognormal(mean=-1.75, sigma=0.55, size=n_terms)
        return {
            "center": centers.astype(np.float64),
            "normal": normal.astype(np.float64),
            "tangent": tangent.astype(np.float64),
            "width": width.astype(np.float64),
            "length": length.astype(np.float64),
            "amplitude": amplitude.astype(np.float64),
        }

    @property
    def rough_count(self) -> int:
        return int(self.rough["center"].shape[0])

    def inside_domain(self, points: np.ndarray) -> np.ndarray:
        z, scalar = _as_points(points)
        mask = (
            (z[:, 0] >= self.xlim[0])
            & (z[:, 0] <= self.xlim[1])
            & (z[:, 1] >= self.ylim[0])
            & (z[:, 1] <= self.ylim[1])
        )
        return bool(mask[0]) if scalar else mask

    def _basin_energy_grad(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energies = []
        grads = []
        for basin in self.basins:
            diff = z - basin.center[None, :]
            rot = basin.rotation
            local = diff @ rot
            inv_axes2 = 1.0 / np.maximum(basin.axes, 1.0e-12) ** 2
            e = basin.offset + 0.5 * np.sum((local * local) * inv_axes2[None, :], axis=1)
            grad_local = local * inv_axes2[None, :]
            grad = grad_local @ rot.T
            energies.append(e)
            grads.append(grad)
        energy_mat = np.column_stack(energies)
        grad_stack = np.stack(grads, axis=1)
        weights_log = -energy_mat / self.softmin_tau
        weights = np.exp(weights_log - logsumexp(weights_log, axis=1)[:, None])
        soft_energy = -self.softmin_tau * logsumexp(weights_log, axis=1)
        soft_grad = np.sum(weights[:, :, None] * grad_stack, axis=1)
        return soft_energy, soft_grad

    @staticmethod
    def _ridge_energy_grad(z: np.ndarray, ridge: Ridge) -> tuple[np.ndarray, np.ndarray]:
        diff = z - ridge.center[None, :]
        sn = (diff @ ridge.normal) / ridge.width
        st = (diff @ ridge.tangent) / ridge.length
        value = ridge.amplitude * np.exp(-0.5 * (sn * sn + st * st))
        grad = value[:, None] * (
            -(sn / ridge.width)[:, None] * ridge.normal[None, :]
            - (st / ridge.length)[:, None] * ridge.tangent[None, :]
        )
        return value, grad

    def _fixed_ridge_energy_grad(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energy = np.zeros(z.shape[0], dtype=np.float64)
        grad = np.zeros_like(z)
        for ridge in self.ridges:
            e, g = self._ridge_energy_grad(z, ridge)
            energy += e
            grad += g
        return energy, grad

    def _rough_energy_grad(
        self,
        z: np.ndarray,
        batch_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_total = self.rough_count
        if n_total <= 0 or self.rough_scale == 0.0:
            return np.zeros(z.shape[0], dtype=np.float64), np.zeros_like(z)
        if batch_indices is None:
            idx = np.arange(n_total)
        else:
            idx = np.asarray(batch_indices, dtype=np.int64)
            if idx.size == 0:
                raise ValueError("rough minibatch is empty")
        centers = self.rough["center"][idx]
        normal = self.rough["normal"][idx]
        tangent = self.rough["tangent"][idx]
        width = self.rough["width"][idx]
        length = self.rough["length"][idx]
        amplitude = self.rough["amplitude"][idx]
        diff = z[:, None, :] - centers[None, :, :]
        sn = np.sum(diff * normal[None, :, :], axis=2) / width[None, :]
        st = np.sum(diff * tangent[None, :, :], axis=2) / length[None, :]
        value_terms = amplitude[None, :] * np.exp(-0.5 * (sn * sn + st * st))
        grad_terms = value_terms[:, :, None] * (
            -(sn / width[None, :])[:, :, None] * normal[None, :, :]
            - (st / length[None, :])[:, :, None] * tangent[None, :, :]
        )
        # The minibatch mean is an unbiased estimate of the full rough mean.
        energy = self.rough_scale * np.mean(value_terms, axis=1)
        grad = self.rough_scale * np.mean(grad_terms, axis=1)
        return energy, grad

    def _confinement_energy_grad(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = 4.4
        r2 = np.sum(z * z, axis=1)
        energy = 0.012 * (r2 / (scale * scale)) ** 3
        grad = 0.012 * 3.0 * (r2 / (scale * scale)) ** 2
        grad = grad[:, None] * (2.0 * z / (scale * scale))
        return energy, grad

    def _l2_energy_grad(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.l2_scale == 0.0:
            return np.zeros(z.shape[0], dtype=np.float64), np.zeros_like(z)
        energy = self.l2_scale * np.sum(z * z, axis=1)
        grad = 2.0 * self.l2_scale * z
        return energy, grad

    def energy_and_grad(
        self,
        points: np.ndarray,
        rough_batch_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        z, scalar = _as_points(points)
        basin_e, basin_g = self._basin_energy_grad(z)
        ridge_e, ridge_g = self._fixed_ridge_energy_grad(z)
        rough_e, rough_g = self._rough_energy_grad(z, rough_batch_indices)
        conf_e, conf_g = self._confinement_energy_grad(z)
        l2_e, l2_g = self._l2_energy_grad(z)
        energy = basin_e + ridge_e + rough_e + conf_e + l2_e
        grad = basin_g + ridge_g + rough_g + conf_g + l2_g
        inside = self.inside_domain(z)
        if not np.all(inside):
            energy = energy.copy()
            grad = grad.copy()
            outside = ~inside
            energy[outside] = 1.0e9
            grad[outside] = 0.0
        if scalar:
            return energy[:1], grad[:1]
        return energy, grad

    def energy(self, points: np.ndarray) -> np.ndarray:
        energy, _ = self.energy_and_grad(points)
        return energy

    def grad(self, points: np.ndarray, rough_batch_indices: np.ndarray | None = None) -> np.ndarray:
        _, grad = self.energy_and_grad(points, rough_batch_indices)
        return grad

    def region_names(self) -> list[str]:
        return [basin.name for basin in self.basins]

    def region_mask(self, points: np.ndarray) -> np.ndarray:
        z, scalar = _as_points(points)
        masks = []
        distances = []
        for basin in self.basins:
            diff = z - basin.center[None, :]
            local = diff @ basin.rotation
            dist = np.sqrt(np.sum((local / basin.axes[None, :]) ** 2, axis=1))
            distances.append(dist)
            masks.append(dist <= basin.region_radius)
        raw = np.column_stack(masks)
        distance_mat = np.column_stack(distances)
        nearest = np.argmin(distance_mat, axis=1)
        out = np.zeros_like(raw, dtype=bool)
        for idx in range(len(self.basins)):
            out[:, idx] = raw[:, idx] & (nearest == idx)
        return out[:1] if scalar else out

    def region_reference_frame(self) -> list[dict[str, float | str]]:
        rows = []
        for basin in self.basins:
            radius = float(np.linalg.norm(basin.center))
            angle = float(np.arctan2(basin.center[1], basin.center[0]))
            rows.append(
                {
                    "region": basin.name,
                    "center_x": float(basin.center[0]),
                    "center_y": float(basin.center[1]),
                    "radius": radius,
                    "angle": angle,
                    "axis_major": float(max(basin.axes)),
                    "axis_minor": float(min(basin.axes)),
                    "region_radius": float(basin.region_radius),
                }
            )
        return rows

    def grid_reference(self, grid_n: int) -> dict[str, np.ndarray]:
        x = np.linspace(self.xlim[0], self.xlim[1], int(grid_n))
        y = np.linspace(self.ylim[0], self.ylim[1], int(grid_n))
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        energy = self.energy(points).reshape(xx.shape)
        logp = -self.beta * energy
        logp = logp - np.max(logp)
        density = np.exp(logp)
        density /= np.sum(density)
        masks = self.region_mask(points)
        region_mass = density.ravel() @ masks.astype(np.float64)
        return {
            "x": x,
            "y": y,
            "xx": xx,
            "yy": yy,
            "energy": energy,
            "density": density,
            "region_mass": region_mass,
        }
