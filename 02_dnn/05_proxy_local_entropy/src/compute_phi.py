from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


P = 2545
DEFAULT_LAMBDA_REG = 220.0


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["cell_id"]), str(row["dataset_tag"]), int(row["ref_id"]))


def _theta_norm_sq_from_path(path_value: object, *, repo_root: Path | None, cache: dict[str, float]) -> float:
    path_text = str(path_value or "")
    if not path_text:
        raise ValueError("theta_ref_norm_sq is missing and theta_path is empty")
    if path_text in cache:
        return cache[path_text]
    path = Path(path_text)
    if not path.is_absolute():
        if repo_root is None:
            raise ValueError(f"theta_ref_norm_sq is missing and theta_path is relative without repo_root: {path_text}")
        path = repo_root / path
    theta = np.load(path)
    norm_sq = float(np.dot(theta.reshape(-1), theta.reshape(-1)))
    cache[path_text] = norm_sq
    return norm_sq


def enrich_unit_rows_for_full_phi(
    unit_rows: list[dict[str, Any]],
    *,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    param_count: int = P,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Add explicit stripped/full logZ fields needed for absolute phi(d)."""
    cache: dict[str, float] = {}
    enriched: list[dict[str, Any]] = []
    for row in unit_rows:
        out = dict(row)
        stripped = _finite_float(out.get("logZ_inf_stripped", out.get("logZ_inf")))
        norm_sq = _finite_float(out.get("theta_ref_norm_sq"))
        if not np.isfinite(norm_sq):
            norm = _finite_float(out.get("theta_ref_norm"))
            if np.isfinite(norm):
                norm_sq = norm * norm
        if not np.isfinite(norm_sq):
            norm_sq = _theta_norm_sq_from_path(out.get("theta_path"), repo_root=repo_root, cache=cache)
        correction = _finite_float(out.get("reference_prior_log_weight"))
        if not np.isfinite(correction):
            correction = -float(lambda_reg) * norm_sq / (2.0 * float(param_count))
        full = _finite_float(out.get("logZ_inf_full"))
        if not np.isfinite(full):
            full = stripped + correction
        out["theta_ref_norm_sq"] = norm_sq
        out["reference_prior_log_weight"] = correction
        out["logZ_inf_stripped"] = stripped
        out["logZ_inf_full"] = full
        enriched.append(out)
    return enriched


def delta_phi_rows(unit_rows: list[dict[str, Any]], *, r0: float = 0.10) -> list[dict[str, Any]]:
    by_beta_ref: dict[float, dict[tuple[str, str, int], dict[float, dict[str, float]]]] = {}
    for row in unit_rows:
        beta = round(float(row["beta"]), 8)
        key = _row_key(row)
        radius = round(float(row["radius"]), 8)
        stripped = _finite_float(row.get("logZ_inf_stripped", row.get("logZ_inf")))
        full = _finite_float(row.get("logZ_inf_full"))
        if not np.isfinite(full):
            full = stripped
        by_beta_ref.setdefault(beta, {}).setdefault(key, {})[radius] = {
            "stripped": stripped,
            "full": full,
        }
    out: list[dict[str, Any]] = []
    for beta, ref_map in sorted(by_beta_ref.items()):
        radii = sorted({radius for values in ref_map.values() for radius in values})
        for radius in radii:
            paired = [
                (values[round(float(r0), 8)], values[radius])
                for values in ref_map.values()
                if round(float(r0), 8) in values
                and radius in values
                and np.isfinite(values[round(float(r0), 8)]["stripped"])
                and np.isfinite(values[radius]["stripped"])
                and np.isfinite(values[round(float(r0), 8)]["full"])
                and np.isfinite(values[radius]["full"])
            ]
            if not paired:
                out.append(
                    {
                        "beta": beta,
                        "radius": radius,
                        "ref_count": 0,
                        "delta_phi_full": float("nan"),
                        "delta_phi_energy": float("nan"),
                        "delta_phi_stripped_proxy": float("nan"),
                        "delta_phi_energy_stripped": float("nan"),
                        "delta_phi_reference_prior_correction": float("nan"),
                        "area_term": float((P - 1) / P * math.log(radius / r0)) if radius > 0 else float("nan"),
                        "claim": "no_claim",
                    }
                )
                continue
            base_full = np.asarray([a["full"] for a, _b in paired], dtype=np.float64)
            vals_full = np.asarray([b["full"] for _a, b in paired], dtype=np.float64)
            base_stripped = np.asarray([a["stripped"] for a, _b in paired], dtype=np.float64)
            vals_stripped = np.asarray([b["stripped"] for _a, b in paired], dtype=np.float64)
            energy_full = float(np.mean(vals_full - base_full) / P)
            energy_stripped = float(np.mean(vals_stripped - base_stripped) / P)
            area = float((P - 1) / P * math.log(radius / r0))
            delta_correction = energy_full - energy_stripped
            out.append(
                {
                    "beta": beta,
                    "radius": radius,
                    "ref_count": len(paired),
                    "delta_phi_full": area + energy_full,
                    "delta_phi_energy": energy_full,
                    "delta_phi_stripped_proxy": area + energy_stripped,
                    "delta_phi_energy_stripped": energy_stripped,
                    "delta_phi_reference_prior_correction": delta_correction,
                    "area_term": area,
                    "claim": "pass",
                }
            )
    return out
