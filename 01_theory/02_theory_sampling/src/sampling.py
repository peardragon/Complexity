from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln, ive, logsumexp

import make_summarized_outputs as final_tables


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "summarized_outputs"
SHELL_SUMMARY_ROOT = SUMMARY_ROOT
RAW_OUTPUT_ROOT = PROJECT_ROOT / "01_theory" / "02_theory_sampling" / "raw_outputs"
DATASET_POOL_ROOT = RAW_OUTPUT_ROOT / "dataset_pool"
REFERENCE_POOL_ROOT = RAW_OUTPUT_ROOT / "reference_pool"

DEFAULT_METHOD = "exact_shell_l2_vmf_adaptive_ce_tempered_smc"
RUN_ID = "two_pool_perceptron_alpha0p1_shell_sampling"


@dataclass(frozen=True)
class SMCParams:
    target_cess_fraction: float = 0.85
    resample_ess_fraction: float = 0.85
    max_steps: int = 160
    min_delta_t: float = 5.0e-5
    bisection_steps: int = 36
    mh_sweeps: int = 2
    move_kappa_factor: float = 80.0


def log_sphere_mgf(dim: int, kappa: float) -> float:
    """Return log E_{u~uniform sphere} exp(kappa * mu.u)."""
    k = float(abs(kappa))
    if k == 0.0:
        return 0.0
    nu = float(dim) / 2.0 - 1.0
    scaled = ive(nu, k)
    if scaled <= 0.0 or not np.isfinite(scaled):
        return float((k * k) / (2.0 * float(dim)))
    return float(gammaln(float(dim) / 2.0) + nu * np.log(2.0 / k) + np.log(scaled) + k)


def sample_vmf(mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(mu))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("vMF mean direction has zero or non-finite norm")
    mu = mu / norm
    dim = int(mu.size)
    if dim < 2:
        raise ValueError("vMF dimension must be at least 2")
    if float(kappa) <= 1.0e-12:
        out = rng.normal(size=(int(n), dim))
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out.astype(np.float64)

    b = (-2.0 * float(kappa) + np.sqrt(4.0 * float(kappa) ** 2 + float(dim - 1) ** 2)) / float(dim - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = float(kappa) * x0 + float(dim - 1) * np.log(max(1.0 - x0 * x0, 1.0e-300))
    w = np.empty(int(n), dtype=np.float64)
    filled = 0
    alpha = 0.5 * float(dim - 1)
    while filled < int(n):
        draw = max(1024, int((int(n) - filled) * 1.25))
        z = rng.beta(alpha, alpha, size=draw)
        candidate = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        log_accept = (
            float(kappa) * candidate
            + float(dim - 1) * np.log(np.maximum(1.0 - x0 * candidate, 1.0e-300))
            - c
        )
        accepted = candidate[np.log(rng.random(size=draw)) <= log_accept]
        take = min(accepted.size, int(n) - filled)
        if take:
            w[filled : filled + take] = accepted[:take]
            filled += take

    tangent = rng.normal(size=(int(n), dim))
    tangent -= (tangent @ mu)[:, None] * mu[None, :]
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    bad = tangent_norm[:, 0] <= 0.0
    while np.any(bad):
        tangent[bad] = rng.normal(size=(int(np.sum(bad)), dim))
        tangent[bad] -= (tangent[bad] @ mu)[:, None] * mu[None, :]
        tangent_norm[bad] = np.linalg.norm(tangent[bad], axis=1, keepdims=True)
        bad = tangent_norm[:, 0] <= 0.0
    tangent /= tangent_norm
    return (w[:, None] * mu[None, :] + np.sqrt(np.maximum(1.0 - w * w, 0.0))[:, None] * tangent).astype(
        np.float64
    )


def sample_vmf_batch(mus: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    mus = np.asarray(mus, dtype=np.float64)
    if mus.ndim != 2:
        raise ValueError(f"expected a matrix of mean directions, got shape {mus.shape}")
    n, dim = mus.shape
    norms = np.linalg.norm(mus, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("vMF batch contains zero or non-finite mean directions")
    mu = mus / norms
    if float(kappa) <= 1.0e-12:
        out = rng.normal(size=(int(n), int(dim)))
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out.astype(np.float64)

    b = (-2.0 * float(kappa) + np.sqrt(4.0 * float(kappa) ** 2 + float(dim - 1) ** 2)) / float(dim - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = float(kappa) * x0 + float(dim - 1) * np.log(max(1.0 - x0 * x0, 1.0e-300))
    w = np.empty(int(n), dtype=np.float64)
    filled = 0
    alpha = 0.5 * float(dim - 1)
    while filled < int(n):
        draw = max(1024, int((int(n) - filled) * 1.25))
        z = rng.beta(alpha, alpha, size=draw)
        candidate = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        log_accept = (
            float(kappa) * candidate
            + float(dim - 1) * np.log(np.maximum(1.0 - x0 * candidate, 1.0e-300))
            - c
        )
        accepted = candidate[np.log(rng.random(size=draw)) <= log_accept]
        take = min(accepted.size, int(n) - filled)
        if take:
            w[filled : filled + take] = accepted[:take]
            filled += take

    tangent = rng.normal(size=(int(n), int(dim)))
    tangent -= np.sum(tangent * mu, axis=1, keepdims=True) * mu
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    bad = tangent_norm[:, 0] <= 0.0
    while np.any(bad):
        tangent[bad] = rng.normal(size=(int(np.sum(bad)), int(dim)))
        tangent[bad] -= np.sum(tangent[bad] * mu[bad], axis=1, keepdims=True) * mu[bad]
        tangent_norm[bad] = np.linalg.norm(tangent[bad], axis=1, keepdims=True)
        bad = tangent_norm[:, 0] <= 0.0
    tangent /= tangent_norm
    return (w[:, None] * mu + np.sqrt(np.maximum(1.0 - w * w, 0.0))[:, None] * tangent).astype(np.float64)


def stable_softplus_neg_margin(theta_batch: np.ndarray, a_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dim = int(theta_batch.shape[1])
    margins = (theta_batch @ np.asarray(a_matrix, dtype=np.float64).T) / math.sqrt(dim)
    ce = np.logaddexp(0.0, -margins).sum(axis=1)
    err = np.mean(margins <= 0.0, axis=1)
    return ce.astype(np.float64), err.astype(np.float64)


def normalise_logw(logw: np.ndarray) -> np.ndarray:
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return logw
    return logw - logsumexp(logw)


def ess_from_logw(logw: np.ndarray) -> float:
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return 0.0
    return float(np.exp(2.0 * logsumexp(logw) - logsumexp(2.0 * logw)))


def ess_fraction_from_normalised_logw(logw_norm: np.ndarray) -> float:
    logw_norm = np.asarray(logw_norm, dtype=np.float64)
    if logw_norm.size == 0:
        return 0.0
    return float(np.exp(-logsumexp(2.0 * logw_norm)) / max(1, logw_norm.size))


def weighted_mean(values: np.ndarray, logw_norm: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    logw_norm = normalise_logw(np.asarray(logw_norm, dtype=np.float64))
    if values.size == 0:
        return float("nan")
    return float(np.sum(np.exp(logw_norm) * values))


def cess_fraction(logw_norm: np.ndarray, ce: np.ndarray, *, delta_t: float) -> float:
    if logw_norm.size == 0:
        return 0.0
    loga = -float(delta_t) * np.asarray(ce, dtype=np.float64)
    log_num = 2.0 * logsumexp(logw_norm + loga)
    log_den = logsumexp(logw_norm + 2.0 * loga)
    return float(np.exp(log_num - log_den))


def choose_next_temperature(t: float, ce: np.ndarray, logw_norm: np.ndarray, params: SMCParams) -> tuple[float, float]:
    full_cess = cess_fraction(logw_norm, ce, delta_t=1.0 - float(t))
    if full_cess >= float(params.target_cess_fraction):
        return 1.0, full_cess
    low = float(t)
    high = 1.0
    for _ in range(max(1, int(params.bisection_steps))):
        mid = 0.5 * (low + high)
        mid_cess = cess_fraction(logw_norm, ce, delta_t=mid - float(t))
        if mid_cess >= float(params.target_cess_fraction):
            low = mid
        else:
            high = mid
    t_new = low
    if t_new <= float(t) + 1.0e-15:
        t_new = min(1.0, float(t) + max(float(params.min_delta_t), 1.0e-12))
    return float(t_new), cess_fraction(logw_norm, ce, delta_t=t_new - float(t))


def systematic_resample(logw_norm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = np.exp(normalise_logw(logw_norm))
    n = int(weights.size)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    positions = (float(rng.random()) + np.arange(n, dtype=np.float64)) / float(n)
    return np.searchsorted(cdf, positions, side="left").astype(np.int64)


def rejuvenate(
    *,
    directions: np.ndarray,
    ce: np.ndarray,
    err: np.ndarray,
    theta_ref: np.ndarray,
    a_matrix: np.ndarray,
    radius: float,
    mu: np.ndarray,
    base_kappa: float,
    t: float,
    params: SMCParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    sweeps = max(0, int(params.mh_sweeps))
    if sweeps <= 0 or directions.size == 0:
        return directions, ce, err, []
    dim = int(theta_ref.size)
    move_kappa = float(params.move_kappa_factor) * float(dim)
    accept_rates: list[float] = []
    current_projection = directions @ mu
    for _ in range(sweeps):
        proposal = sample_vmf_batch(directions, move_kappa, rng)
        theta_prop = theta_ref[None, :] + math.sqrt(dim) * float(radius) * proposal
        ce_prop, err_prop = stable_softplus_neg_margin(theta_prop, a_matrix)
        proposal_projection = proposal @ mu
        log_accept = -float(t) * (ce_prop - ce) + float(base_kappa) * (
            proposal_projection - current_projection
        )
        accept = np.log(rng.random(size=ce.size)) <= np.minimum(0.0, log_accept)
        if np.any(accept):
            directions[accept] = proposal[accept]
            ce[accept] = ce_prop[accept]
            err[accept] = err_prop[accept]
            current_projection[accept] = proposal_projection[accept]
        accept_rates.append(float(np.mean(accept)) if accept.size else float("nan"))
    return directions, ce, err, accept_rates


def run_smc_single(
    *,
    theta_ref: np.ndarray,
    a_matrix: np.ndarray,
    radius: float,
    n_particles: int,
    mu: np.ndarray,
    base_kappa: float,
    rng: np.random.Generator,
    params: SMCParams,
) -> dict[str, Any]:
    dim = int(theta_ref.size)
    directions = sample_vmf(mu, base_kappa, int(n_particles), rng)
    theta_batch = theta_ref[None, :] + math.sqrt(dim) * float(radius) * directions
    ce, err = stable_softplus_neg_margin(theta_batch, a_matrix)
    logw_norm = np.full(int(n_particles), -math.log(int(n_particles)), dtype=np.float64)
    t = 0.0
    logz_ce = 0.0
    history: list[dict[str, Any]] = []
    resample_count = 0
    completed = True
    for step in range(max(1, int(params.max_steps))):
        if t >= 1.0 - 1.0e-12:
            break
        t_new, cess = choose_next_temperature(t, ce, logw_norm, params)
        delta_t = max(0.0, float(t_new) - float(t))
        loga = -delta_t * ce
        log_increment = float(logsumexp(logw_norm + loga))
        logz_ce += log_increment
        logw_norm = normalise_logw(logw_norm + loga)
        ess_frac = ess_fraction_from_normalised_logw(logw_norm)
        did_resample = ess_frac < float(params.resample_ess_fraction)
        if did_resample:
            idx = systematic_resample(logw_norm, rng)
            directions = directions[idx].copy()
            ce = ce[idx].copy()
            err = err[idx].copy()
            logw_norm = np.full(int(n_particles), -math.log(int(n_particles)), dtype=np.float64)
            ess_frac = 1.0
            resample_count += 1
        directions, ce, err, accept_rates = rejuvenate(
            directions=directions,
            ce=ce,
            err=err,
            theta_ref=theta_ref,
            a_matrix=a_matrix,
            radius=float(radius),
            mu=mu,
            base_kappa=float(base_kappa),
            t=float(t_new),
            params=params,
            rng=rng,
        )
        history.append(
            {
                "step": int(step + 1),
                "t_start": float(t),
                "t_end": float(t_new),
                "delta_t": float(delta_t),
                "cess_fraction": float(cess),
                "ess_fraction_after_reweight": float(ess_frac),
                "resampled": bool(did_resample),
                "mh_acceptance": float(np.mean(accept_rates)) if accept_rates else float("nan"),
            }
        )
        t = float(t_new)
    else:
        completed = t >= 1.0 - 1.0e-12
    return {
        "directions": directions,
        "ce": ce,
        "err": err,
        "logw_norm": normalise_logw(logw_norm),
        "logZ_CE": float(logz_ce) if completed else float("nan"),
        "completed": bool(completed),
        "history": history,
        "resample_count": int(resample_count),
    }


def run_smc_split(
    *,
    theta_ref: np.ndarray,
    a_matrix: np.ndarray,
    radius: float,
    n_samples: int,
    mu: np.ndarray,
    base_kappa: float,
    seed: int,
    params: SMCParams,
) -> dict[str, Any]:
    n_total = int(n_samples)
    n0 = max(1, n_total // 2)
    n1 = n_total - n0
    splits = []
    for split_id, n_particles in enumerate((n0, n1)):
        splits.append(
            run_smc_single(
                theta_ref=theta_ref,
                a_matrix=a_matrix,
                radius=float(radius),
                n_particles=int(n_particles),
                mu=mu,
                base_kappa=float(base_kappa),
                rng=np.random.default_rng(int(seed) + 7919 * (split_id + 1)),
                params=params,
            )
        )
    completed = all(bool(split["completed"]) for split in splits)
    logz_values = np.asarray([float(split["logZ_CE"]) for split in splits], dtype=np.float64)
    counts = np.asarray([n0, n1], dtype=np.float64)
    dim = int(theta_ref.size)
    if completed and np.all(np.isfinite(logz_values)):
        logz_ce = float(logsumexp(np.log(counts / np.sum(counts)) + logz_values))
        split_diff = float(abs(logz_values[0] - logz_values[1]) / dim)
    else:
        logz_ce = float("nan")
        split_diff = float("inf")

    directions = np.concatenate([split["directions"] for split in splits], axis=0)
    ce = np.concatenate([split["ce"] for split in splits], axis=0)
    err = np.concatenate([split["err"] for split in splits], axis=0)
    split_labels = np.concatenate([np.full(int(count), idx, dtype=np.int32) for idx, count in enumerate((n0, n1))])
    logw = np.concatenate(
        [
            math.log(float(count) / float(np.sum(counts))) + np.asarray(split["logw_norm"], dtype=np.float64)
            for count, split in zip((n0, n1), splits)
        ]
    )
    histories = [split["history"] for split in splits]
    flat_history = [row for history in histories for row in history]
    cess_values = [float(row["cess_fraction"]) for row in flat_history if np.isfinite(float(row["cess_fraction"]))]
    ess_values = [
        float(row["ess_fraction_after_reweight"])
        for row in flat_history
        if np.isfinite(float(row["ess_fraction_after_reweight"]))
    ]
    mh_values = [float(row["mh_acceptance"]) for row in flat_history if np.isfinite(float(row["mh_acceptance"]))]
    return {
        "directions": directions,
        "ce": ce,
        "err": err,
        "split": split_labels,
        "target_logw": normalise_logw(logw),
        "logZ_CE": logz_ce,
        "split_logZ": logz_values.tolist(),
        "split_logZ_per_N_diff": split_diff,
        "completed": bool(completed),
        "smc_step_count": int(max(len(split["history"]) for split in splits)),
        "smc_total_step_count": int(sum(len(split["history"]) for split in splits)),
        "smc_resample_count": int(sum(int(split["resample_count"]) for split in splits)),
        "smc_min_cess_fraction": float(min(cess_values)) if cess_values else float("nan"),
        "smc_mean_cess_fraction": float(np.mean(cess_values)) if cess_values else float("nan"),
        "smc_min_ess_fraction": float(min(ess_values)) if ess_values else float("nan"),
        "smc_mean_mh_acceptance": float(np.mean(mh_values)) if mh_values else float("nan"),
        "smc_histories": histories,
    }


def deterministic_seed(row: pd.Series, radius_index: dict[float, int]) -> int:
    n_value = int(row["N"])
    dataset_id = int(row["dataset_id"])
    ref_id = int(row["ref_id"])
    ridx = int(radius_index[float(row["radius"])])
    return int(2026062915 + n_value * 100000 + dataset_id * 1000 + ref_id * 100 + (ridx - 1) * 5)


def load_dataset_and_reference(n_value: int, dataset_id: int, ref_id: int) -> tuple[np.ndarray, np.ndarray]:
    n_dir = f"N_{int(n_value)}"
    dataset_name = f"dataset_{int(dataset_id) + 1:03d}"
    ref_name = f"ref_{int(ref_id) + 1:03d}"
    dataset_dir = DATASET_POOL_ROOT / n_dir / dataset_name
    reference_path = REFERENCE_POOL_ROOT / n_dir / dataset_name / ref_name / "reference.npz"
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    a_matrix = np.load(dataset_dir / "dataset.npz")["A"].astype(np.float64)
    theta_ref = np.load(reference_path)["theta"].astype(np.float64)
    return a_matrix, theta_ref


def write_npz_exact(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def project_relative_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def compute_unit(
    row: pd.Series,
    *,
    params: SMCParams,
    radius_index: dict[float, int],
    seed_offset: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.time()
    n_value = int(row["N"])
    dataset_id = int(row["dataset_id"])
    ref_id = int(row["ref_id"])
    radius = float(row["radius"])
    n_particles = int(row["n_particles"])
    a_matrix, theta_ref = load_dataset_and_reference(n_value, dataset_id, ref_id)
    dim = int(theta_ref.size)
    if dim != n_value:
        raise ValueError(f"reference dimension mismatch for dataset={dataset_id} ref={ref_id}: {dim} != {n_value}")
    ce_ref, err_ref = stable_softplus_neg_margin(theta_ref[None, :], a_matrix)
    ref_norm = float(np.linalg.norm(theta_ref))
    mu = -theta_ref / ref_norm
    kappa = math.sqrt(dim) * radius * ref_norm
    seed = deterministic_seed(row, radius_index) + int(seed_offset)

    smc = run_smc_split(
        theta_ref=theta_ref,
        a_matrix=a_matrix,
        radius=radius,
        n_samples=n_particles,
        mu=mu,
        base_kappa=kappa,
        seed=seed,
        params=params,
    )
    directions = np.asarray(smc["directions"], dtype=np.float64)
    ce = np.asarray(smc["ce"], dtype=np.float64)
    err = np.asarray(smc["err"], dtype=np.float64)
    split = np.asarray(smc["split"], dtype=np.int32)
    target_logw = np.asarray(smc["target_logw"], dtype=np.float64)
    projection = directions @ mu
    theta_norm_sq = ref_norm * ref_norm + dim * radius * radius - 2.0 * math.sqrt(dim) * radius * ref_norm * projection
    l2_penalty = 0.5 * theta_norm_sq

    logm = log_sphere_mgf(dim, kappa)
    log_prefactor = -0.5 * dim * radius * radius + logm
    ref_l2_constant = -0.5 * ref_norm * ref_norm
    logz_ce = float(smc["logZ_CE"])
    logz_shell_stripped = float(log_prefactor + logz_ce) if np.isfinite(logz_ce) else float("nan")
    logz_shell_full = float(logz_shell_stripped + ref_l2_constant) if np.isfinite(logz_shell_stripped) else float("nan")
    split_logz = [float(log_prefactor + value) if np.isfinite(float(value)) else float("nan") for value in smc["split_logZ"]]
    split_logz_per_n_diff = (
        float(abs(split_logz[0] - split_logz[1]) / dim) if all(np.isfinite(value) for value in split_logz) else float("inf")
    )
    ess = ess_from_logw(target_logw)
    sample_rel = str(row["sample_payload_path"] if pd.notna(row["sample_payload_path"]) else row["samples_path"])
    sample_path = project_relative_path(sample_rel)
    write_npz_exact(
        sample_path,
        ce=ce.astype(np.float64),
        error=err.astype(np.float64),
        logw_target=target_logw.astype(np.float64),
        logw_ce=(-ce).astype(np.float64),
        split=split.astype(np.int32),
        l2_penalty=l2_penalty.astype(np.float64),
        direction_projection=projection.astype(np.float64),
        smc_history_json=np.asarray(json.dumps(smc["smc_histories"], allow_nan=True)),
    )
    row_update = {
        "N": n_value,
        "M": int(a_matrix.shape[0]),
        "dataset_id": dataset_id,
        "ref_id": ref_id,
        "radius": radius,
        "n_particles": n_particles,
        "ce_ref": float(ce_ref[0]),
        "err_ref": float(err_ref[0]),
        "ref_norm": ref_norm,
        "sampler_method": DEFAULT_METHOD,
        "direct_qc_pass": np.nan,
        "fallback_used": False,
        "direct_logZ_CE": np.nan,
        "direct_ess_frac": np.nan,
        "direct_split_logZ_per_N_diff": np.nan,
        "logZ_CE": logz_ce,
        "logZ_shell_stripped": logz_shell_stripped,
        "logZ_shell_full": logz_shell_full,
        "ref_l2_constant": ref_l2_constant,
        "logM": logm,
        "kappa": kappa,
        "ess": ess,
        "ess_frac": float(ess / max(1, n_particles)),
        "weighted_ce": weighted_mean(ce, target_logw),
        "weighted_error": weighted_mean(err, target_logw),
        "weighted_accuracy": 1.0 - weighted_mean(err, target_logw),
        "weighted_l2_penalty": weighted_mean(l2_penalty, target_logw),
        "split0_logZ_shell": split_logz[0],
        "split1_logZ_shell": split_logz[1],
        "split_logZ_per_N_diff": split_logz_per_n_diff,
        "samples_path": sample_rel,
        "smc_completed": bool(smc["completed"]),
        "smc_step_count": int(smc["smc_step_count"]),
        "smc_total_step_count": int(smc["smc_total_step_count"]),
        "smc_resample_count": int(smc["smc_resample_count"]),
        "smc_min_cess_fraction": float(smc["smc_min_cess_fraction"]),
        "smc_mean_cess_fraction": float(smc["smc_mean_cess_fraction"]),
        "smc_mean_mh_acceptance": float(smc["smc_mean_mh_acceptance"]),
        "smc_mean_global_mh_acceptance": np.nan,
        "source_run_id": row.get("source_run_id", RUN_ID),
        "sample_payload_path": sample_rel,
        "split_qc_threshold_per_N": 0.006,
        "elapsed_s": float(time.time() - start),
        "seed": seed,
        "reused": False,
        "legacy_direct_qc_pass": row.get("direct_qc_pass", np.nan),
        "legacy_fallback_used": row.get("fallback_used", np.nan),
        "legacy_direct_logZ_CE": row.get("direct_logZ_CE", np.nan),
        "legacy_direct_ess_frac": row.get("direct_ess_frac", np.nan),
        "legacy_direct_split_logZ_per_N_diff": row.get("direct_split_logZ_per_N_diff", np.nan),
    }
    diagnostics = {
        "sample_path": str(sample_path),
        "split_diff": split_logz_per_n_diff,
        "min_cess": float(smc["smc_min_cess_fraction"]),
        "steps": int(smc["smc_step_count"]),
        "elapsed_s": row_update["elapsed_s"],
    }
    return row_update, diagnostics


def rebuild_tables(
    units: pd.DataFrame,
    *,
    input_unit_count: int,
    processed_unit_count: int,
    output_root: Path = SHELL_SUMMARY_ROOT,
) -> dict[str, Any]:
    phi = final_tables.build_phi_summary(units, split_threshold=0.006, smc_cess_threshold=0.8)
    qc = final_tables.build_qc_summary(phi, split_threshold=0.006, smc_cess_threshold=0.8)
    validation = final_tables.validation_summary(
        units,
        phi,
        qc,
        base={
            "input_unit_count": int(input_unit_count),
            "processed_unit_count": int(processed_unit_count),
            "nondefault_method_count_before_validation": int((units["sampler_method"] != DEFAULT_METHOD).sum()),
            "over_particle_cap_count_before_validation": int((units["n_particles"] > 32768).sum()),
        },
        particle_cap=32768,
        split_threshold=0.006,
        smc_cess_threshold=0.8,
    )
    output_root = project_relative_path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sample_summary_path = output_root / "sample_unit_summary.csv"
    units.to_csv(sample_summary_path, index=False)
    logz_split = final_tables.build_logz_split_frame(units, source_path=sample_summary_path)
    final_tables.write_phi_figure_inputs(phi, output_root / "figure_inputs" / "phi_by_sampling")
    final_tables.write_phi_energetic_figure_inputs(phi, output_root / "figure_inputs" / "phi_energetic_by_sampling")
    final_tables.write_logz_figure_inputs(logz_split, output_root / "figure_inputs" / "logZ_split")
    return validation


def main() -> None:
    global DATASET_POOL_ROOT, REFERENCE_POOL_ROOT

    parser = argparse.ArgumentParser(description="Run two-pool shell sampling units and refresh summarized outputs.")
    parser.add_argument("--summary", type=Path, default=SHELL_SUMMARY_ROOT / "sample_unit_summary.csv")
    parser.add_argument("--n", type=int, action="append", default=[], help="Optional N filter; repeat for multiple values.")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means all selected rows.")
    parser.add_argument("--start-index", type=int, default=0, help="Debug offset into selected rows.")
    parser.add_argument("--only-index", type=int, action="append", default=[], help="Process specific dataframe index.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Deterministic offset added to selected unit seeds.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_POOL_ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_POOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=SHELL_SUMMARY_ROOT)
    parser.add_argument("--skip-summary-write", action="store_true", help="Do not refresh summarized outputs after sampling.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing payloads or summaries.")
    args = parser.parse_args()

    DATASET_POOL_ROOT = project_relative_path(args.dataset_root).resolve()
    REFERENCE_POOL_ROOT = project_relative_path(args.reference_root).resolve()

    summary_path = args.summary if args.summary.is_absolute() else PROJECT_ROOT / args.summary
    units = pd.read_csv(summary_path, low_memory=False).astype(object)
    input_unit_count = int(len(units))
    radius_index = {float(radius): idx + 1 for idx, radius in enumerate(sorted(units["radius"].unique()))}
    if args.only_index:
        selected_indices = [int(index) for index in args.only_index]
    else:
        selected = pd.Series(True, index=units.index)
        if args.n:
            selected &= units["N"].astype(int).isin([int(value) for value in args.n])
        selected_indices = list(units.index[selected])
    if int(args.start_index) > 0 and not args.only_index:
        selected_indices = selected_indices[int(args.start_index) :]
    if int(args.limit) > 0 and not args.only_index:
        selected_indices = selected_indices[: int(args.limit)]
    print(f"selected rows: {len(selected_indices)}")
    if args.dry_run:
        for index in selected_indices[:5]:
            row = units.loc[index]
            a_matrix, theta_ref = load_dataset_and_reference(int(row["N"]), int(row["dataset_id"]), int(row["ref_id"]))
            ce_ref, err_ref = stable_softplus_neg_margin(theta_ref[None, :], a_matrix)
            print(
                json.dumps(
                    {
                        "index": int(index),
                        "N": int(row["N"]),
                        "dataset_id": int(row["dataset_id"]),
                        "ref_id": int(row["ref_id"]),
                        "radius": float(row["radius"]),
                        "summary_ce_ref": float(row["ce_ref"]),
                        "computed_ce_ref": float(ce_ref[0]),
                        "computed_err_ref": float(err_ref[0]),
                    },
                    sort_keys=True,
                )
            )
        return

    params = SMCParams()
    diagnostics: list[dict[str, Any]] = []
    for processed, index in enumerate(selected_indices, start=1):
        update, diag = compute_unit(
            units.loc[index],
            params=params,
            radius_index=radius_index,
            seed_offset=int(args.seed_offset),
        )
        for key, value in update.items():
            if key not in units.columns:
                units[key] = np.nan
            units.at[index, key] = value
        diagnostics.append({"index": int(index), **diag})
        if processed == 1 or processed % 100 == 0 or processed == len(selected_indices):
            elapsed = sum(float(item["elapsed_s"]) for item in diagnostics)
            print(
                f"processed {processed}/{len(selected_indices)} "
                f"last_split={diag['split_diff']:.6g} last_min_cess={diag['min_cess']:.6g} "
                f"sum_unit_elapsed={elapsed:.1f}s",
                flush=True,
            )

    if args.skip_summary_write:
        print("payloads written; summary tables not updated because --skip-summary-write was supplied")
        return

    validation = rebuild_tables(
        units,
        input_unit_count=input_unit_count,
        processed_unit_count=len(selected_indices),
        output_root=args.output_root,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
