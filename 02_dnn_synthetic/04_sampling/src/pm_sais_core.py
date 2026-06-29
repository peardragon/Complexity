from __future__ import annotations

import math
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from dnn_model import P, ce_and_error, ce_error_batch, ce_radial_grad_batch
from io_utils import ensure_dir, save_json
from loaders import ReferenceRecord, load_dataset, load_theta
from vmf import log_sphere_mgf, sample_vmf, sample_vmf_batch


DERIVATIVE_METHODOLOGY_ID = "exact_shell_l2_vmf_ce_tempered_smc_radial_score_derivative_v1"
_REF_STATIC_CACHE: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
_REF_STATIC_CACHE_MAX = int(os.environ.get("COMPLEXITY_REF_STATIC_CACHE_MAX", "2048"))


@dataclass(frozen=True)
class PMSAISParams:
    gamma_ce: float = 0.4
    lambda_reg: float = 220.0
    chunk_size: int = 64
    device: str = "cpu"
    dtype: str = "float64"
    h_ladder: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0)
    delta_ce_thresholds: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    relative_delta_ce_thresholds: tuple[float, ...] = (1.0, 5.0, 10.0, 50.0, 100.0)
    sampler: str = "direct"
    smc_target_cess_fraction: float = 0.70
    smc_resample_ess_fraction: float = 0.70
    smc_resample_every_step: bool = False
    smc_max_steps: int = 200
    smc_min_delta_t: float = 1.0e-4
    smc_bisection_steps: int = 32
    smc_mh_sweeps: int = 2
    smc_move_kappa: float = 0.0
    smc_move_kappa_factor: float = 50.0
    radial_derivative_enabled: bool = False
    radial_derivative_chunk_size: int = 0


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("-inf")
    return float(logsumexp(values) - math.log(values.size))


def ess_from_logw(logw: np.ndarray) -> float:
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return 0.0
    return float(np.exp(2.0 * logsumexp(logw) - logsumexp(2.0 * logw)))


def weighted_mean(values: np.ndarray, logw: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    logw = np.asarray(logw, dtype=np.float64)
    if values.size == 0 or logw.size == 0:
        return float("nan")
    weights = np.exp(logw - logsumexp(logw))
    return float(np.sum(weights * values))


def weighted_sd(values: np.ndarray, logw: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    logw = np.asarray(logw, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(logw)
    if not np.any(mask):
        return float("nan")
    values = values[mask]
    logw = logw[mask]
    weights = np.exp(logw - logsumexp(logw))
    mean = float(np.sum(weights * values))
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2)))


def weighted_ratio(logw: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    logw = np.asarray(logw, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if logw.size == 0 or not np.any(mask):
        return 0.0, float("-inf"), 0.0
    log_ratio = float(logsumexp(logw[mask]) - logsumexp(logw))
    return float(np.exp(log_ratio)), log_ratio, ess_from_logw(logw[mask])


def _reference_static_fields(
    *,
    record: ReferenceRecord,
    theta_ref: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    key = (str(Path(record.theta_path).resolve()), str(Path(record.dataset_path).resolve()))
    try:
        cached = _REF_STATIC_CACHE.pop(key)
    except KeyError:
        ref_norm = float(np.linalg.norm(theta_ref))
        if not np.isfinite(ref_norm) or ref_norm <= 0.0:
            raise ValueError(f"Bad theta_ref norm for {record.theta_path}: {ref_norm}")
        ce_ref, err_ref = ce_and_error(theta_ref, data["X_train"], data["y"])
        cached = {
            "theta_ref_norm": ref_norm,
            "mu": -theta_ref / ref_norm,
            "ce_ref": float(ce_ref),
            "err_ref": float(err_ref),
        }
    _REF_STATIC_CACHE[key] = cached
    while len(_REF_STATIC_CACHE) > _REF_STATIC_CACHE_MAX:
        _REF_STATIC_CACHE.popitem(last=False)
    return cached


def _split_weighted_means(values: np.ndarray, logw: np.ndarray, split: np.ndarray) -> tuple[float, float]:
    outs: list[float] = []
    split = np.asarray(split, dtype=np.int32)
    for split_id in (0, 1):
        mask = split == int(split_id)
        outs.append(weighted_mean(values[mask], logw[mask]) if np.any(mask) else float("nan"))
    return outs[0], outs[1]


def _normalise_logw(logw: np.ndarray) -> np.ndarray:
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return logw
    return logw - logsumexp(logw)


def _ess_fraction_from_normalised_logw(logw_norm: np.ndarray) -> float:
    logw_norm = np.asarray(logw_norm, dtype=np.float64)
    if logw_norm.size == 0:
        return 0.0
    ess = float(np.exp(-logsumexp(2.0 * logw_norm)))
    return float(ess / max(1, logw_norm.size))


def _cess_fraction(logw_norm: np.ndarray, ce: np.ndarray, *, delta_t: float, gamma_ce: float) -> float:
    logw_norm = np.asarray(logw_norm, dtype=np.float64)
    ce = np.asarray(ce, dtype=np.float64)
    if logw_norm.size == 0:
        return 0.0
    loga = -float(delta_t) * float(gamma_ce) * ce
    log_num = 2.0 * logsumexp(logw_norm + loga)
    log_den = logsumexp(logw_norm + 2.0 * loga)
    return float(np.exp(log_num - log_den))


def _choose_next_temperature(
    *,
    t: float,
    ce: np.ndarray,
    logw_norm: np.ndarray,
    params: PMSAISParams,
) -> tuple[float, float]:
    target = float(params.smc_target_cess_fraction)
    if target <= 0.0:
        cess = _cess_fraction(logw_norm, ce, delta_t=1.0 - float(t), gamma_ce=params.gamma_ce)
        return 1.0, cess
    full_cess = _cess_fraction(logw_norm, ce, delta_t=1.0 - float(t), gamma_ce=params.gamma_ce)
    if full_cess >= target:
        return 1.0, full_cess
    low = float(t)
    high = 1.0
    for _ in range(max(1, int(params.smc_bisection_steps))):
        mid = 0.5 * (low + high)
        mid_cess = _cess_fraction(logw_norm, ce, delta_t=mid - float(t), gamma_ce=params.gamma_ce)
        if mid_cess >= target:
            low = mid
        else:
            high = mid
    t_new = low
    if t_new <= float(t) + 1.0e-15:
        t_new = min(1.0, float(t) + max(float(params.smc_min_delta_t), 1.0e-12))
    actual = _cess_fraction(logw_norm, ce, delta_t=t_new - float(t), gamma_ce=params.gamma_ce)
    return float(t_new), actual


def _systematic_resample(logw_norm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = np.exp(_normalise_logw(logw_norm))
    n = int(weights.size)
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    positions = (float(rng.random()) + np.arange(n, dtype=np.float64)) / float(n)
    return np.searchsorted(cdf, positions, side="left").astype(np.int64)


def _rejuvenate_vmf_rw(
    *,
    directions: np.ndarray,
    ce: np.ndarray,
    err: np.ndarray,
    theta_ref: np.ndarray,
    data: dict[str, np.ndarray],
    radius: float,
    mu: np.ndarray,
    base_kappa: float,
    t: float,
    params: PMSAISParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    sweeps = max(0, int(params.smc_mh_sweeps))
    if sweeps <= 0 or directions.size == 0:
        return directions, ce, err, []
    move_kappa = float(params.smc_move_kappa)
    if move_kappa <= 0.0:
        move_kappa = float(params.smc_move_kappa_factor) * float(P)
    if move_kappa <= 0.0:
        return directions, ce, err, []
    rates: list[float] = []
    current_projection = directions @ mu
    for _ in range(sweeps):
        proposal = sample_vmf_batch(directions, move_kappa, rng)
        theta_prop = theta_ref[None, :] + math.sqrt(P) * float(radius) * proposal
        ce_prop, err_prop = ce_error_batch(
            theta_prop,
            data["X_train"],
            data["y"],
            chunk_size=int(params.chunk_size),
            device=str(params.device),
            dtype=str(params.dtype),
        )
        proposal_projection = proposal @ mu
        log_accept = (
            -float(t) * float(params.gamma_ce) * (ce_prop - ce)
            + float(base_kappa) * (proposal_projection - current_projection)
        )
        accept = np.log(rng.random(size=ce.size)) <= np.minimum(0.0, log_accept)
        if np.any(accept):
            directions[accept] = proposal[accept]
            ce[accept] = ce_prop[accept]
            err[accept] = err_prop[accept]
            current_projection[accept] = proposal_projection[accept]
        rates.append(float(np.mean(accept)) if accept.size else float("nan"))
    return directions, ce, err, rates


def _run_ce_smc_single(
    *,
    theta_ref: np.ndarray,
    data: dict[str, np.ndarray],
    radius: float,
    n_particles: int,
    mu: np.ndarray,
    base_kappa: float,
    rng: np.random.Generator,
    params: PMSAISParams,
) -> dict[str, Any]:
    n_particles = int(n_particles)
    if n_particles <= 0:
        raise ValueError("SMC split must contain at least one particle")
    directions = sample_vmf(mu, base_kappa, n_particles, rng)
    theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * directions
    ce, err = ce_error_batch(
        theta_batch,
        data["X_train"],
        data["y"],
        chunk_size=int(params.chunk_size),
        device=str(params.device),
        dtype=str(params.dtype),
    )
    logw_norm = np.full(n_particles, -math.log(n_particles), dtype=np.float64)
    t = 0.0
    logz_ce = 0.0
    history: list[dict[str, Any]] = []
    resample_count = 0
    completed = True
    max_steps = max(1, int(params.smc_max_steps))
    for step in range(max_steps):
        if t >= 1.0 - 1.0e-12:
            break
        t_new, cess_frac = _choose_next_temperature(t=t, ce=ce, logw_norm=logw_norm, params=params)
        delta_t = max(0.0, float(t_new) - float(t))
        loga = -delta_t * float(params.gamma_ce) * ce
        log_increment = float(logsumexp(logw_norm + loga))
        logz_ce += log_increment
        logw_norm = _normalise_logw(logw_norm + loga)
        ess_frac = _ess_fraction_from_normalised_logw(logw_norm)
        did_resample = bool(params.smc_resample_every_step) or ess_frac < float(params.smc_resample_ess_fraction)
        if did_resample:
            idx = _systematic_resample(logw_norm, rng)
            directions = directions[idx].copy()
            ce = ce[idx].copy()
            err = err[idx].copy()
            logw_norm = np.full(n_particles, -math.log(n_particles), dtype=np.float64)
            ess_frac = 1.0
            resample_count += 1
        directions, ce, err, accept_rates = _rejuvenate_vmf_rw(
            directions=directions,
            ce=ce,
            err=err,
            theta_ref=theta_ref,
            data=data,
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
                "cess_fraction": float(cess_frac),
                "ess_fraction_after_reweight": float(ess_frac),
                "resampled": did_resample,
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
        "logw_norm": _normalise_logw(logw_norm),
        "logZ_CE": float(logz_ce) if completed else float("nan"),
        "completed": completed,
        "history": history,
        "resample_count": resample_count,
    }


def _run_ce_smc_split(
    *,
    theta_ref: np.ndarray,
    data: dict[str, np.ndarray],
    radius: float,
    n_samples: int,
    mu: np.ndarray,
    base_kappa: float,
    seed: int,
    params: PMSAISParams,
) -> dict[str, Any]:
    n_total = int(n_samples)
    if n_total < 2:
        raise ValueError("SMC split estimator requires at least two particles")
    n0 = max(1, n_total // 2)
    n1 = n_total - n0
    if n1 <= 0:
        n0, n1 = 1, 1
        n_total = 2
    splits = []
    for split_id, n_particles in enumerate((n0, n1)):
        splits.append(
            _run_ce_smc_single(
                theta_ref=theta_ref,
                data=data,
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
    if completed and np.all(np.isfinite(logz_values)):
        logz_ce = float(logsumexp(np.log(counts / np.sum(counts)) + logz_values))
        split_diff = float(abs(logz_values[0] - logz_values[1]) / P)
    else:
        logz_ce = float("nan")
        split_diff = float("inf")
    directions = np.concatenate([np.asarray(split["directions"], dtype=np.float64) for split in splits], axis=0)
    ce = np.concatenate([np.asarray(split["ce"], dtype=np.float64) for split in splits], axis=0)
    err = np.concatenate([np.asarray(split["err"], dtype=np.float64) for split in splits], axis=0)
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
    ess_values = [float(row["ess_fraction_after_reweight"]) for row in flat_history if np.isfinite(float(row["ess_fraction_after_reweight"]))]
    mh_values = [float(row["mh_acceptance"]) for row in flat_history if np.isfinite(float(row["mh_acceptance"]))]
    return {
        "directions": directions,
        "ce": ce,
        "err": err,
        "split": split_labels,
        "target_logw": _normalise_logw(logw),
        "logZ_CE": logz_ce,
        "split_logZ": logz_values.tolist(),
        "split_logZ_per_P_diff": split_diff,
        "completed": completed,
        "smc_step_count": int(max(len(split["history"]) for split in splits)),
        "smc_total_step_count": int(sum(len(split["history"]) for split in splits)),
        "smc_resample_count": int(sum(int(split["resample_count"]) for split in splits)),
        "smc_min_cess_fraction": float(min(cess_values)) if cess_values else float("nan"),
        "smc_mean_cess_fraction": float(np.mean(cess_values)) if cess_values else float("nan"),
        "smc_min_ess_fraction": float(min(ess_values)) if ess_values else float("nan"),
        "smc_mean_mh_acceptance": float(np.mean(mh_values)) if mh_values else float("nan"),
        "smc_min_mh_acceptance": float(min(mh_values)) if mh_values else float("nan"),
        "smc_histories": histories,
    }


def _radial_derivative_fields(
    *,
    theta_ref: np.ndarray,
    data: dict[str, np.ndarray],
    radius: float,
    directions: np.ndarray,
    ce: np.ndarray,
    target_logw: np.ndarray,
    split: np.ndarray,
    params: PMSAISParams,
) -> dict[str, Any]:
    theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * directions
    chunk_size = int(params.radial_derivative_chunk_size or params.chunk_size)
    replay_ce, radial_grad_ce = ce_radial_grad_batch(
        theta_batch,
        directions,
        data["X_train"],
        data["y"],
        chunk_size=max(1, chunk_size),
        device=str(params.device),
        dtype=str(params.dtype),
    )
    theta_ref_dot_u_over_sqrt_p = (directions @ theta_ref) / math.sqrt(P)
    prior_radial_score = -float(params.lambda_reg) * theta_ref_dot_u_over_sqrt_p
    ce_radial_score = -float(params.gamma_ce) * math.sqrt(P) * radial_grad_ce
    variable_score = prior_radial_score + ce_radial_score
    total_score = -float(params.lambda_reg) * float(radius) + variable_score
    split0, split1 = _split_weighted_means(total_score, target_logw, split)
    split_diff = float(abs(split0 - split1) / P) if np.isfinite(split0) and np.isfinite(split1) else float("inf")
    ce_diff = np.abs(np.asarray(replay_ce, dtype=np.float64) - np.asarray(ce, dtype=np.float64))
    dlogz = weighted_mean(total_score, target_logw)
    return {
        "radial_derivative_methodology_id": DERIVATIVE_METHODOLOGY_ID,
        "dlogZ_inf_dr": dlogz,
        "dlogZ_inf_stripped_dr": dlogz,
        "dlogZ_inf_full_dr": dlogz,
        "dlogZ_dr_split0": split0,
        "dlogZ_dr_split1": split1,
        "split_dlogZ_dr_per_P_diff": split_diff,
        "weighted_prior_radial_score": weighted_mean(prior_radial_score, target_logw),
        "weighted_ce_radial_score": weighted_mean(ce_radial_score, target_logw),
        "weighted_variable_radial_score": weighted_mean(variable_score, target_logw),
        "weighted_total_radial_score": dlogz,
        "weighted_radial_grad_ce": weighted_mean(radial_grad_ce, target_logw),
        "weighted_total_radial_score_sd": weighted_sd(total_score, target_logw),
        "ce_replay_max_abs_diff": float(np.max(ce_diff)) if ce_diff.size else float("nan"),
        "ce_replay_mean_abs_diff": float(np.mean(ce_diff)) if ce_diff.size else float("nan"),
    }


def _format_radius(radius: float) -> str:
    return f"{float(radius):.2f}".replace(".", "p")


def _unit_rel_path(record: ReferenceRecord, radius: float) -> Path:
    return (
        Path(record.cell_id)
        / record.dataset_tag
        / f"ref_{int(record.ref_id):03d}"
        / f"r_{_format_radius(radius)}"
    )


def sample_unit(
    *,
    record: ReferenceRecord,
    radius: float,
    n_samples: int,
    seed: int,
    raw_root: Path,
    params: PMSAISParams,
    force: bool,
) -> dict[str, Any]:
    unit_dir = raw_root / _unit_rel_path(record, radius)
    summary_path = unit_dir / "unit_summary.json"
    samples_path = unit_dir / "samples.npz"
    if summary_path.exists() and samples_path.exists() and not force:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            payload["reused"] = True
            return payload

    ensure_dir(unit_dir)
    data = load_dataset(record.dataset_path)
    theta_ref = load_theta(record.theta_path)
    ref_static = _reference_static_fields(record=record, theta_ref=theta_ref, data=data)
    ref_norm = float(ref_static["theta_ref_norm"])
    ce_ref = float(ref_static["ce_ref"])
    err_ref = float(ref_static["err_ref"])
    mu = np.asarray(ref_static["mu"], dtype=np.float64)
    kappa = float(params.lambda_reg * float(radius) * ref_norm / math.sqrt(P))
    sampler = str(params.sampler).strip().lower()
    if sampler in {
        "smc",
        "ce_smc",
        "adaptive_ce_smc",
        "tempered_smc",
        "tpis",
        "tempered_path_importance_sampling",
        "exact_shell_l2_vmf_ce_smc",
        "exact_shell_l2_vmf_ce_tempered_smc",
        "exact_shell_l2_vmf_adaptive_ce_smc",
        "exact_shell_l2_vmf_adaptive_ce_tempered_smc",
    }:
        smc = _run_ce_smc_split(
            theta_ref=theta_ref,
            data=data,
            radius=float(radius),
            n_samples=int(n_samples),
            mu=mu,
            base_kappa=kappa,
            seed=int(seed),
            params=params,
        )
        directions = np.asarray(smc["directions"], dtype=np.float64)
        ce = np.asarray(smc["ce"], dtype=np.float64)
        err = np.asarray(smc["err"], dtype=np.float64)
        split = np.asarray(smc["split"], dtype=np.int32)
        target_logw = np.asarray(smc["target_logw"], dtype=np.float64)
        logz_ce = float(smc["logZ_CE"])
        split_logz_ce = [float(value) for value in smc["split_logZ"]]
        smc_payload = smc
        sampler_method = "exact_shell_l2_vmf_adaptive_ce_tempered_smc"
    else:
        rng = np.random.default_rng(int(seed))
        directions = sample_vmf(mu, kappa, int(n_samples), rng)
        theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * directions
        ce, err = ce_error_batch(
            theta_batch,
            data["X_train"],
            data["y"],
            chunk_size=int(params.chunk_size),
            device=str(params.device),
            dtype=str(params.dtype),
        )
        split = np.arange(int(n_samples), dtype=np.int32) % 2
        target_logw = -float(params.gamma_ce) * ce
        logz_ce = logmeanexp(target_logw)
        split_logz_ce = [logmeanexp(target_logw[split == split_id]) for split_id in (0, 1)]
        smc_payload = {}
        sampler_method = "prior_matched_vmf_direct_is"
    delta_ce = np.maximum(ce - float(ce_ref), 0.0)
    h = np.sqrt(2.0 * delta_ce)
    logw_ce_direct = -float(params.gamma_ce) * ce

    log_prefactor = -float(params.lambda_reg) * float(radius) * float(radius) / 2.0 + log_sphere_mgf(P, kappa)
    reference_prior_log_weight = -float(params.lambda_reg) * ref_norm * ref_norm / (2.0 * P)
    logz_inf_stripped = float(log_prefactor + logz_ce) if np.isfinite(logz_ce) else float("nan")
    logz_inf_full = float(logz_inf_stripped + reference_prior_log_weight)
    ess_inf = ess_from_logw(target_logw)
    split_logz = [float(log_prefactor + value) if np.isfinite(value) else float("nan") for value in split_logz_ce]
    split_logz_per_p_diff = float(abs(split_logz[0] - split_logz[1]) / P) if all(np.isfinite(split_logz)) else float("inf")

    h_rows: list[dict[str, Any]] = []
    for h_value in params.h_ladder:
        mask = h <= float(h_value)
        ratio, log_ratio, ess_num = weighted_ratio(target_logw, mask)
        logz_h = float(logz_inf_stripped + log_ratio) if np.isfinite(logz_inf_stripped) and np.isfinite(log_ratio) else float("-inf")
        h_rows.append(
            {
                "H": float(h_value),
                "logZ_H": logz_h,
                "R_H": ratio,
                "gate_count": int(np.sum(mask)),
                "ess_H": ess_num,
            }
        )

    gap_rows: list[dict[str, Any]] = []
    for h_value in params.h_ladder:
        ratio, log_ratio, ess_num = weighted_ratio(target_logw, h <= float(h_value))
        gap_rows.append(
            {
                "mode": "absolute_h",
                "threshold": float(h_value),
                "Z_ratio": ratio,
                "log_Z_ratio": log_ratio,
                "ess_num": ess_num,
                "gate_count": int(np.sum(h <= float(h_value))),
            }
        )
    for threshold in params.delta_ce_thresholds:
        ratio, log_ratio, ess_num = weighted_ratio(target_logw, delta_ce <= float(threshold))
        gap_rows.append(
            {
                "mode": "delta_ce",
                "threshold": float(threshold),
                "Z_ratio": ratio,
                "log_Z_ratio": log_ratio,
                "ess_num": ess_num,
                "gate_count": int(np.sum(delta_ce <= float(threshold))),
            }
        )
    rel_delta = delta_ce / max(float(ce_ref), 1.0e-12)
    for threshold in params.relative_delta_ce_thresholds:
        ratio, log_ratio, ess_num = weighted_ratio(target_logw, rel_delta <= float(threshold))
        gap_rows.append(
            {
                "mode": "relative_delta_ce_to_ref_ce",
                "threshold": float(threshold),
                "Z_ratio": ratio,
                "log_Z_ratio": log_ratio,
                "ess_num": ess_num,
                "gate_count": int(np.sum(rel_delta <= float(threshold))),
            }
        )

    projection = directions @ mu
    theta_norm_sq = ref_norm * ref_norm + P * float(radius) * float(radius) - 2.0 * math.sqrt(P) * float(radius) * ref_norm * projection
    l2_penalty = float(params.lambda_reg) * theta_norm_sq / (2.0 * P)
    weighted_ce_value = weighted_mean(ce, target_logw)
    weighted_l2_penalty = weighted_mean(l2_penalty, target_logw)
    weighted_projection = weighted_mean(projection, target_logw)
    np.savez_compressed(
        samples_path,
        ce=ce.astype(np.float64),
        error=err.astype(np.float64),
        h=h.astype(np.float64),
        delta_ce=delta_ce.astype(np.float64),
        logw_ce=target_logw.astype(np.float64),
        logw_target=target_logw.astype(np.float64),
        logw_ce_direct=logw_ce_direct.astype(np.float64),
        split=split.astype(np.int32),
        direction_projection=projection.astype(np.float64),
        theta_norm_sq=theta_norm_sq.astype(np.float64),
        l2_penalty=l2_penalty.astype(np.float64),
    )
    payload: dict[str, Any] = {
        "beta": float(record.beta),
        "cell_id": record.cell_id,
        "dataset_tag": record.dataset_tag,
        "dataset_id": int(record.dataset_id),
        "ref_id": int(record.ref_id),
        "radius": float(radius),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "sampler_method": sampler_method,
        "theta_path": str(record.theta_path),
        "dataset_path": str(record.dataset_path),
        "samples_path": str(samples_path),
        "ce_ref": float(ce_ref),
        "err_ref": float(err_ref),
        "theta_ref_norm": ref_norm,
        "theta_ref_norm_sq": ref_norm * ref_norm,
        "kappa": kappa,
        "logM": log_sphere_mgf(P, kappa),
        "log_prefactor": log_prefactor,
        "reference_prior_log_weight": reference_prior_log_weight,
        "logZ_CE": logz_ce,
        "logZ_inf_stripped": logz_inf_stripped,
        "logZ_inf_full": logz_inf_full,
        "logZ_inf": logz_inf_stripped,
        "ess_inf": ess_inf,
        "ess_frac": float(ess_inf / max(1, int(n_samples))),
        "weighted_ce": weighted_ce_value,
        "weighted_gamma_ce": float(params.gamma_ce) * weighted_ce_value,
        "unweighted_l2_penalty": float(np.mean(l2_penalty)),
        "weighted_l2_penalty": weighted_l2_penalty,
        "weighted_target_energy": float(params.gamma_ce) * weighted_ce_value + weighted_l2_penalty,
        "weighted_ce_l2_ratio": float((float(params.gamma_ce) * weighted_ce_value) / max(weighted_l2_penalty, 1.0e-300)),
        "weighted_direction_projection": weighted_projection,
        "unweighted_theta_norm_sq": float(np.mean(theta_norm_sq)),
        "weighted_theta_norm_sq": weighted_mean(theta_norm_sq, target_logw),
        "weighted_error": weighted_mean(err, target_logw),
        "weighted_accuracy": 1.0 - weighted_mean(err, target_logw),
        "weighted_h": weighted_mean(h, target_logw),
        "split0_logZ_inf": split_logz[0],
        "split1_logZ_inf": split_logz[1],
        "split_logZ_per_P_diff": split_logz_per_p_diff,
        "direction_mean_projection": float(np.mean(projection)),
        "direction_sd_projection": float(np.std(projection)),
        "h_rows": h_rows,
        "gap_rows": gap_rows,
        "raw_unit_dir": str(unit_dir),
        "reused": False,
    }
    if sampler_method in {"exact_shell_l2_vmf_adaptive_ce_smc", "exact_shell_l2_vmf_adaptive_ce_tempered_smc"}:
        payload.update(
            {
                "smc_completed": bool(smc_payload.get("completed", False)),
                "smc_step_count": int(smc_payload.get("smc_step_count", 0)),
                "smc_total_step_count": int(smc_payload.get("smc_total_step_count", 0)),
                "smc_resample_count": int(smc_payload.get("smc_resample_count", 0)),
                "smc_min_cess_fraction": float(smc_payload.get("smc_min_cess_fraction", float("nan"))),
                "smc_mean_cess_fraction": float(smc_payload.get("smc_mean_cess_fraction", float("nan"))),
                "smc_min_ess_fraction": float(smc_payload.get("smc_min_ess_fraction", float("nan"))),
                "smc_mean_mh_acceptance": float(smc_payload.get("smc_mean_mh_acceptance", float("nan"))),
                "smc_min_mh_acceptance": float(smc_payload.get("smc_min_mh_acceptance", float("nan"))),
                "smc_target_cess_fraction": float(params.smc_target_cess_fraction),
                "smc_resample_ess_fraction": float(params.smc_resample_ess_fraction),
                "smc_move_kappa": float(params.smc_move_kappa) if float(params.smc_move_kappa) > 0.0 else float(params.smc_move_kappa_factor) * float(P),
                "smc_histories": smc_payload.get("smc_histories", []),
            }
        )
    if bool(params.radial_derivative_enabled):
        payload.update(
            _radial_derivative_fields(
                theta_ref=theta_ref,
                data=data,
                radius=float(radius),
                directions=directions,
                ce=ce,
                target_logw=target_logw,
                split=split,
                params=params,
            )
        )
    save_json(summary_path, payload)
    return payload


def params_from_config(config: dict[str, Any]) -> PMSAISParams:
    target = config.get("target") or config.get("model") or {}
    sampling = config.get("sampling") or {}
    smc = config.get("smc") or config.get("tpis_controls") or sampling.get("smc") or {}
    derivative = config.get("derivative") or {}
    h_values = sampling.get("h_ladder") or config.get("H_ladder") or [1, 2, 4, 6, 8]
    finite_h = [float(value) for value in h_values if str(value).lower() not in {"inf", ".inf", "infinity"}]
    gap = config.get("loss_gap_ratios", {})
    method = str(
        config.get(
            "method",
            config.get(
                "method_name",
                config.get("sampler", sampling.get("method", sampling.get("sampler", "direct"))),
            ),
        )
    ).lower()
    proposal = str(sampling.get("proposal", "")).lower()
    uses_tempered_smc = any(token in method for token in ("smc", "temper", "tpis")) or any(
        token in proposal for token in ("smc", "temper", "tpis")
    )
    sampler = "exact_shell_l2_vmf_adaptive_ce_tempered_smc" if uses_tempered_smc else "direct"
    return PMSAISParams(
        gamma_ce=float(target.get("gamma_ce", target.get("gamma", 0.4))),
        lambda_reg=float(target.get("lambda_reg", target.get("lambda_l2_per_param", target.get("lambda_l2", 220.0)))),
        chunk_size=int(config.get("compute", {}).get("chunk_size", 64)),
        device=str(config.get("compute", {}).get("device", "cpu")),
        dtype=str(config.get("compute", {}).get("dtype", "float64")),
        h_ladder=tuple(finite_h),
        delta_ce_thresholds=tuple(float(x) for x in gap.get("delta_ce_thresholds", [0.25, 0.5, 1.0, 2.0, 4.0])),
        relative_delta_ce_thresholds=tuple(float(x) for x in gap.get("relative_delta_ce_thresholds", [1, 5, 10, 50, 100])),
        sampler=sampler,
        smc_target_cess_fraction=float(smc.get("target_cess_fraction", smc.get("cess_target", 0.70))),
        smc_resample_ess_fraction=float(smc.get("resample_ess_fraction", 0.70)),
        smc_resample_every_step=bool(smc.get("resample_every_step", False)),
        smc_max_steps=int(smc.get("max_steps", 200)),
        smc_min_delta_t=float(smc.get("min_delta_t", 1.0e-4)),
        smc_bisection_steps=int(smc.get("bisection_steps", 32)),
        smc_mh_sweeps=int(smc.get("mh_sweeps", 2)),
        smc_move_kappa=float(smc.get("move_kappa", 0.0)),
        smc_move_kappa_factor=float(smc.get("move_kappa_factor", 50.0)),
        radial_derivative_enabled=bool(derivative.get("enabled", False)),
        radial_derivative_chunk_size=int(derivative.get("chunk_size", 0) or config.get("compute", {}).get("derivative_chunk_size", 0) or 0),
    )
