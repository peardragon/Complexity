from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .dnn_model import P, ce_and_error, ce_error_batch, ce_radial_grad_batch
from .loaders import load_dataset, load_theta, project_relative, resolve_existing_path
from .vmf import log_sphere_mgf, sample_vmf, sample_vmf_batch


SAMPLING_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = SAMPLING_ROOT / "config" / "default.json"
DERIVATIVE_METHODOLOGY_ID = "mnist10_exact_shell_l2_vmf_ce_tempered_smc_radial_score_derivative_v1"
TEMPERED_SMC_METHOD = "exact_shell_l2_vmf_adaptive_ce_tempered_smc"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def load_config() -> dict[str, Any]:
    import json

    cfg = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    sampling = dict(cfg.get("sampling") or {})
    if "radii" not in sampling:
        start = float(sampling.get("radius_start", 0.01))
        stop = float(sampling.get("radius_stop", 1.0))
        step = float(sampling.get("radius_step", 0.01))
        count = int(round((stop - start) / step)) + 1
        sampling["radii"] = [round(start + idx * step, 10) for idx in range(count)]
    sampling.setdefault("lambda_reg", 1.0)
    sampling.setdefault("fallback_policies_enabled", False)
    sampling.setdefault("radial_derivative_enabled", bool(sampling.get("radial_derivative_enabled", True)))
    cfg["sampling"] = sampling
    cfg.setdefault("dataset", {})
    cfg["dataset"].setdefault("n_train", 512)
    cfg["dataset"].setdefault("input_dim", 100)
    cfg.setdefault("smc", {})
    cfg["smc"].setdefault("target_cess_fraction", 0.95)
    cfg["smc"].setdefault("resample_ess_fraction", 0.50)
    cfg["smc"].setdefault("max_steps", 180)
    cfg["smc"].setdefault("min_delta_t", 0.0001)
    cfg["smc"].setdefault("bisection_steps", 32)
    cfg["smc"].setdefault("mh_sweeps", 2)
    cfg["smc"].setdefault("move_kappa_factor", 80.0)
    cfg.setdefault("compute", {})
    cfg["compute"].setdefault("chunk_size", 256)
    cfg["compute"].setdefault("derivative_chunk_size", 64)
    cfg["compute"].setdefault("device", "cpu")
    cfg["compute"].setdefault("dtype", "float32")
    cfg.setdefault("outputs", {})
    cfg["outputs"].setdefault("run_root", project_relative(SAMPLING_ROOT / "raw_outputs" / "shell_pool"))
    cfg.setdefault("reference_search", {})
    cfg["reference_search"].setdefault("selected_refs_per_dataset", 30)
    cfg.setdefault("qc", {})
    cfg["python"] = sys.executable
    cfg["resolved_at_unix"] = time.time()
    return cfg


def _normalise_logw(logw: np.ndarray) -> np.ndarray:
    return np.asarray(logw, dtype=np.float64) - logsumexp(logw)


def _ess_fraction(logw_norm: np.ndarray) -> float:
    logw_norm = np.asarray(logw_norm, dtype=np.float64)
    return float(np.exp(-logsumexp(2.0 * logw_norm)) / max(1, logw_norm.size))


def _weighted_mean(values: np.ndarray, logw: np.ndarray) -> float:
    weights = np.exp(np.asarray(logw, dtype=np.float64) - logsumexp(logw))
    return float(np.sum(weights * np.asarray(values, dtype=np.float64)))


def _weighted_sd(values: np.ndarray, logw: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.exp(np.asarray(logw, dtype=np.float64) - logsumexp(logw))
    mean = float(np.sum(weights * values))
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2)))


def _cess_fraction(logw_norm: np.ndarray, ce: np.ndarray, delta_t: float, gamma_ce: float) -> float:
    loga = -float(delta_t) * float(gamma_ce) * np.asarray(ce, dtype=np.float64)
    return float(np.exp(2.0 * logsumexp(logw_norm + loga) - logsumexp(logw_norm + 2.0 * loga)))


def _choose_temperature(t: float, ce: np.ndarray, logw_norm: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    target = float(cfg["smc"]["target_cess_fraction"])
    gamma_ce = float(cfg["dataset"]["n_train"])
    full = _cess_fraction(logw_norm, ce, 1.0 - t, gamma_ce)
    if full >= target:
        return 1.0, full
    low, high = float(t), 1.0
    for _ in range(int(cfg["smc"]["bisection_steps"])):
        mid = 0.5 * (low + high)
        val = _cess_fraction(logw_norm, ce, mid - t, gamma_ce)
        if val >= target:
            low = mid
        else:
            high = mid
    out = min(1.0, max(low, t + float(cfg["smc"]["min_delta_t"])))
    return out, _cess_fraction(logw_norm, ce, out - t, gamma_ce)


def _systematic_resample(logw_norm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = np.exp(_normalise_logw(logw_norm))
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cdf, positions, side="left")


def _rejuvenate(
    directions: np.ndarray,
    ce: np.ndarray,
    err: np.ndarray,
    theta_ref: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    mu: np.ndarray,
    base_kappa: float,
    t: float,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    move_kappa = float(cfg["smc"]["move_kappa_factor"]) * P
    proposal = sample_vmf_batch(directions, move_kappa, rng)
    theta_prop = theta_ref[None, :] + math.sqrt(P) * float(radius) * proposal
    ce_prop, err_prop = ce_error_batch(
        theta_prop,
        x,
        y,
        chunk_size=int(cfg["compute"]["chunk_size"]),
        device=str(cfg["compute"]["device"]),
        dtype=str(cfg["compute"]["dtype"]),
    )
    current_proj = directions @ mu
    prop_proj = proposal @ mu
    log_accept = -float(t) * float(cfg["dataset"]["n_train"]) * (ce_prop - ce) + float(base_kappa) * (prop_proj - current_proj)
    accept = np.log(rng.random(size=ce.size)) <= np.minimum(0.0, log_accept)
    if np.any(accept):
        directions[accept] = proposal[accept]
        ce[accept] = ce_prop[accept]
        err[accept] = err_prop[accept]
    return directions, ce, err, float(np.mean(accept))


def _radial_derivative_fields(
    theta_ref: np.ndarray,
    data: dict[str, np.ndarray],
    radius: float,
    directions: np.ndarray,
    ce: np.ndarray,
    target_logw: np.ndarray,
    split: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * directions
    replay_ce, radial_grad_ce = ce_radial_grad_batch(
        theta_batch,
        directions,
        data["X_train"],
        data["y_train"],
        chunk_size=int(cfg["compute"].get("derivative_chunk_size") or cfg["compute"]["chunk_size"]),
        device=str(cfg["compute"]["device"]),
        dtype=str(cfg["compute"]["dtype"]),
    )
    theta_ref_dot_u_over_sqrt_p = (directions @ theta_ref) / math.sqrt(P)
    prior_radial_score = -float(cfg["sampling"]["lambda_reg"]) * theta_ref_dot_u_over_sqrt_p
    ce_radial_score = -float(cfg["dataset"]["n_train"]) * math.sqrt(P) * radial_grad_ce
    variable_score = prior_radial_score + ce_radial_score
    total_score = -float(cfg["sampling"]["lambda_reg"]) * float(radius) + variable_score
    split_values = []
    for split_id in (0, 1):
        mask = np.asarray(split, dtype=np.int32) == split_id
        split_values.append(_weighted_mean(total_score[mask], target_logw[mask]) if np.any(mask) else float("nan"))
    split_diff = float(abs(split_values[0] - split_values[1]) / P) if np.all(np.isfinite(split_values)) else float("inf")
    ce_diff = np.abs(np.asarray(replay_ce, dtype=np.float64) - np.asarray(ce, dtype=np.float64))
    dlogz = _weighted_mean(total_score, target_logw)
    return {
        "radial_derivative_methodology_id": DERIVATIVE_METHODOLOGY_ID,
        "dlogZ_inf_dr": dlogz,
        "dlogZ_inf_stripped_dr": dlogz,
        "dlogZ_inf_full_dr": dlogz,
        "dlogZ_dr_split0": split_values[0],
        "dlogZ_dr_split1": split_values[1],
        "split_dlogZ_dr_per_P_diff": split_diff,
        "weighted_prior_radial_score": _weighted_mean(prior_radial_score, target_logw),
        "weighted_ce_radial_score": _weighted_mean(ce_radial_score, target_logw),
        "weighted_variable_radial_score": _weighted_mean(variable_score, target_logw),
        "weighted_total_radial_score": dlogz,
        "weighted_radial_grad_ce": _weighted_mean(radial_grad_ce, target_logw),
        "weighted_total_radial_score_sd": _weighted_sd(total_score, target_logw),
        "ce_replay_max_abs_diff": float(np.max(ce_diff)) if ce_diff.size else float("nan"),
        "ce_replay_mean_abs_diff": float(np.mean(ce_diff)) if ce_diff.size else float("nan"),
    }


def run_smc_split(
    theta_ref: np.ndarray,
    ds: dict[str, np.ndarray],
    radius: float,
    n_samples: int,
    lambda_reg: float,
    seed: int,
    cfg: dict[str, Any],
    reference_ce: float,
) -> dict[str, Any]:
    theta_ref = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    ref_norm = float(np.linalg.norm(theta_ref))
    if not np.isfinite(ref_norm) or ref_norm <= 0.0:
        raise ValueError("reference theta has zero or non-finite norm")
    mu = -theta_ref / ref_norm
    base_kappa = float(lambda_reg * float(radius) * ref_norm / math.sqrt(P))
    gamma_ce = float(cfg["dataset"]["n_train"])
    split_outputs: list[dict[str, Any]] = []
    for split_idx, n_particles in enumerate([n_samples // 2, n_samples - n_samples // 2]):
        rng = np.random.default_rng(int(seed) + 7919 * (split_idx + 1))
        directions = sample_vmf(mu, base_kappa, int(n_particles), rng)
        theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * directions
        ce, err = ce_error_batch(
            theta_batch,
            ds["X_train"],
            ds["y_train"],
            chunk_size=int(cfg["compute"]["chunk_size"]),
            device=str(cfg["compute"]["device"]),
            dtype=str(cfg["compute"]["dtype"]),
        )
        logw_norm = np.full(int(n_particles), -math.log(int(n_particles)), dtype=np.float64)
        t = 0.0
        logz_ce = 0.0
        history: list[dict[str, Any]] = []
        completed = True
        for step in range(int(cfg["smc"]["max_steps"])):
            if t >= 1.0 - 1.0e-12:
                break
            t_new, cess = _choose_temperature(t, ce, logw_norm, cfg)
            delta_t = max(0.0, t_new - t)
            loga = -delta_t * gamma_ce * ce
            logz_ce += float(logsumexp(logw_norm + loga))
            logw_norm = _normalise_logw(logw_norm + loga)
            ess_after = _ess_fraction(logw_norm)
            resampled = ess_after < float(cfg["smc"]["resample_ess_fraction"])
            if resampled:
                idx = _systematic_resample(logw_norm, rng)
                directions = directions[idx].copy()
                ce = ce[idx].copy()
                err = err[idx].copy()
                logw_norm = np.full(int(n_particles), -math.log(int(n_particles)), dtype=np.float64)
                ess_after = 1.0
            acc = float("nan")
            for _ in range(int(cfg["smc"]["mh_sweeps"])):
                directions, ce, err, acc = _rejuvenate(
                    directions,
                    ce,
                    err,
                    theta_ref,
                    ds["X_train"],
                    ds["y_train"],
                    radius,
                    mu,
                    base_kappa,
                    t_new,
                    cfg,
                    rng,
                )
            history.append({"step": step + 1, "t_start": t, "t_end": t_new, "cess_fraction": cess, "ess_fraction_after_reweight": ess_after, "resampled": resampled, "mh_acceptance": acc})
            t = t_new
        else:
            completed = t >= 1.0 - 1.0e-12
        split_outputs.append({"logZ_CE": logz_ce if completed else float("nan"), "ce": ce, "err": err, "directions": directions, "logw_norm": _normalise_logw(logw_norm), "history": history, "completed": completed})

    logz_values = np.asarray([float(s["logZ_CE"]) for s in split_outputs], dtype=np.float64)
    counts = np.asarray([len(split_outputs[0]["ce"]), len(split_outputs[1]["ce"])], dtype=np.float64)
    logz_ce = float(logsumexp(np.log(counts / np.sum(counts)) + logz_values)) if np.all(np.isfinite(logz_values)) else float("nan")
    log_prefactor = -float(lambda_reg) * float(radius) * float(radius) / 2.0 + log_sphere_mgf(P, base_kappa)
    logz_stripped = float(log_prefactor + logz_ce) if np.isfinite(logz_ce) else float("nan")
    reference_prior_log_weight = -float(lambda_reg) * ref_norm * ref_norm / (2.0 * P)
    ce = np.concatenate([s["ce"] for s in split_outputs])
    err = np.concatenate([s["err"] for s in split_outputs])
    logw = np.concatenate([math.log(counts[i] / np.sum(counts)) + split_outputs[i]["logw_norm"] for i in range(2)])
    logw = _normalise_logw(logw)
    dirs = np.concatenate([s["directions"] for s in split_outputs], axis=0)
    split = np.concatenate([np.full(len(split_outputs[i]["ce"]), i, dtype=np.int32) for i in range(2)])
    flat_history = [h for s in split_outputs for h in s["history"]]
    mh_acceptances = np.asarray([hrow["mh_acceptance"] for hrow in flat_history], dtype=np.float64)
    mh_acceptances = mh_acceptances[np.isfinite(mh_acceptances)]
    h = np.sqrt(2.0 * np.maximum(ce - float(reference_ce), 0.0))
    theta_batch = theta_ref[None, :] + math.sqrt(P) * float(radius) * dirs
    theta_norm_sq = np.sum(theta_batch * theta_batch, axis=1)
    l2_penalty = float(lambda_reg) * theta_norm_sq / (2.0 * P)
    direction_projection = dirs @ mu
    payload = {
        "logZ": logz_stripped,
        "logZ_CE": logz_ce,
        "logZ_inf_stripped": logz_stripped,
        "reference_prior_log_weight": reference_prior_log_weight,
        "logZ_inf_full": float(logz_stripped + reference_prior_log_weight) if np.isfinite(logz_stripped) else float("nan"),
        "split0_logZ": float(log_prefactor + logz_values[0]) if np.isfinite(logz_values[0]) else float("nan"),
        "split1_logZ": float(log_prefactor + logz_values[1]) if np.isfinite(logz_values[1]) else float("nan"),
        "split_logZ_per_P_diff": float(abs(logz_values[0] - logz_values[1]) / P) if np.all(np.isfinite(logz_values)) else float("inf"),
        "ess": float(_ess_fraction(logw) * max(1, logw.size)),
        "ess_fraction": _ess_fraction(logw),
        "weighted_ce": _weighted_mean(ce, logw),
        "weighted_error": _weighted_mean(err, logw),
        "weighted_h": _weighted_mean(h, logw),
        "smc_completed": bool(all(s["completed"] for s in split_outputs)),
        "smc_step_count": int(max(len(s["history"]) for s in split_outputs)),
        "smc_total_step_count": int(sum(len(s["history"]) for s in split_outputs)),
        "smc_min_cess_fraction": float(np.min([hrow["cess_fraction"] for hrow in flat_history])) if flat_history else float("nan"),
        "smc_mean_mh_acceptance": float(np.mean(mh_acceptances)) if mh_acceptances.size else 0.0,
        "hard_shell_distance_max_abs_err": float(np.max(np.abs(np.linalg.norm(theta_batch - theta_ref[None, :], axis=1) / math.sqrt(P) - float(radius)))),
        "direction_unit_norm_max_abs_err": float(np.max(np.abs(np.linalg.norm(dirs, axis=1) - 1.0))),
        "kappa": base_kappa,
        "logM": log_sphere_mgf(P, base_kappa),
        "log_prefactor": log_prefactor,
        "_samples_npz": {
            "ce": ce,
            "error": err,
            "h": h,
            "logw_target": logw,
            "split": split,
            "direction_projection": direction_projection,
            "theta_norm_sq": theta_norm_sq,
            "l2_penalty": l2_penalty,
        },
    }
    if bool(cfg.get("sampling", {}).get("radial_derivative_enabled", False)):
        payload.update(_radial_derivative_fields(theta_ref, ds, radius, dirs, ce, logw, split, cfg))
    return payload


def fallback_policy_for(_rule: str, _radius: float, _ref_id: int | None = None) -> dict[str, Any] | None:
    return None


def cfg_for_fallback_policy(base_cfg: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    cfg = {**base_cfg, "smc": dict(base_cfg["smc"])}
    cfg["smc"]["target_cess_fraction"] = float(policy["target_cess_fraction"])
    cfg["smc"]["mh_sweeps"] = int(policy["mh_sweeps"])
    cfg["smc"]["move_kappa_factor"] = float(policy["move_kappa_factor"])
    cfg["smc"]["max_steps"] = int(policy["max_steps"])
    return cfg


def run_replicated_smc(
    row: dict[str, Any],
    radius: float,
    cfg: dict[str, Any],
    *,
    n_samples_each: int,
    replicates: int,
    lambda_reg: float,
    seed: int,
) -> dict[str, Any]:
    ds = load_dataset(row["dataset_path"])
    theta_ref = load_theta(row["theta_path"])
    rows = [
        run_smc_split(theta_ref, ds, radius, int(n_samples_each), lambda_reg, int(seed) + rep * 1000003, cfg, float(row["CE_mean_train"]))
        for rep in range(int(replicates))
    ]
    full = np.asarray([row["logZ_inf_full"] for row in rows], dtype=np.float64)
    weights = _normalise_logw(full)
    return {
        "split_id": int(row.get("split_id", 0)),
        "rule": str(row["rule"]),
        "ref_id": int(row["ref_id"]),
        "radius": float(radius),
        "replicates": int(replicates),
        "n_samples_each": int(n_samples_each),
        "n_samples_total": int(replicates) * int(n_samples_each),
        "lambda_reg": float(lambda_reg),
        "seed": int(seed),
        "theta_path": str(row["theta_path"]),
        "dataset_path": str(row["dataset_path"]),
        "theta_ref_norm": float(np.linalg.norm(theta_ref)),
        "sampler_method": "replicated_exact_shell_l2_vmf_adaptive_ce_tempered_smc",
        "logZ": float(logsumexp([row["logZ"] for row in rows]) - math.log(len(rows))),
        "logZ_inf_full": float(logsumexp(full) - math.log(len(full))),
        "split_logZ_per_P_diff": float(max(row["split_logZ_per_P_diff"] for row in rows)),
        "ess_fraction": float(np.sum(np.exp(weights) * np.asarray([row["ess_fraction"] for row in rows], dtype=np.float64))),
        "weighted_ce": float(np.sum(np.exp(weights) * np.asarray([row["weighted_ce"] for row in rows], dtype=np.float64))),
        "weighted_error": float(np.sum(np.exp(weights) * np.asarray([row["weighted_error"] for row in rows], dtype=np.float64))),
        "smc_completed": bool(all(row["smc_completed"] for row in rows)),
        "replicate_summaries": rows,
    }
