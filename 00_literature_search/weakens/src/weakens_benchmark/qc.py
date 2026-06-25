from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .importance import ImportanceResult, normalized_weights
from .landscape import ProxyLandscape
from .samplers import SampleResult


def _autocorr_ess(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    if n < 4 or float(np.var(x)) <= 0.0:
        return float(n)
    x = x - float(np.mean(x))
    corr = np.correlate(x, x, mode="full")[n - 1 :]
    corr = corr / max(corr[0], 1.0e-300)
    tau = 1.0
    for k in range(1, min(n, 1000)):
        if corr[k] <= 0.0:
            break
        tau += 2.0 * float(corr[k])
    return float(max(1.0, min(n, n / tau)))


def summarize_chain(
    landscape: ProxyLandscape,
    result: SampleResult,
    truth_mass: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masks = landscape.region_mask(result.samples)
    sample_count = max(1, result.samples.shape[0])
    est_mass = np.mean(masks, axis=0) if result.samples.size else np.zeros(len(truth_mass))
    hits = np.sum(masks, axis=0).astype(int) if result.samples.size else np.zeros(len(truth_mass), dtype=int)
    region_rows = []
    for idx, region in enumerate(landscape.region_names()):
        region_rows.append(
            {
                "method": result.name,
                "region": region,
                "truth_mass": float(truth_mass[idx]),
                "estimated_mass": float(est_mass[idx]),
                "hit_count": int(hits[idx]),
                "weighted": False,
            }
        )
    important = truth_mass >= float(config["qc"]["min_truth_region_mass"])
    l1 = float(np.sum(np.abs(est_mass[important] - truth_mass[important])))
    ess = _autocorr_ess(result.energies)
    ess_fraction = float(ess / sample_count)
    covered = int(np.sum(hits[important] >= int(config["qc"]["min_region_hits"])))
    all_covered = bool(covered == int(np.sum(important)))
    pass_qc = bool(
        all_covered
        and l1 <= float(config["qc"]["max_region_l1_error"])
        and ess_fraction >= float(config["qc"]["min_ess_fraction"])
    )
    summary = {
        "method": result.name,
        "method_role": result.metadata.get("method_role", ""),
        "attempted_sample_count": int(result.metadata.get("attempted_sample_count", result.samples.shape[0])),
        "elapsed_seconds": result.metadata.get("elapsed_seconds"),
        "sample_count": int(result.samples.shape[0]),
        "accept_rate": result.accept_rate,
        "ess_fraction": ess_fraction,
        "important_regions": int(np.sum(important)),
        "covered_important_regions": covered,
        "region_l1_error": l1,
        "pass_qc": pass_qc,
        "failure_reason": ""
        if pass_qc
        else _failure_reason(all_covered, l1, ess_fraction, config),
    }
    return region_rows, summary


def summarize_importance(
    landscape: ProxyLandscape,
    result: ImportanceResult,
    truth_mass: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masks = landscape.region_mask(result.samples)
    weights = normalized_weights(result.log_weights)
    est_mass = weights @ masks.astype(np.float64)
    hits = np.sum(masks, axis=0).astype(int)
    region_rows = []
    for idx, region in enumerate(landscape.region_names()):
        region_rows.append(
            {
                "method": "vmf_l2_final",
                "region": region,
                "truth_mass": float(truth_mass[idx]),
                "estimated_mass": float(est_mass[idx]),
                "hit_count": int(hits[idx]),
                "weighted": True,
            }
        )
    important = truth_mass >= float(config["qc"]["min_truth_region_mass"])
    l1 = float(np.sum(np.abs(est_mass[important] - truth_mass[important])))
    covered = int(np.sum(hits[important] >= int(config["qc"]["min_region_hits"])))
    all_covered = bool(covered == int(np.sum(important)))
    pass_qc = bool(
        all_covered
        and l1 <= float(config["qc"]["max_region_l1_error"])
        and result.ess_fraction >= float(config["qc"]["min_ess_fraction"])
    )
    summary = {
        "method": "vmf_l2_final",
        "method_role": result.metadata.get("method_role", ""),
        "attempted_sample_count": int(result.metadata.get("attempted_sample_count", result.samples.shape[0])),
        "elapsed_seconds": result.metadata.get("elapsed_seconds"),
        "sample_count": int(result.samples.shape[0]),
        "accept_rate": None,
        "ess_fraction": float(result.ess_fraction),
        "important_regions": int(np.sum(important)),
        "covered_important_regions": covered,
        "region_l1_error": l1,
        "pass_qc": pass_qc,
        "failure_reason": ""
        if pass_qc
        else _failure_reason(all_covered, l1, result.ess_fraction, config),
    }
    return region_rows, summary


def _failure_reason(
    all_covered: bool,
    l1: float,
    ess_fraction: float,
    config: dict[str, Any],
) -> str:
    reasons = []
    if not all_covered:
        reasons.append("missing_important_region")
    if l1 > float(config["qc"]["max_region_l1_error"]):
        reasons.append("region_mass_l1_error_high")
    if ess_fraction < float(config["qc"]["min_ess_fraction"]):
        reasons.append("ess_fraction_low")
    return ",".join(reasons)


def build_qc_tables(
    landscape: ProxyLandscape,
    baseline_results: dict[str, SampleResult],
    importance_result: ImportanceResult,
    truth_mass: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for result in baseline_results.values():
        rows, summary = summarize_chain(landscape, result, truth_mass, config)
        region_rows.extend(rows)
        summaries.append(summary)
    rows, summary = summarize_importance(landscape, importance_result, truth_mass, config)
    region_rows.extend(rows)
    summaries.append(summary)
    return pd.DataFrame(region_rows), pd.DataFrame(summaries)
