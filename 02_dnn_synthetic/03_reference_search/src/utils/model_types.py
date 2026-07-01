from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    series: str
    beta_ising: float
    rewire_p: float
    sweep_value: float
    display_label: str


@dataclass
class BudgetSpec:
    name: str
    datasets_per_cell: int
    refs_per_width: int
    train_epochs: int
    n_windows: int
    steps_per_window: int
    burn_in: int
    wham_bins: int
    ising_sweeps: int
    hmc_L: int
    hmc_eps: float
    k_spring: float
    d_min: float
    d_max: float
    swap_interval: int
    target_accept: float
    adapt_rate: float
    thinning: int
    sampling_dtype: str
    sampling_threads: int
    train_lr: float
    train_weight_decay: float
    train_momentum: float
    probe_steps: int
    probe_burn_in: int
    ref_restarts: int
    adam_epochs_ref: int
    lbfgs_max_iter: int
    start_top_k: int
    force_anchor_windows: bool
    small_d_window_exponent: float
    prep_steps: int
    prep_k: float
    trajectory_L_jitter: float
    entropic_compensation_strength: float
    entropic_compensation_dim: float


@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    momentum: float
    epochs: int
    seed: int
    optimizer_name: str = "adam"
    lbfgs_max_iter: int = 0
    init_scale_multiplier: float = 1.0
    activation: str = "softplus"
    loss: str = "logistic"
    margin: float = 1.0


@dataclass
class DNNArch:
    input_dim: int
    width1: int
    width2: int


@dataclass
class UmbrellaWindow:
    center: float
    k_spring: float


@dataclass
class DistanceSpec:
    mode: str = "plain_l2"
    eps: float = 1e-12


@dataclass
class REUSDiag:
    eps_by_window: List[float]
    accept_rate_by_window: List[float]
    mean_abs_dH_by_window: List[float]
    mean_dH_by_window: List[float]
    swap_attempts: int
    swap_accepts: int
    swap_accept_rate: float
    observed_d_min: float
    observed_d_max: float
    beta_swap_attempts: int = 0
    beta_swap_accepts: int = 0
    beta_swap_accept_rate: float = 0.0


@dataclass
class WHAMResult:
    d_centers: np.ndarray
    log_p: np.ndarray
    f: np.ndarray
    converged: bool
    n_iter: int
    max_delta_f: float
    bin_width: float
    dropped_windows: List[int]


@dataclass
class RunConfig:
    stage: str
    budget: str
    results_dir: str
    cells: List[str]
    widths: List[int]
    seed: int
    force: bool
    sampling_device: str
    datasets_per_cell: int
    refs_per_width: int
    train_epochs: int
    n_windows: int
    steps_per_window: int
    burn_in: int
    wham_bins: int
    n_points: int
    input_dim: int
    k_graph: int
    nmstv_scales: List[float]
    rewire_mode: str
    ising_sweeps: int
    train_lr: float
    train_weight_decay: float
    train_momentum: float
    activation: str
    loss_name: str
    ref_loss_name: str
    margin: float
    gibbs_beta: float
    dist_mode: str
    hmc_L: int
    hmc_eps: float
    k_spring: float
    d_min: float
    d_max: float
    swap_interval: int
    target_accept: float
    adapt_rate: float
    thinning: int
    sampling_dtype: str
    sampling_threads: int
    probe_steps: int
    probe_burn_in: int
    ref_restarts: int
    ref_target_valid_count: int
    ref_rescue_enabled: bool
    ref_rescue_topk: int
    ref_rescue_subspace_dim: int
    ref_rescue_center_limit: int
    ref_rescue_attempt_budget: int
    ref_rescue_shgo_maxev: int
    ref_rescue_continuation_loss: str
    ref_rescue_continuation_lr: float
    ref_rescue_continuation_steps: int
    ref_rescue_dedup_scale: float
    adam_epochs_ref: int
    lbfgs_max_iter: int
    init_probe_restarts: int
    sample_width_mode: str
    allow_fake_references: bool
    fallback_reference_min_train_accuracy: float
    enable_line_anchors: bool
    line_anchor_splits: int
    samplers: List[str]
    start_top_k: int
    force_anchor_windows: bool
    small_d_window_exponent: float
    prep_steps: int
    prep_k: float
    trajectory_L_jitter: float
    entropic_compensation_strength: float
    entropic_compensation_dim: float


@dataclass
class ReferenceAttemptSummary:
    attempt_id: int
    is_exact_solution: bool
    reference_status: str
    sampler_eligible: bool
    optimizer_chain: str
    final_train_loss: float
    final_cls_err: float
    final_train_accuracy: float


@dataclass
class SelectedReference:
    ref_id: int
    attempt_id: int
    theta_path: str
    theta_init_path: str
    summary_path: str
    reference_status: str
    sampler_eligible: bool


__all__ = [
    "BudgetSpec",
    "CellSpec",
    "DistanceSpec",
    "DNNArch",
    "REUSDiag",
    "ReferenceAttemptSummary",
    "RunConfig",
    "SelectedReference",
    "TrainConfig",
    "UmbrellaWindow",
    "WHAMResult",
]
