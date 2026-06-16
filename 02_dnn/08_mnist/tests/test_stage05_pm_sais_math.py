from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "02_dnn" / "08_mnist" / "src"
sys.path.insert(0, str(SRC))

from mnist14_model import P
from mnist14_vmf import log_sphere_mgf, sample_vmf


def test_vmf_samples_are_unit_norm_and_logm_zero() -> None:
    rng = np.random.default_rng(123)
    mu = rng.normal(size=P)
    samples = sample_vmf(mu, 0.0, 32, rng)
    assert samples.shape == (32, P)
    assert np.max(np.abs(np.linalg.norm(samples, axis=1) - 1.0)) < 1.0e-10
    assert log_sphere_mgf(P, 0.0) == 0.0


def test_hard_shell_formula_distance() -> None:
    rng = np.random.default_rng(456)
    theta_ref = rng.normal(size=P)
    u = sample_vmf(-theta_ref / np.linalg.norm(theta_ref), 1.5, 16, rng)
    radius = 0.45
    theta = theta_ref[None, :] + math.sqrt(P) * radius * u
    d = np.linalg.norm(theta - theta_ref[None, :], axis=1) / math.sqrt(P)
    assert np.max(np.abs(d - radius)) < 1.0e-10
