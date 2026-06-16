from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "02_dnn" / "08_mnist" / "src"
sys.path.insert(0, str(SRC))

from mnist14_model import P, ce_and_error_np, flatten_parts, init_parts, init_theta, unpack_theta


def test_model_param_count_and_roundtrip() -> None:
    assert P == 3441
    theta = init_theta(123)
    parts = unpack_theta(theta)
    assert flatten_parts(parts).shape == (P,)
    assert np.allclose(flatten_parts(parts), theta)


def test_ce_and_error_are_finite() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 196))
    y = np.where(rng.random(12) > 0.5, 1, -1)
    theta = flatten_parts(init_parts(rng))
    ce, err = ce_and_error_np(theta, x, y)
    assert np.isfinite(ce)
    assert 0.0 <= err <= 1.0
