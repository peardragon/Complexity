from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "02_dnn" / "08_mnist" / "runs" / "smoke" / "04_exact_reference_search"


def test_stage04_reference_index_and_payloads() -> None:
    index = pd.read_csv(RUN_DIR / "reference_index.csv")
    assert len(index) == 45
    assert (index["train_error"] == 0.0).all()
    counts = index.groupby(["split_id", "rule"]).size()
    assert counts.min() == 5
    theta_rows = []
    for row in index.to_dict("records"):
        theta_path = REPO_ROOT / row["theta_path"]
        assert theta_path.exists()
        theta = np.load(theta_path).reshape(-1)
        assert theta.size == 3441
        assert np.isfinite(theta).all()
        theta_rows.append(theta)
    for (_, rule), sub in index.groupby(["split_id", "rule"]):
        rows = [np.load(REPO_ROOT / row["theta_path"]).reshape(-1) for row in sub.to_dict("records")]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                assert np.linalg.norm(rows[i] - rows[j]) > 1.0e-6
