from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "02_dnn" / "08_mnist" / "runs" / "smoke" / "01_dataset_prepare"


def test_stage01_dataset_payloads() -> None:
    index_path = RUN_DIR / "dataset_index.csv"
    assert index_path.exists()
    index = pd.read_csv(index_path)
    assert len(index) == 9
    assert sorted(index["rule"].unique().tolist()) == ["random_label", "real_even_odd", "teacher_nn"]
    for row in index.to_dict("records"):
        path = REPO_ROOT / row["dataset_path"]
        assert path.exists()
        payload = np.load(path)
        assert payload["X_train"].shape == (256, 196)
        assert payload["X_test"].shape == (2048, 196)
        assert payload["X_train_raw14"].shape == (256, 196)
        assert payload["X_test_raw14"].shape == (2048, 196)
        assert payload["X_train"].dtype == np.float32
        assert payload["y_train"].dtype == np.int8
        assert set(np.unique(payload["y_train"]).tolist()) == {-1, 1}
        assert np.isfinite(payload["X_train"]).all()
        balance = float(np.mean(payload["y_train"] == 1))
        assert 0.45 <= balance <= 0.55
    assert (RUN_DIR / "figures" / "fig01_mnist_28_vs_14_montage.png").exists()
