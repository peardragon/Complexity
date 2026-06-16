from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "02_dnn" / "08_mnist" / "runs" / "smoke" / "02_complexity_measure"


def test_stage02_complexity_tables() -> None:
    dataset = pd.read_csv(RUN_DIR / "complexity_by_dataset.csv")
    summary = pd.read_csv(RUN_DIR / "complexity_by_rule_summary.csv")
    graph = pd.read_csv(RUN_DIR / "graph_stats_by_dataset_k.csv")
    assert len(dataset) == 9
    assert len(summary) == 3
    assert len(graph) == 27
    assert graph["edge_count"].min() > 0
    assert np.isfinite(graph[["tv", "nmstv", "sigma_k"]].to_numpy()).all()
    assert (RUN_DIR / "figures" / "fig01_nmstv_by_rule_boxplot.png").exists()
