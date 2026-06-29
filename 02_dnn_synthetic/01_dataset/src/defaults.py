from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    series: str
    beta_ising: float
    rewire_p: float
    sweep_value: float
    display_label: str


BETA_SERIES = [0.10, 0.20, 0.35, 0.60, 1.00, 1.60]
P_SERIES = [0.00, 0.03, 0.07, 0.15, 0.30, 0.50]
VALID_DIMS = [2]

DEFAULT_CONFIG = {
    "pipeline_id": "synthetic_dataset",
    "methodology_id": "ws_ising_dataset_v1",
    "seed": 0,
    "n_points": 512,
    "datasets_per_cell": 30,
    "k_graph": 10,
    "rewire_mode": "degree_preserve",
    "ising_sweeps": 2000,
    "nmstv_scales": [0.5, 1.0, 2.0, 4.0],
    "beta_series": BETA_SERIES,
    "p_series": P_SERIES,
    "reuse_duplicate_cell_datasets": True,
}


def make_cell_id(beta_ising: float, rewire_p: float) -> str:
    beta_tag = f"{beta_ising:.2f}".replace(".", "p")
    p_tag = f"{rewire_p:.2f}".replace(".", "p")
    return f"cell_beta_{beta_tag}_p_{p_tag}"


def build_cell_specs(*, beta_series: list[float] | None = None, p_series: list[float] | None = None) -> List[CellSpec]:
    beta_values = BETA_SERIES if beta_series is None else list(beta_series)
    p_values = P_SERIES if p_series is None else list(p_series)
    cells = [
        CellSpec(
            cell_id=make_cell_id(beta, 0.0),
            series="beta",
            beta_ising=float(beta),
            rewire_p=0.0,
            sweep_value=float(beta),
            display_label=f"beta={beta:.2f}, p=0.00",
        )
        for beta in beta_values
    ]
    cells.extend(
        [
            CellSpec(
                cell_id=make_cell_id(0.60, p),
                series="p",
                beta_ising=0.60,
                rewire_p=float(p),
                sweep_value=float(p),
                display_label=f"beta=0.60, p={p:.2f}",
            )
            for p in p_values
        ]
    )
    return cells


CELL_SPECS = build_cell_specs()


def dataset_seed(base_seed: int, cell_index: int, dataset_id: int) -> int:
    return int(base_seed + 1000 * cell_index + 100000 * int(dataset_id) + 1234)


def cell_specs_as_dicts() -> list[dict]:
    return [asdict(cell) for cell in CELL_SPECS]
