from __future__ import annotations


DEFAULT_CONFIG = {
    "pipeline_id": "mnist10_eta_reference_search",
    "methodology_id": "mnist10_adam_exact_reference_v1",
    "run_name": "eta_reference_search_gapfill_0p02_0p05_0p15_0p25_30ref_cpu60_gpu0",
    "etas": [0.02, 0.05, 0.15, 0.25],
    "selected_refs_per_eta": 30,
    "max_attempts_per_eta": 240,
    "batch_size": 10,
    "max_epochs": 4200,
    "lr": 0.022,
    "device": "auto",
    "base_seed": 2026062500,
    "attempt_seed_starts": {
        "0.02": 2026164500,
        "0.05": 2026167500,
        "0.15": 2026277500,
        "0.25": 2026187500,
    },
    "canonical_ref_offset": 1,
    "force": False,
}
