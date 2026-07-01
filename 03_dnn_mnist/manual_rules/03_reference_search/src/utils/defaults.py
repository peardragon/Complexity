from __future__ import annotations


DEFAULT_CONFIG = {
    "pipeline_id": "mnist10_manual_rule_reference_search",
    "methodology_id": "mnist10_adam_exact_reference_manual_rules_v1",
    "rules": [
        "very_low_tv_spectral_teacher",
        "real_even_odd",
        "teacher_nn",
        "random_label",
    ],
    "selected_refs_per_rule": 30,
    "max_attempts_per_rule": 240,
    "batch_size": 10,
    "max_epochs": 4200,
    "lr": 0.022,
    "device": "auto",
    "canonical_ref_offset": 1,
    "attempt_seed_starts": {
        "rule_001": 2840000,
        "rule_002": 2701000,
        "rule_003": 2702000,
        "rule_004": 2703000,
        "very_low_tv_spectral_teacher": 2840000,
        "real_even_odd": 2701000,
        "teacher_nn": 2702000,
        "random_label": 2703000,
    },
    "resample_seed_offsets": {
        "rule_001": 2026061900,
        "rule_002": 2026061800,
        "rule_003": 2026061800,
        "rule_004": 2026061800,
    },
    "force": False,
}
