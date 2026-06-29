from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STAGE_ROOT = Path(__file__).resolve().parents[1]
MANUAL_ROOT = STAGE_ROOT.parent
RULE_MAPPING = MANUAL_ROOT / "config" / "rule_mapping.csv"
RAW_ROOT = STAGE_ROOT / "raw_outputs" / "shell_pool"


def _radius_from_name(name: str) -> float:
    return float(name.removeprefix("r_").replace("p", "."))


def build_unit_index() -> pd.DataFrame:
    rules = pd.read_csv(RULE_MAPPING)
    rule_lookup = rules.set_index("rule_id")
    rows: list[dict[str, object]] = []
    methods: set[str] = set()
    for rule_dir in sorted(RAW_ROOT.glob("rule_*")):
        if rule_dir.name not in rule_lookup.index:
            raise ValueError(f"unknown rule directory: {rule_dir}")
        rule = rule_lookup.loc[rule_dir.name]
        for ref_dir in sorted(rule_dir.glob("ref_*")):
            for radius_dir in sorted(ref_dir.glob("r_*"), key=lambda path: _radius_from_name(path.name)):
                samples_path = radius_dir / "samples.npz"
                summary_path = radius_dir / "unit_summary.json"
                if not samples_path.exists() or not summary_path.exists():
                    raise FileNotFoundError(f"missing samples or unit summary under {radius_dir}")
                payload = json.loads(summary_path.read_text())
                methods.add(str(payload.get("sampler_method", "")))
                rows.append(
                    {
                        "rule_id": rule_dir.name,
                        "rule_name": rule.rule_name,
                        "label": rule.label,
                        "ref_id": ref_dir.name,
                        "original_ref_id": int(payload.get("ref_id", ref_dir.name.removeprefix("ref_"))),
                        "radius_path_id": radius_dir.name,
                        "radius": _radius_from_name(radius_dir.name),
                        "sampler_method": payload.get("sampler_method"),
                        "samples_path": (
                            "Complexity/03_dnn_mnist/manual_rules/04_sampling/raw_outputs/shell_pool/"
                            f"{rule_dir.name}/{ref_dir.name}/{radius_dir.name}/samples.npz"
                        ),
                        "unit_summary_path": (
                            "Complexity/03_dnn_mnist/manual_rules/04_sampling/raw_outputs/shell_pool/"
                            f"{rule_dir.name}/{ref_dir.name}/{radius_dir.name}/unit_summary.json"
                        ),
                    }
                )
    out = pd.DataFrame(rows).sort_values(["rule_id", "ref_id", "radius"]).reset_index(drop=True)
    if methods != {"exact_shell_l2_vmf_adaptive_ce_tempered_smc"}:
        raise RuntimeError(f"unexpected sampler methods: {sorted(methods)}")
    out.to_csv(RAW_ROOT / "unit_index.csv", index=False)
    return out


def main() -> None:
    print(RAW_ROOT / "unit_index.csv")
    build_unit_index()


if __name__ == "__main__":
    main()
