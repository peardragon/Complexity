from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_loss_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (round(float(row["beta"]), 8), round(float(row["radius"]), 8), str(row["mode"]), str(row["H_or_threshold"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (beta, radius, mode, threshold), group in sorted(groups.items()):
        ratios = np.asarray([float(row["Z_ratio"]) for row in group], dtype=np.float64)
        logs = np.asarray([float(row["log_Z_ratio"]) for row in group], dtype=np.float64)
        finite_logs = logs[np.isfinite(logs)]
        out.append(
            {
                "beta": beta,
                "radius": radius,
                "mode": mode,
                "H_or_threshold": threshold,
                "ref_count": len(group),
                "mean_Z_ratio": float(np.mean(ratios)) if ratios.size else float("nan"),
                "mean_log_Z_ratio": float(np.mean(finite_logs)) if finite_logs.size else float("-inf"),
                "total_gate_count": int(sum(int(float(row["gate_count"])) for row in group)),
                "mean_ess_num": float(np.mean([float(row["ess_num"]) for row in group])) if group else 0.0,
                "claim": "pass" if finite_logs.size else "no_claim",
            }
        )
    return out

