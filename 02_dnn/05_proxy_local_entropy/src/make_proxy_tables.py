from __future__ import annotations

import argparse
from pathlib import Path

from compute_phi import delta_phi_rows
from io_utils import read_csv, save_csv
from plot_summary import DEFAULT_Q_VALUES, hq_phase_rows
from plot_summary_derivative import absolute_phi_rows, accuracy_q_phase_rows, derivative_phi_rows


def _write_table(path: Path, rows: list[dict[str, object]], fallback_fields: list[str]) -> None:
    fields = list(rows[0].keys()) if rows else fallback_fields
    save_csv(path, rows, fields)


def write_run_report(range_label: str, output_root: Path, *, include_accuracy: bool) -> None:
    report = f"""# Proxy local entropy raw_outputs report: {range_label}

이 폴더는 05_proxy_local_entropy figure를 재현하기 위한 proxy summary table을 보관한다.

## Config

- input unit summary: `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/{range_label}/summary_tables/sample_unit_summary.csv`
- input R-H summary: `02_dnn/04_sampling/raw_outputs/shell_pool/9_beta_cell_10_dataset_10_reference/{range_label}/summary_tables/rh_by_ref_radius_h.csv`
- proxy method: full regularized local entropy view using `logZ_inf_full`, `logZ_inf_stripped`, and `reference_prior_log_weight` from the 04 sampling unit summary
- regularization fallback: `compute_phi.DEFAULT_LAMBDA_REG=220` only if the full/correction fields are absent
- q values: 0.5, 0.9, 0.99
- accuracy quantile table regenerated: {'yes' if include_accuracy else 'no'}

## Output files

- `summary_tables/absolute_phi_by_beta_radius.csv`: 04 sampling unit summary에서 beta/radius별 absolute full phi, energy term, area term을 집계한 table이다.
- `summary_tables/delta_phi_by_beta_radius.csv`: 기준 radius 대비 delta phi와 energy/area split을 beta/radius별로 집계한 table이다.
- `summary_tables/dphi_dr_by_beta_radius.csv`: radial derivative 관련 proxy quantity를 beta/radius별로 정리한 table이다.
- `summary_tables/hq_by_beta_radius.csv`: `rh_by_ref_radius_h.csv`에서 q별 H threshold phase map 입력을 만든 table이다.
- `summary_tables/accuracy_q_by_beta_radius.csv`: sample payload NPZ의 per-sample error/log weight를 다시 읽어 q별 weighted accuracy cutoff를 만든 table이다. 이 table은 `--include-accuracy` 실행 때만 생성한다.

## Reproduction chain

`04_sampling/.../summary_tables/`와 필요 시 `04_sampling/.../sample_payloads/`를 입력으로 이 폴더의 proxy summary tables를 만들고, `05_proxy_local_entropy/figures/`의 local entropy, derivative, phase-map figure가 이 table들을 사용한다.
"""
    (output_root / "run_report.md").write_text(report, encoding="utf-8")


def make_tables(range_root: Path, output_root: Path, *, include_accuracy: bool) -> list[Path]:
    summary_root = range_root / "summary_tables"
    unit_rows = read_csv(summary_root / "sample_unit_summary.csv")
    rh_rows = read_csv(summary_root / "rh_by_ref_radius_h.csv")
    table_dir = output_root / "summary_tables"

    outputs: list[Path] = []

    abs_rows = absolute_phi_rows(unit_rows)
    abs_path = table_dir / "absolute_phi_by_beta_radius.csv"
    _write_table(abs_path, abs_rows, ["beta", "radius", "phi_full", "claim"])
    outputs.append(abs_path)

    delta_rows = delta_phi_rows(unit_rows)
    delta_path = table_dir / "delta_phi_by_beta_radius.csv"
    _write_table(delta_path, delta_rows, ["beta", "radius", "delta_phi_full", "claim"])
    outputs.append(delta_path)

    derivative_rows = derivative_phi_rows(abs_rows)
    derivative_path = table_dir / "dphi_dr_by_beta_radius.csv"
    _write_table(derivative_path, derivative_rows, ["beta", "radius", "dphi_full_dr", "claim"])
    outputs.append(derivative_path)

    hq_rows = hq_phase_rows(rh_rows, q_values=DEFAULT_Q_VALUES)
    hq_path = table_dir / "hq_by_beta_radius.csv"
    _write_table(hq_path, hq_rows, ["q", "beta", "radius", "H_q"])
    outputs.append(hq_path)

    if include_accuracy:
        accuracy_rows = accuracy_q_phase_rows(
            unit_rows,
            delta_rows,
            q_values=DEFAULT_Q_VALUES,
            progress_path=output_root / "logs" / "accuracy_phase_status.json",
        )
        accuracy_path = table_dir / "accuracy_q_by_beta_radius.csv"
        _write_table(accuracy_path, accuracy_rows, ["q", "beta", "radius", "accuracy_q", "claim"])
        outputs.append(accuracy_path)

    write_run_report(range_root.name, output_root, include_accuracy=include_accuracy)
    outputs.append(output_root / "run_report.md")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 05 proxy-local-entropy raw output tables from 04 sampling summaries.")
    parser.add_argument(
        "--range-root",
        type=Path,
        required=True,
        help="Range root under 02_dnn/04_sampling/raw_outputs/shell_pool/.../d_*.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Range root under 02_dnn/05_proxy_local_entropy/raw_outputs/.../d_*.",
    )
    parser.add_argument(
        "--include-accuracy",
        action="store_true",
        help="Also regenerate accuracy_q_by_beta_radius.csv by reading every sample payload NPZ.",
    )
    args = parser.parse_args()

    outputs = make_tables(args.range_root, args.output_root, include_accuracy=bool(args.include_accuracy))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
