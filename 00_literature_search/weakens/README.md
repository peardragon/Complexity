# Weakens Proxy Sampling Benchmark

This workspace is a focused reproduction scaffold for a failure mode:
local chain samplers can look valid near one low-loss solution while missing
low-loss mass in distant, narrow, or barrier-separated regions of a complex
dataset-induced landscape. A vMF-direction plus L2-radius importance proposal
is used as the recovery baseline.

The layout mirrors the numbered style used in `../../02_dnn` while keeping this
experiment self-contained:

- `00_method_notes/`: baseline papers, claim boundaries, and evidence matrix.
- `01_dataset_proxy/`: proxy landscape and dataset-complexity configuration.
- `02_region_protocol/`: pre-registered regions used for reachability tests.
- `03_baseline_samplers/`: HMC, pseudo-Langevin, and random-walk MCMC outputs.
- `04_vmf_l2_importance/`: vMF direction plus L2 radius importance sampling.
- `05_qc_and_figures/`: QC tables and final figure.
- `src/weakens_benchmark/`: shared implementation.
- `scripts/run_reproduce.py`: one-command reproduction entrypoint.

Run the default reproduction. This is the report-grade setting and may take a
couple of minutes because HMC uses full gradients:

```bash
python scripts/run_reproduce.py --config 01_dataset_proxy/config/default.json
```

For a faster smoke check:

```bash
python scripts/run_reproduce.py --config 01_dataset_proxy/config/smoke.json
```

Run the L2-common target reproduction. In this setting every method targets
the same `E_proxy(z) + lambda ||z||^2` landscape; only the sampling/proposal
mechanism differs, and attempted sample counts are recorded in QC:

```bash
python scripts/run_reproduce.py --config 01_dataset_proxy/config/default_l2_common.json
```

For a faster L2-common smoke check:

```bash
python scripts/run_reproduce.py --config 01_dataset_proxy/config/smoke_l2_common.json
```

Primary output:

```text
05_qc_and_figures/figures/<experiment_id>/final_sampling_failure_vmf_recovery.png
05_qc_and_figures/figures/<experiment_id>/final_sampling_failure_vmf_recovery_AB.png
05_qc_and_figures/figures/<experiment_id>/schematic_problem_statement_vmf_l2.png
05_qc_and_figures/figures/<experiment_id>/method_sampling_results_l2_common/
```

Interpretation:

- HMC and pL are recorded as existing-method reproductions or proxy
  reproductions with explicit metadata.
- `vmf_l2_final` is the proxy-coordinate instantiation of the final method,
  not an unrelated extra baseline.
- QC failure means finite-budget failure on this registered proxy landscape:
  missing important regions, high region-mass error, or low ESS.
