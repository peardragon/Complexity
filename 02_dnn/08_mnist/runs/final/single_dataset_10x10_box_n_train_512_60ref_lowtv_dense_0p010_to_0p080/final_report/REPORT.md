# MNIST10 BOX Single-Dataset 10x10 PM-SAIS Final Report

Architecture: `100-20-20-1-tanh`, P=2461

Scale: one split, 1 label rules, 60 exact references per rule, selected all-pass hard d_raw line [0.01, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.04, 0.05, 0.065, 0.08].

Main estimator: PM-SAIS averaged over references. `phi_by_rule_radius.csv` contains the selected line after every rule/radius row passed QC.

Sampling rows: 840 unit summaries. Baseline sample-count mix: {4096: 840}. Fallback total-sample mix: {}.

## Complexity

| rule | TV | NMSTV |
| --- | --- | --- |
| low_tv_spectral_teacher | 0.203528 | 0.407056 |

## Sampling QC Coverage

| rule | QC-pass radii | total radii | pass radius range |
| --- | --- | --- | --- |
| low_tv_spectral_teacher | 14 | 14 | 0.0100..0.0800 |

Common supported radii across all rules: 14 / 14.

## Phi Outputs

| scope | file | rows | claim status |
| --- | --- | --- | --- |
| raw all-pass line diagnostic | phi_by_rule_radius_raw_diagnostic.csv | 14 | same rows as selected line |
| QC-pass rule/radius | phi_by_rule_radius.csv | 14 | all selected rows claimable |
| common-radius claim table | final_claim_table.csv | 14 | 14 supported, 0 no_claim |

## Figures

| figure set | path | content |
| --- | --- | --- |
| label representatives | ../01_dataset_prepare/figures/label_representatives/ | 1 rule figures |
| raw selected-line energy | figures/fig08_phi_energy_raw_dense_diagnostic.png | all rules, 14 radii each |
| raw selected-line full | figures/fig09_phi_full_raw_dense_diagnostic.png | all rules, 14 radii each |
| QC-pass heatmap | figures/fig07_sampling_qc_pass_heatmap.png | pass/no-claim by rule and radius |
| QC-pass energy | figures/fig04_phi_energy_qc_pass_main.png | only passing rule/radius rows |

## Pipeline Comparison To Retained 3NN Synthetic

| item | retained 3NN synthetic | MNIST10 BOX run |
| --- | --- | --- |
| model | 2-48-48-1 tanh, P=2545 | 100-20-20-1 tanh, P=2461 |
| input/data | synthetic 2D beta-cell datasets | one fixed MNIST 1/4 marginal, 10x10 BOX inputs |
| label axis | many beta-indexed synthetic datasets | same input marginal, low-TV spectral teacher rule |
| complexity measure | dataset/rule complexity used for ordering | graph TV/NMSTV on the fixed MNIST marginal |
| reference ensemble | 30 references per retained beta/dataset setting | 60 exact optimizer-induced references per rule |
| shell sampler | hard L2 shell PM-SAIS with adaptive CE-tempered SMC | same hard L2 shell PM-SAIS estimator |
| phi aggregation | averaged over retained sample/reference units by beta/radius | averaged over 60 references for each rule/radius |
| QC claim policy | retained dense production claim where sampler QC passed | selected 10x10 line must pass every rule/radius gate |

## Interpretation

The selected phi(d) curve was obtained for the low-TV spectral teacher over 14 hard radii. Since every selected rule/radius row passed QC, the energy curve is the formal all-pass claim line rather than a no-claim diagnostic subset.

## Low-TV Dense Extension

Added dense all-QC low-TV-only `phi(d)_energy` and numerical derivative outputs. The derivative uses `numpy.gradient` over the QC-passed dense hard-radius line.

## Runtime

Pilot timing used `max_units=14`, covering one reference across all radii; 5 newly computed units took 22.02 s, estimating roughly 22.0 min for 300 missing units before reuse and summary overhead. The full Stage 05 run completed 840 unit rows in 740.94 s, with 540 copied/reused unit summaries and 300 newly computed unit summaries.
