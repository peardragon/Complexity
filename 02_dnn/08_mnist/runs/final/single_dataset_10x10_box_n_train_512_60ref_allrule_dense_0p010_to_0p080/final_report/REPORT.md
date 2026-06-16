# MNIST10 BOX Single-Dataset 10x10 PM-SAIS Final Report

Architecture: `100-20-20-1-tanh`, P=2461

Scale: one split, 4 label rules, 60 exact references per rule, selected all-pass hard d_raw line [0.01, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.04, 0.05, 0.065, 0.08].

Main estimator: PM-SAIS averaged over references. `phi_by_rule_radius.csv` contains the selected line after every rule/radius row passed QC.

Sampling rows: 3360 unit summaries. Baseline sample-count mix: {4096: 2507}. Fallback total-sample mix: {16384: 721, 32768: 57, 65536: 75}.

## Complexity

| rule | TV | NMSTV |
| --- | --- | --- |
| low_tv_spectral_teacher | 0.203528 | 0.407056 |
| real_even_odd | 0.246643 | 0.493286 |
| teacher_nn | 0.342189 | 0.684377 |
| random_label | 0.492779 | 0.985559 |

## Sampling QC Coverage

| rule | QC-pass radii | total radii | pass radius range |
| --- | --- | --- | --- |
| low_tv_spectral_teacher | 14 | 14 | 0.0100..0.0800 |
| real_even_odd | 14 | 14 | 0.0100..0.0800 |
| teacher_nn | 14 | 14 | 0.0100..0.0800 |
| random_label | 14 | 14 | 0.0100..0.0800 |

Common supported radii across all rules: 14 / 14.

## Phi Outputs

| scope | file | rows | claim status |
| --- | --- | --- | --- |
| raw all-pass line diagnostic | phi_by_rule_radius_raw_diagnostic.csv | 56 | same rows as selected line |
| QC-pass rule/radius | phi_by_rule_radius.csv | 56 | all selected rows claimable |
| common-radius claim table | final_claim_table.csv | 14 | 14 supported, 0 no_claim |

## Figures

| figure set | path | content |
| --- | --- | --- |
| label representatives | ../01_dataset_prepare/figures/label_representatives/ | 4 rule figures |
| raw selected-line energy | figures/fig08_phi_energy_raw_dense_diagnostic.png | all rules, 14 radii each |
| raw selected-line full | figures/fig09_phi_full_raw_dense_diagnostic.png | all rules, 14 radii each |
| QC-pass heatmap | figures/fig07_sampling_qc_pass_heatmap.png | pass/no-claim by rule and radius |
| QC-pass energy | figures/fig04_phi_energy_qc_pass_main.png | only passing rule/radius rows |

## Pipeline Comparison To Retained 3NN Synthetic

| item | retained 3NN synthetic | MNIST10 BOX run |
| --- | --- | --- |
| model | 2-48-48-1 tanh, P=2545 | 100-20-20-1 tanh, P=2461 |
| input/data | synthetic 2D beta-cell datasets | one fixed MNIST 1/4 marginal, 10x10 BOX inputs |
| label axis | many beta-indexed synthetic datasets | same input marginal, four label rules |
| complexity measure | dataset/rule complexity used for ordering | graph TV/NMSTV on the fixed MNIST marginal |
| reference ensemble | 30 references per retained beta/dataset setting | 60 exact optimizer-induced references per rule |
| shell sampler | hard L2 shell PM-SAIS with adaptive CE-tempered SMC | same hard L2 shell PM-SAIS estimator |
| phi aggregation | averaged over retained sample/reference units by beta/radius | averaged over 60 references for each rule/radius |
| QC claim policy | retained dense production claim where sampler QC passed | selected 10x10 line must pass every rule/radius gate |

## Interpretation

The selected phi(d) curves were obtained for all four rules over 14 hard radii. Since every selected rule/radius row passed QC, the energy curve is the formal all-pass claim line rather than a no-claim diagnostic subset.

## All-Rule Dense Extension

Added all-rule dense `phi(d)_energy` and numerical derivative outputs. Derivative outputs use `numpy.gradient` over the dense hard-radius line. If Stage 06 passed, `dphi_dd_energy_by_rule_radius.csv` is based only on QC-passed rows; raw diagnostics are also retained with QC labels.

## Runtime And Reuse

Stage 05 used 3360 unit summaries: 2436 copied/reused unit JSON payloads and 924 newly computed unit JSON payloads. Newly computed unit elapsed time summed to 4326.89 s across unit payloads.

Newly computed units by rule:

| rule | copied units | newly computed units | new-unit elapsed s |
| --- | ---: | ---: | ---: |
| low_tv_spectral_teacher | 840 | 0 | 0.00 |
| random_label | 518 | 322 | 2679.14 |
| real_even_odd | 538 | 302 | 559.93 |
| teacher_nn | 540 | 300 | 1087.82 |

Five known bad broad-run outlier units were intentionally not copied and were recomputed with `rep16_n4096_cess95_mh2_outlier_recompute`: `random_label/ref_004/r=0.0200`, `random_label/ref_007/r=0.0300`, `random_label/ref_018/r=0.0300`, `real_even_odd/ref_000/r=0.0500`, and `real_even_odd/ref_027/r=0.0500`.
