# MNIST manual-rule QC report

- axis policy: manual rules, not beta.
- units: `12000` = 4 rules x 30 references x 100 radii.
- sampler methods: `exact_shell_l2_vmf_adaptive_ce_tempered_smc`.
- tempered-path default: `True`.
- QC A: reference variability is summarized as SD and SE across references for each rule/radius.
- QC B: split logZ stability is summarized by q95/max split logZ/P difference for each rule/radius.
- Dataset variability is not plotted because this manual-rule MNIST stage uses a single dataset per rule.

## Figure inputs

- `figure_inputs/logZ_split_qc_results/logZ_split_qc_results.csv` -> `figures/logZ_split_qc_results.png`
- `figure_inputs/reference_variability_results/reference_variability_results.csv` -> `figures/reference_variability_results.png`
