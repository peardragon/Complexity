# Two-pool sampling shell_pool run report

## Config

- Final result was rebuilt as default adaptive CE-tempered SMC only.
- Replaced all legacy direct-vMF IS units and all p262144 bad-split units.
- Particle cap: `32768` observed, target cap `32768`.
- Split logZ threshold per N: `0.006`.
- Legacy direct/fallback decision columns are preserved as `legacy_*`; final policy columns set no direct fallback.

## Output files

- `near_split/N_*/dataset_*/ref_*/r_*/samples.npy`: canonical 2048-particle near-radius payloads.
- `far_split/N_*/dataset_*/ref_*/r_*/samples.npy`: canonical 32768-particle far-radius payloads.
- `sample_unit_summary.csv`: unit-level final summary.
- `sampling_phi_by_N_alpha0p1.csv`: sampling empirical phi table for figures.
- `sampling_logz_stability_by_N_radius.csv`: radius/N split-logZ and SMC stability summary.
- `default_smc_final_validation.json`: merge and QC validation summary.

## Validation

- Units: `16800`.
- Method counts: `{'exact_shell_l2_vmf_adaptive_ce_smc': 16800}`.
- Particle counts: `{'2048': 13200, '32768': 3600}`.
- Replacement reason counts: `{'nondefault_direct_is': 5731, 'n40_default_smc_tempered_path': 4200, 'cap_particles_32768': 15}`.
- Non-default method count: `0`.
- Over-cap unit count: `0`.
- Final fallback unit count: `0`.
- Max split logZ / N: `0.0059277203723576`.
- Split-fail QC cells: `0`.
- Min SMC CESS fraction: `0.8500000000000005`.
- SMC-fail QC cells: `0`.
- Payload paths missing: `0`.
- Full default-SMC pass: `True`.

## Reproduction chain

Canonical near/far raw sample files are aggregated into `sample_unit_summary.csv`; the compact phi and QC tables are regenerated from this summary. Sampling-only and theory-comparison figures read the compact CSVs.
