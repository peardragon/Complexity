# Eta Flip Phi Advanced 90ref Full Run Monitoring

- Run name: `eta_reference_phi_advanced_4eta_90ref_r0p1_to_2p5_step0p05_n1024_cpu35_gpu0`
- Reference pool: `eta_reference_search_advanced_4eta_90ref_cpu35_gpu0`
- Basis: 4 eta values, 90 references per eta, 49 radii from 0.10 to 2.50 in 0.05 steps, 1024 samples per ref/radius.
- Expected units: 17,640.
- Smoke trial: 4 units, one per shard.
- Smoke unit elapsed seconds: 16.922, 24.158, 28.862, 29.996.
- Smoke mean unit elapsed seconds: 24.985.
- Estimated remaining wall time with 4 shards from smoke mean: about 30.6 hours after the smoke units.
- Conservative estimate from previous advanced mean unit time: about 57 hours.
- Resource policy: CPU-only, 4 shards, 2 threads per shard, expected below 35% of a 32-CPU machine; GPU disabled.
- Detached runner: `tmux` session `eta_phi_adv90_20260626`.
- On completion the manager runs final aggregation, eta phi(d) figure generation, and the combined advanced-plus-flip figure generation.

## Stop Note

This run was stopped after the radius contract was corrected to `d <= 1.0`.
The partial payload is preserved for provenance only; promoted label-flip
advanced sampling continues in
`eta_reference_phi_advanced_4eta_90ref_r0p1_to_1p0_step0p05_n1024_cpu35_gpu0`.
