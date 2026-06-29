# Eta Flip Phi Advanced 90ref Full Run Monitoring

- Run name: `eta_reference_phi_advanced_4eta_90ref_r0p1_to_1p0_step0p05_n1024_cpu60_gpu0`
- Reference pool: `eta_reference_search_advanced_4eta_90ref_cpu35_gpu0`
- Basis: 4 eta values, 90 references per eta, 19 radii from 0.10 to 1.00 in 0.05 steps, 1024 samples per ref/radius.
- Expected units: 6,840.
- Smoke trial: 8 units, one per shard.
- Smoke unit elapsed seconds: 20.305, 28.266, 32.783, 32.578, 36.337, 37.464, 37.751, 37.742.
- Smoke mean unit elapsed seconds: 32.903.
- Estimated remaining wall time with 8 shards from smoke mean: about 7.8 hours after the smoke units.
- Resource policy: CPU-only, 8 shards, 2 threads per shard, expected near but below 60% of a 32-CPU machine; GPU disabled for this wrapper path, so GPU usage remains below the 50% cap.
- Detached runner: `tmux` session `eta_phi_adv90_r1p0_cpu60_20260626`.
- On completion the manager runs final aggregation, eta phi(d) figure generation, and the combined advanced-plus-flip figure generation.
