# Reference Cloud Proxy Metrics Run Report

- Dataset units: 540
- Pairwise rows: 234900
- Backend: `.venv` Python, torch CUDA auto device, float32 evaluation
- CPU smoke: 18 dataset units in 157.488 seconds, 8.749 seconds per dataset
- CUDA smoke: 18 dataset units in 7.025 seconds, 0.390 seconds per dataset
- CUDA smoke estimate for 540 dataset units: 210.753 seconds
- Elapsed seconds: 302.965
- Seconds per dataset: 0.561
- t_grid_count: 21
- Figures generated: True

Claim boundary: `B_lin` is a straight-line CE/error barrier diagnostic only. It is not a nonlinear connectivity proof, and `G_tau` was not computed.
