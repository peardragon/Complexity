# DNN Proxy Local Entropy

This stage keeps compact proxy summary tables under `raw_outputs/` and active
3NN proxy figures under `figures/`.

The reproduction chain is:

`02_dnn/04_sampling/raw_outputs/.../summary_tables` -> `raw_outputs/.../summary_tables` -> `figures/...`.

Per-sample accuracy quantile tables are optional because they require reading
all retained sampling NPZ payloads again. The retained sample payloads remain in
`02_dnn/04_sampling/raw_outputs/.../sample_payloads`, and
`src/make_proxy_tables.py --include-accuracy` regenerates the accuracy table
when that full recovery is needed.
