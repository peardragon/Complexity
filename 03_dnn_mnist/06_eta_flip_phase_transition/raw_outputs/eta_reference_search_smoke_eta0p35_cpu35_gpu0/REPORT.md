# Eta-Specific Reference Search Smoke

- Status: `complete`
- References: `1` / `1`
- Attempts: `4`
- Elapsed seconds: `4.667`
- CPU threads: `8`
- Device: `cpu`

This smoke trains eta-specific exact references for label-flipped MNIST even/odd datasets.
It is the formal-reference-search counterpart to the earlier fixed-anchor phi smoke.

Primary files:

- `01_dataset_gen/eta_dataset_manifest.csv`
- `04_exact_reference_search/attempt_logs/attempts.csv`
- `04_exact_reference_search/reference_index.csv`
- `04_exact_reference_search/selected_reference_pool/`

Datasets:

split_id,rule,eta,n_train,n_test,train_pos_fraction,dataset_path,flip_rate_train,flip_rate_test,reused
0,eta_0p35,0.35,512,2048,0.482421875,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_smoke_eta0p35_cpu35_gpu0/01_dataset_gen/split_000/eta_0p35/rep_000/dataset.npz,0.357421875,0.3603515625,False


Selected references:

rule,eta,ref_id,CE_mean_train,train_error,theta_norm
eta_0p35,0.35,0,0.002257074427163305,0.0,17.416036583580503

