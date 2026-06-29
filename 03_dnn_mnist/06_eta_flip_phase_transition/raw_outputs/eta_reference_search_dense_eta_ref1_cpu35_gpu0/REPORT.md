# Eta-Specific Reference Search Smoke

- Status: `complete`
- References: `6` / `6`
- Attempts: `24`
- Elapsed seconds: `6.318`
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
0,eta_0p20,0.2,512,2048,0.49609375,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_dense_eta_ref1_cpu35_gpu0/01_dataset_gen/split_000/eta_0p20/rep_000/dataset.npz,0.171875,0.1953125,False
0,eta_0p25,0.25,512,2048,0.494140625,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_dense_eta_ref1_cpu35_gpu0/01_dataset_gen/split_000/eta_0p25/rep_000/dataset.npz,0.287109375,0.2626953125,False
0,eta_0p30,0.3,512,2048,0.48046875,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_dense_eta_ref1_cpu35_gpu0/01_dataset_gen/split_000/eta_0p30/rep_000/dataset.npz,0.30078125,0.30078125,False
0,eta_0p35,0.35,512,2048,0.51171875,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_dense_eta_ref1_cpu35_gpu0/01_dataset_gen/split_000/eta_0p35/rep_000/dataset.npz,0.34375,0.36181640625,False
0,eta_0p40,0.4,512,2048,0.509765625,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_dense_eta_ref1_cpu35_gpu0/01_dataset_gen/split_000/eta_0p40/rep_000/dataset.npz,0.333984375,0.40625,False
0,eta_0p50,0.5,512,2048,0.509765625,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_dense_eta_ref1_cpu35_gpu0/01_dataset_gen/split_000/eta_0p50/rep_000/dataset.npz,0.541015625,0.4892578125,False


Selected references:

rule,eta,ref_id,CE_mean_train,train_error,theta_norm
eta_0p20,0.2,0,0.0012216558466707473,0.0,18.7819560875015
eta_0p25,0.25,0,0.0029456756343830747,0.0,16.638071187516644
eta_0p30,0.3,0,0.0010013522300583323,0.0,18.47810222092992
eta_0p35,0.35,0,0.0012615531247670906,0.0,17.90444473259974
eta_0p40,0.4,0,0.005329630529541957,0.0,15.23584703347892
eta_0p50,0.5,0,0.0013816742153038141,0.0,17.046604318818027

