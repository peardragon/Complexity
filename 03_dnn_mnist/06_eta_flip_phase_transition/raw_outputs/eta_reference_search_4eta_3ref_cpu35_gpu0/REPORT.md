# Eta-Specific Reference Search Smoke

- Status: `complete`
- References: `12` / `12`
- Attempts: `16`
- Elapsed seconds: `5.559`
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
0,eta_0p00,0.0,512,2048,0.5,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_4eta_3ref_cpu35_gpu0/01_dataset_gen/split_000/eta_0p00/rep_000/dataset.npz,0.0,0.0,False
0,eta_0p20,0.2,512,2048,0.478515625,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_4eta_3ref_cpu35_gpu0/01_dataset_gen/split_000/eta_0p20/rep_000/dataset.npz,0.201171875,0.1865234375,False
0,eta_0p35,0.35,512,2048,0.544921875,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_4eta_3ref_cpu35_gpu0/01_dataset_gen/split_000/eta_0p35/rep_000/dataset.npz,0.345703125,0.34130859375,False
0,eta_0p50,0.5,512,2048,0.4921875,/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs/eta_reference_search_4eta_3ref_cpu35_gpu0/01_dataset_gen/split_000/eta_0p50/rep_000/dataset.npz,0.4453125,0.5078125,False


Selected references:

rule,eta,ref_id,CE_mean_train,train_error,theta_norm
eta_0p00,0.0,0,0.0018833008850675098,0.0,12.815650008581265
eta_0p00,0.0,1,0.0028950254893227835,0.0,13.965838688474937
eta_0p00,0.0,2,0.0008582590459593344,0.0,13.935842952637815
eta_0p20,0.2,0,0.0012130485596157348,0.0,16.57950195563767
eta_0p20,0.2,1,0.001148368067514653,0.0,17.461216361169857
eta_0p20,0.2,2,0.0017074979131422347,0.0,18.141008348521748
eta_0p35,0.35,0,0.0018494231960365514,0.0,16.464693303006868
eta_0p35,0.35,1,0.003981932391421011,0.0,17.248209130097454
eta_0p35,0.35,2,0.004444618718684838,0.0,18.992417588694984
eta_0p50,0.5,0,0.0023441583419617907,0.0,15.520819623152663
eta_0p50,0.5,1,0.003517266047503663,0.0,16.85979882345147
eta_0p50,0.5,2,0.003455172308523421,0.0,17.101015519020955

