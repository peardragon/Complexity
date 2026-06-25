# Two-pool sampling shell_pool run report

## Config

- 최종 결과는 base dense p2048 sampling, far-radius p32768 replacement, bad-split p262144 replacement를 병합한 shell sampling 결과이다.
- 유지 원칙: final CSV에 직접 쓰인 payload와 replacement audit payload를 모두 보존한다.

## Output files

- `sample_payloads/base_10x10_N40_80_160_320_p2048_r0p05/`: base raw dataset/reference/sample payload. final `sample_unit_summary.csv`에 13,200행 사용된다.
- `sample_payloads/far_split_N40_p32768/`: N=40 far-radius replacement payload. final CSV에 400행 사용된다.
- `sample_payloads/far_split_N80_p32768/`: N=80 far-radius replacement payload. final CSV에 600행 사용된다.
- `sample_payloads/far_split_N160_p32768/`: N=160 far-radius replacement payload. final CSV에 1,000행 사용된다.
- `sample_payloads/far_split_N320_p32768/`: N=320 far-radius replacement payload. final CSV에 1,585행 사용된다. superseded 15개 sample은 QC replacement audit를 위해 보존한다.
- `sample_payloads/replacement_bad_split_p262144/`: N=320 bad-split unit replacement payload. final CSV에 15행 사용된다.
- `sample_unit_summary.csv`: raw sample payload에서 집계한 unit-level summary이다. `source_run_id`와 `sample_payload_path`가 compact payload 위치를 추적한다.
- `sampling_phi_by_N_alpha0p1.csv`: analytic curve 없이 sampling empirical phi만 담은 N별 figure 입력 table이다.
- `sampling_qc_by_N_radius.csv`: radius/N별 split 및 SMC QC summary이다.

## Reproduction chain

`sample_payloads/`의 raw `samples.npz`들을 unit-level로 집계해 `sample_unit_summary.csv`를 만들고, 이 summary에서 `sampling_phi_by_N_alpha0p1.csv`와 `sampling_qc_by_N_radius.csv`를 만든다. sampling-only figure는 `sampling_phi_by_N_alpha0p1.csv`만 사용한다.
