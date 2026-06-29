#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(cd "${SCRIPT_DIR}/.." && pwd)"
LABEL_ROOT="$(cd "${STAGE}/.." && pwd)"
RUN_ROOT="${STAGE}/raw_outputs/shell_pool"
REFERENCE_ROOT="${LABEL_ROOT}/03_reference_search/raw_outputs"
LOG_DIR="${RUN_ROOT}/logs_eta0p02_add"

ETA_LIST="${ETA_DENSE_LIST:-0.02,0.05,0.15,0.25}"
REF_COUNT="${ETA_DENSE_REF_COUNT:-30}"
SHARDS="${ETA_DENSE_SHARDS:-15}"
THREADS="${ETA_DENSE_THREADS_PER_SHARD:-1}"
SAMPLES_PER_REF_RADIUS="${ETA_DENSE_SAMPLES_PER_REF_RADIUS:-1024}"

RADIUS_GRID="$(
  python - <<'PY'
print(",".join(f"{idx / 100:.2f}".rstrip("0").rstrip(".") for idx in range(1, 101)))
PY
)"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}/run_metadata"
date -Is > "${RUN_ROOT}/run_metadata/ADD_ETA0P02_STARTED_AT.txt"
: > "${RUN_ROOT}/run_metadata/RUNNING_PIDS_ETA0P02.txt"

export CUDA_VISIBLE_DEVICES=""
export MNIST14_DEVICE="cpu"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TORCH_NUM_THREADS="${THREADS}"
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONUNBUFFERED=1

for shard in $(seq 0 $((SHARDS - 1))); do
  (
    cd "${LABEL_ROOT}"
    python "${SCRIPT_DIR}/run_eta_reference_phi_smoke.py" \
      --run-root "${RUN_ROOT}" \
      --reference-run-root "${REFERENCE_ROOT}" \
      --etas "${ETA_LIST}" \
      --radii "${RADIUS_GRID}" \
      --ref-count "${REF_COUNT}" \
      --samples-per-ref-radius "${SAMPLES_PER_REF_RADIUS}" \
      --cpu-threads "${THREADS}" \
      --shard-index "${shard}" \
      --shard-count "${SHARDS}" \
      --save-samples-npz \
      --no-final-aggregate
  ) > "${LOG_DIR}/shard${shard}_of_${SHARDS}.log" 2>&1 &
  echo "$! shard${shard}_of_${SHARDS}" >> "${RUN_ROOT}/run_metadata/RUNNING_PIDS_ETA0P02.txt"
done

status=0
while read -r pid _; do
  wait "${pid}" || status=1
done < "${RUN_ROOT}/run_metadata/RUNNING_PIDS_ETA0P02.txt"

python "${SCRIPT_DIR}/run_eta_reference_phi_smoke.py" \
  --run-root "${RUN_ROOT}" \
  --reference-run-root "${REFERENCE_ROOT}" \
  --etas "${ETA_LIST}" \
  --radii "${RADIUS_GRID}" \
  --ref-count "${REF_COUNT}" \
  --samples-per-ref-radius "${SAMPLES_PER_REF_RADIUS}" \
  --cpu-threads 1 \
  --save-samples-npz \
  --aggregate-only > "${LOG_DIR}/final_aggregate.log" 2>&1 || status=1

python "${SCRIPT_DIR}/stage_paths.py" >> "${LOG_DIR}/final_aggregate.log" 2>&1 || status=1
date -Is > "${RUN_ROOT}/run_metadata/ADD_ETA0P02_FINISHED_AT.txt"
exit "${status}"
