#!/usr/bin/env bash
set -u -o pipefail

LOCAL_ROOT="/home/bjyong/Complexity/local_project/03_dnn_mnist"
STAGE_ROOT="${LOCAL_ROOT}/06_eta_flip_phase_transition"
META_ROOT="${META_ROOT:-${STAGE_ROOT}/raw_outputs/direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0}"
RUN_NAME="$(python - <<'PY'
from pathlib import Path
root = Path("/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/raw_outputs")
meta = Path(__import__("os").environ.get("META_ROOT", root / "direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"))
print(str((meta / "02_eta_flip_sampling").relative_to(root)))
PY
)"
REFERENCE_RUN_ROOT="${REFERENCE_RUN_ROOT:-${STAGE_ROOT}/raw_outputs/eta_reference_search_gapfill_0p02_0p05_0p15_0p25_30ref_cpu60_gpu0}"
LOG_DIR="${META_ROOT}/03_run_logs/eta"
SHARD_COUNT="${SHARD_COUNT:-9}"
THREADS_PER_SHARD="${THREADS_PER_SHARD:-2}"
SAMPLES_PER_UNIT="${SAMPLES_PER_UNIT:-1024}"
DERIVATIVE_CHUNK_SIZE="${DERIVATIVE_CHUNK_SIZE:-64}"
ETAS="${ETAS:-0.05,0.15,0.25}"

mkdir -p "${LOG_DIR}"
cd "${LOCAL_ROOT}" || exit 1

RADII="$(python - <<'PY'
print(",".join(f"{i / 100:.2f}" for i in range(1, 101)))
PY
)"

export CUDA_VISIBLE_DEVICES=""
export MNIST14_DEVICE="cpu"
export OMP_NUM_THREADS="${THREADS_PER_SHARD}"
export MKL_NUM_THREADS="${THREADS_PER_SHARD}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_SHARD}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_SHARD}"
export TORCH_NUM_THREADS="${THREADS_PER_SHARD}"
export TORCH_NUM_INTEROP_THREADS="${THREADS_PER_SHARD}"

echo "[direct-eta] meta_root=${META_ROOT}"
echo "[direct-eta] run_name=${RUN_NAME}"
echo "[direct-eta] reference_run_root=${REFERENCE_RUN_ROOT}"
echo "[direct-eta] etas=${ETAS} shards=${SHARD_COUNT} threads_per_shard=${THREADS_PER_SHARD} samples=${SAMPLES_PER_UNIT}"

pids=()
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  log_path="${LOG_DIR}/shard_${shard}_of_${SHARD_COUNT}.log"
  (
    python 06_eta_flip_phase_transition/src/run_eta_reference_phi_smoke.py \
      --run-name "${RUN_NAME}" \
      --reference-run-root "${REFERENCE_RUN_ROOT}" \
      --etas "${ETAS}" \
      --radii "${RADII}" \
      --ref-count 30 \
      --samples-per-ref-radius "${SAMPLES_PER_UNIT}" \
      --direct-derivative \
      --derivative-chunk-size "${DERIVATIVE_CHUNK_SIZE}" \
      --cpu-threads "${THREADS_PER_SHARD}" \
      --shard-index "${shard}" \
      --shard-count "${SHARD_COUNT}" \
      --save-samples-npz \
      --no-final-aggregate
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "[direct-eta] launched shard=${shard} pid=${pids[-1]} log=${log_path}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[direct-eta] at least one shard failed; see ${LOG_DIR}" >&2
  exit "${status}"
fi

python 06_eta_flip_phase_transition/src/run_eta_reference_phi_smoke.py \
  --run-name "${RUN_NAME}" \
  --reference-run-root "${REFERENCE_RUN_ROOT}" \
  --etas "${ETAS}" \
  --radii "${RADII}" \
  --ref-count 30 \
  --samples-per-ref-radius "${SAMPLES_PER_UNIT}" \
  --direct-derivative \
  --derivative-chunk-size "${DERIVATIVE_CHUNK_SIZE}" \
  --cpu-threads "${THREADS_PER_SHARD}" \
  --shard-count "${SHARD_COUNT}" \
  --save-samples-npz \
  --aggregate-only

echo "[direct-eta] complete"
