#!/usr/bin/env bash
set -u -o pipefail

LOCAL_ROOT="/home/bjyong/Complexity/local_project/03_dnn_mnist"
META_ROOT="${META_ROOT:-${LOCAL_ROOT}/06_eta_flip_phase_transition/raw_outputs/direct_derivative_methodology_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0}"
RUN_ROOT="${META_ROOT}/01_active_rules_sampling"
LOG_DIR="${META_ROOT}/03_run_logs/rules"
SHARD_COUNT="${SHARD_COUNT:-9}"
THREADS_PER_SHARD="${THREADS_PER_SHARD:-2}"
SAMPLES_PER_UNIT="${SAMPLES_PER_UNIT:-1024}"
DERIVATIVE_CHUNK_SIZE="${DERIVATIVE_CHUNK_SIZE:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"

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

echo "[direct-rules] meta_root=${META_ROOT}"
echo "[direct-rules] run_root=${RUN_ROOT}"
echo "[direct-rules] shards=${SHARD_COUNT} threads_per_shard=${THREADS_PER_SHARD} samples=${SAMPLES_PER_UNIT}"

pids=()
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  log_path="${LOG_DIR}/shard_${shard}_of_${SHARD_COUNT}.log"
  (
    python 04_sampling/src/sample_refpool1024_all_radii.py \
      --run-root "${RUN_ROOT}" \
      --target-refs 30 \
      --radii "${RADII}" \
      --samples-per-ref-radius "${SAMPLES_PER_UNIT}" \
      --direct-derivative \
      --device cpu \
      --cpu-threads "${THREADS_PER_SHARD}" \
      --chunk-size "${CHUNK_SIZE}" \
      --derivative-chunk-size "${DERIVATIVE_CHUNK_SIZE}" \
      --shard-index "${shard}" \
      --shard-count "${SHARD_COUNT}" \
      --aggregate-every 0 \
      --save-samples-npz \
      --no-final-aggregate
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "[direct-rules] launched shard=${shard} pid=${pids[-1]} log=${log_path}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[direct-rules] at least one shard failed; see ${LOG_DIR}" >&2
  exit "${status}"
fi

python 04_sampling/src/sample_refpool1024_all_radii.py \
  --run-root "${RUN_ROOT}" \
  --target-refs 30 \
  --radii "${RADII}" \
  --samples-per-ref-radius "${SAMPLES_PER_UNIT}" \
  --direct-derivative \
  --device cpu \
  --cpu-threads "${THREADS_PER_SHARD}" \
  --chunk-size "${CHUNK_SIZE}" \
  --derivative-chunk-size "${DERIVATIVE_CHUNK_SIZE}" \
  --shard-count "${SHARD_COUNT}" \
  --aggregate-only \
  --save-samples-npz

echo "[direct-rules] complete"
