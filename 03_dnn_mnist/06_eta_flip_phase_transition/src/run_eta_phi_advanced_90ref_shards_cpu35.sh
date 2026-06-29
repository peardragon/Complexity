#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bjyong/Complexity/local_project/03_dnn_mnist"
STAGE="$ROOT/06_eta_flip_phase_transition"
RUN_NAME="eta_reference_phi_advanced_4eta_90ref_r0p1_to_2p5_step0p05_n1024_cpu35_gpu0"
RUN_ROOT="$STAGE/raw_outputs/$RUN_NAME"
REF_ROOT="$STAGE/raw_outputs/eta_reference_search_advanced_4eta_90ref_cpu35_gpu0"
LOG_DIR="$RUN_ROOT/logs"
SHARDS="${ETA_ADV_SHARDS:-4}"
THREADS="${ETA_ADV_THREADS_PER_SHARD:-2}"
RADIUS_GRID="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5"

mkdir -p "$LOG_DIR"
date -Is > "$RUN_ROOT/MANAGER_STARTED_AT.txt"
: > "$RUN_ROOT/RUNNING_PIDS.txt"

export CUDA_VISIBLE_DEVICES=""
export MNIST14_DEVICE="cpu"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TORCH_NUM_THREADS="$THREADS"
export TORCH_NUM_INTEROP_THREADS=1

for shard in $(seq 0 $((SHARDS - 1))); do
  (
    cd "$ROOT"
    python "$STAGE/src/run_eta_reference_phi_smoke.py" \
      --run-name "$RUN_NAME" \
      --reference-run-root "$REF_ROOT" \
      --etas 0.25,0.30,0.35,0.40 \
      --radii "$RADIUS_GRID" \
      --ref-count 90 \
      --samples-per-ref-radius 1024 \
      --cpu-threads "$THREADS" \
      --shard-index "$shard" \
      --shard-count "$SHARDS" \
      --no-final-aggregate
  ) > "$LOG_DIR/shard${shard}_of_${SHARDS}.log" 2>&1 &
  echo "$!" >> "$RUN_ROOT/RUNNING_PIDS.txt"
done

wait

cd "$ROOT"
python "$STAGE/src/run_eta_reference_phi_smoke.py" \
  --run-name "$RUN_NAME" \
  --reference-run-root "$REF_ROOT" \
  --etas 0.25,0.30,0.35,0.40 \
  --radii "$RADIUS_GRID" \
  --ref-count 90 \
  --samples-per-ref-radius 1024 \
  --cpu-threads "$THREADS" \
  --aggregate-only > "$LOG_DIR/final_aggregate.log" 2>&1

python "$STAGE/src/plot_eta_phi_energy_only.py" \
  --run-root "$RUN_ROOT" \
  --out-dir "$STAGE/figures/eta_phi_energy_advanced_90ref_n1024" \
  --d-min 0.1 \
  --d-max 2.5 > "$LOG_DIR/plot_eta_phi_energy.log" 2>&1

python "$STAGE/src/plot_combined_advanced_and_eta_phi_energy.py" \
  --advanced-run-root "$ROOT/04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref" \
  --eta-run-root "$RUN_ROOT" \
  --graph-run-root "$STAGE/raw_outputs/eta_sweep_pilot_cpu35_gpu0" \
  --out-dir "$STAGE/figures/combined_advanced_eta_phi_energy_90ref" \
  --d-min 0.1 \
  --d-max 2.5 > "$LOG_DIR/plot_combined.log" 2>&1

date -Is > "$RUN_ROOT/MANAGER_FINISHED_AT.txt"
