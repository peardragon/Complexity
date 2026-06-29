#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bjyong/Complexity/local_project/03_dnn_mnist"
STAGE="$ROOT/06_eta_flip_phase_transition"
RULE_RUN="$ROOT/04_sampling/raw_outputs/active_rules_refpool1024_dense30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
ETA_RUN="$STAGE/raw_outputs/eta_reference_phi_dense_4eta_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
MASTER_LOG_DIR="$STAGE/raw_outputs/dense30ref_rules_flips_r1p0_cpu60_gpu0_master/logs"
GRAPH_RUN="$STAGE/raw_outputs/eta_sweep_pilot_cpu35_gpu0"

mkdir -p "$MASTER_LOG_DIR"
date -Is > "$MASTER_LOG_DIR/MASTER_STARTED_AT.txt"

{
  echo "Dense 30ref full run"
  echo "rule_run=$RULE_RUN"
  echo "eta_run=$ETA_RUN"
  echo "radii=0.01..1.00 step 0.01"
  echo "samples_per_unit=1024"
  echo "resource_policy=sequential rule then eta; each stage CPU-only and capped <=19/32 logical CPUs"
  echo "rule_smoke=/home/bjyong/Complexity/local_project/03_dnn_mnist/04_sampling/smoke_runs/dense30ref_r0p01_to_1p0_step0p01_n1024_timing"
  echo "eta_smoke=/home/bjyong/Complexity/local_project/03_dnn_mnist/06_eta_flip_phase_transition/smoke_runs/eta_reference_phi_dense30ref_r0p01_to_1p0_step0p01_n1024_timing"
} > "$MASTER_LOG_DIR/RUN_PLAN.txt"

bash "$ROOT/04_sampling/src/run_refpool1024_dense30ref_r1p0_cpu60_gpu0.sh" \
  > "$MASTER_LOG_DIR/rule_manager.log" 2>&1

python "$ROOT/04_sampling/src/plot_advanced_phi_energy_spaghetti.py" \
  --run-root "$RULE_RUN" \
  --d-min 0.01 \
  --d-max 1.0 \
  > "$MASTER_LOG_DIR/plot_rule_phi.log" 2>&1

bash "$STAGE/src/run_eta_phi_dense30ref_r1p0_shards_cpu60_gpu0.sh" \
  > "$MASTER_LOG_DIR/eta_manager.log" 2>&1

python "$STAGE/src/plot_eta_phi_energy_only.py" \
  --run-root "$ETA_RUN" \
  --out-dir "$STAGE/figures/eta_phi_energy_dense30ref_r1p0_n1024_cpu60_gpu0" \
  --d-min 0.01 \
  --d-max 1.0 \
  > "$MASTER_LOG_DIR/plot_eta_phi_energy.log" 2>&1

python "$STAGE/src/plot_combined_advanced_and_eta_phi_energy.py" \
  --advanced-run-root "$RULE_RUN" \
  --eta-run-root "$ETA_RUN" \
  --graph-run-root "$GRAPH_RUN" \
  --out-dir "$STAGE/figures/combined_dense30ref_rules_eta_phi_energy_r1p0_cpu60_gpu0" \
  --d-min 0.01 \
  --d-max 1.0 \
  > "$MASTER_LOG_DIR/plot_combined.log" 2>&1

python "$STAGE/src/plot_eta_positive_curvature_mass_composite.py" \
  --run-root "$ETA_RUN" \
  --out-dir "$STAGE/figures/eta_positive_curvature_mass_dense30ref_r1p0_n1024_cpu60_gpu0" \
  --d-min 0.01 \
  --d-max 1.0 \
  > "$MASTER_LOG_DIR/plot_eta_curvature.log" 2>&1

date -Is > "$MASTER_LOG_DIR/MASTER_FINISHED_AT.txt"
