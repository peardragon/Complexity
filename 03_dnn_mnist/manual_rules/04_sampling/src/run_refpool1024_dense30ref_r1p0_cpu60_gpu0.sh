#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bjyong/Complexity/Complexity/03_dnn_mnist/manual_rules"
RUN_NAME="shell_pool"
RUN_ROOT="$ROOT/04_sampling/raw_outputs/$RUN_NAME"

RADIUS_GRID="$(
  python - <<'PY'
print(",".join(f"{idx / 100:.2f}".rstrip("0").rstrip(".") for idx in range(1, 101)))
PY
)"

export REFPOOL1024_RUN_ROOT="${REFPOOL1024_RUN_ROOT:-$RUN_ROOT}"
export REFPOOL1024_TARGET_REFS="${REFPOOL1024_TARGET_REFS:-30}"
export REFPOOL1024_RADIUS_GRID="${REFPOOL1024_RADIUS_GRID:-production}"
export REFPOOL1024_RADII="${REFPOOL1024_RADII:-$RADIUS_GRID}"
export REFPOOL1024_SHARDS="${REFPOOL1024_SHARDS:-17}"
export REFPOOL1024_CPU_THREADS_PER_SHARD="${REFPOOL1024_CPU_THREADS_PER_SHARD:-1}"
export REFPOOL1024_MAX_LOGICAL_CPUS="${REFPOOL1024_MAX_LOGICAL_CPUS:-19}"
export REFPOOL1024_REBALANCE_AFFINITY="${REFPOOL1024_REBALANCE_AFFINITY:-1}"

mkdir -p "$RUN_ROOT/logs"
cat > "$RUN_ROOT/DENSE30_RUN_PLAN.txt" <<EOF
run_name=$RUN_NAME
run_root=$RUN_ROOT
scope=4 active MNIST rules, 30 references per rule, n=1024
radii=0.01..1.00 step 0.01, anchor r0=0.1
cpu_policy=max 19 logical CPUs out of 32 (<=60%), CPU-only, GPU 0%
shards=$REFPOOL1024_SHARDS
threads_per_shard=$REFPOOL1024_CPU_THREADS_PER_SHARD
smoke_rule_units=8
smoke_rule_mean_elapsed_s=2.556
conservative_estimate_note=small-d smoke underestimates larger radii; rule stage was increased to 17 shards after confirming the 15-shard run used about 52% of 32 logical CPUs.
EOF

export REFPOOL1024_DIRECT_DERIVATIVE="${REFPOOL1024_DIRECT_DERIVATIVE:-1}"

exec "$ROOT/04_sampling/src/run_refpool1024_pinned.sh"
