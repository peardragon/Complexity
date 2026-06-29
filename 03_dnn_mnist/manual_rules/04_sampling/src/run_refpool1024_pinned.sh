#!/usr/bin/env bash
set -u

ROOT="/home/bjyong/Complexity/Complexity/03_dnn_mnist/manual_rules"
RUN_ROOT="${REFPOOL1024_RUN_ROOT:-$ROOT/04_sampling/raw_outputs/refpool1024_all_radii_60ref}"
TARGET_REFS="${REFPOOL1024_TARGET_REFS:-60}"
RADIUS_GRID="${REFPOOL1024_RADIUS_GRID:-production}"
CUSTOM_RADII="${REFPOOL1024_RADII:-}"
LOG_DIR="$RUN_ROOT/logs"
PID_FILE="$RUN_ROOT/RUNNING_PIDS.txt"
STATUS_FILE="$RUN_ROOT/MANAGER_STATUS.txt"
REBALANCER_PID_FILE="$RUN_ROOT/AFFINITY_REBALANCER_PID.txt"

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

export PYTHONUNBUFFERED=1

total_cpus="$(nproc)"
max_logical_cpus="${REFPOOL1024_MAX_LOGICAL_CPUS:-$((total_cpus * 70 / 100))}"
shards="${REFPOOL1024_SHARDS:-16}"
threads_per_shard="${REFPOOL1024_CPU_THREADS_PER_SHARD:-2}"
rebalance_affinity="${REFPOOL1024_REBALANCE_AFFINITY:-1}"
rebalance_interval="${REFPOOL1024_REBALANCE_INTERVAL_SECONDS:-300}"
status=0
started_at="$(date -Is)"
printf 'started_at=%s\n' "$started_at" > "$STATUS_FILE"
printf 'run_root=%s\n' "$RUN_ROOT" >> "$STATUS_FILE"
printf 'target_refs=%s\n' "$TARGET_REFS" >> "$STATUS_FILE"
printf 'radius_grid=%s\n' "$RADIUS_GRID" >> "$STATUS_FILE"
printf 'custom_radii=%s\n' "$CUSTOM_RADII" >> "$STATUS_FILE"
printf 'shards=%s\n' "$shards" >> "$STATUS_FILE"
printf 'threads_per_shard=%s\n' "$threads_per_shard" >> "$STATUS_FILE"
printf 'resource_policy=taskset_nonoverlap_logical_cpu_set_cap_device_cpu\n' >> "$STATUS_FILE"
printf 'total_logical_cpus=%s\n' "$total_cpus" >> "$STATUS_FILE"
printf 'max_logical_cpus=%s\n' "$max_logical_cpus" >> "$STATUS_FILE"
printf 'rebalance_affinity=%s\n' "$rebalance_affinity" >> "$STATUS_FILE"
printf 'rebalance_interval_seconds=%s\n' "$rebalance_interval" >> "$STATUS_FILE"

cpu_cursor=0
two_core_shards=$((max_logical_cpus - shards))
if [[ "$two_core_shards" -lt 0 ]]; then
  two_core_shards=0
fi
if [[ "$two_core_shards" -gt "$shards" ]]; then
  two_core_shards="$shards"
fi
printf 'two_core_shards=%s\n' "$two_core_shards" >> "$STATUS_FILE"

extra_args=(--radius-grid "$RADIUS_GRID")
if [[ -n "$CUSTOM_RADII" ]]; then
  extra_args+=(--radii "$CUSTOM_RADII")
fi
if [[ "${REFPOOL1024_DIRECT_DERIVATIVE:-1}" == "1" ]]; then
  extra_args+=(--direct-derivative)
fi

for i in $(seq 0 "$((shards - 1))"); do
  if [[ "$max_logical_cpus" -ge "$shards" ]]; then
    if [[ "$i" -lt "$two_core_shards" ]]; then
      cpu_list="${cpu_cursor},$((cpu_cursor + 1))"
      cpu_cursor=$((cpu_cursor + 2))
    else
      cpu_list="$cpu_cursor"
      cpu_cursor=$((cpu_cursor + 1))
    fi
  else
    cpu_list="$((i % max_logical_cpus))"
  fi
  (
    cd "$ROOT" || exit 1
    exec taskset -c "$cpu_list" nice -n 10 python \
      04_sampling/src/sample_refpool1024_all_radii.py \
      --run-root "$RUN_ROOT" \
      --target-refs "$TARGET_REFS" \
      --shard-index "$i" \
      --shard-count "$shards" \
      --no-final-aggregate \
      --aggregate-every 0 \
      --device cpu \
      --cpu-threads "$threads_per_shard" \
      "${extra_args[@]}"
  ) > "$LOG_DIR/shard${shards}_pinned_${i}.log" 2>&1 &
  printf '%s shard%s_pinned_%s cpus_%s\n' "$!" "$shards" "$i" "$cpu_list" >> "$PID_FILE"
done

rebalance_pid=""
if [[ "$rebalance_affinity" == "1" ]]; then
  (
    cd "$ROOT" || exit 1
    exec python 04_sampling/src/rebalance_refpool1024_affinity.py \
      --run-root "$RUN_ROOT" \
      --max-logical-cpus "$max_logical_cpus" \
      --interval-seconds "$rebalance_interval"
  ) > "$LOG_DIR/affinity_rebalance.log" 2>&1 &
  rebalance_pid="$!"
  printf '%s\n' "$rebalance_pid" > "$REBALANCER_PID_FILE"
fi

for pid in $(awk '{print $1}' "$PID_FILE"); do
  wait "$pid" || status=1
done

if [[ -n "${rebalance_pid:-}" ]] && kill -0 "$rebalance_pid" 2>/dev/null; then
  kill "$rebalance_pid" 2>/dev/null || true
  wait "$rebalance_pid" 2>/dev/null || true
fi

(
  cd "$ROOT" || exit 1
  python 04_sampling/src/sample_refpool1024_all_radii.py \
    --aggregate-only \
    --run-root "$RUN_ROOT" \
    --target-refs "$TARGET_REFS" \
    "${extra_args[@]}" \
    --device cpu \
    --cpu-threads 1
) >> "$LOG_DIR/final_aggregate.log" 2>&1 || true

finished_at="$(date -Is)"
printf 'finished_at=%s\n' "$finished_at" >> "$STATUS_FILE"
printf 'exit_status=%s\n' "$status" >> "$STATUS_FILE"
exit "$status"
