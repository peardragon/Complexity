#!/usr/bin/env bash
set -u

ROOT="/home/bjyong/Complexity/local_project/03_dnn_mnist"
RUN_ROOT="${REFPOOL1024_RUN_ROOT:-$ROOT/04_sampling/raw_outputs/refpool1024_all_radii_60ref}"
TARGET_REFS="${REFPOOL1024_TARGET_REFS:-60}"
RADIUS_GRID="${REFPOOL1024_RADIUS_GRID:-production}"
CUSTOM_RADII="${REFPOOL1024_RADII:-}"
PID_FILE="$RUN_ROOT/RUNNING_PIDS.txt"
LOG_DIR="$RUN_ROOT/logs"
MONITOR_STATUS="$RUN_ROOT/MONITOR_STATUS.txt"
PROGRESS_LOG="$LOG_DIR/progress_monitor.log"
INTERVAL_SECONDS="${REFPOOL1024_MONITOR_INTERVAL_SECONDS:-600}"

mkdir -p "$LOG_DIR"
printf 'started_at=%s\n' "$(date -Is)" > "$MONITOR_STATUS"
printf 'run_root=%s\n' "$RUN_ROOT" >> "$MONITOR_STATUS"
printf 'target_refs=%s\n' "$TARGET_REFS" >> "$MONITOR_STATUS"
printf 'radius_grid=%s\n' "$RADIUS_GRID" >> "$MONITOR_STATUS"
printf 'custom_radii=%s\n' "$CUSTOM_RADII" >> "$MONITOR_STATUS"
printf 'interval_seconds=%s\n' "$INTERVAL_SECONDS" >> "$MONITOR_STATUS"

extra_args=(--radius-grid "$RADIUS_GRID")
if [[ -n "$CUSTOM_RADII" ]]; then
  extra_args+=(--radii "$CUSTOM_RADII")
fi

aggregate_once() {
  (
    cd "$ROOT" || exit 1
    python 04_sampling/src/sample_refpool1024_all_radii.py \
      --aggregate-only \
      --run-root "$RUN_ROOT" \
      --target-refs "$TARGET_REFS" \
      "${extra_args[@]}" \
      --device cpu \
      --cpu-threads 1
  ) >> "$PROGRESS_LOG" 2>&1 || true
}

while true; do
  alive=0
  if [[ -f "$PID_FILE" ]]; then
    while read -r pid _rest; do
      [[ -z "${pid:-}" ]] && continue
      if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done < "$PID_FILE"
  fi

  unit_count="$(find "$RUN_ROOT/05_pool2_pm_sais_sampling/unit_summaries" -name unit_summary.json 2>/dev/null | wc -l | tr -d ' ')"
  npz_count="$(find "$RUN_ROOT/05_pool2_pm_sais_sampling/unit_summaries" -name samples.npz 2>/dev/null | wc -l | tr -d ' ')"
  {
    printf 'checked_at=%s\n' "$(date -Is)"
    printf 'alive_shards=%s\n' "$alive"
    printf 'unit_summary_count=%s\n' "$unit_count"
    printf 'samples_npz_count=%s\n' "$npz_count"
  } > "$MONITOR_STATUS.tmp"
  mv "$MONITOR_STATUS.tmp" "$MONITOR_STATUS"

  aggregate_once

  if [[ "$alive" -eq 0 ]]; then
    printf 'finished_at=%s\n' "$(date -Is)" >> "$MONITOR_STATUS"
    exit 0
  fi

  sleep "$INTERVAL_SECONDS"
done
