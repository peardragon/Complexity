#!/usr/bin/env bash
set -u

ROOT="/home/bjyong/Complexity/local_project/03_dnn_mnist"
RUN60="${REFPOOL1024_RUN60_ROOT:-$ROOT/04_sampling/raw_outputs/refpool1024_all_radii_60ref}"
RUN90="${REFPOOL1024_RUN90_ROOT:-$ROOT/04_sampling/raw_outputs/refpool1024_all_radii_90ref}"
INTERVAL_SECONDS="${REFPOOL1024_CHAIN_INTERVAL_SECONDS:-60}"
CHAIN_STATUS="$RUN60/CHAIN_60_TO_90_STATUS.txt"
CHAIN_LOG="$RUN60/logs/chain_60_to_90.log"

mkdir -p "$RUN60/logs" "$RUN90/logs"

write_status() {
  {
    printf 'checked_at=%s\n' "$(date -Is)"
    printf 'stage=%s\n' "$1"
    shift || true
    for line in "$@"; do
      printf '%s\n' "$line"
    done
  } > "$CHAIN_STATUS.tmp"
  mv "$CHAIN_STATUS.tmp" "$CHAIN_STATUS"
}

alive_from_pid_file() {
  local pid_file="$1"
  local alive=0
  if [[ -f "$pid_file" ]]; then
    while read -r pid _rest; do
      [[ -z "${pid:-}" ]] && continue
      if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done < "$pid_file"
  fi
  printf '%s\n' "$alive"
}

aggregate_run() {
  local run_root="$1"
  local target_refs="$2"
  (
    cd "$ROOT" || exit 1
    python 04_sampling/src/sample_refpool1024_all_radii.py \
      --aggregate-only \
      --run-root "$run_root" \
      --target-refs "$target_refs" \
      --device cpu \
      --cpu-threads 1
  ) >> "$CHAIN_LOG" 2>&1
}

seed_90_from_60() {
  local src_root="$RUN60/05_pool2_pm_sais_sampling/unit_summaries/split_000"
  local dst_root="$RUN90/05_pool2_pm_sais_sampling/unit_summaries/split_000"
  local seeded_refs=0
  local units
  local npz

  if [[ ! -d "$src_root" ]]; then
    write_status "seed_90ref_failed" "missing_source=$src_root"
    return 3
  fi

  write_status "seeding_90ref_from_60ref" "source=$RUN60" "target=$RUN90"
  mkdir -p "$dst_root"
  for rule_dir in "$src_root"/*; do
    [[ -d "$rule_dir" ]] || continue
    local rule
    rule="$(basename "$rule_dir")"
    mkdir -p "$dst_root/$rule"
    for ref_dir in "$rule_dir"/ref_*; do
      [[ -d "$ref_dir" ]] || continue
      local dst_ref
      dst_ref="$dst_root/$rule/$(basename "$ref_dir")"
      mkdir -p "$dst_ref"
      cp -an "$ref_dir"/. "$dst_ref"/ 2>> "$CHAIN_LOG"
      seeded_refs=$((seeded_refs + 1))
    done
  done

  (
    cd "$ROOT" || exit 1
    python - "$RUN90" <<'PY'
import json
import os
from pathlib import Path
import sys

run_root = Path(sys.argv[1])
summary_root = run_root / "05_pool2_pm_sais_sampling" / "unit_summaries"
for path in summary_root.rglob("unit_summary.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    samples_path = path.with_name("samples.npz")
    if samples_path.exists():
        payload["samples_path"] = str(samples_path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
PY
  ) >> "$CHAIN_LOG" 2>&1

  units="$(find "$RUN90/05_pool2_pm_sais_sampling/unit_summaries" -name unit_summary.json 2>/dev/null | wc -l | tr -d ' ')"
  npz="$(find "$RUN90/05_pool2_pm_sais_sampling/unit_summaries" -name samples.npz 2>/dev/null | wc -l | tr -d ' ')"
  write_status "seeded_90ref_from_60ref" "seeded_ref_dirs=$seeded_refs" "unit_summary_count=$units" "samples_npz_count=$npz"
  if [[ "$units" -lt 6000 || "$npz" -lt 6000 ]]; then
    return 3
  fi
}

audit_run() {
  local run_root="$1"
  local target_refs="$2"
  (
    cd "$ROOT" || exit 1
    python 04_sampling/src/audit_refpool1024_run.py \
      --run-root "$run_root" \
      --target-refs "$target_refs" \
      --samples-per-ref-radius 1024 \
      --check-npz-arrays
  ) > "$run_root/AUDIT.json.tmp" 2>> "$CHAIN_LOG"
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    mv "$run_root/AUDIT.json.tmp" "$run_root/AUDIT.json"
  else
    mv "$run_root/AUDIT.json.tmp" "$run_root/AUDIT_FAILED.json" 2>/dev/null || true
  fi
  return "$rc"
}

start_90_if_needed() {
  local alive
  alive="$(alive_from_pid_file "$RUN90/RUNNING_PIDS.txt")"
  if [[ "$alive" -gt 0 ]]; then
    write_status "90ref_already_running" "run90=$RUN90" "alive_shards=$alive"
    return 0
  fi
  if [[ -f "$RUN90/AUDIT.json" ]]; then
    write_status "90ref_already_audited" "run90=$RUN90"
    return 0
  fi

  write_status "starting_90ref" "run90=$RUN90"
  (
    cd "$ROOT" || exit 1
    env \
      REFPOOL1024_RUN_ROOT="$RUN90" \
      REFPOOL1024_TARGET_REFS=90 \
      REFPOOL1024_SHARDS="${REFPOOL1024_SHARDS:-16}" \
      REFPOOL1024_CPU_THREADS_PER_SHARD="${REFPOOL1024_CPU_THREADS_PER_SHARD:-2}" \
      setsid 04_sampling/src/run_refpool1024_pinned.sh \
        > "$RUN90/logs/manager_stdout.log" 2>&1 &
    printf '%s\n' "$!" > "$RUN90/MANAGER_PID.txt"
    env \
      REFPOOL1024_RUN_ROOT="$RUN90" \
      REFPOOL1024_TARGET_REFS=90 \
      REFPOOL1024_MONITOR_INTERVAL_SECONDS="${REFPOOL1024_MONITOR_INTERVAL_SECONDS:-600}" \
      setsid 04_sampling/src/monitor_refpool1024_progress.sh \
        > "$RUN90/logs/monitor_stdout.log" 2>&1 &
    printf '%s\n' "$!" > "$RUN90/MONITOR_PID.txt"
  )
}

wait_for_run_to_finish() {
  local run_root="$1"
  local label="$2"
  local expected_units="$3"
  local pid_file="$run_root/RUNNING_PIDS.txt"
  local manager_pid_file="$run_root/MANAGER_PID.txt"
  while true; do
    local alive manager_alive units npz
    alive="$(alive_from_pid_file "$pid_file")"
    manager_alive="$(alive_from_pid_file "$manager_pid_file")"
    units="$(find "$run_root/05_pool2_pm_sais_sampling/unit_summaries" -name unit_summary.json 2>/dev/null | wc -l | tr -d ' ')"
    npz="$(find "$run_root/05_pool2_pm_sais_sampling/unit_summaries" -name samples.npz 2>/dev/null | wc -l | tr -d ' ')"
    write_status "waiting_${label}" "run_root=$run_root" "alive_shards=$alive" "alive_manager=$manager_alive" "unit_summary_count=$units" "samples_npz_count=$npz" "expected_units=$expected_units"
    if [[ "$alive" -eq 0 && "$manager_alive" -eq 0 && -f "$pid_file" && "$units" -ge "$expected_units" && "$npz" -ge "$expected_units" ]]; then
      break
    fi
    sleep "$INTERVAL_SECONDS"
  done
}

main() {
  printf '[chain] started_at=%s\n' "$(date -Is)" >> "$CHAIN_LOG"

  wait_for_run_to_finish "$RUN60" "60ref" 6000
  write_status "aggregating_60ref" "run60=$RUN60"
  aggregate_run "$RUN60" 60
  write_status "auditing_60ref" "run60=$RUN60"
  if ! audit_run "$RUN60" 60; then
    write_status "60ref_audit_failed" "run60=$RUN60" "audit=$RUN60/AUDIT_FAILED.json"
    return 2
  fi
  write_status "60ref_audit_passed" "run60=$RUN60" "audit=$RUN60/AUDIT.json"

  if ! seed_90_from_60; then
    write_status "90ref_seed_failed" "run60=$RUN60" "run90=$RUN90"
    return 3
  fi
  start_90_if_needed
  wait_for_run_to_finish "$RUN90" "90ref" 9000
  write_status "aggregating_90ref" "run90=$RUN90"
  aggregate_run "$RUN90" 90
  write_status "auditing_90ref" "run90=$RUN90"
  if ! audit_run "$RUN90" 90; then
    write_status "90ref_audit_failed" "run90=$RUN90" "audit=$RUN90/AUDIT_FAILED.json"
    return 2
  fi
  write_status "complete" "run60=$RUN60" "run90=$RUN90" "audit60=$RUN60/AUDIT.json" "audit90=$RUN90/AUDIT.json"
}

main "$@"
