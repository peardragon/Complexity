#!/usr/bin/env bash
set -u

ROOT="/home/bjyong/Complexity/local_project/03_dnn_mnist"

export REFPOOL1024_RUN_ROOT="${REFPOOL1024_RUN_ROOT:-$ROOT/04_sampling/raw_outputs/very_low_tv_spectral_teacher_refpool1024_advanced_90ref}"
export REFPOOL1024_TARGET_REFS="${REFPOOL1024_TARGET_REFS:-90}"
export REFPOOL1024_RADIUS_GRID="${REFPOOL1024_RADIUS_GRID:-advanced}"
export REFPOOL1024_RADIUS_COUNT="${REFPOOL1024_RADIUS_COUNT:-49}"
export REFPOOL1024_SHARDS="${REFPOOL1024_SHARDS:-16}"
export REFPOOL1024_CPU_THREADS_PER_SHARD="${REFPOOL1024_CPU_THREADS_PER_SHARD:-2}"

exec "$ROOT/04_sampling/src/run_refpool1024_pinned.sh"
