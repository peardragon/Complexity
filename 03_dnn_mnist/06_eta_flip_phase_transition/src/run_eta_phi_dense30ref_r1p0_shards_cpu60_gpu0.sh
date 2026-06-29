#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bjyong/Complexity/local_project/03_dnn_mnist"
STAGE="$ROOT/06_eta_flip_phase_transition"
RUN_NAME="eta_reference_phi_dense_4eta_30ref_r0p01_to_1p0_step0p01_n1024_cpu60_gpu0"
RUN_ROOT="$STAGE/raw_outputs/$RUN_NAME"
ETA_LIST="${ETA_DENSE_LIST:-0.05,0.15,0.25}"
REF_COUNT="${ETA_DENSE_REF_COUNT:-30}"
OLD_REF_ROOT="$STAGE/raw_outputs/eta_reference_search_advanced_4eta_90ref_cpu35_gpu0"
LOW_REF_RUN_NAME="eta_reference_search_gapfill_0p05_0p15_30ref_cpu60_gpu0"
LOW_REF_ROOT="$STAGE/raw_outputs/$LOW_REF_RUN_NAME"
REF_ROOT="$STAGE/raw_outputs/eta_reference_search_gapfill_0p05_0p15_0p25_30ref_cpu60_gpu0"
LOG_DIR="$RUN_ROOT/logs"
SHARDS="${ETA_DENSE_SHARDS:-15}"
THREADS="${ETA_DENSE_THREADS_PER_SHARD:-1}"
MAX_LOGICAL_CPUS="${ETA_DENSE_MAX_LOGICAL_CPUS:-19}"
REFERENCE_THREADS="${ETA_DENSE_REFERENCE_THREADS:-8}"

RADIUS_GRID="$(
  python - <<'PY'
print(",".join(f"{idx / 100:.2f}".rstrip("0").rstrip(".") for idx in range(1, 101)))
PY
)"

mkdir -p "$LOG_DIR"
date -Is > "$RUN_ROOT/MANAGER_STARTED_AT.txt"
: > "$RUN_ROOT/RUNNING_PIDS.txt"

cat > "$RUN_ROOT/DENSE30_RUN_PLAN.txt" <<EOF
run_name=$RUN_NAME
run_root=$RUN_ROOT
reference_run_root=$REF_ROOT
eta_list=$ETA_LIST
scope=eta 0.05,0.15,0.25, 30 references per eta, n=1024
radii=0.01..1.00 step 0.01, anchor r0=0.1
cpu_policy=max $MAX_LOGICAL_CPUS logical CPUs out of 32 (<=60%), CPU-only, GPU 0%
shards=$SHARDS
threads_per_shard=$THREADS
reference_search_threads=$REFERENCE_THREADS
save_unit_samples_npz=true
smoke_eta_units=8
smoke_eta_mean_elapsed_s=5.349
conservative_estimate_note=small-d smoke underestimates larger radii; 3 eta x 30 ref x 100 radii gives 9000 units after the gap-fill reference search.
EOF

needs_reference_pool() {
  python - "$REF_ROOT" "$ETA_LIST" "$REF_COUNT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
etas = [float(part) for part in sys.argv[2].split(",") if part.strip()]
ref_count = int(sys.argv[3])
path = root / "04_exact_reference_search" / "reference_index.csv"
if not path.exists():
    raise SystemExit(1)
refs = pd.read_csv(path)
if "eta" not in refs.columns:
    raise SystemExit(1)
refs["eta"] = pd.to_numeric(refs["eta"], errors="coerce")
for eta in etas:
    if int((refs["eta"].sub(eta).abs() < 1e-9).sum()) < ref_count:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

prepare_reference_pool() {
  if needs_reference_pool; then
    echo "[eta-dense] reference pool already ready: $REF_ROOT"
    return 0
  fi

  echo "[eta-dense] preparing gap-fill reference pool for eta list: $ETA_LIST"
  mkdir -p "$LOW_REF_ROOT"
  if ! python - "$LOW_REF_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
path = root / "04_exact_reference_search" / "reference_index.csv"
if not path.exists():
    raise SystemExit(1)
refs = pd.read_csv(path)
refs["eta"] = pd.to_numeric(refs["eta"], errors="coerce")
for eta in (0.05, 0.15):
    if int((refs["eta"].sub(eta).abs() < 1e-9).sum()) < 30:
        raise SystemExit(1)
raise SystemExit(0)
PY
  then
    (
      cd "$ROOT"
      exec taskset -c "0-$((MAX_LOGICAL_CPUS - 1))" nice -n 10 python "$STAGE/src/run_eta_reference_search_smoke.py" \
        --run-name "$LOW_REF_RUN_NAME" \
        --etas 0.05,0.15 \
        --target-refs 30 \
        --max-attempts 600 \
        --batch-size 8 \
        --max-epochs 1200 \
        --lr 0.022 \
        --cpu-threads "$REFERENCE_THREADS"
    ) > "$LOG_DIR/reference_search_gapfill_0p05_0p15.log" 2>&1
  fi

  python - "$REF_ROOT" "$LOW_REF_ROOT" "$OLD_REF_ROOT" "$ETA_LIST" "$REF_COUNT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

out_root = Path(sys.argv[1])
low_root = Path(sys.argv[2])
old_root = Path(sys.argv[3])
etas = [float(part) for part in sys.argv[4].split(",") if part.strip()]
ref_count = int(sys.argv[5])

def eta_token(eta: float) -> str:
    return f"eta_{eta:.2f}".replace(".", "p")

def load_refs(root: Path) -> pd.DataFrame:
    path = root / "04_exact_reference_search" / "reference_index.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    refs = pd.read_csv(path)
    refs["eta"] = pd.to_numeric(refs.get("eta"), errors="coerce")
    return refs

low_refs = load_refs(low_root)
old_refs = load_refs(old_root)
parts = []
available = {}
for eta in etas:
    source = old_refs if abs(eta - 0.25) < 1e-9 else low_refs
    sub = source[source["eta"].sub(eta).abs() < 1e-9].sort_values("ref_id").head(ref_count).copy()
    if len(sub) < ref_count:
        raise RuntimeError(f"reference pool short for eta={eta:.2f}: {len(sub)} < {ref_count}")
    sub["rule"] = eta_token(eta)
    sub["eta"] = float(eta)
    available[eta_token(eta)] = int(len(sub))
    parts.append(sub)

merged = pd.concat(parts, ignore_index=True)
ref_dir = out_root / "04_exact_reference_search"
ref_dir.mkdir(parents=True, exist_ok=True)
(out_root / "01_dataset_gen").mkdir(parents=True, exist_ok=True)
merged.to_csv(ref_dir / "reference_index.csv", index=False)
status = {
    "status": "complete",
    "etas": etas,
    "reference_rows": int(len(merged)),
    "expected_reference_rows": int(len(etas) * ref_count),
    "refs_per_eta": available,
    "ref_count_for_sampling": ref_count,
    "sources": {
        "eta_0p05_0p15": str(low_root),
        "eta_0p25": str(old_root),
    },
    "note": "Merged gap-fill pool: new eta 0.05/0.15 exact references plus existing eta 0.25 references.",
}
payload = json.dumps(status, indent=2, sort_keys=True) + "\n"
(out_root / "REFERENCE_SEARCH_STATUS.json").write_text(payload, encoding="utf-8")
(ref_dir / "REFERENCE_SEARCH_STATUS.json").write_text(payload, encoding="utf-8")
(out_root / "run_config_resolved.json").write_text(payload, encoding="utf-8")
print(payload, end="")
PY

  needs_reference_pool
}

prepare_reference_pool

export CUDA_VISIBLE_DEVICES=""
export MNIST14_DEVICE="cpu"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TORCH_NUM_THREADS="$THREADS"
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONUNBUFFERED=1

cpu_cursor=0
two_core_shards=$((MAX_LOGICAL_CPUS - SHARDS))
if [[ "$two_core_shards" -lt 0 ]]; then
  two_core_shards=0
fi
if [[ "$two_core_shards" -gt "$SHARDS" ]]; then
  two_core_shards="$SHARDS"
fi

for shard in $(seq 0 $((SHARDS - 1))); do
  if [[ "$MAX_LOGICAL_CPUS" -ge "$SHARDS" ]]; then
    if [[ "$shard" -lt "$two_core_shards" ]]; then
      cpu_list="${cpu_cursor},$((cpu_cursor + 1))"
      cpu_cursor=$((cpu_cursor + 2))
    else
      cpu_list="$cpu_cursor"
      cpu_cursor=$((cpu_cursor + 1))
    fi
  else
    cpu_list="$((shard % MAX_LOGICAL_CPUS))"
  fi
  (
    cd "$ROOT"
    exec taskset -c "$cpu_list" nice -n 10 python "$STAGE/src/run_eta_reference_phi_smoke.py" \
      --run-name "$RUN_NAME" \
      --reference-run-root "$REF_ROOT" \
      --etas "$ETA_LIST" \
      --radii "$RADIUS_GRID" \
      --ref-count "$REF_COUNT" \
      --samples-per-ref-radius 1024 \
      --cpu-threads "$THREADS" \
      --shard-index "$shard" \
      --shard-count "$SHARDS" \
      --save-samples-npz \
      --no-final-aggregate
  ) > "$LOG_DIR/shard${shard}_of_${SHARDS}.log" 2>&1 &
  echo "$! shard${shard}_of_${SHARDS} cpus_${cpu_list}" >> "$RUN_ROOT/RUNNING_PIDS.txt"
done

status=0
while read -r pid _; do
  wait "$pid" || status=1
done < "$RUN_ROOT/RUNNING_PIDS.txt"

cd "$ROOT"
python "$STAGE/src/run_eta_reference_phi_smoke.py" \
  --run-name "$RUN_NAME" \
  --reference-run-root "$REF_ROOT" \
  --etas "$ETA_LIST" \
  --radii "$RADIUS_GRID" \
  --ref-count "$REF_COUNT" \
  --samples-per-ref-radius 1024 \
  --cpu-threads 1 \
  --save-samples-npz \
  --aggregate-only > "$LOG_DIR/final_aggregate.log" 2>&1 || status=1

date -Is > "$RUN_ROOT/MANAGER_FINISHED_AT.txt"
exit "$status"
