#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_direct_derivative_full_rules_30ref_cpu60_gpu0.sh"
bash "${SCRIPT_DIR}/run_direct_derivative_full_eta_30ref_cpu60_gpu0.sh"

echo "[direct-all] complete"
