#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"
STAGE="${2:?stage name required, e.g. 01_dataset_prepare}"

cd "$REPO_ROOT"

PROMPT="02_dnn/08_mnist/stages/${STAGE}/START_PROMPT.md"
if [[ ! -f "$PROMPT" ]]; then
  echo "Prompt not found: $PROMPT" >&2
  exit 1
fi

codex exec --cd "$REPO_ROOT" --ask-for-approval never --sandbox workspace-write - < "$PROMPT"
