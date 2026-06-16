#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"
cd "$REPO_ROOT"

codex exec --cd "$REPO_ROOT" --ask-for-approval never --sandbox workspace-write - < 02_dnn/08_mnist/prompts/MASTER_GOAL_MODE_START_PROMPT.md
