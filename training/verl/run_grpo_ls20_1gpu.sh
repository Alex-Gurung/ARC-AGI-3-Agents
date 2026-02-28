#!/usr/bin/env bash
set -euo pipefail

# veRL grouped-rollout collection script for ls20 on 1 GPU.
#
# This script collects grouped candidate decisions + reward annotations
# using the LoopAgent runtime and semantic surprise signals.
#
# Optional env vars:
#   TRAIN_MODEL=google/gemma-3-4b-it
#   MAX_STEPS=50
#   ATTEMPT_ID=attempt_0001
#   OUTPUT_JSONL=training/verl/data/ls20_grouped_rollouts.jsonl
#   K_CURIOSITY=4
#   K_LEARNER=4
#   K_SOLVER=4
#   SURPRISE_REWARD_SOURCE=debiased_nll|self_rated|hybrid

TRAIN_MODEL="${TRAIN_MODEL:-google/gemma-3-4b-it}"
MAX_STEPS="${MAX_STEPS:-50}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt_0001}"
OUTPUT_JSONL="${OUTPUT_JSONL:-training/verl/data/ls20_grouped_rollouts.jsonl}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

if ! uv run python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('arc_agi') else 1)
PY
then
  echo "arc_agi is not installed in this environment."
  exit 1
fi

echo "Running grouped rollout collector"
echo "TRAIN_MODEL=${TRAIN_MODEL}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"

uv run python -m training.verl.agent_loop_ls20 \
  --game-id ls20 \
  --model "${TRAIN_MODEL}" \
  --attempt-id "${ATTEMPT_ID}" \
  --max-steps "${MAX_STEPS}" \
  --output-jsonl "${OUTPUT_JSONL}"
