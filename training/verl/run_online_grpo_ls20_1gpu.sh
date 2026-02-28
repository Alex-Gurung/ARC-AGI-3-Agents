#!/usr/bin/env bash
set -euo pipefail

# Online on-policy grouped rollout loop for ls20.
#
# This script runs iterative collection + optional GRPO update adapter.
#
# Optional env vars:
#   TRAIN_MODEL=google/gemma-3-4b-it
#   ONLINE_ITERS=3
#   ATTEMPTS_PER_ITER=8
#   MAX_STEPS=50
#   OUTPUT_DIR=training/verl/data/online
#   VERL_GRPO_UPDATE_CMD="<your update command template>"
#   REQUIRE_UPDATE_STEP=true|false
#
# Command template placeholders:
#   {model} {rollout_jsonl} {rollouts} {output_dir} {iteration_dir} {iteration}

TRAIN_MODEL="${TRAIN_MODEL:-google/gemma-3-4b-it}"
ONLINE_ITERS="${ONLINE_ITERS:-3}"
ATTEMPTS_PER_ITER="${ATTEMPTS_PER_ITER:-8}"
MAX_STEPS="${MAX_STEPS:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-training/verl/data/online}"
REQUIRE_UPDATE_STEP="${REQUIRE_UPDATE_STEP:-true}"

mkdir -p "${OUTPUT_DIR}"

echo "Running online grouped loop"
echo "TRAIN_MODEL=${TRAIN_MODEL}"
echo "ONLINE_ITERS=${ONLINE_ITERS}"
echo "ATTEMPTS_PER_ITER=${ATTEMPTS_PER_ITER}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REQUIRE_UPDATE_STEP=${REQUIRE_UPDATE_STEP}"

EXTRA_ARGS=()
if [[ "${REQUIRE_UPDATE_STEP}" != "true" ]]; then
  EXTRA_ARGS+=(--allow-collector-only)
fi

uv run python -m training.verl.online_grpo_ls20 \
  --game-id ls20 \
  --model "${TRAIN_MODEL}" \
  --iterations "${ONLINE_ITERS}" \
  --attempts-per-iter "${ATTEMPTS_PER_ITER}" \
  --max-steps "${MAX_STEPS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
