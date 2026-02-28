#!/usr/bin/env bash
set -euo pipefail

# Iterative on-policy OpenRLHF GRPO loop for ls20.
#
# Optional env vars:
#   MODEL=google/gemma-3-1b-it
#   ONLINE_ITERS=3
#   MAX_SAMPLES_PER_ITER=1024
#   DATASET_PATH=training/openrlhf/data/ls20_prompts.jsonl
#   OUTPUT_DIR=training/openrlhf/data/online

MODEL="${MODEL:-google/gemma-3-1b-it}"
ONLINE_ITERS="${ONLINE_ITERS:-3}"
MAX_SAMPLES_PER_ITER="${MAX_SAMPLES_PER_ITER:-1024}"
DATASET_PATH="${DATASET_PATH:-training/openrlhf/data/ls20_prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-training/openrlhf/data/online}"

mkdir -p "${OUTPUT_DIR}"

echo "Running OpenRLHF online GRPO loop"
echo "MODEL=${MODEL}"
echo "ONLINE_ITERS=${ONLINE_ITERS}"
echo "MAX_SAMPLES_PER_ITER=${MAX_SAMPLES_PER_ITER}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

uv run python -m training.openrlhf.online_grpo_ls20 \
  --model "${MODEL}" \
  --iterations "${ONLINE_ITERS}" \
  --max-samples-per-iter "${MAX_SAMPLES_PER_ITER}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}"
