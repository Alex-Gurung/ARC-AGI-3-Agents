#!/usr/bin/env bash
set -euo pipefail

# Iterative on-policy OpenRLHF GRPO loop for ls20.
#
# Optional env vars:
#   MODEL=google/gemma-3-1b-it
#   AGENT_FUNC_PATH=training/openrlhf/agent_func_loopagent_ls20.py
#   ONLINE_ITERS=3
#   MAX_SAMPLES_PER_ITER=1024
#   DATASET_PATH=training/openrlhf/data/ls20_prompts.jsonl
#   OUTPUT_DIR=training/openrlhf/data/online
#   MEMORY_INIT_CARRY_P=0.60
#   MEMORY_INIT_NOISY_P=0.25
#   MEMORY_INIT_BLANK_P=0.15
#   NOISY_DELETE_FRACTION=0.20
#   NOISY_CONF_JITTER=0.10

MODEL="${MODEL:-google/gemma-3-1b-it}"
AGENT_FUNC_PATH="${AGENT_FUNC_PATH:-training/openrlhf/agent_func_loopagent_ls20.py}"
ONLINE_ITERS="${ONLINE_ITERS:-3}"
MAX_SAMPLES_PER_ITER="${MAX_SAMPLES_PER_ITER:-1024}"
DATASET_PATH="${DATASET_PATH:-training/openrlhf/data/ls20_prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-training/openrlhf/data/online}"
MEMORY_INIT_CARRY_P="${MEMORY_INIT_CARRY_P:-0.60}"
MEMORY_INIT_NOISY_P="${MEMORY_INIT_NOISY_P:-0.25}"
MEMORY_INIT_BLANK_P="${MEMORY_INIT_BLANK_P:-0.15}"
NOISY_DELETE_FRACTION="${NOISY_DELETE_FRACTION:-0.20}"
NOISY_CONF_JITTER="${NOISY_CONF_JITTER:-0.10}"

mkdir -p "${OUTPUT_DIR}"

echo "Running OpenRLHF online GRPO loop"
echo "MODEL=${MODEL}"
echo "AGENT_FUNC_PATH=${AGENT_FUNC_PATH}"
echo "ONLINE_ITERS=${ONLINE_ITERS}"
echo "MAX_SAMPLES_PER_ITER=${MAX_SAMPLES_PER_ITER}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MEMORY_INIT_CARRY_P=${MEMORY_INIT_CARRY_P}"
echo "MEMORY_INIT_NOISY_P=${MEMORY_INIT_NOISY_P}"
echo "MEMORY_INIT_BLANK_P=${MEMORY_INIT_BLANK_P}"

uv run python -m training.openrlhf.online_grpo_ls20 \
  --model "${MODEL}" \
  --agent-func-path "${AGENT_FUNC_PATH}" \
  --iterations "${ONLINE_ITERS}" \
  --max-samples-per-iter "${MAX_SAMPLES_PER_ITER}" \
  --memory-init-carry-p "${MEMORY_INIT_CARRY_P}" \
  --memory-init-noisy-p "${MEMORY_INIT_NOISY_P}" \
  --memory-init-blank-p "${MEMORY_INIT_BLANK_P}" \
  --noisy-delete-fraction "${NOISY_DELETE_FRACTION}" \
  --noisy-conf-jitter "${NOISY_CONF_JITTER}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}"
