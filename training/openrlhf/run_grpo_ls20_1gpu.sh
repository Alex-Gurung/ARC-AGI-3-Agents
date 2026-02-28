#!/usr/bin/env bash
set -euo pipefail

# Single-GPU OpenRLHF GRPO (group_norm) + hybrid/colocate setup for ARC ls20.
#
# Prereqs:
#   1) openrlhf installed in current environment
#   2) CUDA GPU visible
#   3) arc-agi env dependencies available
#
# Usage:
#   bash training/openrlhf/run_grpo_ls20_1gpu.sh
#
# Optional env vars:
#   MODEL=google/gemma-3-1b-it
#   DATASET_PATH=training/openrlhf/data/ls20_prompts.jsonl
#   SAVE_PATH=checkpoints/openrlhf-ls20-grpo
#   MAX_SAMPLES=2048
#   N_SAMPLES_PER_PROMPT=4
#   ROLLOUT_BATCH_SIZE=8
#   TRAIN_BATCH_SIZE=32
#   MAX_STEPS_PER_EPISODE=200
#   VLLM_GPU_UTIL=0.45

MODEL="${MODEL:-google/gemma-3-1b-it}"
DATASET_PATH="${DATASET_PATH:-training/openrlhf/data/ls20_prompts.jsonl}"
SAVE_PATH="${SAVE_PATH:-checkpoints/openrlhf-ls20-grpo}"
CKPT_PATH="${CKPT_PATH:-${SAVE_PATH}/ckpt}"
MAX_SAMPLES="${MAX_SAMPLES:-2048}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-1}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-1}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-200}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-4096}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-1024}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.5}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
INIT_KL_COEF="${INIT_KL_COEF:-0.00}"

mkdir -p "$(dirname "${DATASET_PATH}")" "${SAVE_PATH}" "${CKPT_PATH}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found at ${DATASET_PATH}; generating default ls20 prompts..."
  uv run python training/openrlhf/make_ls20_dataset.py --output "${DATASET_PATH}" --num-rows 512
fi

if ! uv run python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('openrlhf') else 1)
PY
then
  echo "openrlhf is not installed in this environment."
  echo "Install it first, then rerun this script."
  exit 1
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "ray CLI not found. Install ray/openrlhf dependencies first."
  exit 1
fi

# Start a local single-node ray head if none is running.
if ! ray status >/dev/null 2>&1; then
  echo "Starting Ray head..."
  ray start --head --num-gpus 1 --disable-usage-stats
fi

echo "Launching OpenRLHF GRPO training on ls20"
echo "MODEL=${MODEL}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "SAVE_PATH=${SAVE_PATH}"

# NOTE:
# - group_norm = GRPO-style grouped advantages
# - colocate_all_models enables hybrid colocated model execution
# - do NOT combine async_train with colocate_all_models
uv run python -m openrlhf.cli.train_ppo_ray \
  --pretrain "${MODEL}" \
  --prompt_data "${DATASET_PATH}" \
  --input_key observation \
  --label_key label \
  --advantage_estimator group_norm \
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}" \
  --agent_func_path training/openrlhf/agent_func_ls20.py \
  --max_steps_per_episode "${MAX_STEPS_PER_EPISODE}" \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --ref_num_nodes 0 \
  --ref_num_gpus_per_node 0 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL}" \
  --colocate_all_models \
  --vllm_enable_sleep \
  --deepspeed_enable_sleep \
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
  --prompt_max_len "${PROMPT_MAX_LEN}" \
  --generate_max_len "${GENERATE_MAX_LEN}" \
  --max_samples "${MAX_SAMPLES}" \
  --actor_learning_rate "${ACTOR_LR}" \
  --init_kl_coef "${INIT_KL_COEF}" \
  --use_kl_loss \
  --normalize_reward \
  --packing_samples \
  --save_path "${SAVE_PATH}" \
  --ckpt_path "${CKPT_PATH}" \
  --save_steps 100 \
  --logging_steps 1 \
  --bf16
