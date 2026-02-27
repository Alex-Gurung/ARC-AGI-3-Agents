#!/bin/bash
# Start VLLM server with gemma-3-1b-it on 1 GPU.
#
# Usage:
#   bash scripts/run_vllm.sh
#
# Environment variables:
#   VLLM_MODEL  - model to serve (default: google/gemma-3-1b-it)
#   VLLM_PORT   - port to serve on (default: 8000)
#   VLLM_GPU    - CUDA device (default: 0)

set -euo pipefail

MODEL="${VLLM_MODEL:-google/gemma-3-1b-it}"
PORT="${VLLM_PORT:-8000}"
GPU="${VLLM_GPU:-0}"

echo "Starting VLLM server..."
echo "  Model: ${MODEL}"
echo "  Port:  ${PORT}"
echo "  GPU:   ${GPU}"

CUDA_VISIBLE_DEVICES="${GPU}" vllm serve "${MODEL}" \
    --port "${PORT}" \
    --max-model-len 8192 \
    --dtype auto \
    --api-key dummy
