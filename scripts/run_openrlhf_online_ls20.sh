#!/usr/bin/env bash
set -euo pipefail

exec bash training/openrlhf/run_online_grpo_ls20_1gpu.sh "$@"
