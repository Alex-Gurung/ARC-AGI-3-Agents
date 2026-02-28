#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper.
exec bash training/openrlhf/run_grpo_ls20_1gpu.sh "$@"
