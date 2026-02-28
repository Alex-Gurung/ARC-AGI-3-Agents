# Training Backends Overview

This repo currently supports two training backends:

1. `openrlhf` (`training/openrlhf/`)
2. `verl` (`training/verl/`)

Both paths are configured for `ls20` first and support iterative on-policy-style
training loops on a single GPU.

## OpenRLHF

- Single job:
  - `bash training/openrlhf/run_grpo_ls20_1gpu.sh`
- Iterative loop:
  - `bash training/openrlhf/run_online_grpo_ls20_1gpu.sh`
  - `bash scripts/run_openrlhf_online_ls20.sh`

See: `training/openrlhf/README.md`

## veRL

- Grouped rollout collection:
  - `bash training/verl/run_grpo_ls20_1gpu.sh`
- Online grouped loop:
  - `bash training/verl/run_online_grpo_ls20_1gpu.sh`
  - `bash scripts/run_verl_online_ls20.sh`

For veRL online runs, set `VERL_GRPO_UPDATE_CMD` so each iteration performs an
actual update step (default behavior enforces this).

See: `training/verl/README.md`

## Shared Notes

- Inference remains single-trajectory.
- Grouped candidate branching is training-only.
- `GRID/DIFF` is treated as ground truth.
- `OBJECTS/RELATIONS` is heuristic/noisy.
