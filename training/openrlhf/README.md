# OpenRLHF LS20 Scaffold (Baseline)

This directory contains a minimal GRPO-style OpenRLHF setup for ARC-AGI-3 `ls20`.
It is kept as a baseline/simple path alongside the richer veRL grouped-rollout
path in `training/verl/`.

## Files

- `make_ls20_dataset.py`: creates JSONL prompt data for rollout initialization.
- `agent_func_ls20.py`: OpenRLHF multi-turn agent function that runs the ARC env.
- `run_grpo_ls20_1gpu.sh`: single-GPU launch script with hybrid colocate settings.

## What this trains

This scaffold trains a single policy to emit one environment action per turn:

- expected output: `ANSWER: ACTION1` (or `ANSWER: ACTION6 x y`)
- reward: novelty + transition magnitude + level progress + terminal outcomes
- game: fixed to `ls20` by default

## Quick start

```bash
uv run python training/openrlhf/make_ls20_dataset.py
bash training/openrlhf/run_grpo_ls20_1gpu.sh
```

## Notes

- `--advantage_estimator group_norm` is used (GRPO-style grouped advantages).
- `--colocate_all_models` is enabled for hybrid colocated execution.
- Do not combine `--async_train` with `--colocate_all_models`.
- Vision in this scaffold is not wired into OpenRLHF actor prompts by default; it relies on text state encoding from `StateEncoder`.
- State text includes `OBJECTS/RELATIONS` as heuristic/noisy descriptors; `GRID/DIFF` should be treated as ground truth.
- For LoopAgent-complete grouped branching + semantic surprise rewards, use `training/verl/`.

## Common knobs

Environment variables for `run_grpo_ls20_1gpu.sh`:

- `MODEL`
- `N_SAMPLES_PER_PROMPT`
- `MAX_SAMPLES`
- `MAX_STEPS_PER_EPISODE`
- `ROLLOUT_BATCH_SIZE`, `TRAIN_BATCH_SIZE`
- `VLLM_GPU_UTIL`
