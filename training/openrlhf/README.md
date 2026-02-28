# OpenRLHF LS20 Training Path

This directory contains the OpenRLHF GRPO training path for ARC-AGI-3 `ls20`.
It is maintained alongside the veRL grouped-rollout path in `training/verl/`.

## Files

- `make_ls20_dataset.py`: creates JSONL prompt data for rollout initialization.
- `agent_func_ls20.py`: OpenRLHF multi-turn agent function that runs the ARC env.
- `run_grpo_ls20_1gpu.sh`: single-GPU launch script with hybrid colocate settings.
- `online_grpo_ls20.py`: iterative on-policy runner wrapper for multi-iteration jobs.
- `run_online_grpo_ls20_1gpu.sh`: shell entrypoint for iterative OpenRLHF runs.

## What This Trains

This path trains a single policy to emit one environment action per turn:

- expected output: `ANSWER: ACTION1` (or `ANSWER: ACTION6 x y`)
- reward: novelty + transition magnitude + level progress + terminal outcomes
- game: fixed to `ls20` by default

## Quick Start (Single Job)

```bash
uv run python training/openrlhf/make_ls20_dataset.py
bash training/openrlhf/run_grpo_ls20_1gpu.sh
```

## Online Loop (Iterative)

```bash
bash training/openrlhf/run_online_grpo_ls20_1gpu.sh
# or
bash scripts/run_openrlhf_online_ls20.sh
```

Per iteration:

1. run one OpenRLHF GRPO job (`run_grpo_ls20_1gpu.sh`) with the current model,
2. resolve latest model/checkpoint path,
3. feed that model into the next iteration.

Outputs:

- per-iteration model directories under `OUTPUT_DIR/iter_xxxx/`
- `OUTPUT_DIR/online_summary.json`
- `OUTPUT_DIR/latest_model.txt`

## Notes

- `--advantage_estimator group_norm` is used (GRPO-style grouped advantages).
- `--colocate_all_models` is enabled for hybrid colocated execution.
- Do not combine `--async_train` with `--colocate_all_models`.
- Vision in this scaffold is not wired into OpenRLHF actor prompts by default; it relies on text state encoding from `StateEncoder`.
- State text includes `OBJECTS/RELATIONS` as heuristic/noisy descriptors; `GRID/DIFF` should be treated as ground truth.
- For LoopAgent-complete grouped branching + semantic surprise rewards, use `training/verl/`.

## Common Knobs

Environment variables for `run_grpo_ls20_1gpu.sh`:

- `MODEL`
- `N_SAMPLES_PER_PROMPT`
- `MAX_SAMPLES`
- `MAX_STEPS_PER_EPISODE`
- `ROLLOUT_BATCH_SIZE`, `TRAIN_BATCH_SIZE`
- `VLLM_GPU_UTIL`

Additional environment variables for `run_online_grpo_ls20_1gpu.sh`:

- `ONLINE_ITERS`
- `MAX_SAMPLES_PER_ITER`
- `OUTPUT_DIR`
- `DATASET_PATH`
