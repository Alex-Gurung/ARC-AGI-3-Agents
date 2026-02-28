# OpenRLHF LS20 Training Path

This directory contains the OpenRLHF GRPO training path for ARC-AGI-3 `ls20`.
It is maintained alongside the veRL grouped-rollout path in `training/verl/`.

## Files

- `make_ls20_dataset.py`: creates JSONL prompt data for rollout initialization.
- `agent_func_loopagent_ls20.py`: full LoopAgent-style OpenRLHF multi-turn agent function.
- `agent_func_ls20.py`: legacy simple action-only baseline.
- `run_grpo_ls20_1gpu.sh`: single-GPU launch script with hybrid colocate settings.
- `online_grpo_ls20.py`: iterative on-policy runner wrapper for multi-iteration jobs.
- `run_online_grpo_ls20_1gpu.sh`: shell entrypoint for iterative OpenRLHF runs.
- `training/reward_channels.py`: shared role-gated scalarization from reward vectors.

## What This Trains

This path trains a single shared policy across two staged roles:

- `DECIDER`: mode-aware curiosity/solver boundary decisions (action/subgoal/plan)
- `LEARNER`: memory updates from observed transitions

Each boundary transition is followed by a learner step so the model learns
both exploration/exploitation behavior and rulebook updates.

## Full Rollout Flow (Current)

Each environment trajectory alternates:

1. `DECIDER` receives mode + state + memory and outputs:
   - `MODE`, optional `PLAN`, optional `SUBGOAL`
   - `ACTION_SEQUENCE`
   - `EXPECTED` transition
2. Environment executes to boundary:
   - action boundary in `LEARN_ACTION`
   - sequence boundary in `LEARN_SUBGOAL` / `LEARN_PLAN` / `SOLVE`
3. Boundary reward is emitted (curiosity-shaped in learn modes, solver-shaped in solve mode).
4. `LEARNER` receives `BEFORE/AFTER/DIFF` + memory and emits memory ops:
   - `ADD`, `MODIFY`, `REMOVE`, `NONE`
   - optional `NEXT_MODE`
5. Learner reward is emitted and control returns to `DECIDER`.

Done conditions:

- terminal game state (`WIN` / `GAME_OVER`) after learner step
- action budget exhaustion (`MAX_STEPS_PER_EPISODE`)

- expected output: `ANSWER: ACTION1` (or `ANSWER: ACTION6 x y`)
- reward:
  - decider: curiosity/solver-style boundary reward (surprise/progress/terminal)
  - learner: memory-improvement reward from transition explanation/update quality
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
- Default agent path in run scripts is:
  - `AGENT_FUNC_PATH=training/openrlhf/agent_func_loopagent_ls20.py`
- Vision in this OpenRLHF path is not wired into actor prompts by default; it relies on text state encoding from `StateEncoder`.
- State text includes `OBJECTS/RELATIONS` as heuristic/noisy descriptors; `GRID/DIFF` should be treated as ground truth.
- For LoopAgent-complete grouped branching + semantic surprise rewards, use `training/verl/`.

## Grouping Semantics (Important)

OpenRLHF grouping is at the prompt level via:

- `--n_samples_per_prompt` + `--advantage_estimator group_norm`

In this staged agent, that means grouping naturally happens for both roles, because
observations are role-conditioned:

- `ROLE: DECIDER` prompts form one family of grouped experiences
- `ROLE: LEARNER` prompts form another family of grouped experiences

This is not the same as nested per-boundary branching (`Kc/Kl/Ks`) in the veRL path.
If you need explicit nested branch evaluation and canonical commit logic, use `training/verl/`.

## Boundary Reward Schema

Use the shared boundary event + scalarization spec:

- `training/BOUNDARY_REWARD_SCHEMA.md`
- `training/boundary_event.schema.json`

In this OpenRLHF path:

- `DECIDER` boundaries map to `module=curiosity` in learn modes and `module=solver` in solve mode.
- `LEARNER` boundaries map to `module=learner`.
- Grouped normalization is still handled by OpenRLHF (`group_norm`), while this schema
  standardizes what gets logged/scored per boundary.

## Memory Curriculum in OpenRLHF Path

`agent_func_loopagent_ls20.py` supports per-episode memory initialization modes:

- `carry`: start from previous episode memory snapshot
- `noisy`: start from snapshot with random perturbation
- `blank`: start with empty memory

The mode is sampled with:

- `MEMORY_INIT_CARRY_P` (default `0.60`)
- `MEMORY_INIT_NOISY_P` (default `0.25`)
- `MEMORY_INIT_BLANK_P` (default `0.15`)

Noisy perturbation uses:

- `NOISY_DELETE_FRACTION` (default `0.2`)
- `NOISY_CONF_JITTER` (default `0.1`)

## Common Knobs

Environment variables for `run_grpo_ls20_1gpu.sh`:

- `MODEL`
- `AGENT_FUNC_PATH`
- `N_SAMPLES_PER_PROMPT`
- `MAX_SAMPLES`
- `MAX_STEPS_PER_EPISODE`
- `ROLLOUT_BATCH_SIZE`, `TRAIN_BATCH_SIZE`
- `VLLM_GPU_UTIL`
- `MEMORY_INIT_CARRY_P`, `MEMORY_INIT_NOISY_P`, `MEMORY_INIT_BLANK_P`
- `NOISY_DELETE_FRACTION`, `NOISY_CONF_JITTER`

Additional environment variables for `run_online_grpo_ls20_1gpu.sh`:

- `ONLINE_ITERS`
- `MAX_SAMPLES_PER_ITER`
- `OUTPUT_DIR`
- `DATASET_PATH`
- `MEMORY_INIT_CARRY_P`, `MEMORY_INIT_NOISY_P`, `MEMORY_INIT_BLANK_P`
- `NOISY_DELETE_FRACTION`, `NOISY_CONF_JITTER`
