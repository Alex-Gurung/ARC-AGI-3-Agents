# Loop Agent Status

## Current State

The `LoopAgent` is now an explore-learn-exploit harness with:

- Dynamic action space from `frame.available_actions`
- Curiosity / Learner / Solver split
- Phase control: `EXPLORE` <-> `EXPLOIT`
- Level control: `action` -> `subgoal` -> `plan` via `LevelController`
- Surprise strategies:
  - `heuristic` (default)
  - `logprob` (VLLM completions scoring pass)
- Keyframes:
  - periodic via `STATE_KEYFRAME_INTERVAL`
  - event-driven (`subgoal_attempt`, `surprise_spike`, reset/level transition)
- Subgoal burst execution:
  - solver can emit short action sequences to reduce per-step LLM calls
  - queued burst actions execute before new model queries
- Configurable memory persistence mode on full reset:
  - `strict`: clear memory
  - `carry`: keep memory
  - `noisy`: perturb memory (delete/jitter/shuffle)

## Memory Model

Memory remains list-based, but each entry now has a stable ID:

- Display format: `[idx=<n>][id=M####][TYPE] ...`
- Operations support index or ID refs:
  - `MODIFY 3 ...` or `MODIFY M0003 ...`
  - `REMOVE 3 ...` or `REMOVE M0003 ...`
- Internal references should prefer IDs for stability.

## Key Runtime Env Vars

- `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`
- `SURPRISE_STRATEGY` = `heuristic` | `logprob`
- `SURPRISE_THRESHOLD`
- `EXPLORE_BUDGET_RATIO`, `RE_EXPLORE_BUDGET`
- `MEMORY_PERSISTENCE_MODE` = `strict` | `carry` | `noisy`
- `NOISY_DELETE_FRACTION`, `NOISY_CONF_JITTER`
- `STATE_KEYFRAME_INTERVAL`
- `USE_SUBGOAL_BURSTS`, `BURST_MAX_STEPS`
- `LEARNER_UPDATE_INTERVAL`

## Known Gaps / TODOs

1. Add a strict evaluation preset script (forces blank memory at new attempts and fixed deterministic settings).
2. Add stronger burst stop conditions in-loop (e.g. `NO_CHANGE_2`, explicit `subgoal_done`, per-burst surprise guard).
3. Add stable-ID-aware learner prompt examples (`REMOVE M####`, `MODIFY M####`) so model natively uses IDs.
4. Add calibration tooling for surprise thresholds per game family and per memory mode.
5. Add trajectory logger schema for RL (group candidates + rewards + advantages + chosen sample).
6. Implement warm/cold/noisy rollout mix for training data generation:
   - suggested starting ratio: 60/30/10.
7. Add held-out evaluation suite for robustness:
   - blank-memory strict eval
   - carry-memory eval
   - noisy-memory stress eval
8. Remove accidental build artifact from tracking: `arc_agi_3_agents.egg-info/`.

## RL Direction (not implemented yet)

Recommended next training approach:

- Grouped on-policy sampling per active module (curiosity OR learner OR solver, not all branched at once)
- Within-group normalized advantages
- Softmax selection over candidate rewards (not uniform random)
- Learner dense reward from predictive improvement (`delta log p`) + terminal bonus
- Framework decision (`OpenRLHF` vs custom) can be deferred until rollout schema is frozen.
