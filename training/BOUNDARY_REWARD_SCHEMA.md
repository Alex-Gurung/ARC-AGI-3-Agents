# Boundary Reward Schema and Scalarization (PipelineRL + PRIME-RL)

This document defines a shared format for turning one game rollout into many
boundary-level reward events (multi-sub-reward), aligned with the current
LoopAgent behavior:

- modes: `LEARN_ACTION | LEARN_SUBGOAL | LEARN_PLAN | SOLVE`
- modules: `curiosity | learner | solver`
- boundaries: `action | subgoal | plan`
- subgoal sequences and soft resets
- memory/rulebook updates and diagnosis routing
- state visit/repeat counters and action-coverage signals

## Why

One episode should emit many supervised RL items:

- one item per decision boundary
- each item stores a reward vector (`reward_components`)
- training uses a role-gated scalarization (`reward_scalar`)

This preserves dense signal while keeping framework optimizers simple.

## Boundary Event (JSON)

Each emitted row should follow this shape:

```json
{
  "episode_id": "ls20_ep_000123",
  "attempt_id": "attempt_000123",
  "boundary_index": 17,
  "step": 64,
  "policy_version": "ckpt_0017",
  "framework": "openrlhf",
  "group_id": "ls20_ep_000123::17::curiosity::LEARN_SUBGOAL::subgoal",

  "role": "DECIDER",
  "module": "curiosity",
  "mode": "LEARN_SUBGOAL",
  "phase": "LEARN",
  "boundary_type": "subgoal",
  "boundary_reason": "exhausted",
  "subgoal_index": 1,

  "active_plan": "1) touch blue switch; 2) pass right gate; 3) reach border",
  "active_subgoal": "touch blue switch",
  "action_trace": ["ACTION4", "ACTION4", "ACTION1"],
  "action_trace_len": 3,
  "expected": "player reaches blue switch and gate opens",

  "state_before_hash": "a1b2c3d4e5f60708",
  "state_after_hash": "1f2e3d4c5b6a7980",
  "state_visits_before": 2,
  "state_visits_after": 1,
  "repeat_state_streak_before": 1,
  "repeat_state_streak_after": 0,
  "changed_cells": 11,
  "level_delta": 0,

  "terminal_state": "NOT_FINISHED",
  "is_win": false,
  "is_game_over": false,
  "is_soft_reset": false,
  "parse_warning": "none",

  "memory_before_count": 7,
  "memory_after_count": 8,
  "missing_action_lessons_before": ["ACTION3", "ACTION6"],
  "missing_action_lessons_after": ["ACTION6"],

  "semantic_report": "The player moved right and contacted the blue switch...",
  "surprise": {
    "source": "debiased_nll",
    "debiased_nll": 0.42,
    "self_rated_x10": 6.0,
    "reward_value": 0.42,
    "mean_nll_full": 2.13,
    "mean_nll_stripped": 1.71
  },

  "reward_components": {
    "surprise": 0.42,
    "novelty": 1.0,
    "transition_magnitude": 11.0,
    "boundary_progress": 0.0,
    "unknown_action_bonus": 0.0,
    "loop_break_bonus": 1.0,
    "repeat_penalty": 0.0,

    "mismatch_reduction": 0.0,
    "parse_bonus": 0.0,
    "dedup_bonus": 0.0,
    "contradiction_cleanup_bonus": 0.0,
    "missing_action_reduction_bonus": 0.0,
    "parse_warning_penalty": 0.0,

    "level_delta_reward": 0.0,
    "win_bonus": 0.0,
    "game_over_penalty": 0.0,
    "step_penalty": 0.0,
    "solver_surprise_bonus": 0.0
  },

  "reward_scalar": 0.91,
  "reward_channel_version": "v2",
  "diagnostics": {
    "last_diagnosis_level": "none",
    "mode_switch_cooldown_remaining": 1,
    "rulebook_status": "ACTION lessons covered=5/7"
  }
}
```

## Scalarization (Role-Gated)

For all formulas below, reward vector is logged in full, but only
`reward_scalar` is used for policy update.

### 1) Curiosity / DECIDER in learn modes

Use when `module == "curiosity"`:

`reward_scalar =`

- `+ 1.0 * surprise`
- `+ 0.2 * novelty`
- `+ 0.01 * transition_magnitude`
- `+ 1.0 * boundary_progress` if `boundary_type=subgoal`
- `+ 2.0 * boundary_progress` if `boundary_type=plan`
- `+ 1.0 * unknown_action_bonus`
- `+ 0.5 * loop_break_bonus`
- `- 0.5 * repeat_penalty`
- `- 0.25 * parse_warning_penalty`

Notes:

- `surprise` is from configured source (`debiased_nll` / `self_rated` / `hybrid`).
- `novelty` should be `1.0` when post-transition state is first visit in attempt.
- `unknown_action_bonus` should be `1.0` when candidate tests an action missing from rulebook.

### 2) Learner

Use when `module == "learner"`:

`reward_scalar =`

- `+ 1.0 * mismatch_reduction`
- `+ 0.1 * parse_bonus`
- `+ 0.1 * dedup_bonus`
- `+ 0.1 * contradiction_cleanup_bonus`
- `+ 0.25 * missing_action_reduction_bonus`
- `- 0.1 * parse_warning_penalty`

Recommended definitions:

- `mismatch_reduction = mismatch_before - mismatch_after`
- `parse_bonus = 1` when output parseable into valid ops, else `0`
- `dedup_bonus = 1` when memory growth is non-duplicate / bounded, else `0`
- `missing_action_reduction_bonus = (#missing_before - #missing_after)`

### 3) Solver / DECIDER in SOLVE mode

Use when `module == "solver"`:

`reward_scalar =`

- `+ 10.0 * level_delta_reward`
- `+ 50.0 * win_bonus`
- `- 5.0 * game_over_penalty`
- `- 0.05 * step_penalty`
- `+ 0.5 * solver_surprise_bonus`

Recommended definitions:

- `level_delta_reward = level_delta`
- `win_bonus = 1` iff win reached at this boundary
- `game_over_penalty = 1` iff game-over at this boundary
- `step_penalty = action_trace_len`
- `solver_surprise_bonus = surprise`

## Grouping for Advantage Normalization

Do not normalize across mixed roles/modules.

Use:

- `group_id = episode_id + boundary_index + module + mode + boundary_type`

Within each `group_id`:

1. collect `K` candidate scalar rewards
2. z-normalize (`(r-mean)/(std+eps)`)
3. compute sampling probs via softmax
4. sample selected branch

Commit only selected branch to canonical state/memory.

## PipelineRL Mapping

In `generate_rollout`:

1. run boundary candidate replicas
2. emit one event per candidate with full `reward_components`
3. include `group_id` and `candidate_index`
4. store `reward_scalar` for optimizer

In preprocessor/verifier:

1. validate schema
2. compute/verify scalar from channels
3. attach within-group normalized advantage

## PRIME-RL Mapping

In verifier EnvServer:

1. return boundary `reward_components` in `info`
2. return `reward_scalar` as immediate reward
3. include `group_id`, `module`, `mode`, `boundary_type`, `policy_version`

In trainer/orchestrator:

1. group by `group_id`
2. run group-normalized advantages
3. keep async lag metadata (`policy_version`, staleness diagnostics)

## Current LoopAgent Alignment Checklist

When generating events, include these current signals from LoopAgent:

- `state_visits_*` and `repeat_state_streak_*`
- `missing_action_lessons_*`
- `rulebook_status`
- `subgoal_index`, `active_plan`, `active_subgoal`
- `boundary_reason` (`terminal|no_change|exhausted` for sequence boundaries)
- `parse_warning`
- `last_diagnosis_level` and cooldown metadata
- `soft_reset` marker (control-flow reset, not world evidence)

## Implementation Notes

1. Keep `reward_components` vector immutable in logs.
2. Version the schema (`reward_channel_version`) whenever formulas change.
3. Keep scalarization deterministic and side-effect free.
4. Never mix curiosity/learner/solver rewards into one untagged scalar stream.
