# Loop Agent Status

## Architecture

### Main Loop Cycle

The agent runs a mode-routed loop, up to 200 steps per game by default.
Execution is hierarchical: maintain an active plan and subgoal, execute
subgoal action sequences (up to 20 actions), then learn/update mode at
sequence boundary.

```
┌──────────────────────────────────────────────────────────────────────┐
│ GAME ENV → STATE ENCODER → state_text                               │
│          (grid + actions)                                           │
│                                                                      │
│ LEARN modes: Curiosity chooses action/subgoal/plan                   │
│ SOLVE mode: Solver uses active plan + active subgoal                 │
│                                                                      │
│ If in subgoal sequence mode:                                         │
│   propose_subgoal_actions() → execute queued actions open-loop       │
│   (no per-step memory writes; boundary update is authoritative)       │
│                                                                      │
│ Learn-subgoal / learn-plan attempts use soft RESET after boundary     │
│   (RESET is control-flow only, not learning evidence)                │
│                                                                      │
│ Sequence boundary (or normal single-step path):                      │
│   Learner update (ADD/MODIFY/REMOVE/NONE)                            │
│   Surprise score (for Level Controller + metrics)                    │
│   Mode routing (diagnosis-directed or mode-router LLM)               │
└──────────────────────────────────────────────────────────────────────┘
```

### Mode Transitions

```
       ┌─────────────────────────────────────────────────────────┐
       │ CURRENT_MODE ∈ {LEARN_ACTION, LEARN_SUBGOAL,          │
       │                  LEARN_PLAN, SOLVE}                   │
       └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    boundary diagnosis unexpected (high confidence)?
      yes -> route directly to diagnosed learn level:
             action  -> LEARN_ACTION
             subgoal -> LEARN_SUBGOAL
             plan    -> LEARN_PLAN
      no  -> Curiosity mode-router chooses next mode

    GAME_OVER always overrides:
      learner.diagnose(...) -> diagnosed level -> corresponding LEARN_* mode
```

Note: mode routing happens at boundaries. During active subgoal sequences,
learner writes are boundary-level at `finalize_subgoal_sequence()`.

### Abstraction Levels

Abstraction levels are still represented (`action/subgoal/plan`) but
selection is controlled by `DecisionMode`:

- `LEARN_ACTION` -> action-level exploration
- `LEARN_SUBGOAL` -> subgoal-level exploration with action sequences
- `LEARN_PLAN` -> plan-level exploration with subgoal attempts
- `SOLVE` -> exploit current understanding to complete level

```
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  ┌──────────┐   converge   ┌──────────┐  converge  ┌─────┐│
    │  │  ACTION  │─────────────▶│ SUBGOAL  │───────────▶│PLAN ││
    │  │          │              │          │            │     ││
    │  │ single   │              │ sequence │          │multi││
    │  │ move     │              │ of 1-20  │          │step ││
    │  │ per step │              │ actions  │           │strat││
    │  └──────────┘              └──────────┘            └─────┘│
    │                                                             │
    │  Curiosity:  Curiosity:          Curiosity:                 │
    │  "try        "investigate        "devise overall            │
    │   ACTION2"    the top row"        strategy as               │
    │                                   ordered subgoals"         │
    │                                                             │
    │  Solver:     Solver:             Solver:                    │
    │  picks       proposes 1-4        follows plan,              │
    │  one         actions for         picks subgoal,             │
    │  action      current subgoal     then action sequence       │
    └─────────────────────────────────────────────────────────────┘
```

### Component Data Flow

```
    reads ──▶     writes ══▶

    ┌───────────────────────────────────────────────────────────────┐
    │                                                               │
    │  STATE ENCODER ──────▶ state_text ──▶ Curiosity             │
    │       │                    │     ──▶ Solver                  │
    │       │                    └───────▶ Learner                 │
    │       │                                                       │
    │       └──────────────▶ diff_text  ──▶ Learner               │
    │                                                               │
    │  MEMORY ─────────────▶ memory_text ──▶ Curiosity            │
    │       ▲                          ──▶ Solver                  │
    │       │                          ──▶ Learner                 │
    │       │                                                       │
    │       ╚══════════════════════════════ Learner                 │
    │                                      (ADD/MODIFY/REMOVE)      │
    │                                                               │
    │  SURPRISE ───────────▶ score ────────▶ Level Controller     │
    │                                                               │
    │  Learner changed? ─────▶ phase updater (EXPLORE/EXPLOIT)     │
    │  GAME_OVER ────────────▶ phase updater + diagnosis           │
    │                                                               │
    │  LEVEL CONTROLLER ───▶ current_level ▶ Curiosity            │
    │                                  ──▶ Solver                  │
    │                                                               │
    │  Curiosity/Solver ───▶ prediction ───▶ Learner              │
    │                                                               │
    └───────────────────────────────────────────────────────────────┘
```

### Subgoal Sequence Execution

A subgoal is defined in prompts as:
"a short plan (<=20 actions) expected to produce a significant,
testable state change that helps level completion."

During LEARN_SUBGOAL / LEARN_PLAN, Curiosity proposes:
- `SUBGOAL`
- `SUCCESS_TEST`
- `ACTION_SEQUENCE`
- `EXPECTED`

During SOLVE, Solver proposes action sequences for active subgoals.
Sequences execute open-loop, then evaluate at boundary.

```
    Curiosity/Solver.propose_subgoal_actions()
         │
         ▼
    ┌─────────────────────────────────────────────────┐
    │  action₁ ──▶ ... ──▶ action₂₀               │
    │                                                 │
    │  executed sequentially, no learner between steps│
    │                                                 │
    │  interrupted if:                                │
    │    • terminal state (WIN / GAME_OVER)           │
    │    • 4+ steps with no grid changes              │
    └─────────────────────┬───────────────────────────┘
                          │
                          ▼
                 finalize_subgoal_sequence()
                    • learner sees full before→after
                    • surprise scored once for sequence
                    • may advance to next subgoal
```

### Memory Lifecycle

```
    GAME START
         │
         ▼
    ┌─────────────┐
    │ Empty memory│ (0/50)
    │             │
    │ EXPLORE     │  Learner fills entries:
    │ phase       │  observations, rules, actions
    │             │
    └──────┬──────┘
           │ 5+ consecutive NONEs (stable)
           ▼
    ┌─────────────┐
    │ Stable      │  Solver reads memory to act
    │ memory      │  Learner still watches
    │             │  If prediction wrong → back to explore
    │ EXPLOIT     │
    │ phase       │
    └──────┬──────┘
           │ level complete
           ▼
    ┌─────────────┐
    │ New level   │  Memory persists (mode-dependent)
    │             │  State encoder resets
    │ brief       │  Surprise history resets
    │ re-explore  │  15% of remaining budget
    └──────┬──────┘
           │ full reset (RESET action)
           ▼
    ┌─────────────────────────────────┐
    │ Memory persistence mode:        │
    │   strict → clear all            │
    │   carry  → keep all             │
    │   noisy  → delete 20%, jitter   │
    │            confidence, shuffle  │
    └─────────────────────────────────┘
```

---

## Current State

The `LoopAgent` is now an explore-learn-exploit harness with:

- Dynamic action space from `frame.available_actions`
- Curiosity / Learner / Solver split
- Mode control: `LEARN_ACTION` | `LEARN_SUBGOAL` | `LEARN_PLAN` | `SOLVE`
- Diagnosis-directed routing on mismatch (no strike-based hardcoding)
- Surprise strategies:
  - `heuristic` (default)
  - `logprob` (VLLM completions scoring pass)
- Keyframes:
  - periodic via `STATE_KEYFRAME_INTERVAL`
  - event-driven (`subgoal_attempt`, `subgoal_advance`, reset/level transition)
- Active hierarchical execution:
  - plan is parsed into ordered subgoals
  - each active subgoal proposes an action sequence from current state
  - sequence is executed open-loop, then evaluated at subgoal boundary
  - in `SOLVE`, solver routes through plan/subgoal sequence execution
  - in `LEARN_SUBGOAL` and `LEARN_PLAN`, curiosity uses subgoal sequences
- Soft reset control-flow:
  - explore subgoal attempts queue a soft `RESET`
  - soft reset frames are metadata-only (no learner update, no surprise scoring)
  - memory/phase/level are preserved across soft reset
- Subgoal-boundary updates:
  - learner update, phase transition, and level-controller update happen at sequence boundaries
  - hard interrupts: terminal state or repeated no-change steps
- Diagnosis-directed mode routing:
  - learner expectation assessment returns `expected|unexpected` with confidence
  - high-confidence `unexpected` routes directly to diagnosed learn level
  - `GAME_OVER` always triggers direct diagnosis-based route to a learn level
- Learned mode router:
  - at boundaries without mismatch override, curiosity picks next mode:
    `LEARN_ACTION|LEARN_SUBGOAL|LEARN_PLAN|SOLVE`
- Structured response contract:
  - prompts allow short reasoning but require a final `ANSWER: ...` line
  - parser extracts only the final `ANSWER:` payload for execution
- Explicit stage context in prompts:
  - Curiosity/Solver/Learner all receive `PHASE`, `LEVEL`, and `SUBGOAL_INDEX`
  - Curiosity also receives `ACTIVE_PLAN` and `ACTIVE_SUBGOAL`
- Configurable memory persistence mode on full reset:
  - `strict`: clear memory
  - `carry`: keep memory
  - `noisy`: perturb memory (delete/jitter/shuffle)
- Training-only grouped sampling package:
  - `training/group_sampler.py`
  - `training/rewards.py`
  - `training/rollout_runner.py`
  - `training/trajectory_schema.py`
  - includes replica-prefix replay hooks for counterfactual candidate scoring
  - includes explicit `commit_selected_candidate(...)` helper so only the selected
    branch mutates canonical trajectory state; non-selected candidates are log-only

## Memory Model

Memory remains list-based, but each entry now has a stable ID:

- Display format: `[idx=<n>][id=M####][TYPE] ...`
- Operations support index or ID refs:
  - `MODIFY 3 ...` or `MODIFY M0003 ...`
  - `REMOVE 3 ...` or `REMOVE M0003 ...`
- Parser accepts confidence formats `(0.8)` and `(conf: 0.8)`.
- Internal references should prefer IDs for stability.

## Key Runtime Env Vars

- `MAX_ACTIONS` (default `200`)
- `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`
- `SURPRISE_STRATEGY` = `heuristic` | `logprob`
- `MEMORY_PERSISTENCE_MODE` = `strict` | `carry` | `noisy`
- `NOISY_DELETE_FRACTION`, `NOISY_CONF_JITTER`
- `STATE_KEYFRAME_INTERVAL`
- `USE_SUBGOAL_SEQUENCES` (legacy fallback: `USE_SUBGOAL_BURSTS`)
- `SUBGOAL_MAX_ACTIONS` (legacy fallback: `BURST_MAX_STEPS`)
- `SUBGOAL_NO_CHANGE_LIMIT`
- `LEARNER_UPDATE_INTERVAL`
- `MISMATCH_CONF_THRESHOLD`

## Known Gaps / TODOs

1. Add a strict evaluation preset script (forces blank memory at new attempts and fixed deterministic settings).
2. Add stronger subgoal sequence stop conditions (e.g. explicit `subgoal_done` classifier, surprise guard).
3. Add stable-ID-aware learner prompt examples (`REMOVE M####`, `MODIFY M####`) so model natively uses IDs.
4. Add calibration tooling for surprise scaling (for metrics/RL rewards) per game family and memory mode.
5. Integrate `training/` grouped-sampling utilities with full env-replica prefix replay.
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

## Validation Snapshot (2026-02-27)

Passing checks:

- `uv run python scripts/test_loop_v2_units.py`
- `uv run python scripts/test_memory.py`
- `uv run python scripts/test_state_encoder.py`
- `uv run python scripts/test_surprise.py --strategy heuristic`
- `uv run python scripts/test_full_loop.py`
- `uv run ruff check agents/templates/loop_agent training`

Known repo-wide test blocker:

- `uv run pytest -q` currently fails before execution due to missing module
  `agents.structs` imported by `tests/conftest.py`.
