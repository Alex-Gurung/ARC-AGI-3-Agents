# Loop Agent Status

## Architecture

### Main Loop Cycle

The agent runs an explore-learn-exploit loop, up to 80 steps per game.
Execution is hierarchical: maintain an active plan and subgoal, execute
short subgoal action sequences, then learn/update phase at sequence boundary.

```
┌──────────────────────────────────────────────────────────────────────┐
│ GAME ENV → STATE ENCODER → state_text                               │
│          (grid + actions)                                           │
│                                                                      │
│ EXPLORE path: Curiosity chooses action/subgoal/plan                  │
│ EXPLOIT path: Solver uses active plan + active subgoal               │
│                                                                      │
│ If in subgoal sequence mode:                                         │
│   propose_subgoal_actions() → execute queued actions open-loop       │
│   (no learner/phase update until sequence boundary)                  │
│                                                                      │
│ Sequence boundary (or normal single-step path):                      │
│   Learner update (ADD/MODIFY/REMOVE/NONE)                            │
│   Surprise score (for Level Controller + metrics)                    │
│   Phase update (driven by learner_changed/GAME_OVER)                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase Transitions

```
                ┌──────────────────────────────────┐
                │                                  │
                │   memory stable (5+ NONEs)       │
                │   OR explore budget exhausted    │
                │                                  │
    ╔═══════════▼══╗                  ╔════════════╧═══╗
    ║              ║   learner        ║                ║
    ║   EXPLORE    ║   changed     ◀──║    EXPLOIT     ║
    ║              ║   memory         ║                ║
    ║  Curiosity   ║◀─────────────────║  Solver picks  ║
    ║  proposes    ║                  ║  best action   ║
    ║  info-seeking║   GAME_OVER      ║  from memory   ║
    ║  actions     ║◀──diagnose───────║                ║
    ║              ║   + regress      ║                ║
    ╚══════════════╝   level          ╚════════════════╝
```

Note: during active subgoal sequences, `learner changed memory` is evaluated
at `finalize_subgoal_sequence()` rather than every primitive action.

### Abstraction Levels

The Level Controller advances when surprise converges (<5% change over
a window of 5 steps). On GAME_OVER, the Learner diagnoses which level
broke and the controller regresses to that level.

```
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  ┌──────────┐   converge   ┌──────────┐  converge  ┌─────┐│
    │  │  ACTION  │─────────────▶│ SUBGOAL  │───────────▶│PLAN ││
    │  │          │              │          │            │     ││
    │  │ single   │◀─ diagnose ──│ sequence │◀─diagnose─│multi││
    │  │ move     │              │ of 1-4   │           │step ││
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
    │  STATE ENCODER ──────▶ state_text ──▶ Curiosity               │
    │       │                    │     ──▶ Solver                   │
    │       │                    └───────▶ Learner                  │
    │       │                                                       │
    │       └──────────────▶ diff_text  ──▶ Learner                 │
    │                                                               │
    │  MEMORY ─────────────▶ memory_text ──▶ Curiosity              │
    │       ▲                          ──▶ Solver                  │
    │       │                          ──▶ Learner                 │
    │       │                                                       │
    │       ╚══════════════════════════════ Learner                 │
    │                                      (ADD/MODIFY/REMOVE)      │
    │                                                               │
    │  SURPRISE ───────────▶ score ────────▶ Level Controller       │
    │                                                               │
    │  Learner changed? ─────▶ phase updater (EXPLORE/EXPLOIT)      │
    │  GAME_OVER ────────────▶ phase updater + diagnosis            │
    │                                                               │
    │  LEVEL CONTROLLER ───▶ current_level ▶ Curiosity              │
    │                                  ──▶ Solver                  │
    │                                                               │
    │  Curiosity/Solver ───▶ prediction ───▶ Learner                │
    │                                                               │
    └───────────────────────────────────────────────────────────────┘
```

### Subgoal Sequence Execution

When at subgoal or plan level during EXPLOIT, the Solver proposes
a short action sequence (1-4 actions) for the current subgoal.
These are executed open-loop, then evaluated as a batch.

```
    Solver.propose_subgoal_actions()
         │
         ▼
    ┌─────────────────────────────────────────────────┐
    │  action₁ ──▶ action₂ ──▶ action₃ ──▶ action₄   │
    │                                                  │
    │  executed sequentially, no learner between steps │
    │                                                  │
    │  interrupted if:                                 │
    │    • terminal state (WIN / GAME_OVER)            │
    │    • 2+ steps with no grid changes               │
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
    │ Empty memory │ (0/50)
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
    │            confidence, shuffle   │
    └─────────────────────────────────┘
```

---

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
  - event-driven (`subgoal_attempt`, `subgoal_advance`, reset/level transition)
- Active hierarchical execution:
  - plan is parsed into ordered subgoals
  - each active subgoal proposes a short action sequence from current state
  - sequence is executed open-loop, then evaluated at subgoal boundary
- Subgoal-boundary updates:
  - learner update, phase transition, and level-controller update happen at sequence boundaries
  - hard interrupts: terminal state or repeated no-change steps
- Structured response contract:
  - prompts allow short reasoning but require a final `ANSWER: ...` line
  - parser extracts only the final `ANSWER:` payload for execution
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
- Parser accepts confidence formats `(0.8)` and `(conf: 0.8)`.
- Internal references should prefer IDs for stability.

## Key Runtime Env Vars

- `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`
- `SURPRISE_STRATEGY` = `heuristic` | `logprob`
- `EXPLORE_BUDGET_RATIO`, `RE_EXPLORE_BUDGET`
- `MEMORY_PERSISTENCE_MODE` = `strict` | `carry` | `noisy`
- `NOISY_DELETE_FRACTION`, `NOISY_CONF_JITTER`
- `STATE_KEYFRAME_INTERVAL`
- `USE_SUBGOAL_SEQUENCES` (legacy fallback: `USE_SUBGOAL_BURSTS`)
- `SUBGOAL_MAX_ACTIONS` (legacy fallback: `BURST_MAX_STEPS`)
- `SUBGOAL_NO_CHANGE_LIMIT`
- `LEARNER_UPDATE_INTERVAL`

## Known Gaps / TODOs

1. Add a strict evaluation preset script (forces blank memory at new attempts and fixed deterministic settings).
2. Add stronger subgoal sequence stop conditions (e.g. explicit `subgoal_done` classifier, surprise guard).
3. Add stable-ID-aware learner prompt examples (`REMOVE M####`, `MODIFY M####`) so model natively uses IDs.
4. Add calibration tooling for surprise scaling (for metrics/RL rewards) per game family and memory mode.
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

## Validation Snapshot (2026-02-27)

Passing checks:

- `uv run python scripts/test_memory.py`
- `uv run python scripts/test_state_encoder.py`
- `uv run python scripts/test_surprise.py --strategy heuristic`
- `uv run python scripts/test_full_loop.py`
- `uv run ruff check agents/templates/loop_agent scripts/test_memory.py`

Known repo-wide test blocker:

- `uv run pytest -q` currently fails before execution due to missing module
  `agents.structs` imported by `tests/conftest.py`.
