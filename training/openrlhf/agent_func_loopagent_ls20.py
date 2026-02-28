"""OpenRLHF multi-turn agent function with LoopAgent-style rollout semantics.

This path trains one shared policy to handle:
- DECIDER role: curiosity/solver mode-conditioned boundary decisions
- LEARNER role: memory/rulebook updates after each boundary transition

Each RL step alternates roles:
1) DECIDER proposes mode/plan/subgoal/actions/expected
2) Environment executes to next boundary (action or subgoal/plan boundary)
3) LEARNER updates memory from BEFORE/ACTION/AFTER/DIFF evidence

This preserves module-specific grouped training signals under OpenRLHF's
group_norm estimator while staying within a single-policy interface.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
from dataclasses import dataclass, replace
from typing import Any, Optional

from arc_agi import Arcade
from arcengine import FrameData, FrameDataRaw, GameAction, GameState

from agents.templates.loop_agent.memory import (
    Memory,
    apply_memory_operation,
    parse_memory_operation,
)
from agents.templates.loop_agent.state_encoder import StateEncoder
from agents.templates.loop_agent.surprise import HeuristicSurprise
from training.verl.rewards import curiosity_reward, learner_reward, solver_reward

OPENRLHF_IMPORT_ERROR: ImportError | None = None
try:
    from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
except ImportError as import_err:  # pragma: no cover - local dev without openrlhf
    OPENRLHF_IMPORT_ERROR = import_err
    AgentInstanceBase = object  # type: ignore[assignment]
    MultiTurnAgentExecutor = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MODES = ("LEARN_ACTION", "LEARN_SUBGOAL", "LEARN_PLAN", "SOLVE")
ROLES = ("DECIDER", "LEARNER")

DECIDER_OBS_TEMPLATE = """\
ROLE: DECIDER
You are controlling a hierarchical game agent.
Choose what to test/execute next given CURRENT_MODE and memory rulebook.

CURRENT_MODE: {mode}
ACTIVE_PLAN: {active_plan}
ACTIVE_SUBGOAL: {active_subgoal}
SUBGOAL_INDEX: {subgoal_index}
STATE_VISITS: {state_visits}
MISSING_ACTION_LESSONS: {missing_action_lessons}
ACTION_BUDGET: {step_used}/{step_budget}

AVAILABLE_ACTIONS: {available_actions}

STATE:
{state_text}

MEMORY RULEBOOK:
{memory_text}

Rules:
- LEARN_ACTION: prioritize testing action mechanics and unknown actions.
- LEARN_SUBGOAL: propose short sequences that test object interactions.
- LEARN_PLAN: propose/update high-level strategy and test a subgoal.
- SOLVE: exploit current rulebook/plan to make progress.
- Use GRID/DIFF as ground truth evidence.
- OBJECTS/RELATIONS are heuristic and may be noisy.

Output format (exactly these labels; MODE/PLAN/SUBGOAL optional):
ANSWER:
MODE: <LEARN_ACTION|LEARN_SUBGOAL|LEARN_PLAN|SOLVE>
PLAN: <optional semicolon-separated steps>
SUBGOAL: <optional short phrase>
ACTION_SEQUENCE: <comma-separated actions, may include ACTION6 x y, max {max_subgoal_actions}>
EXPECTED: <short expected transition>
""".strip()


LEARNER_OBS_TEMPLATE = """\
ROLE: LEARNER
Update memory lessons from this observed boundary transition.
Only add claims supported by direct evidence in BEFORE/AFTER/DIFF.
Use low confidence for hypotheses. Remove stale/wrong entries when contradicted.

CURRENT_MODE: {mode}
BOUNDARY_TYPE: {boundary_type}
SUBGOAL_INDEX: {subgoal_index}
ACTION_TRACE: {action_trace}
EXPECTED: {expected}

BEFORE:
{state_before}

AFTER:
{state_after}

DIFF:
{diff_text}

MEMORY RULEBOOK:
{memory_text}

Output format:
ANSWER:
<one or more ops, one per line>
ADD [TYPE] lesson | evidence (0.xx)
MODIFY [n] lesson | evidence (0.xx)
REMOVE [n] | reason
NONE

Optional:
NEXT_MODE: <LEARN_ACTION|LEARN_SUBGOAL|LEARN_PLAN|SOLVE>
""".strip()


@dataclass
class PendingBoundary:
    mode: str
    boundary_type: str
    state_before: str
    state_after: str
    diff_text: str
    expected: str
    action_trace: list[str]
    changed_cells: int
    level_delta: int
    terminal: bool
    parse_warning: str


def _convert_raw_frame_data(raw: FrameDataRaw | None) -> FrameData:
    if raw is None:
        raise ValueError("Received None frame data from environment")
    return FrameData(
        game_id=raw.game_id,
        frame=[arr.tolist() for arr in raw.frame],
        state=raw.state,
        levels_completed=raw.levels_completed,
        win_levels=raw.win_levels,
        guid=raw.guid,
        full_reset=raw.full_reset,
        available_actions=raw.available_actions,
    )


def _available_action_names(frame: FrameData) -> list[str]:
    return [f"ACTION{a}" if a > 0 else "RESET" for a in frame.available_actions]


def _grid_signature(grid: list[list[int]]) -> str:
    if not grid or not grid[0]:
        return "empty"
    height = len(grid)
    width = len(grid[0])
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(f"{width}x{height}|".encode("ascii"))
    for row in grid:
        hasher.update(bytes(row))
    return hasher.hexdigest()


class LS20LoopAgentInstance(AgentInstanceBase):
    """OpenRLHF environment with LoopAgent-like staged rollouts."""

    def __init__(self) -> None:
        super().__init__()
        self._arcade: Arcade | None = None
        self._env: Any = None
        self._card_id: str | None = None
        self._frame: FrameData | None = None

        self._state_encoder = StateEncoder(
            keyframe_interval=int(os.environ.get("STATE_KEYFRAME_INTERVAL", "10"))
        )
        self._surprise = HeuristicSurprise()
        self._memory = Memory(max_entries=int(os.environ.get("MAX_MEMORY_ENTRIES", "50")))

        self._role: str = "DECIDER"
        self._mode: str = "LEARN_ACTION"
        self._active_plan_text: str = ""
        self._active_plan_steps: list[str] = []
        self._active_subgoal: str = ""
        self._active_subgoal_index: int = 0

        self._subgoal_max_actions: int = int(os.environ.get("SUBGOAL_MAX_ACTIONS", "20"))
        self._subgoal_no_change_limit: int = int(
            os.environ.get("SUBGOAL_NO_CHANGE_LIMIT", "4")
        )

        self._pending_boundary: PendingBoundary | None = None
        self._terminal_after_learner: bool = False

        self._episode_actions_used: int = 0
        self._max_actions: int = 200
        self._running_return: float = 0.0
        self._turn_index: int = 0

        self._seen_state_signatures: set[str] = set()
        self._state_visit_counts: dict[str, int] = {}
        self._last_parse_warning: str = "none"
        self._carry_memory_snapshot: list[Any] = []
        self._memory_init_carry_p: float = float(os.environ.get("MEMORY_INIT_CARRY_P", "0.60"))
        self._memory_init_noisy_p: float = float(os.environ.get("MEMORY_INIT_NOISY_P", "0.25"))
        self._memory_init_blank_p: float = float(os.environ.get("MEMORY_INIT_BLANK_P", "0.15"))
        self._noisy_delete_fraction: float = float(
            os.environ.get("NOISY_DELETE_FRACTION", "0.2")
        )
        self._noisy_conf_jitter: float = float(os.environ.get("NOISY_CONF_JITTER", "0.1"))

    async def reset(self, states: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self._close_scorecard()

        game_id = str(states.get("game_id", "ls20"))
        self._max_actions = int(states.get("max_steps", 200))

        self._arcade = Arcade()
        self._card_id = self._arcade.open_scorecard(tags=["openrlhf", "ls20", "loopagent"])
        self._env = self._arcade.make(game_id, scorecard_id=self._card_id)

        raw_initial = self._env.observation_space
        self._frame = _convert_raw_frame_data(raw_initial)

        self._state_encoder.reset()
        self._surprise = HeuristicSurprise()
        self._initialize_memory()

        self._role = "DECIDER"
        self._mode = "LEARN_ACTION"
        self._active_plan_text = ""
        self._active_plan_steps = []
        self._active_subgoal = ""
        self._active_subgoal_index = 0
        self._pending_boundary = None
        self._terminal_after_learner = False
        self._episode_actions_used = 0
        self._running_return = 0.0
        self._turn_index = 0
        self._last_parse_warning = "none"
        self._seen_state_signatures = set()
        self._state_visit_counts = {}

        state_text = self._state_encoder.encode(self._frame)
        self._record_state_visit(self._frame)

        observation = self._build_decider_observation(state_text)
        out = dict(states)
        out.update(
            {
                "game_id": game_id,
                "observation": observation,
                "turn_index": self._turn_index,
                "running_return": self._running_return,
            }
        )
        return out

    async def step(
        self,
        states: dict[str, Any],
        action_text: str,
        **kwargs: Any,
    ) -> tuple[str, float, bool, dict[str, Any], dict[str, Any]]:
        del kwargs
        if self._env is None or self._frame is None:
            raise RuntimeError("Agent instance not initialized. reset() must run before step().")

        if self._role == "DECIDER":
            observation, reward, done, info = self._step_decider(action_text)
        else:
            observation, reward, done, info = self._step_learner(action_text)

        self._running_return += reward
        self._turn_index += 1

        out_states = dict(states)
        out_states.update(
            {
                "observation": observation,
                "turn_index": self._turn_index,
                "running_return": self._running_return,
            }
        )
        if done:
            self._snapshot_memory()
            self._close_scorecard()
        return observation, float(reward), done, info, out_states

    def _step_decider(self, raw_output: str) -> tuple[str, float, bool, dict[str, Any]]:
        assert self._frame is not None
        state_before = self._state_encoder.encode(self._frame)
        grid_before = self._frame.frame[-1] if self._frame.frame else []
        levels_before = self._frame.levels_completed
        available_actions = _available_action_names(self._frame)

        decision = self._parse_decider_output(raw_output, available_actions)
        selected_mode = decision["mode"]
        self._mode = selected_mode

        if decision["plan"]:
            self._active_plan_text = decision["plan"]
            self._active_plan_steps = self._parse_plan_steps(decision["plan"])
            if self._active_plan_steps:
                self._active_subgoal_index = 0
                self._active_subgoal = self._active_plan_steps[0]

        if decision["subgoal"]:
            self._active_subgoal = decision["subgoal"]

        actions = decision["actions"]
        if not actions:
            actions = [self._fallback_action(available_actions)]

        boundary_actions = actions[:1] if selected_mode == "LEARN_ACTION" else actions[: self._subgoal_max_actions]
        action_trace: list[str] = []
        parse_warning = decision["parse_warning"]

        no_change_streak = 0
        current_grid = grid_before
        terminal = False
        for token in boundary_actions:
            if self._episode_actions_used >= self._max_actions:
                break
            action_obj, action_name = self._to_game_action(token, available_actions)
            raw_after = self._env.step(
                action_obj,
                data=action_obj.action_data.model_dump(),
                reasoning={"source": "openrlhf_loopagent"},
            )
            frame_after = _convert_raw_frame_data(raw_after)
            self._frame = frame_after
            next_grid = frame_after.frame[-1] if frame_after.frame else []
            changed_step = self._state_encoder.get_num_changed_cells(current_grid, next_grid)
            current_grid = next_grid

            action_trace.append(action_name)
            self._episode_actions_used += 1

            if changed_step == 0:
                no_change_streak += 1
            else:
                no_change_streak = 0

            if frame_after.state in (GameState.WIN, GameState.GAME_OVER):
                terminal = True
                break
            if selected_mode != "LEARN_ACTION" and no_change_streak >= self._subgoal_no_change_limit:
                break

        if not action_trace:
            # Fallback execution when parse completely fails.
            fallback_action = self._fallback_action(available_actions)
            action_obj, action_name = self._to_game_action(fallback_action, available_actions)
            raw_after = self._env.step(
                action_obj,
                data=action_obj.action_data.model_dump(),
                reasoning={"source": "openrlhf_loopagent_fallback"},
            )
            self._frame = _convert_raw_frame_data(raw_after)
            action_trace = [action_name]
            self._episode_actions_used += 1
            parse_warning = "invalid_output_fallback"

        state_after = self._state_encoder.encode(self._frame)
        grid_after = self._frame.frame[-1] if self._frame.frame else []
        diff_text = self._state_encoder.get_diff_text(grid_before, grid_after)
        changed_cells = self._state_encoder.get_num_changed_cells(grid_before, grid_after)
        level_delta = self._frame.levels_completed - levels_before
        self._record_state_visit(self._frame)

        boundary_type = self._mode_to_boundary(self._mode)
        surprise = self._surprise.compute(
            state_before=state_before,
            action=", ".join(action_trace),
            state_after=state_after,
            memory=self._memory,
            num_changed_cells=changed_cells,
        )
        novelty_bonus = 1.0 if self._is_state_unseen(self._frame) else 0.0

        if self._mode == "SOLVE":
            reward = solver_reward(
                level_delta=level_delta,
                is_win=self._frame.state == GameState.WIN,
                is_game_over=self._frame.state == GameState.GAME_OVER,
                steps_used=max(1, len(action_trace)),
            ) + 0.5 * surprise
        else:
            reward = curiosity_reward(
                surprise_reward=surprise,
                boundary_type=boundary_type,
                novelty_bonus=novelty_bonus,
                transition_magnitude_bonus=float(changed_cells),
                boundary_progress_bonus=float(level_delta),
            )
        if parse_warning != "none":
            reward -= 0.25

        self._pending_boundary = PendingBoundary(
            mode=self._mode,
            boundary_type=boundary_type,
            state_before=state_before,
            state_after=state_after,
            diff_text=diff_text,
            expected=decision["expected"],
            action_trace=action_trace,
            changed_cells=changed_cells,
            level_delta=level_delta,
            terminal=terminal or self._frame.state in (GameState.WIN, GameState.GAME_OVER),
            parse_warning=parse_warning,
        )
        self._role = "LEARNER"
        self._terminal_after_learner = self._pending_boundary.terminal or (
            self._episode_actions_used >= self._max_actions
        )
        self._last_parse_warning = parse_warning

        observation = self._build_learner_observation(self._pending_boundary)
        info = {
            "role": "DECIDER",
            "mode": self._mode,
            "boundary_type": boundary_type,
            "action_trace": action_trace,
            "parse_warning": parse_warning,
            "changed_cells": changed_cells,
            "level_delta": level_delta,
            "game_state": self._frame.state.name,
            "actions_used": self._episode_actions_used,
        }
        # Always allow learner update step after boundary.
        return observation, float(reward), False, info

    def _step_learner(self, raw_output: str) -> tuple[str, float, bool, dict[str, Any]]:
        assert self._frame is not None
        if self._pending_boundary is None:
            # Safety fallback: switch back to DECIDER.
            self._role = "DECIDER"
            observation = self._build_decider_observation(self._state_encoder.encode(self._frame))
            return observation, 0.0, False, {
                "role": "LEARNER",
                "warning": "missing_pending_boundary",
            }

        pending = self._pending_boundary
        answer_output = self._extract_answer(raw_output)
        next_mode = self._parse_mode_label(raw_output)
        operation_lines = self._extract_operation_lines(answer_output)

        parse_ok = bool(operation_lines) or answer_output.strip().upper() == "NONE"
        before_missing = self._memory.missing_action_lessons(
            _available_action_names(self._frame)
        )
        memory_before_count = len(self._memory.entries)
        changed = False
        contradiction_cleanup = False

        for line in operation_lines:
            if line.upper() == "NONE":
                continue
            operation = parse_memory_operation(line, self._episode_actions_used)
            if operation.get("op") == "remove":
                contradiction_cleanup = True
            if apply_memory_operation(self._memory, operation, self._episode_actions_used):
                changed = True

        after_missing = self._memory.missing_action_lessons(
            _available_action_names(self._frame)
        )
        memory_after_count = len(self._memory.entries)
        non_duplicate = memory_after_count <= memory_before_count + 1

        mismatch_before = self._expected_mismatch(pending.expected, pending.state_after)
        mismatch_after = max(0.0, mismatch_before - (0.2 if changed else 0.0))
        reward = learner_reward(
            debiased_before=mismatch_before,
            debiased_after=mismatch_after,
            parse_ok=parse_ok,
            non_duplicate=non_duplicate,
            contradiction_cleanup=contradiction_cleanup,
        )
        reward += 0.25 * float(len(before_missing) - len(after_missing))
        if pending.parse_warning != "none":
            reward -= 0.1

        if next_mode:
            self._mode = next_mode
        elif self._frame.state == GameState.GAME_OVER:
            self._mode = "LEARN_ACTION"

        self._pending_boundary = None
        self._role = "DECIDER"

        done = bool(self._terminal_after_learner)
        self._terminal_after_learner = False
        if done:
            observation = "DONE: true"
        else:
            observation = self._build_decider_observation(self._state_encoder.encode(self._frame))

        info = {
            "role": "LEARNER",
            "mode": self._mode,
            "memory_changed": changed,
            "memory_size": len(self._memory.entries),
            "parse_ok": parse_ok,
            "next_mode": next_mode or "unchanged",
        }
        return observation, float(reward), done, info

    def _build_decider_observation(self, state_text: str) -> str:
        assert self._frame is not None
        available_actions = _available_action_names(self._frame)
        missing = self._memory.missing_action_lessons(available_actions)
        signature = self._frame_signature(self._frame)
        visits = self._state_visit_counts.get(signature, 1)
        return DECIDER_OBS_TEMPLATE.format(
            mode=self._mode,
            active_plan=self._active_plan_text or "none",
            active_subgoal=self._active_subgoal or "none",
            subgoal_index=self._active_subgoal_index if self._active_subgoal else "none",
            state_visits=visits,
            missing_action_lessons=", ".join(missing) if missing else "none",
            step_used=self._episode_actions_used,
            step_budget=self._max_actions,
            available_actions=", ".join(available_actions),
            state_text=state_text,
            memory_text=self._memory.to_text(),
            max_subgoal_actions=self._subgoal_max_actions,
        )

    def _build_learner_observation(self, pending: PendingBoundary) -> str:
        return LEARNER_OBS_TEMPLATE.format(
            mode=pending.mode,
            boundary_type=pending.boundary_type,
            subgoal_index=self._active_subgoal_index if self._active_subgoal else "none",
            action_trace=", ".join(pending.action_trace),
            expected=pending.expected or "none",
            state_before=pending.state_before,
            state_after=pending.state_after,
            diff_text=pending.diff_text,
            memory_text=self._memory.to_text(),
        )

    def _parse_decider_output(
        self,
        raw_output: str,
        available_actions: list[str],
    ) -> dict[str, Any]:
        mode = self._parse_mode_label(raw_output) or self._mode
        plan = self._extract_labeled_value(raw_output, "PLAN")
        subgoal = self._extract_labeled_value(raw_output, "SUBGOAL")
        action_payload = self._extract_labeled_value(raw_output, "ACTION_SEQUENCE")
        if not action_payload:
            action_payload = self._extract_answer(raw_output)
        expected = self._extract_labeled_value(raw_output, "EXPECTED")

        if mode in {"LEARN_SUBGOAL", "LEARN_PLAN", "SOLVE"}:
            actions = self._parse_action_list(action_payload, available_actions, self._subgoal_max_actions)
        else:
            actions = [self._parse_action(action_payload, available_actions)]

        parse_warning = "none"
        if not actions:
            parse_warning = "invalid_output_fallback"

        # If LEARN_PLAN has no explicit subgoal, derive from plan steps.
        if mode == "LEARN_PLAN" and not subgoal and plan:
            plan_steps = self._parse_plan_steps(plan)
            if plan_steps:
                subgoal = plan_steps[0]

        return {
            "mode": mode,
            "plan": (plan or "").strip(),
            "subgoal": (subgoal or "").strip(),
            "actions": actions,
            "expected": (expected or "").strip(),
            "parse_warning": parse_warning,
        }

    def _to_game_action(
        self,
        token: str,
        available_actions: list[str],
    ) -> tuple[GameAction, str]:
        text = token.strip().upper()
        if text.startswith("ACTION6") and "ACTION6" in available_actions:
            match = re.search(r"ACTION6\s*(\d+)?\s*(?:[, ]\s*(\d+))?", text)
            if match and match.group(1) and match.group(2):
                x = max(0, min(63, int(match.group(1))))
                y = max(0, min(63, int(match.group(2))))
            else:
                x, y = 32, 32
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y, "reasoning": "openrlhf_loopagent"})
            return action, f"ACTION6 {x} {y}"

        if text == "RESET" and "RESET" in available_actions:
            return GameAction.RESET, "RESET"

        name = self._parse_action(text, available_actions)
        if name == "RESET":
            return GameAction.RESET, "RESET"
        action = GameAction.from_name(name)
        action.reasoning = "openrlhf_loopagent"
        return action, name

    def _parse_action_list(
        self,
        raw_output: str,
        available_actions: list[str],
        max_steps: int,
    ) -> list[str]:
        text = raw_output.upper()
        out: list[str] = []

        action6_matches = re.findall(r"ACTION6\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?", text)
        action6_full: list[str] = []
        for x_str, y_str in action6_matches:
            x = max(0, min(63, int(x_str)))
            y = max(0, min(63, int(y_str)))
            action6_full.append(f"ACTION6 {x} {y}")

        generic = re.findall(r"RESET|ACTION\s*\d+", text)
        for token in generic:
            token = token.replace(" ", "")
            if token == "ACTION6":
                if action6_full and "ACTION6" in available_actions:
                    out.extend(action6_full)
                elif "ACTION6" in available_actions:
                    out.append("ACTION6 32 32")
            elif token in available_actions:
                out.append(token)

        valid: list[str] = []
        for action in out:
            if action.startswith("ACTION6"):
                if "ACTION6" in available_actions:
                    valid.append(action)
            elif action in available_actions:
                valid.append(action)

        if not valid:
            fallback = self._parse_action(raw_output, available_actions)
            return [fallback] if fallback else []
        return valid[: max(1, max_steps)]

    def _parse_action(self, raw_output: str, available_actions: list[str]) -> str:
        text = raw_output.upper().strip()

        if "ACTION6" in available_actions:
            action6_match = re.search(r"ACTION6\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?", text)
            if action6_match:
                x = max(0, min(63, int(action6_match.group(1))))
                y = max(0, min(63, int(action6_match.group(2))))
                return f"ACTION6 {x} {y}"
            if "ACTION6" in text:
                return "ACTION6 32 32"

        for action in available_actions:
            if action.upper() in text:
                return action

        match = re.search(r"ACTION\s*(\d+)", text)
        if match:
            action_name = f"ACTION{match.group(1)}"
            if action_name in available_actions:
                return action_name

        if "RESET" in text and "RESET" in available_actions:
            return "RESET"

        return self._fallback_action(available_actions)

    @staticmethod
    def _fallback_action(available_actions: list[str]) -> str:
        return next((a for a in available_actions if a != "RESET"), "RESET")

    def _extract_answer(self, raw_output: str) -> str:
        text = raw_output.strip()
        parts = re.split(r"(?im)^\s*ANSWER\s*:\s*", text)
        if len(parts) > 1:
            return parts[-1].strip()
        inline = re.split(r"(?i)\bANSWER\s*:\s*", text)
        if len(inline) > 1:
            return inline[-1].strip()
        return text

    @staticmethod
    def _extract_labeled_value(raw_output: str, label: str) -> str:
        pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+)$"
        matches = re.findall(pattern, raw_output.strip())
        if matches:
            return matches[-1].strip()
        return ""

    def _parse_mode_label(self, raw_output: str) -> Optional[str]:
        candidate = self._extract_labeled_value(raw_output, "MODE")
        text = (candidate or raw_output).upper()
        for mode in MODES:
            if mode in text:
                return mode
        next_mode = self._extract_labeled_value(raw_output, "NEXT_MODE").upper()
        for mode in MODES:
            if mode == next_mode:
                return mode
        return None

    @staticmethod
    def _parse_plan_steps(plan_text: str, max_steps: int = 5) -> list[str]:
        text = plan_text.strip()
        if not text:
            return []
        if ";" in text:
            parts = text.split(";")
        elif "->" in text:
            parts = text.split("->")
        else:
            parts = re.split(r"[\n\r]+", text)

        steps: list[str] = []
        for raw in parts:
            cleaned = re.sub(r"^\s*(?:\d+[\).:\-]?\s*|[-*]\s*)", "", raw).strip()
            cleaned = cleaned.strip().strip("<>").strip().strip("\"'")
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            lowered = cleaned.lower()
            if lowered in {"step", "step 1", "step 2", "step 3", "todo", "tbd"}:
                continue
            if re.fullmatch(r"steps?\s*\d*", lowered):
                continue
            if re.fullmatch(r"step[_\-\s]*\d+", lowered):
                continue
            if cleaned:
                steps.append(cleaned)
        return steps[: max(1, max_steps)]

    @staticmethod
    def _extract_operation_lines(text: str) -> list[str]:
        operations: list[str] = []
        saw_none = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.upper().startswith("NEXT_MODE:"):
                continue
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            line = re.sub(r"(?i)^\s*ANSWER\s*:\s*", "", line).strip()
            if not line:
                continue
            fragments = [frag.strip() for frag in line.split(";") if frag.strip()]
            for fragment in fragments:
                cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", fragment).strip()
                if cleaned.upper() == "NONE":
                    saw_none = True
                    continue
                if re.match(r"^(ADD|MODIFY|REMOVE|NONE)\b", cleaned, flags=re.IGNORECASE):
                    operations.append(cleaned)
        if operations:
            return operations
        if saw_none:
            return ["NONE"]
        return []

    @staticmethod
    def _expected_mismatch(expected: str, actual: str) -> float:
        exp_tokens = set(re.findall(r"[a-z0-9]+", (expected or "").lower()))
        act_tokens = set(re.findall(r"[a-z0-9]+", (actual or "").lower()))
        if not exp_tokens:
            return 0.5
        inter = len(exp_tokens & act_tokens)
        union = max(1, len(exp_tokens | act_tokens))
        similarity = inter / union
        return max(0.0, min(1.0, 1.0 - similarity))

    @staticmethod
    def _mode_to_boundary(mode: str) -> str:
        if mode == "LEARN_ACTION":
            return "action"
        if mode == "LEARN_SUBGOAL":
            return "subgoal"
        return "plan"

    def _frame_signature(self, frame: FrameData) -> str:
        grid = frame.frame[-1] if frame.frame else []
        return f"{frame.state.name}:{_grid_signature(grid)}"

    def _record_state_visit(self, frame: FrameData) -> None:
        sig = self._frame_signature(frame)
        self._state_visit_counts[sig] = self._state_visit_counts.get(sig, 0) + 1
        self._seen_state_signatures.add(sig)

    def _is_state_unseen(self, frame: FrameData) -> bool:
        sig = self._frame_signature(frame)
        return self._state_visit_counts.get(sig, 0) <= 1

    def _close_scorecard(self) -> None:
        if self._arcade is None or self._card_id is None:
            return
        try:
            self._arcade.close_scorecard(self._card_id)
        except Exception as err:  # pragma: no cover - cleanup best effort
            logger.warning("Failed to close scorecard %s: %s", self._card_id, err)
        finally:
            self._card_id = None

    def _initialize_memory(self) -> None:
        """Initialize episode memory using carry/noisy/blank curriculum."""
        self._memory.clear()
        snapshot = self._carry_memory_snapshot
        total = (
            max(0.0, self._memory_init_carry_p)
            + max(0.0, self._memory_init_noisy_p)
            + max(0.0, self._memory_init_blank_p)
        )
        if not snapshot or total <= 0.0:
            return

        carry_p = max(0.0, self._memory_init_carry_p) / total
        noisy_p = max(0.0, self._memory_init_noisy_p) / total
        sample = random.random()
        mode = "blank"
        if sample < carry_p:
            mode = "carry"
        elif sample < carry_p + noisy_p:
            mode = "noisy"

        if mode == "blank":
            return

        self._memory.entries = [replace(entry) for entry in snapshot]
        if mode == "noisy":
            self._memory.perturb(
                delete_fraction=self._noisy_delete_fraction,
                confidence_jitter=self._noisy_conf_jitter,
                shuffle_entries=True,
            )

    def _snapshot_memory(self) -> None:
        """Capture memory snapshot for next episode initialization."""
        self._carry_memory_snapshot = [replace(entry) for entry in self._memory.entries]


if MultiTurnAgentExecutor is None:  # pragma: no cover - hit only without openrlhf
    def _missing_openrlhf_agent(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "OpenRLHF is required to run this agent function. "
            "Install openrlhf and rerun training."
        ) from OPENRLHF_IMPORT_ERROR

    agent = _missing_openrlhf_agent
else:
    agent = MultiTurnAgentExecutor(LS20LoopAgentInstance)
