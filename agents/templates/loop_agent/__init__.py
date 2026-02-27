"""LoopAgent: Explore-Learn-Exploit agent for ARC-AGI-3.

Three-part agent (Curiosity, Learner, Solver) operating at three levels
of abstraction (Actions, Subgoals, Plans) with an adaptive explore-exploit loop.

Uses VLLM-served models via OpenAI-compatible API.
"""

import logging
import os
import re
import time
from enum import Enum
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from ...agent import Agent
from ...tracing import trace_agent_session
from .curiosity import Curiosity
from .learner import Learner
from .memory import Memory
from .solver import Solver
from .state_encoder import StateEncoder
from .surprise import (
    HeuristicSurprise,
    LevelController,
    LogProbSurprise,
    SurpriseStrategy,
)

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    EXPLORE = "EXPLORE"
    EXPLOIT = "EXPLOIT"


class LoopAgent(Agent):
    """Explore-Learn-Exploit agent with adaptive phase transitions.

    Overrides Agent.main() to add post-step handling:
    - Learner updates memory after actions/subgoal sequences
    - Subgoal-sequence execution with boundary evaluation
    - Level controller for abstraction level switching
    """

    MAX_ACTIONS = 80

    # VLLM configuration
    VLLM_BASE_URL: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_MODEL: str = os.environ.get("VLLM_MODEL", "google/gemma-3-1b-it")
    VLLM_API_KEY: str = os.environ.get("VLLM_API_KEY", "dummy")

    # Phase transition budgets
    EXPLORE_BUDGET_RATIO: float = float(os.environ.get("EXPLORE_BUDGET_RATIO", "0.25"))
    RE_EXPLORE_BUDGET: int = int(os.environ.get("RE_EXPLORE_BUDGET", "5"))

    # Strategy switches
    SURPRISE_STRATEGY: str = os.environ.get("SURPRISE_STRATEGY", "heuristic")
    MEMORY_PERSISTENCE_MODE: str = os.environ.get(
        "MEMORY_PERSISTENCE_MODE", "strict"
    ).lower()  # strict | carry | noisy
    # New names; keep backward-compatible env var fallbacks.
    USE_SUBGOAL_SEQUENCES: bool = os.environ.get(
        "USE_SUBGOAL_SEQUENCES",
        os.environ.get("USE_SUBGOAL_BURSTS", "true"),
    ).lower() == "true"
    SUBGOAL_MAX_ACTIONS: int = int(
        os.environ.get("SUBGOAL_MAX_ACTIONS", os.environ.get("BURST_MAX_STEPS", "4"))
    )
    SUBGOAL_NO_CHANGE_LIMIT: int = int(os.environ.get("SUBGOAL_NO_CHANGE_LIMIT", "2"))
    LEARNER_UPDATE_INTERVAL: int = int(os.environ.get("LEARNER_UPDATE_INTERVAL", "1"))
    KEYFRAME_INTERVAL: int = int(os.environ.get("STATE_KEYFRAME_INTERVAL", "10"))

    # Noisy-memory mode controls
    NOISY_DELETE_FRACTION: float = float(os.environ.get("NOISY_DELETE_FRACTION", "0.2"))
    NOISY_CONF_JITTER: float = float(os.environ.get("NOISY_CONF_JITTER", "0.1"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # VLLM client
        self.client = OpenAI(
            base_url=self.VLLM_BASE_URL,
            api_key=self.VLLM_API_KEY,
        )

        # Components
        self.memory = Memory(max_entries=50)
        self.state_encoder = StateEncoder(keyframe_interval=self.KEYFRAME_INTERVAL)
        self.curiosity = Curiosity(self.client, self.VLLM_MODEL)
        self.learner = Learner(self.client, self.VLLM_MODEL)
        self.solver = Solver(self.client, self.VLLM_MODEL)
        self.level_controller = LevelController(
            convergence_threshold=0.05,
            window_size=5,
        )

        # Surprise strategy
        if self.SURPRISE_STRATEGY == "logprob":
            self.surprise: SurpriseStrategy = LogProbSurprise(
                client=self.client,
                model=self.VLLM_MODEL,
            )
        else:
            self.surprise = HeuristicSurprise()

        # Phase management
        self.phase = Phase.EXPLORE
        self.explore_budget = int(self.MAX_ACTIONS * self.EXPLORE_BUDGET_RATIO)
        self.explore_actions_taken = 0
        self.re_explore_remaining = 0

        # State tracking
        self._last_state_text: Optional[str] = None
        self._current_state_text: Optional[str] = None
        self._last_prediction: str = ""
        self._levels_completed_at_reset: int = 0
        self._pending_subgoal_actions: list[GameAction] = []
        self._pending_subgoal_keyframe: bool = False
        self._active_plan_text: str = ""
        self._active_plan_steps: list[str] = []
        self._active_subgoal_index: int = 0
        self._active_subgoal: Optional[str] = None
        self._subgoal_sequence_active: bool = False
        self._subgoal_sequence_start_state: Optional[str] = None
        self._subgoal_sequence_start_grid: Optional[list[list[int]]] = None
        self._subgoal_sequence_expected: str = ""
        self._subgoal_sequence_planned_actions: list[str] = []
        self._subgoal_sequence_taken_actions: list[str] = []
        self._subgoal_no_change_streak: int = 0

    @trace_agent_session
    def main(self) -> None:
        """Overridden main loop with post-step learner/surprise handling."""
        self.timer = time.time()

        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            latest_frame = self._convert_raw_frame_data(
                self.arc_env.observation_space if self.arc_env else None
            )

            state_before = self.state_encoder.encode(latest_frame)
            self._current_state_text = state_before
            grid_before = latest_frame.frame[-1] if latest_frame.frame else []

            action = self.choose_action(self.frames, latest_frame)

            frame_after = self.take_action(action)
            if frame_after:
                self.append_frame(frame_after)
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, "
                    f"levels completed {frame_after.levels_completed}, "
                    f"phase {self.phase.value}, level {self.level_controller.current_level}, "
                    f"memory {len(self.memory)}, queued {len(self._pending_subgoal_actions)}, "
                    f"avg fps {self.fps}"
                )
                self._post_step(state_before, grid_before, action, frame_after)

            self.action_counter += 1

        self.cleanup()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action based on current phase."""
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._reset_subgoal_sequence_state(clear_pending=True)
            return GameAction.RESET

        # A full_reset flag appears on the frame produced by RESET. Reinitialize state
        # but continue with a normal next action (don't send RESET again).
        if latest_frame.full_reset:
            self._on_full_reset()

        if latest_frame.levels_completed > self._levels_completed_at_reset:
            self._on_level_complete(latest_frame)

        # Execute pre-planned subgoal actions first.
        if self._pending_subgoal_actions:
            return self._pending_subgoal_actions.pop(0)

        available_actions = self._get_available_action_names(latest_frame)
        state_text = (
            self._current_state_text
            or self._last_state_text
            or self.state_encoder.encode(latest_frame)
        )

        if self.phase == Phase.EXPLORE:
            return self._explore_action(state_text, available_actions)
        return self._exploit_action(state_text, available_actions, latest_frame)

    def _explore_action(self, state_text: str, available_actions: list[str]) -> GameAction:
        """Pick an action using curiosity."""
        level = self.level_controller.current_level
        result = self.curiosity.propose_action(
            state_text=state_text,
            memory=self.memory,
            available_actions=available_actions,
            level=level,
        )

        self.explore_actions_taken += 1

        if result["type"] == "plan":
            self._set_active_plan(
                plan_text=result["value"],
                plan_steps=result.get("steps", []),
            )
            result = self.curiosity.propose_action(
                state_text=state_text,
                memory=self.memory,
                available_actions=available_actions,
                level="action",
            )
        elif result["type"] == "subgoal":
            self._set_active_subgoal(result["value"])
            self._pending_subgoal_keyframe = True
            result = self.curiosity.propose_action(
                state_text=state_text,
                memory=self.memory,
                available_actions=available_actions,
                level="action",
            )

        self._last_prediction = result.get("prediction", "")
        action_name = result.get("value", "RESET")
        return self._name_to_game_action(action_name, available_actions)

    def _exploit_action(
        self,
        state_text: str,
        available_actions: list[str],
        latest_frame: FrameData,
    ) -> GameAction:
        """Pick an action using solver."""
        if self.USE_SUBGOAL_SEQUENCES and self.level_controller.current_level == "subgoal":
            self._ensure_active_plan(state_text, available_actions)
            self._ensure_active_subgoal()

            if self._active_subgoal:
                sequence = self.solver.propose_subgoal_actions(
                    state_text=state_text,
                    memory=self.memory,
                    available_actions=available_actions,
                    max_steps=self.SUBGOAL_MAX_ACTIONS,
                )
                action_names = sequence.get("actions", [])
                actions = [
                    self._name_to_game_action(action_name, available_actions)
                    for action_name in action_names
                ]
                if actions:
                    self._start_subgoal_sequence(
                        state_before=state_text,
                        grid_before=self._copy_grid(latest_frame.frame[-1] if latest_frame.frame else []),
                        planned_action_names=action_names,
                        expected=sequence.get("prediction", ""),
                    )
                    self._pending_subgoal_actions.extend(actions[1:])
                    self._last_prediction = sequence.get("prediction", "")
                    return actions[0]

        result = self.solver.solve(
            state_text=state_text,
            memory=self.memory,
            available_actions=available_actions,
        )
        self._last_prediction = result.get("prediction", "")
        action_name = result.get("action", "RESET")
        return self._name_to_game_action(action_name, available_actions)

    def _post_step(
        self,
        state_before: str,
        grid_before: list[list[int]],
        action: GameAction,
        frame_after: FrameData,
    ) -> None:
        """Post-step processing: subgoal-boundary learner updates and phase transitions."""
        state_after = self.state_encoder.encode(
            frame_after,
            force_keyframe=self._pending_subgoal_keyframe,
            keyframe_reason="subgoal_attempt" if self._pending_subgoal_keyframe else None,
        )
        self._pending_subgoal_keyframe = False
        grid_after = frame_after.frame[-1] if frame_after.frame else []

        diff_text = self.state_encoder.get_diff_text(grid_before, grid_after)
        num_changed = self.state_encoder.get_num_changed_cells(grid_before, grid_after)

        if self._subgoal_sequence_active:
            self._handle_subgoal_sequence_step(
                action=action,
                state_after=state_after,
                grid_after=grid_after,
                frame_after=frame_after,
                num_changed_cells=num_changed,
            )
        else:
            # Non-sequence step (typically exploration): keep per-step updates.
            learner_changed = False
            if self._should_run_learner(num_changed, frame_after):
                learner_changed = self.learner.update(
                    state_before=state_before,
                    action_taken=action.name,
                    state_after=state_after,
                    diff_text=diff_text,
                    memory=self.memory,
                    current_step=self.action_counter,
                    prediction=self._last_prediction,
                )
            else:
                logger.debug("Skipping learner update on this step")

            surprise_score = self.surprise.compute(
                state_before=state_before,
                action=action.name,
                state_after=state_after,
                memory=self.memory,
                num_changed_cells=num_changed,
            )
            self.level_controller.update(surprise_score)
            self._update_phase(learner_changed, frame_after)

        self._last_state_text = state_after
        self._current_state_text = None

    def _handle_subgoal_sequence_step(
        self,
        action: GameAction,
        state_after: str,
        grid_after: list[list[int]],
        frame_after: FrameData,
        num_changed_cells: int,
    ) -> None:
        """Track one executed subgoal action and finalize at boundary."""
        self._subgoal_sequence_taken_actions.append(action.name)
        if num_changed_cells == 0:
            self._subgoal_no_change_streak += 1
        else:
            self._subgoal_no_change_streak = 0

        terminal = frame_after.state in (GameState.WIN, GameState.GAME_OVER)
        no_change_interrupt = self._subgoal_no_change_streak >= self.SUBGOAL_NO_CHANGE_LIMIT
        if no_change_interrupt and self._pending_subgoal_actions:
            logger.info(
                "Subgoal sequence interrupted: no grid changes for "
                f"{self._subgoal_no_change_streak} consecutive actions"
            )
            self._pending_subgoal_actions.clear()

        sequence_done = terminal or not self._pending_subgoal_actions or no_change_interrupt
        if not sequence_done:
            return

        reason = "terminal" if terminal else "no_change" if no_change_interrupt else "exhausted"
        self._finalize_subgoal_sequence(
            state_after=state_after,
            grid_after=grid_after,
            frame_after=frame_after,
            reason=reason,
        )

    def _finalize_subgoal_sequence(
        self,
        state_after: str,
        grid_after: list[list[int]],
        frame_after: FrameData,
        reason: str,
    ) -> None:
        """Run learner/phase updates once at subgoal boundary."""
        start_state = self._subgoal_sequence_start_state or self._last_state_text or state_after
        start_grid = self._subgoal_sequence_start_grid or self._copy_grid(grid_after)
        action_trace = self._subgoal_sequence_taken_actions or self._subgoal_sequence_planned_actions
        if not action_trace:
            action_trace = ["RESET"]
        action_taken = f"SUBGOAL_SEQ[{', '.join(action_trace)}]"

        diff_text = self.state_encoder.get_diff_text(start_grid, grid_after)
        total_changed = self.state_encoder.get_num_changed_cells(start_grid, grid_after)

        learner_changed = self.learner.update(
            state_before=start_state,
            action_taken=action_taken,
            state_after=state_after,
            diff_text=diff_text,
            memory=self.memory,
            current_step=self.action_counter,
            prediction=self._subgoal_sequence_expected or self._last_prediction,
        )

        surprise_score = self.surprise.compute(
            state_before=start_state,
            action=action_taken,
            state_after=state_after,
            memory=self.memory,
            num_changed_cells=total_changed,
        )
        self.level_controller.update(surprise_score)
        self._update_phase(learner_changed, frame_after)

        terminal = frame_after.state in (GameState.WIN, GameState.GAME_OVER)
        if (
            self.phase == Phase.EXPLOIT
            and not terminal
            and reason == "exhausted"
            and not learner_changed
        ):
            advanced = self._advance_subgoal()
            if advanced:
                self.state_encoder.force_keyframe_next("subgoal_advance")

        self._reset_subgoal_sequence_state(clear_pending=True)

    def _set_active_plan(self, plan_text: str, plan_steps: list[str] | None = None) -> None:
        """Set currently active plan and initialize subgoal pointer."""
        cleaned_plan = plan_text.strip()
        steps = plan_steps if plan_steps is not None else self.curiosity.parse_plan_steps(cleaned_plan)
        steps = [s.strip() for s in steps if s.strip()]

        if not cleaned_plan and steps:
            cleaned_plan = "; ".join(f"{i+1}) {s}" for i, s in enumerate(steps))

        self._active_plan_text = cleaned_plan
        self._active_plan_steps = steps
        self._active_subgoal_index = 0
        self._active_subgoal = None

        if cleaned_plan:
            self.solver.set_plan(cleaned_plan)
        else:
            self.solver.clear_plan()

        if self._active_plan_steps:
            self._set_active_subgoal(self._active_plan_steps[0])
            logger.info(
                f"Active plan set with {len(self._active_plan_steps)} subgoals: "
                f"{self._active_plan_steps[:3]}"
            )
        else:
            self.solver.clear_subgoal()
            logger.info("Active plan set but no parseable subgoals were found")

    def _set_active_subgoal(self, subgoal: str) -> None:
        """Set the currently active subgoal."""
        text = subgoal.strip()
        if not text:
            return
        self._active_subgoal = text
        self.solver.set_subgoal(text)
        self._pending_subgoal_keyframe = True

    def _ensure_active_plan(self, state_text: str, available_actions: list[str]) -> None:
        """Create a plan if none is active."""
        if self._active_plan_steps:
            return
        result = self.curiosity.propose_action(
            state_text=state_text,
            memory=self.memory,
            available_actions=available_actions,
            level="plan",
        )
        if result.get("type") == "plan":
            self._set_active_plan(
                plan_text=result.get("value", ""),
                plan_steps=result.get("steps", []),
            )

    def _ensure_active_subgoal(self) -> None:
        """Set current subgoal from active plan if needed."""
        if self._active_subgoal:
            return
        if not self._active_plan_steps:
            return
        self._active_subgoal_index = min(
            max(self._active_subgoal_index, 0),
            len(self._active_plan_steps) - 1,
        )
        self._set_active_subgoal(self._active_plan_steps[self._active_subgoal_index])

    def _advance_subgoal(self) -> bool:
        """Advance to the next subgoal in the active plan."""
        if not self._active_plan_steps:
            return False
        next_idx = self._active_subgoal_index + 1
        if next_idx >= len(self._active_plan_steps):
            logger.info("Active plan exhausted; clearing plan state")
            self._reset_plan_state()
            return False
        self._active_subgoal_index = next_idx
        self._set_active_subgoal(self._active_plan_steps[next_idx])
        return True

    def _start_subgoal_sequence(
        self,
        state_before: str,
        grid_before: list[list[int]],
        planned_action_names: list[str],
        expected: str,
    ) -> None:
        """Initialize subgoal sequence tracking."""
        self._subgoal_sequence_active = True
        self._subgoal_sequence_start_state = state_before
        self._subgoal_sequence_start_grid = self._copy_grid(grid_before)
        self._subgoal_sequence_expected = expected.strip()
        self._subgoal_sequence_planned_actions = planned_action_names[:]
        self._subgoal_sequence_taken_actions = []
        self._subgoal_no_change_streak = 0
        self._pending_subgoal_keyframe = True
        logger.debug(
            f"Starting subgoal sequence with {len(planned_action_names)} actions: "
            f"{planned_action_names}"
        )

    def _reset_subgoal_sequence_state(self, clear_pending: bool = False) -> None:
        """Clear transient sequence execution state."""
        self._subgoal_sequence_active = False
        self._subgoal_sequence_start_state = None
        self._subgoal_sequence_start_grid = None
        self._subgoal_sequence_expected = ""
        self._subgoal_sequence_planned_actions = []
        self._subgoal_sequence_taken_actions = []
        self._subgoal_no_change_streak = 0
        if clear_pending:
            self._pending_subgoal_actions.clear()

    def _reset_plan_state(self) -> None:
        """Clear active plan and subgoal state."""
        self._active_plan_text = ""
        self._active_plan_steps = []
        self._active_subgoal_index = 0
        self._active_subgoal = None
        self.solver.clear_plan()

    def _should_run_learner(self, num_changed_cells: int, frame_after: FrameData) -> bool:
        """Decide whether to run the learner this step.

        Always runs unless we're mid-sequence AND nothing interesting happened.
        During sequences, we still run on terminal states or large grid changes.
        """
        in_sequence = len(self._pending_subgoal_actions) > 0
        interval_due = self.LEARNER_UPDATE_INTERVAL <= 1 or (
            self.action_counter % self.LEARNER_UPDATE_INTERVAL == 0
        )
        terminal = frame_after.state in (GameState.WIN, GameState.GAME_OVER)
        large_change = num_changed_cells > 20
        return interval_due and (not in_sequence or terminal or large_change)

    def _update_phase(self, learner_changed: bool, frame_after: FrameData) -> None:
        """Manage explore/exploit phase transitions.

        Phase transitions are driven by the learner's behavior:
        - Explore -> Exploit: when learner stops updating memory (consecutive NONEs)
        - Exploit -> Explore: when learner updates memory (something unexpected) or GAME_OVER

        The insight: if the learner changes memory, our world model was wrong.
        That IS the surprise signal. No threshold needed.
        """
        if self.phase == Phase.EXPLORE:
            if self.re_explore_remaining > 0:
                self.re_explore_remaining -= 1
                if self.learner.memory_is_stable:
                    logger.info("Phase: EXPLORE -> EXPLOIT (memory stable during re-explore)")
                    self.phase = Phase.EXPLOIT
                    self.re_explore_remaining = 0
                elif self.re_explore_remaining <= 0:
                    logger.info("Phase: EXPLORE -> EXPLOIT (re-explore budget exhausted)")
                    self.phase = Phase.EXPLOIT
                return

            if self.learner.memory_is_stable:
                logger.info(
                    f"Phase: EXPLORE -> EXPLOIT (memory stable after "
                    f"{self.explore_actions_taken} explore actions)"
                )
                self.phase = Phase.EXPLOIT
            elif self.explore_actions_taken >= self.explore_budget:
                logger.info(
                    f"Phase: EXPLORE -> EXPLOIT (explore budget exhausted: "
                    f"{self.explore_budget} actions)"
                )
                self.phase = Phase.EXPLOIT
            return

        if self.phase == Phase.EXPLOIT:
            # Re-enter explore when the learner updates memory (our model was wrong)
            # or on GAME_OVER (something clearly went wrong).
            should_reexplore = frame_after.state == GameState.GAME_OVER or learner_changed

            if should_reexplore:
                reason = (
                    "GAME_OVER"
                    if frame_after.state == GameState.GAME_OVER
                    else "learner updated memory (unexpected outcome)"
                )
                logger.info(f"Phase: EXPLOIT -> EXPLORE ({reason})")
                self.phase = Phase.EXPLORE
                self.re_explore_remaining = self.RE_EXPLORE_BUDGET
                self.learner.reset_stability()
                self.state_encoder.force_keyframe_next("phase_reentry")

                # Diagnose which level broke on GAME_OVER
                if frame_after.state == GameState.GAME_OVER:
                    diagnosis = self.learner.diagnose(
                        expected=self._last_prediction or "continued progress",
                        actual="GAME_OVER",
                        memory=self.memory,
                    )
                    if diagnosis:
                        self.level_controller.diagnose_to_level(diagnosis["level"])

    def _on_full_reset(self) -> None:
        """Handle full game reset with configurable memory persistence."""
        mode = self.MEMORY_PERSISTENCE_MODE
        if mode == "strict":
            logger.info("Full reset detected — clearing memory (strict mode)")
            self.memory.clear()
        elif mode == "noisy":
            logger.info("Full reset detected — perturbing memory (noisy mode)")
            self.memory.perturb(
                delete_fraction=self.NOISY_DELETE_FRACTION,
                confidence_jitter=self.NOISY_CONF_JITTER,
                shuffle_entries=True,
            )
        else:
            logger.info("Full reset detected — keeping memory (carry mode)")

        self.state_encoder.reset()
        self.phase = Phase.EXPLORE
        self.explore_actions_taken = 0
        self.re_explore_remaining = 0
        self.explore_budget = int(
            (self.MAX_ACTIONS - self.action_counter) * self.EXPLORE_BUDGET_RATIO
        )
        self.level_controller.reset()
        self.learner.reset_stability()
        self._reset_plan_state()
        self._reset_subgoal_sequence_state(clear_pending=True)
        self._pending_subgoal_keyframe = False
        self._last_state_text = None
        self._current_state_text = None
        self._levels_completed_at_reset = 0

    def _on_level_complete(self, frame: FrameData) -> None:
        """Handle level completion — brief re-explore for new level."""
        logger.info(
            f"Level complete! levels_completed={frame.levels_completed}. "
            "Re-entering explore for new level."
        )
        self._levels_completed_at_reset = frame.levels_completed
        self.state_encoder.reset()
        self.phase = Phase.EXPLORE
        self.explore_actions_taken = 0
        self.re_explore_remaining = 0
        remaining = self.MAX_ACTIONS - self.action_counter
        self.explore_budget = max(5, int(remaining * 0.15))
        self.level_controller.reset()
        self.learner.reset_stability()
        self._reset_plan_state()
        self._reset_subgoal_sequence_state(clear_pending=True)
        self._pending_subgoal_keyframe = False
        self._last_state_text = None
        self._current_state_text = None

    def _get_available_action_names(self, frame: FrameData) -> list[str]:
        """Build action name list from frame.available_actions."""
        if frame.available_actions:
            seen = set()
            names: list[str] = []
            for action_id in frame.available_actions:
                if action_id == 0:
                    name = "RESET"
                elif 1 <= action_id <= 7:
                    name = f"ACTION{action_id}"
                else:
                    continue
                if name not in seen:
                    names.append(name)
                    seen.add(name)
            if names:
                return names
        return [
            "RESET",
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION5",
            "ACTION6",
            "ACTION7",
        ]

    @staticmethod
    def _copy_grid(grid: list[list[int]]) -> list[list[int]]:
        return [row[:] for row in grid]

    def _fallback_action(self, available_actions: list[str]) -> GameAction:
        candidate = available_actions[0] if available_actions else "RESET"
        if candidate == "ACTION6":
            action = GameAction.ACTION6
            action.set_data({"x": 32, "y": 32})
            action.reasoning = {"x": 32, "y": 32, "agent": "loop_fallback"}
            return action
        try:
            action = GameAction.from_name(candidate)
            action.reasoning = f"LoopAgent fallback {self.phase.value}"
            return action
        except ValueError:
            return GameAction.RESET

    def _name_to_game_action(self, name: str, available_actions: list[str]) -> GameAction:
        """Convert action text to GameAction with strict available-action checks."""
        text = name.strip().upper()

        # ACTION6 with coordinates: "ACTION6 32 16" or "ACTION6 (32, 16)"
        action6_match = re.search(r"ACTION6\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?", text)
        if action6_match:
            if "ACTION6" not in available_actions:
                logger.warning("ACTION6 predicted but unavailable, using fallback action")
                return self._fallback_action(available_actions)
            x = max(0, min(63, int(action6_match.group(1))))
            y = max(0, min(63, int(action6_match.group(2))))
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y})
            action.reasoning = {"x": x, "y": y, "agent": "loop"}
            return action

        if "ACTION6" in text:
            if "ACTION6" not in available_actions:
                return self._fallback_action(available_actions)
            action = GameAction.ACTION6
            action.set_data({"x": 32, "y": 32})
            action.reasoning = {"x": 32, "y": 32, "agent": "loop_default_click"}
            return action

        match = re.search(r"\b(RESET|ACTION\s*\d+)\b", text)
        candidate = match.group(1).replace(" ", "") if match else text.replace(" ", "")
        try:
            action = GameAction.from_name(candidate)
            if action.name not in available_actions:
                logger.warning(
                    f"Action {action.name} predicted but unavailable "
                    f"(available={available_actions}); using fallback."
                )
                return self._fallback_action(available_actions)
            if action.is_simple():
                action.reasoning = f"LoopAgent {self.phase.value}"
            return action
        except ValueError:
            logger.warning(f"Unknown action name: {name}, using fallback action")
            return self._fallback_action(available_actions)


__all__ = ["LoopAgent"]
