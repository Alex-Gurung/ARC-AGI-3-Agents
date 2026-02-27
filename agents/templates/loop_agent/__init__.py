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
    - Learner updates memory after actions
    - Surprise computation for phase transitions
    - Level controller for abstraction level switching
    """

    MAX_ACTIONS = 80

    # VLLM configuration
    VLLM_BASE_URL: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_MODEL: str = os.environ.get("VLLM_MODEL", "google/gemma-3-1b-it")
    VLLM_API_KEY: str = os.environ.get("VLLM_API_KEY", "dummy")

    # Phase transition thresholds
    SURPRISE_THRESHOLD: float = float(os.environ.get("SURPRISE_THRESHOLD", "0.3"))
    EXPLORE_BUDGET_RATIO: float = float(os.environ.get("EXPLORE_BUDGET_RATIO", "0.25"))
    RE_EXPLORE_BUDGET: int = int(os.environ.get("RE_EXPLORE_BUDGET", "5"))

    # Strategy switches
    SURPRISE_STRATEGY: str = os.environ.get("SURPRISE_STRATEGY", "heuristic")
    MEMORY_PERSISTENCE_MODE: str = os.environ.get(
        "MEMORY_PERSISTENCE_MODE", "strict"
    ).lower()  # strict | carry | noisy
    USE_SUBGOAL_BURSTS: bool = os.environ.get("USE_SUBGOAL_BURSTS", "true").lower() == "true"
    BURST_MAX_STEPS: int = int(os.environ.get("BURST_MAX_STEPS", "4"))
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
        self._levels_completed_at_reset: int = 0
        self._pending_action_queue: list[GameAction] = []
        self._pending_subgoal_keyframe: bool = False

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
                    f"memory {len(self.memory)}, queued {len(self._pending_action_queue)}, "
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
        # Execute pre-planned burst actions first to reduce solver call overhead.
        if self._pending_action_queue:
            return self._pending_action_queue.pop(0)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # A full_reset flag appears on the frame produced by RESET. Reinitialize state
        # but continue with a normal next action (don't send RESET again).
        if latest_frame.full_reset:
            self._on_full_reset()

        if latest_frame.levels_completed > self._levels_completed_at_reset:
            self._on_level_complete(latest_frame)

        available_actions = self._get_available_action_names(latest_frame)
        state_text = (
            self._current_state_text
            or self._last_state_text
            or self.state_encoder.encode(latest_frame)
        )

        if self.phase == Phase.EXPLORE:
            return self._explore_action(state_text, available_actions)
        return self._exploit_action(state_text, available_actions)

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
            self.solver.set_plan(result["value"])
            result = self.curiosity.propose_action(
                state_text=state_text,
                memory=self.memory,
                available_actions=available_actions,
                level="action",
            )
        elif result["type"] == "subgoal":
            self.solver.set_subgoal(result["value"])
            # Force a keyframe after the next action to anchor the subgoal attempt.
            self._pending_subgoal_keyframe = True
            result = self.curiosity.propose_action(
                state_text=state_text,
                memory=self.memory,
                available_actions=available_actions,
                level="action",
            )

        action_name = result.get("value", "RESET")
        return self._name_to_game_action(action_name, available_actions)

    def _exploit_action(self, state_text: str, available_actions: list[str]) -> GameAction:
        """Pick an action using solver."""
        if (
            self.USE_SUBGOAL_BURSTS
            and self.level_controller.current_level == "subgoal"
            and self.solver.has_subgoal
        ):
            burst = self.solver.propose_burst(
                state_text=state_text,
                memory=self.memory,
                available_actions=available_actions,
                max_steps=self.BURST_MAX_STEPS,
            )
            burst_actions = [
                self._name_to_game_action(action_name, available_actions)
                for action_name in burst.get("actions", [])
            ]
            if burst_actions:
                self._pending_action_queue.extend(burst_actions[1:])
                logger.debug(
                    f"Using burst ({len(burst_actions)} steps): "
                    f"{[a.name for a in burst_actions]}"
                )
                return burst_actions[0]

        result = self.solver.solve(
            state_text=state_text,
            memory=self.memory,
            available_actions=available_actions,
        )
        action_name = result.get("action", "RESET")
        return self._name_to_game_action(action_name, available_actions)

    def _post_step(
        self,
        state_before: str,
        grid_before: list[list[int]],
        action: GameAction,
        frame_after: FrameData,
    ) -> None:
        """Post-step processing: surprise, learner, phase transitions, keyframes."""
        state_after = self.state_encoder.encode(
            frame_after,
            force_keyframe=self._pending_subgoal_keyframe,
            keyframe_reason="subgoal_attempt" if self._pending_subgoal_keyframe else None,
        )
        self._pending_subgoal_keyframe = False
        grid_after = frame_after.frame[-1] if frame_after.frame else []

        diff_text = self.state_encoder.get_diff_text(grid_before, grid_after)
        num_changed = self.state_encoder.get_num_changed_cells(grid_before, grid_after)

        surprise_score = self.surprise.compute(
            state_before=state_before,
            action=action.name,
            state_after=state_after,
            memory=self.memory,
            num_changed_cells=num_changed,
        )
        if surprise_score > self.SURPRISE_THRESHOLD:
            self.state_encoder.force_keyframe_next("surprise_spike")

        if self._should_run_learner(surprise_score, frame_after):
            self.learner.update(
                state_before=state_before,
                action_taken=action.name,
                state_after=state_after,
                diff_text=diff_text,
                memory=self.memory,
                current_step=self.action_counter,
            )
        else:
            logger.debug("Skipping learner update during burst execution")

        self.level_controller.update(surprise_score)
        self._update_phase(surprise_score, frame_after)

        self._last_state_text = state_after
        self._current_state_text = None

    def _should_run_learner(self, surprise_score: float, frame_after: FrameData) -> bool:
        in_burst = len(self._pending_action_queue) > 0
        interval_due = self.LEARNER_UPDATE_INTERVAL <= 1 or (
            self.action_counter % self.LEARNER_UPDATE_INTERVAL == 0
        )
        terminal = frame_after.state in (GameState.WIN, GameState.GAME_OVER)
        high_surprise = surprise_score > self.SURPRISE_THRESHOLD
        return interval_due and (not in_burst or terminal or high_surprise)

    def _update_phase(self, surprise_score: float, frame_after: FrameData) -> None:
        """Manage explore/exploit phase transitions."""
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
            if frame_after.state == GameState.GAME_OVER or surprise_score > self.SURPRISE_THRESHOLD:
                reason = (
                    "GAME_OVER"
                    if frame_after.state == GameState.GAME_OVER
                    else f"surprise={surprise_score:.3f}"
                )
                logger.info(f"Phase: EXPLOIT -> EXPLORE ({reason})")
                self.phase = Phase.EXPLORE
                self.re_explore_remaining = self.RE_EXPLORE_BUDGET
                self.learner.reset_stability()
                self.state_encoder.force_keyframe_next("phase_reentry")
                if frame_after.state == GameState.GAME_OVER:
                    diagnosis = self.learner.diagnose(
                        expected="continued progress",
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
        self.solver.clear_plan()
        self._pending_action_queue.clear()
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
        self.solver.clear_plan()
        self._pending_action_queue.clear()
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
