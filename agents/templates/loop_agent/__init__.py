"""LoopAgent: mode-routed agent for ARC-AGI-3.

Three-part agent (Curiosity, Learner, Solver) operating at three levels
of abstraction (Actions, Subgoals, Plans) with learned mode routing:
LEARN_ACTION / LEARN_SUBGOAL / LEARN_PLAN / SOLVE.

Uses VLLM-served models via OpenAI-compatible API.
"""

import hashlib
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
from .memory import Memory, MemoryEntry
from .runtime import LoopRuntime
from .solver import Solver
from .state_encoder import StateEncoder
from .surprise import (
    LevelController,
)

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    EXPLORE = "EXPLORE"
    EXPLOIT = "EXPLOIT"


class DecisionMode(str, Enum):
    LEARN_ACTION = "LEARN_ACTION"
    LEARN_SUBGOAL = "LEARN_SUBGOAL"
    LEARN_PLAN = "LEARN_PLAN"
    SOLVE = "SOLVE"


class LoopAgent(Agent):
    """Mode-routed agent with hierarchical subgoal-sequence execution.

    Overrides Agent.main() to add post-step handling:
    - Learner updates memory after actions/subgoal sequences
    - Subgoal-sequence execution with boundary evaluation
    - Diagnosis-directed mode routing + mode-router decisions
    """

    MAX_ACTIONS = int(os.environ.get("MAX_ACTIONS", "200"))

    # VLLM configuration
    VLLM_BASE_URL: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_MODEL: str = os.environ.get("VLLM_MODEL", "google/gemma-3-1b-it")
    VLLM_API_KEY: str = os.environ.get("VLLM_API_KEY", "dummy")

    # Strategy switches
    # Surprise computed via World Model + Observer + Judge pipeline.
    MEMORY_PERSISTENCE_MODE: str = os.environ.get(
        "MEMORY_PERSISTENCE_MODE", "strict"
    ).lower()  # strict | carry | noisy
    # New names; keep backward-compatible env var fallbacks.
    USE_SUBGOAL_SEQUENCES: bool = os.environ.get(
        "USE_SUBGOAL_SEQUENCES",
        os.environ.get("USE_SUBGOAL_BURSTS", "true"),
    ).lower() == "true"
    SUBGOAL_MAX_ACTIONS: int = int(
        os.environ.get("SUBGOAL_MAX_ACTIONS", os.environ.get("BURST_MAX_STEPS", "20"))
    )
    SUBGOAL_NO_CHANGE_LIMIT: int = int(os.environ.get("SUBGOAL_NO_CHANGE_LIMIT", "4"))
    LEARNER_UPDATE_INTERVAL: int = int(os.environ.get("LEARNER_UPDATE_INTERVAL", "1"))
    KEYFRAME_INTERVAL: int = int(os.environ.get("STATE_KEYFRAME_INTERVAL", "10"))
    USE_VISION: bool = os.environ.get("USE_VISION", "false").lower() == "true"
    VISION_CELL_SIZE: int = int(os.environ.get("VISION_CELL_SIZE", "8"))
    CURIOSITY_NUM_SAMPLES: int = int(os.environ.get("CURIOSITY_NUM_SAMPLES", "4"))
    LEARNER_NUM_SAMPLES: int = int(os.environ.get("LEARNER_NUM_SAMPLES", "4"))
    MISMATCH_CONF_THRESHOLD: float = float(os.environ.get("MISMATCH_CONF_THRESHOLD", "0.7"))
    MODE_SWITCH_COOLDOWN_BOUNDARIES: int = int(
        os.environ.get("MODE_SWITCH_COOLDOWN_BOUNDARIES", "2")
    )
    MEMORY_CONSOLIDATION_INTERVAL: int = int(
        os.environ.get("MEMORY_CONSOLIDATION_INTERVAL", "5")
    )

    # Noisy-memory mode controls
    NOISY_DELETE_FRACTION: float = float(os.environ.get("NOISY_DELETE_FRACTION", "0.2"))
    NOISY_CONF_JITTER: float = float(os.environ.get("NOISY_CONF_JITTER", "0.1"))
    ACTION_HISTORY_MAX: int = int(os.environ.get("ACTION_HISTORY_MAX", "15"))

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
        self.runtime = LoopRuntime(self)

        # Decision-state management
        self.phase = Phase.EXPLORE  # Backward-compatible label for prompt plumbing.
        self.current_mode: DecisionMode = DecisionMode.LEARN_ACTION
        self.explore_actions_taken = 0

        # State tracking
        self._last_state_text: Optional[str] = None
        self._current_state_text: Optional[str] = None
        self._last_prediction: str = ""
        self._last_surprise_score: float = 0.0
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
        self._subgoal_sequence_is_explore: bool = False
        self._subgoal_sequence_level: str = "action"
        self._pending_soft_reset: bool = False
        self._in_soft_reset: bool = False
        self._plan_attempt_active: bool = False
        self._plan_attempt_start_state: Optional[str] = None
        self._plan_attempt_start_grid: Optional[list[list[int]]] = None
        self._last_boundary_diagnosis_level: str = "none"
        self._repeat_state_streak: int = 0
        self._last_state_signature: Optional[str] = None
        self._state_visit_counts: dict[str, int] = {}
        self._state_action_attempt_counts: dict[str, dict[str, int]] = {}
        self._action_attempt_counts: dict[str, int] = {}
        self._mode_boundary_counts: dict[str, int] = {
            DecisionMode.LEARN_ACTION.value: 0,
            DecisionMode.LEARN_SUBGOAL.value: 0,
            DecisionMode.LEARN_PLAN.value: 0,
            DecisionMode.SOLVE.value: 0,
        }
        self._mode_switch_cooldown_remaining: int = 0
        self._action_history: list[dict[str, Any]] = []
        self._learner_steps_since_consolidation: int = 0

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

            action = self.runtime.step_decide(
                frames=self.frames,
                latest_frame=latest_frame,
                state_before=state_before,
                grid_before=grid_before,
            )

            frame_after = self.take_action(action)
            if frame_after:
                self.append_frame(frame_after)
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, "
                    f"levels completed {frame_after.levels_completed}, "
                    f"mode {self.current_mode.value}, level {self.level_controller.current_level}, "
                    f"memory {len(self.memory)}, queued {len(self._pending_subgoal_actions)}, "
                    f"avg fps {self.fps}"
                )
                self.runtime.step_observe(
                    state_before=state_before,
                    grid_before=grid_before,
                    action=action,
                    frame_after=frame_after,
                )

            self.action_counter += 1

        self.cleanup()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action based on current decision mode."""
        handled_full_reset = False
        if latest_frame.full_reset:
            handled_full_reset = True
            if self._in_soft_reset:
                self._on_soft_reset()
                self._in_soft_reset = False
            else:
                self._on_full_reset()

        if self._pending_soft_reset:
            self._pending_soft_reset = False
            self._in_soft_reset = True
            self._reset_subgoal_sequence_state(clear_pending=True)
            return GameAction.RESET

        if not handled_full_reset and latest_frame.state in (
            GameState.NOT_PLAYED,
            GameState.GAME_OVER,
        ):
            self._reset_subgoal_sequence_state(clear_pending=True)
            return GameAction.RESET

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
        state_image_url = self._state_image_data_url(latest_frame)
        state_signature = self._frame_signature(latest_frame)
        state_visit_count = self._state_visit_counts.get(state_signature, 0)
        missing_action_lessons_list = self.memory.missing_action_lessons(available_actions)
        missing_action_lessons = (
            ", ".join(missing_action_lessons_list)
            if missing_action_lessons_list
            else "none"
        )
        rulebook_status = self._rulebook_status_text(
            available_actions=available_actions,
            missing_action_lessons=missing_action_lessons_list,
        )
        semantic_discovery_status = self._semantic_discovery_status(state_text)
        action_try_counts = self._action_try_counts_text(
            available_actions=available_actions,
            state_signature=state_signature,
        )
        early_exploration_hint = self._early_exploration_hint(
            available_actions=available_actions,
            state_visit_count=state_visit_count,
            state_signature=state_signature,
        )

        if self.current_mode == DecisionMode.SOLVE:
            return self._exploit_action(
                state_text,
                available_actions,
                latest_frame,
                forced_level="plan",
                state_image_url=state_image_url,
            )

        learn_level = self._mode_to_level(self.current_mode)
        return self._explore_action(
            state_text,
            available_actions,
            latest_frame,
            forced_level=learn_level,
            state_image_url=state_image_url,
            state_visit_count=state_visit_count,
            action_try_counts=action_try_counts,
            early_exploration_hint=early_exploration_hint,
            state_signature=state_signature,
            rulebook_status=rulebook_status,
            missing_action_lessons=missing_action_lessons,
            missing_action_lessons_list=missing_action_lessons_list,
            semantic_discovery_status=semantic_discovery_status,
        )

    def _explore_action(
        self,
        state_text: str,
        available_actions: list[str],
        latest_frame: FrameData,
        forced_level: str | None = None,
        state_image_url: str | None = None,
        state_visit_count: int = 0,
        action_try_counts: str = "none",
        early_exploration_hint: str = "none",
        state_signature: str | None = None,
        rulebook_status: str = "none",
        missing_action_lessons: str = "none",
        missing_action_lessons_list: list[str] | None = None,
        semantic_discovery_status: str = "none",
    ) -> GameAction:
        """Pick an exploratory action or subgoal sequence using curiosity."""
        level = forced_level or self.level_controller.current_level
        self.level_controller.current_level = level
        stage_subgoal_index = self._active_subgoal_index if self._active_subgoal else None
        grid_before = self._copy_grid(latest_frame.frame[-1] if latest_frame.frame else [])

        if level == "plan":
            plan_result = self._sample_curiosity_plan(
                state_text=state_text,
                available_actions=available_actions,
                phase=self._phase_label(),
                active_plan=self._active_plan_text,
                active_subgoal=self._active_subgoal or "none",
                subgoal_index=stage_subgoal_index,
                image_data_url=state_image_url,
                state_visit_count=state_visit_count,
                action_try_counts=action_try_counts,
                early_exploration_hint=early_exploration_hint,
                rulebook_status=rulebook_status,
                missing_action_lessons=missing_action_lessons,
                semantic_discovery_status=semantic_discovery_status,
            )
            if plan_result.get("type") == "plan":
                self._set_active_plan(
                    plan_text=plan_result.get("value", ""),
                    plan_steps=plan_result.get("steps", []),
                )

        if level in {"subgoal", "plan"} or self._active_subgoal:
            if self._active_subgoal:
                seq = self._sample_curiosity_subgoal_sequence(
                    state_text=state_text,
                    available_actions=available_actions,
                    max_steps=self.SUBGOAL_MAX_ACTIONS,
                    active_subgoal=self._active_subgoal,
                    phase=self._phase_label(),
                    level=level,
                    subgoal_index=self._active_subgoal_index if self._active_subgoal else None,
                    image_data_url=state_image_url,
                    state_visit_count=state_visit_count,
                    action_try_counts=action_try_counts,
                    early_exploration_hint=early_exploration_hint,
                    state_signature=state_signature,
                    rulebook_status=rulebook_status,
                    missing_action_lessons=missing_action_lessons,
                    missing_action_lessons_list=missing_action_lessons_list or [],
                    semantic_discovery_status=semantic_discovery_status,
                )
                proposed_subgoal = str(seq.get("subgoal", "")).strip()
                if proposed_subgoal:
                    self._set_active_subgoal(proposed_subgoal)
                success_test = str(seq.get("success_test", "")).strip()
                action_names = seq.get("actions", [])
                actions = [
                    self._name_to_game_action(action_name, available_actions)
                    for action_name in action_names
                ]
                if actions:
                    self._start_subgoal_sequence(
                        state_before=state_text,
                        grid_before=grid_before,
                        planned_action_names=action_names,
                        expected=seq.get("prediction", "") or success_test,
                        is_explore=True,
                        level=level,
                    )
                    self._pending_subgoal_actions.extend(actions[1:])
                    self._last_prediction = seq.get("prediction", "")
                    return actions[0]
            else:
                seq = self._sample_curiosity_subgoal_sequence(
                    state_text=state_text,
                    available_actions=available_actions,
                    max_steps=self.SUBGOAL_MAX_ACTIONS,
                    active_subgoal="none",
                    phase=self._phase_label(),
                    level=level,
                    subgoal_index=None,
                    image_data_url=state_image_url,
                    state_visit_count=state_visit_count,
                    action_try_counts=action_try_counts,
                    early_exploration_hint=early_exploration_hint,
                    state_signature=state_signature,
                    rulebook_status=rulebook_status,
                    missing_action_lessons=missing_action_lessons,
                    missing_action_lessons_list=missing_action_lessons_list or [],
                    semantic_discovery_status=semantic_discovery_status,
                )
                proposed_subgoal = str(seq.get("subgoal", "")).strip()
                if proposed_subgoal:
                    self._set_active_subgoal(proposed_subgoal)
                    action_names = seq.get("actions", [])
                    actions = [
                        self._name_to_game_action(action_name, available_actions)
                        for action_name in action_names
                    ]
                    if actions:
                        self._start_subgoal_sequence(
                            state_before=state_text,
                            grid_before=grid_before,
                            planned_action_names=action_names,
                            expected=seq.get("prediction", "") or str(seq.get("success_test", "")).strip(),
                            is_explore=True,
                            level=level,
                        )
                        self._pending_subgoal_actions.extend(actions[1:])
                        self._last_prediction = seq.get("prediction", "")
                        return actions[0]

        action_result = self._sample_curiosity_action(
            state_text=state_text,
            available_actions=available_actions,
            phase=self._phase_label(),
            active_plan=self._active_plan_text,
            active_subgoal=self._active_subgoal or "none",
            subgoal_index=self._active_subgoal_index if self._active_subgoal else None,
            image_data_url=state_image_url,
            state_visit_count=state_visit_count,
            action_try_counts=action_try_counts,
            early_exploration_hint=early_exploration_hint,
            state_signature=state_signature,
            rulebook_status=rulebook_status,
            missing_action_lessons=missing_action_lessons,
            missing_action_lessons_list=missing_action_lessons_list or [],
            semantic_discovery_status=semantic_discovery_status,
        )
        self.explore_actions_taken += 1
        self._last_prediction = action_result.get("prediction", "")
        action_name = action_result.get("value", "RESET")
        return self._name_to_game_action(action_name, available_actions)

    def _exploit_action(
        self,
        state_text: str,
        available_actions: list[str],
        latest_frame: FrameData,
        forced_level: str | None = None,
        state_image_url: str | None = None,
    ) -> GameAction:
        """Pick an action using solver."""
        level = forced_level or self.level_controller.current_level
        self.level_controller.current_level = level

        if level == "plan":
            self._ensure_active_plan(state_text, available_actions, state_image_url)
            self._ensure_active_subgoal()

        if self.USE_SUBGOAL_SEQUENCES and level in {"subgoal", "plan"}:
            self._ensure_active_plan(state_text, available_actions, state_image_url)
            self._ensure_active_subgoal()

            if self._active_subgoal:
                stage_subgoal_index = (
                    self._active_subgoal_index if self._active_subgoal else None
                )
                sequence = self.solver.propose_subgoal_actions(
                    state_text=state_text,
                    memory=self.memory,
                    available_actions=available_actions,
                    max_steps=self.SUBGOAL_MAX_ACTIONS,
                    phase=self._phase_label(),
                    level=level,
                    subgoal_index=stage_subgoal_index,
                    image_data_url=state_image_url,
                    action_history=self._action_history_text(),
                )
                action_names = sequence.get("actions", [])
                actions = [
                    self._name_to_game_action(action_name, available_actions)
                    for action_name in action_names
                ]
                if actions:
                    if level == "plan" and not self._plan_attempt_active:
                        self._plan_attempt_active = True
                        self._plan_attempt_start_state = state_text
                        self._plan_attempt_start_grid = self._copy_grid(
                            latest_frame.frame[-1] if latest_frame.frame else []
                        )
                    self._start_subgoal_sequence(
                        state_before=state_text,
                        grid_before=self._copy_grid(latest_frame.frame[-1] if latest_frame.frame else []),
                        planned_action_names=action_names,
                        expected=sequence.get("prediction", ""),
                        is_explore=False,
                        level=level,
                    )
                    self._pending_subgoal_actions.extend(actions[1:])
                    self._last_prediction = sequence.get("prediction", "")
                    return actions[0]

        stage_subgoal_index = self._active_subgoal_index if self._active_subgoal else None
        result = self.solver.solve(
            state_text=state_text,
            memory=self.memory,
            available_actions=available_actions,
            phase=self._phase_label(),
            level=level,
            subgoal_index=stage_subgoal_index,
            image_data_url=state_image_url,
            action_history=self._action_history_text(),
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
        if action == GameAction.RESET and self._in_soft_reset and frame_after.full_reset:
            logger.debug("Soft reset step processed (metadata only; skipping learn/surprise)")
            self._current_state_text = None
            return

        state_after = self.state_encoder.encode(
            frame_after,
            force_keyframe=self._pending_subgoal_keyframe,
            keyframe_reason="subgoal_attempt" if self._pending_subgoal_keyframe else None,
        )
        self._pending_subgoal_keyframe = False
        grid_after = frame_after.frame[-1] if frame_after.frame else []
        available_actions_after = self._get_available_action_names(frame_after)
        missing_after_list = self.memory.missing_action_lessons(available_actions_after)
        missing_after_text = ", ".join(missing_after_list) if missing_after_list else "none"
        rulebook_after = self._rulebook_status_text(
            available_actions=available_actions_after,
            missing_action_lessons=missing_after_list,
        )

        diff_text = self.state_encoder.get_diff_text(grid_before, grid_after)
        num_changed = self.state_encoder.get_num_changed_cells(grid_before, grid_after)
        state_before_image = self._grid_image_data_url(grid_before)
        state_after_image = self._grid_image_data_url(grid_after)
        transition_image = self._transition_image_data_url(grid_before, grid_after)

        if self._subgoal_sequence_active:
            self._handle_subgoal_sequence_step(
                action=action,
                state_after=state_after,
                grid_after=grid_after,
                frame_after=frame_after,
                num_changed_cells=num_changed,
            )
        else:
            # --- World Model + Observer + Judge + Scribe pipeline ---

            # 1. WORLD MODEL: predict what should have happened
            predicted = self.learner.predict_outcome(
                state_before=state_before,
                action_taken=action.name,
                memory=self.memory,
                image_before_url=state_before_image,
            )

            # 2. OBSERVER: describe what actually happened
            observed = self.learner.observe_transition(
                state_before=state_before,
                state_after=state_after,
                diff_text=diff_text,
                image_before_url=state_before_image,
                image_after_url=state_after_image,
                image_diff_url=transition_image,
            )

            # 3. JUDGE: score prediction accuracy
            similarity = self.learner.judge_similarity(predicted, observed)
            surprise_score = (6 - similarity) / 5.0  # normalize to [0, 1]
            self._last_surprise_score = surprise_score

            # 4. SCRIBE: update memory with full context
            if self._should_run_learner(num_changed, frame_after):
                self._learner_update_best_of_n(
                    state_before=state_before,
                    action_taken=action.name,
                    state_after=state_after,
                    diff_text=diff_text,
                    memory=self.memory,
                    current_step=self.action_counter,
                    prediction=self._last_prediction,
                    phase=self._phase_label(),
                    level=self.level_controller.current_level,
                    subgoal_index=self._active_subgoal_index if self._active_subgoal else None,
                    image_before_url=state_before_image,
                    image_after_url=state_after_image,
                    image_diff_url=transition_image,
                    rulebook_status=rulebook_after,
                    missing_action_lessons=missing_after_text,
                    predicted_description=predicted,
                    observed_description=observed,
                    similarity=similarity,
                )
                self._learner_steps_since_consolidation += 1
            else:
                logger.debug("Skipping learner update on this step")

            self._maybe_consolidate_memory()

            # Record surprise per level
            level = self._mode_to_level(self.current_mode)
            self._record_surprise(level, surprise_score)
            game_event = None
            if frame_after.state == GameState.GAME_OVER:
                game_event = "GAME_OVER"
            self._route_mode_after_boundary(
                state_text=state_after,
                frame_after=frame_after,
                state_image_url=state_after_image,
                game_event=game_event,
            )

        state_signature_after = self._frame_signature(frame_after)
        prev_state_signature = self._last_state_signature
        self._state_visit_counts[state_signature_after] = (
            self._state_visit_counts.get(state_signature_after, 0) + 1
        )
        if state_signature_after not in self._state_action_attempt_counts:
            self._state_action_attempt_counts[state_signature_after] = {}
        per_state_action_counts = self._state_action_attempt_counts[state_signature_after]
        per_state_action_counts[action.name] = per_state_action_counts.get(action.name, 0) + 1
        self._action_attempt_counts[action.name] = self._action_attempt_counts.get(action.name, 0) + 1

        self._last_state_text = state_after
        self._last_state_signature = state_signature_after
        if action != GameAction.RESET and prev_state_signature == state_signature_after:
            self._repeat_state_streak += 1
        else:
            self._repeat_state_streak = 0
        self._current_state_text = None

        # Record action in history ring buffer
        event = None
        if frame_after.state == GameState.GAME_OVER:
            event = "GAME_OVER"
        history_entry: dict[str, Any] = {
            "step": self.action_counter,
            "action": action.name,
            "changed": num_changed,
            "event": event,
        }
        self._action_history.append(history_entry)
        if len(self._action_history) > self.ACTION_HISTORY_MAX:
            self._action_history.pop(0)

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
        start_image = self._grid_image_data_url(start_grid)
        end_image = self._grid_image_data_url(grid_after)
        transition_image = self._transition_image_data_url(start_grid, grid_after)
        available_actions_after = self._get_available_action_names(frame_after)
        missing_after_list = self.memory.missing_action_lessons(available_actions_after)
        missing_after_text = ", ".join(missing_after_list) if missing_after_list else "none"
        rulebook_after = self._rulebook_status_text(
            available_actions=available_actions_after,
            missing_action_lessons=missing_after_list,
        )

        # --- World Model + Observer + Judge + Scribe pipeline (boundary) ---

        # 1. WORLD MODEL: predict what the subgoal sequence should have produced
        predicted = self.learner.predict_outcome(
            state_before=start_state,
            action_taken=action_taken,
            memory=self.memory,
            image_before_url=start_image,
        )

        # 2. OBSERVER: describe what actually happened
        observed = self.learner.observe_transition(
            state_before=start_state,
            state_after=state_after,
            diff_text=diff_text,
            image_before_url=start_image,
            image_after_url=end_image,
            image_diff_url=transition_image,
        )

        # 3. JUDGE: score prediction accuracy
        similarity = self.learner.judge_similarity(predicted, observed)
        surprise_score = (6 - similarity) / 5.0  # normalize to [0, 1]
        self._last_surprise_score = surprise_score
        self._record_surprise(self._subgoal_sequence_level, surprise_score)

        # 4. SCRIBE: update memory with full context
        learner_changed = self._learner_update_best_of_n(
            state_before=start_state,
            action_taken=action_taken,
            state_after=state_after,
            diff_text=diff_text,
            memory=self.memory,
            current_step=self.action_counter,
            prediction=self._subgoal_sequence_expected or self._last_prediction,
            phase=self._phase_label(),
            level=self.level_controller.current_level,
            subgoal_index=self._active_subgoal_index if self._active_subgoal else None,
            image_before_url=start_image,
            image_after_url=end_image,
            image_diff_url=transition_image,
            rulebook_status=rulebook_after,
            missing_action_lessons=missing_after_text,
            predicted_description=predicted,
            observed_description=observed,
            similarity=similarity,
        )
        self._learner_steps_since_consolidation += 1
        self._maybe_consolidate_memory()

        if self._subgoal_sequence_is_explore:
            self.explore_actions_taken += 1
            self._pending_soft_reset = True

        terminal = frame_after.state in (GameState.WIN, GameState.GAME_OVER)
        advanced = False
        if (
            not self._subgoal_sequence_is_explore
            and self.phase == Phase.EXPLOIT
            and not terminal
            and reason == "exhausted"
            and not learner_changed
        ):
            advanced = self._advance_subgoal()
            if advanced:
                self.state_encoder.force_keyframe_next("subgoal_advance")

        if self._subgoal_sequence_level == "plan":
            plan_attempt_ended = terminal or (reason == "exhausted" and not advanced)
            if plan_attempt_ended:
                self._plan_attempt_active = False
                self._plan_attempt_start_state = None
                self._plan_attempt_start_grid = None

        if (
            not self._subgoal_sequence_is_explore
            and self.phase == Phase.EXPLOIT
            and not terminal
            and reason == "exhausted"
            and not learner_changed
        ):
            if not advanced:
                self._plan_attempt_active = False
                self._plan_attempt_start_state = None
                self._plan_attempt_start_grid = None

        game_event = None
        if frame_after.state == GameState.GAME_OVER:
            game_event = "GAME_OVER"
        self._route_mode_after_boundary(
            state_text=state_after,
            frame_after=frame_after,
            state_image_url=end_image,
            game_event=game_event,
        )

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
        self._plan_attempt_active = bool(steps)
        self._plan_attempt_start_state = self._current_state_text or self._last_state_text
        self._plan_attempt_start_grid = None

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

    def _ensure_active_plan(
        self,
        state_text: str,
        available_actions: list[str],
        state_image_url: str | None = None,
    ) -> None:
        """Create a plan if none is active."""
        if self._active_plan_steps:
            return
        missing_actions = self.memory.missing_action_lessons(available_actions)
        result = self.curiosity.propose_action(
            state_text=state_text,
            memory=self.memory,
            available_actions=available_actions,
            level="plan",
            phase=self._phase_label(),
            active_plan=self._active_plan_text,
            active_subgoal=self._active_subgoal or "none",
            subgoal_index=self._active_subgoal_index if self._active_subgoal else None,
            image_data_url=state_image_url,
            state_visit_count=self._state_visit_counts.get(self._last_state_signature or "", 0),
            action_try_counts=self._action_try_counts_text(
                available_actions=available_actions,
                state_signature=self._last_state_signature,
            ),
            early_exploration_hint=self._early_exploration_hint(
                available_actions=available_actions,
                state_visit_count=self._state_visit_counts.get(
                    self._last_state_signature or "",
                    0,
                ),
                state_signature=self._last_state_signature,
            ),
            rulebook_status=self._rulebook_status_text(
                available_actions=available_actions,
                missing_action_lessons=missing_actions,
            ),
            missing_action_lessons=", ".join(missing_actions) if missing_actions else "none",
            semantic_discovery_status=self._semantic_discovery_status(state_text),
            action_history=self._action_history_text(),
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
        is_explore: bool,
        level: str,
    ) -> None:
        """Initialize subgoal sequence tracking."""
        self._subgoal_sequence_active = True
        self._subgoal_sequence_start_state = state_before
        self._subgoal_sequence_start_grid = self._copy_grid(grid_before)
        self._subgoal_sequence_expected = expected.strip()
        self._subgoal_sequence_planned_actions = planned_action_names[:]
        self._subgoal_sequence_taken_actions = []
        self._subgoal_no_change_streak = 0
        self._subgoal_sequence_is_explore = is_explore
        self._subgoal_sequence_level = level
        self._pending_subgoal_keyframe = True
        if level == "plan" and self._plan_attempt_active and self._plan_attempt_start_grid is None:
            self._plan_attempt_start_state = state_before
            self._plan_attempt_start_grid = self._copy_grid(grid_before)
        logger.debug(
            f"Starting subgoal sequence with {len(planned_action_names)} actions: "
            f"{planned_action_names} (explore={is_explore}, level={level})"
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
        self._subgoal_sequence_is_explore = False
        self._subgoal_sequence_level = "action"
        if clear_pending:
            self._pending_subgoal_actions.clear()

    def _reset_plan_state(self) -> None:
        """Clear active plan and subgoal state."""
        self._active_plan_text = ""
        self._active_plan_steps = []
        self._active_subgoal_index = 0
        self._active_subgoal = None
        self._plan_attempt_active = False
        self._plan_attempt_start_state = None
        self._plan_attempt_start_grid = None
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

    def _phase_label(self) -> str:
        return "SOLVE" if self.current_mode == DecisionMode.SOLVE else "LEARN"

    def _action_history_text(self) -> str:
        """Format recent action history as compact text for prompts."""
        if not self._action_history:
            return "no actions taken yet"
        lines: list[str] = []
        for entry in self._action_history:
            action = entry["action"]
            line = f"  step {entry['step']}: {action}"
            changed = entry.get("changed", 0)
            if changed > 0:
                line += f" -> {changed} cells changed"
            elif action not in ("RESET", "LEVEL_COMPLETE", "STUCK"):
                line += " -> no change"
            event = entry.get("event")
            if event == "GAME_OVER":
                line += " [GAME_OVER: level failed, resetting]"
            elif event == "SOFT_RESET":
                line += " [SOFT_RESET: agent chose to reset after exploration]"
            elif event == "LEVEL_COMPLETE":
                line += " [LEVEL_COMPLETE: solved! advancing to next level]"
            elif event == "STUCK_RESET":
                line += " [STUCK: same state repeated, switching to LEARN_ACTION]"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _mode_to_level(mode: DecisionMode) -> str:
        return {
            DecisionMode.LEARN_ACTION: "action",
            DecisionMode.LEARN_SUBGOAL: "subgoal",
            DecisionMode.LEARN_PLAN: "plan",
            DecisionMode.SOLVE: "plan",
        }[mode]

    @staticmethod
    def _level_to_mode(level: str) -> DecisionMode:
        return {
            "action": DecisionMode.LEARN_ACTION,
            "subgoal": DecisionMode.LEARN_SUBGOAL,
            "plan": DecisionMode.LEARN_PLAN,
        }.get(level, DecisionMode.LEARN_ACTION)

    def _set_mode(self, mode: DecisionMode, reason: str) -> None:
        if mode == self.current_mode:
            return
        old_mode = self.current_mode
        self.current_mode = mode
        self.phase = Phase.EXPLOIT if mode == DecisionMode.SOLVE else Phase.EXPLORE
        self.level_controller.current_level = self._mode_to_level(mode)
        self._mode_switch_cooldown_remaining = max(0, self.MODE_SWITCH_COOLDOWN_BOUNDARIES)
        self._mode_boundary_counts[mode.value] = 0
        logger.info("Mode transition: %s -> %s (%s)", old_mode.value, mode.value, reason)

        if mode == DecisionMode.LEARN_ACTION:
            self._reset_plan_state()
        elif mode == DecisionMode.LEARN_PLAN:
            self._reset_plan_state()

    @staticmethod
    def _action_base_name(action_name: str) -> str:
        text = action_name.strip().upper()
        if text.startswith("ACTION6"):
            return "ACTION6"
        match = re.search(r"\b(RESET|ACTION\d+)\b", text)
        return match.group(1) if match else text

    def _sample_curiosity_action(
        self,
        *,
        state_text: str,
        available_actions: list[str],
        phase: str,
        active_plan: str,
        active_subgoal: str,
        subgoal_index: int | None,
        image_data_url: str | None,
        state_visit_count: int,
        action_try_counts: str,
        early_exploration_hint: str,
        state_signature: str | None,
        rulebook_status: str,
        missing_action_lessons: str,
        missing_action_lessons_list: list[str],
        semantic_discovery_status: str,
    ) -> dict[str, Any]:
        n = max(1, self.CURIOSITY_NUM_SAMPLES)
        candidates: list[dict[str, Any]] = []
        for _ in range(n):
            candidates.append(
                self.curiosity.propose_action(
                    state_text=state_text,
                    memory=self.memory,
                    available_actions=available_actions,
                    level="action",
                    phase=phase,
                    active_plan=active_plan,
                    active_subgoal=active_subgoal,
                    subgoal_index=subgoal_index,
                    image_data_url=image_data_url,
                    state_visit_count=state_visit_count,
                    action_try_counts=action_try_counts,
                    early_exploration_hint=early_exploration_hint,
                    rulebook_status=rulebook_status,
                    missing_action_lessons=missing_action_lessons,
                    semantic_discovery_status=semantic_discovery_status,
                    action_history=self._action_history_text(),
                )
            )
        if n == 1 or len(candidates) == 1:
            return candidates[0]

        best_idx = 0
        best_score = float("-inf")
        state_counts = (
            self._state_action_attempt_counts.get(state_signature, {})
            if state_signature
            else {}
        )
        for idx, candidate in enumerate(candidates):
            action_value = self._action_base_name(str(candidate.get("value", "")))
            local_count = state_counts.get(action_value, 0)
            global_count = self._action_attempt_counts.get(action_value, 0)
            score = 0.0
            score += 3.0 / (1.0 + local_count)
            score += 1.0 / (1.0 + global_count)
            if action_value in missing_action_lessons_list:
                score += 4.0
            if action_value == "RESET" and missing_action_lessons_list:
                score -= 4.0
            if action_value == "RESET":
                score -= 1.0
            prediction = str(candidate.get("prediction", "")).strip()
            if prediction:
                score += 0.2
            if score > best_score:
                best_score = score
                best_idx = idx
        return candidates[best_idx]

    def _sample_curiosity_plan(
        self,
        *,
        state_text: str,
        available_actions: list[str],
        phase: str,
        active_plan: str,
        active_subgoal: str,
        subgoal_index: int | None,
        image_data_url: str | None,
        state_visit_count: int,
        action_try_counts: str,
        early_exploration_hint: str,
        rulebook_status: str,
        missing_action_lessons: str,
        semantic_discovery_status: str,
    ) -> dict[str, Any]:
        n = max(1, self.CURIOSITY_NUM_SAMPLES)
        candidates: list[dict[str, Any]] = []
        for _ in range(n):
            candidates.append(
                self.curiosity.propose_action(
                    state_text=state_text,
                    memory=self.memory,
                    available_actions=available_actions,
                    level="plan",
                    phase=phase,
                    active_plan=active_plan,
                    active_subgoal=active_subgoal,
                    subgoal_index=subgoal_index,
                    image_data_url=image_data_url,
                    state_visit_count=state_visit_count,
                    action_try_counts=action_try_counts,
                    early_exploration_hint=early_exploration_hint,
                    rulebook_status=rulebook_status,
                    missing_action_lessons=missing_action_lessons,
                    semantic_discovery_status=semantic_discovery_status,
                    action_history=self._action_history_text(),
                )
            )
        if n == 1 or len(candidates) == 1:
            return candidates[0]

        best_idx = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(candidates):
            steps = candidate.get("steps", []) or []
            text = str(candidate.get("value", "")).lower()
            if not steps:
                score = -6.0
            else:
                placeholder_steps = 0
                generic_steps = 0
                informative_steps = 0
                for step in steps:
                    s = str(step).strip().lower()
                    if re.fullmatch(r"steps?\s*\d*", s) or re.fullmatch(r"step[_\-\s]*\d+", s):
                        placeholder_steps += 1
                        continue
                    tokens = re.findall(r"[a-z0-9]+", s)
                    if len(tokens) < 3:
                        generic_steps += 1
                    else:
                        informative_steps += 1
                score = 0.0
                score -= 4.0 * placeholder_steps
                score -= 1.5 * generic_steps
                score += 1.5 * informative_steps
                score += 0.5 * len(set(str(step).strip().lower() for step in steps if str(step).strip()))
            if re.fullmatch(r"(step[_\-\s]*\d+\s*;?\s*)+", text.strip()):
                score -= 8.0
            if "hypothesis" in text:
                score += 0.25
            if score > best_score:
                best_score = score
                best_idx = idx
        return candidates[best_idx]

    def _sample_curiosity_subgoal_sequence(
        self,
        *,
        state_text: str,
        available_actions: list[str],
        max_steps: int,
        active_subgoal: str,
        phase: str,
        level: str,
        subgoal_index: int | None,
        image_data_url: str | None,
        state_visit_count: int,
        action_try_counts: str,
        early_exploration_hint: str,
        state_signature: str | None,
        rulebook_status: str,
        missing_action_lessons: str,
        missing_action_lessons_list: list[str],
        semantic_discovery_status: str,
    ) -> dict[str, Any]:
        n = max(1, self.CURIOSITY_NUM_SAMPLES)
        candidates: list[dict[str, Any]] = []
        for _ in range(n):
            candidates.append(
                self.curiosity.propose_subgoal_actions(
                    state_text=state_text,
                    memory=self.memory,
                    available_actions=available_actions,
                    max_steps=max_steps,
                    active_subgoal=active_subgoal,
                    phase=phase,
                    level=level,
                    subgoal_index=subgoal_index,
                    image_data_url=image_data_url,
                    state_visit_count=state_visit_count,
                    action_try_counts=action_try_counts,
                    early_exploration_hint=early_exploration_hint,
                    rulebook_status=rulebook_status,
                    missing_action_lessons=missing_action_lessons,
                    semantic_discovery_status=semantic_discovery_status,
                    action_history=self._action_history_text(),
                )
            )
        if n == 1 or len(candidates) == 1:
            only = candidates[0]
            only_actions = [
                self._action_base_name(str(a))
                for a in only.get("actions", [])
            ]
            only["actions"] = only_actions
            return only

        best_idx = 0
        best_score = float("-inf")
        state_counts = (
            self._state_action_attempt_counts.get(state_signature, {})
            if state_signature
            else {}
        )
        for idx, candidate in enumerate(candidates):
            actions = [self._action_base_name(str(a)) for a in candidate.get("actions", [])]
            if not actions:
                continue
            unique_actions = set(actions)
            score = 0.0
            score += 0.5 * len(unique_actions)
            for action_name in unique_actions:
                local_count = state_counts.get(action_name, 0)
                global_count = self._action_attempt_counts.get(action_name, 0)
                score += 2.0 / (1.0 + local_count)
                score += 0.5 / (1.0 + global_count)
                if action_name in missing_action_lessons_list:
                    score += 2.5
                if action_name == "RESET":
                    score -= 2.0
            if score > best_score:
                best_score = score
                best_idx = idx
        selected = candidates[best_idx]
        selected_actions = [
            self._action_base_name(str(a))
            for a in selected.get("actions", [])
        ]
        selected["actions"] = selected_actions
        return selected

    @staticmethod
    def _clone_memory(memory: Memory) -> Memory:
        cloned = Memory(max_entries=memory.MAX_ENTRIES)
        cloned._next_id = memory._next_id
        cloned.entries = [
            MemoryEntry(
                type=entry.type,
                content=entry.content,
                justification=entry.justification,
                confidence=entry.confidence,
                created_step=entry.created_step,
                last_modified_step=entry.last_modified_step,
                memory_id=entry.memory_id,
            )
            for entry in memory.entries
        ]
        return cloned

    def _score_learner_candidate(
        self,
        *,
        before_memory: Memory,
        after_memory: Memory,
        action_taken: str,
        answer_text: str,
        state_after: str,
        available_actions: list[str],
    ) -> float:
        score = 0.0
        missing_before = before_memory.missing_action_lessons(available_actions)
        missing_after = after_memory.missing_action_lessons(available_actions)
        score += 2.0 * (len(missing_before) - len(missing_after))
        if len(after_memory.entries) != len(before_memory.entries):
            score += 0.25

        action_base = self._action_base_name(action_taken)
        if action_base.startswith("ACTION") and action_base in missing_before and action_base not in missing_after:
            score += 4.0

        return score

    @staticmethod
    def _estimate_changed_cells(diff_text: str) -> int:
        """Estimate changed-cell count from DIFF text for surprise scoring."""
        match = re.search(r"CHANGED\s*\((\d+)\s*cells?\)", diff_text, flags=re.IGNORECASE)
        if match:
            return max(0, int(match.group(1)))
        if "no changes" in diff_text.lower():
            return 0
        return 1

    def _learner_update_best_of_n(
        self,
        *,
        state_before: str,
        action_taken: str,
        state_after: str,
        diff_text: str,
        memory: Memory,
        current_step: int,
        prediction: str,
        phase: str,
        level: str,
        subgoal_index: int | None,
        image_before_url: str | None,
        image_after_url: str | None,
        image_diff_url: str | None,
        rulebook_status: str,
        missing_action_lessons: str,
        predicted_description: str = "",
        observed_description: str = "",
        similarity: int = 0,
    ) -> bool:
        action_history = self._action_history_text()
        n = max(1, self.LEARNER_NUM_SAMPLES)
        if n == 1:
            return self.learner.update(
                state_before=state_before,
                action_taken=action_taken,
                state_after=state_after,
                diff_text=diff_text,
                memory=memory,
                current_step=current_step,
                prediction=prediction,
                phase=phase,
                level=level,
                subgoal_index=subgoal_index,
                image_before_url=image_before_url,
                image_after_url=image_after_url,
                image_diff_url=image_diff_url,
                rulebook_status=rulebook_status,
                missing_action_lessons=missing_action_lessons,
                action_history=action_history,
                predicted_description=predicted_description,
                observed_description=observed_description,
                similarity=similarity,
            )

        original_nones = self.learner.consecutive_nones
        available_actions = self._extract_available_actions_from_text(state_after)
        before_snapshot = self._clone_memory(memory)
        candidates: list[dict[str, Any]] = []

        for _ in range(n):
            candidate_memory = self._clone_memory(memory)
            changed = self.learner.update(
                state_before=state_before,
                action_taken=action_taken,
                state_after=state_after,
                diff_text=diff_text,
                memory=candidate_memory,
                current_step=current_step,
                prediction=prediction,
                phase=phase,
                level=level,
                subgoal_index=subgoal_index,
                image_before_url=image_before_url,
                image_after_url=image_after_url,
                image_diff_url=image_diff_url,
                rulebook_status=rulebook_status,
                missing_action_lessons=missing_action_lessons,
                action_history=action_history,
                predicted_description=predicted_description,
                observed_description=observed_description,
                similarity=similarity,
            )
            answer_text = self.learner.last_answer_output
            raw_text = self.learner.last_raw_output
            score = self._score_learner_candidate(
                before_memory=before_snapshot,
                after_memory=candidate_memory,
                action_taken=action_taken,
                answer_text=answer_text,
                state_after=state_after,
                available_actions=available_actions,
            )
            # Surprise scoring deferred to real self-rated call after
            # candidate selection; no per-candidate LLM calls.
            surprise_score = 0.0
            candidates.append(
                {
                    "memory": candidate_memory,
                    "changed": changed,
                    "answer": answer_text,
                    "raw": raw_text,
                    "score": score,
                    "surprise": surprise_score,
                }
            )

        best = (
            min(
                candidates,
                key=lambda c: (
                    float(c["surprise"]),
                    -float(c["score"]),
                ),
            )
            if candidates
            else None
        )
        if not best:
            self.learner.consecutive_nones = original_nones + 1
            return False

        best_memory: Memory = best["memory"]
        memory.entries = best_memory.entries
        memory._next_id = best_memory._next_id
        self.learner.last_answer_output = str(best["answer"])
        self.learner.last_raw_output = str(best["raw"])
        if bool(best["changed"]):
            self.learner.consecutive_nones = 0
        else:
            self.learner.consecutive_nones = original_nones + 1
        return bool(best["changed"])

    @staticmethod
    def _extract_available_actions_from_text(state_text: str) -> list[str]:
        """Parse AVAILABLE_ACTIONS from encoded state text, fallback to default set."""
        match = re.search(r"(?im)^\s*AVAILABLE_ACTIONS\s*:\s*(.+)$", state_text)
        if not match:
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
        actions = [token.strip().upper() for token in match.group(1).split(",") if token.strip()]
        return actions or [
            "RESET",
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION5",
            "ACTION6",
            "ACTION7",
        ]

    def _maybe_consolidate_memory(self) -> None:
        """Run periodic memory consolidation if interval is reached."""
        if self.MEMORY_CONSOLIDATION_INTERVAL <= 0:
            return
        if self._learner_steps_since_consolidation >= self.MEMORY_CONSOLIDATION_INTERVAL:
            self._learner_steps_since_consolidation = 0
            ops = self.learner.consolidate(self.memory, self.action_counter)
            if ops:
                logger.info("Memory consolidation applied %d operations", ops)

    def _record_surprise(self, level: str, surprise: float) -> None:
        level_key = level if level in self.level_controller.surprise_history else "action"
        self.level_controller.surprise_history[level_key].append(surprise)
        self.level_controller.current_level = level_key

    def _surprise_summary_text(self) -> str:
        """Build per-level surprise summary for the router prompt."""
        lines = []
        for level_key in ("action", "subgoal", "plan"):
            history = self.level_controller.surprise_history.get(level_key, [])
            if not history:
                lines.append(f"  {level_key}: no data")
                continue
            avg = sum(history) / len(history)
            last_5 = history[-5:]
            last_5_str = ", ".join(f"{v:.2f}" for v in last_5)
            lines.append(
                f"  {level_key}: avg={avg:.2f}, last_5=[{last_5_str}], total={len(history)}"
            )
        return "\n".join(lines)

    def _route_mode_after_boundary(
        self,
        state_text: str,
        frame_after: FrameData,
        state_image_url: str | None = None,
        game_event: str | None = None,
    ) -> None:
        self._mode_boundary_counts[self.current_mode.value] += 1

        if self._mode_switch_cooldown_remaining > 0:
            self._mode_switch_cooldown_remaining -= 1
            logger.debug(
                "Skipping mode-router during cooldown (%d boundaries remaining)",
                self._mode_switch_cooldown_remaining,
            )
            return

        available_actions = self._get_available_action_names(frame_after)
        missing_actions = self.memory.missing_action_lessons(available_actions)
        mode_result = self.curiosity.propose_mode(
            state_text=state_text,
            memory=self.memory,
            available_actions=available_actions,
            current_mode=self.current_mode.value,
            active_plan=self._active_plan_text or "none",
            active_subgoal=self._active_subgoal or "none",
            last_prediction=self._last_prediction or "none",
            last_diagnosis_level=self._last_boundary_diagnosis_level,
            rulebook_status=self._rulebook_status_text(
                available_actions=available_actions,
                missing_action_lessons=missing_actions,
            ),
            missing_action_lessons=", ".join(missing_actions) if missing_actions else "none",
            semantic_discovery_status=self._semantic_discovery_status(state_text),
            action_history=self._action_history_text(),
            game_event=game_event or "none",
            surprise_summary=self._surprise_summary_text(),
        )
        proposed_mode = str(mode_result.get("mode", self.current_mode.value)).upper()
        try:
            target_mode = DecisionMode(proposed_mode)
        except ValueError:
            target_mode = self.current_mode
        target_mode = self._gate_router_mode(target_mode)
        self._set_mode(target_mode, reason="mode router decision")

    def _gate_router_mode(self, proposed_mode: DecisionMode) -> DecisionMode:
        """Pass through the router's mode decision without heuristic gates."""
        return proposed_mode

    def _on_full_reset(self) -> None:
        """Handle full game reset with configurable memory persistence."""
        self._action_history.append({
            "step": self.action_counter,
            "action": "RESET",
            "changed": 0,
            "event": "GAME_OVER",
        })
        if len(self._action_history) > self.ACTION_HISTORY_MAX:
            self._action_history.pop(0)
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
        self.current_mode = DecisionMode.LEARN_ACTION
        self.phase = Phase.EXPLORE
        self.explore_actions_taken = 0
        self.level_controller.reset()
        self.learner.reset_stability()
        self._reset_plan_state()
        self._reset_subgoal_sequence_state(clear_pending=True)
        self._pending_subgoal_keyframe = False
        self._pending_soft_reset = False
        self._in_soft_reset = False
        self._last_boundary_diagnosis_level = "none"
        self._last_state_text = None
        self._current_state_text = None
        self._last_state_signature = None
        self._levels_completed_at_reset = 0
        self._repeat_state_streak = 0
        self._state_visit_counts = {}
        self._state_action_attempt_counts = {}
        self._action_attempt_counts = {}
        self._mode_boundary_counts = {
            DecisionMode.LEARN_ACTION.value: 0,
            DecisionMode.LEARN_SUBGOAL.value: 0,
            DecisionMode.LEARN_PLAN.value: 0,
            DecisionMode.SOLVE.value: 0,
        }
        self._mode_switch_cooldown_remaining = 0

    def _on_soft_reset(self) -> None:
        """Handle a control-flow reset after exploratory subgoal attempts."""
        self._action_history.append({
            "step": self.action_counter,
            "action": "RESET",
            "changed": 0,
            "event": "SOFT_RESET",
        })
        if len(self._action_history) > self.ACTION_HISTORY_MAX:
            self._action_history.pop(0)
        logger.info("Soft reset detected — keeping memory/phase/level, clearing transient plan state")
        self.state_encoder.reset()
        self.state_encoder.force_keyframe_next("soft_reset")
        self._reset_plan_state()
        self._reset_subgoal_sequence_state(clear_pending=True)
        self._pending_subgoal_keyframe = False
        self.phase = Phase.EXPLOIT if self.current_mode == DecisionMode.SOLVE else Phase.EXPLORE
        self._last_state_text = None
        self._current_state_text = None
        self._last_state_signature = None
        self._repeat_state_streak = 0

    def _on_level_complete(self, frame: FrameData) -> None:
        """Handle level completion — brief re-explore for new level."""
        self._action_history.append({
            "step": self.action_counter,
            "action": "LEVEL_COMPLETE",
            "changed": 0,
            "event": "LEVEL_COMPLETE",
        })
        if len(self._action_history) > self.ACTION_HISTORY_MAX:
            self._action_history.pop(0)
        logger.info(
            f"Level complete! levels_completed={frame.levels_completed}. "
            "Re-entering explore for new level."
        )
        self._levels_completed_at_reset = frame.levels_completed
        self.state_encoder.reset()
        self.current_mode = DecisionMode.LEARN_ACTION
        self.phase = Phase.EXPLORE
        self.explore_actions_taken = 0
        self.level_controller.reset()
        self.learner.reset_stability()
        self._reset_plan_state()
        self._reset_subgoal_sequence_state(clear_pending=True)
        self._pending_subgoal_keyframe = False
        self._pending_soft_reset = False
        self._in_soft_reset = False
        self._last_boundary_diagnosis_level = "none"
        self._last_state_text = None
        self._current_state_text = None
        self._last_state_signature = None
        self._repeat_state_streak = 0
        self._state_visit_counts = {}
        self._state_action_attempt_counts = {}
        self._action_attempt_counts = {}
        self._mode_boundary_counts = {
            DecisionMode.LEARN_ACTION.value: 0,
            DecisionMode.LEARN_SUBGOAL.value: 0,
            DecisionMode.LEARN_PLAN.value: 0,
            DecisionMode.SOLVE.value: 0,
        }
        self._mode_switch_cooldown_remaining = 0

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

    def _action_try_counts_text(
        self,
        available_actions: list[str],
        state_signature: str | None = None,
    ) -> str:
        """Summarize per-action attempt counts for prompt context."""
        state_counts = (
            self._state_action_attempt_counts.get(state_signature, {})
            if state_signature
            else {}
        )
        parts: list[str] = []
        for action_name in available_actions:
            local = state_counts.get(action_name, 0)
            global_count = self._action_attempt_counts.get(action_name, 0)
            parts.append(f"{action_name}:{local} (global={global_count})")
        return ", ".join(parts) if parts else "none"

    def _rulebook_status_text(
        self,
        available_actions: list[str],
        missing_action_lessons: list[str],
    ) -> str:
        """Compact summary of rule-book coverage for prompts."""
        total = len(available_actions)
        covered = max(0, total - len(missing_action_lessons))
        return f"ACTION lessons covered={covered}/{total}"

    def _semantic_discovery_status(self, state_text: str) -> str:
        """Summarize semantic-understanding coverage for curiosity prompts."""
        object_ids = sorted(set(re.findall(r"\bO\d+\b", state_text)))
        relation_match = re.search(r"RELATIONS\s*\((\d+)\)", state_text)
        relations_count = int(relation_match.group(1)) if relation_match else 0
        total_lessons = len(self.memory) if self.memory else 0
        return (
            f"total_lessons={total_lessons} "
            f"objects_in_view={len(object_ids)} "
            f"relations_in_view={relations_count}"
        )

    def _early_exploration_hint(
        self,
        available_actions: list[str],
        state_visit_count: int,
        state_signature: str | None = None,
    ) -> str:
        """Give curiosity lightweight guidance for early diversified exploration."""
        if self.current_mode != DecisionMode.LEARN_ACTION:
            return "none"
        state_counts = (
            self._state_action_attempt_counts.get(state_signature, {})
            if state_signature
            else {}
        )
        untested = [
            action_name
            for action_name in available_actions
            if action_name != "RESET" and state_counts.get(action_name, 0) == 0
        ]
        if untested:
            return (
                "Novelty objective: prioritize actions that may reach unseen states. "
                "In this exact state, try untested primitive actions first: "
                + ", ".join(untested)
            )
        if state_visit_count >= 3:
            return (
                f"This state has been visited {state_visit_count} times; "
                "prefer the least-tried action in this state to break loops."
            )
        return "none"

    @staticmethod
    def _grid_signature(grid: list[list[int]]) -> str:
        """Compact stable hash for a grid state."""
        if not grid or not grid[0]:
            return "empty"
        height = len(grid)
        width = len(grid[0])
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(f"{width}x{height}|".encode("ascii"))
        for row in grid:
            hasher.update(bytes(row))
        return hasher.hexdigest()

    def _frame_signature(self, frame: FrameData) -> str:
        """Signature for visit counting using rendered grid + game state."""
        grid = frame.frame[-1] if frame.frame else []
        base = self._grid_signature(grid)
        return f"{frame.state.name}:{base}"

    def _state_image_data_url(self, frame: FrameData) -> str | None:
        """Render current frame as a PNG data URL when vision is enabled."""
        if not self.USE_VISION:
            return None
        image_url = self.state_encoder.frame_to_image_data_url(
            frame=frame,
            cell_size=self.VISION_CELL_SIZE,
        )
        return image_url or None

    def _grid_image_data_url(self, grid: list[list[int]]) -> str | None:
        """Render a raw grid as a PNG data URL when vision is enabled."""
        if not self.USE_VISION:
            return None
        image_url = self.state_encoder.grid_to_image_data_url(
            grid=grid,
            cell_size=self.VISION_CELL_SIZE,
        )
        return image_url or None

    def _transition_image_data_url(
        self,
        grid_before: list[list[int]],
        grid_after: list[list[int]],
    ) -> str | None:
        """Render BEFORE|AFTER|DIFF transition visualization for multimodal calls."""
        if not self.USE_VISION:
            return None
        image_url = self.state_encoder.transition_image_data_url(
            grid_before=grid_before,
            grid_after=grid_after,
            cell_size=self.VISION_CELL_SIZE,
        )
        return image_url or None

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
            action.reasoning = f"LoopAgent {self.current_mode.value}"
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
                action.reasoning = f"LoopAgent {self.current_mode.value}"
            return action
        except ValueError:
            logger.warning(f"Unknown action name: {name}, using fallback action")
            return self._fallback_action(available_actions)


__all__ = ["DecisionMode", "LoopAgent"]
