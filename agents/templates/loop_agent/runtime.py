"""Runtime helpers for LoopAgent decision/observation orchestration.

This module provides a light extraction layer so training harnesses can
reuse LoopAgent's decision and post-step logic while supporting snapshot/
restore around grouped branch evaluation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arcengine import FrameData, GameAction

if TYPE_CHECKING:
    from . import LoopAgent


@dataclass
class RuntimeSnapshot:
    """Serializable runtime state for branch replay."""

    scalar_state: dict[str, Any]
    memory_state: Any
    state_encoder_state: dict[str, Any]
    level_controller_state: dict[str, Any]
    solver_state: dict[str, Any]
    learner_state: dict[str, Any]


class LoopRuntime:
    """Thin runtime adapter around LoopAgent's current behavior."""

    SNAPSHOT_FIELDS = (
        "phase",
        "current_mode",
        "explore_actions_taken",
        "_last_state_text",
        "_current_state_text",
        "_last_prediction",
        "_levels_completed_at_reset",
        "_pending_subgoal_actions",
        "_pending_subgoal_keyframe",
        "_active_plan_text",
        "_active_plan_steps",
        "_active_subgoal_index",
        "_active_subgoal",
        "_subgoal_sequence_active",
        "_subgoal_sequence_start_state",
        "_subgoal_sequence_start_grid",
        "_subgoal_sequence_expected",
        "_subgoal_sequence_planned_actions",
        "_subgoal_sequence_taken_actions",
        "_subgoal_no_change_streak",
        "_subgoal_sequence_is_explore",
        "_subgoal_sequence_level",
        "_pending_soft_reset",
        "_in_soft_reset",
        "_plan_attempt_active",
        "_plan_attempt_start_state",
        "_plan_attempt_start_grid",
        "_last_boundary_diagnosis_level",
        "_repeat_state_streak",
        "_last_state_signature",
        "_state_visit_counts",
        "_state_action_attempt_counts",
        "_action_attempt_counts",
        "_mode_boundary_counts",
        "_mode_switch_cooldown_remaining",
    )

    def __init__(self, agent: "LoopAgent") -> None:
        self.agent = agent

    def step_decide(
        self,
        *,
        frames: list[FrameData],
        latest_frame: FrameData,
        state_before: str | None = None,
        grid_before: list[list[int]] | None = None,
    ) -> GameAction:
        """Delegate one action decision to LoopAgent behavior."""
        del state_before, grid_before
        return self.agent.choose_action(frames, latest_frame)

    def step_observe(
        self,
        *,
        state_before: str,
        grid_before: list[list[int]],
        action: GameAction,
        frame_after: FrameData,
    ) -> None:
        """Delegate one post-step update to LoopAgent behavior."""
        self.agent._post_step(state_before, grid_before, action, frame_after)

    def snapshot(self) -> RuntimeSnapshot:
        """Capture runtime state for branch replay."""
        scalar_state: dict[str, Any] = {}
        for field in self.SNAPSHOT_FIELDS:
            if hasattr(self.agent, field):
                scalar_state[field] = copy.deepcopy(getattr(self.agent, field))

        memory_state = copy.deepcopy(self.agent.memory)
        state_encoder_state = {
            "previous_grid": copy.deepcopy(self.agent.state_encoder.previous_grid),
            "step_count": self.agent.state_encoder.step_count,
            "frames_since_keyframe": self.agent.state_encoder.frames_since_keyframe,
            "_force_next_keyframe": self.agent.state_encoder._force_next_keyframe,
            "_forced_reason": copy.deepcopy(self.agent.state_encoder._forced_reason),
            "_prev_objects": copy.deepcopy(self.agent.state_encoder._prev_objects),
            "_next_object_id": self.agent.state_encoder._next_object_id,
        }
        level_controller_state = {
            "current_level": self.agent.level_controller.current_level,
            "surprise_history": copy.deepcopy(self.agent.level_controller.surprise_history),
        }
        solver_state = {
            "current_plan": copy.deepcopy(self.agent.solver.current_plan),
            "current_subgoal": copy.deepcopy(self.agent.solver.current_subgoal),
        }
        learner_state = {
            "consecutive_nones": self.agent.learner.consecutive_nones,
            "last_raw_output": self.agent.learner.last_raw_output,
            "last_answer_output": self.agent.learner.last_answer_output,
        }

        return RuntimeSnapshot(
            scalar_state=scalar_state,
            memory_state=memory_state,
            state_encoder_state=state_encoder_state,
            level_controller_state=level_controller_state,
            solver_state=solver_state,
            learner_state=learner_state,
        )

    def restore(self, snapshot: RuntimeSnapshot) -> None:
        """Restore runtime state from snapshot."""
        for field, value in snapshot.scalar_state.items():
            setattr(self.agent, field, copy.deepcopy(value))

        self.agent.memory = copy.deepcopy(snapshot.memory_state)
        self.agent.state_encoder.previous_grid = copy.deepcopy(
            snapshot.state_encoder_state.get("previous_grid")
        )
        self.agent.state_encoder.step_count = int(
            snapshot.state_encoder_state.get("step_count", 0)
        )
        self.agent.state_encoder.frames_since_keyframe = int(
            snapshot.state_encoder_state.get("frames_since_keyframe", 0)
        )
        self.agent.state_encoder._force_next_keyframe = bool(
            snapshot.state_encoder_state.get("_force_next_keyframe", False)
        )
        self.agent.state_encoder._forced_reason = snapshot.state_encoder_state.get(
            "_forced_reason"
        )
        self.agent.state_encoder._prev_objects = copy.deepcopy(
            snapshot.state_encoder_state.get("_prev_objects", [])
        )
        self.agent.state_encoder._next_object_id = int(
            snapshot.state_encoder_state.get("_next_object_id", 0)
        )

        self.agent.level_controller.current_level = str(
            snapshot.level_controller_state.get("current_level", "action")
        )
        self.agent.level_controller.surprise_history = copy.deepcopy(
            snapshot.level_controller_state.get(
                "surprise_history",
                {"action": [], "subgoal": [], "plan": []},
            )
        )

        self.agent.solver.current_plan = copy.deepcopy(
            snapshot.solver_state.get("current_plan")
        )
        self.agent.solver.current_subgoal = copy.deepcopy(
            snapshot.solver_state.get("current_subgoal")
        )

        self.agent.learner.consecutive_nones = int(
            snapshot.learner_state.get("consecutive_nones", 0)
        )
        self.agent.learner.last_raw_output = str(
            snapshot.learner_state.get("last_raw_output", "")
        )
        self.agent.learner.last_answer_output = str(
            snapshot.learner_state.get("last_answer_output", "")
        )
