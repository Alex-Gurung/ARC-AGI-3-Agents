"""Shared boundary reward channels and scalarization helpers.

Use this module across training backends so each boundary can emit:
1) a multi-channel reward vector
2) a role-gated scalar used by the optimizer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardComponents:
    """Dense reward channels logged per boundary."""

    # Shared / curiosity channels
    surprise: float = 0.0
    novelty: float = 0.0
    transition_magnitude: float = 0.0
    boundary_progress: float = 0.0
    unknown_action_bonus: float = 0.0
    loop_break_bonus: float = 0.0
    repeat_penalty: float = 0.0

    # Learner channels
    mismatch_reduction: float = 0.0
    parse_bonus: float = 0.0
    dedup_bonus: float = 0.0
    contradiction_cleanup_bonus: float = 0.0
    missing_action_reduction_bonus: float = 0.0
    parse_warning_penalty: float = 0.0

    # Solver channels
    level_delta_reward: float = 0.0
    win_bonus: float = 0.0
    game_over_penalty: float = 0.0
    step_penalty: float = 0.0
    solver_surprise_bonus: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class RewardWeights:
    """Default weights aligned with current repository behavior."""

    # Curiosity
    curiosity_w_surprise: float = 1.0
    curiosity_w_novelty: float = 0.2
    curiosity_w_transition: float = 0.01
    curiosity_w_subgoal_progress: float = 1.0
    curiosity_w_plan_progress: float = 2.0
    curiosity_w_unknown_action: float = 1.0
    curiosity_w_loop_break: float = 0.5
    curiosity_w_repeat_penalty: float = 0.5
    curiosity_w_parse_warning_penalty: float = 0.25

    # Learner
    learner_w_mismatch_reduction: float = 1.0
    learner_w_parse_bonus: float = 0.1
    learner_w_dedup_bonus: float = 0.1
    learner_w_contradiction_cleanup_bonus: float = 0.1
    learner_w_missing_action_reduction_bonus: float = 0.25
    learner_w_parse_warning_penalty: float = 0.1

    # Solver
    solver_w_level_delta: float = 10.0
    solver_w_win_bonus: float = 50.0
    solver_w_game_over_penalty: float = 5.0
    solver_w_step_penalty: float = 0.05
    solver_w_surprise_bonus: float = 0.5


def scalarize_curiosity(
    components: RewardComponents,
    *,
    boundary_type: str,
    weights: RewardWeights | None = None,
) -> float:
    w = weights or RewardWeights()
    score = 0.0
    score += w.curiosity_w_surprise * components.surprise
    score += w.curiosity_w_novelty * components.novelty
    score += w.curiosity_w_transition * components.transition_magnitude
    if boundary_type == "subgoal":
        score += w.curiosity_w_subgoal_progress * components.boundary_progress
    elif boundary_type == "plan":
        score += w.curiosity_w_plan_progress * components.boundary_progress
    score += w.curiosity_w_unknown_action * components.unknown_action_bonus
    score += w.curiosity_w_loop_break * components.loop_break_bonus
    score -= w.curiosity_w_repeat_penalty * components.repeat_penalty
    score -= w.curiosity_w_parse_warning_penalty * components.parse_warning_penalty
    return score


def scalarize_learner(
    components: RewardComponents,
    *,
    weights: RewardWeights | None = None,
) -> float:
    w = weights or RewardWeights()
    score = 0.0
    score += w.learner_w_mismatch_reduction * components.mismatch_reduction
    score += w.learner_w_parse_bonus * components.parse_bonus
    score += w.learner_w_dedup_bonus * components.dedup_bonus
    score += (
        w.learner_w_contradiction_cleanup_bonus
        * components.contradiction_cleanup_bonus
    )
    score += (
        w.learner_w_missing_action_reduction_bonus
        * components.missing_action_reduction_bonus
    )
    score -= w.learner_w_parse_warning_penalty * components.parse_warning_penalty
    return score


def scalarize_solver(
    components: RewardComponents,
    *,
    weights: RewardWeights | None = None,
) -> float:
    w = weights or RewardWeights()
    score = 0.0
    score += w.solver_w_level_delta * components.level_delta_reward
    score += w.solver_w_win_bonus * components.win_bonus
    score -= w.solver_w_game_over_penalty * components.game_over_penalty
    score -= w.solver_w_step_penalty * components.step_penalty
    score += w.solver_w_surprise_bonus * components.solver_surprise_bonus
    return score


def group_z_normalize(values: list[float], eps: float = 1e-6) -> list[float]:
    """Standard score normalization used for grouped advantages."""
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = (var + eps) ** 0.5
    return [(v - mean) / std for v in values]
