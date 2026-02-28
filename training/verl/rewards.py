"""Reward definitions for veRL grouped LoopAgent training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CuriosityRewardConfig:
    novelty_weight: float = 0.2
    transition_weight: float = 0.01
    subgoal_progress_weight: float = 1.0
    plan_progress_weight: float = 2.0


@dataclass
class LearnerRewardConfig:
    parse_bonus: float = 0.1
    dedup_bonus: float = 0.1
    contradiction_cleanup_bonus: float = 0.1


@dataclass
class SolverRewardConfig:
    level_delta_weight: float = 10.0
    win_bonus: float = 50.0
    game_over_penalty: float = 5.0
    step_penalty: float = 0.05


def curiosity_reward(
    *,
    surprise_reward: float,
    boundary_type: str,
    novelty_bonus: float = 0.0,
    transition_magnitude_bonus: float = 0.0,
    boundary_progress_bonus: float = 0.0,
    cfg: CuriosityRewardConfig | None = None,
) -> float:
    """Curiosity reward with small shaping by boundary type."""
    config = cfg or CuriosityRewardConfig()
    boundary = boundary_type.lower()
    if boundary == "action":
        return (
            surprise_reward
            + config.novelty_weight * novelty_bonus
            + config.transition_weight * transition_magnitude_bonus
        )
    if boundary == "subgoal":
        return surprise_reward + config.subgoal_progress_weight * boundary_progress_bonus
    if boundary == "plan":
        return surprise_reward + config.plan_progress_weight * boundary_progress_bonus
    return surprise_reward


def learner_reward(
    *,
    debiased_before: float,
    debiased_after: float,
    parse_ok: bool = True,
    non_duplicate: bool = True,
    contradiction_cleanup: bool = False,
    cfg: LearnerRewardConfig | None = None,
) -> float:
    """Learner reward prioritizing reduction in transition unpredictability."""
    config = cfg or LearnerRewardConfig()
    score = debiased_before - debiased_after
    if parse_ok:
        score += config.parse_bonus
    if non_duplicate:
        score += config.dedup_bonus
    if contradiction_cleanup:
        score += config.contradiction_cleanup_bonus
    return score


def solver_reward(
    *,
    level_delta: int,
    is_win: bool,
    is_game_over: bool,
    steps_used: int,
    cfg: SolverRewardConfig | None = None,
) -> float:
    """Solver reward favoring progress and efficiency."""
    config = cfg or SolverRewardConfig()
    reward = config.level_delta_weight * float(level_delta)
    if is_win:
        reward += config.win_bonus
    if is_game_over:
        reward -= config.game_over_penalty
    reward -= config.step_penalty * max(0, steps_used)
    return reward
