"""Mode-local reward shaping utilities for grouped rollouts."""

from __future__ import annotations


def action_reward(
    *,
    surprise: float,
    changed_cells: int,
    level_delta: int = 0,
    terminal_penalty: float = 0.0,
) -> float:
    """Reward immediate action outcomes for curiosity/solver action mode."""
    return (0.5 * surprise) + (0.01 * changed_cells) + (3.0 * level_delta) - terminal_penalty


def subgoal_reward(
    *,
    boundary_surprise: float,
    progress: float,
    learner_changed: bool,
    steps_used: int,
    terminal: bool,
) -> float:
    """Reward full subgoal-sequence outcomes at the boundary."""
    learner_bonus = 0.5 if learner_changed else 0.0
    terminal_penalty = 2.0 if terminal else 0.0
    efficiency_penalty = 0.02 * max(steps_used, 0)
    return (progress + boundary_surprise + learner_bonus) - (efficiency_penalty + terminal_penalty)


def plan_reward(
    *,
    subgoals_completed: int,
    total_subgoals: int,
    solved: bool,
    terminal_failure: bool,
) -> float:
    """Reward plan-attempt outcomes at plan boundary."""
    completion_ratio = 0.0
    if total_subgoals > 0:
        completion_ratio = subgoals_completed / total_subgoals
    solved_bonus = 5.0 if solved else 0.0
    failure_penalty = 2.0 if terminal_failure else 0.0
    return (2.0 * completion_ratio) + solved_bonus - failure_penalty
