"""Trajectory schemas for veRL grouped boundary rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SurpriseMetrics:
    """Per-boundary surprise metrics for analysis and reward routing."""

    debiased_nll: float
    self_rated_x10: float
    reward_value: float
    source: str
    mean_nll_full: float = 0.0
    mean_nll_stripped: float = 0.0


@dataclass
class CandidateDecision:
    """One candidate output in grouped sampling."""

    candidate_id: str
    output: str
    reward: float
    advantage: float
    probability: float
    reward_components: dict[str, float] = field(default_factory=dict)
    surprise: SurpriseMetrics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    """One grouped decision event in an episode."""

    episode_id: str
    attempt_id: str
    step: int
    boundary_type: str
    mode: str
    module: str
    prompt_fingerprint: str
    state_before: str
    state_after: str
    memory_before: str
    memory_after: str
    semantic_report: str
    selected_index: int
    reward_channel_version: str = "v2"
    candidates: list[CandidateDecision] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeRecord:
    """Compact per-episode summary."""

    episode_id: str
    game_id: str
    total_steps: int
    done: bool
    game_state: str
    levels_completed: int
    running_return: float
    memory_size: int
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
