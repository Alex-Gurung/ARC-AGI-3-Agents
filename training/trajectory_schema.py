"""Structured trajectory records for grouped training rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CandidateEvaluation:
    """A single candidate sampled for a grouped decision."""

    candidate_id: str
    output: str
    reward: float
    advantage: float = 0.0
    probability: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupDecisionRecord:
    """One grouped sampling decision in a rollout trajectory."""

    attempt_id: str
    step: int
    phase: str
    mode: str
    prompt: str
    state_before: str
    state_after: str
    memory_before: str
    memory_after: str
    selected_index: int
    candidates: list[CandidateEvaluation] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)
