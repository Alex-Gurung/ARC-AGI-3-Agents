"""Grouped candidate evaluation helpers for veRL-style rollouts."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

BoundaryType = str


@dataclass
class CandidateSample:
    """One sampled candidate at a decision boundary."""

    candidate_id: str
    output: str
    reward: float
    metadata: dict[str, Any] = field(default_factory=dict)
    advantage: float = 0.0
    probability: float = 0.0


@dataclass
class GroupedBranchingConfig:
    """Configuration for grouped softmax branching."""

    k_curiosity: int = int(os.environ.get("K_CURIOSITY", "4"))
    k_learner: int = int(os.environ.get("K_LEARNER", "4"))
    k_solver: int = int(os.environ.get("K_SOLVER", "4"))
    temperature: float = 1.0
    epsilon: float = 1e-8

    def count_for_module(self, module: str) -> int:
        key = module.lower()
        if key == "curiosity":
            return max(1, self.k_curiosity)
        if key == "learner":
            return max(1, self.k_learner)
        if key == "solver":
            return max(1, self.k_solver)
        return 1


class GroupedBranchingEngine:
    """Compute group-normalized advantages and select one canonical branch."""

    def __init__(
        self,
        config: GroupedBranchingConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or GroupedBranchingConfig()
        self.rng = random.Random(seed)

    def evaluate_group(
        self,
        *,
        module: str,
        sample_fn: Callable[[int], dict[str, Any]],
        reward_fn: Callable[[dict[str, Any]], float],
    ) -> list[CandidateSample]:
        """Sample candidates for one module and attach reward/adv/prob."""
        n = self.config.count_for_module(module)
        out: list[CandidateSample] = []
        for idx in range(n):
            payload = sample_fn(idx)
            reward = reward_fn(payload)
            out.append(
                CandidateSample(
                    candidate_id=f"{module}_{idx}",
                    output=str(payload.get("output", "")),
                    reward=float(reward),
                    metadata=payload,
                )
            )
        self._attach_advantages(out)
        return out

    def select_index(self, candidates: list[CandidateSample]) -> int:
        """Sample a candidate index from softmax over advantages."""
        if not candidates:
            return 0
        weights = [max(0.0, c.probability) for c in candidates]
        if not any(weights):
            return 0
        return self.rng.choices(range(len(candidates)), weights=weights, k=1)[0]

    def commit_selected(
        self,
        *,
        candidates: list[CandidateSample],
        selected_index: int,
        commit_fn: Callable[[CandidateSample], None],
        log_fn: Callable[[CandidateSample], None] | None = None,
    ) -> None:
        """Apply only selected branch to canonical state; log others."""
        for idx, candidate in enumerate(candidates):
            if idx == selected_index:
                commit_fn(candidate)
            elif log_fn is not None:
                log_fn(candidate)

    def _attach_advantages(self, candidates: list[CandidateSample]) -> None:
        if not candidates:
            return
        rewards = [c.reward for c in candidates]
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(variance + self.config.epsilon)
        advantages = [(r - mean) / std for r in rewards]
        probs = self._softmax(advantages)
        for idx, candidate in enumerate(candidates):
            candidate.advantage = advantages[idx]
            candidate.probability = probs[idx]

    def _softmax(self, values: list[float]) -> list[float]:
        if not values:
            return []
        temperature = max(self.config.temperature, self.config.epsilon)
        scaled = [v / temperature for v in values]
        max_val = max(scaled)
        exps = [math.exp(v - max_val) for v in scaled]
        total = sum(exps)
        if total <= self.config.epsilon:
            return [1.0 / len(values)] * len(values)
        return [e / total for e in exps]
