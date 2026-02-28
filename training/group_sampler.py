"""Group-relative sampling helpers used by training rollouts."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class GroupSamplingConfig:
    """Configuration for within-group softmax selection."""

    temperature: float = 1.0
    epsilon: float = 1e-8


class GroupSampler:
    """Computes group-normalized advantages and softmax-selects candidates."""

    def __init__(self, config: GroupSamplingConfig | None = None) -> None:
        self.config = config or GroupSamplingConfig()

    def advantages(self, rewards: list[float]) -> list[float]:
        """Compute zero-mean unit-variance advantages for a reward group."""
        if not rewards:
            return []
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var + self.config.epsilon)
        return [(r - mean) / std for r in rewards]

    def probabilities(self, advantages: list[float]) -> list[float]:
        """Compute softmax probabilities from normalized advantages."""
        if not advantages:
            return []
        temperature = max(self.config.temperature, self.config.epsilon)
        scaled = [a / temperature for a in advantages]
        max_val = max(scaled)
        exps = [math.exp(v - max_val) for v in scaled]
        total = sum(exps)
        if total <= 0:
            return [1.0 / len(advantages)] * len(advantages)
        return [v / total for v in exps]

    def sample_index(self, rewards: list[float], rng: random.Random | None = None) -> int:
        """Sample an index from softmax probabilities over group advantages."""
        if not rewards:
            return 0
        random_gen = rng or random
        adv = self.advantages(rewards)
        probs = self.probabilities(adv)
        return random_gen.choices(range(len(rewards)), weights=probs, k=1)[0]
