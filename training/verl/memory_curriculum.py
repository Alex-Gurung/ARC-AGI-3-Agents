"""Memory initialization curriculum for training rollouts."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass

from agents.templates.loop_agent.memory import Memory


@dataclass
class MemoryCurriculumConfig:
    carry_p: float = float(os.environ.get("MEMORY_INIT_CARRY_P", "0.60"))
    noisy_p: float = float(os.environ.get("MEMORY_INIT_NOISY_P", "0.25"))
    blank_p: float = float(os.environ.get("MEMORY_INIT_BLANK_P", "0.15"))
    noisy_delete_fraction: float = float(os.environ.get("NOISY_DELETE_FRACTION", "0.2"))
    noisy_conf_jitter: float = float(os.environ.get("NOISY_CONF_JITTER", "0.1"))

    def normalized(self) -> tuple[float, float, float]:
        values = [max(0.0, self.carry_p), max(0.0, self.noisy_p), max(0.0, self.blank_p)]
        total = sum(values)
        if total <= 1e-8:
            return 1.0, 0.0, 0.0
        return values[0] / total, values[1] / total, values[2] / total


class MemoryCurriculum:
    """Sample training memory initialization modes (carry/noisy/blank)."""

    def __init__(self, config: MemoryCurriculumConfig | None = None, seed: int = 0) -> None:
        self.config = config or MemoryCurriculumConfig()
        self.rng = random.Random(seed)

    def sample_mode(self) -> str:
        carry_p, noisy_p, blank_p = self.config.normalized()
        modes = ["carry", "noisy", "blank"]
        probs = [carry_p, noisy_p, blank_p]
        return self.rng.choices(modes, weights=probs, k=1)[0]

    def initialize(
        self,
        *,
        base_memory: Memory | None,
        max_entries: int = 50,
        forced_mode: str | None = None,
    ) -> tuple[Memory, str]:
        """Return initialized memory and chosen mode."""
        mode = (forced_mode or self.sample_mode()).lower()
        if mode == "blank" or base_memory is None:
            return Memory(max_entries=max_entries), mode

        memory = copy.deepcopy(base_memory)
        memory.MAX_ENTRIES = max_entries
        if mode == "noisy":
            memory.perturb(
                delete_fraction=self.config.noisy_delete_fraction,
                confidence_jitter=self.config.noisy_conf_jitter,
                shuffle_entries=True,
            )
        return memory, mode
