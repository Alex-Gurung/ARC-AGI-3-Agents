"""Surprise computation for the LoopAgent.

Strategy pattern supporting multiple surprise metrics:
- LogProbSurprise: uses VLLM logprobs API (default)
- HeuristicSurprise: no LLM, based on state changes

The surprise score is a system-level signal for:
- Phase transitions (explore <-> exploit)
- Level controller (abstraction level switching)
- Training rewards (curiosity/learner objectives)

It is NOT passed to the model in prompts.
"""

import logging
from abc import ABC, abstractmethod
from collections import deque

from openai import OpenAI

from .memory import Memory

logger = logging.getLogger(__name__)


class SurpriseStrategy(ABC):
    """Base class for surprise computation strategies."""

    @abstractmethod
    def compute(
        self,
        state_before: str,
        action: str,
        state_after: str,
        memory: Memory,
        num_changed_cells: int,
    ) -> float:
        """Compute surprise score. Higher = more surprising."""
        raise NotImplementedError


class LogProbSurprise(SurpriseStrategy):
    """Compute surprise via VLLM logprobs scoring pass.

    Uses the completions API with forced completion to measure
    -log p(new_state | state, action, memory).

    Normalizes by number of changed cells and debiases against
    a running mean.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        window_size: int = 20,
    ) -> None:
        self.client = client
        self.model = model
        self.window_size = window_size
        self.history: deque[float] = deque(maxlen=window_size)

    def compute(
        self,
        state_before: str,
        action: str,
        state_after: str,
        memory: Memory,
        num_changed_cells: int,
    ) -> float:
        """Compute surprise using logprob scoring pass."""
        memory_text = memory.to_text() if memory else "empty"

        prompt = (
            f"State: {state_before}\n"
            f"Memory: {memory_text}\n"
            f"Action: {action}\n"
            f"Result: "
        )
        forced_completion = state_after

        try:
            prompt_only = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
            prompt_tokens = 0
            prompt_choice = prompt_only.choices[0]
            if prompt_choice.logprobs and prompt_choice.logprobs.tokens:
                prompt_tokens = len(prompt_choice.logprobs.tokens)

            response = self.client.completions.create(
                model=self.model,
                prompt=prompt + forced_completion,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )

            # Extract logprobs for the forced completion tokens only
            choice = response.choices[0]
            logprobs_data = choice.logprobs

            if logprobs_data is None or logprobs_data.token_logprobs is None:
                logger.warning("No logprobs returned from VLLM")
                return 0.0

            tokens = logprobs_data.tokens or []
            token_logprobs = logprobs_data.token_logprobs or []

            # Slice completion tokens by prompt token count.
            completion_start_idx = min(prompt_tokens, len(tokens))

            # Sum logprobs of completion tokens
            completion_logprobs = [
                lp
                for lp in token_logprobs[completion_start_idx:]
                if lp is not None
            ]

            if not completion_logprobs:
                return 0.0

            raw_surprisal = -sum(completion_logprobs) / len(completion_logprobs)
            normalized = raw_surprisal / max(num_changed_cells, 1)

            # Debias against running mean
            running_mean = self.running_mean
            debiased = normalized - running_mean
            self.history.append(normalized)

            logger.debug(
                f"Surprise: raw={raw_surprisal:.3f}, "
                f"norm={normalized:.3f}, debiased={debiased:.3f}, "
                f"changed_cells={num_changed_cells}"
            )

            return debiased

        except Exception as e:
            logger.error(f"LogProbSurprise failed: {e}")
            return 0.0

    @property
    def running_mean(self) -> float:
        """Current running mean of normalized surprise."""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)


class HeuristicSurprise(SurpriseStrategy):
    """Compute surprise from structural heuristics (no LLM needed).

    Useful as a baseline or fallback when VLLM logprobs aren't available.
    Scores based on:
    - Number of changed cells
    - Whether the state transitioned
    - Whether the score changed
    """

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self.history: deque[float] = deque(maxlen=window_size)

    def compute(
        self,
        state_before: str,
        action: str,
        state_after: str,
        memory: Memory,
        num_changed_cells: int,
    ) -> float:
        """Compute surprise heuristically."""
        score = 0.0

        # More changed cells = more surprising
        if num_changed_cells > 50:
            score += 1.0
        elif num_changed_cells > 20:
            score += 0.7
        elif num_changed_cells > 5:
            score += 0.4
        elif num_changed_cells > 0:
            score += 0.2

        # State transition markers
        if "GAME_OVER" in state_after:
            score += 1.0
        if "WIN" in state_after:
            score += 1.0
        if "LEVELS_COMPLETED" in state_after:
            # Check if level changed
            for line in state_after.split("\n"):
                if "LEVELS_COMPLETED:" in line:
                    for line_before in state_before.split("\n"):
                        if "LEVELS_COMPLETED:" in line_before:
                            if line != line_before:
                                score += 1.5

        # Debias
        running_mean = self.running_mean
        debiased = score - running_mean
        self.history.append(score)

        return debiased

    @property
    def running_mean(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)


class LevelController:
    """Controls which abstraction level (action/subgoal/plan) is active.

    Uses surprise-rate convergence: when the rate of change in surprise
    flattens (<5% over last N steps), move to the next level.

    Supports targeted regression via diagnosis (plan broke -> investigate at that level).
    """

    LEVELS = ("action", "subgoal", "plan")

    def __init__(self, convergence_threshold: float = 0.05, window_size: int = 5) -> None:
        self.current_level: str = "action"
        self.convergence_threshold = convergence_threshold
        self.window_size = window_size
        self.surprise_history: dict[str, list[float]] = {
            "action": [],
            "subgoal": [],
            "plan": [],
        }

    def update(self, surprise: float) -> str:
        """Update with new surprise score, potentially advance level.

        Returns the current (possibly updated) level.
        """
        history = self.surprise_history[self.current_level]
        history.append(surprise)

        if len(history) >= self.window_size:
            recent = history[-self.window_size:]
            # Rate of change in surprise over the window
            rate_of_change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-6)
            if rate_of_change < self.convergence_threshold:
                next_level = self._next_level(self.current_level)
                if next_level != self.current_level:
                    logger.info(
                        f"Level controller: {self.current_level} -> {next_level} "
                        f"(surprise converged, rate={rate_of_change:.4f})"
                    )
                    self.current_level = next_level

        return self.current_level

    def diagnose_to_level(self, level: str) -> None:
        """Force regression to a specific level (from solver diagnosis)."""
        if level in self.LEVELS:
            logger.info(f"Level controller: diagnosed regression to {level}")
            self.current_level = level
        else:
            logger.warning(f"Invalid level for diagnosis: {level}")

    def regress_one_level(self) -> str:
        """Regress one abstraction level (plan->subgoal->action)."""
        previous = self._prev_level(self.current_level)
        if previous != self.current_level:
            logger.info(f"Level controller: {self.current_level} -> {previous} (regress)")
            self.current_level = previous
        return self.current_level

    def reset(self) -> None:
        """Reset level controller state."""
        self.current_level = "action"
        self.surprise_history = {"action": [], "subgoal": [], "plan": []}

    def _next_level(self, level: str) -> str:
        return {"action": "subgoal", "subgoal": "plan", "plan": "plan"}[level]

    def _prev_level(self, level: str) -> str:
        return {"action": "action", "subgoal": "action", "plan": "subgoal"}[level]

    @property
    def is_at_plan_level(self) -> bool:
        return self.current_level == "plan"
