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
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from openai import OpenAI

from .memory import Memory

logger = logging.getLogger(__name__)


@dataclass
class SurpriseBundle:
    """Structured surprise outputs for semantic/debiased scoring."""

    debiased_nll: float
    self_rated_x10: float
    reward_value: float
    mean_nll_full: float = 0.0
    mean_nll_stripped: float = 0.0


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


class SemanticDebiasedNLLSurprise(SurpriseStrategy):
    """Semantic surprise with debiased mean NLL + optional self-rated source."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        reward_source: str | None = None,
        self_rated_weight: float | None = None,
        window_size: int = 100,
    ) -> None:
        self.client = client
        self.model = model
        self.reward_source = (
            reward_source
            or os.environ.get("SURPRISE_REWARD_SOURCE", "debiased_nll")
        ).strip().lower()
        self.self_rated_weight = float(
            self_rated_weight
            if self_rated_weight is not None
            else os.environ.get("SELF_RATED_SURPRISE_WEIGHT", "0.25")
        )
        self.debiased_history: deque[float] = deque(maxlen=window_size)
        self.self_rated_history: deque[float] = deque(maxlen=window_size)
        self.last_bundle: SurpriseBundle = SurpriseBundle(
            debiased_nll=0.0,
            self_rated_x10=0.0,
            reward_value=0.0,
        )

    def compute(
        self,
        state_before: str,
        action: str,
        state_after: str,
        memory: Memory,
        num_changed_cells: int,
    ) -> float:
        """Compatibility API: treat state_after as semantic report fallback."""
        del num_changed_cells
        bundle = self.compute_bundle(
            state_before=state_before,
            action=action,
            memory=memory,
            semantic_report=state_after,
            self_rated_x10=0.0,
        )
        return bundle.reward_value

    def compute_bundle(
        self,
        *,
        state_before: str,
        action: str,
        memory: Memory,
        semantic_report: str,
        self_rated_x10: float | None = None,
        visual_context_hint: str = "",
    ) -> SurpriseBundle:
        """Compute semantic surprise bundle.

        debiased_nll = mean_nll_full - mean_nll_stripped
        """
        memory_text = memory.to_text() if memory else "empty"
        report = semantic_report.strip() or "No meaningful transition was observed."
        full_prompt = (
            f"BEFORE_STATE:\n{state_before}\n\n"
            f"ACTION:\n{action}\n\n"
            f"MEMORY:\n{memory_text}\n\n"
            f"VISUAL_HINT:\n{visual_context_hint or 'none'}\n\n"
            "TRANSITION_REPORT:\n"
        )
        stripped_prompt = (
            f"BEFORE_STATE:\n{state_before}\n\n"
            f"MEMORY:\n{memory_text}\n\n"
            f"VISUAL_HINT:\n{visual_context_hint or 'none'}\n\n"
            "TRANSITION_REPORT:\n"
        )

        mean_nll_full = self._mean_nll_forced_completion(
            prompt=full_prompt,
            forced_completion=report,
        )
        mean_nll_stripped = self._mean_nll_forced_completion(
            prompt=stripped_prompt,
            forced_completion=report,
        )
        debiased = mean_nll_full - mean_nll_stripped
        self_rated = max(0.0, min(10.0, float(self_rated_x10 or 0.0)))

        z_debiased = self._zscore(debiased, self.debiased_history)
        z_self_rated = self._zscore(self_rated, self.self_rated_history)
        self.debiased_history.append(debiased)
        self.self_rated_history.append(self_rated)

        reward_source = self.reward_source
        if reward_source == "self_rated":
            reward_value = z_self_rated
        elif reward_source == "hybrid":
            w = max(0.0, min(1.0, self.self_rated_weight))
            reward_value = (1.0 - w) * z_debiased + w * z_self_rated
        else:
            reward_value = z_debiased

        bundle = SurpriseBundle(
            debiased_nll=debiased,
            self_rated_x10=self_rated,
            reward_value=reward_value,
            mean_nll_full=mean_nll_full,
            mean_nll_stripped=mean_nll_stripped,
        )
        self.last_bundle = bundle
        return bundle

    def _mean_nll_forced_completion(self, *, prompt: str, forced_completion: str) -> float:
        """Return per-token mean NLL for forced completion using completions API."""
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
            choice = response.choices[0]
            logprobs_data = choice.logprobs
            if logprobs_data is None or logprobs_data.token_logprobs is None:
                return 0.0
            tokens = logprobs_data.tokens or []
            token_logprobs = logprobs_data.token_logprobs or []
            completion_start_idx = min(prompt_tokens, len(tokens))
            completion_logprobs = [
                lp for lp in token_logprobs[completion_start_idx:] if lp is not None
            ]
            if not completion_logprobs:
                return 0.0
            return -sum(completion_logprobs) / len(completion_logprobs)
        except Exception as e:
            logger.error("SemanticDebiasedNLLSurprise scoring failed: %s", e)
            return 0.0

    @staticmethod
    def _zscore(value: float, history: deque[float]) -> float:
        if not history:
            return 0.0
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / max(len(history), 1)
        std = variance**0.5
        if std <= 1e-8:
            return 0.0
        return (value - mean) / std


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
