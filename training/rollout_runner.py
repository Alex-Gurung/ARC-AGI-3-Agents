"""Training-only grouped rollout runner.

This module is intentionally separate from online inference logic.
It supports grouped candidate sampling and softmax selection while keeping
one canonical trajectory per attempt.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .group_sampler import GroupSampler, GroupSamplingConfig
from .trajectory_schema import CandidateEvaluation, GroupDecisionRecord


@dataclass
class ModuleCandidateConfig:
    """Candidate count per module for grouped sampling."""

    solver: int = 4
    curiosity: int = 3
    learner: int = 2


@dataclass
class PrefixAction:
    """One action in a canonical prefix replay."""

    action: Any
    data: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None


class EnvironmentLike(Protocol):
    """Minimal protocol for ARC wrappers used during replica rollouts."""

    def reset(self) -> Any: ...

    def step(
        self,
        action: Any,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> Any: ...


class TrainingRolloutRunner:
    """Coordinates grouped candidate generation and canonical selection."""

    def __init__(
        self,
        candidate_config: ModuleCandidateConfig | None = None,
        sampler_config: GroupSamplingConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.candidate_config = candidate_config or ModuleCandidateConfig()
        self.group_sampler = GroupSampler(sampler_config)
        self.rng = random.Random(seed)

    def module_candidate_count(self, module: str) -> int:
        """Return configured candidate count for a module."""
        if module == "solver":
            return self.candidate_config.solver
        if module == "curiosity":
            return self.candidate_config.curiosity
        if module == "learner":
            return self.candidate_config.learner
        return 1

    def build_group(
        self,
        *,
        module: str,
        candidate_fn: Callable[[int], dict[str, Any]],
        reward_fn: Callable[[dict[str, Any]], float],
    ) -> list[CandidateEvaluation]:
        """Sample and score candidate outputs for one grouped decision."""
        out: list[CandidateEvaluation] = []
        n = self.module_candidate_count(module)
        for idx in range(n):
            sample = candidate_fn(idx)
            reward = reward_fn(sample)
            out.append(
                CandidateEvaluation(
                    candidate_id=f"{module}_{idx}",
                    output=sample.get("output", ""),
                    reward=reward,
                    metadata=sample,
                )
            )
        rewards = [c.reward for c in out]
        advantages = self.group_sampler.advantages(rewards)
        probabilities = self.group_sampler.probabilities(advantages)
        for i, candidate in enumerate(out):
            candidate.advantage = advantages[i]
            candidate.probability = probabilities[i]
        return out

    def select_candidate(self, candidates: list[CandidateEvaluation]) -> int:
        """Softmax-sample selected candidate index."""
        if not candidates:
            return 0
        rewards = [c.reward for c in candidates]
        return self.group_sampler.sample_index(rewards, rng=self.rng)

    def decision_record(
        self,
        *,
        attempt_id: str,
        step: int,
        phase: str,
        mode: str,
        prompt: str,
        state_before: str,
        state_after: str,
        memory_before: str,
        memory_after: str,
        candidates: list[CandidateEvaluation],
        selected_index: int,
        diagnostics: dict[str, Any] | None = None,
    ) -> GroupDecisionRecord:
        """Build a trajectory record for one grouped decision."""
        return GroupDecisionRecord(
            attempt_id=attempt_id,
            step=step,
            phase=phase,
            mode=mode,
            prompt=prompt,
            state_before=state_before,
            state_after=state_after,
            memory_before=memory_before,
            memory_after=memory_after,
            selected_index=selected_index,
            candidates=candidates,
            diagnostics=diagnostics or {},
        )

    def commit_selected_candidate(
        self,
        *,
        candidates: list[CandidateEvaluation],
        selected_index: int,
        commit_fn: Callable[[CandidateEvaluation], None],
        log_fn: Callable[[CandidateEvaluation], None] | None = None,
    ) -> None:
        """Apply only the selected candidate to canonical trajectory state.

        Non-selected candidates are optionally logged but never committed.
        """
        for idx, candidate in enumerate(candidates):
            if idx == selected_index:
                commit_fn(candidate)
            elif log_fn is not None:
                log_fn(candidate)

    def replay_prefix(
        self,
        *,
        env_factory: Callable[[], EnvironmentLike],
        prefix_actions: list[PrefixAction],
    ) -> tuple[EnvironmentLike, Any]:
        """Create a replica env and replay canonical prefix actions."""
        env = env_factory()
        latest = env.reset()
        for event in prefix_actions:
            latest = env.step(
                event.action,
                data=event.data or {},
                reasoning=event.reasoning or {},
            )
        return env, latest

    def evaluate_candidates_with_replicas(
        self,
        *,
        module: str,
        env_factory: Callable[[], EnvironmentLike],
        prefix_actions: list[PrefixAction],
        candidate_fn: Callable[[int], dict[str, Any]],
        execute_fn: Callable[[EnvironmentLike, dict[str, Any]], tuple[float, dict[str, Any]]],
    ) -> tuple[list[CandidateEvaluation], int]:
        """Evaluate grouped candidates from identical replayed state replicas.

        Returns:
            candidates: scored candidate evaluations with advantages/probabilities
            selected_index: softmax-sampled candidate index for canonical trajectory
        """
        candidates: list[CandidateEvaluation] = []
        n = self.module_candidate_count(module)
        for idx in range(n):
            env, _ = self.replay_prefix(env_factory=env_factory, prefix_actions=prefix_actions)
            sample = candidate_fn(idx)
            reward, metadata = execute_fn(env, sample)
            candidates.append(
                CandidateEvaluation(
                    candidate_id=f"{module}_{idx}",
                    output=sample.get("output", ""),
                    reward=reward,
                    metadata=metadata,
                )
            )

        rewards = [c.reward for c in candidates]
        advantages = self.group_sampler.advantages(rewards)
        probabilities = self.group_sampler.probabilities(advantages)
        for i, candidate in enumerate(candidates):
            candidate.advantage = advantages[i]
            candidate.probability = probabilities[i]

        selected_index = self.select_candidate(candidates)
        return candidates, selected_index
