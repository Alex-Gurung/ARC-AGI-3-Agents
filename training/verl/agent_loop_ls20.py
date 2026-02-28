"""Grouped rollout collector for veRL-style LoopAgent training on ls20.

This module is intentionally dependency-light: it does not require veRL to be
importable for local dry runs, but emits grouped decision trajectories that can
be consumed by downstream RL tooling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arc_agi import Arcade
from arcengine import FrameData, GameAction, GameState

from agents.templates.loop_agent import DecisionMode, LoopAgent
from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory
from agents.templates.loop_agent.surprise import SemanticDebiasedNLLSurprise
from training.verl.grouped_branching import (
    CandidateSample,
    GroupedBranchingConfig,
    GroupedBranchingEngine,
)
from training.verl.memory_curriculum import MemoryCurriculum
from training.verl.rewards import curiosity_reward
from training.verl.trajectory_schema import (
    CandidateDecision,
    DecisionRecord,
    EpisodeRecord,
    SurpriseMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class SerializedAction:
    """Serializable action event used for prefix replay."""

    name: str
    data: dict[str, Any]
    reasoning: Any


@dataclass
class CandidateEvalContext:
    """Replica result bundle for one candidate."""

    sample: CandidateSample
    agent: LoopAgent
    frame_after: FrameData
    action_event: SerializedAction
    semantic_report: str
    surprise_metrics: SurpriseMetrics
    changed_cells: int
    level_delta: int


class LS20GroupedRunner:
    """Run grouped-candidate training attempts on ls20."""

    def __init__(
        self,
        *,
        model: str,
        game_id: str = "ls20",
        max_steps: int = 200,
        seed: int = 0,
        grouped_config: GroupedBranchingConfig | None = None,
    ) -> None:
        self.model = model
        self.game_id = game_id
        self.max_steps = max_steps
        self.branching = GroupedBranchingEngine(grouped_config, seed=seed)
        self.memory_curriculum = MemoryCurriculum(seed=seed)
        self.surprise_reward_source = os.environ.get(
            "SURPRISE_REWARD_SOURCE", "debiased_nll"
        ).strip().lower()
        self.self_rated_weight = float(
            os.environ.get("SELF_RATED_SURPRISE_WEIGHT", "0.25")
        )

    def run_attempt(
        self,
        *,
        attempt_id: str,
        output_jsonl: Path,
        base_memory: Memory | None = None,
    ) -> EpisodeRecord:
        """Run one grouped rollout attempt and append JSONL decisions."""
        canonical_actions: list[SerializedAction] = []
        initial_memory, memory_mode = self.memory_curriculum.initialize(
            base_memory=base_memory,
            max_entries=50,
        )
        canonical = self._new_agent_with_replay(
            replay_actions=[],
            memory_override=initial_memory,
        )
        decision_records: list[DecisionRecord] = []
        running_return = 0.0

        for step in range(self.max_steps):
            latest_frame = canonical._convert_raw_frame_data(
                canonical.arc_env.observation_space if canonical.arc_env else None
            )
            if latest_frame.state in (GameState.WIN, GameState.GAME_OVER):
                break

            module = "solver" if canonical.current_mode == DecisionMode.SOLVE else "curiosity"
            k = self.branching.config.count_for_module(module)
            candidates: list[CandidateEvalContext] = []

            for idx in range(k):
                replica = self._new_agent_with_replay(
                    replay_actions=canonical_actions,
                    memory_override=canonical.memory,
                )
                candidate = self._evaluate_one_step_candidate(
                    agent=replica,
                    candidate_id=f"{module}_{idx}",
                )
                candidates.append(candidate)

            branch_samples = [c.sample for c in candidates]
            self.branching._attach_advantages(branch_samples)
            selected_index = self.branching.select_index(branch_samples)
            selected = candidates[selected_index]
            canonical = selected.agent
            canonical_actions.append(selected.action_event)
            running_return += selected.sample.reward

            record = self._build_decision_record(
                attempt_id=attempt_id,
                step=step,
                module=module,
                selected_index=selected_index,
                candidates=candidates,
                selected=selected,
            )
            decision_records.append(record)
            self._append_jsonl(output_jsonl, record.to_dict())

            if selected.frame_after.state in (GameState.WIN, GameState.GAME_OVER):
                break

        final_frame = canonical.frames[-1] if canonical.frames else FrameData()
        episode = EpisodeRecord(
            episode_id=f"{attempt_id}_episode",
            game_id=self.game_id,
            total_steps=len(canonical_actions),
            done=final_frame.state in (GameState.WIN, GameState.GAME_OVER),
            game_state=final_frame.state.name,
            levels_completed=final_frame.levels_completed,
            running_return=running_return,
            memory_size=len(canonical.memory),
            notes={
                "decision_count": len(decision_records),
                "memory_mode": memory_mode,
            },
        )
        self._append_jsonl(output_jsonl, {"episode_summary": episode.to_dict()})
        return episode

    def _new_agent_with_replay(
        self,
        *,
        replay_actions: list[SerializedAction],
        memory_override: Memory | None,
    ) -> LoopAgent:
        arcade = Arcade()
        card_id = arcade.open_scorecard(tags=["verl", self.game_id])
        env = arcade.make(self.game_id, scorecard_id=card_id)
        agent = LoopAgent(
            card_id=card_id,
            game_id=self.game_id,
            agent_name="loop_agent_verl",
            ROOT_URL="",
            record=False,
            arc_env=env,
            tags=["verl"],
        )
        if memory_override is not None:
            agent.memory = self._clone_memory(memory_override)

        latest_frame = agent._convert_raw_frame_data(
            agent.arc_env.observation_space if agent.arc_env else None
        )
        for event in replay_actions:
            state_before = agent.state_encoder.encode(latest_frame)
            grid_before = latest_frame.frame[-1] if latest_frame.frame else []
            action = self._deserialize_action(event)
            frame_after = agent.take_action(action)
            if frame_after is None:
                break
            agent.append_frame(frame_after)
            agent.runtime.step_observe(
                state_before=state_before,
                grid_before=grid_before,
                action=action,
                frame_after=frame_after,
            )
            agent.action_counter += 1
            latest_frame = frame_after
        return agent

    def _evaluate_one_step_candidate(
        self,
        *,
        agent: LoopAgent,
        candidate_id: str,
    ) -> CandidateEvalContext:
        latest_frame = agent._convert_raw_frame_data(
            agent.arc_env.observation_space if agent.arc_env else None
        )
        state_before = agent.state_encoder.encode(latest_frame)
        grid_before = latest_frame.frame[-1] if latest_frame.frame else []
        image_before = agent._grid_image_data_url(grid_before)

        action = agent.runtime.step_decide(
            frames=agent.frames,
            latest_frame=latest_frame,
            state_before=state_before,
            grid_before=grid_before,
        )
        frame_after = agent.take_action(action)
        if frame_after is None:
            # invalid step fallback
            frame_after = latest_frame
        else:
            agent.append_frame(frame_after)

        agent.runtime.step_observe(
            state_before=state_before,
            grid_before=grid_before,
            action=action,
            frame_after=frame_after,
        )
        agent.action_counter += 1

        grid_after = frame_after.frame[-1] if frame_after.frame else []
        state_after = agent.state_encoder.encode(frame_after)
        diff_text = agent.state_encoder.get_diff_text(grid_before, grid_after)
        changed_cells = agent.state_encoder.get_num_changed_cells(grid_before, grid_after)
        image_after = agent._grid_image_data_url(grid_after)
        image_diff = agent._transition_image_data_url(grid_before, grid_after)

        learner: Learner = agent.learner
        semantic_report = learner.generate_semantic_transition_report(
            state_before=state_before,
            action_taken=action.name,
            state_after=state_after,
            diff_text=diff_text,
            memory=agent.memory,
            image_before_url=image_before,
            image_after_url=image_after,
            image_diff_url=image_diff,
        )
        self_rated = learner.self_rate_surprise(
            state_before=state_before,
            action_taken=action.name,
            state_after=state_after,
            diff_text=diff_text,
            memory=agent.memory,
            mode=agent.current_mode.value,
            image_before_url=image_before,
            image_after_url=image_after,
            image_diff_url=image_diff,
        )
        surprise_engine = SemanticDebiasedNLLSurprise(
            client=agent.client,
            model=self.model,
            reward_source=self.surprise_reward_source,
            self_rated_weight=self.self_rated_weight,
        )
        bundle = surprise_engine.compute_bundle(
            state_before=state_before,
            action=action.name,
            memory=agent.memory,
            semantic_report=semantic_report,
            self_rated_x10=self_rated,
            visual_context_hint="BEFORE/AFTER/DIFF images provided in observer stage",
        )
        if self.surprise_reward_source == "self_rated":
            surprise_signal = bundle.self_rated_x10 / 10.0
        elif self.surprise_reward_source == "hybrid":
            w = max(0.0, min(1.0, self.self_rated_weight))
            surprise_signal = (1.0 - w) * bundle.debiased_nll + w * (
                bundle.self_rated_x10 / 10.0
            )
        else:
            surprise_signal = bundle.debiased_nll
        level_delta = frame_after.levels_completed - latest_frame.levels_completed
        reward = curiosity_reward(
            surprise_reward=surprise_signal,
            boundary_type="action",
            novelty_bonus=1.0 if changed_cells > 0 else 0.0,
            transition_magnitude_bonus=float(changed_cells),
            boundary_progress_bonus=float(level_delta),
        )
        sample = CandidateSample(
            candidate_id=candidate_id,
            output=action.name,
            reward=reward,
            metadata={
                "action_name": action.name,
                "state_before_hash": self._hash_text(state_before),
                "state_after_hash": self._hash_text(state_after),
                "changed_cells": changed_cells,
                "level_delta": level_delta,
                "mode": agent.current_mode.value,
                "surprise_signal": surprise_signal,
            },
        )

        surprise_metrics = SurpriseMetrics(
            debiased_nll=bundle.debiased_nll,
            self_rated_x10=bundle.self_rated_x10,
            reward_value=bundle.reward_value,
            source=surprise_engine.reward_source,
            mean_nll_full=bundle.mean_nll_full,
            mean_nll_stripped=bundle.mean_nll_stripped,
        )
        return CandidateEvalContext(
            sample=sample,
            agent=agent,
            frame_after=frame_after,
            action_event=self._serialize_action(action),
            semantic_report=semantic_report,
            surprise_metrics=surprise_metrics,
            changed_cells=changed_cells,
            level_delta=level_delta,
        )

    def _build_decision_record(
        self,
        *,
        attempt_id: str,
        step: int,
        module: str,
        selected_index: int,
        candidates: list[CandidateEvalContext],
        selected: CandidateEvalContext,
    ) -> DecisionRecord:
        candidate_rows: list[CandidateDecision] = []
        for c in candidates:
            candidate_rows.append(
                CandidateDecision(
                    candidate_id=c.sample.candidate_id,
                    output=c.sample.output,
                    reward=c.sample.reward,
                    advantage=c.sample.advantage,
                    probability=c.sample.probability,
                    surprise=c.surprise_metrics,
                    metadata=c.sample.metadata,
                )
            )
        mode = selected.sample.metadata.get("mode", "LEARN_ACTION")
        state_before_hash = selected.sample.metadata.get("state_before_hash", "unknown")
        state_after_hash = selected.sample.metadata.get("state_after_hash", "unknown")
        return DecisionRecord(
            episode_id=f"{attempt_id}_episode",
            attempt_id=attempt_id,
            step=step,
            boundary_type="action",
            mode=str(mode),
            module=module,
            prompt_fingerprint=f"{module}:{state_before_hash}",
            state_before=state_before_hash,
            state_after=state_after_hash,
            memory_before=f"size={len(selected.agent.memory)}",
            memory_after=f"size={len(selected.agent.memory)}",
            semantic_report=selected.semantic_report,
            selected_index=selected_index,
            candidates=candidate_rows,
            diagnostics={
                "changed_cells": selected.changed_cells,
                "level_delta": selected.level_delta,
                "game_state": selected.frame_after.state.name,
            },
        )

    @staticmethod
    def _serialize_action(action: GameAction) -> SerializedAction:
        return SerializedAction(
            name=action.name,
            data=action.action_data.model_dump(),
            reasoning=action.reasoning,
        )

    @staticmethod
    def _deserialize_action(event: SerializedAction) -> GameAction:
        action = GameAction.from_name(event.name)
        data = dict(event.data)
        action.set_data(data)
        if event.reasoning is not None:
            action.reasoning = event.reasoning
        return action

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    @staticmethod
    def _clone_memory(memory: Memory) -> Memory:
        copied = Memory(max_entries=memory.MAX_ENTRIES)
        copied.entries = [
            type(entry)(
                type=entry.type,
                content=entry.content,
                justification=entry.justification,
                confidence=entry.confidence,
                created_step=entry.created_step,
                last_modified_step=entry.last_modified_step,
                memory_id=entry.memory_id,
            )
            for entry in memory.entries
        ]
        copied._next_id = memory._next_id
        return copied

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", type=str, default="ls20")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--attempt-id", type=str, default="attempt_0001")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("training/verl/data/ls20_grouped_rollouts.jsonl"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    runner = LS20GroupedRunner(
        model=args.model,
        game_id=args.game_id,
        max_steps=args.max_steps,
    )
    episode = runner.run_attempt(
        attempt_id=args.attempt_id,
        output_jsonl=args.output_jsonl,
    )
    print(json.dumps({"episode": asdict(episode)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
