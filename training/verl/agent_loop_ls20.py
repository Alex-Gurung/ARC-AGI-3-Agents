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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arc_agi import Arcade
from arcengine import FrameData, GameAction, GameState

from agents.templates.loop_agent import DecisionMode, LoopAgent
from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory
from agents.templates.loop_agent.surprise import SemanticDebiasedNLLSurprise
from training.reward_channels import (
    RewardComponents,
    scalarize_curiosity,
    scalarize_learner,
    scalarize_solver,
)
from training.verl.grouped_branching import (
    CandidateSample,
    GroupedBranchingConfig,
    GroupedBranchingEngine,
)
from training.verl.memory_curriculum import MemoryCurriculum
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
    action_events: list[SerializedAction]
    semantic_report: str
    surprise_metrics: SurpriseMetrics
    changed_cells: int
    level_delta: int
    boundary_type: str
    steps_used: int
    learner_group: list[CandidateSample] = field(default_factory=list)
    learner_selected_index: int = 0


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
        episode, _ = self.run_attempt_with_memory(
            attempt_id=attempt_id,
            output_jsonl=output_jsonl,
            base_memory=base_memory,
        )
        return episode

    def run_attempt_with_memory(
        self,
        *,
        attempt_id: str,
        output_jsonl: Path,
        base_memory: Memory | None = None,
    ) -> tuple[EpisodeRecord, Memory]:
        """Run one grouped rollout attempt and return final canonical memory."""
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
        decision_step = 0
        action_steps_used = 0

        while action_steps_used < self.max_steps:
            latest_frame = canonical._convert_raw_frame_data(
                canonical.arc_env.observation_space if canonical.arc_env else None
            )
            if latest_frame.state in (GameState.WIN, GameState.GAME_OVER):
                break

            module = "solver" if canonical.current_mode == DecisionMode.SOLVE else "curiosity"
            k = self.branching.config.count_for_module(module)
            candidates: list[CandidateEvalContext] = []

            for idx in range(k):
                candidate = self._evaluate_one_step_candidate(
                    module=module,
                    replay_actions=canonical_actions,
                    memory_override=canonical.memory,
                    candidate_id=f"{module}_{idx}",
                )
                candidates.append(candidate)

            branch_samples = [c.sample for c in candidates]
            self.branching._attach_advantages(branch_samples)
            selected_index = self.branching.select_index(branch_samples)
            selected = candidates[selected_index]
            canonical = selected.agent
            if selected.action_events:
                canonical_actions.extend(selected.action_events)
                action_steps_used += len(selected.action_events)
            else:
                # Safety guard against malformed candidates.
                action_steps_used += 1
            running_return += selected.sample.reward

            record = self._build_decision_record(
                attempt_id=attempt_id,
                step=decision_step,
                module=module,
                selected_index=selected_index,
                candidates=candidates,
                selected=selected,
            )
            decision_records.append(record)
            self._append_jsonl(output_jsonl, record.to_dict())
            decision_step += 1

            if selected.frame_after.state in (GameState.WIN, GameState.GAME_OVER):
                break

        final_frame = canonical.frames[-1] if canonical.frames else FrameData()
        episode = EpisodeRecord(
            episode_id=f"{attempt_id}_episode",
            game_id=self.game_id,
            total_steps=action_steps_used,
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
        return episode, self._clone_memory(canonical.memory)

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
        module: str,
        replay_actions: list[SerializedAction],
        memory_override: Memory,
        candidate_id: str,
    ) -> CandidateEvalContext:
        """Evaluate one top-level action candidate + grouped learner boundary."""
        agent = self._new_agent_with_replay(
            replay_actions=replay_actions,
            memory_override=memory_override,
        )
        latest_frame = agent._convert_raw_frame_data(
            agent.arc_env.observation_space if agent.arc_env else None
        )
        state_before = agent.state_encoder.encode(latest_frame)
        _ = latest_frame.frame[-1] if latest_frame.frame else []

        action = agent.runtime.step_decide(
            frames=agent.frames,
            latest_frame=latest_frame,
            state_before=state_before,
            grid_before=_,
        )
        action_event = self._serialize_action(action)
        boundary_level_hint = (
            str(agent._subgoal_sequence_level)
            if agent._subgoal_sequence_active
            else self._mode_to_boundary_level(agent.current_mode.value)
        )

        learner_candidates, learner_selected_index = self._evaluate_learner_group(
            replay_actions=replay_actions,
            memory_override=memory_override,
            action_event=action_event,
            mode_hint=agent.current_mode.value,
            boundary_level_hint=boundary_level_hint,
        )
        if learner_candidates:
            selected = learner_candidates[learner_selected_index]
        else:
            # Safety fallback: evaluate once without grouping.
            selected = self._evaluate_single_learner_candidate(
                replay_actions=replay_actions,
                memory_override=memory_override,
                action_event=action_event,
                mode_hint=agent.current_mode.value,
                candidate_id=f"{candidate_id}_learner_fallback",
                boundary_level_hint=boundary_level_hint,
            )

        reward, top_components = self._module_reward(
            module=module,
            boundary_type=str(selected.metadata.get("boundary_type", "action")),
            surprise_signal=float(selected.metadata.get("surprise_signal", 0.0)),
            changed_cells=int(selected.metadata.get("changed_cells", 0)),
            level_delta=int(selected.metadata.get("level_delta", 0)),
            frame_state=str(selected.metadata.get("game_state", "NOT_FINISHED")),
            steps_used=max(1, int(selected.metadata.get("steps_used", 1))),
        )

        sample = CandidateSample(
            candidate_id=candidate_id,
            output=action.name,
            reward=reward,
            metadata={
                "action_name": action.name,
                "state_before_hash": selected.metadata.get("state_before_hash", "unknown"),
                "state_after_hash": selected.metadata.get("state_after_hash", "unknown"),
                "changed_cells": int(selected.metadata.get("changed_cells", 0)),
                "level_delta": int(selected.metadata.get("level_delta", 0)),
                "mode": selected.metadata.get("mode", agent.current_mode.value),
                "boundary_type": selected.metadata.get("boundary_type", "action"),
                "surprise_signal": float(selected.metadata.get("surprise_signal", 0.0)),
                "learner_selected_index": learner_selected_index,
                "learner_rewards": [c.reward for c in learner_candidates],
                "memory_before_count": int(selected.metadata.get("memory_before_count", 0)),
                "memory_after_count": int(selected.metadata.get("memory_after_count", 0)),
                "steps_used": int(selected.metadata.get("steps_used", 1)),
                "reward_components": top_components.to_dict(),
                "reward_channel_version": "v2",
            },
        )
        surprise_metrics = SurpriseMetrics(
            debiased_nll=float(selected.metadata.get("debiased_nll", 0.0)),
            self_rated_x10=float(selected.metadata.get("self_rated_x10", 0.0)),
            reward_value=float(selected.metadata.get("bundle_reward_value", 0.0)),
            source=str(selected.metadata.get("bundle_source", self.surprise_reward_source)),
            mean_nll_full=float(selected.metadata.get("mean_nll_full", 0.0)),
            mean_nll_stripped=float(selected.metadata.get("mean_nll_stripped", 0.0)),
        )

        final_agent: LoopAgent = selected.metadata["agent"]
        final_frame: FrameData = selected.metadata["frame_after"]
        return CandidateEvalContext(
            sample=sample,
            agent=final_agent,
            frame_after=final_frame,
            action_events=self._metadata_action_events(
                selected.metadata.get("action_events"),
                fallback=action_event,
            ),
            semantic_report=str(selected.metadata.get("semantic_report", "")),
            surprise_metrics=surprise_metrics,
            changed_cells=int(selected.metadata.get("changed_cells", 0)),
            level_delta=int(selected.metadata.get("level_delta", 0)),
            boundary_type=str(selected.metadata.get("boundary_type", "action")),
            steps_used=max(1, int(selected.metadata.get("steps_used", 1))),
            learner_group=learner_candidates,
            learner_selected_index=learner_selected_index,
        )

    def _evaluate_learner_group(
        self,
        *,
        replay_actions: list[SerializedAction],
        memory_override: Memory,
        action_event: SerializedAction,
        mode_hint: str,
        boundary_level_hint: str,
    ) -> tuple[list[CandidateSample], int]:
        """Evaluate grouped learner candidates at one boundary."""
        k = self.branching.config.count_for_module("learner")
        learner_candidates: list[CandidateSample] = []
        for idx in range(k):
            candidate = self._evaluate_single_learner_candidate(
                replay_actions=replay_actions,
                memory_override=memory_override,
                action_event=action_event,
                mode_hint=mode_hint,
                candidate_id=f"learner_{idx}",
                boundary_level_hint=boundary_level_hint,
            )
            learner_candidates.append(candidate)
        self.branching._attach_advantages(learner_candidates)
        selected = self.branching.select_index(learner_candidates)
        return learner_candidates, selected

    def _evaluate_single_learner_candidate(
        self,
        *,
        replay_actions: list[SerializedAction],
        memory_override: Memory,
        action_event: SerializedAction,
        mode_hint: str,
        candidate_id: str,
        boundary_level_hint: str,
    ) -> CandidateSample:
        """Evaluate one learner candidate by replaying prefix + one boundary."""
        agent = self._new_agent_with_replay(
            replay_actions=replay_actions,
            memory_override=memory_override,
        )
        latest_before = agent._convert_raw_frame_data(
            agent.arc_env.observation_space if agent.arc_env else None
        )
        state_before_boundary = agent.state_encoder.encode(latest_before)
        grid_before_boundary = latest_before.frame[-1] if latest_before.frame else []
        start_levels = latest_before.levels_completed
        memory_before_count = len(agent.memory.entries)

        action_events: list[SerializedAction] = []
        current_frame = latest_before
        next_action = self._deserialize_action(action_event)
        steps_used = 0
        frame_after = latest_before

        while True:
            step_state_before = agent.state_encoder.encode(current_frame)
            step_grid_before = current_frame.frame[-1] if current_frame.frame else []
            frame_candidate = agent.take_action(next_action)
            if frame_candidate is None:
                break
            frame_after = frame_candidate
            agent.append_frame(frame_after)
            agent.runtime.step_observe(
                state_before=step_state_before,
                grid_before=step_grid_before,
                action=next_action,
                frame_after=frame_after,
            )
            agent.action_counter += 1
            action_events.append(self._serialize_action(next_action))
            steps_used += 1
            current_frame = frame_after

            if frame_after.state in (GameState.WIN, GameState.GAME_OVER):
                break
            if agent._pending_subgoal_actions:
                next_action = agent._pending_subgoal_actions.pop(0)
                continue
            # Reached a decision boundary when no queued sequence action remains.
            break

        if not action_events:
            # Degenerate fallback to keep schema valid.
            action_events.append(action_event)
            steps_used = 1

        grid_after = frame_after.frame[-1] if frame_after.frame else []
        state_after = agent.state_encoder.encode(frame_after)
        diff_text = agent.state_encoder.get_diff_text(grid_before_boundary, grid_after)
        changed_cells = agent.state_encoder.get_num_changed_cells(grid_before_boundary, grid_after)
        image_before = agent._grid_image_data_url(grid_before_boundary)
        image_after = agent._grid_image_data_url(grid_after)
        image_diff = agent._transition_image_data_url(grid_before_boundary, grid_after)
        action_trace = [evt.name for evt in action_events]
        boundary_action_name = (
            action_trace[0]
            if len(action_trace) == 1
            else f"SUBGOAL_SEQ[{', '.join(action_trace)}]"
        )

        learner: Learner = agent.learner
        # World Model + Observer + Judge pipeline (replaces old semantic_report + self_rate_surprise)
        predicted = learner.predict_outcome(
            state_before=state_before_boundary,
            action_taken=boundary_action_name,
            memory=agent.memory,
            image_before_url=image_before,
        )
        observed = learner.observe_transition(
            state_before=state_before_boundary,
            state_after=state_after,
            diff_text=diff_text,
            image_before_url=image_before,
            image_after_url=image_after,
            image_diff_url=image_diff,
        )
        similarity = learner.judge_similarity(predicted, observed)
        # Map to legacy interfaces: semantic_report ≈ observer description,
        # self_rated ≈ surprise on 0-10 scale
        semantic_report = observed
        self_rated = int((6 - similarity) * 2)  # 0-10 scale for compat
        surprise_engine = SemanticDebiasedNLLSurprise(
            client=agent.client,
            model=self.model,
            reward_source=self.surprise_reward_source,
            self_rated_weight=self.self_rated_weight,
        )
        bundle_before = surprise_engine.compute_bundle(
            state_before=state_before_boundary,
            action=boundary_action_name,
            memory=memory_override,
            semantic_report=semantic_report,
            self_rated_x10=self_rated,
            visual_context_hint="BEFORE/AFTER/DIFF images provided in observer stage",
        )
        memory_after_count = len(agent.memory.entries)
        boundary_type = boundary_level_hint if boundary_level_hint in {"action", "subgoal", "plan"} else "action"

        bundle_after = surprise_engine.compute_bundle(
            state_before=state_before_boundary,
            action=boundary_action_name,
            memory=agent.memory,
            semantic_report=semantic_report,
            self_rated_x10=self_rated,
            visual_context_hint="BEFORE/AFTER/DIFF images provided in observer stage",
        )
        parse_ok = bool((agent.learner.last_answer_output or "").strip())
        contradiction_cleanup = "REMOVE" in (agent.learner.last_answer_output or "").upper()
        learner_components = RewardComponents(
            mismatch_reduction=float(bundle_before.debiased_nll - bundle_after.debiased_nll),
            parse_bonus=1.0 if parse_ok else 0.0,
            dedup_bonus=1.0 if memory_after_count <= memory_before_count + 1 else 0.0,
            contradiction_cleanup_bonus=1.0 if contradiction_cleanup else 0.0,
            parse_warning_penalty=0.0,
        )
        learner_signal = scalarize_learner(learner_components)
        if self.surprise_reward_source == "self_rated":
            surprise_signal = bundle_after.self_rated_x10 / 10.0
        elif self.surprise_reward_source == "hybrid":
            w = max(0.0, min(1.0, self.self_rated_weight))
            surprise_signal = (1.0 - w) * bundle_after.debiased_nll + w * (
                bundle_after.self_rated_x10 / 10.0
            )
        else:
            surprise_signal = bundle_after.debiased_nll

        level_delta = frame_after.levels_completed - start_levels
        metadata: dict[str, Any] = {
            "agent": agent,
            "frame_after": frame_after,
            "semantic_report": semantic_report,
            "mode": mode_hint,
            "changed_cells": changed_cells,
            "level_delta": level_delta,
            "state_before_hash": self._hash_text(state_before_boundary),
            "state_after_hash": self._hash_text(state_after),
            "game_state": frame_after.state.name,
            "boundary_type": boundary_type,
            "surprise_signal": surprise_signal,
            "debiased_nll": bundle_after.debiased_nll,
            "self_rated_x10": bundle_after.self_rated_x10,
            "bundle_reward_value": bundle_after.reward_value,
            "bundle_source": surprise_engine.reward_source,
            "mean_nll_full": bundle_after.mean_nll_full,
            "mean_nll_stripped": bundle_after.mean_nll_stripped,
            "memory_before_count": memory_before_count,
            "memory_after_count": memory_after_count,
            "steps_used": steps_used,
            "action_events": [asdict(event) for event in action_events],
            "reward_components": learner_components.to_dict(),
            "reward_channel_version": "v2",
        }
        return CandidateSample(
            candidate_id=candidate_id,
            output=boundary_action_name,
            reward=learner_signal,
            metadata=metadata,
        )

    def _module_reward(
        self,
        *,
        module: str,
        boundary_type: str,
        surprise_signal: float,
        changed_cells: int,
        level_delta: int,
        frame_state: str,
        steps_used: int,
    ) -> tuple[float, RewardComponents]:
        if module == "solver":
            components = RewardComponents(
                level_delta_reward=float(level_delta),
                win_bonus=1.0 if frame_state == GameState.WIN.name else 0.0,
                game_over_penalty=1.0 if frame_state == GameState.GAME_OVER.name else 0.0,
                step_penalty=float(max(1, steps_used)),
                solver_surprise_bonus=float(surprise_signal),
            )
            return scalarize_solver(components), components
        components = RewardComponents(
            surprise=float(surprise_signal),
            novelty=1.0 if changed_cells > 0 else 0.0,
            transition_magnitude=float(changed_cells),
            boundary_progress=float(level_delta),
        )
        return scalarize_curiosity(components, boundary_type=boundary_type), components

    @staticmethod
    def _mode_to_boundary_level(mode_value: str) -> str:
        return {
            DecisionMode.LEARN_ACTION.value: "action",
            DecisionMode.LEARN_SUBGOAL.value: "subgoal",
            DecisionMode.LEARN_PLAN.value: "plan",
            DecisionMode.SOLVE.value: "plan",
        }.get(mode_value, "action")

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
            learner_group_rows = [
                {
                    "candidate_id": lc.candidate_id,
                    "reward": lc.reward,
                    "advantage": lc.advantage,
                    "probability": lc.probability,
                    "memory_after_count": int(lc.metadata.get("memory_after_count", 0)),
                    "boundary_type": str(lc.metadata.get("boundary_type", "action")),
                    "steps_used": int(lc.metadata.get("steps_used", 1)),
                }
                for lc in c.learner_group
            ]
            candidate_rows.append(
                CandidateDecision(
                    candidate_id=c.sample.candidate_id,
                    output=c.sample.output,
                    reward=c.sample.reward,
                    advantage=c.sample.advantage,
                    probability=c.sample.probability,
                    reward_components={
                        k: float(v)
                        for k, v in (
                            c.sample.metadata.get("reward_components", {}) or {}
                        ).items()
                    },
                    surprise=c.surprise_metrics,
                    metadata={
                        **c.sample.metadata,
                        "learner_selected_index": c.learner_selected_index,
                        "learner_group": learner_group_rows,
                    },
                )
            )
        mode = selected.sample.metadata.get("mode", "LEARN_ACTION")
        state_before_hash = selected.sample.metadata.get("state_before_hash", "unknown")
        state_after_hash = selected.sample.metadata.get("state_after_hash", "unknown")
        return DecisionRecord(
            episode_id=f"{attempt_id}_episode",
            attempt_id=attempt_id,
            step=step,
            boundary_type=selected.boundary_type,
            mode=str(mode),
            module=module,
            prompt_fingerprint=f"{module}:{state_before_hash}",
            state_before=state_before_hash,
            state_after=state_after_hash,
            memory_before=f"size={int(selected.sample.metadata.get('memory_before_count', len(selected.agent.memory)))}",
            memory_after=f"size={int(selected.sample.metadata.get('memory_after_count', len(selected.agent.memory)))}",
            semantic_report=selected.semantic_report,
            selected_index=selected_index,
            reward_channel_version=str(
                selected.sample.metadata.get("reward_channel_version", "v2")
            ),
            candidates=candidate_rows,
            diagnostics={
                "changed_cells": selected.changed_cells,
                "level_delta": selected.level_delta,
                "game_state": selected.frame_after.state.name,
                "steps_used": selected.steps_used,
                "action_trace": [event.name for event in selected.action_events],
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

    @staticmethod
    def _metadata_action_events(
        raw_events: Any,
        *,
        fallback: SerializedAction,
    ) -> list[SerializedAction]:
        if isinstance(raw_events, list) and raw_events:
            parsed: list[SerializedAction] = []
            for item in raw_events:
                if isinstance(item, SerializedAction):
                    parsed.append(item)
                    continue
                if isinstance(item, dict) and "name" in item:
                    parsed.append(
                        SerializedAction(
                            name=str(item.get("name", fallback.name)),
                            data=dict(item.get("data", {})),
                            reasoning=item.get("reasoning"),
                        )
                    )
            if parsed:
                return parsed
        return [fallback]


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
