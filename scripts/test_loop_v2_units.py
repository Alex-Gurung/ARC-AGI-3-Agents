"""Offline unit/integration checks for Loop Agent v2 behavior."""

from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcengine import FrameData, GameAction, GameState

from agents.templates.loop_agent import DecisionMode, LoopAgent, Phase
from agents.templates.loop_agent.curiosity import Curiosity
from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory
from agents.templates.loop_agent.surprise import LevelController
from training.group_sampler import GroupSampler
from training.rollout_runner import ModuleCandidateConfig, TrainingRolloutRunner
from training.trajectory_schema import CandidateEvaluation


def _make_agent() -> LoopAgent:
    return LoopAgent(
        card_id="card",
        game_id="game",
        agent_name="loop_agent_test",
        ROOT_URL="",
        record=False,
        arc_env=None,  # type: ignore[arg-type]
    )


def test_subgoal_sequence_parser_truncation() -> None:
    print("=" * 60)
    print("TEST: Subgoal action parser truncates to 20")
    print("=" * 60)
    curiosity = Curiosity(client=None, model="dummy")  # type: ignore[arg-type]
    available = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
    raw = "ACTION_SEQUENCE: " + ", ".join(["ACTION1"] * 30)
    parsed = curiosity._parse_action_list(raw_output=raw, available_actions=available, max_steps=20)
    assert len(parsed) == 20, parsed
    assert all(action == "ACTION1" for action in parsed)
    print("  PASS\n")


def test_no_change_interrupt_and_terminal_precedence() -> None:
    print("=" * 60)
    print("TEST: No-change interrupt + terminal precedence")
    print("=" * 60)

    class SeqHarness:
        SUBGOAL_NO_CHANGE_LIMIT = 4

        def __init__(self) -> None:
            self._subgoal_sequence_taken_actions: list[str] = []
            self._subgoal_no_change_streak = 0
            self._pending_subgoal_actions = [GameAction.ACTION1]
            self.finalize_calls = 0
            self.finalize_reason = ""

        def _finalize_subgoal_sequence(self, **kwargs: Any) -> None:
            self.finalize_calls += 1
            self.finalize_reason = str(kwargs.get("reason"))

    harness = SeqHarness()
    for _ in range(3):
        LoopAgent._handle_subgoal_sequence_step(
            harness,  # type: ignore[arg-type]
            action=GameAction.ACTION1,
            state_after="state",
            grid_after=[[0]],
            frame_after=SimpleNamespace(state=GameState.NOT_FINISHED),
            num_changed_cells=0,
        )
    assert harness.finalize_calls == 0

    LoopAgent._handle_subgoal_sequence_step(
        harness,  # type: ignore[arg-type]
        action=GameAction.ACTION1,
        state_after="state",
        grid_after=[[0]],
        frame_after=SimpleNamespace(state=GameState.NOT_FINISHED),
        num_changed_cells=0,
    )
    assert harness.finalize_calls == 1
    assert harness.finalize_reason == "no_change"

    terminal_harness = SeqHarness()
    terminal_harness._subgoal_no_change_streak = 3
    LoopAgent._handle_subgoal_sequence_step(
        terminal_harness,  # type: ignore[arg-type]
        action=GameAction.ACTION1,
        state_after="state",
        grid_after=[[0]],
        frame_after=SimpleNamespace(state=GameState.GAME_OVER),
        num_changed_cells=0,
    )
    assert terminal_harness.finalize_calls == 1
    assert terminal_harness.finalize_reason == "terminal"
    print("  PASS\n")


def test_soft_reset_step_skip_gate() -> None:
    print("=" * 60)
    print("TEST: Soft reset step skips learner/surprise updates")
    print("=" * 60)
    harness = SimpleNamespace(_in_soft_reset=True, _current_state_text="before")
    LoopAgent._post_step(
        harness,  # type: ignore[arg-type]
        state_before="state_before",
        grid_before=[[0]],
        action=GameAction.RESET,
        frame_after=SimpleNamespace(full_reset=True),
    )
    assert harness._current_state_text is None
    print("  PASS\n")


def test_level_regression() -> None:
    print("=" * 60)
    print("TEST: LevelController.regress_one_level")
    print("=" * 60)
    ctrl = LevelController()
    ctrl.current_level = "plan"
    assert ctrl.regress_one_level() == "subgoal"
    assert ctrl.regress_one_level() == "action"
    assert ctrl.regress_one_level() == "action"
    print("  PASS\n")


def test_expectation_parser() -> None:
    print("=" * 60)
    print("TEST: Learner expectation parser")
    print("=" * 60)
    learner = Learner(client=None, model="dummy")  # type: ignore[arg-type]
    parsed = learner._parse_expectation_assessment(
        "verdict=unexpected conf=0.82 level=subgoal ref=M0012"
    )
    assert parsed is not None
    assert parsed["verdict"] == "unexpected"
    assert abs(parsed["confidence"] - 0.82) < 1e-6
    assert parsed["level"] == "subgoal"
    assert parsed["entry_ref"] == "M0012"
    print("  PASS\n")


def test_diagnosis_directed_level() -> None:
    print("=" * 60)
    print("TEST: Diagnosis-directed level routing")
    print("=" * 60)

    class LearnerStub:
        def __init__(self, outputs: list[dict[str, Any]]) -> None:
            self.outputs = outputs
            self.idx = 0

        def assess_expectation(self, **_: Any) -> dict[str, Any]:
            out = self.outputs[min(self.idx, len(self.outputs) - 1)]
            self.idx += 1
            return out

    harness = SimpleNamespace(
        learner=LearnerStub(
            [
                {"verdict": "unexpected", "confidence": 0.9, "level": "subgoal"},
                {"verdict": "expected", "confidence": 0.95, "level": "subgoal"},
            ]
        ),
        memory=Memory(),
        MISMATCH_CONF_THRESHOLD=0.7,
        _last_boundary_diagnosis_level="none",
    )

    routed = LoopAgent._run_mode_diagnosis(
        harness,  # type: ignore[arg-type]
        mode="subgoal",
        expected="exp",
        actual="obs",
        subgoal_index=0,
    )
    assert routed == "subgoal"

    routed = LoopAgent._run_mode_diagnosis(
        harness,  # type: ignore[arg-type]
        mode="subgoal",
        expected="exp",
        actual="obs",
        subgoal_index=0,
    )
    assert routed is None
    assert harness._last_boundary_diagnosis_level == "none"
    print("  PASS\n")


def test_plan_end_only_diagnosis() -> None:
    print("=" * 60)
    print("TEST: Plan diagnosis only at plan-attempt end")
    print("=" * 60)
    agent = _make_agent()
    agent.phase = Phase.EXPLOIT
    agent.current_mode = DecisionMode.LEARN_PLAN
    agent._subgoal_sequence_start_state = "s0"
    agent._subgoal_sequence_start_grid = [[0]]
    agent._subgoal_sequence_taken_actions = ["ACTION1"]
    agent._subgoal_sequence_planned_actions = ["ACTION1"]
    agent._subgoal_sequence_expected = "expect progress"
    agent._last_prediction = "expect progress"
    agent._subgoal_sequence_is_explore = False
    agent._subgoal_sequence_level = "plan"
    agent._active_subgoal = "reach target"
    agent._active_subgoal_index = 0
    agent._plan_attempt_active = True

    agent.learner.update = lambda **_: False  # type: ignore[method-assign]
    agent.surprise.compute = lambda **_: 0.0  # type: ignore[method-assign]
    agent._record_surprise = lambda *_: None  # type: ignore[method-assign]
    agent._route_mode_after_boundary = lambda **_: None  # type: ignore[method-assign]
    agent._reset_subgoal_sequence_state = lambda **_: None  # type: ignore[method-assign]

    calls: list[str] = []
    agent._run_mode_diagnosis = lambda **kwargs: calls.append(str(kwargs.get("mode")))  # type: ignore[method-assign]

    # Non-terminal + exhausted + advanced => not plan-end yet.
    agent._advance_subgoal = lambda: True  # type: ignore[method-assign]
    agent._finalize_subgoal_sequence(
        state_after="s1",
        grid_after=[[0]],
        frame_after=FrameData(state=GameState.NOT_FINISHED),
        reason="exhausted",
    )
    assert calls == []

    # Non-terminal + exhausted + not advanced => plan-attempt end.
    agent._advance_subgoal = lambda: False  # type: ignore[method-assign]
    agent._finalize_subgoal_sequence(
        state_after="s2",
        grid_after=[[0]],
        frame_after=FrameData(state=GameState.NOT_FINISHED),
        reason="exhausted",
    )
    assert calls == ["plan"]
    print("  PASS\n")


def test_game_over_diagnostic_override() -> None:
    print("=" * 60)
    print("TEST: GAME_OVER diagnostic level jump")
    print("=" * 60)
    agent = _make_agent()
    agent.phase = Phase.EXPLOIT
    agent.current_mode = DecisionMode.SOLVE

    agent.learner.diagnose = lambda **_: {"level": "subgoal", "entry_index": None}  # type: ignore[method-assign]
    agent._route_mode_after_boundary(
        state_text="state_after",
        frame_after=FrameData(state=GameState.GAME_OVER),
        diagnosis_level=None,
    )

    assert agent.current_mode == DecisionMode.LEARN_SUBGOAL
    print("  PASS\n")


def test_group_sampler_and_training_selection() -> None:
    print("=" * 60)
    print("TEST: Group sampler + training selection")
    print("=" * 60)
    sampler = GroupSampler()
    rewards = [0.1, 0.5, -0.3, 0.0]
    advantages = sampler.advantages(rewards)
    assert len(advantages) == len(rewards)
    assert abs(sum(advantages)) < 1e-6

    idx = sampler.sample_index(rewards, rng=random.Random(123))
    assert 0 <= idx < len(rewards)

    runner_a = TrainingRolloutRunner(
        candidate_config=ModuleCandidateConfig(solver=4, curiosity=3, learner=2),
        seed=999,
    )
    runner_b = TrainingRolloutRunner(
        candidate_config=ModuleCandidateConfig(solver=4, curiosity=3, learner=2),
        seed=999,
    )
    assert runner_a.module_candidate_count("solver") == 4
    assert runner_a.module_candidate_count("curiosity") == 3
    assert runner_a.module_candidate_count("learner") == 2
    assert runner_a.select_candidate(
        [
            CandidateEvaluation(candidate_id="c0", output="", reward=0.1),
            CandidateEvaluation(candidate_id="c1", output="", reward=0.6),
            CandidateEvaluation(candidate_id="c2", output="", reward=0.2),
        ]
    ) == runner_b.select_candidate(
        [
            CandidateEvaluation(candidate_id="c0", output="", reward=0.1),
            CandidateEvaluation(candidate_id="c1", output="", reward=0.6),
            CandidateEvaluation(candidate_id="c2", output="", reward=0.2),
        ]
    )

    canonical_memory: list[str] = []
    side_logs: list[str] = []
    candidates = [
        CandidateEvaluation(candidate_id="c0", output="A", reward=0.1),
        CandidateEvaluation(candidate_id="c1", output="B", reward=0.7),
        CandidateEvaluation(candidate_id="c2", output="C", reward=0.2),
    ]
    runner_a.commit_selected_candidate(
        candidates=candidates,
        selected_index=1,
        commit_fn=lambda c: canonical_memory.append(c.candidate_id),
        log_fn=lambda c: side_logs.append(c.candidate_id),
    )
    assert canonical_memory == ["c1"]
    assert side_logs == ["c0", "c2"]
    print("  PASS\n")


if __name__ == "__main__":
    test_subgoal_sequence_parser_truncation()
    test_no_change_interrupt_and_terminal_precedence()
    test_soft_reset_step_skip_gate()
    test_level_regression()
    test_expectation_parser()
    test_diagnosis_directed_level()
    test_plan_end_only_diagnosis()
    test_game_over_diagnostic_override()
    test_group_sampler_and_training_selection()
    print("All Loop v2 unit tests passed!")
