"""Test the Surprise computation module.

Tests both LogProbSurprise (with VLLM) and HeuristicSurprise (no LLM).

Usage:
    # Heuristic only (no VLLM needed):
    uv run python scripts/test_surprise.py --strategy heuristic

    # LogProb (needs VLLM):
    uv run python scripts/test_surprise.py --strategy logprob [--base-url URL] [--model MODEL]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.templates.loop_agent.memory import Memory
from agents.templates.loop_agent.surprise import (
    HeuristicSurprise,
    LevelController,
    LogProbSurprise,
)


def test_heuristic_surprise():
    """Test heuristic surprise with various scenarios."""
    print("=" * 60)
    print("TEST: Heuristic Surprise")
    print("=" * 60)

    surprise = HeuristicSurprise()
    memory = Memory()

    # Scenario 1: No changes
    score = surprise.compute(
        state_before="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
        action="ACTION1",
        state_after="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
        memory=memory,
        num_changed_cells=0,
    )
    print(f"  No changes: surprise={score:.3f}")

    # Scenario 2: Small change (player moved)
    score = surprise.compute(
        state_before="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
        action="ACTION1",
        state_after="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
        memory=memory,
        num_changed_cells=2,
    )
    print(f"  Small change (2 cells): surprise={score:.3f}")

    # Scenario 3: Large change
    score = surprise.compute(
        state_before="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
        action="ACTION4",
        state_after="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
        memory=memory,
        num_changed_cells=60,
    )
    print(f"  Large change (60 cells): surprise={score:.3f}")

    # Scenario 4: Game Over
    score = surprise.compute(
        state_before="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
        action="ACTION2",
        state_after="STATE: GAME_OVER\nLEVELS_COMPLETED: 0",
        memory=memory,
        num_changed_cells=5,
    )
    print(f"  Game Over: surprise={score:.3f}")

    # Scenario 5: Level Complete
    score = surprise.compute(
        state_before="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
        action="ACTION4",
        state_after="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 1",
        memory=memory,
        num_changed_cells=100,
    )
    print(f"  Level Complete: surprise={score:.3f}")

    print()


def test_logprob_surprise(base_url: str, model: str):
    """Test logprob surprise with VLLM."""
    print("=" * 60)
    print("TEST: LogProb Surprise")
    print("=" * 60)

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="dummy")
    surprise = LogProbSurprise(client=client, model=model)
    memory = Memory()

    # Test with a simple state transition
    state_before = "GRID: 0*64 per row, player at (30,30)"
    state_after_expected = "CHANGED (2 cells): (30,30):9->0, (30,29):0->9"
    state_after_unexpected = "STATE: GAME_OVER. CHANGED (200 cells): massive explosion"

    # Expected outcome
    score_expected = surprise.compute(
        state_before=state_before,
        action="ACTION1",
        state_after=state_after_expected,
        memory=memory,
        num_changed_cells=2,
    )
    print(f"  Expected outcome: surprise={score_expected:.3f}")

    # Unexpected outcome
    score_unexpected = surprise.compute(
        state_before=state_before,
        action="ACTION1",
        state_after=state_after_unexpected,
        memory=memory,
        num_changed_cells=200,
    )
    print(f"  Unexpected outcome: surprise={score_unexpected:.3f}")

    print(f"  Running mean: {surprise.running_mean:.3f}")
    print()


def test_level_controller():
    """Test the LevelController surprise-rate convergence."""
    print("=" * 60)
    print("TEST: Level Controller")
    print("=" * 60)

    ctrl = LevelController(convergence_threshold=0.05, window_size=5)
    assert ctrl.current_level == "action"
    print(f"  Initial level: {ctrl.current_level}")

    # Simulate converging surprise at action level
    for i in range(10):
        # Surprise decreasing and stabilizing
        surprise = 1.0 / (i + 1) + 0.1
        level = ctrl.update(surprise)
        print(f"  Step {i}: surprise={surprise:.3f}, level={level}")

    print(f"  Final level after convergence: {ctrl.current_level}")

    # Test diagnosis regression
    ctrl.diagnose_to_level("action")
    print(f"  After diagnosis to action: {ctrl.current_level}")
    assert ctrl.current_level == "action"

    # Test one-level regression helper
    ctrl.current_level = "plan"
    ctrl.regress_one_level()
    assert ctrl.current_level == "subgoal"
    ctrl.regress_one_level()
    assert ctrl.current_level == "action"
    print(f"  After regress_one_level x2: {ctrl.current_level}")

    # Test reset
    ctrl.reset()
    print(f"  After reset: {ctrl.current_level}")
    assert ctrl.current_level == "action"

    print("  PASS\n")


def test_debiasing():
    """Test that surprise debiasing works (running mean subtraction)."""
    print("=" * 60)
    print("TEST: Surprise Debiasing")
    print("=" * 60)

    surprise = HeuristicSurprise(window_size=5)
    memory = Memory()

    scores = []
    # Run 10 identical actions — debiased surprise should trend toward 0
    for i in range(10):
        score = surprise.compute(
            state_before="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
            action="ACTION1",
            state_after="STATE: NOT_FINISHED\nLEVELS_COMPLETED: 0",
            memory=memory,
            num_changed_cells=2,
        )
        scores.append(score)
        print(f"  Step {i}: debiased_surprise={score:.3f}")

    # After enough identical steps, debiased should be near 0
    assert abs(scores[-1]) < 0.1, f"Debiased surprise should converge to ~0, got {scores[-1]:.3f}"
    print("  PASS: debiased surprise converges to ~0 for identical actions\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["heuristic", "logprob", "both"], default="heuristic")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    args = parser.parse_args()

    test_level_controller()
    test_debiasing()

    if args.strategy in ("heuristic", "both"):
        test_heuristic_surprise()

    if args.strategy in ("logprob", "both"):
        test_logprob_surprise(args.base_url, args.model)

    print("All Surprise tests complete!")
