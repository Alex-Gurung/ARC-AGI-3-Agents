"""Test the Solver component with a live VLLM server.

Checks:
- Does the model pick valid actions?
- With good memory, does it make reasonable choices?

Usage:
    # Start VLLM first: bash scripts/run_vllm.sh
    uv run python scripts/test_solver.py [--base-url URL] [--model MODEL] [--n-trials N]
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from agents.templates.loop_agent.solver import Solver
from agents.templates.loop_agent.memory import Memory, MemoryEntry


def make_rich_memory() -> Memory:
    """Create a well-populated memory simulating learned game knowledge."""
    mem = Memory()
    mem.add(MemoryEntry(
        type="ACTION", content="ACTION1 moves player up by 1 cell",
        justification="confirmed 5 times", confidence=0.95,
        created_step=1, last_modified_step=10,
    ))
    mem.add(MemoryEntry(
        type="ACTION", content="ACTION2 moves player down by 1 cell",
        justification="confirmed 5 times", confidence=0.95,
        created_step=2, last_modified_step=11,
    ))
    mem.add(MemoryEntry(
        type="ACTION", content="ACTION3 moves player left by 1 cell",
        justification="confirmed 3 times", confidence=0.9,
        created_step=3, last_modified_step=12,
    ))
    mem.add(MemoryEntry(
        type="ACTION", content="ACTION4 moves player right by 1 cell",
        justification="confirmed 3 times", confidence=0.9,
        created_step=4, last_modified_step=13,
    ))
    mem.add(MemoryEntry(
        type="RULE", content="Black cells (5) are walls that block movement",
        justification="tried 4 times, never moved through them", confidence=0.95,
        created_step=5, last_modified_step=14,
    ))
    mem.add(MemoryEntry(
        type="VOCAB", content="Color 11 = door border, touching it may complete level",
        justification="observed door-like structure with color 11 border", confidence=0.6,
        created_step=8, last_modified_step=8,
    ))
    mem.add(MemoryEntry(
        type="PLAN", content="1) navigate to door at (50,30) 2) touch the door",
        justification="door is the only interactive-looking object", confidence=0.5,
        created_step=15, last_modified_step=15,
    ))
    return mem


def test_with_plan(solver: Solver, n_trials: int = 5):
    """Test solver with a well-populated memory and active plan."""
    print("=" * 60)
    print("TEST: Solver with Plan and Rich Memory")
    print("=" * 60)

    memory = make_rich_memory()
    solver.set_plan("1) navigate to door at (50,30) 2) touch the door")
    solver.set_subgoal("Move right toward the door at (50,30)")

    state = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
AVAILABLE_ACTIONS: RESET, ACTION1, ACTION2, ACTION3, ACTION4
CHANGED (2 cells):
  (32,30):0->9
  (31,30):9->0
SUMMARY: 64x64 grid
  Colors: {0:4000, 5:60, 9:1, 11:16, 12:1}
  Unique colors: 5"""

    available_actions = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    valid_count = 0
    latencies = []

    for i in range(n_trials):
        start = time.time()
        result = solver.solve(
            state_text=state,
            memory=memory,
            available_actions=available_actions,
        )
        elapsed = time.time() - start
        latencies.append(elapsed)

        is_valid = result["action"] in available_actions
        valid_count += int(is_valid)

        print(f"  Trial {i+1}: action={result['action']}, valid={is_valid}, "
              f"latency={elapsed:.3f}s, raw={result['raw'][:80]}")

    print(f"\n  Valid outputs: {valid_count}/{n_trials} ({100*valid_count/n_trials:.0f}%)")
    print(f"  Avg latency: {sum(latencies)/len(latencies):.3f}s")

    # Check if solver preferentially picks ACTION4 (move right, toward door)
    print(f"  (Expect ACTION4 = move right, toward the door at x=50)")
    print()


def test_empty_memory(solver: Solver, n_trials: int = 3):
    """Test solver with no memory (should still pick something valid)."""
    print("=" * 60)
    print("TEST: Solver with Empty Memory")
    print("=" * 60)

    memory = Memory()
    solver.clear_plan()

    state = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
AVAILABLE_ACTIONS: RESET, ACTION1, ACTION2, ACTION3, ACTION4
SUMMARY: 64x64 grid
  Colors: {0:4090, 5:25, 9:1, 12:1}
  Unique colors: 4"""

    available_actions = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    for i in range(n_trials):
        start = time.time()
        result = solver.solve(
            state_text=state,
            memory=memory,
            available_actions=available_actions,
        )
        elapsed = time.time() - start
        print(f"  Trial {i+1}: action={result['action']}, latency={elapsed:.3f}s")
    print()


def test_game_over_recovery(solver: Solver):
    """Test solver when recovering from game over."""
    print("=" * 60)
    print("TEST: Solver After Game Over")
    print("=" * 60)

    memory = make_rich_memory()
    # Add a learning from game over
    memory.add(MemoryEntry(
        type="RULE", content="Touching red cells (8) causes GAME_OVER",
        justification="died when moving into red area", confidence=0.85,
        created_step=20, last_modified_step=20,
    ))

    solver.set_plan("1) navigate to door avoiding red cells 2) touch door")
    solver.set_subgoal("Move up to avoid the red area below")

    state = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
AVAILABLE_ACTIONS: RESET, ACTION1, ACTION2, ACTION3, ACTION4
SUMMARY: 64x64 grid
  Colors: {0:3990, 5:60, 8:20, 9:1, 11:16, 12:1}
  Unique colors: 6"""

    available_actions = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    result = solver.solve(
        state_text=state,
        memory=memory,
        available_actions=available_actions,
    )

    print(f"  Action: {result['action']}")
    print(f"  Raw: {result['raw'][:80]}")
    print(f"  (Expect ACTION1 = move up, away from red danger zone)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    solver = Solver(client, args.model)

    print(f"Using VLLM at {args.base_url} with model {args.model}\n")

    test_with_plan(solver, args.n_trials)
    test_empty_memory(solver)
    test_game_over_recovery(solver)

    print("All Solver tests complete!")
