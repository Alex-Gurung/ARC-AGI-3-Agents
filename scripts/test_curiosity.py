"""Test the Curiosity component with a live VLLM server.

Checks:
- Does the model produce valid output (one of ACTION1-7, RESET)?
- Does it follow the format?
- How often does it produce garbage?
- What's the latency?

Usage:
    # Start VLLM first: bash scripts/run_vllm.sh
    uv run python scripts/test_curiosity.py [--base-url URL] [--model MODEL] [--n-trials N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from agents.templates.loop_agent.curiosity import Curiosity
from agents.templates.loop_agent.memory import Memory, MemoryEntry


def render_raw(raw: str, show_full_raw: bool, preview_len: int = 80) -> str:
    """Render raw output as single-line preview or full block."""
    if show_full_raw:
        return f"\n----- RAW START -----\n{raw}\n----- RAW END -----"
    return raw[:preview_len]


def make_sample_state() -> str:
    """Create a sample compressed state for testing."""
    return """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
AVAILABLE_ACTIONS: RESET, ACTION1, ACTION2, ACTION3, ACTION4
CHANGED (2 cells):
  (30,30):9->0
  (30,29):0->9
SUMMARY: 64x64 grid
  Colors: {0:4094, 5:25, 9:1, 12:1}
  Unique colors: 4"""


def make_sample_memory() -> Memory:
    """Create a sample memory with some entries."""
    mem = Memory()
    mem.add(MemoryEntry(
        type="ACTION", content="ACTION1 moves player up by 1 cell",
        justification="player sprite shifted up after ACTION1",
        confidence=0.8, created_step=1, last_modified_step=1,
    ))
    mem.add(MemoryEntry(
        type="RULE", content="Black cells (5) are walls, cannot pass through",
        justification="tried moving into them twice with no effect",
        confidence=0.7, created_step=3, last_modified_step=3,
    ))
    mem.add(MemoryEntry(
        type="OBSERVATION", content="There is a colored region at (20,20)",
        justification="seen in initial grid",
        confidence=0.4, created_step=0, last_modified_step=0,
    ))
    return mem


def test_action_level(curiosity: Curiosity, n_trials: int = 5, show_full_raw: bool = False):
    """Test curiosity at action level."""
    print("=" * 60)
    print("TEST: Curiosity Action Level")
    print("=" * 60)

    state = make_sample_state()
    memory = make_sample_memory()
    available_actions = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    valid_count = 0
    latencies = []

    for i in range(n_trials):
        start = time.time()
        result = curiosity.propose_action(
            state_text=state,
            memory=memory,
            available_actions=available_actions,
            level="action",
        )
        elapsed = time.time() - start
        latencies.append(elapsed)

        is_valid = result["value"] in available_actions
        valid_count += int(is_valid)

        print(
            f"  Trial {i+1}: action={result['value']}, valid={is_valid}, "
            f"latency={elapsed:.3f}s, raw={render_raw(result['raw'], show_full_raw)}"
        )

    print(f"\n  Valid outputs: {valid_count}/{n_trials} ({100*valid_count/n_trials:.0f}%)")
    print(f"  Avg latency: {sum(latencies)/len(latencies):.3f}s")
    print()


def test_subgoal_level(curiosity: Curiosity, n_trials: int = 3, show_full_raw: bool = False):
    """Test curiosity at subgoal level."""
    print("=" * 60)
    print("TEST: Curiosity Subgoal Level")
    print("=" * 60)

    state = make_sample_state()
    memory = make_sample_memory()

    for i in range(n_trials):
        start = time.time()
        result = curiosity.propose_action(
            state_text=state,
            memory=memory,
            available_actions=["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            level="subgoal",
        )
        elapsed = time.time() - start

        subgoal_text = result["value"] if show_full_raw else result["value"][:80]
        print(f"  Trial {i+1}: subgoal='{subgoal_text}', latency={elapsed:.3f}s")
        if show_full_raw:
            print(render_raw(result["raw"], show_full_raw))
    print()


def test_plan_level(curiosity: Curiosity, n_trials: int = 3, show_full_raw: bool = False):
    """Test curiosity at plan level."""
    print("=" * 60)
    print("TEST: Curiosity Plan Level")
    print("=" * 60)

    memory = make_sample_memory()

    for i in range(n_trials):
        start = time.time()
        result = curiosity.propose_action(
            state_text="",
            memory=memory,
            available_actions=["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            level="plan",
        )
        elapsed = time.time() - start

        plan_text = result["value"] if show_full_raw else result["value"][:120]
        print(f"  Trial {i+1}: plan='{plan_text}', latency={elapsed:.3f}s")
        if show_full_raw:
            print(f"    parsed_steps={result.get('steps', [])}")
            print(render_raw(result["raw"], show_full_raw))
    print()


def test_empty_memory(curiosity: Curiosity, show_full_raw: bool = False):
    """Test curiosity with no memory (fresh start)."""
    print("=" * 60)
    print("TEST: Curiosity with Empty Memory")
    print("=" * 60)

    state = make_sample_state()
    memory = Memory()
    available_actions = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    result = curiosity.propose_action(
        state_text=state,
        memory=memory,
        available_actions=available_actions,
        level="action",
    )

    print(f"  Action: {result['value']}, raw: {render_raw(result['raw'], show_full_raw)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument(
        "--show-full-raw",
        action="store_true",
        help="Print full raw model outputs instead of short previews",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    curiosity = Curiosity(client, args.model)

    print(f"Using VLLM at {args.base_url} with model {args.model}\n")

    test_action_level(curiosity, args.n_trials, args.show_full_raw)
    test_subgoal_level(curiosity, min(args.n_trials, 3), args.show_full_raw)
    test_plan_level(curiosity, min(args.n_trials, 3), args.show_full_raw)
    test_empty_memory(curiosity, args.show_full_raw)

    print("All Curiosity tests complete!")
