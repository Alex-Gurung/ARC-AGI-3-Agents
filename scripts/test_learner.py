"""Test the Learner component with a live VLLM server.

Checks:
- Does the model produce valid memory operations (ADD/REMOVE/MODIFY/NONE)?
- Does the output parse correctly?
- Are the memory entries semantically reasonable?

Usage:
    # Start VLLM first: bash scripts/run_vllm.sh
    uv run python scripts/test_learner.py [--base-url URL] [--model MODEL]
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory, MemoryEntry


def test_first_action_discovery(learner: Learner):
    """Test learning from the very first action (no prior memory)."""
    print("=" * 60)
    print("TEST: First Action Discovery")
    print("=" * 60)

    memory = Memory()

    state_before = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
GRID (64x64):
  r29: 0*30 12 0*33
  r30: 0*30 9 0*33
  (other rows: all 0)
SUMMARY: 64x64 grid
  Colors: {0:4094, 9:1, 12:1}
  Unique colors: 3"""

    state_after = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
CHANGED (2 cells):
  (30,29):12->0
  (30,28):0->12
SUMMARY: 64x64 grid
  Colors: {0:4094, 9:1, 12:1}
  Unique colors: 3"""

    diff_text = """CHANGED (2 cells):
  (30,29):12->0
  (30,28):0->12"""

    start = time.time()
    changed = learner.update(
        state_before=state_before,
        action_taken="ACTION1",
        state_after=state_after,
        diff_text=diff_text,
        memory=memory,
        current_step=1,
    )
    elapsed = time.time() - start

    print(f"  Memory changed: {changed}")
    print(f"  Latency: {elapsed:.3f}s")
    print(f"  Memory state: {memory.to_text()}")
    print()


def test_confirming_knowledge(learner: Learner):
    """Test learner when action confirms existing knowledge."""
    print("=" * 60)
    print("TEST: Confirming Existing Knowledge")
    print("=" * 60)

    memory = Memory()
    memory.add(MemoryEntry(
        type="ACTION", content="ACTION1 moves the colored pixel up",
        justification="observed pixel shift up once",
        confidence=0.6, created_step=1, last_modified_step=1,
    ))

    state_before = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
CHANGED (2 cells):
  (30,28):0->12
  (30,29):12->0"""

    state_after = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
CHANGED (2 cells):
  (30,27):0->12
  (30,28):12->0"""

    diff_text = """CHANGED (2 cells):
  (30,27):0->12
  (30,28):12->0"""

    start = time.time()
    changed = learner.update(
        state_before=state_before,
        action_taken="ACTION1",
        state_after=state_after,
        diff_text=diff_text,
        memory=memory,
        current_step=5,
    )
    elapsed = time.time() - start

    print(f"  Memory changed: {changed}")
    print(f"  Latency: {elapsed:.3f}s")
    print(f"  Memory state: {memory.to_text()}")
    print()


def test_no_change(learner: Learner):
    """Test learner when nothing happened (wall collision)."""
    print("=" * 60)
    print("TEST: No Change (Wall Collision)")
    print("=" * 60)

    memory = Memory()
    memory.add(MemoryEntry(
        type="ACTION", content="ACTION1 moves player up",
        justification="observed",
        confidence=0.8, created_step=1, last_modified_step=1,
    ))

    state_before = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0"""

    state_after = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0
DIFF: no changes"""

    diff_text = "no changes"

    start = time.time()
    changed = learner.update(
        state_before=state_before,
        action_taken="ACTION1",
        state_after=state_after,
        diff_text=diff_text,
        memory=memory,
        current_step=10,
    )
    elapsed = time.time() - start

    print(f"  Memory changed: {changed}")
    print(f"  Latency: {elapsed:.3f}s")
    print(f"  Memory state: {memory.to_text()}")
    print()


def test_game_over_discovery(learner: Learner):
    """Test learner when game over happens."""
    print("=" * 60)
    print("TEST: Game Over Discovery")
    print("=" * 60)

    memory = Memory()
    memory.add(MemoryEntry(
        type="ACTION", content="ACTION2 moves player down",
        justification="observed",
        confidence=0.8, created_step=1, last_modified_step=1,
    ))

    state_before = """STATE: NOT_FINISHED
LEVELS_COMPLETED: 0"""

    state_after = """STATE: GAME_OVER
LEVELS_COMPLETED: 0
CHANGED (5 cells):
  (30,60):0->8
  (31,60):0->8
  (30,61):0->8
  (31,61):0->8
  (30,30):9->0"""

    diff_text = """CHANGED (5 cells):
  (30,60):0->8
  (31,60):0->8
  (30,61):0->8
  (31,61):0->8
  (30,30):9->0"""

    start = time.time()
    changed = learner.update(
        state_before=state_before,
        action_taken="ACTION2",
        state_after=state_after,
        diff_text=diff_text,
        memory=memory,
        current_step=15,
    )
    elapsed = time.time() - start

    print(f"  Memory changed: {changed}")
    print(f"  Latency: {elapsed:.3f}s")
    print(f"  Memory state: {memory.to_text()}")
    print()


def test_diagnosis(learner: Learner):
    """Test solver diagnosis."""
    print("=" * 60)
    print("TEST: Solver Diagnosis")
    print("=" * 60)

    memory = Memory()
    memory.add(MemoryEntry(
        type="ACTION", content="ACTION1 moves player up",
        justification="observed", confidence=0.9,
        created_step=0, last_modified_step=0,
    ))
    memory.add(MemoryEntry(
        type="PLAN", content="1) move to door 2) touch door to win",
        justification="guessed from layout", confidence=0.4,
        created_step=5, last_modified_step=5,
    ))

    start = time.time()
    result = learner.diagnose(
        expected="touching the door should win",
        actual="GAME_OVER - touching the door killed us",
        memory=memory,
    )
    elapsed = time.time() - start

    print(f"  Diagnosis: {result}")
    print(f"  Latency: {elapsed:.3f}s")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    learner = Learner(client, args.model)

    print(f"Using VLLM at {args.base_url} with model {args.model}\n")

    test_first_action_discovery(learner)
    test_confirming_knowledge(learner)
    test_no_change(learner)
    test_game_over_discovery(learner)
    test_diagnosis(learner)

    print("All Learner tests complete!")
