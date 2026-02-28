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
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory, MemoryEntry


def render_raw(raw: str, show_full_raw: bool, preview_len: int = 120) -> str:
    """Render raw output as single-line preview or full block."""
    if show_full_raw:
        return f"\n----- RAW START -----\n{raw}\n----- RAW END -----"
    return raw[:preview_len]


def test_first_action_discovery(learner: Learner, show_full_raw: bool = False):
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
    print(f"  Raw: {render_raw(learner.last_raw_output, show_full_raw)}")
    print(f"  Parsed answer: {learner.last_answer_output}")
    print(f"  Memory state: {memory.to_text()}")
    print()


def test_confirming_knowledge(learner: Learner, show_full_raw: bool = False):
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
    print(f"  Raw: {render_raw(learner.last_raw_output, show_full_raw)}")
    print(f"  Parsed answer: {learner.last_answer_output}")
    print(f"  Memory state: {memory.to_text()}")
    print()


def test_no_change(learner: Learner, show_full_raw: bool = False):
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
    print(f"  Raw: {render_raw(learner.last_raw_output, show_full_raw)}")
    print(f"  Parsed answer: {learner.last_answer_output}")
    print(f"  Memory state: {memory.to_text()}")
    print()


def test_game_over_discovery(learner: Learner, show_full_raw: bool = False):
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
    print(f"  Raw: {render_raw(learner.last_raw_output, show_full_raw)}")
    print(f"  Parsed answer: {learner.last_answer_output}")
    print(f"  Memory state: {memory.to_text()}")
    print()


def test_judge_similarity(learner: Learner, show_full_raw: bool = False):
    """Test judge similarity scoring."""
    print("=" * 60)
    print("TEST: Judge Similarity")
    print("=" * 60)

    predicted = "The blue object at row 5 col 3 should move up by 1 row to row 4 col 3. The dark walls remain unchanged. The rest of the grid stays black."
    observed = "The blue object moved from row 5 col 3 to row 4 col 3. All dark wall cells remained in place. No other changes occurred."

    start = time.time()
    similarity = learner.judge_similarity(predicted, observed)
    elapsed = time.time() - start

    surprise = (6 - similarity) / 5.0
    print(f"  Similarity: {similarity}/5")
    print(f"  Surprise: {surprise:.3f}")
    print(f"  Latency: {elapsed:.3f}s")
    print(f"  Raw: {render_raw(learner.last_raw_output, show_full_raw)}")
    print(f"  Parsed answer: {learner.last_answer_output}")
    assert 1 <= similarity <= 5, f"Similarity {similarity} out of range"
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--show-full-raw",
        action="store_true",
        help="Print full raw model outputs instead of short previews",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    learner = Learner(client, args.model)

    print(f"Using VLLM at {args.base_url} with model {args.model}\n")

    test_first_action_discovery(learner, args.show_full_raw)
    test_confirming_knowledge(learner, args.show_full_raw)
    test_no_change(learner, args.show_full_raw)
    test_game_over_discovery(learner, args.show_full_raw)
    test_judge_similarity(learner, args.show_full_raw)

    print("All Learner tests complete!")
