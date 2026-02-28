"""Test surprise estimation quality.

Part 1: Hand-crafted prediction/observation pairs → judge scores them.
Part 2: Full WM→Observer→Judge pipeline on synthetic transitions → see
        what the model says about its own outputs.

Usage:
    VLLM_BASE_URL=http://localhost:8000/v1 VLLM_MODEL=google/gemma-3-1b-it \
        uv run python scripts/test_surprise_estimation.py
"""

from __future__ import annotations

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory, MemoryEntry

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-3-1b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")

HR = "=" * 70


def make_learner() -> Learner:
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    return Learner(client, VLLM_MODEL)


# ──────────────────────────────────────────────────────────────────────
# Part 1: Hand-crafted pairs — expected judge behavior
# ──────────────────────────────────────────────────────────────────────

CRAFTED_CASES: list[dict] = [
    # Case A: perfect match — should score 4-5
    {
        "label": "A: Perfect match",
        "predicted": (
            "The blue cell at row 5 col 3 moves up by one row to row 4 col 3. "
            "Its previous position becomes black (empty). All other cells remain unchanged. "
            "The grey border walls stay in place."
        ),
        "observed": (
            "The blue cell (color 3) shifted from row 5 col 3 to row 4 col 3. "
            "The cell at row 5 col 3 is now black (0). Everything else is identical. "
            "Grey walls (color 5) along the border are unchanged."
        ),
        "expected_range": (4, 5),
    },
    # Case B: wrong direction — should score 1-2
    {
        "label": "B: Wrong direction",
        "predicted": (
            "The red player moves left by one column, from col 7 to col 6. "
            "The target green cell remains at col 10."
        ),
        "observed": (
            "The red player moved right by one column, from col 7 to col 8. "
            "A new yellow cell appeared at col 5. The green cell is still at col 10."
        ),
        "expected_range": (1, 2),
    },
    # Case C: partially right — should score 2-3
    {
        "label": "C: Partially right (got movement, missed side effects)",
        "predicted": (
            "ACTION1 moves the orange block down by one row. "
            "No other changes occur on the grid."
        ),
        "observed": (
            "The orange block moved down by one row as expected. "
            "However, a chain reaction triggered: three grey cells below it "
            "also shifted down, and the bottom-most grey cell disappeared off the grid."
        ),
        "expected_range": (2, 3),
    },
    # Case D: completely wrong — should score 1
    {
        "label": "D: Completely wrong",
        "predicted": (
            "Nothing changes. The grid remains identical to before the action. "
            "The player stays at row 2 col 2."
        ),
        "observed": (
            "The entire grid reset. All colored cells cleared to black. "
            "A new pattern of blue and red cells appeared in the center. "
            "Game state changed to a new level."
        ),
        "expected_range": (1, 1),
    },
    # Case E: near-miss — should score 3-4
    {
        "label": "E: Near-miss (right idea, wrong magnitude)",
        "predicted": (
            "The green cluster shifts right by 2 columns. "
            "The leftmost column of the cluster becomes empty."
        ),
        "observed": (
            "The green cluster shifted right by 1 column (not 2). "
            "The leftmost column of the cluster is now empty. "
            "Otherwise the shape is preserved."
        ),
        "expected_range": (3, 4),
    },
]


def run_part1(learner: Learner) -> list[dict]:
    """Score hand-crafted prediction/observation pairs."""
    print(f"\n{HR}")
    print("PART 1: Hand-crafted prediction/observation → Judge scores")
    print(HR)

    results = []
    for case in CRAFTED_CASES:
        similarity = learner.judge_similarity(case["predicted"], case["observed"])
        surprise = (6 - similarity) / 5.0
        lo, hi = case["expected_range"]
        in_range = lo <= similarity <= hi
        status = "OK" if in_range else "MISS"

        result = {
            "label": case["label"],
            "similarity": similarity,
            "surprise": surprise,
            "expected_range": case["expected_range"],
            "in_range": in_range,
            "raw_output": learner.last_raw_output,
        }
        results.append(result)

        print(f"\n  [{status}] {case['label']}")
        print(f"       similarity={similarity}/5  surprise={surprise:.2f}  expected={lo}-{hi}")
        print(f"       raw judge output: {learner.last_raw_output[:120]}")

    hits = sum(1 for r in results if r["in_range"])
    print(f"\n  Summary: {hits}/{len(results)} within expected range")
    return results


# ──────────────────────────────────────────────────────────────────────
# Part 2: Full pipeline on synthetic transitions
# ──────────────────────────────────────────────────────────────────────

# Synthetic game states (RLE-style, mimicking state_encoder output)
TRANSITIONS: list[dict] = [
    {
        "label": "Simple movement: blue cell moves up",
        "state_before": textwrap.dedent("""\
            STATE: NOT_FINISHED
            LEVELS_COMPLETED: 0
            AVAILABLE_ACTIONS: ACTION1, ACTION2, ACTION3, ACTION4, RESET
            KEYFRAME: initial
            GRID (5x5):
            r0: 5*5
            r1: 5 0*3 5
            r2: 5 0 3 0 5
            r3: 5 0*3 5
            r4: 5*5
            SUMMARY: 5x5 grid, colors: 0:6, 3:1, 5:18"""),
        "action": "ACTION1",
        "state_after": textwrap.dedent("""\
            STATE: NOT_FINISHED
            LEVELS_COMPLETED: 0
            AVAILABLE_ACTIONS: ACTION1, ACTION2, ACTION3, ACTION4, RESET
            CHANGED CELLS:
            (2,2):3->0
            (2,1):0->3
            SUMMARY: 5x5 grid, colors: 0:6, 3:1, 5:18"""),
        "diff": "(2,2):3->0, (2,1):0->3",
        "memory_entries": [],
    },
    {
        "label": "No change: blocked by wall",
        "state_before": textwrap.dedent("""\
            STATE: NOT_FINISHED
            LEVELS_COMPLETED: 0
            AVAILABLE_ACTIONS: ACTION1, ACTION2, ACTION3, ACTION4, RESET
            KEYFRAME: initial
            GRID (5x5):
            r0: 5*5
            r1: 5 3 0*2 5
            r2: 5 0*3 5
            r3: 5 0*3 5
            r4: 5*5
            SUMMARY: 5x5 grid, colors: 0:9, 3:1, 5:15"""),
        "action": "ACTION1",
        "state_after": textwrap.dedent("""\
            STATE: NOT_FINISHED
            LEVELS_COMPLETED: 0
            AVAILABLE_ACTIONS: ACTION1, ACTION2, ACTION3, ACTION4, RESET
            GRID: no changes
            SUMMARY: 5x5 grid, colors: 0:9, 3:1, 5:15"""),
        "diff": "no changes",
        "memory_entries": [
            ("ACTION", "ACTION1 moves the blue cell (3) upward by one row", "observed once", 0.4),
        ],
    },
    {
        "label": "Game over: player falls into trap",
        "state_before": textwrap.dedent("""\
            STATE: NOT_FINISHED
            LEVELS_COMPLETED: 0
            AVAILABLE_ACTIONS: ACTION1, ACTION2, ACTION3, ACTION4, RESET
            KEYFRAME: initial
            GRID (7x7):
            r0: 5*7
            r1: 5 0*5 5
            r2: 5 0 0 3 0 0 5
            r3: 5 0 0 2 0 0 5
            r4: 5 0 0 2 0 0 5
            r5: 5 0*5 5
            r6: 5*7
            SUMMARY: 7x7 grid, colors: 0:20, 2:2, 3:1, 5:26"""),
        "action": "ACTION2",
        "state_after": textwrap.dedent("""\
            STATE: GAME_OVER
            LEVELS_COMPLETED: 0
            AVAILABLE_ACTIONS: RESET
            CHANGED CELLS:
            (3,2):3->0
            (3,3):2->3
            SUMMARY: 7x7 grid, colors: 0:21, 2:1, 3:1, 5:26"""),
        "diff": "(3,2):3->0, (3,3):2->3",
        "memory_entries": [
            ("ACTION", "ACTION1 moves the blue cell (3) upward by one row", "observed twice", 0.6),
            ("RULE", "Red cells (2) are dangerous — touching them ends the game", "hypothesis from layout", 0.3),
        ],
    },
]


def run_part2(learner: Learner) -> list[dict]:
    """Run full WM→Observer→Judge pipeline on synthetic transitions."""
    print(f"\n{HR}")
    print("PART 2: Full pipeline — World Model predicts, Observer reports, Judge scores")
    print(HR)

    results = []
    for trans in TRANSITIONS:
        print(f"\n  --- {trans['label']} ---")
        print(f"  Action: {trans['action']}")

        # Build memory
        memory = Memory(max_entries=50)
        for mtype, content, justification, conf in trans["memory_entries"]:
            memory.add(MemoryEntry(
                type=mtype,
                content=content,
                justification=justification,
                confidence=conf,
                created_step=0,
                last_modified_step=0,
            ))

        # 1. World Model predicts
        predicted = learner.predict_outcome(
            state_before=trans["state_before"],
            action_taken=trans["action"],
            memory=memory,
        )
        print(f"\n  [World Model prediction]")
        print(textwrap.indent(predicted[:300], "    "))

        # 2. Observer describes what actually happened
        observed = learner.observe_transition(
            state_before=trans["state_before"],
            state_after=trans["state_after"],
            diff_text=trans["diff"],
        )
        print(f"\n  [Observer report]")
        print(textwrap.indent(observed[:300], "    "))

        # 3. Judge scores
        similarity = learner.judge_similarity(predicted, observed)
        surprise = (6 - similarity) / 5.0
        print(f"\n  [Judge verdict]")
        print(f"    similarity={similarity}/5  surprise={surprise:.2f}")
        print(f"    raw: {learner.last_raw_output[:120]}")

        results.append({
            "label": trans["label"],
            "action": trans["action"],
            "predicted": predicted,
            "observed": observed,
            "similarity": similarity,
            "surprise": surprise,
            "judge_raw": learner.last_raw_output,
        })

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Connecting to VLLM: {VLLM_BASE_URL}  model: {VLLM_MODEL}")
    learner = make_learner()

    # Quick connectivity check
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"Available models: {available}")
        if VLLM_MODEL not in available:
            print(f"WARNING: {VLLM_MODEL} not in available models!")
    except Exception as e:
        print(f"WARNING: Could not list models ({e}). Proceeding anyway...")

    p1_results = run_part1(learner)
    p2_results = run_part2(learner)

    # Final summary
    print(f"\n{HR}")
    print("SUMMARY")
    print(HR)

    print("\nPart 1 — Judge calibration on hand-crafted pairs:")
    for r in p1_results:
        lo, hi = r["expected_range"]
        tag = "OK  " if r["in_range"] else "MISS"
        print(f"  [{tag}] {r['label']:50s}  sim={r['similarity']}  expected={lo}-{hi}")

    print("\nPart 2 — WM self-assessment via pipeline:")
    for r in p2_results:
        print(f"  {r['label']:50s}  sim={r['similarity']}/5  surprise={r['surprise']:.2f}")

    avg_sim = sum(r["similarity"] for r in p2_results) / max(len(p2_results), 1)
    print(f"\n  Average pipeline similarity: {avg_sim:.1f}/5")
    print(f"  Average pipeline surprise:   {(6 - avg_sim) / 5.0:.2f}")


if __name__ == "__main__":
    main()
