"""Test surprise estimation quality.

Part 1: Hand-crafted prediction/observation pairs → judge scores them.
Part 2: Full WM→Observer→Judge pipeline on synthetic transitions → see
        what the model says about its own outputs.
Part 3: Grid representation comprehension — can the model read the RLE
        format and answer factual questions about cell positions/colors?

Usage:
    VLLM_BASE_URL=http://localhost:8000/v1 VLLM_MODEL=google/gemma-3-4b-it \
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
from agents.templates.loop_agent.state_encoder import StateEncoder

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
# Part 3: Grid representation comprehension
# ──────────────────────────────────────────────────────────────────────

GRID_5x5 = textwrap.dedent("""\
    GRID (5x5):
    r0: 5*5
    r1: 5 0*3 5
    r2: 5 0 3 0 5
    r3: 5 0*3 5
    r4: 5*5
    SUMMARY: 5x5 grid, colors: 0:6, 3:1, 5:18""")

GRID_7x7 = textwrap.dedent("""\
    GRID (7x7):
    r0: 5*7
    r1: 5 0*5 5
    r2: 5 0 0 3 0 0 5
    r3: 5 0 0 2 0 0 5
    r4: 5 0 0 2 0 0 5
    r5: 5 0*5 5
    r6: 5*7
    SUMMARY: 7x7 grid, colors: 0:20, 2:2, 3:1, 5:26""")

DIFF_TEXT = textwrap.dedent("""\
    BEFORE GRID (5x5):
    r0: 5*5
    r1: 5 0*3 5
    r2: 5 0 3 0 5
    r3: 5 0*3 5
    r4: 5*5

    CHANGED CELLS:
    (2,2):3->0
    (1,2):0->3

    AFTER GRID (5x5):
    r0: 5*5
    r1: 5 0 3 0 5
    r2: 5 0*3 5
    r3: 5 0*3 5
    r4: 5*5""")

COMPREHENSION_QS: list[dict] = [
    # --- Cell lookup ---
    {
        "label": "Cell lookup: r2,c2 in 5x5",
        "grid": GRID_5x5,
        "question": "What color is the cell at row 2, column 2?",
        "accept": ["3"],
        "category": "cell_lookup",
    },
    {
        "label": "Cell lookup: r0,c0 in 5x5 (corner)",
        "grid": GRID_5x5,
        "question": "What color is the cell at row 0, column 0?",
        "accept": ["5"],
        "category": "cell_lookup",
    },
    {
        "label": "Cell lookup: r1,c2 in 5x5 (interior)",
        "grid": GRID_5x5,
        "question": "What color is the cell at row 1, column 2?",
        "accept": ["0"],
        "category": "cell_lookup",
    },
    {
        "label": "Cell lookup: r3,c3 in 7x7",
        "grid": GRID_7x7,
        "question": "What color is the cell at row 3, column 3?",
        "accept": ["2"],
        "category": "cell_lookup",
    },
    {
        "label": "Cell lookup: r2,c3 in 7x7",
        "grid": GRID_7x7,
        "question": "What color is the cell at row 2, column 3?",
        "accept": ["3"],
        "category": "cell_lookup",
    },
    # --- Counting ---
    {
        "label": "Count: how many color-3 cells in 5x5",
        "grid": GRID_5x5,
        "question": "How many cells have color 3?",
        "accept": ["1", "one"],
        "category": "counting",
    },
    {
        "label": "Count: how many color-5 cells in 5x5",
        "grid": GRID_5x5,
        "question": "How many cells have color 5?",
        "accept": ["18", "eighteen"],
        "category": "counting",
    },
    {
        "label": "Count: how many color-2 cells in 7x7",
        "grid": GRID_7x7,
        "question": "How many cells have color 2?",
        "accept": ["2", "two"],
        "category": "counting",
    },
    # --- RLE expansion ---
    {
        "label": "RLE expand: list row 1 of 5x5",
        "grid": GRID_5x5,
        "question": "List the color of each cell in row 1, left to right, separated by spaces.",
        "accept": ["5 0 0 0 5"],
        "category": "rle_expand",
    },
    {
        "label": "RLE expand: list row 2 of 7x7",
        "grid": GRID_7x7,
        "question": "List the color of each cell in row 2, left to right, separated by spaces.",
        "accept": ["5 0 0 3 0 0 5"],
        "category": "rle_expand",
    },
    # --- Grid dimensions ---
    {
        "label": "Dimensions: 5x5",
        "grid": GRID_5x5,
        "question": "How many rows and columns does this grid have?",
        "accept": ["5", "5x5", "5 rows and 5 columns", "5 rows, 5 columns"],
        "category": "dimensions",
    },
    # --- Spatial / adjacency ---
    {
        "label": "Adjacency: what is directly above color 3 in 5x5",
        "grid": GRID_5x5,
        "question": "The cell with color 3 is at row 2, column 2. What color is the cell directly above it (row 1, column 2)?",
        "accept": ["0"],
        "category": "spatial",
    },
    {
        "label": "Adjacency: what is directly below color 3 in 7x7",
        "grid": GRID_7x7,
        "question": "The cell with color 3 is at row 2, column 3. What color is the cell directly below it (row 3, column 3)?",
        "accept": ["2"],
        "category": "spatial",
    },
    # --- Diff comprehension ---
    {
        "label": "Diff: which cell gained color 3",
        "grid": DIFF_TEXT,
        "question": "After the change, which cell now has color 3 that didn't have it before? Give row and column.",
        "accept": ["1,2", "row 1, column 2", "r1,c2", "(1,2)", "row 1 column 2", "1, 2"],
        "category": "diff",
    },
    {
        "label": "Diff: what happened to old color-3 cell",
        "grid": DIFF_TEXT,
        "question": "Cell (2,2) changed from color 3 to what color?",
        "accept": ["0"],
        "category": "diff",
    },
]

COMPREHENSION_PROMPT_TEXT = """\
You are reading a game grid encoded in RLE (run-length encoding) format.

In this format:
- "r0:", "r1:", etc. are row indices (top to bottom)
- Numbers are cell colors
- "N*K" means color N repeated K times
- For example: "r0: 5*3 0 2" means row 0 has cells [5, 5, 5, 0, 2]

{grid}

{question}

Think step by step, then output exactly one final line:
ANSWER: <your answer>"""

COMPREHENSION_PROMPT_VISUAL = """\
You are looking at a game grid. An image of the grid is attached.

The grid uses these colors (by index):
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=pink, 7=orange, 8=light-blue, 9=dark-red

Row 0 is the top row. Column 0 is the leftmost column.

{grid}

{question}

Think step by step, then output exactly one final line:
ANSWER: <your answer>"""

# Actual grid arrays matching the RLE definitions above
GRID_5x5_ARRAY = [
    [5, 5, 5, 5, 5],
    [5, 0, 0, 0, 5],
    [5, 0, 3, 0, 5],
    [5, 0, 0, 0, 5],
    [5, 5, 5, 5, 5],
]

GRID_7x7_ARRAY = [
    [5, 5, 5, 5, 5, 5, 5],
    [5, 0, 0, 0, 0, 0, 5],
    [5, 0, 0, 3, 0, 0, 5],
    [5, 0, 0, 2, 0, 0, 5],
    [5, 0, 0, 2, 0, 0, 5],
    [5, 0, 0, 0, 0, 0, 5],
    [5, 5, 5, 5, 5, 5, 5],
]

# After-state for the diff questions (color 3 moved from r2c2 to r1c2)
GRID_5x5_AFTER_ARRAY = [
    [5, 5, 5, 5, 5],
    [5, 0, 3, 0, 5],
    [5, 0, 0, 0, 5],
    [5, 0, 0, 0, 5],
    [5, 5, 5, 5, 5],
]


def _extract_answer(raw: str) -> str:
    """Pull text after the last ANSWER: marker."""
    if "ANSWER:" in raw.upper():
        idx = raw.upper().rfind("ANSWER:")
        return raw[idx + 7:].strip()
    return raw.strip()


def _check_answer(answer: str, accept: list[str]) -> bool:
    answer_lower = answer.lower().strip().rstrip(".")
    return any(acc.lower() in answer_lower for acc in accept)


def _ask_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    image_url: str | None = None,
) -> str:
    """Single LLM call, optionally with an image."""
    if image_url:
        content: list[dict] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        content = prompt  # type: ignore[assignment]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def run_part3(client: OpenAI, model: str) -> list[dict]:
    """Test grid comprehension: text-only vs text+image, side by side."""
    print(f"\n{HR}")
    print("PART 3: Grid representation comprehension (text vs text+image)")
    print(HR)

    encoder = StateEncoder()

    # Map each grid text block to its rendered image
    grid_images: dict[str, str] = {}
    grid_images[id(GRID_5x5)] = encoder.grid_to_image_data_url(GRID_5x5_ARRAY, cell_size=16)
    grid_images[id(GRID_7x7)] = encoder.grid_to_image_data_url(GRID_7x7_ARRAY, cell_size=16)
    # For diff questions, render before|after triptych
    grid_images[id(DIFF_TEXT)] = encoder.transition_image_data_url(
        GRID_5x5_ARRAY, GRID_5x5_AFTER_ARRAY, cell_size=16,
    )

    # Build grid-id lookup for each question
    grid_obj_ids = {}
    for q in COMPREHENSION_QS:
        if q["grid"] is GRID_5x5:
            grid_obj_ids[q["label"]] = id(GRID_5x5)
        elif q["grid"] is GRID_7x7:
            grid_obj_ids[q["label"]] = id(GRID_7x7)
        else:
            grid_obj_ids[q["label"]] = id(DIFF_TEXT)

    results = []
    cat_text: dict[str, list[bool]] = {}
    cat_visual: dict[str, list[bool]] = {}

    for q in COMPREHENSION_QS:
        cat = q["category"]
        if cat not in cat_text:
            cat_text[cat] = []
            cat_visual[cat] = []

        # --- Text-only ---
        prompt_t = COMPREHENSION_PROMPT_TEXT.format(grid=q["grid"], question=q["question"])
        raw_t = _ask_llm(client, model, prompt_t)
        ans_t = _extract_answer(raw_t)
        ok_t = _check_answer(ans_t, q["accept"])
        cat_text[cat].append(ok_t)

        # --- Text + Image ---
        img_url = grid_images.get(grid_obj_ids[q["label"]], "")
        prompt_v = COMPREHENSION_PROMPT_VISUAL.format(grid=q["grid"], question=q["question"])
        raw_v = _ask_llm(client, model, prompt_v, image_url=img_url if img_url else None)
        ans_v = _extract_answer(raw_v)
        ok_v = _check_answer(ans_v, q["accept"])
        cat_visual[cat].append(ok_v)

        tag_t = "OK" if ok_t else "X "
        tag_v = "OK" if ok_v else "X "

        results.append({
            "label": q["label"],
            "category": cat,
            "text_correct": ok_t,
            "visual_correct": ok_v,
            "text_answer": ans_t,
            "visual_answer": ans_v,
            "accepted": q["accept"],
        })

        print(f"\n  {q['label']}")
        print(f"    Q: {q['question']}")
        print(f"    text  [{tag_t}]: {ans_t[:80]}")
        print(f"    image [{tag_v}]: {ans_v[:80]}")
        print(f"    expected: {q['accept']}")

    # Category summary
    print(f"\n  {'─' * 60}")
    print(f"  {'Category':<15s}  {'Text':>8s}  {'Image':>8s}  {'Delta':>6s}")
    print(f"  {'─' * 60}")
    total_t = total_v = total_n = 0
    for cat in sorted(cat_text.keys()):
        nt = sum(cat_text[cat])
        nv = sum(cat_visual[cat])
        n = len(cat_text[cat])
        total_t += nt
        total_v += nv
        total_n += n
        delta = nv - nt
        sign = "+" if delta > 0 else ""
        print(f"    {cat:<15s}  {nt}/{n:>3d}      {nv}/{n:>3d}      {sign}{delta}")

    dt = total_v - total_t
    sign = "+" if dt > 0 else ""
    print(f"    {'TOTAL':<15s}  {total_t}/{total_n:>3d}      {total_v}/{total_n:>3d}      {sign}{dt}")
    print(f"\n  Text accuracy:  {100 * total_t / total_n:.0f}%")
    print(f"  Image accuracy: {100 * total_v / total_n:.0f}%")

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Connecting to VLLM: {VLLM_BASE_URL}  model: {VLLM_MODEL}")
    learner = make_learner()

    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    # Quick connectivity check
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"Available models: {available}")
        if VLLM_MODEL not in available:
            print(f"WARNING: {VLLM_MODEL} not in available models!")
    except Exception as e:
        print(f"WARNING: Could not list models ({e}). Proceeding anyway...")

    p1_results = run_part1(learner)
    p2_results = run_part2(learner)
    p3_results = run_part3(client, VLLM_MODEL)

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

    p3_text = sum(1 for r in p3_results if r["text_correct"])
    p3_img = sum(1 for r in p3_results if r["visual_correct"])
    n3 = len(p3_results)
    print(f"\nPart 3 — Grid comprehension: text={p3_text}/{n3}  image={p3_img}/{n3}")


if __name__ == "__main__":
    main()
