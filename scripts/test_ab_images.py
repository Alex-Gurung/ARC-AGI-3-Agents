"""A/B test: image+text vs text-only through the WM→Observer→Judge pipeline.

Connects to a real game, takes random actions, then runs each transition
through the full pipeline twice:
  A) Text + images (the new image-aware prompts)
  B) Text only (no images passed)

Compares similarity scores to measure whether images help the pipeline.

Usage:
    uv run python scripts/test_ab_images.py --game <game_id>
    uv run python scripts/test_ab_images.py --game <game_id> --steps 20
    uv run python scripts/test_ab_images.py --game <game_id> --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory
from agents.templates.loop_agent.state_encoder import StateEncoder

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-3-4b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")

HR = "=" * 70
THIN = "─" * 60


# ──────────────────────────────────────────────────────────────────────
# Game connection (reused from test_visual_comprehension.py)
# ──────────────────────────────────────────────────────────────────────

def convert_raw_frame(raw):
    """Convert FrameDataRaw → FrameData."""
    from arcengine import FrameData
    return FrameData(
        game_id=raw.game_id,
        frame=[arr.tolist() for arr in raw.frame],
        state=raw.state,
        levels_completed=raw.levels_completed,
        win_levels=raw.win_levels,
        guid=raw.guid,
        full_reset=raw.full_reset,
        available_actions=raw.available_actions,
    )


def collect_transitions(game_id: str, num_steps: int = 10, interactive: bool = False) -> list[dict]:
    """Play the game and collect (before, action, after) transition triples."""
    from arc_agi import Arcade
    from arcengine import GameAction

    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=["ab_test"])
    env = arcade.make(game_id, scorecard_id=card_id)

    raw = env.reset()
    if raw is None:
        raise RuntimeError(f"Could not reset game {game_id}")
    frame = convert_raw_frame(raw)

    encoder = StateEncoder()
    grid = frame.frame[-1] if frame.frame else []
    state_text = encoder.encode(frame, force_keyframe=True)

    transitions = []
    prev = {
        "grid": [row[:] for row in grid],
        "state_text": state_text,
        "frame": frame,
    }

    print(f"  step 0: RESET  grid={len(grid)}x{len(grid[0]) if grid else 0}")

    for step in range(1, num_steps + 1):
        available = frame.available_actions or [1, 2, 3, 4, 5]
        non_reset = [a for a in available if a != 0]

        if interactive:
            action_names = [GameAction.from_id(a).name for a in available]
            print(f"\n  Available: {', '.join(action_names)}")
            choice = input("  Action (name, number, or 'q'): ").strip()
            if choice.lower() in ("q", "quit", "done"):
                break
            try:
                if choice.isdigit():
                    action = GameAction.from_id(int(choice))
                else:
                    action = GameAction.from_name(choice.upper())
            except Exception:
                print(f"  Invalid: {choice}")
                continue
        else:
            action_id = random.choice(non_reset if non_reset else available)
            action = GameAction.from_id(action_id)

        if action.is_complex():
            grid_h = len(grid) if grid else 30
            grid_w = len(grid[0]) if grid and grid[0] else 30
            action.set_data({
                "game_id": game_id,
                "x": random.randint(0, min(63, grid_w - 1)),
                "y": random.randint(0, min(63, grid_h - 1)),
            })
        else:
            action.set_data({"game_id": game_id})

        raw = env.step(action, data=action.action_data.model_dump())
        if raw is None:
            print(f"  step {step}: {action.name} returned None, stopping")
            break

        frame = convert_raw_frame(raw)
        grid = frame.frame[-1] if frame.frame else []
        # Force keyframe so each transition has full state
        state_text = encoder.encode(frame, force_keyframe=True)

        print(f"  step {step}: {action.name}  state={frame.state.name}")

        transitions.append({
            "step": step,
            "action": action.name,
            "grid_before": prev["grid"],
            "grid_after": [row[:] for row in grid],
            "state_before": prev["state_text"],
            "state_after": state_text,
        })

        prev = {
            "grid": [row[:] for row in grid],
            "state_text": state_text,
            "frame": frame,
        }

        if frame.state.name in ("WIN", "GAME_OVER"):
            break

    arcade.close_scorecard(card_id)
    return transitions


# ──────────────────────────────────────────────────────────────────────
# A/B pipeline runner
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    learner: Learner,
    encoder: StateEncoder,
    memory: Memory,
    transition: dict,
    use_images: bool,
) -> dict:
    """Run WM→Observer→Judge on a single transition.

    Returns dict with predicted, observed, similarity, and raw outputs.
    """
    grid_before = transition["grid_before"]
    grid_after = transition["grid_after"]
    state_before = transition["state_before"]
    state_after = transition["state_after"]
    action = transition["action"]

    diff_text = encoder.get_diff_text(grid_before, grid_after)

    if use_images:
        img_before = encoder.grid_to_image_data_url(grid_before, cell_size=16)
        img_after = encoder.grid_to_image_data_url(grid_after, cell_size=16)
        img_diff = encoder.transition_image_data_url(grid_before, grid_after, cell_size=16)
    else:
        img_before = None
        img_after = None
        img_diff = None

    # 1. World Model
    predicted = learner.predict_outcome(
        state_before=state_before,
        action_taken=action,
        memory=memory,
        image_before_url=img_before,
    )
    wm_raw = learner.last_raw_output

    # 2. Observer
    observed = learner.observe_transition(
        state_before=state_before,
        state_after=state_after,
        diff_text=diff_text,
        image_before_url=img_before,
        image_after_url=img_after,
        image_diff_url=img_diff,
    )
    obs_raw = learner.last_raw_output

    # 3. Judge
    similarity = learner.judge_similarity(predicted, observed)
    surprise = (6 - similarity) / 5.0

    return {
        "predicted": predicted,
        "observed": observed,
        "similarity": similarity,
        "surprise": surprise,
        "wm_raw": wm_raw,
        "obs_raw": obs_raw,
    }


def run_ab_test(transitions: list[dict], max_transitions: int = 8) -> dict:
    """Run A/B test on collected transitions."""
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    learner = Learner(client, VLLM_MODEL)
    encoder = StateEncoder()

    # Shared empty memory (no learned rules yet — tests raw comprehension)
    memory = Memory(max_entries=50)

    # Filter to transitions where the grid actually changed
    changed = [t for t in transitions if t["grid_before"] != t["grid_after"]]
    if not changed:
        print("  No grid-changing transitions found, using all transitions")
        changed = transitions

    subset = changed[:max_transitions]
    print(f"\n  Testing {len(subset)} transitions ({len(changed)} had grid changes)")

    results_a = []  # image + text
    results_b = []  # text only

    for i, trans in enumerate(subset):
        print(f"\n{THIN}")
        print(f"  Transition {i + 1}/{len(subset)}: step {trans['step']}, {trans['action']}")
        print(THIN)

        # --- Condition A: image + text ---
        print("\n  [A] Image + Text:")
        result_a = run_pipeline(learner, encoder, memory, trans, use_images=True)
        results_a.append(result_a)

        print("    WM prediction:")
        print(textwrap.indent(result_a["predicted"], "      "))
        print("    Observer:")
        print(textwrap.indent(result_a["observed"], "      "))
        print(f"    Similarity:    {result_a['similarity']}/5  (surprise={result_a['surprise']:.2f})")

        # --- Condition B: text only ---
        print("\n  [B] Text Only:")
        result_b = run_pipeline(learner, encoder, memory, trans, use_images=False)
        results_b.append(result_b)

        print("    WM prediction:")
        print(textwrap.indent(result_b["predicted"], "      "))
        print("    Observer:")
        print(textwrap.indent(result_b["observed"], "      "))
        print(f"    Similarity:    {result_b['similarity']}/5  (surprise={result_b['surprise']:.2f})")

        delta = result_a["similarity"] - result_b["similarity"]
        winner = "IMAGE" if delta > 0 else ("TEXT" if delta < 0 else "TIE")
        print(f"\n    >>> {winner}  (delta={delta:+d})")

    return {"results_a": results_a, "results_b": results_b, "transitions": subset}


def print_summary(results: dict) -> None:
    """Print aggregate summary of A/B results."""
    results_a = results["results_a"]
    results_b = results["results_b"]
    n = len(results_a)

    if n == 0:
        print("  No results to summarize")
        return

    sims_a = [r["similarity"] for r in results_a]
    sims_b = [r["similarity"] for r in results_b]
    surprises_a = [r["surprise"] for r in results_a]
    surprises_b = [r["surprise"] for r in results_b]

    avg_sim_a = sum(sims_a) / n
    avg_sim_b = sum(sims_b) / n
    avg_surp_a = sum(surprises_a) / n
    avg_surp_b = sum(surprises_b) / n

    wins_a = sum(1 for a, b in zip(sims_a, sims_b) if a > b)
    wins_b = sum(1 for a, b in zip(sims_a, sims_b) if b > a)
    ties = sum(1 for a, b in zip(sims_a, sims_b) if a == b)

    print(f"\n{HR}")
    print("A/B TEST RESULTS")
    print(HR)
    print(f"  Transitions tested:    {n}")
    print()
    print(f"  {'Metric':<25} {'[A] Image+Text':>15} {'[B] Text Only':>15}")
    print(f"  {'─' * 55}")
    print(f"  {'Avg similarity (1-5)':<25} {avg_sim_a:>15.2f} {avg_sim_b:>15.2f}")
    print(f"  {'Avg surprise (0-1)':<25} {avg_surp_a:>15.2f} {avg_surp_b:>15.2f}")
    print(f"  {'Min similarity':<25} {min(sims_a):>15d} {min(sims_b):>15d}")
    print(f"  {'Max similarity':<25} {max(sims_a):>15d} {max(sims_b):>15d}")
    print()
    print(f"  Head-to-head: Image wins {wins_a}, Text wins {wins_b}, Ties {ties}")
    print()

    # Per-transition breakdown
    print(f"  {'Step':<8} {'Action':<12} {'[A] sim':>8} {'[B] sim':>8} {'Winner':>8}")
    print(f"  {'─' * 44}")
    for i, trans in enumerate(results["transitions"]):
        sa = sims_a[i]
        sb = sims_b[i]
        w = "IMG" if sa > sb else ("TXT" if sb > sa else "TIE")
        print(f"  {trans['step']:<8} {trans['action']:<12} {sa:>8d} {sb:>8d} {w:>8}")

    # Show detailed outputs for the most interesting cases
    print(f"\n{'─' * 60}")
    print("DETAILED OUTPUTS (largest image vs text divergence)")
    print(f"{'─' * 60}")

    deltas = [(abs(sims_a[i] - sims_b[i]), i) for i in range(n)]
    deltas.sort(reverse=True)

    shown = 0
    for delta, idx in deltas:
        if delta == 0:
            break
        if shown >= 3:
            break

        trans = results["transitions"][idx]
        ra = results_a[idx]
        rb = results_b[idx]

        print(f"\n  Step {trans['step']}: {trans['action']}  "
              f"(Image={ra['similarity']}/5, Text={rb['similarity']}/5)")

        print("\n  [A] Image+Text WM prediction:")
        print(textwrap.indent(ra["predicted"], "      "))
        print("\n  [A] Image+Text Observer:")
        print(textwrap.indent(ra["observed"], "      "))

        print("\n  [B] Text-only WM prediction:")
        print(textwrap.indent(rb["predicted"], "      "))
        print("\n  [B] Text-only Observer:")
        print(textwrap.indent(rb["observed"], "      "))

        shown += 1

    if shown == 0:
        # All ties — still show one full example
        if n > 0:
            trans = results["transitions"][0]
            ra = results_a[0]
            rb = results_b[0]
            print(f"\n  Step {trans['step']}: {trans['action']}  "
                  f"(Image={ra['similarity']}/5, Text={rb['similarity']}/5)  [TIE]")

            print("\n  [A] Image+Text WM prediction:")
            print(textwrap.indent(ra["predicted"], "      "))
            print("\n  [A] Image+Text Observer:")
            print(textwrap.indent(ra["observed"], "      "))

            print("\n  [B] Text-only WM prediction:")
            print(textwrap.indent(rb["predicted"], "      "))
            print("\n  [B] Text-only Observer:")
            print(textwrap.indent(rb["observed"], "      "))


def save_results(results: dict, output_path: Path) -> None:
    """Save full results to JSON for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "num_transitions": len(results["transitions"]),
        "results_a": [
            {
                "predicted": r["predicted"],
                "observed": r["observed"],
                "similarity": r["similarity"],
                "surprise": r["surprise"],
            }
            for r in results["results_a"]
        ],
        "results_b": [
            {
                "predicted": r["predicted"],
                "observed": r["observed"],
                "similarity": r["similarity"],
                "surprise": r["surprise"],
            }
            for r in results["results_b"]
        ],
        "transitions": [
            {
                "step": t["step"],
                "action": t["action"],
            }
            for t in results["transitions"]
        ],
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Full results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B test: image+text vs text-only pipeline"
    )
    parser.add_argument("--game", "-g", type=str, required=True, help="Game ID")
    parser.add_argument("--steps", "-n", type=int, default=10, help="Steps to play")
    parser.add_argument("--max-transitions", "-m", type=int, default=8,
                        help="Max transitions to test")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Choose actions manually")
    parser.add_argument("--save", "-s", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print(f"VLLM: {VLLM_BASE_URL}  model: {VLLM_MODEL}")

    # Phase 1: Collect game transitions
    print(f"\n{HR}")
    print(f"PHASE 1: Collecting transitions from game {args.game}")
    print(HR)

    transitions = collect_transitions(
        args.game,
        num_steps=args.steps,
        interactive=args.interactive,
    )

    if not transitions:
        print("  No transitions collected!")
        return

    print(f"\n  Collected {len(transitions)} transitions")

    # Phase 2: Run A/B pipeline
    print(f"\n{HR}")
    print("PHASE 2: Running A/B pipeline comparison")
    print(HR)

    results = run_ab_test(transitions, max_transitions=args.max_transitions)

    # Phase 3: Summary
    print_summary(results)

    # Save if requested
    save_path = Path(args.save) if args.save else Path(f"output/ab_test_{args.game}.json")
    save_results(results, save_path)


if __name__ == "__main__":
    main()
