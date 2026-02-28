"""Test visual comprehension on real game frames.

Connects to a game, takes random actions (or lets you play),
captures actual screenshots, and probes the model with scene-level
questions:
  - "Describe what you see" (open-ended scene comprehension)
  - "What changed?" (before/after diff comprehension)
  - "What do you think would happen if...?" (prediction)

Usage:
    # Random play, 15 steps, then probe:
    uv run python scripts/test_visual_comprehension.py --game <game_id>

    # Interactive (you choose actions):
    uv run python scripts/test_visual_comprehension.py --game <game_id> --interactive

    # Just probe saved frames (skip game connection):
    uv run python scripts/test_visual_comprehension.py --from-dir output/game_xyz
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

from openai import OpenAI
from PIL import Image

from agents.templates.loop_agent.state_encoder import ARC_RGB_PALETTE, StateEncoder

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-3-4b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")

HR = "=" * 70


# ──────────────────────────────────────────────────────────────────────
# Game connection helpers
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


def play_random(game_id: str, num_steps: int = 15) -> list[dict]:
    """Connect to game, take random actions, return captured frames."""
    from arc_agi import Arcade
    from arcengine import GameAction

    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=["visual_test"])
    env = arcade.make(game_id, scorecard_id=card_id)

    raw = env.reset()
    if raw is None:
        raise RuntimeError(f"Could not reset game {game_id}")
    frame = convert_raw_frame(raw)

    encoder = StateEncoder()
    captures = []

    grid = frame.frame[-1] if frame.frame else []
    state_text = encoder.encode(frame, force_keyframe=True)
    captures.append({
        "step": 0,
        "action": "RESET",
        "grid": grid,
        "state_text": state_text,
        "game_state": frame.state.name,
    })
    print(f"  step 0: RESET  state={frame.state.name}  grid={len(grid)}x{len(grid[0]) if grid else 0}")

    for step in range(1, num_steps + 1):
        available = frame.available_actions or [1, 2, 3, 4, 5]
        # Avoid RESET (0) most of the time
        non_reset = [a for a in available if a != 0]
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
        state_text = encoder.encode(frame)
        captures.append({
            "step": step,
            "action": action.name,
            "grid": grid,
            "state_text": state_text,
            "game_state": frame.state.name,
        })
        print(f"  step {step}: {action.name}  state={frame.state.name}  levels={frame.levels_completed}")

        if frame.state.name in ("WIN", "GAME_OVER"):
            break

    arcade.close_scorecard(card_id)
    return captures


def play_interactive(game_id: str) -> list[dict]:
    """Interactive mode: user chooses actions."""
    from arc_agi import Arcade
    from arcengine import GameAction

    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=["visual_test_interactive"])
    env = arcade.make(game_id, scorecard_id=card_id)

    raw = env.reset()
    if raw is None:
        raise RuntimeError(f"Could not reset game {game_id}")
    frame = convert_raw_frame(raw)

    encoder = StateEncoder()
    captures = []

    grid = frame.frame[-1] if frame.frame else []
    state_text = encoder.encode(frame, force_keyframe=True)
    captures.append({
        "step": 0,
        "action": "RESET",
        "grid": grid,
        "state_text": state_text,
        "game_state": frame.state.name,
    })

    # Save and show initial frame
    _save_and_show_grid(grid, 0)

    step = 0
    while True:
        available = frame.available_actions or [0, 1, 2, 3, 4, 5]
        action_names = [GameAction.from_id(a).name for a in available]
        print(f"\n  Available: {', '.join(action_names)}")
        choice = input("  Action (name, number, or 'q' to finish): ").strip()

        if choice.lower() in ("q", "quit", "done"):
            break

        try:
            if choice.isdigit():
                action = GameAction.from_id(int(choice))
            else:
                action = GameAction.from_name(choice.upper())
        except Exception:
            print(f"  Invalid action: {choice}")
            continue

        if action.is_complex():
            try:
                coords = input("  x,y coordinates: ").strip().split(",")
                x, y = int(coords[0]), int(coords[1])
            except (ValueError, IndexError):
                print("  Invalid coordinates")
                continue
            action.set_data({"game_id": game_id, "x": x, "y": y})
        else:
            action.set_data({"game_id": game_id})

        raw = env.step(action, data=action.action_data.model_dump())
        if raw is None:
            print("  Step returned None")
            continue

        step += 1
        frame = convert_raw_frame(raw)
        grid = frame.frame[-1] if frame.frame else []
        state_text = encoder.encode(frame)
        captures.append({
            "step": step,
            "action": action.name,
            "grid": grid,
            "state_text": state_text,
            "game_state": frame.state.name,
        })

        _save_and_show_grid(grid, step)
        print(f"  state={frame.state.name}  levels={frame.levels_completed}")

        if frame.state.name in ("WIN", "GAME_OVER"):
            print(f"  Game ended: {frame.state.name}")
            break

    arcade.close_scorecard(card_id)
    return captures


def _save_and_show_grid(grid: list[list[int]], step: int) -> None:
    """Render grid to terminal using half-block characters."""
    if not grid:
        return
    for y in range(0, len(grid), 2):
        line = ""
        for x in range(len(grid[0])):
            top = ARC_RGB_PALETTE.get(grid[y][x], (128, 128, 128))
            if y + 1 < len(grid):
                bot = ARC_RGB_PALETTE.get(grid[y + 1][x], (128, 128, 128))
            else:
                bot = (0, 0, 0)
            # Upper half block with fg=top, bg=bot
            line += f"\033[38;2;{top[0]};{top[1]};{top[2]}m\033[48;2;{bot[0]};{bot[1]};{bot[2]}m\u2580\033[0m"
        print(f"  {line}")


# ──────────────────────────────────────────────────────────────────────
# LLM probes
# ──────────────────────────────────────────────────────────────────────

def ask_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    image_urls: list[str] | None = None,
    max_tokens: int = 512,
) -> str:
    """Call model with text + optional images."""
    if image_urls:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        content = prompt  # type: ignore[assignment]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


DESCRIBE_PROMPT = """\
You are looking at a screenshot of a grid-based game. Describe what you see.

Focus on:
- What distinct objects or regions are visible (shapes, clusters, isolated cells)
- What colors they are and roughly where they are (center, top-left, border, etc.)
- Any obvious structure (walls, paths, borders, patterns)
- What you think the player might control or interact with

Keep it to 3-5 sentences. Be concrete — describe what you actually see, not what you guess the game might be."""

DIFF_PROMPT = """\
You are looking at two screenshots of a grid-based game: BEFORE and AFTER an action was taken.

The action taken was: {action}

Describe what changed between the two frames:
- What moved, appeared, or disappeared?
- Did any object change position or color?
- Did the overall structure change?
- What do you think the action did?

Keep it to 3-5 sentences. Be specific about what you actually see changed."""

PREDICT_PROMPT = """\
You are looking at a screenshot of a grid-based game.

The available actions are simple directional-style commands (ACTION1 through ACTION5) plus RESET.

Based on what you see, what do you think would happen if the player takes {action}?

Describe your prediction in 2-3 sentences. Focus on what you expect to visually change on the grid."""


def probe_scene_description(
    client: OpenAI, model: str, encoder: StateEncoder,
    captures: list[dict], indices: list[int],
) -> list[dict]:
    """Ask model to describe individual frames."""
    results = []
    for idx in indices:
        cap = captures[idx]
        img_url = encoder.grid_to_image_data_url(cap["grid"], cell_size=16)
        if not img_url:
            continue

        print(f"\n  --- Frame {cap['step']} (after {cap['action']}) ---")
        response = ask_llm(client, model, DESCRIBE_PROMPT, image_urls=[img_url])
        print(textwrap.indent(response, "    "))

        results.append({
            "type": "describe",
            "step": cap["step"],
            "action": cap["action"],
            "response": response,
        })
    return results


def probe_diff_description(
    client: OpenAI, model: str, encoder: StateEncoder,
    captures: list[dict], pairs: list[tuple[int, int]],
) -> list[dict]:
    """Ask model to describe what changed between pairs of frames."""
    results = []
    for before_idx, after_idx in pairs:
        cap_before = captures[before_idx]
        cap_after = captures[after_idx]

        img_before = encoder.grid_to_image_data_url(cap_before["grid"], cell_size=16)
        img_after = encoder.grid_to_image_data_url(cap_after["grid"], cell_size=16)
        img_diff = encoder.transition_image_data_url(
            cap_before["grid"], cap_after["grid"], cell_size=16,
        )

        if not img_before or not img_after:
            continue

        action = cap_after["action"]
        prompt = DIFF_PROMPT.format(action=action)

        print(f"\n  --- Diff: frame {cap_before['step']} → {cap_after['step']} ({action}) ---")

        # Try with triptych (before|after|diff)
        images = [img_before, img_after]
        if img_diff:
            images.append(img_diff)

        response = ask_llm(client, model, prompt, image_urls=images)
        print(textwrap.indent(response, "    "))

        results.append({
            "type": "diff",
            "step_before": cap_before["step"],
            "step_after": cap_after["step"],
            "action": action,
            "response": response,
        })
    return results


def probe_prediction(
    client: OpenAI, model: str, encoder: StateEncoder,
    captures: list[dict], probes: list[tuple[int, str]],
) -> list[dict]:
    """Ask model to predict what will happen, then compare with reality."""
    results = []
    for frame_idx, action_name in probes:
        cap = captures[frame_idx]
        img_url = encoder.grid_to_image_data_url(cap["grid"], cell_size=16)
        if not img_url:
            continue

        prompt = PREDICT_PROMPT.format(action=action_name)
        print(f"\n  --- Predict: frame {cap['step']} + {action_name} ---")

        prediction = ask_llm(client, model, prompt, image_urls=[img_url])
        print(f"    [Prediction]")
        print(textwrap.indent(prediction, "      "))

        # If the next frame exists and used this action, show what actually happened
        next_idx = frame_idx + 1
        if next_idx < len(captures) and captures[next_idx]["action"] == action_name:
            cap_after = captures[next_idx]
            img_after = encoder.grid_to_image_data_url(cap_after["grid"], cell_size=16)
            diff_prompt = DIFF_PROMPT.format(action=action_name)
            actual = ask_llm(
                client, model, diff_prompt,
                image_urls=[img_url, img_after] if img_after else [img_url],
            )
            print(f"    [What actually happened]")
            print(textwrap.indent(actual, "      "))
            results.append({
                "type": "predict",
                "step": cap["step"],
                "action": action_name,
                "prediction": prediction,
                "actual": actual,
            })
        else:
            results.append({
                "type": "predict",
                "step": cap["step"],
                "action": action_name,
                "prediction": prediction,
                "actual": None,
            })

    return results


# ──────────────────────────────────────────────────────────────────────
# Save/load captures
# ──────────────────────────────────────────────────────────────────────

def save_captures(captures: list[dict], output_dir: Path) -> None:
    """Save captured frames as images + metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = StateEncoder()

    meta = []
    for cap in captures:
        grid = cap["grid"]
        if grid:
            img = encoder._render_grid_image(grid)
            img_upscaled = img.resize(
                (img.width * 8, img.height * 8),
                resample=Image.Resampling.NEAREST,
            )
            img_path = output_dir / f"frame_{cap['step']:03d}.png"
            img_upscaled.save(str(img_path))

        meta.append({
            "step": cap["step"],
            "action": cap["action"],
            "game_state": cap["game_state"],
            "grid_size": f"{len(grid)}x{len(grid[0])}" if grid and grid[0] else "0x0",
        })

    with open(output_dir / "captures.json", "w") as f:
        json.dump({"frames": meta, "num_frames": len(captures)}, f, indent=2)

    print(f"\n  Saved {len(captures)} frames to {output_dir}/")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test visual comprehension on real game frames"
    )
    parser.add_argument("--game", "-g", type=str, help="Game ID to play")
    parser.add_argument("--steps", "-n", type=int, default=15, help="Random steps to take")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--from-dir", type=str, help="Load saved captures instead of playing")
    parser.add_argument("--save-dir", type=str, help="Save frames to this directory")
    parser.add_argument("--skip-play", action="store_true", help="Only probe, don't play")
    args = parser.parse_args()

    if not args.game and not args.from_dir:
        parser.error("Either --game or --from-dir is required")

    # Connect to VLLM
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    encoder = StateEncoder()
    print(f"VLLM: {VLLM_BASE_URL}  model: {VLLM_MODEL}")

    # Get frames
    if args.from_dir:
        print(f"\nLoading frames from {args.from_dir}...")
        # TODO: reload grid data from saved PNGs
        print("  (--from-dir not yet implemented, use --game)")
        return

    print(f"\nPlaying game: {args.game}")
    if args.interactive:
        captures = play_interactive(args.game)
    else:
        captures = play_random(args.game, num_steps=args.steps)

    if not captures:
        print("No frames captured!")
        return

    # Save frames
    save_dir = Path(args.save_dir or f"output/visual_test_{args.game}")
    save_captures(captures, save_dir)

    # ── Probes ──
    print(f"\n{HR}")
    print("PROBING MODEL WITH GAME SCREENSHOTS")
    print(HR)

    # 1. Describe scenes — first frame, a mid frame, last frame
    print(f"\n{'─' * 60}")
    print("Scene descriptions (image only — no text state)")
    print(f"{'─' * 60}")

    desc_indices = [0]  # always first
    if len(captures) > 3:
        desc_indices.append(len(captures) // 2)  # middle
    if len(captures) > 1:
        desc_indices.append(len(captures) - 1)  # last
    desc_results = probe_scene_description(client, VLLM_MODEL, encoder, captures, desc_indices)

    # 2. Diff descriptions — pick pairs where grids actually changed
    print(f"\n{'─' * 60}")
    print("Diff descriptions (before/after image pairs)")
    print(f"{'─' * 60}")

    diff_pairs = []
    for i in range(1, len(captures)):
        if captures[i]["grid"] != captures[i - 1]["grid"]:
            diff_pairs.append((i - 1, i))
        if len(diff_pairs) >= 4:
            break

    if not diff_pairs and len(captures) > 1:
        # No grid changes found, just use first pair
        diff_pairs = [(0, 1)]

    diff_results = probe_diff_description(client, VLLM_MODEL, encoder, captures, diff_pairs)

    # 3. Predictions — pick a frame and ask what would happen
    print(f"\n{'─' * 60}")
    print("Predictions (what would happen if...?)")
    print(f"{'─' * 60}")

    predict_probes = []
    for i in range(min(3, len(captures) - 1)):
        next_action = captures[i + 1]["action"]
        predict_probes.append((i, next_action))

    predict_results = probe_prediction(client, VLLM_MODEL, encoder, captures, predict_probes)

    # Summary
    print(f"\n{HR}")
    print("SUMMARY")
    print(HR)
    print(f"  Frames captured:       {len(captures)}")
    print(f"  Scene descriptions:    {len(desc_results)}")
    print(f"  Diff descriptions:     {len(diff_results)}")
    print(f"  Predictions:           {len(predict_results)}")
    print(f"  Frames saved to:       {save_dir}/")


if __name__ == "__main__":
    main()
