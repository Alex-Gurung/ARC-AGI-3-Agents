"""Interactive game player with live Observer/WM feedback.

Play the game in your terminal and see what the model observes after
each action. Lets you evaluate whether the model's descriptions match
what you actually see on screen.

Usage:
    uv run python scripts/interactive_observe.py --game <game_id>
    uv run python scripts/interactive_observe.py --game <game_id> --cell-sizes 4 8 16 32

Controls:
    1-5     Take ACTION1-ACTION5
    0 / r   RESET
    q       Quit
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory
from agents.templates.loop_agent.state_encoder import ARC_RGB_PALETTE, StateEncoder

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-3-4b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def render_grid(grid: list[list[int]]) -> None:
    """Render grid to terminal using half-block Unicode characters."""
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
            line += (
                f"\033[38;2;{top[0]};{top[1]};{top[2]}m"
                f"\033[48;2;{bot[0]};{bot[1]};{bot[2]}m\u2580\033[0m"
            )
        print(f"  {line}")


def convert_raw_frame(raw):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive game with live Observer feedback")
    parser.add_argument("--game", "-g", type=str, required=True, help="Game ID")
    parser.add_argument("--no-wm", action="store_true", help="Skip world model predictions")
    parser.add_argument(
        "--cell-sizes", "-c", type=int, nargs="+", default=None,
        help="Test multiple cell sizes (e.g. --cell-sizes 4 8 16 32). "
             "Runs observer at each size to compare quality.",
    )
    args = parser.parse_args()

    cell_sizes = args.cell_sizes or [16]

    from arc_agi import Arcade
    from arcengine import GameAction

    # Setup
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    learner = Learner(client, VLLM_MODEL)
    encoder = StateEncoder()
    memory = Memory(max_entries=50)

    print(f"{BOLD}VLLM:{RESET} {VLLM_BASE_URL}  model: {VLLM_MODEL}")
    print(f"{BOLD}Game:{RESET} {args.game}")
    if len(cell_sizes) > 1:
        print(f"{BOLD}Cell sizes:{RESET} {cell_sizes}")
    print(f"{DIM}Controls: 1-5 = ACTION1-5, 0/r = RESET, q = quit{RESET}\n")

    # Connect
    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=["interactive_observe"])
    env = arcade.make(args.game, scorecard_id=card_id)

    raw = env.reset()
    if raw is None:
        print("Could not reset game")
        return
    frame = convert_raw_frame(raw)
    grid = frame.frame[-1] if frame.frame else []
    state_text = encoder.encode(frame, force_keyframe=True)

    step = 0
    print(f"{BOLD}Step 0: RESET{RESET}  state={frame.state.name}  "
          f"grid={len(grid)}x{len(grid[0]) if grid else 0}")
    render_grid(grid)

    while True:
        # Show available actions
        available = frame.available_actions or [0, 1, 2, 3, 4, 5]
        action_names = [GameAction.from_id(a).name for a in available]
        print(f"\n{DIM}Available: {', '.join(action_names)}{RESET}")

        choice = input(f"{BOLD}> {RESET}").strip().lower()

        if choice in ("q", "quit"):
            break

        # Parse action
        try:
            if choice in ("r", "reset"):
                action = GameAction.from_id(0)
            elif choice.isdigit():
                action = GameAction.from_id(int(choice))
            else:
                action = GameAction.from_name(choice.upper())
        except Exception:
            print(f"  {RED}Invalid: {choice}{RESET}")
            continue

        if action.is_complex():
            try:
                coords = input("  x,y: ").strip().split(",")
                x, y = int(coords[0]), int(coords[1])
            except (ValueError, IndexError):
                print(f"  {RED}Invalid coordinates{RESET}")
                continue
            action.set_data({"game_id": args.game, "x": x, "y": y})
        else:
            action.set_data({"game_id": args.game})

        # Save before state
        grid_before = [row[:] for row in grid]
        state_before = state_text

        # --- World Model prediction (before executing) ---
        # WM uses the largest cell size for best quality
        wm_cell_size = max(cell_sizes)
        if not args.no_wm:
            img_before_wm = encoder.grid_to_image_data_url(grid_before, cell_size=wm_cell_size)
            grid_h = len(grid_before)
            grid_w = len(grid_before[0]) if grid_before else 0
            px_w, px_h = grid_w * wm_cell_size, grid_h * wm_cell_size
            print(f"\n{YELLOW}{BOLD}World Model predicts (cell_size={wm_cell_size}, {px_w}x{px_h}px):{RESET}")
            predicted = learner.predict_outcome(
                state_before=state_before,
                action_taken=action.name,
                memory=memory,
                image_before_url=img_before_wm,
            )
            print(f"  {YELLOW}{predicted}{RESET}")

        # Execute action
        raw = env.step(action, data=action.action_data.model_dump())
        if raw is None:
            print(f"  {RED}Step returned None{RESET}")
            continue

        step += 1
        frame = convert_raw_frame(raw)
        grid = frame.frame[-1] if frame.frame else []
        state_text = encoder.encode(frame, force_keyframe=True)
        diff_text = encoder.get_diff_text(grid_before, grid)
        num_changed = encoder.get_num_changed_cells(grid_before, grid)

        # Show the new state
        print(f"\n{BOLD}Step {step}: {action.name}{RESET}  "
              f"state={frame.state.name}  changed={num_changed} cells")
        render_grid(grid)

        # Show diff summary
        if num_changed == 0:
            print(f"  {DIM}No grid changes{RESET}")
        else:
            print(f"  {DIM}{diff_text}{RESET}")

        # --- Observer at each cell size ---
        grid_h = len(grid)
        grid_w = len(grid[0]) if grid else 0
        observer_results = []

        for cs in cell_sizes:
            px_w, px_h = grid_w * cs, grid_h * cs
            img_before = encoder.grid_to_image_data_url(grid_before, cell_size=cs)
            img_after = encoder.grid_to_image_data_url(grid, cell_size=cs)
            img_diff = encoder.transition_image_data_url(grid_before, grid, cell_size=cs)

            label = f"cell_size={cs} ({px_w}x{px_h}px)"
            print(f"\n{GREEN}{BOLD}Observer [{label}]:{RESET}")
            observed = learner.observe_transition(
                state_before=state_before,
                state_after=state_text,
                diff_text=diff_text,
                image_before_url=img_before,
                image_after_url=img_after,
                image_diff_url=img_diff,
            )
            print(f"  {GREEN}{observed}{RESET}")
            observer_results.append((cs, observed))

        # --- Judge scores if WM was used ---
        if not args.no_wm:
            for cs, observed in observer_results:
                similarity = learner.judge_similarity(predicted, observed)
                surprise = (6 - similarity) / 5.0
                bar = "█" * similarity + "░" * (5 - similarity)
                label = f"cell_size={cs}" if len(cell_sizes) > 1 else ""
                print(f"\n{BLUE}{BOLD}Judge{' [' + label + ']' if label else ''}:{RESET} "
                      f"similarity={similarity}/5  surprise={surprise:.2f}  [{bar}]")

        if frame.state.name in ("WIN", "GAME_OVER"):
            print(f"\n{BOLD}Game ended: {frame.state.name}{RESET}")
            break

    arcade.close_scorecard(card_id)
    print("\nDone.")


if __name__ == "__main__":
    main()
