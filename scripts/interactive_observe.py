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
import random
import sys
import time

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agi import Arcade
from arcengine import GameAction
from openai import OpenAI

from agents.templates.loop_agent.entity_registry import EntityRegistry
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


EXPLORE_PROMPT = """\
You are watching a short video of an 8-bit style puzzle game being played. \
Each frame shows the game grid after a different action was taken. The game \
is on a 64x64 pixel grid where each cell is one solid color.

Game elements are abstract — colored blocks, patterns, shapes, and \
indicators — not realistic objects. A single game element may be composed \
of multiple colors (e.g. a player with a colored head and body, a bordered \
region with a door of a different color, a patterned tile).

The actions taken between frames were:
{action_list}

Watch carefully how the grid changes between frames. Identify:
1. Which element is the PLAYER (the thing that moves in response to actions)? \
It may be a single color or a multi-color shape.
2. What are the static elements (walls, borders, background)? A border and \
its door may be different colors but form one element.
3. Are there any other dynamic elements (items, indicators, triggers)?

List the distinct game elements you can identify. Group colors that belong \
to the same element together:
ENTITIES:
- <element_role>: <brief description> (colors: <color(N)>, <color(M)>, ...)

Then summarize what you learned about the game mechanics.
ANSWER: <summary>
"""


def run_exploration(env, encoder, learner, entity_reg, game_id, n_actions, cell_size,
                    convert_fn):
    """Auto-play N random actions, build a video, ask model to ID elements."""
    grids = []
    actions_taken = []

    # Get initial state
    raw = env.reset()
    if raw is None:
        print(f"  {RED}Could not reset for exploration{RESET}")
        return raw
    frame = convert_fn(raw)
    grid = frame.frame[-1] if frame.frame else []
    grids.append([row[:] for row in grid])

    for i in range(n_actions):
        available = frame.available_actions or [1, 2, 3, 4, 5]
        # Pick a random non-reset action
        non_reset = [a for a in available if a != 0]
        if not non_reset:
            non_reset = [1]
        action_id = random.choice(non_reset)
        action = GameAction.from_id(action_id)
        action.set_data({"game_id": game_id})

        raw = env.step(action, data=action.action_data.model_dump())
        if raw is None:
            break
        frame = convert_fn(raw)
        grid = frame.frame[-1] if frame.frame else []
        grids.append([row[:] for row in grid])
        actions_taken.append(action.name)

        # If game ended, reset and keep going
        if frame.state.name in ("WIN", "GAME_OVER"):
            raw = env.reset()
            if raw is None:
                break
            frame = convert_fn(raw)
            grid = frame.frame[-1] if frame.frame else []
            grids.append([row[:] for row in grid])
            actions_taken.append("RESET")

    if len(grids) < 2:
        print(f"  {RED}Not enough frames for exploration{RESET}")
        return raw

    # Build video
    print(f"  {DIM}Building video from {len(grids)} frames...{RESET}")
    video_path = encoder.grid_sequence_to_video(grids, cell_size=cell_size)

    # Build action list text
    action_list = "\n".join(
        f"  Frame {i} -> Frame {i+1}: {a}" for i, a in enumerate(actions_taken)
    )

    prompt = EXPLORE_PROMPT.format(action_list=action_list)

    print(f"  {DIM}Asking model to identify game elements...{RESET}")
    video_ref = f"file://{video_path}"
    try:
        response = learner._call_llm(
            prompt,
            max_tokens=1024,
            temperature=0.3,
            video_url=video_ref,
        )
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass

    # Display response
    answer = learner._extract_answer(response)
    if response != answer and response:
        print(f"  {DIM}{response}{RESET}")
        print(f"  {GREEN}{BOLD}=> {answer}{RESET}")
    else:
        print(f"  {GREEN}{answer}{RESET}")

    # Update entity registry
    entity_reg.update_census(grid)
    raw_entities = Learner.extract_entities(response)
    if raw_entities:
        entity_reg.update_labels(raw_entities)
        # Give a confidence boost since this was a multi-frame analysis
        for color in raw_entities:
            if color in entity_reg._entries:
                e = entity_reg._entries[color]
                e.confidence = min(0.95, e.confidence + 0.15)
        print(f"\n{DIM}Entity Registry (after exploration):{RESET}")
        print(f"{DIM}{entity_reg.to_display_text()}{RESET}")

    # Reset game after exploration
    raw = env.reset()
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive game with live Observer feedback")
    parser.add_argument("--game", "-g", type=str, required=True, help="Game ID")
    parser.add_argument("--no-wm", action="store_true", help="Skip world model predictions")
    parser.add_argument(
        "--cell-sizes", "-c", type=int, nargs="+", default=None,
        help="Test multiple cell sizes (e.g. --cell-sizes 4 8 16 32). "
             "Runs observer at each size to compare quality.",
    )
    parser.add_argument(
        "--images", type=int, default=2, choices=[0, 1, 2, 3],
        help="Number of images to send: 0=text only, 1=AFTER only, "
             "2=BEFORE+AFTER (default), 3=BEFORE+AFTER+composite",
    )
    parser.add_argument(
        "--no-objects", action="store_true",
        help="Strip OBJECTS/RELATIONS from state text sent to all prompts",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Send BEFORE+AFTER as a video instead of separate images (requires Qwen3-VL)",
    )
    parser.add_argument(
        "--explore", type=int, default=0, metavar="N",
        help="Auto-play N random actions at start, build a video, and ask "
             "the model to identify game elements (seeds entity registry)",
    )
    args = parser.parse_args()

    cell_sizes = args.cell_sizes or [16]
    num_images = args.images
    strip_obj = Learner._strip_objects if args.no_objects else lambda s: s

    # Setup
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    learner = Learner(client, VLLM_MODEL)
    encoder = StateEncoder()
    memory = Memory(max_entries=50)
    entity_reg = EntityRegistry()

    print(f"{BOLD}VLLM:{RESET} {VLLM_BASE_URL}  model: {VLLM_MODEL}")
    print(f"{BOLD}Game:{RESET} {args.game}")
    if len(cell_sizes) > 1:
        print(f"{BOLD}Cell sizes:{RESET} {cell_sizes}")
    print(f"{DIM}Controls: 1-5 = ACTION1-5, 0/r = RESET, q = quit{RESET}")

    # Connect
    t0 = time.time()
    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=["interactive_observe"])
    env = arcade.make(args.game, scorecard_id=card_id)
    print(f"{DIM}Game connected in {time.time() - t0:.1f}s{RESET}\n")

    # --- Exploration phase (optional) ---
    if args.explore > 0:
        print(f"{BOLD}Exploring: {args.explore} random actions...{RESET}")
        raw = run_exploration(
            env, encoder, learner, entity_reg, args.game,
            n_actions=args.explore,
            cell_size=max(cell_sizes),
            convert_fn=convert_raw_frame,
        )
        if raw is None:
            raw = env.reset()
        if raw is None:
            print("Could not reset game after exploration")
            return
        frame = convert_raw_frame(raw)
        print(f"\n{BOLD}--- Entering interactive mode ---{RESET}")
    else:
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
                state_before=strip_obj(state_before),
                action_taken=action.name,
                memory=memory,
                image_before_url=img_before_wm,
            )
            raw = learner.last_raw_output
            if raw != predicted and raw:
                print(f"  {DIM}{raw}{RESET}")
                print(f"  {YELLOW}{BOLD}=> {predicted}{RESET}")
            else:
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

        known_el = entity_reg.to_prompt_text()

        if num_changed == 0:
            observed = "No changes occurred; the grid is identical before and after the action."
            print(f"\n{GREEN}{BOLD}Observer:{RESET}")
            print(f"  {GREEN}{observed}{RESET}")
            observer_results = [(cs, observed) for cs in cell_sizes]
        else:
            for cs in cell_sizes:
                px_w, px_h = grid_w * cs, grid_h * cs

                # Video mode: generate temp video instead of separate images
                video_path = None
                img_before = None
                img_after = None
                img_diff = None
                if args.video:
                    video_path = encoder.grid_sequence_to_video(
                        [grid_before, grid], cell_size=cs,
                    )
                    imgs_label = "video"
                else:
                    # 0=none, 1=AFTER only, 2=BEFORE+AFTER, 3=BEFORE+AFTER+composite
                    img_before = encoder.grid_to_image_data_url(grid_before, cell_size=cs) if num_images >= 2 else None
                    img_after = encoder.grid_to_image_data_url(grid, cell_size=cs) if num_images >= 1 else None
                    img_diff = encoder.transition_image_data_url(grid_before, grid, cell_size=cs) if num_images >= 3 else None
                    imgs_label = f"{num_images}img" if num_images > 0 else "text-only"

                label = f"cell_size={cs} ({px_w}x{px_h}px, {imgs_label})" if num_images > 0 or args.video else "text-only"
                print(f"\n{GREEN}{BOLD}Observer [{label}]:{RESET}")
                observed = learner.observe_transition(
                    state_before=strip_obj(state_before),
                    state_after=strip_obj(state_text),
                    diff_text=diff_text,
                    image_before_url=img_before,
                    image_after_url=img_after,
                    image_diff_url=img_diff,
                    video_url=video_path,
                    known_elements=known_el,
                )
                # Show full thinking (raw) then the extracted answer
                raw_out = learner.last_raw_output
                if raw_out != observed and raw_out:
                    print(f"  {DIM}{raw_out}{RESET}")
                    print(f"  {GREEN}{BOLD}=> {observed}{RESET}")
                else:
                    print(f"  {GREEN}{observed}{RESET}")
                observer_results.append((cs, observed))

                # Clean up temp video
                if video_path:
                    try:
                        os.unlink(video_path)
                    except OSError:
                        pass

        # Update entity registry from observer output
        entity_reg.update_census(grid)
        if observer_results:
            # Use the last observer result for entity extraction
            raw_entities = Learner.extract_entities(learner.last_raw_output)
            if raw_entities:
                entity_reg.update_labels(raw_entities)

        # Display entity registry
        reg_text = entity_reg.to_display_text()
        print(f"\n{DIM}Entity Registry:{RESET}")
        print(f"{DIM}{reg_text}{RESET}")

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
