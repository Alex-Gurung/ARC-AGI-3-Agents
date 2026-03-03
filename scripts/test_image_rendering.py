"""Quick visual check of image rendering at different cell sizes.

Connects to a game, takes one step, and saves the BEFORE, AFTER, and
transition composite images at each requested cell size to output/img_test/.

Usage:
    uv run python scripts/test_image_rendering.py --game <game_id>
    uv run python scripts/test_image_rendering.py --game <game_id> --cell-sizes 4 8 16 32
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.templates.loop_agent.state_encoder import ARC_RGB_PALETTE, StateEncoder


def data_url_to_png(data_url: str, path: Path) -> None:
    """Decode a data:image/png;base64,... URL and save to disk."""
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        print(f"  WARNING: unexpected data URL prefix for {path}")
        return
    raw = base64.b64decode(data_url[len(prefix):])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    print(f"  Saved {path}  ({len(raw)} bytes)")


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
    parser = argparse.ArgumentParser(description="Visual check of image rendering")
    parser.add_argument("--game", "-g", type=str, required=True, help="Game ID")
    parser.add_argument(
        "--cell-sizes", "-c", type=int, nargs="+", default=[4, 8, 16, 32],
        help="Cell sizes to test (default: 4 8 16 32)",
    )
    parser.add_argument("--action", "-a", type=int, default=1,
                        help="Action ID to take (default: 1 = ACTION1)")
    parser.add_argument("--outdir", "-o", type=str, default="output/img_test",
                        help="Output directory for images")
    args = parser.parse_args()

    from arc_agi import Arcade
    from arcengine import GameAction

    outdir = Path(args.outdir)
    encoder = StateEncoder()

    # Connect and reset
    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=["img_test"])
    env = arcade.make(args.game, scorecard_id=card_id)

    raw = env.reset()
    if raw is None:
        print("Could not reset game")
        return
    frame = convert_raw_frame(raw)
    grid_before = frame.frame[-1] if frame.frame else []

    h = len(grid_before)
    w = len(grid_before[0]) if grid_before else 0
    print(f"Game: {args.game}  grid: {h}x{w}")

    # Print color palette for reference
    print(f"\nColor palette (values present in grid):")
    vals = set()
    for row in grid_before:
        vals.update(row)
    for v in sorted(vals):
        rgb = ARC_RGB_PALETTE.get(v, (128, 128, 128))
        print(f"  {v:>2}: RGB{rgb}")

    # Take one action
    action = GameAction.from_id(args.action)
    if action.is_complex():
        action.set_data({"game_id": args.game, "x": w // 2, "y": h // 2})
    else:
        action.set_data({"game_id": args.game})

    raw = env.step(action, data=action.action_data.model_dump())
    if raw is None:
        print("Step returned None")
        arcade.close_scorecard(card_id)
        return
    frame = convert_raw_frame(raw)
    grid_after = frame.frame[-1] if frame.frame else []

    print(f"\nAction: {action.name}")
    num_changed = encoder.get_num_changed_cells(grid_before, grid_after)
    print(f"Changed cells: {num_changed}")

    # Save images at each cell size
    for cs in args.cell_sizes:
        px_w, px_h = w * cs, h * cs
        print(f"\n--- cell_size={cs}  ({px_w}x{px_h}px) ---")

        before_url = encoder.grid_to_image_data_url(grid_before, cell_size=cs)
        after_url = encoder.grid_to_image_data_url(grid_after, cell_size=cs)
        trans_url = encoder.transition_image_data_url(grid_before, grid_after, cell_size=cs)

        data_url_to_png(before_url, outdir / f"before_cs{cs}.png")
        data_url_to_png(after_url, outdir / f"after_cs{cs}.png")
        data_url_to_png(trans_url, outdir / f"transition_cs{cs}.png")

    arcade.close_scorecard(card_id)
    print(f"\nAll images saved to {outdir}/")
    print("Open them to verify colors and layout match what you see in the game.")


if __name__ == "__main__":
    main()
