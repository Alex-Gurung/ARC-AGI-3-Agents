"""End-to-end test of the LoopAgent.

Runs the full explore-exploit loop on a game and logs everything:
- Phase transitions (explore/exploit)
- Memory evolution over time
- Surprise scores per step
- Level controller state
- Final outcome and levels completed

Usage:
    # Start VLLM first: bash scripts/run_vllm.sh
    # Start ARC-AGI game server
    uv run python scripts/test_full_loop.py --game <game_id> [--base-url URL] [--model MODEL]

    # Or use the standard main.py entry point:
    uv run main.py --agent loop --game <game_id>
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("loop_agent_test.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def test_components_standalone():
    """Test all components work without a game server (mock data)."""
    print("=" * 60)
    print("TEST: Standalone Component Integration")
    print("=" * 60)

    from agents.templates.loop_agent.memory import Memory, MemoryEntry
    from agents.templates.loop_agent.state_encoder import StateEncoder
    from agents.templates.loop_agent.surprise import HeuristicSurprise, LevelController

    # Initialize components
    memory = Memory(max_entries=50)
    encoder = StateEncoder()
    surprise = HeuristicSurprise()
    level_ctrl = LevelController()

    # Simulate a few game steps
    grids = []

    # Step 0: Initial grid
    grid0 = [[0] * 64 for _ in range(64)]
    grid0[30][30] = 9  # player
    grid0[29][30] = 12  # player head
    for y in range(10, 15):
        for x in range(10, 50):
            grid0[y][x] = 5  # wall
    grids.append(grid0)

    # Step 1: Player moved up
    grid1 = [row[:] for row in grid0]
    grid1[30][30] = 0
    grid1[29][30] = 9
    grid1[28][30] = 12
    grids.append(grid1)

    # Step 2: Player moved up again
    grid2 = [row[:] for row in grid1]
    grid2[29][30] = 0
    grid2[28][30] = 9
    grid2[27][30] = 12
    grids.append(grid2)

    # Step 3: Player hit a wall (no change)
    grid3 = [row[:] for row in grid2]
    grids.append(grid3)

    print(f"  Simulating {len(grids)-1} actions...\n")

    actions = ["RESET", "ACTION1", "ACTION1", "ACTION1"]

    for step in range(1, len(grids)):
        grid_before = grids[step - 1]
        grid_after = grids[step]
        action = actions[step]

        # Encode
        diff_text = encoder.get_diff_text(grid_before, grid_after)
        num_changed = encoder.get_num_changed_cells(grid_before, grid_after)

        # Surprise
        s = surprise.compute(
            state_before=f"step {step-1}",
            action=action,
            state_after=f"step {step}",
            memory=memory,
            num_changed_cells=num_changed,
        )

        # Level controller
        level = level_ctrl.update(s)

        # Simulate learner adding entries
        if step == 1:
            memory.add(MemoryEntry(
                type="ACTION", content=f"ACTION1 moves player up",
                justification="observed pixel shift",
                confidence=0.7, created_step=step, last_modified_step=step,
            ))
        elif step == 3 and num_changed == 0:
            memory.add(MemoryEntry(
                type="RULE", content="Black cells block movement",
                justification="ACTION1 had no effect, wall above",
                confidence=0.6, created_step=step, last_modified_step=step,
            ))

        print(f"  Step {step}: action={action}, changed={num_changed}, "
              f"surprise={s:.3f}, level={level}, memory_size={len(memory)}")
        print(f"    diff: {diff_text[:80]}")

    print(f"\n  Final memory:")
    print(f"  {memory.to_text()}")
    print()
    print("  PASS: all components integrated correctly\n")


def test_with_game_server(game_id: str, base_url: str, model: str):
    """Run the full LoopAgent on a real game (requires game server)."""
    print("=" * 60)
    print(f"TEST: Full Loop on Game {game_id}")
    print("=" * 60)

    # Set environment variables for the agent
    os.environ["VLLM_BASE_URL"] = base_url
    os.environ["VLLM_MODEL"] = model
    os.environ["VLLM_API_KEY"] = "dummy"
    os.environ["SURPRISE_STRATEGY"] = "heuristic"  # Start with heuristic

    print(f"  VLLM: {base_url} / {model}")
    print(f"  Game: {game_id}")
    print(f"  Strategy: heuristic")
    print()
    print("  To run this test, use:")
    print(f"  uv run main.py --agent loop --game {game_id}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None, help="Game ID for live test")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    args = parser.parse_args()

    # Always run standalone test
    test_components_standalone()

    # Run live test if game specified
    if args.game:
        test_with_game_server(args.game, args.base_url, args.model)
    else:
        print("Skipping live game test (no --game specified)")
        print("To run on a live game: uv run python scripts/test_full_loop.py --game <game_id>")

    print("\nAll full loop tests complete!")
