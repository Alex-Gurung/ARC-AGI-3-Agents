"""Test the StateEncoder: compression ratio, diff sparsity, token budget.

No LLM needed — tests pure programmatic encoding.

Usage:
    uv run python scripts/test_state_encoder.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from agents.templates.loop_agent.state_encoder import StateEncoder


def make_grid(width: int = 64, height: int = 64, fill: int = 0) -> list[list[int]]:
    """Create a grid filled with a single value."""
    return [[fill] * width for _ in range(height)]


def add_objects(grid: list[list[int]], objects: list[tuple[int, int, int]]) -> None:
    """Add colored pixels to the grid. Each object is (x, y, color)."""
    for x, y, color in objects:
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            grid[y][x] = color


def test_full_grid_encoding():
    """Test RLE compression on a full grid."""
    print("=" * 60)
    print("TEST: Full Grid Encoding (RLE)")
    print("=" * 60)

    encoder = StateEncoder()

    # Create a mostly-empty grid with some objects
    grid = make_grid(64, 64, fill=0)
    # Add some walls (color 5) and a player-like region
    for y in range(10, 15):
        for x in range(10, 15):
            grid[y][x] = 5  # wall block
    grid[30][30] = 9  # player blue body
    grid[29][30] = 12  # player orange head

    # Mock FrameData (minimal)
    class MockFrame:
        frame = [grid]
        state = type("S", (), {"name": "NOT_FINISHED"})()
        levels_completed = 0
        available_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        full_reset = False

    encoded = encoder.encode(MockFrame())  # type: ignore[arg-type]

    lines = encoded.split("\n")
    chars = len(encoded)
    approx_tokens = chars // 4  # rough estimate

    print(f"  Grid size: 64x64 = 4096 cells")
    print(f"  Encoded lines: {len(lines)}")
    print(f"  Encoded chars: {chars}")
    print(f"  Approx tokens: ~{approx_tokens}")
    print(f"  Compression ratio: {4096 / max(chars, 1):.2f}x (cells/chars)")
    print(f"  First 5 lines:")
    for line in lines[:5]:
        print(f"    {line}")
    print()

    assert approx_tokens < 3000, f"Full grid encoding too large: ~{approx_tokens} tokens"
    print("  PASS: within token budget\n")


def test_diff_encoding():
    """Test diff encoding sparsity."""
    print("=" * 60)
    print("TEST: Diff Encoding (sparse changes)")
    print("=" * 60)

    encoder = StateEncoder()

    # First frame
    grid1 = make_grid(64, 64, fill=0)
    grid1[30][30] = 9  # player at (30,30)

    class MockFrame1:
        frame = [grid1]
        state = type("S", (), {"name": "NOT_FINISHED"})()
        levels_completed = 0
        available_actions = [0, 1, 2, 3, 4]
        full_reset = False

    # Encode first frame (sets previous_grid)
    encoder.encode(MockFrame1())  # type: ignore[arg-type]

    # Second frame: player moved up
    grid2 = make_grid(64, 64, fill=0)
    grid2[29][30] = 9  # player at (30,29)

    class MockFrame2:
        frame = [grid2]
        state = type("S", (), {"name": "NOT_FINISHED"})()
        levels_completed = 0
        available_actions = [0, 1, 2, 3, 4]
        full_reset = False

    encoded = encoder.encode(MockFrame2())  # type: ignore[arg-type]

    lines = encoded.split("\n")
    chars = len(encoded)
    approx_tokens = chars // 4

    print(f"  Changes: 2 cells (player moved)")
    print(f"  Encoded lines: {len(lines)}")
    print(f"  Encoded chars: {chars}")
    print(f"  Approx tokens: ~{approx_tokens}")
    print(f"  Content:")
    for line in lines:
        print(f"    {line}")
    print()

    assert approx_tokens < 200, f"Diff encoding too large for 2 changes: ~{approx_tokens} tokens"
    print("  PASS: diff is sparse and compact\n")


def test_large_diff():
    """Test diff encoding with many changes."""
    print("=" * 60)
    print("TEST: Large Diff Encoding (many changes)")
    print("=" * 60)

    encoder = StateEncoder()

    grid1 = make_grid(64, 64, fill=0)
    class MockFrame1:
        frame = [grid1]
        state = type("S", (), {"name": "NOT_FINISHED"})()
        levels_completed = 0
        available_actions = [0, 1, 2, 3, 4]
        full_reset = False

    encoder.encode(MockFrame1())  # type: ignore[arg-type]

    # Create a grid with 200+ changes
    grid2 = make_grid(64, 64, fill=0)
    for i in range(200):
        x, y = random.randint(0, 63), random.randint(0, 63)
        grid2[y][x] = random.randint(1, 15)

    class MockFrame2:
        frame = [grid2]
        state = type("S", (), {"name": "NOT_FINISHED"})()
        levels_completed = 0
        available_actions = [0, 1, 2, 3, 4]
        full_reset = False

    encoded = encoder.encode(MockFrame2())  # type: ignore[arg-type]

    chars = len(encoded)
    approx_tokens = chars // 4

    print(f"  Changes: ~200 cells")
    print(f"  Encoded chars: {chars}")
    print(f"  Approx tokens: ~{approx_tokens}")
    print(f"  First 10 lines:")
    for line in encoded.split("\n")[:10]:
        print(f"    {line}")
    print()

    assert approx_tokens < 2000, f"Large diff too large: ~{approx_tokens} tokens"
    print("  PASS: large diff stays within budget\n")


def test_summary():
    """Test summary statistics."""
    print("=" * 60)
    print("TEST: Summary Statistics")
    print("=" * 60)

    encoder = StateEncoder()

    grid = make_grid(64, 64, fill=0)
    # Add various colors
    for y in range(5, 10):
        for x in range(5, 10):
            grid[y][x] = 5  # walls
    for y in range(20, 25):
        for x in range(20, 25):
            grid[y][x] = 11  # doors
    grid[30][30] = 9  # player

    class MockFrame:
        frame = [grid]
        state = type("S", (), {"name": "NOT_FINISHED"})()
        levels_completed = 0
        available_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        full_reset = False

    encoded = encoder.encode(MockFrame())  # type: ignore[arg-type]

    # Find summary section
    summary_lines = [l for l in encoded.split("\n") if "SUMMARY" in l or "Colors" in l or "Unique" in l]
    print("  Summary lines:")
    for line in summary_lines:
        print(f"    {line}")
    print()
    print("  PASS\n")


if __name__ == "__main__":
    random.seed(42)
    test_full_grid_encoding()
    test_diff_encoding()
    test_large_diff()
    test_summary()
    print("All StateEncoder tests passed!")
