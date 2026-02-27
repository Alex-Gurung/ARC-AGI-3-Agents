"""State encoder for the LoopAgent.

Converts game grid data to compressed text representation.
Purely structural — no game-specific assumptions about what
colors mean or what specific rows represent.
"""

import logging
from collections import Counter
from typing import Optional

from arcengine import FrameData

logger = logging.getLogger(__name__)


class StateEncoder:
    """Encodes game state as compressed text for LLM context.

    Three layers:
    1. Base: RLE-compressed grid (first frame) or diff (subsequent frames)
    2. Summary: Color histogram, region info, dimensions
    3. Game metadata: state, score, available actions
    """

    def __init__(self, keyframe_interval: int = 10) -> None:
        self.previous_grid: Optional[list[list[int]]] = None
        self.step_count: int = 0
        self.keyframe_interval = max(1, keyframe_interval)
        self.frames_since_keyframe: int = 0
        self._force_next_keyframe: bool = False
        self._forced_reason: Optional[str] = None

    def reset(self) -> None:
        """Reset encoder state (e.g., on game reset)."""
        self.previous_grid = None
        self.step_count = 0
        self.frames_since_keyframe = 0
        self._force_next_keyframe = False
        self._forced_reason = None

    def force_keyframe_next(self, reason: str = "manual") -> None:
        """Force the next encode() to emit a full keyframe."""
        self._force_next_keyframe = True
        self._forced_reason = reason

    def encode(
        self,
        frame: FrameData,
        force_keyframe: bool = False,
        keyframe_reason: Optional[str] = None,
    ) -> str:
        """Encode a FrameData into compressed text.

        Returns compact text suitable for LLM context (~500-2000 tokens).
        """
        parts: list[str] = []

        # Game metadata
        parts.append(self._encode_metadata(frame))

        # Grid data
        current_grid = frame.frame[-1] if frame.frame else []
        periodic_keyframe = (
            self.previous_grid is not None
            and self.frames_since_keyframe >= self.keyframe_interval
        )
        should_emit_keyframe = (
            self.previous_grid is None
            or force_keyframe
            or self._force_next_keyframe
            or periodic_keyframe
        )

        if not current_grid:
            parts.append("GRID: empty")
        elif should_emit_keyframe:
            reason = (
                keyframe_reason
                or self._forced_reason
                or ("periodic" if periodic_keyframe else "initial")
            )
            parts.append(f"KEYFRAME: {reason}")
            parts.append(self._encode_full_grid(current_grid))
            self.frames_since_keyframe = 0
            self._force_next_keyframe = False
            self._forced_reason = None
        else:
            # Subsequent frames: diff only
            diff = self._compute_diff(self.previous_grid, current_grid)
            if diff:
                parts.append(self._encode_diff(diff))
            else:
                parts.append("GRID: no changes")
            self.frames_since_keyframe += 1

        # Summary statistics
        if current_grid:
            parts.append(self._encode_summary(current_grid))

        # Store for next diff
        self.previous_grid = [row[:] for row in current_grid] if current_grid else None
        self.step_count += 1

        return "\n".join(parts)

    def encode_for_diff(self, frame_before: FrameData, frame_after: FrameData) -> str:
        """Encode the diff between two specific frames."""
        grid_before = frame_before.frame[-1] if frame_before.frame else []
        grid_after = frame_after.frame[-1] if frame_after.frame else []

        parts: list[str] = []
        parts.append(self._encode_metadata(frame_after))

        if not grid_before or not grid_after:
            parts.append("DIFF: cannot compute (missing grid)")
            return "\n".join(parts)

        diff = self._compute_diff(grid_before, grid_after)
        if diff:
            parts.append(self._encode_diff(diff))
        else:
            parts.append("DIFF: no changes")

        parts.append(self._encode_summary(grid_after))
        return "\n".join(parts)

    def _encode_metadata(self, frame: FrameData) -> str:
        """Encode game metadata."""
        parts = [
            f"STATE: {frame.state.name}",
            f"LEVELS_COMPLETED: {frame.levels_completed}",
        ]
        if frame.available_actions:
            action_names = [f"ACTION{a}" if a > 0 else "RESET" for a in frame.available_actions]
            parts.append(f"AVAILABLE_ACTIONS: {', '.join(action_names)}")
        if frame.full_reset:
            parts.append("FULL_RESET: true")
        return "\n".join(parts)

    def _encode_full_grid(self, grid: list[list[int]]) -> str:
        """RLE-compress the full grid.

        Uses run-length encoding per row: color*count pairs.
        E.g., row [0,0,0,5,5,0,0] -> "0*3 5*2 0*2"
        """
        if not grid:
            return "GRID: empty"

        height = len(grid)
        width = len(grid[0]) if grid else 0

        lines = [f"GRID ({width}x{height}):"]

        for y, row in enumerate(grid):
            encoded_row = self._rle_encode_row(row)
            lines.append(f"  r{y}: {encoded_row}")

        return "\n".join(lines)

    def _rle_encode_row(self, row: list[int]) -> str:
        """Run-length encode a single row."""
        if not row:
            return ""

        runs: list[str] = []
        current_val = row[0]
        count = 1

        for val in row[1:]:
            if val == current_val:
                count += 1
            else:
                runs.append(f"{current_val}*{count}" if count > 1 else str(current_val))
                current_val = val
                count = 1

        runs.append(f"{current_val}*{count}" if count > 1 else str(current_val))
        return " ".join(runs)

    def _compute_diff(
        self, grid_before: list[list[int]], grid_after: list[list[int]]
    ) -> list[tuple[int, int, int, int]]:
        """Compute changed cells between two grids.

        Returns list of (x, y, old_value, new_value) sorted by (y, x).
        """
        changes: list[tuple[int, int, int, int]] = []

        height_before = len(grid_before)
        height_after = len(grid_after)
        max_height = max(height_before, height_after)

        for y in range(max_height):
            row_before = grid_before[y] if y < height_before else []
            row_after = grid_after[y] if y < height_after else []
            max_width = max(len(row_before), len(row_after))

            for x in range(max_width):
                val_before = row_before[x] if x < len(row_before) else -1
                val_after = row_after[x] if x < len(row_after) else -1
                if val_before != val_after:
                    changes.append((x, y, val_before, val_after))

        return changes

    def _encode_diff(self, diff: list[tuple[int, int, int, int]]) -> str:
        """Encode a diff as compact text."""
        lines = [f"CHANGED ({len(diff)} cells):"]

        # If too many changes, summarize
        if len(diff) > 100:
            # Group by change type
            change_types: Counter[tuple[int, int]] = Counter()
            for x, y, old, new in diff:
                change_types[(old, new)] += 1
            lines.append("  (large diff, summarized)")
            for (old, new), count in change_types.most_common(20):
                lines.append(f"  {old}->{new}: {count} cells")
        else:
            for x, y, old, new in diff:
                lines.append(f"  ({x},{y}):{old}->{new}")

        return "\n".join(lines)

    def _encode_summary(self, grid: list[list[int]]) -> str:
        """Encode structural summary statistics."""
        if not grid:
            return "SUMMARY: empty grid"

        height = len(grid)
        width = len(grid[0]) if grid else 0

        # Color histogram
        color_counts: Counter[int] = Counter()
        for row in grid:
            for val in row:
                color_counts[val] += 1

        # Format as compact histogram
        histogram_parts = []
        for color, count in sorted(color_counts.items()):
            histogram_parts.append(f"{color}:{count}")

        lines = [
            f"SUMMARY: {width}x{height} grid",
            f"  Colors: {{{', '.join(histogram_parts)}}}",
            f"  Unique colors: {len(color_counts)}",
        ]

        return "\n".join(lines)

    def get_diff_text(
        self, grid_before: list[list[int]], grid_after: list[list[int]]
    ) -> str:
        """Get just the diff text between two grids (for learner prompts)."""
        diff = self._compute_diff(grid_before, grid_after)
        if not diff:
            return "no changes"
        return self._encode_diff(diff)

    def get_num_changed_cells(
        self, grid_before: list[list[int]], grid_after: list[list[int]]
    ) -> int:
        """Count the number of changed cells between two grids."""
        return len(self._compute_diff(grid_before, grid_after))
