"""State encoder for the LoopAgent.

Converts game grid data to compressed text representation.
Purely structural — no game-specific assumptions about what
colors mean or what specific rows represent.
"""

import base64
import io
import logging
from collections import Counter
from typing import Optional

from arcengine import FrameData
from PIL import Image

logger = logging.getLogger(__name__)

ARC_RGB_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
    10: (255, 255, 255),
    11: (104, 195, 163),
    12: (230, 126, 34),
    13: (142, 68, 173),
    14: (39, 174, 96),
    15: (22, 160, 133),
}

ARC_COLOR_NAMES: dict[int, str] = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "pink",
    7: "orange",
    8: "light_blue",
    9: "dark_red",
    10: "white",
    11: "teal",
    12: "dark_orange",
    13: "purple",
    14: "dark_green",
    15: "dark_teal",
}


def color_label(val: int) -> str:
    """Return a human-readable label for a color value, e.g. 'green(3)'."""
    name = ARC_COLOR_NAMES.get(val)
    if name:
        return f"{name}({val})"
    return str(val)


def _bbox_iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    """Intersection-over-union of two (min_x, min_y, max_x, max_y) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
        # Compact object/relation representation (game-agnostic).
        self.object_max_count: int = 16
        self.relation_max_count: int = 30
        self.object_min_area: int = 1
        # Stable object tracking across frames.
        self._prev_objects: list[dict] = []
        self._next_object_id: int = 0

    def reset(self) -> None:
        """Reset encoder state (e.g., on game reset)."""
        self.previous_grid = None
        self.step_count = 0
        self.frames_since_keyframe = 0
        self._force_next_keyframe = False
        self._forced_reason = None
        self._prev_objects = []
        self._next_object_id = 0

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
            parts.append(self._encode_object_representation(current_grid))

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

    def grid_to_image_data_url(
        self,
        grid: list[list[int]],
        cell_size: int = 8,
    ) -> str:
        """Render a grid to a PNG data URL for multimodal LLM calls."""
        if not grid or not grid[0]:
            return ""

        image = self._render_grid_image(grid)
        return self._image_to_data_url(image=image, cell_size=cell_size)

    def transition_image_data_url(
        self,
        grid_before: list[list[int]],
        grid_after: list[list[int]],
        cell_size: int = 8,
    ) -> str:
        """Render a visual transition as BEFORE | AFTER | REMOVED | ADDED.

        REMOVED panel shows the old color of changed cells (what disappeared).
        ADDED panel shows the new color of changed cells (what appeared).
        Unchanged cells are dim gray in both panels.
        """
        if (not grid_before or not grid_before[0]) and (not grid_after or not grid_after[0]):
            return ""

        height = max(len(grid_before), len(grid_after))
        width = max(
            len(grid_before[0]) if grid_before and grid_before[0] else 0,
            len(grid_after[0]) if grid_after and grid_after[0] else 0,
        )
        if width == 0 or height == 0:
            return ""

        before_norm = self._normalize_grid(grid_before, width=width, height=height)
        after_norm = self._normalize_grid(grid_after, width=width, height=height)
        before_image = self._render_grid_image(before_norm)
        after_image = self._render_grid_image(after_norm)

        removed_image = Image.new("RGB", (width, height))
        added_image = Image.new("RGB", (width, height))
        rem_px = removed_image.load()
        add_px = added_image.load()
        if rem_px is None or add_px is None:
            return ""

        dim = (22, 22, 22)
        for y in range(height):
            for x in range(width):
                old_val = before_norm[y][x]
                new_val = after_norm[y][x]
                if old_val == new_val:
                    rem_px[x, y] = dim
                    add_px[x, y] = dim
                    continue
                # REMOVED: show old color (what was there before)
                or_, og, ob = ARC_RGB_PALETTE.get(old_val, (255, 255, 255))
                rem_px[x, y] = (
                    min(255, or_ + 40),
                    min(255, og + 40),
                    min(255, ob + 40),
                )
                # ADDED: show new color (what appeared)
                nr, ng, nb = ARC_RGB_PALETTE.get(new_val, (255, 255, 255))
                add_px[x, y] = (
                    min(255, nr + 40),
                    min(255, ng + 40),
                    min(255, nb + 40),
                )

        separator = 1
        num_panels = 4
        composite = Image.new(
            "RGB",
            (width * num_panels + separator * (num_panels - 1), height),
            (0, 0, 0),
        )
        composite.paste(before_image, (0, 0))
        composite.paste(after_image, (width + separator, 0))
        composite.paste(removed_image, (2 * (width + separator), 0))
        composite.paste(added_image, (3 * (width + separator), 0))

        # Draw separators
        px = composite.load()
        if px is not None:
            for s in range(1, num_panels):
                sep_x = s * width + s * separator - 1
                for y in range(height):
                    px[sep_x, y] = (235, 235, 235)

        return self._image_to_data_url(image=composite, cell_size=cell_size)

    def grid_sequence_to_video(
        self,
        grids: list[list[list[int]]],
        cell_size: int = 16,
        fps: float = 24.0,
        hold_seconds: float = 0.5,
    ) -> str:
        """Render a sequence of grids as video frames, return temp file path.

        Uses imageio+ffmpeg to create an MP4 file. Each grid is held for
        hold_seconds at the given fps, producing a natural-looking video.

        Args:
            grids: List of grids (each is list[list[int]])
            cell_size: Pixels per cell (upscale factor)
            fps: Frames per second (24.0 = standard video framerate)
            hold_seconds: How long each grid state is shown (0.5 = half second)

        Returns:
            Path to temporary .mp4 file. Caller is responsible for cleanup.
        """
        import tempfile

        import imageio.v3 as iio
        import numpy as np

        if not grids:
            return ""

        # Each grid is held for hold_seconds at the given fps
        repeats_per_frame = max(1, round(fps * hold_seconds))
        frames = []
        for grid in grids:
            img = self._render_grid_image(grid)
            if cell_size > 1:
                img = img.resize(
                    (img.width * cell_size, img.height * cell_size),
                    resample=Image.Resampling.NEAREST,
                )
            arr = np.array(img)
            for _ in range(repeats_per_frame):
                frames.append(arr)

        fd, path = tempfile.mkstemp(suffix=".mp4")
        try:
            iio.imwrite(
                path,
                np.stack(frames),
                fps=fps,
                codec="libx264",
                plugin="pyav",
            )
        finally:
            import os

            os.close(fd)
        return path

    def frame_to_image_data_url(
        self,
        frame: FrameData,
        cell_size: int = 8,
    ) -> str:
        """Render the latest frame grid to a PNG data URL."""
        grid = frame.frame[-1] if frame.frame else []
        return self.grid_to_image_data_url(grid=grid, cell_size=cell_size)

    @staticmethod
    def _normalize_grid(
        grid: list[list[int]],
        *,
        width: int,
        height: int,
        fill: int = 0,
    ) -> list[list[int]]:
        out = [[fill for _ in range(width)] for _ in range(height)]
        for y in range(min(height, len(grid))):
            row = grid[y]
            for x in range(min(width, len(row))):
                out[y][x] = row[x]
        return out

    def _render_grid_image(self, grid: list[list[int]]) -> Image.Image:
        """Render a grid into a non-upscaled PIL RGB image."""
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        image = Image.new("RGB", (width, height))
        px = image.load()
        if px is None:
            return image
        for y in range(height):
            row = grid[y]
            for x in range(width):
                px[x, y] = ARC_RGB_PALETTE.get(row[x], (128, 128, 128))
        return image

    @staticmethod
    def _image_to_data_url(image: Image.Image, cell_size: int) -> str:
        """Resize nearest-neighbor and encode image as PNG data URL."""
        cell_size = max(1, int(cell_size))
        if cell_size > 1:
            image = image.resize(
                (image.width * cell_size, image.height * cell_size),
                resample=Image.Resampling.NEAREST,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

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

    def _encode_diff(
        self, diff: list[tuple[int, int, int, int]], max_clusters: int = 20
    ) -> str:
        """Encode a diff as clustered, color-named text.

        Groups adjacent cells with the same (old, new) color change into
        spatial clusters, then describes each cluster compactly.
        Shows up to *max_clusters* largest clusters; summarizes the rest.
        """
        if not diff:
            return "CHANGED (0 cells): no changes"

        clusters = self._cluster_diff(diff)
        lines = [f"CHANGED ({len(diff)} cells, {len(clusters)} clusters):"]

        shown = clusters[:max_clusters]
        omitted = clusters[max_clusters:]

        for cluster in shown:
            old_c, new_c = cluster["old"], cluster["new"]
            cells = cluster["cells"]
            n = len(cells)
            cx, cy = cluster["center"]
            min_x, min_y = cluster["min_x"], cluster["min_y"]
            max_x, max_y = cluster["max_x"], cluster["max_y"]

            old_label = color_label(old_c)
            new_label = color_label(new_c)

            if n == 1:
                lines.append(f"  ({cx},{cy}): {old_label} -> {new_label}")
            else:
                span = f"({min_x},{min_y})-({max_x},{max_y})"
                lines.append(
                    f"  {n} cells near ({cx},{cy}) [{span}]: "
                    f"{old_label} -> {new_label}"
                )

        if omitted:
            omitted_cells = sum(len(c["cells"]) for c in omitted)
            # Summarize omitted by change type
            type_counts: Counter[tuple[int, int]] = Counter()
            for c in omitted:
                type_counts[(c["old"], c["new"])] += len(c["cells"])
            parts = []
            for (old_c, new_c), count in type_counts.most_common(5):
                parts.append(f"{color_label(old_c)}->{color_label(new_c)}: {count}")
            lines.append(
                f"  ... {len(omitted)} more clusters ({omitted_cells} cells): "
                + ", ".join(parts)
            )

        return "\n".join(lines)

    @staticmethod
    def _cluster_diff(
        diff: list[tuple[int, int, int, int]],
    ) -> list[dict]:
        """Cluster diff cells by (old, new) color pair and spatial adjacency.

        Uses flood-fill on 4-connected neighbors that share the same
        (old, new) change type. Returns list of cluster dicts sorted by
        size descending.
        """
        # Group by change type
        by_type: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for x, y, old, new in diff:
            key = (old, new)
            if key not in by_type:
                by_type[key] = set()
            by_type[key].add((x, y))

        clusters: list[dict] = []

        for (old_c, new_c), positions in by_type.items():
            remaining = set(positions)
            while remaining:
                # Flood-fill one cluster
                seed = next(iter(remaining))
                component: list[tuple[int, int]] = []
                stack = [seed]
                while stack:
                    pos = stack.pop()
                    if pos not in remaining:
                        continue
                    remaining.discard(pos)
                    component.append(pos)
                    px, py = pos
                    for nx, ny in [(px - 1, py), (px + 1, py),
                                   (px, py - 1), (px, py + 1)]:
                        if (nx, ny) in remaining:
                            stack.append((nx, ny))

                xs = [p[0] for p in component]
                ys = [p[1] for p in component]
                clusters.append({
                    "old": old_c,
                    "new": new_c,
                    "cells": component,
                    "center": (round(sum(xs) / len(xs)), round(sum(ys) / len(ys))),
                    "min_x": min(xs),
                    "min_y": min(ys),
                    "max_x": max(xs),
                    "max_y": max(ys),
                })

        clusters.sort(key=lambda c: len(c["cells"]), reverse=True)
        return clusters

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

    def _encode_object_representation(self, grid: list[list[int]]) -> str:
        """Encode a compact object table + relation graph.

        Objects are connected components (4-neighbor) on non-background colors.
        Background is detected from border pixels. Objects have stable IDs
        across frames via color + bbox IoU matching.
        """
        if not grid or not grid[0]:
            return "OBJECTS: none"

        objects, background_color, total_components = self._extract_objects(grid)
        if not objects:
            return f"OBJECTS (bg={background_color}): none"

        selected = objects[: self.object_max_count]
        omitted = max(0, len(objects) - len(selected))

        lines = [
            (
                f"OBJECTS ({len(selected)} shown, total={total_components}, bg={background_color}):"
            )
        ]
        lines.append(
            "  NOTE: object IDs/relations are heuristic inferences from connected components; "
            "verify with GRID/DIFF (and image when available)."
        )
        for obj in selected:
            min_x, min_y, max_x, max_y = obj["bbox"]
            lines.append(
                "  "
                f"{obj['id']}: c={obj['color']} area={obj['area']} "
                f"bbox=({min_x},{min_y})-({max_x},{max_y}) "
                f"fill={obj['fill']:.2f} aspect={obj['aspect']:.2f} "
                f"center=({obj['cx']:.1f},{obj['cy']:.1f})"
            )
        if omitted > 0:
            lines.append(f"  ... {omitted} smaller objects omitted")

        relations = self._build_relations(selected)
        if not relations:
            lines.append("RELATIONS: none")
            return "\n".join(lines)

        lines.append(f"RELATIONS ({len(relations)}):")
        for rel in relations:
            lines.append(f"  {rel}")
        return "\n".join(lines)

    def _detect_background(self, grid: list[list[int]], height: int, width: int) -> int:
        """Detect background color from border pixels.

        Uses the most frequent color on the grid border (top/bottom rows,
        left/right columns). Falls back to global most-frequent for tiny grids.
        """
        if height < 3 or width < 3:
            # Tiny grid — fall back to global most-frequent.
            color_counts: Counter[int] = Counter()
            for row in grid:
                for color in row:
                    color_counts[color] += 1
            return max(color_counts.items(), key=lambda kv: kv[1])[0]

        border_colors: Counter[int] = Counter()
        for x in range(width):
            border_colors[grid[0][x]] += 1
            border_colors[grid[height - 1][x]] += 1
        for y in range(1, height - 1):
            border_colors[grid[y][0]] += 1
            border_colors[grid[y][width - 1]] += 1
        return border_colors.most_common(1)[0][0]

    def _extract_objects(
        self, grid: list[list[int]]
    ) -> tuple[list[dict[str, float | int | str | tuple[int, int, int, int]]], int, int]:
        """Extract connected components as objects and return sorted by area desc.

        Assigns stable IDs by matching to previous frame's objects via color + bbox IoU.
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        if width == 0:
            return [], 0, 0

        background_color = self._detect_background(grid, height, width)

        visited = [[False] * width for _ in range(height)]
        objects: list[dict[str, float | int | str | tuple[int, int, int, int]]] = []

        for y in range(height):
            for x in range(width):
                if visited[y][x]:
                    continue
                color = grid[y][x]
                if color == background_color:
                    visited[y][x] = True
                    continue

                # Flood-fill same-color 4-neighbor component.
                stack = [(x, y)]
                visited[y][x] = True
                area = 0
                sum_x = 0
                sum_y = 0
                min_x = x
                max_x = x
                min_y = y
                max_y = y

                while stack:
                    cx, cy = stack.pop()
                    area += 1
                    sum_x += cx
                    sum_y += cy
                    if cx < min_x:
                        min_x = cx
                    if cx > max_x:
                        max_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cy > max_y:
                        max_y = cy

                    if cx > 0 and not visited[cy][cx - 1] and grid[cy][cx - 1] == color:
                        visited[cy][cx - 1] = True
                        stack.append((cx - 1, cy))
                    if cx + 1 < width and not visited[cy][cx + 1] and grid[cy][cx + 1] == color:
                        visited[cy][cx + 1] = True
                        stack.append((cx + 1, cy))
                    if cy > 0 and not visited[cy - 1][cx] and grid[cy - 1][cx] == color:
                        visited[cy - 1][cx] = True
                        stack.append((cx, cy - 1))
                    if cy + 1 < height and not visited[cy + 1][cx] and grid[cy + 1][cx] == color:
                        visited[cy + 1][cx] = True
                        stack.append((cx, cy + 1))

                if area < self.object_min_area:
                    continue

                bbox_w = max_x - min_x + 1
                bbox_h = max_y - min_y + 1
                bbox_area = bbox_w * bbox_h

                objects.append(
                    {
                        "id": "",
                        "color": color,
                        "area": area,
                        "bbox": (min_x, min_y, max_x, max_y),
                        "cx": (sum_x / area),
                        "cy": (sum_y / area),
                        "fill": round(area / bbox_area, 2) if bbox_area > 0 else 1.0,
                        "aspect": round(bbox_w / bbox_h, 2) if bbox_h > 0 else 1.0,
                    }
                )

        objects.sort(key=lambda o: int(o["area"]), reverse=True)
        self._match_objects(objects)
        self._prev_objects = objects
        return objects, background_color, len(objects)

    def _match_objects(
        self, new_objects: list[dict], max_center_dist: float = 5.0
    ) -> None:
        """Assign stable IDs by matching to previous frame's objects.

        Matches by same color + nearest center distance (up to *max_center_dist*).
        """
        if not self._prev_objects:
            for obj in new_objects:
                obj["id"] = f"O{self._next_object_id}"
                self._next_object_id += 1
            return

        used_prev: set[int] = set()
        for obj in new_objects:
            best_dist = max_center_dist
            best_prev_id: Optional[str] = None
            best_pi = -1
            for pi, prev in enumerate(self._prev_objects):
                if pi in used_prev or prev["color"] != obj["color"]:
                    continue
                dist = abs(float(obj["cx"]) - float(prev["cx"])) + abs(
                    float(obj["cy"]) - float(prev["cy"])
                )
                if dist < best_dist:
                    best_dist = dist
                    best_prev_id = str(prev["id"])
                    best_pi = pi
            if best_prev_id is not None:
                obj["id"] = best_prev_id
                used_prev.add(best_pi)
            else:
                obj["id"] = f"O{self._next_object_id}"
                self._next_object_id += 1

    def _build_relations(
        self,
        objects: list[dict[str, float | int | str | tuple[int, int, int, int]]],
    ) -> list[str]:
        """Build spatial relations between objects.

        Includes proximity (touching/near), direction (above/below/left/right),
        and containment (inside).
        """
        relations_with_score: list[tuple[int, str]] = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj_a = objects[i]
                obj_b = objects[j]
                ax1, ay1, ax2, ay2 = obj_a["bbox"]  # type: ignore[assignment]
                bx1, by1, bx2, by2 = obj_b["bbox"]  # type: ignore[assignment]

                # Check containment (smaller inside larger).
                a_inside_b = ax1 >= bx1 and ax2 <= bx2 and ay1 >= by1 and ay2 <= by2
                b_inside_a = bx1 >= ax1 and bx2 <= ax2 and by1 >= ay1 and by2 <= ay2
                if a_inside_b:
                    relations_with_score.append((0, f"{obj_a['id']} --inside--> {obj_b['id']}"))
                    continue
                if b_inside_a:
                    relations_with_score.append((0, f"{obj_b['id']} --inside--> {obj_a['id']}"))
                    continue

                # Manhattan distance between bounding boxes.
                dx = max(0, max(bx1 - ax2 - 1, ax1 - bx2 - 1))
                dy = max(0, max(by1 - ay2 - 1, ay1 - by2 - 1))
                bbox_distance = dx + dy

                if bbox_distance == 0:
                    relations_with_score.append(
                        (0, f"{obj_a['id']} --touching--> {obj_b['id']}")
                    )
                elif bbox_distance <= 3:
                    # Directional relation based on center-of-mass.
                    cdx = float(obj_b["cx"]) - float(obj_a["cx"])
                    cdy = float(obj_b["cy"]) - float(obj_a["cy"])
                    if abs(cdx) >= abs(cdy):
                        direction = "right-of" if cdx > 0 else "left-of"
                    else:
                        direction = "below" if cdy > 0 else "above"
                    relations_with_score.append(
                        (bbox_distance, f"{obj_a['id']} --{direction}(d={bbox_distance})--> {obj_b['id']}")
                    )

        relations_with_score.sort(key=lambda item: item[0])
        return [rel for _, rel in relations_with_score[: self.relation_max_count]]

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
