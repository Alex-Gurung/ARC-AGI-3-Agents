"""Entity registry for the Observer's accumulated beliefs about game elements.

Tracks per-color semantic labels (from Observer output) with confidence,
plus a factual grid census. Separate from the main memory/rulebook to
keep the Observer independent from World Model predictions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agents.templates.loop_agent.state_encoder import color_label


@dataclass
class EntityEntry:
    """A belief about what a specific color represents in the game."""

    color: int
    label: str = ""  # semantic role from Observer ("player piece", "wall", etc.)
    confidence: float = 0.0
    observations: int = 0  # times Observer has labeled this color
    area: int = 0  # latest cell count from census


class EntityRegistry:
    """Tracks the Observer's beliefs about game elements across steps.

    Two sources of information:
    1. Factual census — which colors exist and how many cells (always correct)
    2. Semantic labels — Observer's accumulated descriptions with confidence
    """

    CONF_INITIAL = 0.4
    CONF_INCREMENT = 0.15
    CONF_CAP = 0.95
    CONF_CONTRADICTION = 0.3
    CONF_THRESHOLD = 0.5  # minimum to include in prompt

    def __init__(self) -> None:
        self._entries: dict[int, EntityEntry] = {}

    def update_census(self, grid: list[list[int]]) -> None:
        """Update factual color census from current grid."""
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        # Update area for known colors, add new ones
        seen_colors = set(counts.keys())
        for color, count in counts.items():
            if color not in self._entries:
                self._entries[color] = EntityEntry(color=color)
            self._entries[color].area = count
        # Zero out area for colors no longer present
        for color in list(self._entries):
            if color not in seen_colors:
                self._entries[color].area = 0

    def update_labels(self, observer_entities: dict[int, str]) -> None:
        """Update semantic labels from Observer's ENTITIES output.

        Args:
            observer_entities: mapping of color_int -> role_label
        """
        for color, new_label in observer_entities.items():
            if color not in self._entries:
                self._entries[color] = EntityEntry(color=color)
            entry = self._entries[color]
            entry.observations += 1

            if not entry.label:
                # First time labeling this color
                entry.label = new_label
                entry.confidence = self.CONF_INITIAL
            elif self._labels_match(entry.label, new_label):
                # Consistent with previous — increase confidence
                entry.confidence = min(
                    self.CONF_CAP, entry.confidence + self.CONF_INCREMENT
                )
            else:
                # Contradiction — reset confidence, replace label
                entry.label = new_label
                entry.confidence = self.CONF_CONTRADICTION

    @staticmethod
    def _labels_match(a: str, b: str) -> bool:
        """Check if two labels are semantically the same (case-insensitive)."""
        return a.strip().lower() == b.strip().lower()

    def to_prompt_text(self, min_confidence: float | None = None) -> str:
        """Format for injection into Observer/WM prompts.

        Only includes semantic labels above the confidence threshold.
        Always includes the factual census line.
        """
        if min_confidence is None:
            min_confidence = self.CONF_THRESHOLD

        # Semantic labels (above threshold)
        labeled = [
            e
            for e in self._entries.values()
            if e.label and e.confidence >= min_confidence
        ]
        labeled.sort(key=lambda e: (-e.confidence, e.color))

        lines: list[str] = []
        if labeled:
            lines.append(
                "KNOWN ELEMENTS (from your prior observations "
                "— use as context, revise if wrong):"
            )
            for e in labeled:
                conf_word = (
                    "high" if e.confidence >= 0.7 else "medium"
                )
                lines.append(
                    f"  {color_label(e.color)}: {e.label} "
                    f"({conf_word} confidence, {e.area} cells)"
                )

        # Census line (always included if we have data)
        census_parts = []
        for e in sorted(self._entries.values(), key=lambda x: -x.area):
            if e.area > 0:
                census_parts.append(f"{color_label(e.color)}: {e.area}")
        if census_parts:
            lines.append("Grid census: " + ", ".join(census_parts))

        return "\n".join(lines)

    def to_display_text(self) -> str:
        """Compact display for interactive testing."""
        if not self._entries:
            return "  (empty)"
        lines: list[str] = []
        for e in sorted(self._entries.values(), key=lambda x: -x.area):
            if e.area == 0 and not e.label:
                continue
            label_part = f'"{e.label}"' if e.label else "(unlabeled)"
            conf_part = f"conf={e.confidence:.2f}" if e.label else ""
            obs_part = f"obs={e.observations}" if e.observations else ""
            parts = [f"{color_label(e.color)}: {label_part}"]
            if conf_part:
                parts.append(conf_part)
            if obs_part:
                parts.append(obs_part)
            parts.append(f"{e.area} cells")
            lines.append("  " + ", ".join(parts))
        return "\n".join(lines) if lines else "  (empty)"

    def reset(self) -> None:
        """Clear all entries. Called on level change."""
        self._entries.clear()
