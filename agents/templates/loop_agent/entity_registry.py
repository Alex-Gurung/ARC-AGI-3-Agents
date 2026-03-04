"""Entity registry for the Observer's accumulated beliefs about game elements.

Tracks per-color semantic labels (from Observer output) with confidence,
plus a factual grid census. Colors that share the same label are grouped
into multi-color elements for display. Separate from the main memory/rulebook
to keep the Observer independent from World Model predictions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from agents.templates.loop_agent.state_encoder import color_label


@dataclass
class EntityEntry:
    """A belief about what a specific color's role is in the game."""

    color: int
    label: str = ""  # element role from Observer ("player", "border", etc.)
    confidence: float = 0.0
    observations: int = 0  # times Observer has labeled this color
    area: int = 0  # latest cell count from census


class EntityRegistry:
    """Tracks the Observer's beliefs about game elements across steps.

    Two sources of information:
    1. Factual census — which colors exist and how many cells (always correct)
    2. Semantic labels — Observer's accumulated element descriptions with confidence

    Colors with the same label are grouped into multi-color elements for display.
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
        seen_colors = set(counts.keys())
        for color, count in counts.items():
            if color not in self._entries:
                self._entries[color] = EntityEntry(color=color)
            self._entries[color].area = count
        for color in list(self._entries):
            if color not in seen_colors:
                self._entries[color].area = 0

    def update_labels(self, observer_entities: dict[int, str]) -> None:
        """Update semantic labels from Observer's ENTITIES output.

        Args:
            observer_entities: mapping of color_int -> element_role
                Multiple colors may map to the same role (multi-color element).
        """
        for color, new_label in observer_entities.items():
            if color not in self._entries:
                self._entries[color] = EntityEntry(color=color)
            entry = self._entries[color]
            entry.observations += 1

            if not entry.label:
                entry.label = new_label
                entry.confidence = self.CONF_INITIAL
            elif self._labels_match(entry.label, new_label):
                entry.confidence = min(
                    self.CONF_CAP, entry.confidence + self.CONF_INCREMENT
                )
            else:
                entry.label = new_label
                entry.confidence = self.CONF_CONTRADICTION

    @staticmethod
    def _labels_match(a: str, b: str) -> bool:
        """Check if two labels are semantically the same (case-insensitive)."""
        return a.strip().lower() == b.strip().lower()

    def _group_by_label(
        self, min_confidence: float,
    ) -> list[tuple[str, float, list[EntityEntry]]]:
        """Group entries by label, returning (label, avg_confidence, entries).

        Only includes entries above min_confidence. Sorted by confidence desc.
        """
        groups: defaultdict[str, list[EntityEntry]] = defaultdict(list)
        for e in self._entries.values():
            if e.label and e.confidence >= min_confidence:
                groups[e.label.strip().lower()].append(e)

        result = []
        for entries in groups.values():
            avg_conf = sum(e.confidence for e in entries) / len(entries)
            # Use the original-cased label from the first entry
            label = entries[0].label
            entries.sort(key=lambda e: -e.area)
            result.append((label, avg_conf, entries))

        result.sort(key=lambda x: -x[1])
        return result

    def to_prompt_text(self, min_confidence: float | None = None) -> str:
        """Format for injection into Observer/WM prompts.

        Groups colors into multi-color elements by shared label.
        Only includes elements above the confidence threshold.
        Always includes the factual census line.
        """
        if min_confidence is None:
            min_confidence = self.CONF_THRESHOLD

        groups = self._group_by_label(min_confidence)

        lines: list[str] = []
        if groups:
            lines.append(
                "KNOWN ELEMENTS (from your prior observations "
                "— use as context, revise if wrong):"
            )
            for label, avg_conf, entries in groups:
                conf_word = "high" if avg_conf >= 0.7 else "medium"
                colors = ", ".join(color_label(e.color) for e in entries)
                total_area = sum(e.area for e in entries)
                lines.append(
                    f"  {label} ({conf_word} confidence, "
                    f"{total_area} cells, colors: {colors})"
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

        # Show grouped elements first
        groups = self._group_by_label(0.0)
        lines: list[str] = []
        grouped_colors: set[int] = set()
        for label, avg_conf, entries in groups:
            colors = " + ".join(color_label(e.color) for e in entries)
            total_area = sum(e.area for e in entries)
            obs = max(e.observations for e in entries)
            lines.append(
                f"  {label}: {colors} "
                f"(conf={avg_conf:.2f}, obs={obs}, {total_area} cells)"
            )
            grouped_colors.update(e.color for e in entries)

        # Show unlabeled colors
        for e in sorted(self._entries.values(), key=lambda x: -x.area):
            if e.color in grouped_colors:
                continue
            if e.area == 0:
                continue
            lines.append(f"  (unlabeled): {color_label(e.color)} ({e.area} cells)")

        return "\n".join(lines) if lines else "  (empty)"

    def reset(self) -> None:
        """Clear all entries. Called on level change."""
        self._entries.clear()
