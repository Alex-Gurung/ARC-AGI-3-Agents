"""Memory data structure for the LoopAgent.

Bounded list of entries with ADD/REMOVE/MODIFY operations,
confidence tracking, and justifications. Max 50 entries with
lowest-confidence eviction when full.

Types are optional cosmetic tags for organization — no logic branches on them.
"""

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single entry in the agent's memory."""

    type: str = ""  # optional cosmetic tag (e.g. "ACTION", "RULE")
    content: str = ""  # what we believe
    justification: str = ""  # why we believe it
    confidence: float = 0.5  # 0-1
    created_step: int = 0
    last_modified_step: int = 0
    memory_id: str = field(default="")

    def to_text(self) -> str:
        """Serialize to single-line text for LLM context."""
        prefix = f"[{self.type}] " if self.type else ""
        return (
            f"{prefix}{self.content} | "
            f"{self.justification} (conf: {self.confidence:.2f})"
        )


class Memory:
    """Bounded memory database with ADD/REMOVE/MODIFY operations."""

    MAX_ENTRIES: int = 50

    def __init__(self, max_entries: int = 50) -> None:
        self.MAX_ENTRIES = max_entries
        self.entries: list[MemoryEntry] = []
        self._next_id: int = 1

    def _new_memory_id(self) -> str:
        memory_id = f"M{self._next_id:04d}"
        self._next_id += 1
        return memory_id

    def _resolve_index(self, ref: int | str) -> Optional[int]:
        if isinstance(ref, int):
            return ref if 0 <= ref < len(self.entries) else None

        text = str(ref).strip().upper()
        if not text:
            return None

        if text.isdigit():
            idx = int(text)
            return idx if 0 <= idx < len(self.entries) else None

        if text.startswith("M"):
            for idx, entry in enumerate(self.entries):
                if entry.memory_id.upper() == text:
                    return idx

        return None

    def add(self, entry: MemoryEntry, dedup_threshold: float = 0.5) -> int:
        """Add a new entry. Returns index. Evicts lowest-confidence if full.

        If an existing entry has high word overlap (Jaccard >= dedup_threshold),
        boosts its confidence instead of adding a duplicate.
        """
        if dedup_threshold > 0:
            similar_idx = self._find_similar(entry, threshold=dedup_threshold)
            if similar_idx is not None:
                existing = self.entries[similar_idx]
                existing.confidence = min(1.0, max(existing.confidence, entry.confidence))
                existing.last_modified_step = entry.created_step
                logger.debug(
                    f"Memory DEDUP: skipped add, boosted [{similar_idx}] "
                    f"'{existing.content[:60]}' confidence to {existing.confidence:.2f}"
                )
                return similar_idx

        if len(self.entries) >= self.MAX_ENTRIES:
            self._evict_lowest_confidence()
        if not entry.memory_id:
            entry.memory_id = self._new_memory_id()
        self.entries.append(entry)
        idx = len(self.entries) - 1
        logger.debug(f"Memory ADD [{idx}]: {entry.to_text()}")
        return idx

    def _find_similar(self, entry: MemoryEntry, threshold: float = 0.5) -> Optional[int]:
        """Find an existing entry with similar content (high word overlap)."""
        new_words = set(entry.content.lower().split())
        if not new_words:
            return None

        for i, existing in enumerate(self.entries):
            existing_words = set(existing.content.lower().split())
            if not existing_words:
                continue
            intersection = new_words & existing_words
            union = new_words | existing_words
            similarity = len(intersection) / len(union) if union else 0
            if similarity >= threshold:
                return i
        return None

    def remove(self, index: int | str, reason: str = "") -> Optional[MemoryEntry]:
        """Remove entry by index or memory_id. Returns removed entry or None if invalid."""
        resolved = self._resolve_index(index)
        if resolved is None:
            logger.warning(
                f"Memory REMOVE failed: ref {index} could not be resolved "
                f"(size={len(self.entries)})"
            )
            return None
        index = resolved
        if 0 <= index < len(self.entries):
            removed = self.entries.pop(index)
            logger.debug(
                f"Memory REMOVE [{index}][{removed.memory_id}]: "
                f"{removed.to_text()} | reason: {reason}"
            )
            return removed
        logger.warning(
            f"Memory REMOVE failed: index {index} out of range "
            f"(0-{len(self.entries) - 1})"
        )
        return None

    def modify(
        self,
        index: int | str,
        content: str,
        justification: str,
        confidence: float,
        step: int,
    ) -> bool:
        """Modify entry by index or memory_id. Returns True if successful."""
        resolved = self._resolve_index(index)
        if resolved is None:
            logger.warning(
                f"Memory MODIFY failed: ref {index} could not be resolved "
                f"(size={len(self.entries)})"
            )
            return False
        index = resolved
        if 0 <= index < len(self.entries):
            entry = self.entries[index]
            old_text = entry.to_text()
            entry.content = content
            entry.justification = justification
            entry.confidence = confidence
            entry.last_modified_step = step
            logger.debug(
                f"Memory MODIFY [{index}][{entry.memory_id}]: "
                f"{old_text} -> {entry.to_text()}"
            )
            return True
        logger.warning(
            f"Memory MODIFY failed: index {index} out of range "
            f"(0-{len(self.entries) - 1})"
        )
        return False

    def get(self, index: int | str) -> Optional[MemoryEntry]:
        """Get entry by index or memory_id."""
        resolved = self._resolve_index(index)
        if resolved is None:
            return None
        index = resolved
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def get_by_type(self, entry_type: str) -> list[tuple[int, MemoryEntry]]:
        """Get all entries of a specific type with their indices."""
        return [(i, e) for i, e in enumerate(self.entries) if e.type == entry_type]

    def get_low_confidence(self, threshold: float = 0.5) -> list[tuple[int, MemoryEntry]]:
        """Get entries with confidence below threshold."""
        return [(i, e) for i, e in enumerate(self.entries) if e.confidence < threshold]

    def known_action_lessons(self) -> set[str]:
        """Return action names mentioned in any entry's content."""
        known: set[str] = set()
        pattern = re.compile(r"\b(RESET|ACTION[1-7])\b", flags=re.IGNORECASE)
        for entry in self.entries:
            for match in pattern.findall(entry.content):
                known.add(match.upper())
        return known

    def missing_action_lessons(self, available_actions: list[str]) -> list[str]:
        """Return available actions that do not yet have explicit ACTION lessons."""
        known = self.known_action_lessons()
        missing: list[str] = []
        for action_name in available_actions:
            upper = action_name.upper()
            if upper not in known:
                missing.append(upper)
        return missing

    def to_text(self) -> str:
        """Serialize full memory for LLM context."""
        if not self.entries:
            return "RULEBOOK (0/50 entries): empty"
        lines = [f"RULEBOOK ({len(self.entries)}/{self.MAX_ENTRIES} entries):"]
        for i, entry in enumerate(self.entries):
            lines.append(f"[{i}] {entry.to_text()}")
        return "\n".join(lines)

    def to_text_with_indices(self) -> str:
        """Serialize with prominent indices for REMOVE/MODIFY references."""
        return self.to_text()  # same format, indices already included

    def _evict_lowest_confidence(self) -> None:
        """Remove the entry with lowest confidence to make room."""
        if not self.entries:
            return
        min_idx = min(range(len(self.entries)), key=lambda i: self.entries[i].confidence)
        evicted = self.entries.pop(min_idx)
        logger.debug(f"Memory EVICT [{min_idx}]: {evicted.to_text()}")

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()

    def perturb(
        self,
        delete_fraction: float = 0.2,
        confidence_jitter: float = 0.1,
        shuffle_entries: bool = True,
    ) -> None:
        """Apply mild random corruption to improve robustness in training runs."""
        if not self.entries:
            return

        delete_fraction = max(0.0, min(1.0, delete_fraction))
        remove_count = min(len(self.entries), int(round(len(self.entries) * delete_fraction)))
        if remove_count > 0:
            remove_indices = sorted(
                random.sample(range(len(self.entries)), remove_count),
                reverse=True,
            )
            for idx in remove_indices:
                removed = self.entries.pop(idx)
                logger.debug(f"Memory PERTURB remove [{idx}][{removed.memory_id}]")

        if confidence_jitter > 0:
            for entry in self.entries:
                delta = random.uniform(-confidence_jitter, confidence_jitter)
                entry.confidence = max(0.0, min(1.0, entry.confidence + delta))

        if shuffle_entries and len(self.entries) > 1:
            random.shuffle(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0


def _strip_type_prefix(content: str) -> tuple[str, str]:
    """Strip a leading [TYPE] tag from content if present.

    Returns (type_tag, cleaned_content). type_tag is "" if no tag found.
    Handles doubled tags like '[RULE] [RULE] content' → ('RULE', 'content').
    """
    cleaned = content.strip()
    tag = ""
    # Strip up to 2 leading [TYPE] prefixes (handles doubling)
    for _ in range(2):
        m = re.match(r"\[(\w+(?:/\w+)?)\]\s*", cleaned)
        if m:
            tag = m.group(1).upper().split("/")[0]
            cleaned = cleaned[m.end():]
        else:
            break
    return tag, cleaned.strip()


def parse_memory_operation(text: str, current_step: int) -> dict:
    """Parse LLM output into a memory operation.

    Expected formats (type tag and confidence are optional):
        ADD [TYPE] content | justification (confidence)
        ADD content | justification (confidence)
        MODIFY index content | justification (confidence)
        REMOVE index | reason
        NONE

    Returns dict with 'op' key ('add', 'modify', 'remove', 'none', 'error')
    and relevant fields.
    """
    text = text.strip()

    # Handle NONE
    if text.upper() == "NONE" or not text:
        return {"op": "none"}

    # Try ADD — optional [TYPE] tag, optional | separator, optional confidence
    # Pattern: ADD [optional type] content | justification (confidence)
    add_match = re.match(
        r"ADD\s+(?:\[?([\w/]+)\]?\s+)?(.+?)\s*\|\s*(.+?)\s*"
        r"(?:\(\s*(?:conf(?:idence)?\s*:\s*)?(\d*\.?\d+)\s*\))?\s*$",
        text,
        re.IGNORECASE,
    )
    if not add_match:
        # Fallback: ADD without | separator
        add_match = re.match(
            r"ADD\s+(?:\[?([\w/]+)\]?\s+)?(.+?)\s*"
            r"(?:\(\s*(?:conf(?:idence)?\s*:\s*)?(\d*\.?\d+)\s*\))?\s*$",
            text,
            re.IGNORECASE,
        )
        if add_match:
            raw_type = (add_match.group(1) or "").upper().split("/")[0]
            content = add_match.group(2).strip()
            # Strip doubled type prefix from content
            content_tag, content = _strip_type_prefix(content)
            if not raw_type and content_tag:
                raw_type = content_tag
            confidence = float(add_match.group(3)) if add_match.group(3) else 0.5
            confidence = max(0.0, min(1.0, confidence))
            return {
                "op": "add",
                "entry": MemoryEntry(
                    type=raw_type,
                    content=content,
                    justification="(no justification provided)",
                    confidence=confidence,
                    created_step=current_step,
                    last_modified_step=current_step,
                ),
            }
    if add_match and add_match.lastindex and add_match.lastindex >= 3:
        raw_type = (add_match.group(1) or "").upper().split("/")[0]
        content = add_match.group(2).strip()
        justification = add_match.group(3).strip()
        # Strip doubled type prefix from content
        content_tag, content = _strip_type_prefix(content)
        if not raw_type and content_tag:
            raw_type = content_tag
        confidence = float(add_match.group(4)) if add_match.group(4) else 0.5
        confidence = max(0.0, min(1.0, confidence))
        return {
            "op": "add",
            "entry": MemoryEntry(
                type=raw_type,
                content=content,
                justification=justification,
                confidence=confidence,
                created_step=current_step,
                last_modified_step=current_step,
            ),
        }

    # Try MODIFY — brackets optional, M#### or int, confidence optional
    modify_match = re.match(
        r"MODIFY\s+\[?([Mm]\d+|\d+)\]?\s+(.+?)\s*\|\s*(.+?)\s*"
        r"(?:\(\s*(?:conf(?:idence)?\s*:\s*)?(\d*\.?\d+)\s*\))?\s*$",
        text,
        re.IGNORECASE,
    )
    if not modify_match:
        # Fallback: MODIFY without | separator
        modify_match = re.match(
            r"MODIFY\s+\[?([Mm]\d+|\d+)\]?\s+(.+?)\s*"
            r"(?:\(\s*(?:conf(?:idence)?\s*:\s*)?(\d*\.?\d+)\s*\))?\s*$",
            text,
            re.IGNORECASE,
        )
        if modify_match:
            confidence = float(modify_match.group(3)) if modify_match.group(3) else 0.5
            confidence = max(0.0, min(1.0, confidence))
            ref = modify_match.group(1)
            index = int(ref) if ref.isdigit() else None
            content = modify_match.group(2).strip()
            _tag, content = _strip_type_prefix(content)
            return {
                "op": "modify",
                "ref": ref,
                "index": index,
                "content": content,
                "justification": "(no justification provided)",
                "confidence": confidence,
            }
    if modify_match and modify_match.lastindex and modify_match.lastindex >= 3:
        confidence = float(modify_match.group(4)) if modify_match.group(4) else 0.5
        confidence = max(0.0, min(1.0, confidence))
        ref = modify_match.group(1)
        index = int(ref) if ref.isdigit() else None
        content = modify_match.group(2).strip()
        justification = modify_match.group(3).strip()
        _tag, content = _strip_type_prefix(content)
        return {
            "op": "modify",
            "ref": ref,
            "index": index,
            "content": content,
            "justification": justification,
            "confidence": confidence,
        }

    # Try REMOVE — brackets optional, M#### or int
    remove_match = re.match(
        r"REMOVE\s+\[?([Mm]\d+|\d+)\]?(?:\s*\|\s*(.+))?$",
        text,
        re.IGNORECASE,
    )
    if remove_match:
        ref = remove_match.group(1)
        index = int(ref) if ref.isdigit() else None
        return {
            "op": "remove",
            "ref": ref,
            "index": index,
            "reason": (remove_match.group(2) or "").strip(),
        }

    return {"op": "error", "reason": f"Could not parse: {text[:100]}"}


def apply_memory_operation(memory: Memory, operation: dict, current_step: int) -> bool:
    """Apply a parsed memory operation to the memory. Returns True if memory changed."""
    op = operation.get("op", "none")

    if op == "none":
        return False

    if op == "add":
        entry = operation["entry"]
        memory.add(entry)
        return True

    if op == "modify":
        return memory.modify(
            index=operation.get("ref", operation.get("index")),
            content=operation["content"],
            justification=operation["justification"],
            confidence=operation["confidence"],
            step=current_step,
        )

    if op == "remove":
        removed = memory.remove(
            index=operation.get("ref", operation.get("index")),
            reason=operation.get("reason", ""),
        )
        return removed is not None

    if op == "error":
        logger.warning(f"Memory operation error: {operation.get('reason', 'unknown')}")
        return False

    return False
