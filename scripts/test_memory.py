"""Test the Memory data structure: ADD/REMOVE/MODIFY, parsing, eviction.

No LLM needed — tests pure data operations.

Usage:
    uv run python scripts/test_memory.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.templates.loop_agent.memory import (
    Memory,
    MemoryEntry,
    apply_memory_operation,
    parse_memory_operation,
)


def test_basic_operations():
    """Test ADD, REMOVE, MODIFY."""
    print("=" * 60)
    print("TEST: Basic Operations")
    print("=" * 60)

    mem = Memory(max_entries=5)
    assert len(mem) == 0

    # ADD
    e1 = MemoryEntry(
        type="ACTION",
        content="ACTION1 moves player up by 1 cell",
        justification="player sprite shifted up after ACTION1",
        confidence=0.8,
        created_step=1,
        last_modified_step=1,
    )
    idx = mem.add(e1)
    assert idx == 0
    assert len(mem) == 1
    print(f"  ADD: {mem.get(0).to_text()}")

    # ADD more
    e2 = MemoryEntry(
        type="RULE",
        content="Black cells (5) block movement",
        justification="tried moving into them twice with no effect",
        confidence=0.7,
        created_step=2,
        last_modified_step=2,
    )
    mem.add(e2)
    assert len(mem) == 2

    # MODIFY
    success = mem.modify(0, "ACTION1 moves player up by 2 cells", "measured more carefully", 0.9, 3)
    assert success
    print(f"  MODIFY: {mem.get(0).to_text()}")

    # REMOVE
    removed = mem.remove(1, "contradicted by evidence")
    assert removed is not None
    assert len(mem) == 1
    print(f"  REMOVE: {removed.to_text()}")

    print("  PASS\n")


def test_eviction():
    """Test that lowest-confidence entry is evicted when full."""
    print("=" * 60)
    print("TEST: Eviction on Overflow")
    print("=" * 60)

    mem = Memory(max_entries=3)

    # Fill to capacity
    for i in range(3):
        mem.add(MemoryEntry(
            type="OBSERVATION",
            content=f"observation {i}",
            justification=f"saw it at step {i}",
            confidence=0.1 * (i + 1),  # 0.1, 0.2, 0.3
            created_step=i,
            last_modified_step=i,
        ))

    assert len(mem) == 3
    print(f"  Before eviction: {[e.confidence for e in mem.entries]}")

    # Add one more — should evict confidence=0.1
    mem.add(MemoryEntry(
        type="RULE",
        content="new important rule",
        justification="clear evidence",
        confidence=0.9,
        created_step=10,
        last_modified_step=10,
    ))

    assert len(mem) == 3
    confidences = [e.confidence for e in mem.entries]
    print(f"  After eviction: {confidences}")
    assert 0.1 not in confidences, "Lowest confidence entry should have been evicted"
    print("  PASS\n")


def test_text_serialization():
    """Test to_text() output format."""
    print("=" * 60)
    print("TEST: Text Serialization")
    print("=" * 60)

    mem = Memory()
    mem.add(MemoryEntry(
        type="ACTION", content="ACTION1 moves up", justification="observed",
        confidence=0.8, created_step=0, last_modified_step=0,
    ))
    mem.add(MemoryEntry(
        type="VOCAB", content="Color 9 = player body", justification="consistent across frames",
        confidence=0.9, created_step=1, last_modified_step=1,
    ))

    text = mem.to_text()
    print(f"  Output:\n{text}")
    print()

    assert "[0]" in text
    assert "[1]" in text
    assert "ACTION" in text
    assert "VOCAB" in text
    print("  PASS\n")


def test_parse_operations():
    """Test parsing LLM output into memory operations."""
    print("=" * 60)
    print("TEST: Parse Memory Operations")
    print("=" * 60)

    # Test ADD
    result = parse_memory_operation(
        "ADD [ACTION] ACTION1 moves player up by 1 cell | player sprite shifted up after ACTION1 (0.8)",
        current_step=5,
    )
    assert result["op"] == "add"
    assert result["entry"].type == "ACTION"
    assert result["entry"].confidence == 0.8
    print(f"  ADD parsed: {result['entry'].to_text()}")

    # Test MODIFY
    result = parse_memory_operation(
        "MODIFY [3] Energy decreases by 2 per move, not 1 | counted cells more carefully (0.6)",
        current_step=10,
    )
    assert result["op"] == "modify"
    assert result["index"] == 3
    assert result["confidence"] == 0.6
    print(f"  MODIFY parsed: index={result['index']}, content={result['content'][:50]}")

    # Test REMOVE
    result = parse_memory_operation(
        "REMOVE [5] | this was contradicted when ACTION3 moved us right",
        current_step=15,
    )
    assert result["op"] == "remove"
    assert result["index"] == 5
    print(f"  REMOVE parsed: index={result['index']}, reason={result['reason'][:50]}")

    # Test NONE
    result = parse_memory_operation("NONE", current_step=20)
    assert result["op"] == "none"
    print("  NONE parsed correctly")

    # Test invalid
    result = parse_memory_operation("garbage output that makes no sense", current_step=25)
    assert result["op"] == "error"
    print(f"  Error handled: {result['reason'][:50]}")

    print("  PASS\n")


def test_apply_operations():
    """Test applying parsed operations to memory."""
    print("=" * 60)
    print("TEST: Apply Operations")
    print("=" * 60)

    mem = Memory()

    # Apply ADD
    op = parse_memory_operation(
        "ADD [RULE] Black cells block movement | tried 3 times, no effect (0.85)",
        current_step=1,
    )
    changed = apply_memory_operation(mem, op, current_step=1)
    assert changed
    assert len(mem) == 1
    print(f"  After ADD: {mem.to_text()}")

    # Apply NONE
    op = parse_memory_operation("NONE", current_step=2)
    changed = apply_memory_operation(mem, op, current_step=2)
    assert not changed
    print("  After NONE: no change (correct)")

    # Apply MODIFY
    op = parse_memory_operation(
        "MODIFY [0] Black cells and dark gray cells block movement | tested with both colors (0.9)",
        current_step=3,
    )
    changed = apply_memory_operation(mem, op, current_step=3)
    assert changed
    print(f"  After MODIFY: {mem.get(0).to_text()}")

    print("  PASS\n")


def test_get_by_type_and_low_confidence():
    """Test filtering methods."""
    print("=" * 60)
    print("TEST: Filtering (by type, low confidence)")
    print("=" * 60)

    mem = Memory()
    mem.add(MemoryEntry(type="ACTION", content="a", justification="j", confidence=0.9, created_step=0, last_modified_step=0))
    mem.add(MemoryEntry(type="RULE", content="b", justification="j", confidence=0.3, created_step=1, last_modified_step=1))
    mem.add(MemoryEntry(type="ACTION", content="c", justification="j", confidence=0.4, created_step=2, last_modified_step=2))
    mem.add(MemoryEntry(type="VOCAB", content="d", justification="j", confidence=0.2, created_step=3, last_modified_step=3))

    actions = mem.get_by_type("ACTION")
    assert len(actions) == 2
    print(f"  ACTION entries: {len(actions)}")

    low_conf = mem.get_low_confidence(0.5)
    assert len(low_conf) == 3
    print(f"  Low confidence (<0.5): {len(low_conf)}")

    print("  PASS\n")


if __name__ == "__main__":
    test_basic_operations()
    test_eviction()
    test_text_serialization()
    test_parse_operations()
    test_apply_operations()
    test_get_by_type_and_low_confidence()
    print("All Memory tests passed!")
