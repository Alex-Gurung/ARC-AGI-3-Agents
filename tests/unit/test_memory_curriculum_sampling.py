from agents.templates.loop_agent.memory import Memory, MemoryEntry
from training.verl.memory_curriculum import MemoryCurriculum, MemoryCurriculumConfig


def test_memory_curriculum_modes() -> None:
    cfg = MemoryCurriculumConfig(carry_p=1.0, noisy_p=0.0, blank_p=0.0)
    curriculum = MemoryCurriculum(cfg, seed=0)
    memory = Memory()
    memory.add(
        MemoryEntry(
            type="RULE",
            content="test",
            justification="obs",
            confidence=0.7,
            created_step=0,
            last_modified_step=0,
        )
    )
    initialized, mode = curriculum.initialize(base_memory=memory)
    assert mode == "carry"
    assert len(initialized.entries) == 1


def test_memory_curriculum_blank_mode() -> None:
    cfg = MemoryCurriculumConfig(carry_p=0.0, noisy_p=0.0, blank_p=1.0)
    curriculum = MemoryCurriculum(cfg, seed=0)
    memory = Memory()
    memory.add(
        MemoryEntry(
            type="RULE",
            content="test",
            justification="obs",
            confidence=0.7,
            created_step=0,
            last_modified_step=0,
        )
    )
    initialized, mode = curriculum.initialize(base_memory=memory)
    assert mode == "blank"
    assert len(initialized.entries) == 0
