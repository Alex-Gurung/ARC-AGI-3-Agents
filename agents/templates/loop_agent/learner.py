"""Learner component for the LoopAgent.

Called after every action with before/after states. Produces memory
operations (ADD/REMOVE/MODIFY/NONE) based on observed changes.

The model sees raw BEFORE/AFTER/DIFF evidence and judges from that —
the surprise score is NOT passed to the model (it's a system-level signal).
"""

import logging
import re
from typing import Optional

from openai import OpenAI

from .memory import Memory, apply_memory_operation, parse_memory_operation

logger = logging.getLogger(__name__)


LEARNER_PROMPT = """\
You are building a knowledge base about a game by observing what happens after each action. Focus on HIGH-LEVEL understanding — the most valuable entries are:

- RULE: Game mechanics and constraints (e.g., "Black cells block movement", "Touching lava resets level")
- VOCAB: What colors/sprites represent (e.g., "Color 9 = player", "Color 11 = exit door")
- PLAN/SUBGOAL: Strategic insights (e.g., "Need key before door opens")
- OBSERVATION: Notable environmental patterns
- ACTION: ONLY basic movement mechanics (e.g., "ACTION1 moves player up 1 cell"). Do not re-add actions already in memory.

IMPORTANT: Check MEMORY below before adding anything. If a similar entry exists, use MODIFY to refine it or output NONE. Do NOT add duplicates. Actively REMOVE outdated, wrong, or redundant entries to keep memory clean.

PHASE: {phase}
LEVEL: {level}
SUBGOAL_INDEX: {subgoal_index}

BEFORE: {state_before}
ACTION: {action_taken}
PREDICTION: {prediction}
AFTER: {state_after}
DIFF: {diff_text}

MEMORY:
{memory_text}

Compare PREDICTION with AFTER/DIFF. Did the outcome match? Did this reveal something new, confirm a belief, or contradict something in memory?

You may output MULTIPLE operations (one per line). Formats:
ADD [TYPE] what we learned | why we think this (confidence 0-1)
MODIFY index corrected belief | why the correction (confidence 0-1)
REMOVE index | why this entry is wrong or redundant
NONE

Examples:
- ADD [RULE] Black cells (5) block movement | tried moving into them twice with no effect (0.7)
- ADD [VOCAB] Color 11 = exit door border | touching it completed the level (0.9)
- MODIFY 3 Energy decreases by 2 per move, not 1 | counted more carefully (0.6)
- REMOVE 5 | duplicate of entry 2
- REMOVE 8 | contradicted when ACTION3 moved us right, not left
- NONE

Think step by step, then output your operations:
ANSWER:
<one or more operations, one per line>"""

DIAGNOSIS_PROMPT = """\
Something unexpected happened while trying to solve the game.

EXPECTED: {expected}
ACTUAL: {actual}

MEMORY:
{memory_text}

Which level of our understanding was wrong?
- action: an action did something different than we recorded
- subgoal: a subgoal had an unexpected outcome or was impossible
- plan: our overall strategy is flawed or incomplete

Which specific memory entry (by index) is most likely wrong?

Briefly think step by step, then output exactly one final line:
ANSWER: level=<action|subgoal|plan> index=<n or none>"""


class Learner:
    """Updates memory based on observed state transitions."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model
        self.consecutive_nones: int = 0  # track how many NONE ops in a row
        self.last_raw_output: str = ""
        self.last_answer_output: str = ""

    def update(
        self,
        state_before: str,
        action_taken: str,
        state_after: str,
        diff_text: str,
        memory: Memory,
        current_step: int,
        prediction: str = "",
        phase: str = "UNKNOWN",
        level: str = "action",
        subgoal_index: int | None = None,
    ) -> bool:
        """Observe a state transition and update memory.

        Args:
            state_before: Compressed state before action.
            action_taken: The action that was taken.
            state_after: Compressed state after action.
            diff_text: Text description of changed cells.
            memory: The memory to update.
            current_step: Current step number.
            prediction: What the agent expected to happen (from curiosity/solver).
            phase: Current high-level phase.
            level: Current abstraction level.
            subgoal_index: Active subgoal index, if any.

        Returns:
            True if memory was changed, False otherwise.
        """
        memory_text = memory.to_text() if memory else "empty"
        stage_subgoal_index = str(subgoal_index) if subgoal_index is not None else "none"

        prompt = LEARNER_PROMPT.format(
            phase=phase,
            level=level,
            subgoal_index=stage_subgoal_index,
            state_before=state_before,
            action_taken=action_taken,
            prediction=prediction or "no prediction",
            state_after=state_after,
            diff_text=diff_text,
            memory_text=memory_text,
        )

        raw_output = self._call_llm(prompt)
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output

        # Parse and apply multiple operations (one per line)
        any_changed = False
        for line in answer_output.strip().splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            # Skip lines that look like markdown list bullets wrapping a real op
            if line.startswith("- "):
                line = line[2:].strip()
            operation = parse_memory_operation(line, current_step)
            changed = apply_memory_operation(memory, operation, current_step)
            if changed:
                any_changed = True

        if any_changed:
            self.consecutive_nones = 0
            logger.info(f"Learner updated memory (step {current_step}): {answer_output[:120]}")
        else:
            self.consecutive_nones += 1
            logger.debug(f"Learner: no update (step {current_step}, {self.consecutive_nones} consecutive)")

        return any_changed

    def diagnose(
        self,
        expected: str,
        actual: str,
        memory: Memory,
    ) -> Optional[dict]:
        """Diagnose which abstraction level's assumption broke.

        Called when exploitation fails (high surprise or GAME_OVER).

        Returns:
            dict with "level" (action/subgoal/plan) and "entry_index" (int or None),
            or None if diagnosis failed.
        """
        memory_text = memory.to_text() if memory else "empty"

        prompt = DIAGNOSIS_PROMPT.format(
            expected=expected,
            actual=actual,
            memory_text=memory_text,
        )

        raw_output = self._call_llm(prompt)
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output
        return self._parse_diagnosis(answer_output)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a single prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=1.0,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.error(f"Learner LLM call failed: {e}")
            return "NONE"

    def _parse_diagnosis(self, raw_output: str) -> Optional[dict]:
        """Parse diagnosis output into level and entry index."""
        text = raw_output.lower().strip()

        level = None
        for candidate in ("action", "subgoal", "plan"):
            if candidate in text:
                level = candidate
                break

        if not level:
            logger.warning(f"Could not parse diagnosis level from: {raw_output[:80]}")
            return None

        # Try to find entry index
        entry_index = None

        match = re.search(r"(?:entry|index)\s*:?\s*(\d+)", text)
        if match:
            entry_index = int(match.group(1))

        return {"level": level, "entry_index": entry_index}

    def _extract_answer(self, raw_output: str) -> str:
        """Extract everything after the final ANSWER: marker.

        Supports multi-line answers (multiple operations after one ANSWER:).
        """
        text = raw_output.strip()

        # Split on ANSWER: and take everything after the last one
        parts = re.split(r"(?im)^\s*ANSWER\s*:\s*", text)
        if len(parts) > 1:
            return parts[-1].strip()

        # Try inline split (ANSWER: not at line start)
        inline = re.split(r"(?i)\bANSWER\s*:\s*", text)
        if len(inline) > 1:
            return inline[-1].strip()

        return text

    @property
    def memory_is_stable(self) -> bool:
        """Whether memory has been stable (no updates) for several steps."""
        return self.consecutive_nones >= 5

    def reset_stability(self) -> None:
        """Reset stability tracker."""
        self.consecutive_nones = 0
