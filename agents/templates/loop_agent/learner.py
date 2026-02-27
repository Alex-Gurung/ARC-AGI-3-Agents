"""Learner component for the LoopAgent.

Called after every action with before/after states. Produces memory
operations (ADD/REMOVE/MODIFY/NONE) based on observed changes.

The model sees raw BEFORE/AFTER/DIFF evidence and judges from that —
the surprise score is NOT passed to the model (it's a system-level signal).
"""

import logging
from typing import Optional

from openai import OpenAI

from .memory import Memory, apply_memory_operation, parse_memory_operation

logger = logging.getLogger(__name__)


LEARNER_PROMPT = """\
You are building a knowledge base about a game by observing what happens after each action. Your job is to notice patterns, confirm or correct existing beliefs, and record new discoveries.

BEFORE: {state_before}
ACTION: {action_taken}
AFTER: {state_after}
DIFF: {diff_text}

MEMORY:
{memory_text}

Compare BEFORE and AFTER. Did this action reveal something new, confirm something we believed, or contradict something in memory?

If something new was learned or something needs correcting, output one line in this format:
ADD [TYPE] what we learned | why we think this (confidence 0-1)
MODIFY index corrected belief | why the correction (confidence 0-1)
REMOVE index | why this entry is wrong

If nothing notable happened, output:
NONE

Type must be one of: ACTION, RULE, SUBGOAL, PLAN, OBSERVATION, VOCAB

Examples:
- ADD [ACTION] ACTION1 moves player up by 1 cell | player sprite shifted up after ACTION1 (0.8)
- ADD [RULE] Black cells (5) block movement | tried moving into them twice with no effect (0.7)
- ADD [VOCAB] Color 11 = border of exit door | touching it completed the level (0.9)
- MODIFY 3 Energy decreases by 2 per move, not 1 | counted cells in row 61 more carefully (0.6)
- REMOVE 5 | this was contradicted when ACTION3 moved us right, not left
- NONE

Update:"""

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

Level:"""


class Learner:
    """Updates memory based on observed state transitions."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model
        self.consecutive_nones: int = 0  # track how many NONE ops in a row

    def update(
        self,
        state_before: str,
        action_taken: str,
        state_after: str,
        diff_text: str,
        memory: Memory,
        current_step: int,
    ) -> bool:
        """Observe a state transition and update memory.

        Args:
            state_before: Compressed state before action.
            action_taken: The action that was taken.
            state_after: Compressed state after action.
            diff_text: Text description of changed cells.
            memory: The memory to update.
            current_step: Current step number.

        Returns:
            True if memory was changed, False otherwise.
        """
        memory_text = memory.to_text() if memory else "empty"

        prompt = LEARNER_PROMPT.format(
            state_before=state_before,
            action_taken=action_taken,
            state_after=state_after,
            diff_text=diff_text,
            memory_text=memory_text,
        )

        raw_output = self._call_llm(prompt)

        # Parse and apply the memory operation
        operation = parse_memory_operation(raw_output, current_step)
        changed = apply_memory_operation(memory, operation, current_step)

        if changed:
            self.consecutive_nones = 0
            logger.info(f"Learner updated memory (step {current_step}): {raw_output[:100]}")
        else:
            self.consecutive_nones += 1
            logger.debug(f"Learner: no update (step {current_step}, {self.consecutive_nones} consecutive)")

        return changed

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
        return self._parse_diagnosis(raw_output)

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
        import re

        match = re.search(r"(?:entry|index)\s*:?\s*(\d+)", text)
        if match:
            entry_index = int(match.group(1))

        return {"level": level, "entry_index": entry_index}

    @property
    def memory_is_stable(self) -> bool:
        """Whether memory has been stable (no updates) for several steps."""
        return self.consecutive_nones >= 5

    def reset_stability(self) -> None:
        """Reset stability tracker."""
        self.consecutive_nones = 0
