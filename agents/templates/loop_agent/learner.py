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
You are building a lesson book about a game by observing what happens after each action.
Each memory entry should be an atomic, reusable lesson that helps future decisions.
Focus on HIGH-LEVEL understanding — the most valuable lessons are:

- RULE: Game mechanics and constraints (e.g., "Black cells block movement", "Touching lava resets level")
- GOAL: Hypothesis of what the level is trying to achieve (e.g., "Goal is to make the right pattern match the left template")
- VOCAB: What colors/sprites represent (e.g., "Color 9 = player", "Color 11 = exit door")
- PLAN/SUBGOAL: Strategic lessons (e.g., "Need key before door opens")
- OBSERVATION: Notable environmental patterns
- ACTION: How actions change state (e.g., "ACTION1 moves player up 1 cell unless blocked"). Do not re-add actions already in memory.

IMPORTANT: Check MEMORY below before adding anything. If a similar entry exists, use MODIFY to refine it or output NONE. Do NOT add duplicates. Actively REMOVE outdated, wrong, or redundant entries to keep memory clean.
CRITICAL EVIDENCE RULE:
- Only add/modify lessons when there is direct evidence in BEFORE/AFTER/DIFF.
- If evidence is weak or ambiguous, output NONE.
- Every lesson must include explicit evidence in the justification after "|".
  Example evidence phrases: "DIFF shows ...", "BEFORE/AFTER changed ...", "repeated over N trials".
CONFIDENCE CALIBRATION:
- 0.20-0.50: early hypothesis from 1 observation or weak evidence.
- 0.50-0.75: moderate evidence (repeated consistent observations).
- 0.75-0.90: strong evidence, but still potentially falsifiable.
- 0.90-1.00: only for directly verified outcomes (e.g., clear WIN/level-complete condition) or many repeated confirmations.
HYPOTHESIS POLICY:
- For uncertain GOAL/PLAN/SUBGOAL lessons, prefix content with "Hypothesis:".
- Do not write high-confidence GOAL/PLAN claims unless completion condition was actually observed.
When useful, explicitly write lesson hypotheses at multiple abstraction levels:
- action lesson: what a specific action does to state
- subgoal lesson: what attempting/completing a subgoal changes
- plan lesson: when a strategy works or fails
- goal lesson: what condition seems to define level completion

REFERENCE RULES:
- For MODIFY/REMOVE, ONLY use numeric indices shown in MEMORY, in brackets.
- Allowed reference form is exactly: [n]  (examples: [0], [5], [12]).
- If no valid index exists, do not guess; output NONE.

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
MODIFY [n] corrected belief | why the correction (confidence 0-1)
REMOVE [n] | why this entry is wrong or redundant
NONE

Examples:
- ADD [RULE] Black cells (5) block movement | tried moving into them twice with no effect (0.7)
- ADD [GOAL] Win seems to require touching the yellow border with the player | level ended when contact happened (0.6)
- ADD [VOCAB] Color 11 = exit door border | touching it completed the level (0.9)
- ADD [ACTION] ACTION4 moves player right by 1 unless blocked by black cells | observed over 4 trials (0.8)
- ADD [ACTION] ACTION2 shifts the wave right by one cell | DIFF shows 0->9 at (x+1,y) and 9->0 at (x,y) across 3 trials (0.85)
- MODIFY [3] Energy decreases by 2 per move, not 1 | counted more carefully (0.6)
- REMOVE [5] | duplicate of [2]
- REMOVE [8] | contradicted when ACTION3 moved us right, not left
- ADD [SUBGOAL] Reaching the blue switch flips the right gate open | after 2 tries, gate changed only when switch was touched (0.7)
- ADD [PLAN] Safe route appears to be: align key color first, then touch door border | direct door attempt caused GAME_OVER twice (0.65)
- ADD [GOAL] Hypothesis: level completes when black square touches yellow square | completion followed contact event while score/state changed (0.55)
- ADD [GOAL] Hypothesis: level completes when matching both shape and color on the right side | not yet verified by level completion; observed partial progress only (0.45)
- NONE

Think step by step, then output ONLY operations (no rationale/prose labels).
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

EXPECTATION_ASSESS_PROMPT = """\
You are checking whether an observed outcome matched what our current memory predicted.

MODE: {mode}
SUBGOAL_INDEX: {subgoal_index}
EXPECTED: {expected}
ACTUAL: {actual}

MEMORY:
{memory_text}

Decide if ACTUAL matched EXPECTED from memory.

Output exactly one final line:
ANSWER: verdict=<expected|unexpected> conf=<0.00-1.00> level=<action|subgoal|plan> ref=<index|none>
"""


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
        image_before_url: str | None = None,
        image_after_url: str | None = None,
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

        images = [url for url in [image_before_url, image_after_url] if url]
        raw_output = self._call_llm(prompt, image_data_urls=images or None)
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output

        # Parse and apply multiple operations (one per line)
        operation_lines = self._extract_operation_lines(answer_output)
        if not operation_lines and answer_output.strip():
            logger.debug("Learner produced no operation lines; treating output as NONE")

        any_changed = False
        for line in operation_lines:
            if line.upper() == "NONE":
                continue
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
        image_data_url: str | None = None,
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

        raw_output = self._call_llm(
            prompt,
            image_data_urls=[image_data_url] if image_data_url else None,
        )
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output
        return self._parse_diagnosis(answer_output)

    def assess_expectation(
        self,
        expected: str,
        actual: str,
        memory: Memory,
        mode: str,
        subgoal_index: int | None = None,
        image_data_url: str | None = None,
    ) -> Optional[dict]:
        """Assess whether the observed result was expected from memory."""
        memory_text = memory.to_text() if memory else "empty"
        prompt = EXPECTATION_ASSESS_PROMPT.format(
            mode=mode,
            subgoal_index=str(subgoal_index) if subgoal_index is not None else "none",
            expected=expected or "no prediction",
            actual=actual,
            memory_text=memory_text,
        )
        raw_output = self._call_llm(
            prompt,
            image_data_urls=[image_data_url] if image_data_url else None,
        )
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output
        return self._parse_expectation_assessment(answer_output)

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        image_data_urls: list[str] | None = None,
    ) -> str:
        """Call the LLM with a single prompt."""
        content: str | list[dict[str, object]]
        image_data_urls = [u for u in (image_data_urls or []) if u]
        if image_data_urls:
            content = [{"type": "text", "text": prompt}]
            for url in image_data_urls:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )
        else:
            content = prompt

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            if image_data_urls:
                logger.warning(
                    "Learner multimodal call failed; retrying text-only: %s",
                    e,
                )
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    content = response.choices[0].message.content or ""
                    return content.strip()
                except Exception as retry_err:
                    logger.error(f"Learner text-only retry failed: {retry_err}")
                    return "NONE"
            logger.error(f"Learner LLM call failed: {e}")
            return "NONE"

    @staticmethod
    def _is_operation_line(line: str) -> bool:
        """Return True iff the line starts with a supported operation token."""
        return re.match(r"^(ADD|MODIFY|REMOVE|NONE)\b", line, flags=re.IGNORECASE) is not None

    def _extract_operation_lines(self, text: str) -> list[str]:
        """Extract strict memory-operation lines from noisy model output.

        Keeps strict op syntax while ignoring rationale prose.
        """
        operations: list[str] = []
        saw_none = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Strip common wrappers without loosening core op schema.
            line = re.sub(r"(?i)^\s*ANSWER\s*:\s*", "", line).strip()
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if not line:
                continue

            # Support compact list formats: "ADD ...; MODIFY ...".
            fragments = [frag.strip() for frag in line.split(";") if frag.strip()]
            for fragment in fragments:
                cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", fragment).strip()
                cleaned = re.sub(r"(?i)^\s*ANSWER\s*:\s*", "", cleaned).strip()
                if not cleaned:
                    continue

                # Light normalization for common model slip: "ADD[TYPE] ..."
                if cleaned.upper().startswith("ADD["):
                    cleaned = f"ADD {cleaned[3:]}"

                if cleaned.upper() == "NONE":
                    saw_none = True
                    continue

                if self._is_operation_line(cleaned):
                    operations.append(cleaned)

        if operations:
            return operations
        if saw_none:
            return ["NONE"]
        return []

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

    def _parse_expectation_assessment(self, raw_output: str) -> Optional[dict]:
        """Parse expectation-assessment schema from model output.

        Accepts both strict key=value format and common model variations:
          verdict=unexpected conf=0.85 level=plan ref=none   (prompt format)
          VERDICT: unexpected conf=0.85 level=plan           (common variation)
          unexpected conf=0.85 level=plan                    (bare)
        """
        text = raw_output.strip()

        # Strip optional VERDICT:/verdict= label prefix
        text_clean = re.sub(
            r"^(?:verdict)\s*[:=]\s*", "", text, count=1, flags=re.IGNORECASE
        )

        # Match verdict value + key=value pairs (ref is optional)
        match = re.search(
            r"(expected|unexpected)\s+"
            r"conf\s*[:=]\s*([01](?:\.\d+)?)\s+"
            r"level\s*[:=]\s*(action|subgoal|plan)"
            r"(?:\s+ref\s*[:=]\s*([A-Za-z0-9_-]+|none))?",
            text_clean,
            flags=re.IGNORECASE,
        )
        if not match:
            logger.warning(f"Could not parse expectation assessment from: {raw_output[:120]}")
            return None

        verdict = match.group(1).lower()
        confidence = max(0.0, min(1.0, float(match.group(2))))
        level = match.group(3).lower()
        entry_ref = match.group(4) if match.group(4) else None
        if entry_ref and entry_ref.lower() == "none":
            entry_ref = None

        return {
            "verdict": verdict,
            "confidence": confidence,
            "level": level,
            "entry_ref": entry_ref,
            "raw": raw_output,
        }

    @property
    def memory_is_stable(self) -> bool:
        """Whether memory has been stable (no updates) for several steps."""
        return self.consecutive_nones >= 5

    def reset_stability(self) -> None:
        """Reset stability tracker."""
        self.consecutive_nones = 0
