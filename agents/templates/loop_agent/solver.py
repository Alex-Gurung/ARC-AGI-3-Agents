"""Solver component for the LoopAgent.

Exploits current knowledge (memory) to make progress toward winning.
Uses the active plan and subgoal to guide action selection.
Falls back to exploratory behavior if memory/plan is empty.
"""

import logging
import re
from typing import Any, Optional

from openai import OpenAI

from .memory import Memory

logger = logging.getLogger(__name__)


SOLVER_PROMPT = """\
You are solving a game. Use your knowledge to win.

PHASE: {phase}
LEVEL: {level}
SUBGOAL_INDEX: {subgoal_index}

STATE:
{state_text}

MEMORY:
{memory_text}

PLAN: {active_plan}
SUBGOAL: {active_subgoal}

Pick the best action to make progress. Choose from: {available_actions_str}

Briefly think step by step, then output exactly two final lines:
ANSWER: <one action from the list>
EXPECTED: <short description of what you expect to happen>
If selecting ACTION6, use coordinates:
ANSWER: ACTION6 x y"""

SUBGOAL_SEQUENCE_PROMPT = """\
You are solving a game and currently have an active subgoal.
Definition: a subgoal is a short plan (up to {max_steps} actions) that should create a significant, observable state change toward level completion.

PHASE: {phase}
LEVEL: {level}
SUBGOAL_INDEX: {subgoal_index}

STATE:
{state_text}

MEMORY:
{memory_text}

PLAN: {active_plan}
SUBGOAL: {active_subgoal}

Propose a short sequence of actions to attempt this subgoal.
MAX_ACTIONS: {max_steps}

Actions must be from: {available_actions_str}
If ACTION6 is used, include coordinates as: ACTION6 x y
Briefly think step by step, then output exactly two final lines:
ANSWER: action1, action2, action3
EXPECTED: <short description of expected subgoal outcome>
"""


class Solver:
    """Selects actions to exploit learned knowledge and solve the game."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model
        self.active_plan: Optional[str] = None
        self.active_subgoal: Optional[str] = None

    def solve(
        self,
        state_text: str,
        memory: Memory,
        available_actions: list[str],
        phase: str = "EXPLOIT",
        level: str = "action",
        subgoal_index: int | None = None,
    ) -> dict[str, Any]:
        """Select an action based on current knowledge.

        Args:
            state_text: Compressed state encoding.
            memory: Current memory.
            available_actions: List of available action names.
            phase: Current high-level phase.
            level: Current abstraction level.
            subgoal_index: Active subgoal index, if any.

        Returns:
            dict with keys:
                - "action": action name string
                - "prediction": what the model expects to happen (str)
                - "raw": raw LLM output
        """
        memory_text = memory.to_text() if memory else "empty"
        available_actions_str = ", ".join(available_actions)
        stage_subgoal_index = str(subgoal_index) if subgoal_index is not None else "none"

        prompt = SOLVER_PROMPT.format(
            phase=phase,
            level=level,
            subgoal_index=stage_subgoal_index,
            state_text=state_text,
            memory_text=memory_text,
            active_plan=self.active_plan or "none",
            active_subgoal=self.active_subgoal or "none",
            available_actions_str=available_actions_str,
        )

        raw_output = self._call_llm(prompt)
        answer_output = self._extract_answer(raw_output)
        prediction = self._extract_expected(raw_output)
        action_name = self._parse_action(answer_output, available_actions)

        return {"action": action_name, "prediction": prediction, "raw": raw_output}

    def set_plan(self, plan: str) -> None:
        """Set the active plan (from curiosity at plan level)."""
        self.active_plan = plan
        logger.info(f"Solver: new plan set: {plan[:100]}")

    def set_subgoal(self, subgoal: str) -> None:
        """Set the active subgoal (from curiosity at subgoal level)."""
        self.active_subgoal = subgoal
        logger.info(f"Solver: new subgoal set: {subgoal[:100]}")

    def clear_plan(self) -> None:
        """Clear active plan and subgoal."""
        self.active_plan = None
        self.active_subgoal = None

    def clear_subgoal(self) -> None:
        """Clear active subgoal while keeping the plan."""
        self.active_subgoal = None

    @property
    def has_subgoal(self) -> bool:
        return bool(self.active_subgoal)

    def propose_subgoal_actions(
        self,
        state_text: str,
        memory: Memory,
        available_actions: list[str],
        max_steps: int = 20,
        phase: str = "EXPLOIT",
        level: str = "subgoal",
        subgoal_index: int | None = None,
    ) -> dict[str, Any]:
        """Propose a short action sequence for the active subgoal."""
        memory_text = memory.to_text() if memory else "empty"
        available_actions_str = ", ".join(available_actions)
        stage_subgoal_index = str(subgoal_index) if subgoal_index is not None else "none"
        prompt = SUBGOAL_SEQUENCE_PROMPT.format(
            phase=phase,
            level=level,
            subgoal_index=stage_subgoal_index,
            state_text=state_text,
            memory_text=memory_text,
            active_plan=self.active_plan or "none",
            active_subgoal=self.active_subgoal or "none",
            available_actions_str=available_actions_str,
            max_steps=max_steps,
        )
        raw_output = self._call_llm(prompt)
        answer_output = self._extract_answer(raw_output)
        prediction = self._extract_expected(raw_output)
        actions = self._parse_subgoal_actions(answer_output, available_actions, max_steps)
        return {"actions": actions, "prediction": prediction, "raw": raw_output}

    # Backward-compatible alias used by older harness code/tests.
    def propose_burst(
        self,
        state_text: str,
        memory: Memory,
        available_actions: list[str],
        max_steps: int = 20,
    ) -> dict[str, Any]:
        return self.propose_subgoal_actions(
            state_text=state_text,
            memory=memory,
            available_actions=available_actions,
            max_steps=max_steps,
        )

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> str:
        """Call the LLM with a single prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.error(f"Solver LLM call failed: {e}")
            return ""

    def _parse_subgoal_actions(
        self,
        raw_output: str,
        available_actions: list[str],
        max_steps: int,
    ) -> list[str]:
        text = raw_output.upper()
        out: list[str] = []

        # Capture ACTION6 with coordinates first so coordinate tokens are preserved.
        action6_matches = re.findall(
            r"ACTION6\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?",
            text,
        )
        action6_full = []
        for x_str, y_str in action6_matches:
            x = max(0, min(63, int(x_str)))
            y = max(0, min(63, int(y_str)))
            action6_full.append(f"ACTION6 {x} {y}")

        generic = re.findall(r"RESET|ACTION\s*\d+", text)
        for token in generic:
            token = token.replace(" ", "")
            if token == "ACTION6":
                if action6_full:
                    out.extend(action6_full)
                elif "ACTION6" in available_actions:
                    out.append("ACTION6 32 32")
            elif token in available_actions:
                out.append(token)

        valid: list[str] = []
        for action in out:
            if action.startswith("ACTION6"):
                if "ACTION6" in available_actions:
                    valid.append(action)
            elif action in available_actions:
                valid.append(action)

        if not valid:
            return [self._parse_action(raw_output, available_actions)]

        return valid[: max(1, max_steps)]

    def _parse_action(self, raw_output: str, available_actions: list[str]) -> str:
        """Parse an action name from LLM output."""
        text = raw_output.upper().strip()

        # ACTION6 with coordinates (preserve coords)
        if "ACTION6" in available_actions:
            action6_match = re.search(
                r"ACTION6\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?",
                text,
            )
            if action6_match:
                x = max(0, min(63, int(action6_match.group(1))))
                y = max(0, min(63, int(action6_match.group(2))))
                return f"ACTION6 {x} {y}"
            if "ACTION6" in text:
                return "ACTION6 32 32"

        # Direct match
        for action in available_actions:
            if action.upper() in text:
                return action

        # Try regex for ACTION followed by a digit
        match = re.search(r"ACTION\s*(\d+)", text)
        if match:
            action_name = f"ACTION{match.group(1)}"
            if action_name in available_actions:
                return action_name

        # Check for RESET
        if "RESET" in text and "RESET" in available_actions:
            return "RESET"

        # Fallback
        logger.warning(
            f"Solver could not parse action from: {raw_output[:80]}. "
            f"Falling back to {available_actions[0]}"
        )
        return available_actions[0] if available_actions else "RESET"

    def _extract_expected(self, raw_output: str) -> str:
        """Extract the prediction after EXPECTED: marker."""
        text = raw_output.strip()
        matches = re.findall(r"(?im)^\s*EXPECTED\s*:\s*(.+)$", text)
        if matches:
            return matches[-1].strip()
        inline = re.split(r"(?i)\bEXPECTED\s*:\s*", text)
        if len(inline) > 1:
            return inline[-1].strip().splitlines()[0].strip()
        return ""

    def _extract_answer(self, raw_output: str) -> str:
        """Extract the payload after the final ANSWER: marker if present."""
        text = raw_output.strip()
        matches = re.findall(r"(?im)^\s*ANSWER\s*:\s*(.+)$", text)
        if matches:
            return matches[-1].strip()
        inline = re.split(r"(?i)\bANSWER\s*:\s*", text)
        if len(inline) > 1:
            return inline[-1].strip().splitlines()[0].strip()
        return text
