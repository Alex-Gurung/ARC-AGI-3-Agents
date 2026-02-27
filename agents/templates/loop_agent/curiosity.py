"""Curiosity component for the LoopAgent.

Proposes actions that maximize information gain at the current
abstraction level (action, subgoal, or plan).

Three separate prompt templates, one per level. The controller
decides which level to operate at; this component only produces
content at the assigned level.
"""

import logging
import re
from typing import Any

from openai import OpenAI

from .memory import Memory

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

ACTION_PROMPT = """\
You are exploring a game to understand how it works. Your goal is to propose an action that increases our knowledge of the game — either by trying something we haven't tried, or by testing an assumption we're unsure about.

PHASE: {phase}
LEVEL: {level}
ACTIVE_PLAN: {active_plan}
ACTIVE_SUBGOAL: {active_subgoal}
SUBGOAL_INDEX: {subgoal_index}

STATE:
{state_text}

MEMORY:
{memory_text}

UNSURE (low confidence entries):
{low_confidence_entries}

Choose an action that teaches us something new or tests a weak assumption. Choose from: {available_actions_str}

Briefly think step by step, then output exactly two final lines:
ANSWER: <one action from the list>
EXPECTED: <short description of what you expect to happen>
If selecting ACTION6, use coordinates:
ANSWER: ACTION6 x y"""

SUBGOAL_PROMPT = """\
You are exploring a game. Your goal is to propose a target or interaction that would increase our understanding of the game, or test something we're unsure about.

PHASE: {phase}
LEVEL: {level}
ACTIVE_PLAN: {active_plan}
ACTIVE_SUBGOAL: {active_subgoal}
SUBGOAL_INDEX: {subgoal_index}

STATE:
{state_text}

MEMORY:
{memory_text}

UNSURE (low confidence entries):
{low_confidence_entries}

Propose one specific thing to investigate or test (short phrase):
Briefly think step by step, then output exactly one final line:
ANSWER: <short goal phrase>"""

PLAN_PROMPT = """\
You are trying to understand how to solve a game. Propose an overall strategy that we can test. Focus on what we don't yet understand about how to win.

PHASE: {phase}
LEVEL: {level}
ACTIVE_PLAN: {active_plan}
ACTIVE_SUBGOAL: {active_subgoal}
SUBGOAL_INDEX: {subgoal_index}

MEMORY:
{memory_text}

Propose a strategy to test as ordered subgoals (max 5):
Briefly think step by step, then output exactly one final line:
ANSWER: <step1>; <step2>; <step3>"""


class Curiosity:
    """Proposes exploratory actions at the assigned abstraction level."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def propose_action(
        self,
        state_text: str,
        memory: Memory,
        available_actions: list[str],
        level: str = "action",
        phase: str = "UNKNOWN",
        active_plan: str = "none",
        active_subgoal: str = "none",
        subgoal_index: int | None = None,
    ) -> dict[str, Any]:
        """Propose an exploratory action/subgoal/plan.

        Args:
            state_text: Compressed state encoding.
            memory: Current memory.
            available_actions: List of available action names.
            level: Abstraction level ("action", "subgoal", "plan").
            phase: Current high-level phase ("EXPLORE"/"EXPLOIT").
            active_plan: Current active plan text, if any.
            active_subgoal: Current active subgoal text, if any.
            subgoal_index: Active subgoal index, if any.

        Returns:
            dict with keys:
                - "type": "action" | "subgoal" | "plan"
                - "value": the action name (for action) or text (for subgoal/plan)
                - "steps": parsed plan steps (for plan only)
                - "prediction": what the model expects to happen (str)
                - "raw": raw LLM output
        """
        memory_text = memory.to_text() if memory else "empty"

        # Build low-confidence entries text
        low_conf = memory.get_low_confidence(threshold=0.5) if memory else []
        if low_conf:
            low_confidence_entries = "\n".join(
                f"[{i}]{e.to_text()}" for i, e in low_conf
            )
        else:
            low_confidence_entries = "none"

        available_actions_str = ", ".join(available_actions)
        stage_subgoal_index = str(subgoal_index) if subgoal_index is not None else "none"
        active_plan_text = active_plan if active_plan else "none"
        active_subgoal_text = active_subgoal if active_subgoal else "none"

        if level == "action":
            prompt = ACTION_PROMPT.format(
                phase=phase,
                level=level,
                active_plan=active_plan_text,
                active_subgoal=active_subgoal_text,
                subgoal_index=stage_subgoal_index,
                state_text=state_text,
                memory_text=memory_text,
                low_confidence_entries=low_confidence_entries,
                available_actions_str=available_actions_str,
            )
        elif level == "subgoal":
            prompt = SUBGOAL_PROMPT.format(
                phase=phase,
                level=level,
                active_plan=active_plan_text,
                active_subgoal=active_subgoal_text,
                subgoal_index=stage_subgoal_index,
                state_text=state_text,
                memory_text=memory_text,
                low_confidence_entries=low_confidence_entries,
            )
        elif level == "plan":
            prompt = PLAN_PROMPT.format(
                phase=phase,
                level=level,
                active_plan=active_plan_text,
                active_subgoal=active_subgoal_text,
                subgoal_index=stage_subgoal_index,
                memory_text=memory_text,
            )
        else:
            logger.warning(f"Unknown level {level}, defaulting to action")
            level = "action"
            prompt = ACTION_PROMPT.format(
                phase=phase,
                level=level,
                active_plan=active_plan_text,
                active_subgoal=active_subgoal_text,
                subgoal_index=stage_subgoal_index,
                state_text=state_text,
                memory_text=memory_text,
                low_confidence_entries=low_confidence_entries,
                available_actions_str=available_actions_str,
            )

        raw_output = self._call_llm(prompt)
        answer_output = self._extract_answer(raw_output)
        prediction = self._extract_expected(raw_output)

        if level == "action":
            action_name = self._parse_action(answer_output, available_actions)
            return {
                "type": "action",
                "value": action_name,
                "prediction": prediction,
                "raw": raw_output,
            }
        elif level == "subgoal":
            return {
                "type": "subgoal",
                "value": (answer_output or raw_output).strip(),
                "prediction": prediction,
                "raw": raw_output,
            }
        else:  # plan
            plan_text = (answer_output or raw_output).strip()
            return {
                "type": "plan",
                "value": plan_text,
                "steps": self.parse_plan_steps(plan_text),
                "prediction": prediction,
                "raw": raw_output,
            }

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a single prompt, return raw text output."""
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
            logger.error(f"Curiosity LLM call failed: {e}")
            return ""

    def _parse_action(self, raw_output: str, available_actions: list[str]) -> str:
        """Parse an action name from LLM output.

        Tries to find a valid action name in the output.
        Falls back to first available action if parsing fails.
        """
        text = raw_output.upper().strip()

        # ACTION6 with coordinates (preserve coords)
        if "ACTION6" in available_actions:
            action6_match = re.search(r"ACTION6\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?", text)
            if action6_match:
                x = max(0, min(63, int(action6_match.group(1))))
                y = max(0, min(63, int(action6_match.group(2))))
                return f"ACTION6 {x} {y}"
            if "ACTION6" in text:
                # If chosen without coords, center is a safe fallback.
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
            f"Curiosity could not parse action from: {raw_output[:80]}. "
            f"Falling back to {available_actions[0]}"
        )
        return available_actions[0] if available_actions else "RESET"

    def _extract_expected(self, raw_output: str) -> str:
        """Extract the prediction after EXPECTED: marker."""
        text = raw_output.strip()
        matches = re.findall(r"(?im)^\s*EXPECTED\s*:\s*(.+)$", text)
        if matches:
            return matches[-1].strip()
        # Fallback: try inline
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
        # Fallback: content after last inline ANSWER:
        inline = re.split(r"(?i)\bANSWER\s*:\s*", text)
        if len(inline) > 1:
            return inline[-1].strip().splitlines()[0].strip()
        return text

    def parse_plan_steps(self, plan_text: str, max_steps: int = 5) -> list[str]:
        """Parse plan text into ordered subgoal steps."""
        text = plan_text.strip()
        if not text:
            return []

        # Prefer explicit separators.
        if ";" in text:
            parts = text.split(";")
        elif "->" in text:
            parts = text.split("->")
        else:
            parts = re.split(r"[\n\r]+", text)

        steps: list[str] = []
        for raw in parts:
            cleaned = re.sub(r"^\s*(?:\d+[\).:\-]?\s*|[-*]\s*)", "", raw).strip()
            if cleaned:
                steps.append(cleaned)

        if not steps and text:
            steps = [text]

        return steps[: max(1, max_steps)]
