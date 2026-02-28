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
STATE_VISITS: {state_visit_count}
ACTION_TRY_COUNTS: {action_try_counts}
EARLY_EXPLORATION_HINT: {early_exploration_hint}
RULEBOOK_STATUS: {rulebook_status}
MISSING_ACTION_LESSONS: {missing_action_lessons}
SEMANTIC_DISCOVERY_STATUS: {semantic_discovery_status}

STATE:
{state_text}

Use GRID/CHANGED DIFF as primary evidence.
Treat OBJECTS/RELATIONS as helpful but possibly noisy heuristics.

RECENT_ACTIONS:
{action_history}

{memory_text}

UNSURE (low confidence entries):
{low_confidence_entries}

Choose an action that teaches us something new or tests a weak assumption. Choose from: {available_actions_str}
If possible, prefer a less-tried action early before repeating the same move.
Prioritize novelty: when STATE_VISITS is high, aim for an unseen next state rather than repeating a known transition.
If MISSING_ACTION_LESSONS is not empty, prioritize those first to build the rule-book from scratch.
Also use actions to identify semantic object roles and mechanics (what each object does, what interactions trigger changes).
If the path forward is unclear, test an assumption that may be wrong (e.g., goal hypothesis, action effect, interaction precondition).

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
STATE_VISITS: {state_visit_count}
ACTION_TRY_COUNTS: {action_try_counts}
EARLY_EXPLORATION_HINT: {early_exploration_hint}
SEMANTIC_DISCOVERY_STATUS: {semantic_discovery_status}

STATE:
{state_text}

Use GRID/CHANGED DIFF as primary evidence.
Treat OBJECTS/RELATIONS as helpful but possibly noisy heuristics.

RECENT_ACTIONS:
{action_history}

{memory_text}

UNSURE (low confidence entries):
{low_confidence_entries}

Propose one specific thing to investigate or test (short phrase):
Prefer subgoals that may reach unseen states if this state has been visited repeatedly.
Prefer subgoals that disambiguate object semantics and mechanics (e.g., touch object A to object B, enter region C, click object D).
If progress is unclear, choose a subgoal that directly challenges a likely-false assumption.
Briefly think step by step, then output exactly one final line:
ANSWER: <short goal phrase>"""

PLAN_PROMPT = """\
You are trying to understand how to solve a game. Propose an overall strategy that we can test. Focus on what we don't yet understand about how to win.

PHASE: {phase}
LEVEL: {level}
ACTIVE_PLAN: {active_plan}
ACTIVE_SUBGOAL: {active_subgoal}
SUBGOAL_INDEX: {subgoal_index}
STATE_VISITS: {state_visit_count}
ACTION_TRY_COUNTS: {action_try_counts}
EARLY_EXPLORATION_HINT: {early_exploration_hint}
SEMANTIC_DISCOVERY_STATUS: {semantic_discovery_status}

RECENT_ACTIONS:
{action_history}

{memory_text}

Propose a strategy to test as ordered subgoals (max 5).
Formatting rules:
- Use plain text steps (no angle brackets like <...>).
- Each step should be concrete and testable from game state.
- Avoid placeholder words (e.g., "systematically", "confirm effect") unless you name the object/interaction.
- Strategy should progressively build semantic understanding: object identity -> interaction mechanics -> win-condition tests.
If the route to solve is unclear, include at least one step that tests a potentially incorrect assumption.
Briefly think step by step, then output exactly one final line:
ANSWER: test touching blue switch; move red block to pink tile; test exit contact after trigger"""

MODE_ROUTER_PROMPT = """\
You are an agent learning to play a game. You need to decide where to focus your learning next.
Look at what you currently know (your memory) and pick the ONE area with the biggest knowledge gap.
Do NOT pick an action or plan a move — only decide what to learn.

There are four modes, each targeting a different level of understanding:
- LEARN_ACTION: learn what individual actions do (e.g. "ACTION1 moves player up", "ACTION3 does nothing at a wall")
- LEARN_SUBGOAL: learn useful multi-step sequences that solve parts of a level (e.g. "go right 3 then up 2 to reach the switch")
- LEARN_PLAN: learn the overall goals of each level and how to combine subgoals into a winning strategy
- SOLVE: we have enough understanding at all levels — execute our best strategy now

When in doubt, pick the LOWEST level with weak understanding — filling low-level gaps first is more efficient.

Here is your current status:
CURRENT_MODE: {current_mode}
LAST_DIAGNOSIS_LEVEL: {last_diagnosis_level}
LAST_PREDICTION: {last_prediction}
ACTIVE_PLAN: {active_plan}
ACTIVE_SUBGOAL: {active_subgoal}
RULEBOOK_STATUS: {rulebook_status}
MISSING_ACTION_LESSONS: {missing_action_lessons}
SEMANTIC_DISCOVERY_STATUS: {semantic_discovery_status}

RECENT_ACTIONS:
{action_history}

{memory_text}

Examples of good reasoning (pick exactly one mode):
- "Memory has no entry for ACTION2 and MISSING_ACTION_LESSONS lists it → ANSWER: LEARN_ACTION"
- "All actions are known but we've never tried combining them to reach the exit → ANSWER: LEARN_SUBGOAL"
- "We can reach objects but don't know what the win condition is → ANSWER: LEARN_PLAN"
- "We have high-confidence entries at every level and a working plan → ANSWER: SOLVE"

Now look at the memory and status above. Identify the weakest area, then pick exactly one mode.
ANSWER: """

EXPLORE_SUBGOAL_SEQUENCE_PROMPT = """\
You are exploring a game and currently testing a subgoal.
Definition: a subgoal is a short plan (up to {max_steps} actions) expected to cause a significant, testable state change that should help level completion.

PHASE: {phase}
LEVEL: {level}
SUBGOAL_INDEX: {subgoal_index}
STATE_VISITS: {state_visit_count}
ACTION_TRY_COUNTS: {action_try_counts}
EARLY_EXPLORATION_HINT: {early_exploration_hint}
SEMANTIC_DISCOVERY_STATUS: {semantic_discovery_status}

STATE:
{state_text}

Use GRID/CHANGED DIFF as primary evidence.
Treat OBJECTS/RELATIONS as helpful but possibly noisy heuristics.

RECENT_ACTIONS:
{action_history}

{memory_text}

ACTIVE_SUBGOAL: {active_subgoal}

Propose a subgoal attempt that teaches us something.
Formatting rules:
- SUBGOAL should name concrete target/object/interaction from current state.
- ACTION_SEQUENCE should be explicit actions only (no prose).
 - Prefer a sequence that is likely to reach an unseen state, not a repeated local loop.
 - Prefer sequences that isolate mechanics (single object interaction, trigger tests, precondition tests).
 - If uncertain, select a sequence designed to falsify a key assumption.
Use only: {available_actions_str}
If ACTION6 is used, include coordinates as ACTION6 x y.

Briefly think step by step, then output exactly four final lines:
SUBGOAL: <short subgoal statement>
SUCCESS_TEST: <observable condition showing subgoal success/failure>
ACTION_SEQUENCE: action1, action2, action3
EXPECTED: <short description of expected outcome>"""


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
        image_data_url: str | None = None,
        state_visit_count: int = 0,
        action_try_counts: str = "none",
        early_exploration_hint: str = "none",
        rulebook_status: str = "none",
        missing_action_lessons: str = "none",
        semantic_discovery_status: str = "none",
        action_history: str = "no actions taken yet",
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
            action_history: Recent action history text.

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
                state_visit_count=state_visit_count,
                action_try_counts=action_try_counts,
                early_exploration_hint=early_exploration_hint,
                rulebook_status=rulebook_status,
                missing_action_lessons=missing_action_lessons,
                semantic_discovery_status=semantic_discovery_status,
                state_text=state_text,
                memory_text=memory_text,
                low_confidence_entries=low_confidence_entries,
                available_actions_str=available_actions_str,
                action_history=action_history,
            )
        elif level == "subgoal":
            prompt = SUBGOAL_PROMPT.format(
                phase=phase,
                level=level,
                active_plan=active_plan_text,
                active_subgoal=active_subgoal_text,
                subgoal_index=stage_subgoal_index,
                state_visit_count=state_visit_count,
                action_try_counts=action_try_counts,
                early_exploration_hint=early_exploration_hint,
                rulebook_status=rulebook_status,
                missing_action_lessons=missing_action_lessons,
                semantic_discovery_status=semantic_discovery_status,
                state_text=state_text,
                memory_text=memory_text,
                low_confidence_entries=low_confidence_entries,
                action_history=action_history,
            )
        elif level == "plan":
            prompt = PLAN_PROMPT.format(
                phase=phase,
                level=level,
                active_plan=active_plan_text,
                active_subgoal=active_subgoal_text,
                subgoal_index=stage_subgoal_index,
                state_visit_count=state_visit_count,
                action_try_counts=action_try_counts,
                early_exploration_hint=early_exploration_hint,
                rulebook_status=rulebook_status,
                missing_action_lessons=missing_action_lessons,
                semantic_discovery_status=semantic_discovery_status,
                memory_text=memory_text,
                action_history=action_history,
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
                state_visit_count=state_visit_count,
                action_try_counts=action_try_counts,
                early_exploration_hint=early_exploration_hint,
                rulebook_status=rulebook_status,
                missing_action_lessons=missing_action_lessons,
                semantic_discovery_status=semantic_discovery_status,
                state_text=state_text,
                memory_text=memory_text,
                low_confidence_entries=low_confidence_entries,
                available_actions_str=available_actions_str,
                action_history=action_history,
            )

        raw_output = self._call_llm(prompt, image_data_urls=[image_data_url] if image_data_url else None)
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

    def propose_mode(
        self,
        state_text: str,
        memory: Memory,
        available_actions: list[str],
        current_mode: str,
        active_plan: str,
        active_subgoal: str,
        last_prediction: str,
        last_diagnosis_level: str,
        image_data_url: str | None = None,
        state_visit_count: int = 0,
        action_try_counts: str = "none",
        early_exploration_hint: str = "none",
        rulebook_status: str = "none",
        missing_action_lessons: str = "none",
        semantic_discovery_status: str = "none",
        action_history: str = "no actions taken yet",
    ) -> dict[str, Any]:
        """Choose the next top-level control mode based on knowledge gaps."""
        memory_text = memory.to_text() if memory else "empty"
        prompt = MODE_ROUTER_PROMPT.format(
            current_mode=current_mode,
            last_diagnosis_level=last_diagnosis_level or "none",
            last_prediction=last_prediction or "none",
            active_plan=active_plan or "none",
            active_subgoal=active_subgoal or "none",
            rulebook_status=rulebook_status,
            missing_action_lessons=missing_action_lessons,
            semantic_discovery_status=semantic_discovery_status,
            memory_text=memory_text,
            action_history=action_history,
        )
        raw_output = self._call_llm(prompt)
        answer_output = self._extract_answer(raw_output)
        mode = self._parse_mode(answer_output)
        return {"mode": mode, "raw": raw_output}

    def _call_llm(
        self,
        prompt: str,
        image_data_urls: list[str] | None = None,
    ) -> str:
        """Call the LLM with a single prompt, return raw text output."""
        content: str | list[dict[str, Any]]
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
                max_tokens=1024,
                temperature=1.0,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            if image_data_urls:
                logger.warning(
                    "Curiosity multimodal call failed; retrying text-only: %s",
                    e,
                )
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                        temperature=1.0,
                    )
                    content = response.choices[0].message.content or ""
                    return content.strip()
                except Exception as retry_err:
                    logger.error(f"Curiosity text-only retry failed: {retry_err}")
                    return ""
            logger.error(f"Curiosity LLM call failed: {e}")
            return ""

    def propose_subgoal_actions(
        self,
        state_text: str,
        memory: Memory,
        available_actions: list[str],
        max_steps: int,
        active_subgoal: str,
        phase: str = "EXPLORE",
        level: str = "subgoal",
        subgoal_index: int | None = None,
        image_data_url: str | None = None,
        state_visit_count: int = 0,
        action_try_counts: str = "none",
        early_exploration_hint: str = "none",
        rulebook_status: str = "none",
        missing_action_lessons: str = "none",
        semantic_discovery_status: str = "none",
        action_history: str = "no actions taken yet",
    ) -> dict[str, Any]:
        """Propose exploratory action sequence for a target subgoal."""
        memory_text = memory.to_text() if memory else "empty"
        stage_subgoal_index = str(subgoal_index) if subgoal_index is not None else "none"
        prompt = EXPLORE_SUBGOAL_SEQUENCE_PROMPT.format(
            phase=phase,
            level=level,
            subgoal_index=stage_subgoal_index,
            state_visit_count=state_visit_count,
            action_try_counts=action_try_counts,
            early_exploration_hint=early_exploration_hint,
            rulebook_status=rulebook_status,
            missing_action_lessons=missing_action_lessons,
            semantic_discovery_status=semantic_discovery_status,
            state_text=state_text,
            memory_text=memory_text,
            active_subgoal=active_subgoal or "none",
            available_actions_str=", ".join(available_actions),
            max_steps=max_steps,
            action_history=action_history,
        )
        raw_output = self._call_llm(prompt, image_data_urls=[image_data_url] if image_data_url else None)
        subgoal_text = self._extract_labeled_value(raw_output, "SUBGOAL") or active_subgoal
        success_test = self._extract_labeled_value(raw_output, "SUCCESS_TEST")
        action_payload = self._extract_labeled_value(raw_output, "ACTION_SEQUENCE")
        answer_output = action_payload or self._extract_answer(raw_output)
        prediction = self._extract_expected(raw_output)
        actions = self._parse_action_list(answer_output, available_actions, max_steps)
        return {
            "subgoal": (subgoal_text or "").strip(),
            "success_test": (success_test or "").strip(),
            "actions": actions,
            "prediction": prediction,
            "raw": raw_output,
        }

    def _parse_action_list(
        self,
        raw_output: str,
        available_actions: list[str],
        max_steps: int,
    ) -> list[str]:
        """Parse a comma/space separated list of actions, preserving ACTION6 coords."""
        text = raw_output.upper()
        out: list[str] = []

        action6_matches = re.findall(r"ACTION6\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?", text)
        action6_full: list[str] = []
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

    def _extract_labeled_value(self, raw_output: str, label: str) -> str:
        """Extract value from 'LABEL: value' style lines."""
        pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+)$"
        matches = re.findall(pattern, raw_output.strip())
        if matches:
            return matches[-1].strip()
        return ""

    def _parse_mode(self, raw_output: str) -> str:
        text = raw_output.upper()
        for mode in ("LEARN_ACTION", "LEARN_SUBGOAL", "LEARN_PLAN", "SOLVE"):
            if mode in text:
                return mode
        logger.warning("Could not parse mode from router output; defaulting to LEARN_ACTION")
        return "LEARN_ACTION"

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
            cleaned = cleaned.strip().strip("<>").strip().strip("\"'")
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            lowered = cleaned.lower()
            if lowered in {"step", "step 1", "step 2", "step 3", "todo", "tbd"}:
                continue
            if re.fullmatch(r"steps?\s*\d*", lowered):
                continue
            if re.fullmatch(r"step[_\-\s]*\d+", lowered):
                continue
            if cleaned:
                steps.append(cleaned)

        if not steps and text:
            lowered_text = text.lower().strip()
            if (
                re.fullmatch(r"steps?\s*\d*", lowered_text)
                or re.fullmatch(r"step[_\-\s]*\d+", lowered_text)
                or re.fullmatch(r"(step[_\-\s]*\d+\s*;?\s*)+", lowered_text)
            ):
                return []
            steps = [text]

        return steps[: max(1, max_steps)]
