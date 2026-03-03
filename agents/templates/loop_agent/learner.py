"""Learner component for the LoopAgent.

Contains multiple "agents" (same model, different prompts):
- Scribe (update): observes transitions, writes memory/rulebook entries
- World Model (predict_outcome): predicts next state from current state + action + memory
- Observer (observe_transition): describes what actually happened in a transition
- Judge (judge_similarity): scores how well a prediction matched observation (1-5)
- Consolidator (consolidate): periodic memory cleanup

Each LLM call is an atomic unit for RL credit assignment.
Surprise = 6 - judge_similarity (low similarity = high surprise).
"""

import logging
import os
import re

from openai import OpenAI

from .memory import Memory, apply_memory_operation, parse_memory_operation

logger = logging.getLogger(__name__)


LEARNER_PROMPT = """\
You are writing a rulebook that describes how a game works. Each entry is one lesson — an atomic, reusable fact that helps future decisions.
The rulebook should be detailed enough that someone reading ONLY the rulebook could understand what the game looks like and how it behaves.

You may optionally prefix entries with a category tag like [ACTION], [RULE], [GOAL], etc. for organization, but this is not required.
Valuable lessons include: what actions do, game mechanics/constraints, what colors/sprites represent, goal hypotheses, strategic insights, notable patterns.
If ACTION is one of MISSING_ACTION_LESSONS, prioritize documenting what that action does.

IMPORTANT: Check the RULEBOOK below before adding anything. If a similar entry already exists, use MODIFY to refine it or output NONE. Do NOT add duplicates. Actively REMOVE outdated, wrong, or redundant entries to keep the rulebook clean.
EVIDENCE RULE:
- Only add/modify lessons when there is direct evidence in BEFORE/AFTER/DIFF.
- If evidence is weak or ambiguous, output NONE.
- Every lesson must include explicit evidence in the justification after "|".
  Write what changed and why that supports the claim.
  Good evidence phrases: "DIFF shows ...", "BEFORE/AFTER changed ...", "observed in N attempts".
DESCRIPTIVE DETAIL:
- In the content, describe the visual/spatial effect: mention colors, positions, directions, and what the grid looks like after the change.
- In the justification, describe what you actually saw change: which cells moved, what colors appeared/disappeared, spatial relationships that shifted.
- Aim for ~20-30 words per side of the "|". Terse entries like "ACTION1 moves player up" are too vague — prefer "ACTION1 shifts the blue object (color 3) upward by 1 row, leaving its previous cell empty (black/0)".
CONFIDENCE IS REQUIRED on every ADD and MODIFY — always include (0.xx) at the end.
Low confidence is encouraged — write early hypotheses at 0.3 or 0.4 and MODIFY to increase later as evidence builds. Calibration:
- 0.20-0.50: early hypothesis from 1 observation or weak evidence. This is fine and expected.
- 0.50-0.75: moderate evidence (repeated consistent observations).
- 0.75-0.90: strong evidence, but still potentially falsifiable.
- 0.90-1.00: only for directly verified outcomes (e.g., clear WIN/level-complete condition) or many repeated confirmations.
For uncertain goal/plan lessons, prefix content with "Hypothesis:".

REFERENCE RULES:
- For MODIFY/REMOVE, ONLY use numeric indices shown in RULEBOOK, in brackets.
- Allowed reference form is exactly: [n]  (examples: [0], [5], [12]).
- If no valid index exists, do not guess; output NONE.

PHASE: {phase}
LEVEL: {level}
SUBGOAL_INDEX: {subgoal_index}
RULEBOOK_STATUS: {rulebook_status}
MISSING_ACTION_LESSONS: {missing_action_lessons}

BEFORE: {state_before}
ACTION: {action_taken}
PREDICTION: {prediction}
AFTER: {state_after}
DIFF: {diff_text}

IMAGES (visual context):
If images are attached, they show the actual game screenshots:
- Image 1: BEFORE (game state before the action)
- Image 2: AFTER (game state after the action)
- Image 3: Composite with BEFORE | AFTER | REMOVED | ADDED panels \
(REMOVED shows old colors of changed cells; ADDED shows new colors)
Use the images to understand what objects look like and the spatial layout. \
The text GRID/DIFF provides precise cell coordinates — use both together.
Treat OBJECTS/RELATIONS as helpful but possibly noisy heuristics.

RECENT_ACTIONS:
{action_history}

{memory_text}

WORLD_MODEL_PREDICTION:
{predicted_description}

OBSERVER_REPORT:
{observed_description}

SIMILARITY_SCORE: {similarity}/5

Compare what the world model predicted with what actually happened. Did the outcome match? Did this reveal something new, confirm a belief, or contradict something in the rulebook?

You may output MULTIPLE operations (one per line). Formats:
ADD detailed lesson (~20-30 words) | specific evidence (~20-30 words) (confidence 0-1)
ADD [TAG] detailed lesson (~20-30 words) | specific evidence (~20-30 words) (confidence 0-1)
MODIFY [n] corrected lesson | specific evidence for the correction (confidence 0-1)
REMOVE [n] | why this entry is wrong or redundant
NONE

Examples from another game (notice the descriptive detail — each entry paints a picture of what the grid looks like):
- ADD ACTION1 shifts the blue square (color 3) upward by 1 row, leaving its old cell empty (black/0); blocked if a dark wall (color 5) is directly above | DIFF shows the blue cell at row 6 col 2 disappeared and reappeared at row 5 col 2 in 2 consecutive attempts; dark cell at row 4 col 2 prevented further upward movement (0.65)
- ADD Dark grey cells (color 5) forming the border walls are impassable — movement actions have no effect when the player is adjacent to them in the movement direction | BEFORE/AFTER grids were identical across 3 attempts where blue object tried to move into color-5 cells at the grid boundary (0.7)
- ADD [VOCAB] Color 9 (bright red) is the player-controlled object — a single cell that responds to movement actions; color 5 (dark grey) forms static walls; color 0 (black) is empty traversable space | the red cell is the only region that changes position after actions while all other colored regions remain fixed across 4 observations (0.6)
- ADD Hypothesis: level completes when the red player (color 9) reaches the green cell (color 4) on the right border — possibly a target or exit | one attempt ended immediately after the red cell moved adjacent to the green cell, but needs more confirmation (0.4)
- MODIFY [3] ACTION1 shifts blue object up by 1 row in open space, but is blocked when a dark wall (color 5) or grid edge is directly above — not a universal upward move | DIFF showed zero cell changes when blue object was at row 1 (top edge) and again when color-5 wall was directly above (0.6)
- REMOVE [5] | contradicted: latest BEFORE/AFTER shows the object passed through what we thought was a wall, so the blocking rule was wrong
- NONE

Do not copy the example wording. Use these as format-only references and ground your output in the current BEFORE/AFTER/DIFF evidence.
Think step by step, then output ONLY operations (no rationale/prose labels).
ANSWER:
<one or more operations, one per line>"""

WORLD_MODEL_PROMPT = """\
You are predicting what will happen next in a grid-based game.

Think step by step through the image:

1. What do you see? Describe each distinct object by its visual appearance — \
color, shape, position (e.g., "a dark red block near the center", "a green \
maze structure filling most of the grid", "a small blue-and-black piece").

2. Based on your rulebook and what you know about the game, predict what \
will happen when the action is taken. Which object will be affected? Where \
will it end up? What will be revealed underneath?

Remember: when an object moves, it reveals the background color where it was.

ACTION about to be taken: {action_taken}

{memory_text}

TEXT STATE (for reference — use the image as primary):
{state_before}

Predict the state after the action in 3-5 sentences. Describe objects by \
their visual appearance, not text IDs. Be specific ("the dark red block \
moves up one row, revealing green background") not abstract ("a change \
occurs"). Account for every visible element.

Think step by step, then output exactly one final line:
ANSWER: <predicted state description>
"""

OBSERVER_PROMPT = """\
You are describing a game state transition that just occurred.

You have two separate images. The FIRST image is the BEFORE state. The \
SECOND image is the AFTER state. You may also have a third composite image \
with four panels: BEFORE | AFTER | REMOVED | ADDED (REMOVED highlights old \
colors of changed cells; ADDED highlights new colors).

Think step by step:

1. BEFORE (first image): List every distinct visual element you see — \
describe each by color, shape, and position. For example: "a large green \
maze structure filling the center", "a dark red block inside the maze near \
row 20", "a small blue-and-black piece at the left edge", "a grey-and-red \
symbol in the bottom-left corner", "a yellow background", "a status bar at \
the bottom with colored indicators".

2. AFTER (second image): Go through the SAME list of elements. For each \
one, say whether it stayed in the same place or changed. Be precise about \
direction — if something moved, say left/right/up/down relative to where \
it was in the BEFORE image.

3. Summary: What happened? Use these common game mechanics to interpret:
- An object moving reveals the background color where it was
- Objects can push other objects or be blocked by walls
- Groups of adjacent changed cells usually mean one event (one object moved)

The CELL CHANGES below are exact — use them to verify direction and extent:
{diff_text}

TEXT BEFORE:
{state_before}

TEXT AFTER:
{state_after}

Now output your description. You MUST mention every distinct visual element \
(player piece, structures, blocks, symbols, background, status bar, etc.) \
and say whether it changed or stayed the same. Describe objects by visual \
appearance, not text IDs.

Think step by step, then output exactly one final line:
ANSWER: <observed state description>
"""

JUDGE_PROMPT = """\
You are comparing two descriptions of a game state transition.

PREDICTED (what was expected to happen):
{predicted_description}

OBSERVED (what actually happened):
{observed_description}

Rate how well the prediction matched the observation on a 1-5 scale:
  1 = Completely wrong — predicted outcome has no relation to what happened
  2 = Wrong direction — predicted some change but the wrong kind
  3 = Partially right — got the general idea but missed key details
  4 = Mostly right — captured the main effect, minor details off
  5 = Exact match — predicted outcome matches observation

Output exactly one final line:
ANSWER: SIMILARITY=<1-5>
"""


CONSOLIDATION_PROMPT = """\
You are reviewing and consolidating a game rulebook. Your goal is to make it concise, non-redundant, and accurate.

{memory_text}

Review the rulebook above. Output operations to clean it up:
- REMOVE [n] | reason — delete redundant or contradicted entries
- MODIFY [n] merged/improved text | reason (confidence) — improve an entry by merging near-duplicates or fixing wording
- NONE — if the rulebook is already clean

Focus on:
1. Merge near-duplicate entries — if two entries say essentially the same thing, keep the better-worded one and REMOVE the other
2. Remove entries that are contradicted by higher-confidence entries
3. Improve clarity of poorly-worded entries

Do NOT add new entries. Only clean up existing ones.
Think step by step, then output ONLY operations (no rationale/prose labels).
ANSWER:
<operations, one per line>"""


class Learner:
    """Updates memory based on observed state transitions."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model
        self.consecutive_nones: int = 0  # track how many NONE ops in a row
        self.last_raw_output: str = ""
        self.last_answer_output: str = ""
        self.semantic_report_max_sentences = int(
            os.environ.get("SEMANTIC_REPORT_MAX_SENTENCES", "10")
        )
        self.semantic_observer_temperature = float(
            os.environ.get("SEMANTIC_OBSERVER_TEMPERATURE", "0.7")
        )
        self.semantic_observer_max_tokens = int(
            os.environ.get("SEMANTIC_OBSERVER_MAX_TOKENS", "1024")
        )

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
        image_diff_url: str | None = None,
        rulebook_status: str = "none",
        missing_action_lessons: str = "none",
        action_history: str = "no actions taken yet",
        predicted_description: str = "",
        observed_description: str = "",
        similarity: int = 0,
    ) -> bool:
        """Observe a state transition and update memory (the "scribe" agent).

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
            predicted_description: World model's predicted state (from predict_outcome).
            observed_description: Observer's description of actual state (from observe_transition).
            similarity: Judge's similarity score 1-5 (from judge_similarity).

        Returns:
            True if memory was changed, False otherwise.
        """
        memory_text = memory.to_text() if memory else "empty"
        stage_subgoal_index = str(subgoal_index) if subgoal_index is not None else "none"

        prompt = LEARNER_PROMPT.format(
            phase=phase,
            level=level,
            subgoal_index=stage_subgoal_index,
            rulebook_status=rulebook_status,
            missing_action_lessons=missing_action_lessons,
            state_before=state_before,
            action_taken=action_taken,
            prediction=prediction or "no prediction",
            state_after=state_after,
            diff_text=diff_text,
            memory_text=memory_text,
            action_history=action_history,
            predicted_description=predicted_description or "none",
            observed_description=observed_description or "none",
            similarity=similarity if similarity > 0 else "N/A",
        )

        images = [url for url in [image_before_url, image_after_url, image_diff_url] if url]
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

    def predict_outcome(
        self,
        *,
        state_before: str,
        action_taken: str,
        memory: Memory,
        image_before_url: str | None = None,
    ) -> str:
        """World Model agent: predict what the state will look like after the action.

        Uses the rulebook (memory) to inform predictions. Returns a semantic
        description of the predicted next state.
        """
        memory_text = memory.to_text() if memory else "empty"
        prompt = WORLD_MODEL_PROMPT.format(
            state_before=state_before,
            action_taken=action_taken,
            memory_text=memory_text,
        )
        images = [image_before_url] if image_before_url else None
        raw_output = self._call_llm(
            prompt,
            max_tokens=self.semantic_observer_max_tokens,
            temperature=self.semantic_observer_temperature,
            image_data_urls=images,
        )
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output
        return answer_output

    def observe_transition(
        self,
        *,
        state_before: str,
        state_after: str,
        diff_text: str,
        image_before_url: str | None = None,
        image_after_url: str | None = None,
        image_diff_url: str | None = None,
    ) -> str:
        """Observer agent: describe what actually happened in the transition.

        Produces a ground-truth semantic description of the observed state change.
        This is the reference side for the judge comparison.
        """
        prompt = OBSERVER_PROMPT.format(
            state_before=state_before,
            state_after=state_after,
            diff_text=diff_text,
        )
        images = [url for url in [image_before_url, image_after_url, image_diff_url] if url]
        raw_output = self._call_llm(
            prompt,
            max_tokens=self.semantic_observer_max_tokens,
            temperature=self.semantic_observer_temperature,
            image_data_urls=images or None,
        )
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output
        return answer_output

    def judge_similarity(
        self,
        predicted_description: str,
        observed_description: str,
    ) -> int:
        """Judge agent: score how well the prediction matched observation.

        Returns similarity score 1-5:
          1 = Completely wrong
          2 = Wrong direction
          3 = Partially right
          4 = Mostly right
          5 = Exact match

        Surprise can be derived as (6 - similarity) / 5.0 → [0, 1].
        """
        prompt = JUDGE_PROMPT.format(
            predicted_description=predicted_description or "(no prediction)",
            observed_description=observed_description or "(no observation)",
        )
        raw_output = self._call_llm(
            prompt,
            max_tokens=128,
            temperature=0.3,  # low temperature for consistent scoring
        )
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output
        return self._parse_similarity(answer_output)

    def consolidate(self, memory: Memory, current_step: int) -> int:
        """Review and consolidate the rulebook to remove duplicates and improve clarity.

        Returns number of operations applied.
        """
        if not memory or len(memory) < 2:
            return 0

        memory_text = memory.to_text()
        prompt = CONSOLIDATION_PROMPT.format(memory_text=memory_text)
        raw_output = self._call_llm(prompt, max_tokens=1024, temperature=0.7)
        answer_output = self._extract_answer(raw_output)
        self.last_raw_output = raw_output
        self.last_answer_output = answer_output

        operation_lines = self._extract_operation_lines(answer_output)
        ops_applied = 0
        # Process REMOVEs in reverse index order to avoid index shifting
        operations = []
        for line in operation_lines:
            if line.upper() == "NONE":
                continue
            operation = parse_memory_operation(line, current_step)
            operations.append(operation)

        # Sort removes by index descending so removals don't shift later indices
        removes = [op for op in operations if op.get("op") == "remove"]
        modifies = [op for op in operations if op.get("op") == "modify"]

        # Apply removes in reverse order
        removes.sort(key=lambda op: int(op.get("index", 0) or 0), reverse=True)
        for op in removes:
            if apply_memory_operation(memory, op, current_step):
                ops_applied += 1

        for op in modifies:
            if apply_memory_operation(memory, op, current_step):
                ops_applied += 1

        if ops_applied:
            logger.info(f"Consolidation applied {ops_applied} operations (step {current_step})")
        else:
            logger.debug(f"Consolidation: no changes needed (step {current_step})")

        return ops_applied

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

    @staticmethod
    def _parse_similarity(raw_output: str) -> int:
        """Parse similarity score 1-5 from judge output."""
        text = raw_output.strip()
        match = re.search(
            r"SIMILARITY\s*[:=]\s*([1-5])",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return int(match.group(1))
        # Fallback: first digit 1-5
        fallback = re.search(r"\b([1-5])\b", text)
        if fallback:
            return int(fallback.group(1))
        logger.warning(f"Could not parse similarity from: {raw_output[:80]}")
        return 3  # default to middle

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
