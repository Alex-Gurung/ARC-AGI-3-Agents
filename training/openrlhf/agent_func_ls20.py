"""OpenRLHF multi-turn agent function for ARC-AGI-3 game ls20.

This trains a single policy to output one environment action each turn.
The environment feedback includes full encoded state text, and reward combines
novel-state bonuses, transition magnitude, level progress, and terminal outcomes.

Expected model output (final line):
    ANSWER: ACTION1
or
    ANSWER: ACTION6 32 16
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from arc_agi import Arcade
from arcengine import FrameData, FrameDataRaw, GameAction, GameState

from agents.templates.loop_agent.state_encoder import StateEncoder
from training.rewards import action_reward

try:
    from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
except ImportError as import_err:  # pragma: no cover - only hit when openrlhf missing
    raise RuntimeError(
        "OpenRLHF is required to use training/openrlhf/agent_func_ls20.py. "
        "Install openrlhf first."
    ) from import_err

logger = logging.getLogger(__name__)


def _convert_raw_frame_data(raw: FrameDataRaw | None) -> FrameData:
    if raw is None:
        raise ValueError("Received None frame data from environment")
    return FrameData(
        game_id=raw.game_id,
        frame=[arr.tolist() for arr in raw.frame],
        state=raw.state,
        levels_completed=raw.levels_completed,
        win_levels=raw.win_levels,
        guid=raw.guid,
        full_reset=raw.full_reset,
        available_actions=raw.available_actions,
    )


def _available_action_names(frame: FrameData) -> list[str]:
    return [f"ACTION{a}" if a > 0 else "RESET" for a in frame.available_actions]


def _grid_signature(grid: list[list[int]]) -> str:
    if not grid or not grid[0]:
        return "empty"
    height = len(grid)
    width = len(grid[0])
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(f"{width}x{height}|".encode("ascii"))
    for row in grid:
        hasher.update(bytes(row))
    return hasher.hexdigest()


class LS20AgentInstance(AgentInstanceBase):
    """One ARC environment instance for OpenRLHF multi-turn rollouts."""

    def __init__(self) -> None:
        super().__init__()
        self._arcade: Arcade | None = None
        self._env: Any = None
        self._card_id: str | None = None
        self._frame: FrameData | None = None
        self._state_encoder = StateEncoder(keyframe_interval=10)
        self._episode_step: int = 0
        self._max_steps: int = 200
        self._seen_signatures: set[str] = set()
        self._running_return: float = 0.0

    async def reset(self, states: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Start a fresh ls20 episode and return initial state dict."""
        del kwargs
        self._close_scorecard()

        game_id = str(states.get("game_id", "ls20"))
        self._max_steps = int(states.get("max_steps", 200))

        self._arcade = Arcade()
        self._card_id = self._arcade.open_scorecard(tags=["openrlhf", "ls20"])
        self._env = self._arcade.make(game_id, scorecard_id=self._card_id)

        raw_initial = self._env.observation_space
        self._frame = _convert_raw_frame_data(raw_initial)

        self._state_encoder.reset()
        self._episode_step = 0
        self._running_return = 0.0

        grid = self._frame.frame[-1] if self._frame.frame else []
        sig = f"{self._frame.state.name}:{_grid_signature(grid)}"
        self._seen_signatures = {sig}

        state_text = self._state_encoder.encode(self._frame)
        observation = self._build_observation(
            state_text=state_text,
            frame=self._frame,
            last_reward=0.0,
            done=False,
            parse_warning="none",
        )

        out = dict(states)
        out.update(
            {
                "game_id": game_id,
                "episode_step": self._episode_step,
                "max_steps": self._max_steps,
                "running_return": self._running_return,
                "observation": observation,
            }
        )
        return out

    async def step(
        self,
        states: dict[str, Any],
        action_text: str,
        **kwargs: Any,
    ) -> tuple[str, float, bool, dict[str, Any], dict[str, Any]]:
        """Apply one model-proposed action and return OpenRLHF step tuple."""
        del kwargs
        if self._env is None or self._frame is None:
            raise RuntimeError("Agent instance not initialized. reset() must run before step().")

        frame_before = self._frame
        grid_before = frame_before.frame[-1] if frame_before.frame else []
        available_actions = _available_action_names(frame_before)

        action, parsed_name, parse_warning = self._parse_action_text(
            action_text=action_text,
            available_actions=available_actions,
        )

        data = action.action_data.model_dump()
        raw_after = self._env.step(
            action,
            data=data,
            reasoning=data.get("reasoning", {}),
        )
        frame_after = _convert_raw_frame_data(raw_after)

        grid_after = frame_after.frame[-1] if frame_after.frame else []
        num_changed = self._state_encoder.get_num_changed_cells(grid_before, grid_after)
        state_text_after = self._state_encoder.encode(frame_after)

        sig_after = f"{frame_after.state.name}:{_grid_signature(grid_after)}"
        unseen_state = sig_after not in self._seen_signatures
        self._seen_signatures.add(sig_after)

        level_delta = frame_after.levels_completed - frame_before.levels_completed
        terminal_penalty = 0.5 if frame_after.state == GameState.GAME_OVER else 0.0
        novelty_signal = 1.0 if unseen_state else -0.15

        reward = action_reward(
            surprise=novelty_signal,
            changed_cells=num_changed,
            level_delta=level_delta,
            terminal_penalty=terminal_penalty,
        )
        if parse_warning != "none":
            reward -= 0.25
        if frame_after.state == GameState.WIN:
            reward += 20.0
        elif frame_after.state == GameState.GAME_OVER:
            reward -= 1.0

        self._running_return += reward
        self._episode_step += 1
        self._frame = frame_after

        done = bool(
            frame_after.state in (GameState.WIN, GameState.GAME_OVER)
            or self._episode_step >= self._max_steps
        )

        feedback = self._build_observation(
            state_text=state_text_after,
            frame=frame_after,
            last_reward=reward,
            done=done,
            parse_warning=parse_warning,
        )

        info = {
            "parsed_action": parsed_name,
            "parse_warning": parse_warning,
            "changed_cells": num_changed,
            "unseen_state": unseen_state,
            "levels_completed": frame_after.levels_completed,
            "game_state": frame_after.state.name,
            "episode_step": self._episode_step,
            "running_return": self._running_return,
        }

        out_states = dict(states)
        out_states.update(
            {
                "episode_step": self._episode_step,
                "running_return": self._running_return,
                "observation": feedback,
            }
        )

        if done:
            self._close_scorecard()

        return feedback, float(reward), done, info, out_states

    def _build_observation(
        self,
        *,
        state_text: str,
        frame: FrameData,
        last_reward: float,
        done: bool,
        parse_warning: str,
    ) -> str:
        available_actions = _available_action_names(frame)
        return (
            "ARC-AGI-3 environment feedback:\n"
            f"EPISODE_STEP: {self._episode_step}/{self._max_steps}\n"
            f"RUNNING_RETURN: {self._running_return:.3f}\n"
            f"LAST_REWARD: {last_reward:.3f}\n"
            f"PARSE_WARNING: {parse_warning}\n"
            f"DONE: {str(done).lower()}\n"
            "Use GRID/DIFF as ground truth; OBJECTS/RELATIONS are heuristic and may be noisy.\n"
            f"AVAILABLE_ACTIONS: {', '.join(available_actions)}\n"
            "\nSTATE:\n"
            f"{state_text}\n\n"
            "Respond with one final line only:\n"
            "ANSWER: <ACTION>\n"
            "or ANSWER: ACTION6 x y"
        )

    def _parse_action_text(
        self,
        *,
        action_text: str,
        available_actions: list[str],
    ) -> tuple[GameAction, str, str]:
        answer = self._extract_answer(action_text)
        text = answer.upper().strip()

        action6_match = re.search(r"ACTION6\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?", text)
        if action6_match and "ACTION6" in available_actions:
            x = max(0, min(63, int(action6_match.group(1))))
            y = max(0, min(63, int(action6_match.group(2))))
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y, "reasoning": "openrlhf_ls20"})
            return action, f"ACTION6 {x} {y}", "none"

        for candidate in available_actions:
            if candidate in text:
                if candidate == "RESET":
                    return GameAction.RESET, "RESET", "none"
                action = GameAction.from_name(candidate)
                action.reasoning = "openrlhf_ls20"
                return action, candidate, "none"

        # Fallback: first available non-reset, otherwise RESET.
        fallback = next((a for a in available_actions if a != "RESET"), "RESET")
        if fallback == "RESET":
            return GameAction.RESET, "RESET", "invalid_output_fallback"

        action = GameAction.from_name(fallback)
        action.reasoning = "openrlhf_ls20_fallback"
        return action, fallback, "invalid_output_fallback"

    @staticmethod
    def _extract_answer(raw_output: str) -> str:
        text = raw_output.strip()
        if not text:
            return ""

        parts = re.split(r"(?im)^\s*ANSWER\s*:\s*", text)
        if len(parts) > 1:
            return parts[-1].strip().splitlines()[0].strip()

        inline = re.split(r"(?i)\bANSWER\s*:\s*", text)
        if len(inline) > 1:
            return inline[-1].strip().splitlines()[0].strip()

        return text.splitlines()[-1].strip()

    def _close_scorecard(self) -> None:
        if self._arcade is None or self._card_id is None:
            return
        try:
            self._arcade.close_scorecard(self._card_id)
        except Exception as err:  # pragma: no cover - cleanup best effort
            logger.warning("Failed to close scorecard %s: %s", self._card_id, err)
        finally:
            self._card_id = None


agent = MultiTurnAgentExecutor(LS20AgentInstance)
