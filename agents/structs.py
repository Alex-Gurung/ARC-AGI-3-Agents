"""Compatibility re-exports for tests expecting agents.structs."""

from arc_agi.scorecard import Card, Scorecard
from arcengine import ActionInput, FrameData, GameAction, GameState

__all__ = [
    "ActionInput",
    "Card",
    "FrameData",
    "GameAction",
    "GameState",
    "Scorecard",
]
