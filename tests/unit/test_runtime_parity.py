from arcengine import FrameData, GameAction, GameState

from agents.templates.loop_agent.runtime import LoopRuntime


class FakeAgent:
    def __init__(self) -> None:
        self.decide_called = 0
        self.observe_called = 0
        self.observed_action: str | None = None

    def choose_action(self, frames, latest_frame):  # noqa: ANN001, ANN201
        del frames, latest_frame
        self.decide_called += 1
        return GameAction.ACTION1

    def _post_step(self, state_before, grid_before, action, frame_after):  # noqa: ANN001, ANN201
        del state_before, grid_before, frame_after
        self.observe_called += 1
        self.observed_action = action.name


def test_runtime_delegates_decide_and_observe() -> None:
    agent = FakeAgent()
    runtime = LoopRuntime(agent)  # type: ignore[arg-type]
    frame = FrameData(
        game_id="t",
        frame=[[[0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1],
    )
    action = runtime.step_decide(
        frames=[frame],
        latest_frame=frame,
        state_before="STATE",
        grid_before=[[0]],
    )
    assert action.name == "ACTION1"
    runtime.step_observe(
        state_before="STATE",
        grid_before=[[0]],
        action=action,
        frame_after=frame,
    )
    assert agent.decide_called == 1
    assert agent.observe_called == 1
    assert agent.observed_action == "ACTION1"
