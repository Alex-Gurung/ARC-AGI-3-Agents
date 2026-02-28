from types import SimpleNamespace

from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory


class _DummyChatCompletions:
    def __init__(self) -> None:
        self.last_messages = None

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.last_messages = kwargs.get("messages")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ANSWER: report."))]
        )


class _DummyClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_DummyChatCompletions())


def test_multimodal_images_are_ordered_before_after_diff() -> None:
    client = _DummyClient()
    learner = Learner(client=client, model="dummy")  # type: ignore[arg-type]
    learner.generate_semantic_transition_report(
        state_before="B",
        action_taken="ACTION1",
        state_after="A",
        diff_text="D",
        memory=Memory(),
        image_before_url="data:image/png;base64,before",
        image_after_url="data:image/png;base64,after",
        image_diff_url="data:image/png;base64,diff",
    )
    messages = client.chat.completions.last_messages
    assert messages is not None
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[1]["image_url"]["url"].endswith("before")
    assert content[2]["image_url"]["url"].endswith("after")
    assert content[3]["image_url"]["url"].endswith("diff")
