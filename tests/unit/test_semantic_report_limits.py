from types import SimpleNamespace

from agents.templates.loop_agent.learner import Learner
from agents.templates.loop_agent.memory import Memory


class _DummyChatCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_messages = None

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.last_messages = kwargs.get("messages")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _DummyClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_DummyChatCompletions(content))


def test_semantic_report_sentence_cap() -> None:
    content = (
        "ANSWER: One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve."
    )
    learner = Learner(client=_DummyClient(content), model="dummy")  # type: ignore[arg-type]
    report = learner.generate_semantic_transition_report(
        state_before="BEFORE",
        action_taken="ACTION1",
        state_after="AFTER",
        diff_text="DIFF",
        memory=Memory(),
        max_sentences=10,
    )
    assert report.count(".") <= 10
    assert "Eleven" not in report


def test_multimodal_payload_order_for_semantic_report() -> None:
    client = _DummyClient("ANSWER: first sentence.")
    learner = Learner(client=client, model="dummy")  # type: ignore[arg-type]
    _ = learner.generate_semantic_transition_report(
        state_before="BEFORE",
        action_taken="ACTION1",
        state_after="AFTER",
        diff_text="DIFF",
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
