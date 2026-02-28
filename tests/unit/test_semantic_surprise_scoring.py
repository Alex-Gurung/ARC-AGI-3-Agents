from types import SimpleNamespace

from agents.templates.loop_agent.memory import Memory
from agents.templates.loop_agent.surprise import (
    SemanticDebiasedNLLSurprise,
)


class _FakeLogprobs:
    def __init__(self, tokens, token_logprobs) -> None:
        self.tokens = tokens
        self.token_logprobs = token_logprobs


class _FakeChoice:
    def __init__(self, tokens, token_logprobs) -> None:
        self.logprobs = _FakeLogprobs(tokens=tokens, token_logprobs=token_logprobs)


class _FakeCompletions:
    def create(self, **kwargs):  # noqa: ANN003, ANN201
        prompt = kwargs.get("prompt", "")
        is_prompt_only = prompt.endswith("TRANSITION_REPORT:\n")
        has_action = "ACTION:\n" in prompt

        if is_prompt_only:
            tokens = ["p1", "p2", "p3", "p4", "p5"]
            lps = [-0.1] * len(tokens)
            return SimpleNamespace(choices=[_FakeChoice(tokens, lps)])

        prompt_tokens = ["p1", "p2", "p3", "p4", "p5"]
        completion_tokens = ["c1", "c2"]
        tokens = prompt_tokens + completion_tokens
        if has_action:
            completion_lps = [-1.0, -1.0]
        else:
            completion_lps = [-2.0, -2.0]
        lps = ([-0.1] * len(prompt_tokens)) + completion_lps
        return SimpleNamespace(choices=[_FakeChoice(tokens, lps)])


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


def test_semantic_debiased_nll_scoring() -> None:
    surprise = SemanticDebiasedNLLSurprise(client=_FakeClient(), model="dummy")  # type: ignore[arg-type]
    bundle = surprise.compute_bundle(
        state_before="S",
        action="ACTION1",
        memory=Memory(),
        semantic_report="moved right.",
        self_rated_x10=6.0,
    )
    assert bundle.mean_nll_full == 1.0
    assert bundle.mean_nll_stripped == 2.0
    assert bundle.debiased_nll == -1.0


def test_reward_source_switches() -> None:
    memory = Memory()
    debiased_engine = SemanticDebiasedNLLSurprise(
        client=_FakeClient(),  # type: ignore[arg-type]
        model="dummy",
        reward_source="debiased_nll",
    )
    self_engine = SemanticDebiasedNLLSurprise(
        client=_FakeClient(),  # type: ignore[arg-type]
        model="dummy",
        reward_source="self_rated",
    )
    hybrid_engine = SemanticDebiasedNLLSurprise(
        client=_FakeClient(),  # type: ignore[arg-type]
        model="dummy",
        reward_source="hybrid",
        self_rated_weight=0.5,
    )
    # Seed histories so z-scores are non-zero.
    for eng in (debiased_engine, self_engine, hybrid_engine):
        eng.debiased_history.extend([0.0, 1.0])
        eng.self_rated_history.extend([2.0, 8.0])

    b1 = debiased_engine.compute_bundle(
        state_before="S",
        action="ACTION1",
        memory=memory,
        semantic_report="moved.",
        self_rated_x10=9.0,
    )
    b2 = self_engine.compute_bundle(
        state_before="S",
        action="ACTION1",
        memory=memory,
        semantic_report="moved.",
        self_rated_x10=9.0,
    )
    b3 = hybrid_engine.compute_bundle(
        state_before="S",
        action="ACTION1",
        memory=memory,
        semantic_report="moved.",
        self_rated_x10=9.0,
    )
    assert b1.reward_value != b2.reward_value
    assert min(b1.reward_value, b2.reward_value) <= b3.reward_value <= max(
        b1.reward_value, b2.reward_value
    )
