from types import SimpleNamespace

from agents.templates.loop_agent.memory import Memory
from agents.templates.loop_agent.surprise import SemanticDebiasedNLLSurprise


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
            return SimpleNamespace(choices=[_FakeChoice(["p"] * 5, [-0.1] * 5)])
        completion_lps = [-1.0, -1.0] if has_action else [-2.0, -2.0]
        return SimpleNamespace(
            choices=[_FakeChoice(["p"] * 5 + ["c1", "c2"], [-0.1] * 5 + completion_lps)]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


def test_reward_source_switch_paths() -> None:
    memory = Memory()
    engines = [
        SemanticDebiasedNLLSurprise(  # type: ignore[arg-type]
            client=_FakeClient(), model="dummy", reward_source="debiased_nll"
        ),
        SemanticDebiasedNLLSurprise(  # type: ignore[arg-type]
            client=_FakeClient(), model="dummy", reward_source="self_rated"
        ),
        SemanticDebiasedNLLSurprise(  # type: ignore[arg-type]
            client=_FakeClient(),
            model="dummy",
            reward_source="hybrid",
            self_rated_weight=0.5,
        ),
    ]
    values: list[float] = []
    for engine in engines:
        engine.debiased_history.extend([0.0, 1.0])
        engine.self_rated_history.extend([2.0, 8.0])
        bundle = engine.compute_bundle(
            state_before="S",
            action="ACTION1",
            memory=memory,
            semantic_report="report",
            self_rated_x10=9.0,
        )
        values.append(bundle.reward_value)

    assert values[0] != values[1]
    assert min(values[0], values[1]) <= values[2] <= max(values[0], values[1])
