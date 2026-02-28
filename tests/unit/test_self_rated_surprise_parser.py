from agents.templates.loop_agent.learner import Learner


def test_self_rated_parser_accepts_common_formats() -> None:
    assert Learner._parse_self_rated_surprise("SURPRISE_X10=7.5") == 7.5
    assert Learner._parse_self_rated_surprise("ANSWER: SURPRISE: 9") == 9.0
    assert Learner._parse_self_rated_surprise("x10: 12") == 10.0
    assert Learner._parse_self_rated_surprise("no score") == 0.0
