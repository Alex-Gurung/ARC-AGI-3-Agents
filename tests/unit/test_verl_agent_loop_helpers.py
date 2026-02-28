from training.verl.agent_loop_ls20 import LS20GroupedRunner, SerializedAction


def test_metadata_action_events_decodes_dict_rows() -> None:
    fallback = SerializedAction(name="ACTION1", data={}, reasoning=None)
    raw = [
        {"name": "ACTION2", "data": {"x": 1}, "reasoning": "r1"},
        {"name": "RESET", "data": {}, "reasoning": None},
    ]
    events = LS20GroupedRunner._metadata_action_events(raw, fallback=fallback)
    assert [event.name for event in events] == ["ACTION2", "RESET"]
    assert events[0].data == {"x": 1}


def test_mode_to_boundary_level_mapping() -> None:
    assert LS20GroupedRunner._mode_to_boundary_level("LEARN_ACTION") == "action"
    assert LS20GroupedRunner._mode_to_boundary_level("LEARN_SUBGOAL") == "subgoal"
    assert LS20GroupedRunner._mode_to_boundary_level("LEARN_PLAN") == "plan"
    assert LS20GroupedRunner._mode_to_boundary_level("SOLVE") == "plan"
    assert LS20GroupedRunner._mode_to_boundary_level("UNKNOWN") == "action"
