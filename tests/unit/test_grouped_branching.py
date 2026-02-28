from training.verl.grouped_branching import (
    GroupedBranchingConfig,
    GroupedBranchingEngine,
)


def test_candidate_counts() -> None:
    config = GroupedBranchingConfig(k_curiosity=4, k_learner=4, k_solver=4)
    assert config.count_for_module("curiosity") == 4
    assert config.count_for_module("learner") == 4
    assert config.count_for_module("solver") == 4


def test_commit_selected_candidate_only_mutates_selected() -> None:
    engine = GroupedBranchingEngine(GroupedBranchingConfig(), seed=0)
    candidates = engine.evaluate_group(
        module="curiosity",
        sample_fn=lambda i: {"output": f"ACTION{i+1}", "idx": i},
        reward_fn=lambda sample: float(sample["idx"]),
    )

    committed: list[str] = []
    logged: list[str] = []
    selected_index = 1
    engine.commit_selected(
        candidates=candidates,
        selected_index=selected_index,
        commit_fn=lambda c: committed.append(c.candidate_id),
        log_fn=lambda c: logged.append(c.candidate_id),
    )

    assert committed == [candidates[selected_index].candidate_id]
    assert len(logged) == len(candidates) - 1
