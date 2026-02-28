from training.verl.grouped_branching import (
    GroupedBranchingConfig,
    GroupedBranchingEngine,
)


def test_only_selected_branch_commits() -> None:
    engine = GroupedBranchingEngine(GroupedBranchingConfig(), seed=0)
    candidates = engine.evaluate_group(
        module="solver",
        sample_fn=lambda i: {"output": f"ACTION{i+1}", "idx": i},
        reward_fn=lambda payload: float(payload["idx"]),
    )
    committed: list[str] = []
    logged: list[str] = []
    engine.commit_selected(
        candidates=candidates,
        selected_index=2,
        commit_fn=lambda c: committed.append(c.candidate_id),
        log_fn=lambda c: logged.append(c.candidate_id),
    )
    assert committed == [candidates[2].candidate_id]
    assert len(logged) == len(candidates) - 1
