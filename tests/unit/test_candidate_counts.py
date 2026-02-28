from training.verl.grouped_branching import GroupedBranchingConfig


def test_candidate_counts_match_defaults() -> None:
    cfg = GroupedBranchingConfig(k_curiosity=4, k_learner=4, k_solver=4)
    assert cfg.count_for_module("curiosity") == 4
    assert cfg.count_for_module("learner") == 4
    assert cfg.count_for_module("solver") == 4
