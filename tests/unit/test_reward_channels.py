from training.reward_channels import (
    RewardComponents,
    group_z_normalize,
    scalarize_curiosity,
    scalarize_learner,
    scalarize_solver,
)


def test_scalarize_curiosity_action_matches_defaults() -> None:
    comps = RewardComponents(
        surprise=1.0,
        novelty=1.0,
        transition_magnitude=10.0,
        parse_warning_penalty=1.0,
    )
    # 1.0 + 0.2 + 0.1 - 0.25 = 1.05
    assert abs(scalarize_curiosity(comps, boundary_type="action") - 1.05) < 1e-6


def test_scalarize_learner_matches_defaults() -> None:
    comps = RewardComponents(
        mismatch_reduction=0.3,
        parse_bonus=1.0,
        dedup_bonus=1.0,
        contradiction_cleanup_bonus=1.0,
        missing_action_reduction_bonus=2.0,
        parse_warning_penalty=1.0,
    )
    # 0.3 + 0.1 + 0.1 + 0.1 + 0.5 - 0.1 = 1.0
    assert abs(scalarize_learner(comps) - 1.0) < 1e-6


def test_scalarize_solver_matches_defaults() -> None:
    comps = RewardComponents(
        level_delta_reward=1.0,
        win_bonus=1.0,
        game_over_penalty=0.0,
        step_penalty=2.0,
        solver_surprise_bonus=0.4,
    )
    # 10 + 50 - 0.1 + 0.2 = 60.1
    assert abs(scalarize_solver(comps) - 60.1) < 1e-6


def test_group_z_normalize_basic() -> None:
    vals = [1.0, 2.0, 3.0]
    z = group_z_normalize(vals)
    assert len(z) == 3
    assert abs(sum(z)) < 1e-6
