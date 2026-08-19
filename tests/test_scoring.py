"""Trajectory scoring tests."""

from pathlib import Path

from dllm_agent.io import parse_json
from dllm_agent.models import ReplayFixture
from dllm_agent.scoring import (
    ScoreInputs,
    no_fixed_point_permutation,
    rank_trajectories,
    trajectory_score,
)

ROOT = Path(__file__).parents[1]


def test_score_uses_replay_fields_when_candidate_is_ranked() -> None:
    # Given
    inputs = ScoreInputs(0.9, 0.8, 0.7, 12)

    # When
    score = trajectory_score(inputs)

    # Then
    assert score == 0.81


def test_candidate_selection_is_deterministic_when_scores_are_ranked() -> None:
    # Given
    fixture = parse_json(
        ROOT / "fixtures/replay/stable-diffcoder-cwe121.json", ReplayFixture, "fixture"
    )

    # When
    ranked = rank_trajectories(fixture.trajectories)

    # Then
    assert tuple(item.candidate_id for item in ranked) == (
        "candidate-01",
        "candidate-02",
        "candidate-03",
        "candidate-04",
    )


def test_permutation_has_no_fixed_points_when_control_is_applied() -> None:
    # Given
    fixture = parse_json(
        ROOT / "fixtures/replay/stable-diffcoder-cwe121.json", ReplayFixture, "fixture"
    )
    candidate = fixture.trajectories[0]

    # When
    control = no_fixed_point_permutation(candidate)

    # Then
    assert all(left != right for left, right in zip(candidate.trace, control, strict=True))
    assert candidate.candidate_id == "candidate-01"


def test_accounting_preserves_nfe_when_candidates_are_ranked() -> None:
    # Given
    fixture = parse_json(
        ROOT / "fixtures/replay/stable-diffcoder-cwe121.json", ReplayFixture, "fixture"
    )

    # When
    ranked = rank_trajectories(fixture.trajectories)

    # Then
    assert sum(item.nfe for item in ranked) == sum(item.nfe for item in fixture.trajectories)
