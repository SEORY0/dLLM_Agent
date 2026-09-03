"""Plateau portfolio policy tests."""

from dllm_agent.revision.models import ProbeFeedback
from dllm_agent.revision.portfolio import (
    LineageState,
    PortfolioPolicy,
    revision_due,
    update_lineage,
)


def test_update_lineage_resets_stalls_when_probe_reward_improves() -> None:
    """Given a plateaued lineage, when probe reward improves, then stalls reset."""
    # Given
    policy = PortfolioPolicy(plateau_epochs=2, proof_family_attempt_cap=3)
    first = update_lineage(LineageState(lineage_id="alpha"), _feedback("alpha", new_edges=1))
    stalled_once = update_lineage(first, _feedback("alpha", new_edges=1))
    plateaued = update_lineage(stalled_once, _feedback("alpha", new_edges=0))
    assert revision_due(plateaued, policy)

    # When
    improved = update_lineage(plateaued, _feedback("alpha", new_edges=3))

    # Then
    assert improved.consecutive_non_improving_epochs == 0
    assert improved.best_feedback_sha256 == improved.latest_feedback_sha256
    assert not revision_due(improved, policy)


def test_update_lineage_counts_tradeoff_as_non_improvement() -> None:
    """Given better branches but worse rare edges, when updated, then plateau advances."""
    # Given
    baseline = update_lineage(
        LineageState(lineage_id="tradeoff"),
        _feedback("tradeoff", branches=10, rare_edges=5, new_edges=2),
    )

    # When
    tradeoff = update_lineage(
        baseline,
        _feedback("tradeoff", branches=11, rare_edges=4, new_edges=2),
    )

    # Then
    assert tradeoff.consecutive_non_improving_epochs == 1
    assert tradeoff.best_feedback_sha256 == baseline.best_feedback_sha256


def test_revision_due_requires_threshold_and_uncapped_family() -> None:
    """Given mixed lineages, when stalls accrue, then only thresholded uncapped work is due."""
    # Given
    policy = PortfolioPolicy(plateau_epochs=2, proof_family_attempt_cap=2)
    warming = update_lineage(
        update_lineage(LineageState(lineage_id="warming"), _feedback("warming", new_edges=2)),
        _feedback("warming", new_edges=2),
    )
    plateaued = update_lineage(
        update_lineage(
            update_lineage(
                LineageState(lineage_id="plateaued"), _feedback("plateaued", new_edges=2)
            ),
            _feedback("plateaued", new_edges=1),
        ),
        _feedback("plateaued", new_edges=1),
    )
    capped_once = update_lineage(
        LineageState(lineage_id="capped"),
        _feedback("capped", new_edges=2),
        proof_family_id="asan-stack-a",
        proof_attempted=True,
    )
    capped_twice = update_lineage(
        capped_once,
        _feedback("capped", new_edges=1),
        proof_family_id="asan-stack-a",
        proof_attempted=True,
    )
    capped_plateau = update_lineage(capped_twice, _feedback("capped", new_edges=1))

    # When
    due_by_lineage = {
        state.lineage_id: revision_due(state, policy)
        for state in (warming, plateaued, capped_plateau)
    }

    # Then
    assert due_by_lineage == {"warming": False, "plateaued": True, "capped": False}


def _feedback(
    lineage_id: str,
    *,
    new_edges: int,
    branches: int | None = None,
    rare_edges: int = 0,
) -> ProbeFeedback:
    target_branches_hit = new_edges if branches is None else branches
    return ProbeFeedback(
        particle_id=lineage_id,
        snapshot_sha256=("1" * 64),
        build_status="skipped",
        run_status="completed",
        valid_snapshot=True,
        sanitizer_observed=False,
        target_branches_hit=target_branches_hit,
        new_edges=new_edges,
        rare_edges=rare_edges,
        value_profile_features=0,
        executions_per_second=100,
        trajectory_uncertainty=0.2,
        corpus_artifact_sha256=(),
        wall_ms=1000,
        build_log_sha256=("2" * 64),
        run_log_sha256=("3" * 64),
        build_argv_sha256=("4" * 64),
        run_argv_sha256=("5" * 64),
        rejected_count=0,
        dedup_count=0,
    )
