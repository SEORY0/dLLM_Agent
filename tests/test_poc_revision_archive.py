"""Tests for deterministic parent-child Pareto archive selection."""

from dataclasses import dataclass

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.archive import ArchiveCapacityError, ProofNovelty, select_parent_child
from dllm_agent.revision.controller_models import Particle, ProbeRecord, RevisionResult
from dllm_agent.revision.models import ProbeFeedback
from dllm_agent.targets.openexr.particle import (
    ExrParticle,
    FieldByteSpan,
    MaterializedSnapshot,
)

_ZERO_SHA = "0" * 64


def test_parent_survives_when_child_is_inferior() -> None:
    """Given dominated child feedback, when selecting archive, then parent remains active."""
    parent = _record(
        "parent",
        _FeedbackVector(
            sanitizer=True,
            branches=10,
            new_edges=8,
            rare_edges=2,
            value_features=4,
            execs=100,
        ),
        snapshot_payload=b"parent",
    )
    child = _record(
        "child",
        _FeedbackVector(
            sanitizer=False,
            branches=9,
            new_edges=7,
            rare_edges=1,
            value_features=3,
            execs=90,
        ),
        nfe=2,
        wall_ms=2,
        snapshot_payload=b"child",
    )

    selection = select_parent_child((parent,), (child,), capacity=1, protected_ids=frozenset())

    assert selection.active_ids == ("parent",)
    assert selection.active_keys == (_key(parent),)
    assert selection.dropped_ids == ("child",)
    assert selection.dropped_keys == (_key(child),)


def test_tradeoff_keeps_parent_and_child_when_capacity_permits() -> None:
    """Given incomparable feedback, when capacity permits both, then both remain active."""
    parent = _record(
        "parent",
        _FeedbackVector(branches=10, new_edges=1, rare_edges=0, value_features=0, execs=100),
        snapshot_payload=b"parent",
    )
    child = _record(
        "child",
        _FeedbackVector(branches=8, new_edges=4, rare_edges=3, value_features=2, execs=80),
        snapshot_payload=b"child",
    )

    selection = select_parent_child((parent,), (child,), capacity=2, protected_ids=frozenset())

    assert selection.active_ids == ("child", "parent")
    assert selection.active_keys == (_key(child), _key(parent))
    assert selection.dropped_ids == ()


def test_protected_anchor_survives_truncation() -> None:
    """Given a protected parent, when capacity truncates archive, then anchor survives."""
    protected = _record(
        "anchor",
        _FeedbackVector(branches=1, new_edges=1, rare_edges=0, value_features=0, execs=10),
        snapshot_payload=b"anchor",
    )
    stronger = _record(
        "stronger",
        _FeedbackVector(branches=100, new_edges=10, rare_edges=4, value_features=3, execs=50),
        snapshot_payload=b"stronger",
    )

    selection = select_parent_child(
        (protected,),
        (stronger,),
        capacity=1,
        protected_ids=frozenset({_key(protected)}),
    )

    assert selection.active_ids == ("anchor",)
    assert selection.active_keys == (_key(protected),)
    assert selection.protected_ids == ("anchor",)
    assert selection.dropped_ids == ("stronger",)
    assert selection.dropped_keys == (_key(stronger),)


def test_proof_novelty_and_snapshot_hash_break_final_ties() -> None:
    """Given equal vectors, when selecting one item, then novelty and hash make ties stable."""
    plain = _record("plain", _FeedbackVector(branches=4, new_edges=4), snapshot_payload=b"b")
    novel = _record("novel", _FeedbackVector(branches=4, new_edges=4), snapshot_payload=b"a")
    first = select_parent_child(
        (plain,),
        (novel,),
        capacity=1,
        protected_ids=frozenset(),
        proof_novelty_by_key={_key(novel): ProofNovelty.NEW_CAUSE},
    )
    second = select_parent_child(
        (novel,),
        (plain,),
        capacity=1,
        protected_ids=frozenset(),
        proof_novelty_by_key={_key(novel): ProofNovelty.NEW_CAUSE},
    )

    assert first.active_ids == ("novel",)
    assert first.active_keys == (_key(novel),)
    assert second.active_ids == ("novel",)
    assert second.active_keys == (_key(novel),)


def test_sanitizer_observed_does_not_dominate_or_rank_archive_candidates() -> None:
    """Given only sanitizer differs, when truncating, then snapshot hash breaks the tie."""
    sanitizer = _record(
        "sanitizer",
        _FeedbackVector(sanitizer=True, branches=5, new_edges=5),
        snapshot_payload=b"a",
    )
    plain = _record(
        "plain",
        _FeedbackVector(sanitizer=False, branches=5, new_edges=5),
        snapshot_payload=b"b",
    )

    selection = select_parent_child((sanitizer,), (plain,), capacity=1, protected_ids=frozenset())

    assert selection.active_keys == (_key(plain),)
    assert selection.dropped_keys == (_key(sanitizer),)


def test_parent_and_child_with_same_particle_id_remain_distinct_by_snapshot() -> None:
    """Given same particle id with different snapshots, when selecting, then both keys remain."""
    parent = _record("lineage", _FeedbackVector(branches=9, new_edges=1), snapshot_payload=b"p")
    child = _record("lineage", _FeedbackVector(branches=1, new_edges=9), snapshot_payload=b"c")

    selection = select_parent_child((parent,), (child,), capacity=2, protected_ids=frozenset())

    assert set(selection.active_keys) == {_key(parent), _key(child)}
    assert selection.dropped_keys == ()


def test_proof_novelty_is_keyed_by_candidate_snapshot_key() -> None:
    """Given duplicate particle ids, when one key is novel, then novelty applies only there."""
    plain = _record("lineage", _FeedbackVector(branches=4, new_edges=4), snapshot_payload=b"plain")
    novel = _record("lineage", _FeedbackVector(branches=4, new_edges=4), snapshot_payload=b"novel")

    selection = select_parent_child(
        (plain,),
        (novel,),
        capacity=1,
        protected_ids=frozenset(),
        proof_novelty_by_key={_key(novel): ProofNovelty.NEW_CAUSE},
    )

    assert selection.active_keys == (_key(novel),)
    assert selection.dropped_keys == (_key(plain),)


def test_crowding_distance_preserves_frontier_extremes_when_truncating() -> None:
    """Given a crowded frontier, when capacity is tight, then objective extremes survive."""
    branch_extreme = _record(
        "branches",
        _FeedbackVector(branches=10, new_edges=1, value_features=1, execs=1),
        snapshot_payload=b"branches",
    )
    middle = _record(
        "middle",
        _FeedbackVector(branches=6, new_edges=6, value_features=6, execs=6),
        snapshot_payload=b"middle",
    )
    edge_extreme = _record(
        "edges",
        _FeedbackVector(branches=1, new_edges=10, value_features=10, execs=10),
        snapshot_payload=b"edges",
    )

    selection = select_parent_child(
        (branch_extreme, middle, edge_extreme),
        (),
        capacity=2,
        protected_ids=frozenset(),
    )

    assert selection.active_keys == (_key(branch_extreme), _key(edge_extreme))
    assert selection.dropped_keys == (_key(middle),)


def test_protected_overflow_fails_closed() -> None:
    """Given too many protected anchors, when selecting, then archive rejects truncation."""
    first = _record("p1", _FeedbackVector(branches=1), snapshot_payload=b"p1")
    second = _record("p2", _FeedbackVector(branches=2), snapshot_payload=b"p2")

    with pytest.raises(ArchiveCapacityError, match="protected"):
        _ = select_parent_child(
            (first, second),
            (),
            capacity=1,
            protected_ids=frozenset({_key(first), _key(second)}),
        )


@dataclass(frozen=True, slots=True)
class _FeedbackVector:
    sanitizer: bool = False
    branches: int = 0
    new_edges: int = 0
    rare_edges: int = 0
    value_features: int = 0
    execs: int = 1


def _record(
    particle_id: str,
    vector: _FeedbackVector,
    *,
    nfe: int = 1,
    wall_ms: int = 1,
    snapshot_payload: bytes,
) -> ProbeRecord:
    snapshot_sha256 = sha256_bytes(snapshot_payload)
    return ProbeRecord(
        particle=Particle(
            particle_id=particle_id,
            result=RevisionResult(
                selectors=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                token_ids=(1,),
                slot_positions=(0,),
                selected_slot_token_ids=(1,),
                token_state_sha256=sha256_bytes(f"{particle_id}:token".encode()),
                transfer_evidence_sha256=sha256_bytes(f"{particle_id}:transfer".encode()),
                rng_state_sha256=sha256_bytes(f"{particle_id}:rng".encode()),
                nfe=nfe,
                forward_calls=nfe,
                gpu_ms=0,
                cache_hit=False,
                trajectory_uncertainty=0.0,
                suggested_field="width",
            ),
            snapshot=MaterializedSnapshot(
                particle=ExrParticle(
                    selectors=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    mode="validity-preserving",
                ),
                content=snapshot_payload,
                sha256=snapshot_sha256,
                anchor_sha256=_ZERO_SHA,
                field_spans=(FieldByteSpan(field="width", start=0, end=len(snapshot_payload)),),
                fault_field=None,
            ),
        ),
        feedback=ProbeFeedback(
            particle_id=particle_id,
            snapshot_sha256=snapshot_sha256,
            build_status="succeeded",
            run_status="completed",
            valid_snapshot=True,
            sanitizer_observed=vector.sanitizer,
            target_branches_hit=vector.branches,
            new_edges=vector.new_edges,
            rare_edges=vector.rare_edges,
            value_profile_features=vector.value_features,
            executions_per_second=vector.execs,
            trajectory_uncertainty=0.0,
            corpus_artifact_sha256=(),
            wall_ms=wall_ms,
            build_log_sha256=_ZERO_SHA,
            run_log_sha256=_ZERO_SHA,
            build_argv_sha256=_ZERO_SHA,
            run_argv_sha256=_ZERO_SHA,
            rejected_count=0,
            dedup_count=0,
        ),
        cache_hit=False,
    )


def _key(record: ProbeRecord) -> tuple[str, str]:
    return (record.particle.particle_id, record.particle.snapshot.sha256)
