"""Tests for per-lineage search-tree frontier selection and backtracking."""

from dataclasses import fields

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.archive import ArchiveCandidateKey, select_parent_child
from dllm_agent.revision.controller_models import Particle, ProbeRecord, RevisionResult
from dllm_agent.revision.models import ProbeFeedback
from dllm_agent.revision.portfolio import EmptyFrontierError, choose_backtrack_key
from dllm_agent.revision.search_tree import (
    LineageFrontier,
    search_action,
    search_action_id,
    select_lineage_frontier,
)
from dllm_agent.targets.openexr.particle import ExrParticle, FieldByteSpan, MaterializedSnapshot

_ZERO_SHA = "0" * 64


def test_search_action_id_is_deterministic() -> None:
    """Given the same inputs, when hashing twice, then the action id matches."""
    first = search_action_id(("p0", "snap-a"), "width", (4, 5))
    second = search_action_id(("p0", "snap-a"), "width", (4, 5))

    assert first == second


def test_search_action_id_is_field_sensitive() -> None:
    """Given a different field, when hashing, then the action id changes."""
    width_id = search_action_id(("p0", "snap-a"), "width", (4, 5))
    height_id = search_action_id(("p0", "snap-a"), "height", (4, 5))

    assert width_id != height_id


def test_search_action_bundles_its_own_id() -> None:
    """Given source/field/positions, when building a SearchAction, then id matches the helper."""
    action = search_action(("p0", "snap-a"), "width", (4, 5))

    assert action.action_id == search_action_id(("p0", "snap-a"), "width", (4, 5))
    assert action.source_key == ("p0", "snap-a")
    assert action.remask_positions == (4, 5)


def test_frontier_width_one_matches_existing_single_active_scout_regression() -> None:
    """Given the archive's own dominated-child fixture, when frontier_width=1, then the
    lineage frontier picks exactly the same survivor as the existing global selection."""
    parent = _record("lineage", branches=10, new_edges=8, snapshot_payload=b"parent")
    child = _record("lineage", branches=9, new_edges=7, snapshot_payload=b"child")
    baseline = select_parent_child((parent,), (child,), capacity=1, protected_ids=frozenset())

    frontier = select_lineage_frontier("lineage", (parent,), (child,), frontier_width=1)

    assert tuple(node.key for node in frontier.nodes) == baseline.active_keys
    assert tuple(node.key for node in frontier.pruned) == baseline.dropped_keys


def test_frontier_width_two_keeps_both_incomparable_snapshots() -> None:
    """Given incomparable parent/child feedback, when frontier_width=2, then both survive
    as live frontier nodes instead of only one active scout."""
    parent = _record("lineage", branches=10, new_edges=1, snapshot_payload=b"parent")
    child = _record("lineage", branches=8, new_edges=4, snapshot_payload=b"child")

    frontier = select_lineage_frontier("lineage", (parent,), (child,), frontier_width=2)

    assert {node.key for node in frontier.nodes} == {_key(parent), _key(child)}
    assert all(node.status == "frontier" for node in frontier.nodes)
    assert frontier.pruned == ()


def test_frontier_ignores_other_lineages() -> None:
    """Given records from another lineage, when selecting one lineage's frontier, then the
    other lineage's snapshots never enter its nodes or pruned set."""
    own_parent = _record("a", branches=5, snapshot_payload=b"a-parent")
    other_parent = _record("b", branches=50, snapshot_payload=b"b-parent")

    frontier = select_lineage_frontier("a", (own_parent, other_parent), (), frontier_width=1)

    assert frontier.lineage_id == "a"
    assert tuple(node.key for node in frontier.nodes) == (_key(own_parent),)
    assert frontier.pruned == ()


def test_pruned_nodes_carry_no_snapshot_payload() -> None:
    """Given a dominated child, when it is pruned, then its DenoNode holds only hash identity."""
    parent = _record("lineage", branches=10, new_edges=8, snapshot_payload=b"parent")
    child = _record("lineage", branches=9, new_edges=7, snapshot_payload=b"child")

    frontier = select_lineage_frontier("lineage", (parent,), (child,), frontier_width=1)

    (pruned_node,) = frontier.pruned
    assert pruned_node.status == "pruned"
    assert pruned_node.key == _key(child)
    # DenoNode has no field capable of holding a ProbeRecord/snapshot payload.
    field_names = {field.name for field in fields(pruned_node)}
    assert field_names == {"key", "parent_key", "depth", "status"}


def test_choose_backtrack_key_is_identity_at_width_one() -> None:
    """Given a single-node frontier, when choosing a backtrack target, then it is that node."""
    key = choose_backtrack_key("lineage", (("lineage", "only"),), epoch=7)

    assert key == ("lineage", "only")


def test_choose_backtrack_key_round_robins_across_a_wider_frontier() -> None:
    """Given a two-node frontier, when epochs advance, then selection alternates
    deterministically."""
    frontier = (("lineage", "snap-a"), ("lineage", "snap-b"))

    chosen = [choose_backtrack_key("lineage", frontier, epoch) for epoch in range(4)]

    assert chosen == [
        ("lineage", "snap-a"),
        ("lineage", "snap-b"),
        ("lineage", "snap-a"),
        ("lineage", "snap-b"),
    ]


def test_choose_backtrack_key_rejects_an_empty_frontier() -> None:
    """Given no live frontier nodes, when choosing a backtrack target, then it fails closed."""
    with pytest.raises(EmptyFrontierError, match="lineage"):
        _ = choose_backtrack_key("lineage", (), epoch=0)


def test_lineage_frontier_is_a_plain_immutable_container() -> None:
    """Given a frontier result, when constructed directly, then equality is value-based."""
    left = LineageFrontier(lineage_id="a", nodes=(), pruned=())
    right = LineageFrontier(lineage_id="a", nodes=(), pruned=())

    assert left == right


def _record(
    particle_id: str, *, branches: int = 0, new_edges: int = 0, snapshot_payload: bytes
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
                nfe=1,
                forward_calls=1,
                gpu_ms=0,
                cache_hit=False,
                trajectory_uncertainty=0.0,
                suggested_field="width",
            ),
            snapshot=MaterializedSnapshot(
                particle=ExrParticle(
                    selectors=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), mode="validity-preserving"
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
            sanitizer_observed=False,
            target_branches_hit=branches,
            new_edges=new_edges,
            rare_edges=0,
            value_profile_features=0,
            executions_per_second=1,
            trajectory_uncertainty=0.0,
            corpus_artifact_sha256=(),
            wall_ms=1,
            build_log_sha256=_ZERO_SHA,
            run_log_sha256=_ZERO_SHA,
            build_argv_sha256=_ZERO_SHA,
            run_argv_sha256=_ZERO_SHA,
            rejected_count=0,
            dedup_count=0,
        ),
        cache_hit=False,
    )


def _key(record: ProbeRecord) -> ArchiveCandidateKey:
    return (record.particle.particle_id, record.particle.snapshot.sha256)
