"""Per-lineage frontier tracking for multi-ancestor revision backtracking.

This module does not reimplement Pareto selection. `select_lineage_frontier`
projects the existing `archive.select_parent_child` dominance/crowding
algorithm onto one lineage at a time so a lineage can keep more than one live
snapshot ("frontier_width" > 1) instead of the single active-scout the
`online-portfolio` pipeline uses today. `frontier_width=1` reduces to exactly
that existing behavior for a lineage with no cross-lineage archive
competitor (see `tests/test_poc_revision_search_tree.py`, which replays
`tests/test_poc_revision_archive.py`'s own fixtures at `frontier_width=1`).

`DenoNode` intentionally stores only hash identity (`key`/`parent_key`), never
the materialized snapshot, trajectory, or `ProbeRecord` payload. A node that
falls out of the frontier ("pruned") therefore has nothing heavy to release:
callers must simply stop carrying its `ProbeRecord` forward once it is no
longer in `LineageFrontier.nodes`.
"""

from dataclasses import dataclass
from typing import Literal

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.archive import (
    ArchiveCandidateCost,
    ArchiveCandidateKey,
    ProofNovelty,
    ProofNoveltyKey,
    lineage_candidates,
    select_parent_child,
)
from dllm_agent.revision.controller_models import ProbeRecord

type NodeStatus = Literal["frontier", "pruned"]


@dataclass(frozen=True, slots=True)
class DenoNode:
    """One archived snapshot's tree identity for one lineage."""

    key: ArchiveCandidateKey
    parent_key: ArchiveCandidateKey | None
    depth: int
    status: NodeStatus


@dataclass(frozen=True, slots=True)
class SearchAction:
    """Deterministic identity for one (node, field, positions) revision attempt."""

    action_id: str
    source_key: ArchiveCandidateKey
    field: str
    remask_positions: tuple[int, ...]


def search_action_id(
    source_key: ArchiveCandidateKey, field: str, remask_positions: tuple[int, ...]
) -> str:
    """Deterministically hash one (node, field, positions) revision attempt."""
    payload = ":".join(
        (source_key[0], source_key[1], field, ",".join(map(str, remask_positions)))
    )
    return sha256_bytes(payload.encode())


def search_action(
    source_key: ArchiveCandidateKey, field: str, remask_positions: tuple[int, ...]
) -> SearchAction:
    """Build one SearchAction with its deterministic action_id filled in."""
    return SearchAction(
        action_id=search_action_id(source_key, field, remask_positions),
        source_key=source_key,
        field=field,
        remask_positions=remask_positions,
    )


@dataclass(frozen=True, slots=True)
class LineageFrontier:
    """One lineage's bounded live frontier plus its pruned history."""

    lineage_id: str
    nodes: tuple[DenoNode, ...]
    pruned: tuple[DenoNode, ...]


def select_lineage_frontier(  # noqa: PLR0913 - frontier selection needs explicit policy inputs.
    lineage_id: str,
    parents: tuple[ProbeRecord, ...],
    children: tuple[ProbeRecord, ...],
    *,
    frontier_width: int,
    protected_ids: frozenset[ArchiveCandidateKey] | None = None,
    proof_novelty_by_key: dict[ProofNoveltyKey, ProofNovelty] | None = None,
    cost_by_key: dict[ArchiveCandidateKey, ArchiveCandidateCost] | None = None,
    parent_key_by_key: dict[ArchiveCandidateKey, ArchiveCandidateKey | None] | None = None,
    depth_by_key: dict[ArchiveCandidateKey, int] | None = None,
) -> LineageFrontier:
    """Select one lineage's own Pareto frontier, reusing `select_parent_child` unchanged."""
    own_parents = lineage_candidates(parents, lineage_id)
    own_children = lineage_candidates(children, lineage_id)
    selection = select_parent_child(
        own_parents,
        own_children,
        capacity=frontier_width,
        protected_ids=protected_ids or frozenset(),
        proof_novelty_by_key=proof_novelty_by_key,
        cost_by_key=cost_by_key,
    )
    resolved_parent_keys = parent_key_by_key or {}
    resolved_depths = depth_by_key or {}
    nodes = tuple(
        _node(candidate.key, "frontier", resolved_parent_keys, resolved_depths)
        for candidate in selection.active
    )
    pruned = tuple(
        _node(candidate.key, "pruned", resolved_parent_keys, resolved_depths)
        for candidate in selection.dropped
    )
    return LineageFrontier(lineage_id=lineage_id, nodes=nodes, pruned=pruned)


def _node(
    key: ArchiveCandidateKey,
    status: NodeStatus,
    parent_key_by_key: dict[ArchiveCandidateKey, ArchiveCandidateKey | None],
    depth_by_key: dict[ArchiveCandidateKey, int],
) -> DenoNode:
    return DenoNode(
        key=key,
        parent_key=parent_key_by_key.get(key),
        depth=depth_by_key.get(key, 0),
        status=status,
    )
