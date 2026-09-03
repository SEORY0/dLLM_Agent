"""Validation and schedule helpers for selective-revision sampling."""

from __future__ import annotations

from typing import Final

from dllm_agent.targets.openexr.particle import FIELD_ORDER, MATERIALIZED_MUTABLE_FIELDS
from tools.runtime.selective_revision_json import ReviseRequest, WireProtocolError
from tools.runtime.selective_revision_template import Template, selected_indexes
from tools.selective_revision_payloads import sha256_token_ids

_MUTABLE_INDEXES: Final = frozenset(
    FIELD_ORDER.index(field) for field in MATERIALIZED_MUTABLE_FIELDS
)


def validate_parent(request: ReviseRequest, template: Template) -> None:
    """Validate parent token and bitmap invariants before selective revision."""
    parent = request.parent
    all_slots = (1 << len(template.slot_positions)) - 1
    if request.remask_bitmap & request.frozen_bitmap:
        raise WireProtocolError(reason="remask and frozen bitmaps overlap")
    if (request.remask_bitmap | request.frozen_bitmap) != all_slots:
        raise WireProtocolError(reason="remask and frozen bitmaps must cover every slot")
    if any(
        request.remask_bitmap & (1 << index)
        for index in range(len(FIELD_ORDER))
        if index not in _MUTABLE_INDEXES
    ):
        raise WireProtocolError(reason="remask selects non-materialized field")
    if request.parent_token_sha256 != sha256_token_ids(parent.token_ids):
        raise WireProtocolError(reason="parent token hash mismatch")
    if parent.token_state_sha256 != request.parent_token_sha256:
        raise WireProtocolError(reason="parent state hash mismatch")
    if parent.slot_positions != template.slot_positions:
        raise WireProtocolError(reason="slot positions mismatch")
    if len(parent.token_ids) != len(template.token_ids):
        raise WireProtocolError(reason="parent token length mismatch")
    _validate_parent_tokens(request, template)


def transfer_counts(mask_count: int, steps: int) -> tuple[int, ...]:
    """Return the upstream even transfer-token schedule."""
    base = mask_count // steps
    remainder = mask_count % steps
    return tuple(base + 1 if index < remainder else base for index in range(steps))


def _validate_parent_tokens(request: ReviseRequest, template: Template) -> None:
    parent = request.parent
    selected = set(selected_indexes(request.remask_bitmap))
    for index, position in enumerate(template.slot_positions):
        if index not in selected and parent.token_ids[position] == template.token_ids[position]:
            raise WireProtocolError(reason="unselected slot is already masked")
    for index, (actual, expected) in enumerate(
        zip(parent.token_ids, template.token_ids, strict=True)
    ):
        if index not in template.slot_positions and actual != expected:
            raise WireProtocolError(reason="frozen grammar token mismatch")
