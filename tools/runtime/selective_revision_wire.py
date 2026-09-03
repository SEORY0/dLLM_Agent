"""Shared strict wire parsing constants for selective revision."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from tools.runtime.selective_revision_json import (
    JsonValue,
    JsonWireError,
    ResponseStatus,
    WireProtocolError,
    WorkerOperation,
    object_from_line,
)

if TYPE_CHECKING:
    from dllm_agent.targets.openexr.particle import FieldName, SelectorSlots

_SLOT_COUNT: Final = 10
READY_KEYS: Final = frozenset(
    {
        "kind",
        "schema_version",
        "model_id",
        "revision",
        "remote_code_sha256",
        "device",
        "load_count",
        "prompt_sha256",
    }
)
REQUEST_KEYS: Final = frozenset({"operation", "request_id", "schema_version", "payload"})
INIT_KEYS: Final = frozenset({"request_id", "particle_id", "seed", "steps"})
PARENT_KEYS: Final = frozenset({"token_ids", "slot_positions", "selectors", "token_state_sha256"})
REVISE_KEYS: Final = frozenset(
    {
        "request_id",
        "particle_id",
        "parent",
        "parent_token_sha256",
        "remask_bitmap",
        "frozen_bitmap",
        "revision_steps",
        "seed",
        "rng_state_sha256",
        "feedback_digest",
        "mode",
    }
)
SHUTDOWN_KEYS: Final = frozenset({"kind", "request_id", "request_sha256", "status"})
RESPONSE_KEYS: Final = frozenset(
    {
        "kind",
        "schema_version",
        "request_id",
        "request_sha256",
        "response_hash",
        "status",
        "selectors",
        "token_ids",
        "slot_positions",
        "selected_slot_token_ids",
        "before_remask_token_ids",
        "remasked_positions",
        "token_state_sha256",
        "transfer_evidence_sha256",
        "rng_state_sha256",
        "nfe",
        "forward_calls",
        "gpu_ms",
        "cache_hit",
        "trajectory_uncertainty",
        "suggested_field",
        "invalidated_from_block",
        "forward_blocks",
        "trajectory_steps",
    }
)
TRAJECTORY_STEP_KEYS: Final = frozenset(
    {
        "step_index",
        "block_index",
        "masked_before",
        "committed_positions",
        "token_ids",
        "confidences",
        "proposal_positions",
        "proposal_token_ids",
        "proposal_confidences",
        "proposal_margins",
        "remaining_masked",
    }
)


def operation(value: str) -> WorkerOperation:
    """Return a known worker operation."""
    match value:  # noqa: MATCH_OK - untrusted wire strings have an open variant set.
        case "initialize" | "revise" | "shutdown":
            return value
        case _:
            raise WireProtocolError(reason=f"unknown operation: {value}")


def field_name(value: str) -> FieldName:
    """Return a known OpenEXR selector field name."""
    match value:  # noqa: MATCH_OK - untrusted wire strings have an open variant set.
        case (
            "origin_x"
            | "origin_y"
            | "width"
            | "height"
            | "compression"
            | "pixel_type"
            | "line_order"
            | "x_sampling"
            | "payload_pattern"
            | "chunk_offset"
        ):
            return value
        case _:
            raise WireProtocolError(reason=f"unknown field: {value}")


def response_status(value: str) -> ResponseStatus:
    """Return a known sampler response status."""
    match value:  # noqa: MATCH_OK - untrusted wire strings have an open variant set.
        case "revised" | "unchanged_revision" | "initialized":
            return value
        case _:
            raise WireProtocolError(reason=f"unknown response status: {value}")


def selector_slots(values: tuple[int, ...]) -> SelectorSlots:
    """Coerce ten selector integers into the selector slot tuple."""
    if len(values) != _SLOT_COUNT:
        raise WireProtocolError(reason=f"expected 10 selectors, got {len(values)}")
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
    )


def wire_object_from_line(line: str) -> dict[str, JsonValue]:
    """Parse one JSONL object as a wire protocol error."""
    try:
        return object_from_line(line)
    except JsonWireError as error:
        raise WireProtocolError(reason=str(error)) from error
