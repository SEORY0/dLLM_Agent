"""Strict JSON payloads for the resident selective-revision worker."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import asdict, replace
from typing import Final, assert_never

from tools.runtime.selective_revision_json import (
    InitializeRequest,
    JsonValue,
    ReadyRecord,
    ReviseRequest,
    SamplerResponse,
    SelectorState,
    ShutdownResponse,
    WireProtocolError,
    WorkerRequest,
    boolean,
    canonical_json,
    integer,
    integers,
    json_value,
    number,
    require,
    require_keys,
    string,
)
from tools.runtime.selective_revision_request_validation import (
    require_bitmaps,
    require_nonnegative,
    require_positive,
    require_sha256,
)
from tools.runtime.selective_revision_trajectory import trajectory_steps
from tools.runtime.selective_revision_wire import (
    INIT_KEYS,
    PARENT_KEYS,
    READY_KEYS,
    REQUEST_KEYS,
    RESPONSE_KEYS,
    REVISE_KEYS,
    SHUTDOWN_KEYS,
    field_name,
    response_status,
    selector_slots,
    wire_object_from_line,
)
from tools.runtime.selective_revision_wire import operation as parse_operation

READY_SCHEMA_VERSION: Final = "dllm-selective-revision-ready/v1"
REQUEST_SCHEMA_VERSION: Final = "dllm-selective-revision-request/v1"
RESPONSE_SCHEMA_VERSION: Final = "dllm-selective-revision-response/v2"
MODEL_ID: Final = "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
MODEL_REVISION: Final = "10cdaf9b486f1cf0273ad968459dbe2d21a1482a"
REMOTE_CODE_SHA256: Final = "298b19a464f32c00d38e43603516174a73fb4c531193369ff1aabbef32593104"


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_token_ids(token_ids: tuple[int, ...]) -> str:
    """Hash token IDs as little-endian signed 64-bit integers."""
    return sha256_bytes(b"".join(struct.pack("<q", token_id) for token_id in token_ids))


def request_hash(request: InitializeRequest | ReviseRequest | WorkerRequest) -> str:
    """Hash a request exactly as canonical JSON."""
    return sha256_bytes(canonical_json(json_value(asdict(request))).encode())


def response_with_hash(response: SamplerResponse) -> SamplerResponse:
    """Return response with its response hash bound to all other response fields."""
    body = asdict(response)
    body["response_hash"] = ""
    return replace(response, response_hash=sha256_bytes(canonical_json(json_value(body)).encode()))


def request_to_line(request: WorkerRequest) -> str:
    """Serialize one worker request as a JSONL line."""
    return canonical_json(json_value(asdict(request))) + "\n"


def ready_record(device: str, load_count: int, prompt_sha256: str) -> ReadyRecord:
    """Create the pinned ready record emitted after model load."""
    return ReadyRecord(
        kind="ready",
        schema_version=READY_SCHEMA_VERSION,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        remote_code_sha256=REMOTE_CODE_SHA256,
        device=device,
        load_count=load_count,
        prompt_sha256=prompt_sha256,
    )


def parse_ready_line(line: str) -> ReadyRecord:
    """Parse and validate the worker ready record."""
    item = wire_object_from_line(line)
    require_keys(item, READY_KEYS)
    require(item, "kind", "ready")
    require(item, "schema_version", READY_SCHEMA_VERSION)
    require(item, "model_id", MODEL_ID)
    require(item, "revision", MODEL_REVISION)
    require(item, "remote_code_sha256", REMOTE_CODE_SHA256)
    if integer(item, "load_count") != 1:
        raise WireProtocolError(reason="ready load_count must be 1")
    return ReadyRecord(
        kind="ready",
        schema_version=READY_SCHEMA_VERSION,
        model_id=string(item, "model_id"),
        revision=string(item, "revision"),
        remote_code_sha256=string(item, "remote_code_sha256"),
        device=string(item, "device"),
        load_count=integer(item, "load_count"),
        prompt_sha256=string(item, "prompt_sha256"),
    )


def parse_response_line(line: str) -> SamplerResponse | ShutdownResponse:
    """Parse a worker response or shutdown acknowledgement."""
    item = wire_object_from_line(line)
    kind = string(item, "kind")
    match kind:  # noqa: MATCH_OK - untrusted wire string has an open set of values.
        case "response":
            return _response(item)
        case "shutdown":
            require_keys(item, SHUTDOWN_KEYS)
            require(item, "status", "ok")
            return ShutdownResponse(
                kind="shutdown",
                request_id=string(item, "request_id"),
                request_sha256=string(item, "request_sha256"),
                status="ok",
            )
        case _:
            raise WireProtocolError(reason=f"unknown response kind: {kind}")


def _response(item: dict[str, JsonValue]) -> SamplerResponse:
    require_keys(item, RESPONSE_KEYS)
    require(item, "schema_version", RESPONSE_SCHEMA_VERSION)
    status = response_status(string(item, "status"))
    return SamplerResponse(
        kind="response",
        schema_version=RESPONSE_SCHEMA_VERSION,
        request_id=string(item, "request_id"),
        request_sha256=string(item, "request_sha256"),
        response_hash=string(item, "response_hash"),
        status=status,
        selectors=selector_slots(integers(item, "selectors")),
        token_ids=integers(item, "token_ids"),
        slot_positions=integers(item, "slot_positions"),
        selected_slot_token_ids=integers(item, "selected_slot_token_ids"),
        before_remask_token_ids=integers(item, "before_remask_token_ids"),
        remasked_positions=integers(item, "remasked_positions"),
        token_state_sha256=string(item, "token_state_sha256"),
        transfer_evidence_sha256=string(item, "transfer_evidence_sha256"),
        rng_state_sha256=string(item, "rng_state_sha256"),
        nfe=integer(item, "nfe"),
        forward_calls=integer(item, "forward_calls"),
        gpu_ms=integer(item, "gpu_ms"),
        cache_hit=boolean(item, "cache_hit"),
        trajectory_uncertainty=float(number(item, "trajectory_uncertainty")),
        suggested_field=field_name(string(item, "suggested_field")),
        invalidated_from_block=integer(item, "invalidated_from_block"),
        forward_blocks=integers(item, "forward_blocks"),
        trajectory_steps=trajectory_steps(item),
    )


def parse_request_line(line: str) -> WorkerRequest:
    """Parse and validate one worker request line."""
    item = wire_object_from_line(line)
    require_keys(item, REQUEST_KEYS)
    operation = parse_operation(string(item, "operation"))
    require(item, "schema_version", REQUEST_SCHEMA_VERSION)
    request_id = string(item, "request_id")
    payload = item.get("payload")
    if operation == "shutdown":
        return WorkerRequest(
            operation=operation,
            request_id=request_id,
            schema_version=REQUEST_SCHEMA_VERSION,
            payload=None,
        )
    if not isinstance(payload, dict):
        raise WireProtocolError(reason="payload must be an object")
    if string(payload, "request_id") != request_id:
        raise WireProtocolError(reason="payload request_id must match request_id")
    match operation:  # noqa: MATCH_OK - shutdown returned above; remaining literals exhaustive.
        case "initialize":
            require_keys(payload, INIT_KEYS)
            require_positive(payload, "steps")
            require_nonnegative(payload, "seed")
            parsed: InitializeRequest | ReviseRequest = InitializeRequest(
                request_id=string(payload, "request_id"),
                particle_id=string(payload, "particle_id"),
                seed=integer(payload, "seed"),
                steps=integer(payload, "steps"),
            )
        case "revise":
            parsed = _revise(payload)
        case "shutdown":
            assert_never(operation)
    return WorkerRequest(
        operation=operation,
        request_id=request_id,
        schema_version=REQUEST_SCHEMA_VERSION,
        payload=parsed,
    )


def _revise(payload: dict[str, JsonValue]) -> ReviseRequest:
    require_keys(payload, REVISE_KEYS)
    require(payload, "mode", "selective-revision/v1")
    require_nonnegative(payload, "seed")
    require_positive(payload, "revision_steps")
    require_sha256(payload, "parent_token_sha256")
    require_sha256(payload, "rng_state_sha256")
    require_sha256(payload, "feedback_digest")
    require_bitmaps(payload)
    parent = payload.get("parent")
    if not isinstance(parent, dict):
        raise WireProtocolError(reason="parent must be an object")
    require_keys(parent, PARENT_KEYS)
    require_sha256(parent, "token_state_sha256")
    return ReviseRequest(
        request_id=string(payload, "request_id"),
        particle_id=string(payload, "particle_id"),
        parent=SelectorState(
            token_ids=integers(parent, "token_ids"),
            slot_positions=integers(parent, "slot_positions"),
            selectors=selector_slots(integers(parent, "selectors")),
            token_state_sha256=string(parent, "token_state_sha256"),
        ),
        parent_token_sha256=string(payload, "parent_token_sha256"),
        remask_bitmap=integer(payload, "remask_bitmap"),
        frozen_bitmap=integer(payload, "frozen_bitmap"),
        revision_steps=integer(payload, "revision_steps"),
        seed=integer(payload, "seed"),
        rng_state_sha256=string(payload, "rng_state_sha256"),
        feedback_digest=string(payload, "feedback_digest"),
        mode="selective-revision/v1",
    )


def shutdown_request(request_id: str) -> WorkerRequest:
    """Build a typed shutdown request."""
    return WorkerRequest(
        operation="shutdown",
        request_id=request_id,
        schema_version=REQUEST_SCHEMA_VERSION,
        payload=None,
    )


def initialize_worker_request(payload: InitializeRequest) -> WorkerRequest:
    """Build a typed initialize worker request."""
    return WorkerRequest(
        operation="initialize",
        request_id=payload.request_id,
        schema_version=REQUEST_SCHEMA_VERSION,
        payload=payload,
    )


def revise_worker_request(payload: ReviseRequest) -> WorkerRequest:
    """Build a typed revise worker request."""
    return WorkerRequest(
        operation="revise",
        request_id=payload.request_id,
        schema_version=REQUEST_SCHEMA_VERSION,
        payload=payload,
    )
