"""JSON helpers for the selective-revision worker wire protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, override

from tools.runtime.selective_revision_decode import coerce_json_value, decode_json_value

if TYPE_CHECKING:
    from pydantic import TypeAdapter

    from dllm_agent.targets.openexr.particle import FieldName, SelectorSlots

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type WorkerOperation = Literal["initialize", "revise", "shutdown"]
type ResponseStatus = Literal["revised", "unchanged_revision", "initialized"]


class _JsonValueAdapter(Protocol):
    def validate_json(self, data: str) -> JsonValue: ...

    def validate_python(self, value: JsonValue) -> JsonValue: ...


@dataclass(frozen=True, slots=True)
class _PydanticJsonValueAdapter:
    adapter: TypeAdapter[JsonValue]

    def validate_json(self, data: str) -> JsonValue:
        return self.adapter.validate_json(data)

    def validate_python(self, value: JsonValue) -> JsonValue:
        return self.adapter.validate_python(value)


def _build_json_adapter() -> _JsonValueAdapter | None:
    try:
        from pydantic import TypeAdapter  # noqa: PLC0415
    except ModuleNotFoundError:
        return None
    return _PydanticJsonValueAdapter(adapter=TypeAdapter(JsonValue))


_JSON_VALUE_ADAPTER: _JsonValueAdapter | None = _build_json_adapter()


class JsonWireError(ValueError):
    """Raised for malformed worker JSON values."""


@dataclass(frozen=True, slots=True)
class WireProtocolError(ValueError):
    """Raised when a JSONL payload violates the worker wire protocol."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the protocol failure reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class SelectorState:
    """Parent or child token/selector state crossing the worker boundary."""

    token_ids: tuple[int, ...]
    slot_positions: tuple[int, ...]
    selectors: SelectorSlots
    token_state_sha256: str


@dataclass(frozen=True, slots=True)
class InitializeRequest:
    """Initial particle request."""

    request_id: str
    particle_id: str
    seed: int
    steps: int


@dataclass(frozen=True, slots=True)
class ReviseRequest:
    """Committed-token selective revision request."""

    request_id: str
    particle_id: str
    parent: SelectorState
    parent_token_sha256: str
    remask_bitmap: int
    frozen_bitmap: int
    revision_steps: int
    seed: int
    rng_state_sha256: str
    feedback_digest: str
    mode: Literal["selective-revision/v1"]


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """One JSONL command sent to the worker."""

    operation: WorkerOperation
    request_id: str
    schema_version: str
    payload: InitializeRequest | ReviseRequest | None


@dataclass(frozen=True, slots=True)
class ReadyRecord:
    """Worker ready/provenance record."""

    kind: Literal["ready"]
    schema_version: str
    model_id: str
    revision: str
    remote_code_sha256: str
    device: str
    load_count: int
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One iterative block diffusion transfer step."""

    step_index: int
    block_index: int
    masked_before: int
    committed_positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    confidences: tuple[float, ...]
    proposal_positions: tuple[int, ...]
    proposal_token_ids: tuple[int, ...]
    proposal_confidences: tuple[float, ...]
    proposal_margins: tuple[float, ...]
    remaining_masked: int


@dataclass(frozen=True, slots=True)
class SamplerResponse:
    """Worker response payload mapped into Task 4 revision results."""

    kind: Literal["response"]
    schema_version: str
    request_id: str
    request_sha256: str
    response_hash: str
    status: ResponseStatus
    selectors: SelectorSlots
    token_ids: tuple[int, ...]
    slot_positions: tuple[int, ...]
    selected_slot_token_ids: tuple[int, ...]
    before_remask_token_ids: tuple[int, ...]
    remasked_positions: tuple[int, ...]
    token_state_sha256: str
    transfer_evidence_sha256: str
    rng_state_sha256: str
    nfe: int
    forward_calls: int
    gpu_ms: int
    cache_hit: bool
    trajectory_uncertainty: float
    suggested_field: FieldName
    invalidated_from_block: int
    forward_blocks: tuple[int, ...]
    trajectory_steps: tuple[TrajectoryStep, ...]


@dataclass(frozen=True, slots=True)
class ShutdownResponse:
    """Worker shutdown acknowledgement."""

    kind: Literal["shutdown"]
    request_id: str
    request_sha256: str
    status: Literal["ok"]


def canonical_json(document: JsonValue) -> str:
    """Serialize a document in canonical worker JSON form."""
    return json.dumps(document, separators=(",", ":"), sort_keys=True)


def object_from_line(line: str) -> dict[str, JsonValue]:
    """Parse a JSON object from one line."""
    try:
        parsed = (
            decode_json_value(line)
            if _JSON_VALUE_ADAPTER is None
            else _JSON_VALUE_ADAPTER.validate_json(line)
        )
    except ValueError as error:
        message = f"malformed JSON: {error}"
        raise JsonWireError(message) from error
    if not isinstance(parsed, dict):
        message = "payload must be a JSON object"
        raise JsonWireError(message)
    return parsed


def json_value(value: JsonValue) -> JsonValue:
    """Coerce a decoded JSON value into the recursive JSON alias."""
    try:
        if _JSON_VALUE_ADAPTER is None:
            return coerce_json_value(value)
        return _JSON_VALUE_ADAPTER.validate_python(value)
    except ValueError as error:
        message = f"unsupported JSON value: {type(value).__name__}"
        raise JsonWireError(message) from error


def require(item: dict[str, JsonValue], key: str, expected: str) -> None:
    """Require a string field to equal an expected value."""
    actual = string(item, key)
    if actual != expected:
        message = f"{key} must be {expected}, got {actual}"
        raise JsonWireError(message)


def require_keys(item: dict[str, JsonValue], expected: frozenset[str]) -> None:
    """Require an object to contain exactly the expected field names."""
    actual = frozenset(item)
    if actual != expected:
        missing = ",".join(sorted(expected - actual))
        extra = ",".join(sorted(actual - expected))
        message = f"unexpected JSON fields missing={missing} extra={extra}"
        raise JsonWireError(message)


def string(item: dict[str, JsonValue], key: str) -> str:
    """Return a required string field."""
    value = item.get(key)
    if not isinstance(value, str):
        message = f"{key} must be a string"
        raise JsonWireError(message)
    return value


def integer(item: dict[str, JsonValue], key: str) -> int:
    """Return a required integer field."""
    value = item.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"{key} must be an integer"
        raise JsonWireError(message)
    return value


def boolean(item: dict[str, JsonValue], key: str) -> bool:
    """Return a required boolean field."""
    value = item.get(key)
    if not isinstance(value, bool):
        message = f"{key} must be a boolean"
        raise JsonWireError(message)
    return value


def number(item: dict[str, JsonValue], key: str) -> int | float:
    """Return a required numeric field."""
    value = item.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    message = f"{key} must be numeric"
    raise JsonWireError(message)


def integers(item: dict[str, JsonValue], key: str) -> tuple[int, ...]:
    """Return a required integer-list field."""
    value = item.get(key)
    if not isinstance(value, list):
        message = f"{key} must be a list"
        raise JsonWireError(message)
    values: list[int] = []
    for entry in value:
        if not isinstance(entry, int) or isinstance(entry, bool):
            message = f"{key} must contain only integers"
            raise JsonWireError(message)
        values.append(entry)
    return tuple(values)
