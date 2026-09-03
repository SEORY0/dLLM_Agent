"""Fixed OpenEXR selector template helpers for selective revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dllm_agent.targets.openexr.particle import (
    FIELD_ORDER,
    MATERIALIZED_MUTABLE_FIELDS,
    FieldName,
    SelectorSlots,
    field_selector_cardinality,
)
from tools.runtime.selective_revision_json import TrajectoryStep, WireProtocolError


@runtime_checkable
class SlotTokenizer(Protocol):
    """Tokenizer surface required by the selector template."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode template text."""
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        """Decode one token for numeric filtering."""
        ...

    def __len__(self) -> int:
        """Return vocabulary size."""
        ...


@dataclass(frozen=True, slots=True)
class FieldSpan:
    """Template span for one typed numeric selector slot."""

    field: FieldName
    position: int


@dataclass(frozen=True, slots=True)
class Template:
    """Frozen selector suffix with exactly one token slot per field."""

    token_ids: tuple[int, ...]
    slot_positions: tuple[int, ...]
    field_spans: tuple[FieldSpan, ...]
    context_token_count: int


@dataclass(frozen=True, slots=True)
class BlockDenoiseResult:
    """Proposed tokens and evidence from one block-level denoising forward."""

    token_ids: tuple[int, ...]
    confidences: tuple[float, ...]
    proposal_margins: tuple[float, ...]
    gpu_ms: int


@dataclass(frozen=True, slots=True)
class FillResult:
    """Sampler token state and evidence after selected slots are committed."""

    token_ids: tuple[int, ...]
    selected_slot_token_ids: tuple[int, ...]
    selected_slot_indexes: tuple[int, ...]
    selectors: SelectorSlots
    confidences: tuple[float, ...]
    forward_blocks: tuple[int, ...]
    nfe: int
    gpu_ms: int
    trajectory_steps: tuple[TrajectoryStep, ...]


def build_template(
    tokenizer: SlotTokenizer,
    mask_token_id: int,
    *,
    block_size: int = 64,
    prompt_context: str = "",
) -> Template:
    """Build the compact fixed OpenEXR selector suffix."""
    token_ids: list[int] = [] if prompt_context == "" else _encode(tokenizer, prompt_context)
    padding = _encode(tokenizer, "\n")
    while len(token_ids) % block_size != 0:
        token_ids.extend(padding)
    context_token_count = len(token_ids)
    token_ids.extend(_encode(tokenizer, "EXR"))
    token_ids.extend(_encode(tokenizer, "\n"))
    token_ids.extend(_encode(tokenizer, "fields"))
    token_ids.extend(_encode(tokenizer, "="))
    for index, field in enumerate(FIELD_ORDER):
        if index > 0:
            token_ids.extend(_encode(tokenizer, ","))
        token_ids.extend(_encode(tokenizer, field))
    token_ids.extend(_encode(tokenizer, "\n"))
    token_ids.extend(_encode(tokenizer, "selectors"))
    token_ids.extend(_encode(tokenizer, "="))
    token_ids.extend(_encode(tokenizer, "["))
    slot_positions: list[int] = []
    spans: list[FieldSpan] = []
    for index, field in enumerate(FIELD_ORDER):
        if index > 0:
            token_ids.extend(_encode(tokenizer, ","))
        slot_positions.append(len(token_ids))
        spans.append(FieldSpan(field=field, position=len(token_ids)))
        token_ids.append(mask_token_id)
    token_ids.extend(_encode(tokenizer, "]"))
    return Template(
        token_ids=tuple(token_ids),
        slot_positions=tuple(slot_positions),
        field_spans=tuple(spans),
        context_token_count=context_token_count,
    )


def numeric_token_ids(tokenizer: SlotTokenizer) -> tuple[int, ...]:
    """Return tokenizer IDs whose stripped decode is an unsigned integer."""
    values: list[int] = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        if _selector_text(text):
            values.append(token_id)
    if not values:
        raise WireProtocolError(reason="tokenizer has no one-token integer IDs")
    return tuple(values)


def selector_slots(values: tuple[int, ...]) -> SelectorSlots:
    """Coerce exactly ten selector values into the selector-slot type."""
    if len(values) != len(FIELD_ORDER):
        reason = f"expected {len(FIELD_ORDER)} selectors, got {len(values)}"
        raise WireProtocolError(reason=reason)
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


def selector_for_token(tokenizer: SlotTokenizer, token_id: int) -> int:
    """Decode a numeric slot token into its selector integer."""
    text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
    if not _selector_text(text):
        reason = f"slot token is not numeric: {token_id}"
        raise WireProtocolError(reason=reason)
    return int(text)


def field_selector_token_ids(
    tokenizer: SlotTokenizer,
    numeric_tokens: tuple[int, ...],
    field: FieldName,
) -> tuple[int, ...]:
    """Return canonical selector-token IDs for one field catalog."""
    by_class: dict[int, int] = {}
    cardinality = field_selector_cardinality(field)
    for token_id in numeric_tokens:
        selector = selector_for_token(tokenizer, token_id)
        if 0 <= selector < cardinality and selector not in by_class:
            by_class[selector] = token_id
    missing = tuple(index for index in range(cardinality) if index not in by_class)
    if missing:
        reason = f"tokenizer missing selector classes for {field}: {missing}"
        raise WireProtocolError(reason=reason)
    return tuple(by_class[index] for index in range(cardinality))


def positions_by_block(
    positions: tuple[int, ...],
    previous_token_ids: tuple[int | None, ...],
    block_size: int,
) -> dict[int, tuple[tuple[int, ...], tuple[int | None, ...]]]:
    """Group selected slot positions by generation block."""
    grouped: dict[int, tuple[list[int], list[int | None]]] = {}
    for position, previous in zip(positions, previous_token_ids, strict=True):
        block = position // block_size
        block_positions, block_previous = grouped.setdefault(block, ([], []))
        block_positions.append(position)
        block_previous.append(previous)
    return {
        block: (tuple(block_positions), tuple(block_previous))
        for block, (block_positions, block_previous) in grouped.items()
    }


def selected_indexes(remask_bitmap: int) -> tuple[int, ...]:
    """Return selected field indexes from a nonzero remask bitmap."""
    indexes = tuple(index for index in range(len(FIELD_ORDER)) if remask_bitmap & (1 << index))
    if not indexes:
        raise WireProtocolError(reason="remask bitmap selects no slots")
    return indexes


def uncertainty(confidences: tuple[float, ...]) -> float:
    """Map transfer confidences to trajectory uncertainty."""
    if not confidences:
        return 0.0
    return max(0.0, min(1.0, 1.0 - min(confidences)))


def suggested_field(slot_indexes: tuple[int, ...], confidences: tuple[float, ...]) -> FieldName:
    """Return the lowest-confidence field suggestion."""
    candidates = tuple(
        index
        for index, slot_index in enumerate(slot_indexes)
        if FIELD_ORDER[slot_index] in MATERIALIZED_MUTABLE_FIELDS
    )
    if not candidates:
        return MATERIALIZED_MUTABLE_FIELDS[0]
    lowest = min(candidates, key=lambda index: confidences[index])
    return FIELD_ORDER[slot_indexes[lowest]]


def _encode(tokenizer: SlotTokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        reason = f"template token is empty: {text}"
        raise WireProtocolError(reason=reason)
    return ids


def _selector_text(text: str) -> bool:
    return text.isascii() and text.isdecimal()
