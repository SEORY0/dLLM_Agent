"""Pure template, selector, validation, and hashing helpers."""

import hashlib
import json

from dllm_agent.experiments.multitarget.slot_sampler_contracts import (
    BlockDenoiseResult,
    SlotSamplerError,
    SlotSchema,
    SlotTokenizer,
)


def validate_schema(schema: SlotSchema) -> None:
    """Reject inconsistent or non-denoisable slot schemas."""
    if len(schema.field_names) != len(schema.cardinalities):
        raise SlotSamplerError(reason="schema field/cardinality mismatch")
    if not schema.field_names:
        raise SlotSamplerError(reason="schema has no slots")
    if any(cardinality <= 0 for cardinality in schema.cardinalities):
        raise SlotSamplerError(reason="slot cardinalities must be positive")
    if any(index < 0 or index >= len(schema.field_names) for index in schema.mutable_indexes):
        raise SlotSamplerError(reason="mutable slot index is out of bounds")


def build_template(
    tokenizer: SlotTokenizer,
    schema: SlotSchema,
    mask_token_id: int,
    prompt_context: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build a target-labelled template with one masked token per field."""
    token_ids = [] if prompt_context == "" else encode(tokenizer, prompt_context)
    token_ids.extend(encode(tokenizer, "\n"))
    token_ids.extend(encode(tokenizer, schema.label))
    token_ids.extend(encode(tokenizer, "\n"))
    slot_positions: list[int] = []
    for field in schema.field_names:
        token_ids.extend(encode(tokenizer, field))
        token_ids.extend(encode(tokenizer, "="))
        slot_positions.append(len(token_ids))
        token_ids.append(mask_token_id)
        token_ids.extend(encode(tokenizer, "\n"))
    return (tuple(token_ids), tuple(slot_positions))


def slot_token_ids(tokenizer: SlotTokenizer, schema: SlotSchema) -> tuple[tuple[int, ...], ...]:
    """Return canonical one-token numeric selector IDs for every field."""
    by_selector: dict[int, int] = {}
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        if text.isascii() and text.isdecimal():
            _ = by_selector.setdefault(int(text), token_id)
    return tuple(
        tokens_for_cardinality(by_selector, cardinality, field)
        for field, cardinality in zip(
            schema.field_names,
            schema.cardinalities,
            strict=True,
        )
    )


def tokens_for_cardinality(
    by_selector: dict[int, int], cardinality: int, field: str
) -> tuple[int, ...]:
    """Return selector tokens in class order for one field."""
    tokens: list[int] = []
    for selector in range(cardinality):
        token_id = by_selector.get(selector)
        if token_id is None:
            raise SlotSamplerError(reason=f"tokenizer missing selector {selector} for {field}")
        tokens.append(token_id)
    return tuple(tokens)


def validate_result(
    result: BlockDenoiseResult,
    positions: tuple[int, ...],
    allowed: tuple[int, ...],
) -> None:
    """Reject model outputs that do not match the constrained request."""
    lengths_match = len(result.token_ids) == len(positions) == len(result.confidences)
    if not lengths_match:
        raise SlotSamplerError(reason="model returned mismatched block result lengths")
    if any(token_id not in allowed for token_id in result.token_ids):
        raise SlotSamplerError(reason="model selected a disallowed token")


def encode(tokenizer: SlotTokenizer, text: str) -> list[int]:
    """Encode required template text and reject an empty tokenizer result."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise SlotSamplerError(reason=f"template token is empty: {text}")
    return token_ids


def selector_for_token(tokenizer: SlotTokenizer, token_id: int) -> int:
    """Decode one numeric selector token."""
    text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
    if not (text.isascii() and text.isdecimal()):
        raise SlotSamplerError(reason=f"slot token is not numeric: {token_id}")
    return int(text)


def selectors(
    tokenizer: SlotTokenizer,
    token_ids: tuple[int, ...],
    slot_positions: tuple[int, ...],
) -> tuple[int, ...]:
    """Decode every committed selector slot in canonical field order."""
    return tuple(selector_for_token(tokenizer, token_ids[position]) for position in slot_positions)


def sha256_tokens(token_ids: tuple[int, ...]) -> str:
    """Hash a canonical compact JSON representation of a token state."""
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
