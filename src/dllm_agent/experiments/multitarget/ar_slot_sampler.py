"""Autoregressive AR-parent control sampler for the dLLM slot particle format.

Fills the identical typed slot template used by `GenericSlotSampler` (same
tokenizer, same schema, same catalog-restricted vocabulary per slot) through
genuine strict left-to-right greedy decoding instead of confidence-ranked
parallel commitment: every position conditions only on positions already
committed earlier in sequence order, never on a not-yet-decided position.
This isolates the AR-vs-dLLM decoding-paradigm variable while holding the
particle format, tokenizer, and slot catalogs fixed -- see
docs/superpowers/specs/2026-09-14-ar-parent-control-arm.md for the model
choice (an architecture/data-matched AR parent) and rationale.

`forward_calls`/`nfe` here count one real forward pass per slot (true
per-token AR cost). This is *not* directly FLOP-comparable to the dLLM arm's
`forward_calls`, which counts block-parallel calls that may resolve several
slots per call -- report both raw counts, never a bare "faster/slower" claim.
"""

import math
from typing import Protocol, final

from dllm_agent.experiments.multitarget.slot_sampler_contracts import (
    SlotSample,
    SlotSamplerError,
    SlotSchema,
    SlotTokenizer,
    TrajectoryStep,
)
from dllm_agent.experiments.multitarget.slot_sampler_utils import (
    build_template,
    selectors,
    sha256_tokens,
    slot_token_ids,
    validate_schema,
)

__all__ = ("ArTokenModel", "AutoregressiveSlotSampler")


class ArTokenModel(Protocol):
    """Next-token logit source required by strict left-to-right AR decoding."""

    def next_token_logits(self, token_ids: tuple[int, ...]) -> tuple[float, ...]:
        """Return full-vocabulary logits for the position after `token_ids`."""
        ...


@final
class AutoregressiveSlotSampler:
    """Fill every typed slot once, strictly left-to-right, via greedy AR decoding."""

    def __init__(
        self,
        tokenizer: SlotTokenizer,
        model: ArTokenModel,
        *,
        schema: SlotSchema,
        mask_token_id: int,
        prompt_context: str = "",
    ) -> None:
        """Construct an AR-parent sampler for one structured target schema."""
        validate_schema(schema)
        self._tokenizer = tokenizer
        self._model = model
        self._template = build_template(tokenizer, schema, mask_token_id, prompt_context)
        self._slot_token_ids = slot_token_ids(tokenizer, schema)

    def initialize(self, *, seed: int) -> SlotSample:
        """Decode every target selector slot once, in ascending position order."""
        del seed  # greedy AR decoding is deterministic; kept for interface parity.
        token_ids = list(self._template[0])
        positions = self._template[1]
        confidences: list[float] = [0.0] * len(positions)
        trajectory: list[TrajectoryStep] = []
        for slot_index, position in enumerate(positions):
            allowed = self._slot_token_ids[slot_index]
            prefix = tuple(token_ids[:position])
            logits = self._model.next_token_logits(prefix)
            token_id, confidence = _select(logits, allowed)
            token_ids[position] = token_id
            confidences[slot_index] = confidence
            trajectory.append(
                TrajectoryStep(
                    step_index=slot_index + 1,
                    block_index=slot_index,
                    masked_before=len(positions) - slot_index,
                    committed_positions=(position,),
                    token_ids=(token_id,),
                    confidences=(confidence,),
                    remaining_masked=len(positions) - slot_index - 1,
                    state_sha256=sha256_tokens(tuple(token_ids)),
                )
            )
        final = tuple(token_ids)
        forward_calls = len(positions)
        return SlotSample(
            token_ids=final,
            selectors=selectors(self._tokenizer, final, positions),
            slot_positions=positions,
            slot_confidences=tuple(confidences),
            selected_slot_token_ids=tuple(final[position] for position in positions),
            requested_steps=forward_calls,
            nfe=forward_calls,
            forward_calls=forward_calls,
            gpu_ms=0,
            trajectory_steps=tuple(trajectory),
            state_sha256=sha256_tokens(final),
            parent_state_sha256=None,
        )


def _select(logits: tuple[float, ...], allowed: tuple[int, ...]) -> tuple[int, float]:
    if not allowed:
        raise SlotSamplerError(reason="slot has no allowed selector tokens")
    restricted = tuple((token_id, logits[token_id]) for token_id in allowed)
    peak = max(value for _, value in restricted)
    weights = tuple((token_id, math.exp(value - peak)) for token_id, value in restricted)
    total = sum(weight for _, weight in weights)
    probabilities = tuple((token_id, weight / total) for token_id, weight in weights)
    return max(probabilities, key=lambda item: (item[1], -item[0]))
