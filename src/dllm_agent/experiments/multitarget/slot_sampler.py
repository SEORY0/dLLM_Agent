"""Target-agnostic constrained dLLM slot sampler."""

from __future__ import annotations

from typing import final

from dllm_agent.experiments.multitarget.slot_sampler_contracts import (
    BlockDenoiseResult,
    SlotModel,
    SlotSample,
    SlotSamplerError,
    SlotSchema,
    SlotTokenizer,
    TrajectoryStep,
)
from dllm_agent.experiments.multitarget.slot_sampler_utils import (
    build_template,
    selector_for_token,
    selectors,
    sha256_tokens,
    slot_token_ids,
    validate_result,
    validate_schema,
)

__all__ = (
    "BlockDenoiseResult",
    "GenericSlotSampler",
    "SlotModel",
    "SlotSample",
    "SlotSamplerError",
    "SlotSchema",
    "SlotTokenizer",
    "TrajectoryStep",
)

type DenoiseGroup = tuple[
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int | None, ...],
]


@final
class GenericSlotSampler:
    """Initialize and selectively reopen typed numeric target slots."""

    def __init__(  # noqa: PLR0913 - mirrors the model-backed sampler boundary.
        self,
        tokenizer: SlotTokenizer,
        model: SlotModel,
        *,
        schema: SlotSchema,
        mask_token_id: int,
        block_size: int,
        prompt_context: str = "",
    ) -> None:
        """Construct a constrained sampler for one structured target schema."""
        validate_schema(schema)
        self._tokenizer = tokenizer
        self._model = model
        self._schema = schema
        self._mask_token_id = mask_token_id
        self._block_size = block_size
        self._template = build_template(tokenizer, schema, mask_token_id, prompt_context)
        self._slot_token_ids = slot_token_ids(tokenizer, schema)

    def initialize(self, *, seed: int, steps: int) -> SlotSample:
        """Denoise all target selector slots from an all-mask state."""
        slot_positions = self._template[1]
        return self._fill(
            token_ids=self._template[0],
            slot_indexes=tuple(range(len(slot_positions))),
            previous_token_ids=tuple(None for _ in slot_positions),
            seed=seed,
            steps=steps,
            parent_state_sha256=None,
        )

    def revise(
        self,
        parent: SlotSample,
        *,
        remask_indexes: tuple[int, ...],
        seed: int,
        steps: int,
    ) -> SlotSample:
        """Reopen only selected mutable slots and preserve all other token state."""
        indexes = self._parse_remask_indexes(remask_indexes)
        current = list(parent.token_ids)
        previous = tuple(parent.token_ids[parent.slot_positions[index]] for index in indexes)
        for index in indexes:
            current[parent.slot_positions[index]] = self._mask_token_id
        return self._fill(
            token_ids=tuple(current),
            slot_indexes=indexes,
            previous_token_ids=previous,
            seed=seed,
            steps=steps,
            parent_state_sha256=parent.state_sha256,
        )

    def _fill(  # noqa: PLR0913 - immutable denoising inputs are explicit.
        self,
        *,
        token_ids: tuple[int, ...],
        slot_indexes: tuple[int, ...],
        previous_token_ids: tuple[int | None, ...],
        seed: int,
        steps: int,
        parent_state_sha256: str | None,
    ) -> SlotSample:
        current = list(token_ids)
        positions = tuple(self._template[1][index] for index in slot_indexes)
        remaining = dict(
            zip(positions, zip(slot_indexes, previous_token_ids, strict=True), strict=True)
        )
        confidences_by_slot = [0.0 for _ in self._template[1]]
        selected_tokens: list[int] = []
        trajectory: list[TrajectoryStep] = []
        forward_calls = 0
        gpu_ms = 0
        for step_index in range(1, max(1, steps) + 1):
            if not remaining:
                break
            masked_before = len(remaining)
            predictions: list[tuple[int, int, int, float]] = []
            for block, allowed, block_positions, block_previous in self._groups(remaining):
                result = self._model.denoise_block(
                    block,
                    tuple(current),
                    block_positions,
                    allowed,
                    block_previous,
                    seed,
                    step_index,
                )
                validate_result(result, block_positions, allowed)
                forward_calls += 1
                gpu_ms += result.gpu_ms
                for position, token_id, confidence in zip(
                    block_positions,
                    result.token_ids,
                    result.confidences,
                    strict=True,
                ):
                    predictions.append((block, position, token_id, confidence))
            steps_left = max(1, max(1, steps) - step_index + 1)
            commit_count = (len(remaining) + steps_left - 1) // steps_left
            committed = sorted(
                predictions,
                key=lambda item: (-item[3], item[1]),
            )[:commit_count]
            by_block: dict[int, list[tuple[int, int, float]]] = {}
            for block, position, token_id, confidence in committed:
                by_block.setdefault(block, []).append((position, token_id, confidence))
            for block, items in sorted(by_block.items()):
                ordered = tuple(sorted(items))
                for position, token_id, confidence in ordered:
                    slot_index = remaining[position][0]
                    current[position] = token_id
                    selected_tokens.append(token_id)
                    confidences_by_slot[slot_index] = confidence
                    del remaining[position]
                trajectory.append(
                    TrajectoryStep(
                        step_index=step_index,
                        block_index=block,
                        masked_before=masked_before,
                        committed_positions=tuple(item[0] for item in ordered),
                        token_ids=tuple(item[1] for item in ordered),
                        confidences=tuple(item[2] for item in ordered),
                        remaining_masked=len(remaining),
                        state_sha256=sha256_tokens(tuple(current)),
                    )
                )
        if remaining:
            raise SlotSamplerError(reason="denoising steps ended before all masks were committed")
        return SlotSample(
            token_ids=tuple(current),
            selectors=selectors(self._tokenizer, tuple(current), self._template[1]),
            slot_positions=self._template[1],
            slot_confidences=tuple(confidences_by_slot),
            selected_slot_token_ids=tuple(selected_tokens),
            requested_steps=steps,
            nfe=forward_calls,
            forward_calls=forward_calls,
            gpu_ms=gpu_ms,
            trajectory_steps=tuple(trajectory),
            state_sha256=sha256_tokens(tuple(current)),
            parent_state_sha256=parent_state_sha256,
        )

    def _groups(self, remaining: dict[int, tuple[int, int | None]]) -> tuple[DenoiseGroup, ...]:
        grouped: dict[tuple[int, tuple[int, ...]], tuple[list[int], list[int | None]]] = {}
        for position, (slot_index, previous_token_id) in remaining.items():
            allowed = self._allowed_for_slot(slot_index, previous_token_id)
            key = (position // self._block_size, allowed)
            positions, previous = grouped.setdefault(key, ([], []))
            positions.append(position)
            previous.append(previous_token_id)
        return tuple(
            (block, allowed, tuple(block_positions), tuple(previous))
            for (block, allowed), (block_positions, previous) in sorted(grouped.items())
        )

    def _allowed_for_slot(self, slot_index: int, previous_token_id: int | None) -> tuple[int, ...]:
        allowed = self._slot_token_ids[slot_index]
        if previous_token_id is None:
            return allowed
        parent_selector = selector_for_token(self._tokenizer, previous_token_id)
        revised = tuple(
            token_id for selector, token_id in enumerate(allowed) if selector != parent_selector
        )
        if not revised:
            raise SlotSamplerError(reason="remasked slot has no alternate selector class")
        return revised

    def _parse_remask_indexes(self, indexes: tuple[int, ...]) -> tuple[int, ...]:
        if not indexes:
            raise SlotSamplerError(reason="revision selected no slots")
        unique = tuple(dict.fromkeys(indexes))
        mutable = set(self._schema.mutable_indexes)
        for index in unique:
            if index not in mutable:
                raise SlotSamplerError(reason=f"slot is not mutable: {index}")
        return unique
