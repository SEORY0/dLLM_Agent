"""Selective committed-token revision sampler for OpenEXR selector slots."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from dllm_agent.targets.openexr.particle import FIELD_ORDER
from tools.runtime.selective_revision_fill import FillRequest, SlotFiller, SlotModel
from tools.runtime.selective_revision_response import sampler_response
from tools.runtime.selective_revision_template import (
    FillResult,
    SlotTokenizer,
    Template,
    build_template,
    field_selector_token_ids,
    numeric_token_ids,
    selected_indexes,
)
from tools.runtime.selective_revision_validation import validate_parent
from tools.selective_revision_payloads import (
    request_hash,
    sha256_bytes,
)

if TYPE_CHECKING:
    from tools.runtime.selective_revision_json import (
        InitializeRequest,
        ResponseStatus,
        ReviseRequest,
        SamplerResponse,
    )


class SelectiveRevisionSampler:
    """Pure selective-revision core over a tiny slot-model protocol."""

    def __init__(
        self,
        tokenizer: SlotTokenizer,
        model: SlotModel,
        *,
        mask_token_id: int,
        block_size: int,
        prompt_context: str = "",
    ) -> None:
        """Create a sampler with a fixed selector template and numeric-token set."""
        self._mask_token_id: int = mask_token_id
        self._template: Template = build_template(
            tokenizer, mask_token_id, block_size=block_size, prompt_context=prompt_context
        )
        numeric_tokens = numeric_token_ids(tokenizer)
        self._filler: SlotFiller = SlotFiller(
            tokenizer=tokenizer,
            model=model,
            template=self._template,
            field_allowed_token_ids=tuple(
                field_selector_token_ids(tokenizer, numeric_tokens, field) for field in FIELD_ORDER
            ),
            block_size=block_size,
        )
        self._cache: dict[str, SamplerResponse] = {}

    def initialize(self, request: InitializeRequest) -> SamplerResponse:
        """Fill every selector slot from an all-mask initial state."""
        fill = self._filler.fill(
            FillRequest(
                token_ids=self._template.token_ids,
                slot_indexes=tuple(range(len(self._template.slot_positions))),
                previous_token_ids=tuple(None for _ in self._template.slot_positions),
                steps=request.steps,
                seed=request.seed,
            )
        )
        return self._response(
            request_id=request.request_id,
            request_sha256=request_hash(request),
            status="initialized",
            fill=fill,
            before_remask_token_ids=tuple(
                self._mask_token_id for _ in self._template.slot_positions
            ),
            remasked_positions=self._template.slot_positions,
            rng_state_sha256=sha256_bytes(f"initial:{request.seed}".encode()),
            cache_hit=False,
        )

    def revise(self, request: ReviseRequest) -> SamplerResponse:
        """Reopen selected committed slots and denoise only those slots."""
        key = request_hash(request)
        cached = self._cache.get(key)
        if cached is not None:
            return replace(cached, cache_hit=True)
        validate_parent(request, self._template)
        indexes = selected_indexes(request.remask_bitmap)
        positions = tuple(request.parent.slot_positions[index] for index in indexes)
        before = tuple(request.parent.token_ids[position] for position in positions)
        remasked = list(request.parent.token_ids)
        for position in positions:
            remasked[position] = self._mask_token_id
        fill = self._filler.fill(
            FillRequest(
                token_ids=tuple(remasked),
                slot_indexes=indexes,
                previous_token_ids=before,
                steps=request.revision_steps,
                seed=request.seed,
            )
        )
        status: ResponseStatus = (
            "revised" if fill.token_ids != request.parent.token_ids else "unchanged_revision"
        )
        response = self._response(
            request_id=request.request_id,
            request_sha256=key,
            status=status,
            fill=fill,
            before_remask_token_ids=before,
            remasked_positions=positions,
            rng_state_sha256=sha256_bytes(f"{request.rng_state_sha256}:{request.seed}".encode()),
            cache_hit=False,
        )
        self._cache[key] = response
        return response

    def _response(  # noqa: PLR0913 - mirrors response payload evidence fields.
        self,
        *,
        request_id: str,
        request_sha256: str,
        status: ResponseStatus,
        fill: FillResult,
        before_remask_token_ids: tuple[int, ...],
        remasked_positions: tuple[int, ...],
        rng_state_sha256: str,
        cache_hit: bool,
    ) -> SamplerResponse:
        return sampler_response(
            template=self._template,
            request_id=request_id,
            request_sha256=request_sha256,
            status=status,
            before_remask_token_ids=before_remask_token_ids,
            fill=fill,
            remasked_positions=remasked_positions,
            rng_state_sha256=rng_state_sha256,
            cache_hit=cache_hit,
        )
