"""Response assembly for selective-revision sampler output."""

from __future__ import annotations

from tools.runtime.selective_revision_json import (
    ResponseStatus,
    SamplerResponse,
    canonical_json,
)
from tools.runtime.selective_revision_template import (
    FillResult,
    Template,
    suggested_field,
    uncertainty,
)
from tools.selective_revision_payloads import (
    RESPONSE_SCHEMA_VERSION,
    response_with_hash,
    sha256_bytes,
    sha256_token_ids,
)


def sampler_response(  # noqa: PLR0913
    *,
    template: Template,
    request_id: str,
    request_sha256: str,
    status: ResponseStatus,
    fill: FillResult,
    before_remask_token_ids: tuple[int, ...],
    remasked_positions: tuple[int, ...],
    rng_state_sha256: str,
    cache_hit: bool,
) -> SamplerResponse:
    """Build the canonical response payload with bound response hash."""
    transfer_hash = sha256_bytes(
        canonical_json(
            {
                "confidences": list(fill.confidences),
                "remasked_positions": list(remasked_positions),
                "tokens": list(fill.selected_slot_token_ids),
            }
        ).encode()
    )
    first_block = min(fill.forward_blocks) if fill.forward_blocks else 0
    response = SamplerResponse(
        kind="response",
        schema_version=RESPONSE_SCHEMA_VERSION,
        request_id=request_id,
        request_sha256=request_sha256,
        response_hash="",
        status=status,
        selectors=fill.selectors,
        token_ids=fill.token_ids,
        slot_positions=template.slot_positions,
        selected_slot_token_ids=fill.selected_slot_token_ids,
        before_remask_token_ids=before_remask_token_ids,
        remasked_positions=remasked_positions,
        token_state_sha256=sha256_token_ids(fill.token_ids),
        transfer_evidence_sha256=transfer_hash,
        rng_state_sha256=rng_state_sha256,
        nfe=fill.nfe,
        forward_calls=fill.nfe,
        gpu_ms=fill.gpu_ms,
        cache_hit=cache_hit,
        trajectory_uncertainty=uncertainty(fill.confidences),
        suggested_field=suggested_field(fill.selected_slot_indexes, fill.confidences),
        invalidated_from_block=first_block,
        forward_blocks=fill.forward_blocks,
        trajectory_steps=fill.trajectory_steps,
    )
    return response_with_hash(response)
