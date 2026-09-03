"""Strict proposal trajectory response boundary tests."""

import json
from dataclasses import asdict, replace

import pytest
from tools.runtime.selective_revision_json import (
    InitializeRequest,
    JsonWireError,
    SamplerResponse,
    TrajectoryStep,
    json_value,
)
from tools.selective_revision_payloads import (
    parse_response_line,
    response_with_hash,
    sha256_token_ids,
)

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.runtime.revision_checks import to_revision_result, validate_sampler_response


def test_response_wire_maps_strict_proposal_trajectory_evidence() -> None:
    """Given a v2 response, when parsed, then proposal arrays reach RevisionResult unchanged."""
    # Given
    response = _response()

    # When
    parsed = parse_response_line(json.dumps(asdict(response)))
    assert isinstance(parsed, SamplerResponse)
    result = to_revision_result(parsed)

    # Then
    assert result.trajectory_steps[0].proposal_positions == (2,)
    assert result.trajectory_steps[0].proposal_token_ids == (12,)
    assert result.trajectory_steps[0].proposal_confidences == (0.75,)
    assert result.trajectory_steps[0].proposal_margins == (0.25,)


def test_response_wire_rejects_missing_proposal_field_and_old_schema() -> None:
    """Given incomplete or v1 evidence, when parsed, then the strict boundary rejects it."""
    # Given
    missing = json_value(asdict(_response()))
    assert isinstance(missing, dict)
    step = json_value(asdict(_response().trajectory_steps[0]))
    assert isinstance(step, dict)
    missing["trajectory_steps"] = [
        {key: value for key, value in step.items() if key != "proposal_margins"}
    ]
    old_schema = json_value(asdict(_response()))
    assert isinstance(old_schema, dict)
    old_schema["schema_version"] = "dllm-selective-revision-response/v1"

    # When / Then
    with pytest.raises(JsonWireError, match="missing=proposal_margins"):
        _ = parse_response_line(json.dumps(missing))
    with pytest.raises(JsonWireError, match="schema_version"):
        _ = parse_response_line(json.dumps(old_schema))


def test_response_validation_rejects_misaligned_proposal_arrays() -> None:
    """Given two positions and one proposal token, when validated, then the row fails closed."""
    # Given
    response = _response()
    step = replace(response.trajectory_steps[0], proposal_positions=(2, 3))
    malformed = replace(response, trajectory_steps=(step,))

    # When / Then
    with pytest.raises(HuntSetupError, match="proposal lengths"):
        validate_sampler_response(malformed, _request())


def test_proposal_evidence_changes_only_response_and_trajectory_hashes() -> None:
    """Given equal commits, changing a margin preserves state and transfer hashes."""
    # Given
    response = _response()
    changed_step = replace(response.trajectory_steps[0], proposal_margins=(0.1,))
    changed = response_with_hash(
        replace(response, response_hash="", trajectory_steps=(changed_step,))
    )

    # When
    result = to_revision_result(response)
    changed_result = to_revision_result(changed)

    # Then
    assert changed.token_state_sha256 == response.token_state_sha256
    assert changed.transfer_evidence_sha256 == response.transfer_evidence_sha256
    assert changed.response_hash != response.response_hash
    assert changed_result.trajectory_evidence_sha256 != result.trajectory_evidence_sha256


def _response() -> SamplerResponse:
    token_ids = (10, 11, 12, 13, 10, 11, 12, 13, 10, 11)
    return response_with_hash(
        SamplerResponse(
            kind="response",
            schema_version="dllm-selective-revision-response/v2",
            request_id="init-1",
            request_sha256="a" * 64,
            response_hash="",
            status="initialized",
            selectors=(0, 1, 2, 3, 0, 1, 2, 3, 0, 1),
            token_ids=token_ids,
            slot_positions=tuple(range(10)),
            selected_slot_token_ids=(12,),
            before_remask_token_ids=(99,),
            remasked_positions=(2,),
            token_state_sha256=sha256_token_ids(token_ids),
            transfer_evidence_sha256="d" * 64,
            rng_state_sha256="e" * 64,
            nfe=1,
            forward_calls=1,
            gpu_ms=2,
            cache_hit=False,
            trajectory_uncertainty=0.25,
            suggested_field="width",
            invalidated_from_block=0,
            forward_blocks=(0,),
            trajectory_steps=(
                TrajectoryStep(
                    step_index=1,
                    block_index=0,
                    masked_before=1,
                    committed_positions=(2,),
                    token_ids=(12,),
                    confidences=(0.75,),
                    proposal_positions=(2,),
                    proposal_token_ids=(12,),
                    proposal_confidences=(0.75,),
                    proposal_margins=(0.25,),
                    remaining_masked=0,
                ),
            ),
        )
    )


def _request() -> InitializeRequest:
    return InitializeRequest(request_id="init-1", particle_id="particle-1", seed=1, steps=1)
