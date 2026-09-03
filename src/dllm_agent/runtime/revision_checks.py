"""Validation and timeout helpers for the resident revision worker client."""

from __future__ import annotations

import concurrent.futures
from typing import IO, TYPE_CHECKING, NoReturn

from tools.runtime.selective_revision_json import (
    InitializeRequest,
    ReadyRecord,
    ReviseRequest,
    SamplerResponse,
    WireProtocolError,
)
from tools.runtime.selective_revision_response_checks import (
    trajectory_sha256,
    validate_response_trajectory,
)
from tools.selective_revision_payloads import (
    MODEL_ID,
    MODEL_REVISION,
    REMOTE_CODE_SHA256,
    sha256_token_ids,
)

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.revision.controller_models import RevisionResult, RevisionTrajectoryStep
from dllm_agent.targets.openexr.particle import FIELD_ORDER, MATERIALIZED_MUTABLE_FIELDS

_MUTABLE_FIELD_SET = frozenset(MATERIALIZED_MUTABLE_FIELDS)

if TYPE_CHECKING:
    from dllm_agent.revision.controller_models import RevisionRequest


def read_line_with_timeout(stream: IO[str], timeout_seconds: float) -> str:
    """Read one line from a stream while enforcing the configured timeout."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(stream.readline)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError as error:
        _ = future.cancel()
        _fail("revision worker response timeout", error)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def validate_ready(ready: ReadyRecord) -> None:
    """Validate the worker ready record against pinned provenance."""
    if (
        ready.model_id != MODEL_ID
        or ready.revision != MODEL_REVISION
        or ready.remote_code_sha256 != REMOTE_CODE_SHA256
    ):
        _fail("revision worker ready provenance mismatch")
    if ready.load_count != 1:
        _fail("revision worker model load count mismatch")


def validate_revision_request(request: ReviseRequest) -> None:
    """Validate Task 4 remask/frozen-field alignment before worker I/O."""
    all_slots = (1 << len(FIELD_ORDER)) - 1
    if request.remask_bitmap & request.frozen_bitmap:
        _fail("revision remask and frozen bitmaps overlap")
    if (request.remask_bitmap | request.frozen_bitmap) != all_slots:
        _fail("revision remask and frozen bitmaps do not cover all slots")
    if request.parent_token_sha256 != sha256_token_ids(request.parent.token_ids):
        _fail("revision parent token hash mismatch")


def validate_controller_revision(request: RevisionRequest) -> None:
    """Validate controller field names before translating to the worker wire payload."""
    if request.changed_field not in request.remask_fields:
        _fail("changed field must be included in remask fields")
    if any(field not in _MUTABLE_FIELD_SET for field in request.remask_fields):
        _fail("remask fields must change validity-preserving materialized bytes")
    remask_bitmap = _field_bitmap(request.remask_fields)
    if request.remask_bitmap != remask_bitmap:
        _fail("remask fields do not match remask bitmap")
    frozen_bitmap = ((1 << len(FIELD_ORDER)) - 1) ^ request.remask_bitmap
    if request.frozen_bitmap != frozen_bitmap:
        _fail("frozen bitmap must complement remask bitmap")


def validate_sampler_response(
    response: SamplerResponse,
    payload: InitializeRequest | ReviseRequest,
) -> None:
    """Validate response state against the request that produced it."""
    if response.token_state_sha256 != sha256_token_ids(response.token_ids):
        _fail("revision worker returned mismatched token hash")
    if len(response.slot_positions) != len(FIELD_ORDER):
        _fail("revision worker returned wrong slot count")
    _validate_trajectory(response)
    if isinstance(payload, ReviseRequest):
        _validate_revise_response(response, payload)


def to_revision_result(response: SamplerResponse) -> RevisionResult:
    """Map a validated worker response into the Task 4 sampler result."""
    return RevisionResult(
        selectors=response.selectors,
        token_ids=response.token_ids,
        slot_positions=response.slot_positions,
        selected_slot_token_ids=response.selected_slot_token_ids,
        token_state_sha256=response.token_state_sha256,
        transfer_evidence_sha256=response.transfer_evidence_sha256,
        rng_state_sha256=response.rng_state_sha256,
        nfe=response.nfe,
        forward_calls=response.forward_calls,
        gpu_ms=response.gpu_ms,
        cache_hit=response.cache_hit,
        trajectory_uncertainty=response.trajectory_uncertainty,
        suggested_field=response.suggested_field,
        trajectory_steps=tuple(
            RevisionTrajectoryStep(
                step_index=step.step_index,
                block_index=step.block_index,
                masked_before=step.masked_before,
                committed_positions=step.committed_positions,
                token_ids=step.token_ids,
                confidences=step.confidences,
                proposal_positions=step.proposal_positions,
                proposal_token_ids=step.proposal_token_ids,
                proposal_confidences=step.proposal_confidences,
                proposal_margins=step.proposal_margins,
                remaining_masked=step.remaining_masked,
            )
            for step in response.trajectory_steps
        ),
        trajectory_evidence_sha256=trajectory_sha256(response),
    )


def _validate_trajectory(response: SamplerResponse) -> None:
    try:
        validate_response_trajectory(response)
    except WireProtocolError as error:
        _fail(str(error), error)


def _validate_revise_response(response: SamplerResponse, payload: ReviseRequest) -> None:
    if response.slot_positions != payload.parent.slot_positions:
        _fail("revision worker returned mismatched slot positions")
    expected_positions = tuple(
        payload.parent.slot_positions[index]
        for index in range(len(FIELD_ORDER))
        if payload.remask_bitmap & (1 << index)
    )
    if response.remasked_positions != expected_positions:
        _fail("revision worker returned mismatched remask positions")
    for position in set(payload.parent.slot_positions) - set(expected_positions):
        if response.token_ids[position] != payload.parent.token_ids[position]:
            _fail("revision worker changed a frozen slot")


def _field_bitmap(fields: tuple[str, ...]) -> int:
    bitmap = 0
    for field in fields:
        bitmap |= 1 << FIELD_ORDER.index(field)
    return bitmap


def _fail(detail: str, cause: BaseException | None = None) -> NoReturn:
    error = HuntSetupError(detail)
    if cause is None:
        raise error
    raise error from cause
