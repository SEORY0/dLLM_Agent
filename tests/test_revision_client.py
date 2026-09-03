import json
from dataclasses import asdict
from pathlib import Path

import pytest
from tools.runtime.selective_revision_json import (
    InitializeRequest,
    ReviseRequest,
    SamplerResponse,
    SelectorState,
    TrajectoryStep,
)
from tools.selective_revision_payloads import (
    initialize_worker_request,
    request_hash,
    response_with_hash,
    revise_worker_request,
    sha256_bytes,
    sha256_token_ids,
    shutdown_request,
)

from dllm_agent.revision.controller_models import InitialParticleRequest, RevisionRequest
from dllm_agent.runtime.revision_client import RevisionWorkerClient
from dllm_agent.runtime.revision_process import WorkerProcessFactory
from dllm_agent.targets.openexr.particle import FIELD_ORDER, SelectorSlots
from tests.revision_client_fakes import FakeFactory, FakePipe, FakeProcess


def test_client_serves_multiple_requests_from_one_ready_worker(tmp_path: Path) -> None:
    # Given
    init_payload = InitializeRequest(
        request_id="init-000001",
        particle_id="particle-1",
        seed=1,
        steps=8,
    )
    init_response = _response("init-000001", request_hash(initialize_worker_request(init_payload)))
    revise_payload = _revise_payload(
        token_ids=(10, 11, 12, 13, 10, 11, 12, 13, 10, 11),
        slot_positions=tuple(range(10)),
        selectors=(0, 1, 2, 3, 0, 1, 2, 3, 0, 1),
    )
    process = FakeProcess(
        stdout=FakePipe(
            (
                _ready(load_count=1),
                init_response,
                _response("revise-000002", request_hash(revise_worker_request(revise_payload))),
                _shutdown("shutdown-000003"),
            )
        )
    )
    factory = FakeFactory(process)
    client = _client(tmp_path, factory)

    # When
    with client as opened:
        initial = opened.initial(_initial())
        revised = opened.revise(
            _revision(initial.token_ids, initial.slot_positions, initial.selectors)
        )

    # Then
    assert opened.ready is not None
    assert opened.ready.load_count == 1
    assert initial.token_state_sha256 == sha256_token_ids(initial.token_ids)
    assert revised.token_state_sha256 == sha256_token_ids(revised.token_ids)
    assert len(factory.calls) == 1
    command, environment = factory.calls[0]
    assert command[0] == str(tmp_path / "python")
    assert command[1].endswith("tools/selective_revision_worker.py")
    assert "--prompt-path" in command
    assert "--prompt-sha256" in command
    assert "HF_HUB_OFFLINE" in environment
    assert process.stdin.writes[0].startswith('{"operation":"initialize"')
    assert process.stdin.writes[1].startswith('{"operation":"revise"')
    assert process.stdin.was_closed is True


def test_client_fails_closed_for_bad_worker_lines(tmp_path: Path) -> None:
    # Given
    too_long = "{" + ("x" * 5000) + "}\n"
    client = _client(tmp_path, FakeFactory(FakeProcess(stdout=FakePipe((too_long,)))))

    # When / Then
    with pytest.raises(Exception, match="line length"), client:
        pass

    malformed = _client(tmp_path, FakeFactory(FakeProcess(stdout=FakePipe(("not-json\n",)))))
    with pytest.raises(Exception, match="malformed"), malformed:
        pass


def test_client_rejects_mismatched_response_hash_and_eof(tmp_path: Path) -> None:
    # Given
    mismatch = _client(
        tmp_path,
        FakeFactory(FakeProcess(stdout=FakePipe((_ready(), _response("wrong-id", "a" * 64))))),
    )

    # When / Then
    with pytest.raises(Exception, match="request id"), mismatch as opened:
        _ = opened.initial(_initial())

    eof = _client(tmp_path, FakeFactory(FakeProcess(stdout=FakePipe((_ready(),)))))
    with pytest.raises(Exception, match="EOF"), eof as opened:
        _ = opened.initial(_initial())


def test_default_factory_uses_no_shell() -> None:
    # Given
    factory = WorkerProcessFactory()

    # When / Then
    assert factory.shell is False


def test_historical_positional_client_construction_keeps_line_limit_and_factory_slots(
    tmp_path: Path,
) -> None:
    # Given
    prompt = tmp_path / "prompt.md"
    _ = prompt.write_text("prompt context", encoding="utf-8")
    factory = FakeFactory(FakeProcess(stdout=FakePipe()))

    # When
    client = RevisionWorkerClient(
        tmp_path / "python",
        tmp_path / "cache",
        prompt,
        sha256_bytes(prompt.read_bytes()),
        "cuda:0",
        1.0,
        4096,
        factory,
    )

    # Then
    assert client.read_only_cache is False
    assert client.max_line_bytes == 4096
    assert client.factory is factory


def _client(tmp_path: Path, factory: FakeFactory) -> RevisionWorkerClient:
    prompt_path = tmp_path / "prompt.md"
    _ = prompt_path.write_text("prompt context", encoding="utf-8")
    prompt_sha256 = sha256_bytes(prompt_path.read_bytes())
    return RevisionWorkerClient(
        runtime_python=tmp_path / "python",
        cache_dir=tmp_path / "cache",
        prompt_path=prompt_path,
        prompt_sha256=prompt_sha256,
        device="cuda:0",
        timeout_seconds=1.0,
        max_line_bytes=4096,
        factory=factory,
    )


def _ready(load_count: int = 1) -> str:
    remote_code_sha256 = "298b19a464f32c00d38e43603516174a73fb4c531193369ff1aabbef32593104"
    return (
        json.dumps(
            {
                "kind": "ready",
                "schema_version": "dllm-selective-revision-ready/v1",
                "model_id": "ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
                "revision": "10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
                "remote_code_sha256": remote_code_sha256,
                "device": "cuda:0",
                "load_count": load_count,
                "prompt_sha256": sha256_bytes(b"prompt context"),
            },
            sort_keys=True,
        )
        + "\n"
    )


def _response(request_id: str, request_sha256: str) -> str:
    token_ids = (10, 11, 12, 13, 10, 11, 12, 13, 10, 11)
    response = response_with_hash(
        SamplerResponse(
            kind="response",
            schema_version="dllm-selective-revision-response/v2",
            request_id=request_id,
            request_sha256=request_sha256,
            response_hash="",
            status="revised",
            selectors=(0, 1, 2, 3, 0, 1, 2, 3, 0, 1),
            token_ids=token_ids,
            slot_positions=tuple(range(10)),
            selected_slot_token_ids=(12,),
            before_remask_token_ids=(12,),
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
    return json.dumps(asdict(response), sort_keys=True) + "\n"


def _shutdown(request_id: str) -> str:
    return (
        json.dumps(
            {
                "kind": "shutdown",
                "request_id": request_id,
                "request_sha256": request_hash(shutdown_request(request_id)),
                "status": "ok",
            },
            sort_keys=True,
        )
        + "\n"
    )


def _initial() -> InitialParticleRequest:
    return InitialParticleRequest(
        particle_id="particle-1",
        seed=1,
        steps=8,
        arm="selective-revision",
        mode="initial",
    )


def _revision(
    token_ids: tuple[int, ...], slot_positions: tuple[int, ...], selectors: SelectorSlots
) -> RevisionRequest:
    return RevisionRequest(
        particle_id="particle-1",
        parent_token_ids=token_ids,
        parent_selectors=selectors,
        slot_positions=slot_positions,
        remask_fields=("width",),
        changed_field="width",
        remask_bitmap=1 << FIELD_ORDER.index("width"),
        frozen_bitmap=((1 << len(FIELD_ORDER)) - 1) ^ (1 << FIELD_ORDER.index("width")),
        revision_steps=8,
        seed=2,
        rng_state_sha256="e" * 64,
        feedback_digest="f" * 64,
        mode="selective-revision/v1",
    )


def _revise_payload(
    token_ids: tuple[int, ...], slot_positions: tuple[int, ...], selectors: SelectorSlots
) -> ReviseRequest:
    parent_sha256 = sha256_token_ids(token_ids)
    return ReviseRequest(
        request_id="revise-000002",
        particle_id="particle-1",
        parent=SelectorState(
            token_ids=token_ids,
            slot_positions=slot_positions,
            selectors=selectors,
            token_state_sha256=parent_sha256,
        ),
        parent_token_sha256=parent_sha256,
        remask_bitmap=1 << FIELD_ORDER.index("width"),
        frozen_bitmap=((1 << len(FIELD_ORDER)) - 1) ^ (1 << FIELD_ORDER.index("width")),
        revision_steps=8,
        seed=2,
        rng_state_sha256="e" * 64,
        feedback_digest="f" * 64,
        mode="selective-revision/v1",
    )
