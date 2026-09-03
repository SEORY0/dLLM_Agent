"""One-candidate Magma probe execution contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from dllm_agent.experiments.multitarget.candidate_bank import Candidate, ProbeObservation
from dllm_agent.experiments.multitarget.container_monitor import (
    CanarySeries,
    ContainerLaunch,
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.docker_runtime import runtime_profile
from dllm_agent.experiments.multitarget.probe_runner import (
    ProbeRunnerError,
    ProbeRunRequest,
    WaitProbe,
    run_candidate_probe,
)
from dllm_agent.experiments.multitarget.slot_sampler import SlotSample
from dllm_agent.targets.structured import MaterializedSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from dllm_agent.experiments.multitarget.docker_runtime import ContainerRequest

IMAGE_ID = "sha256:" + ("a" * 64)


@dataclass(frozen=True, slots=True)
class ProbeCalls:
    request: ContainerRequest | None = None
    shared_dir: Path | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class ContainerState:
    exit_code: int | None = 0
    docker_status: str | None = "exited"
    oom_killed: bool | None = False
    timed_out: bool = False
    process_error: str | None = None


def test_probe_stages_one_candidate_and_returns_observation(tmp_path: Path) -> None:
    # Given
    snapshot = MaterializedSnapshot(
        content=b"candidate-bytes",
        sha256=hashlib.sha256(b"candidate-bytes").hexdigest(),
        anchor_sha256="b" * 64,
        field_spans=(),
    )
    candidate = _candidate(snapshot)
    calls = ProbeCalls()

    def launch(request: ContainerRequest) -> ContainerLaunch:
        object.__setattr__(calls, "request", request)
        return ContainerLaunch(container_id="abc123")

    def wait(
        container_id: str,
        *,
        shared_dir: Path,
        timeout_seconds: int,
    ) -> ContainerRunResult:
        assert container_id == "abc123"
        object.__setattr__(calls, "shared_dir", shared_dir)
        object.__setattr__(calls, "timeout_seconds", timeout_seconds)
        return _result(
            logs_text="prefix\nERROR: AddressSanitizer: heap-buffer-overflow\n",
            canaries={"BUG_A": CanarySeries(reached=((0, 1),), triggered=((30, 1),))},
        )

    # When
    result = run_candidate_probe(
        ProbeRunRequest(
            name="dllm-libpng-probe-p0001",
            profile=runtime_profile("libpng"),
            image_id=IMAGE_ID,
            work_dir=tmp_path / "probe",
            candidate=candidate,
            duration_seconds=30,
            fuzzer_seed=7,
            cpu_index=2,
            poll_seconds=5,
        ),
        launch=launch,
        wait=wait,
    )

    # Then
    assert isinstance(result.observation, ProbeObservation)
    assert result.observation.candidate_sha256 == snapshot.sha256
    assert result.observation.feedback_source_sha256 == snapshot.sha256
    assert result.observation.canary_triggered is True
    assert result.observation.sanitizer_observed is True
    assert result.observation.coverage_edges == 14
    assert result.observation.value_profile_features == 33
    assert result.observation.executions_per_second == 200
    assert result.observation.log_sha256 == result.run_result.log_sha256
    assert result.snapshot_path.name == f"{snapshot.sha256}.bin"
    assert result.snapshot_path.read_bytes() == b"candidate-bytes"
    assert tuple(result.corpus_dir.iterdir()) == (result.snapshot_path,)
    assert result.log_path.read_text(encoding="utf-8").startswith("prefix")
    assert calls.request is not None
    assert calls.request.name == "dllm-libpng-probe-p0001"
    assert calls.request.profile == runtime_profile("libpng")
    assert calls.request.image_id == IMAGE_ID
    assert calls.request.corpus_dir == tmp_path / "probe" / "corpus"
    assert calls.request.shared_dir == tmp_path / "probe" / "shared"
    assert calls.request.duration_seconds == 30
    assert calls.request.fuzzer_seed == 7
    assert calls.request.cpu_index == 2
    assert calls.shared_dir == tmp_path / "probe" / "shared"
    assert calls.timeout_seconds == 90


def test_probe_refuses_to_reuse_preexisting_work_dirs(tmp_path: Path) -> None:
    # Given
    work_dir = tmp_path / "probe"
    (work_dir / "corpus").mkdir(parents=True)
    _ = (work_dir / "corpus" / "old").write_bytes(b"old")

    # When / Then
    with pytest.raises(ProbeRunnerError, match="must not already exist"):
        _ = run_candidate_probe(
            ProbeRunRequest(
                name="dllm-libpng-probe-p0001",
                profile=runtime_profile("libpng"),
                image_id=IMAGE_ID,
                work_dir=work_dir,
                candidate=_candidate(
                    MaterializedSnapshot(
                        content=b"candidate",
                        sha256=hashlib.sha256(b"candidate").hexdigest(),
                        anchor_sha256="b" * 64,
                        field_spans=(),
                    )
                ),
                duration_seconds=30,
                fuzzer_seed=7,
                cpu_index=2,
                poll_seconds=5,
            ),
            launch=_forbidden_launch,
            wait=_forbidden_wait,
        )


@pytest.mark.parametrize(
    ("state", "logs_text"),
    [
        (ContainerState(exit_code=77), "nonzero"),
        (ContainerState(docker_status="running"), "running"),
        (ContainerState(timed_out=True), "timeout"),
        (ContainerState(process_error="docker logs failed"), "error"),
    ],
)
def test_probe_fails_closed_when_container_result_is_not_clean(
    tmp_path: Path,
    state: ContainerState,
    logs_text: str,
) -> None:
    # Given
    candidate = _candidate(
        MaterializedSnapshot(
            content=b"candidate",
            sha256=hashlib.sha256(b"candidate").hexdigest(),
            anchor_sha256="b" * 64,
            field_spans=(),
        )
    )
    run_result = _result(logs_text=logs_text, canaries={}, state=state)

    # When / Then
    with pytest.raises(ProbeRunnerError, match="container did not exit cleanly"):
        _ = run_candidate_probe(
            ProbeRunRequest(
                name="dllm-libpng-probe-p0001",
                profile=runtime_profile("libpng"),
                image_id=IMAGE_ID,
                work_dir=tmp_path / "probe",
                candidate=candidate,
                duration_seconds=30,
                fuzzer_seed=7,
                cpu_index=2,
                poll_seconds=5,
            ),
            launch=_successful_launch,
            wait=_wait_result(run_result),
        )
    assert (tmp_path / "probe" / "probe.log").read_text(encoding="utf-8") == run_result.logs_text


def test_probe_records_descendant_oom_when_supervisor_exits_zero(tmp_path: Path) -> None:
    candidate = _candidate(
        MaterializedSnapshot(
            content=b"candidate",
            sha256=hashlib.sha256(b"candidate").hexdigest(),
            anchor_sha256="b" * 64,
            field_spans=(),
        )
    )

    result = run_candidate_probe(
        ProbeRunRequest(
            name="dllm-libtiff-probe-p0001",
            profile=runtime_profile("libtiff"),
            image_id=IMAGE_ID,
            work_dir=tmp_path / "probe",
            candidate=candidate,
            duration_seconds=30,
            fuzzer_seed=7,
            cpu_index=2,
            poll_seconds=5,
        ),
        launch=_successful_launch,
        wait=_wait_result(
            _result(
                logs_text="fork child OOM was ignored by libFuzzer",
                canaries={},
                state=ContainerState(oom_killed=True),
            )
        ),
    )

    assert result.run_result.exit_code == 0
    assert result.run_result.oom_killed is True


def test_probe_keeps_search_canary_and_asan_feedback_independent(tmp_path: Path) -> None:
    # Given
    candidate = _candidate(
        MaterializedSnapshot(
            content=b"candidate",
            sha256=hashlib.sha256(b"candidate").hexdigest(),
            anchor_sha256="b" * 64,
            field_spans=(),
        )
    )

    # When
    result = run_candidate_probe(
        ProbeRunRequest(
            name="dllm-libpng-probe-p0001",
            profile=runtime_profile("libpng"),
            image_id=IMAGE_ID,
            work_dir=tmp_path / "probe",
            candidate=candidate,
            duration_seconds=30,
            fuzzer_seed=7,
            cpu_index=2,
            poll_seconds=5,
        ),
        launch=_successful_launch,
        wait=_wait_result(
            _result(
                logs_text="libFuzzer fatal signal without sanitizer proof",
                canaries={"BUG_A": CanarySeries(reached=((0, 1),), triggered=((30, 1),))},
            )
        ),
    )

    # Then
    assert result.observation.canary_triggered is True
    assert result.observation.sanitizer_observed is False


def _candidate(snapshot: MaterializedSnapshot) -> Candidate:
    return Candidate(
        sample=SlotSample(
            token_ids=(1,),
            selected_slot_token_ids=(1,),
            selectors=(0,),
            slot_positions=(0,),
            slot_confidences=(0.5,),
            trajectory_steps=(),
            state_sha256="c" * 64,
            parent_state_sha256=None,
            requested_steps=1,
            nfe=1,
            forward_calls=1,
            gpu_ms=1,
        ),
        snapshot=snapshot,
        source_sample_sha256="d" * 64,
    )


def _forbidden_launch(request: ContainerRequest) -> ContainerLaunch:
    _ = request
    pytest.fail("launch must not be called")


def _forbidden_wait(
    container_id: str,
    *,
    shared_dir: Path,
    timeout_seconds: int,
) -> ContainerRunResult:
    _ = (container_id, shared_dir, timeout_seconds)
    pytest.fail("wait must not be called")


def _successful_launch(request: ContainerRequest) -> ContainerLaunch:
    _ = request
    return ContainerLaunch(container_id="abc123")


def _wait_result(run_result: ContainerRunResult) -> WaitProbe:
    def wait(
        container_id: str,
        *,
        shared_dir: Path,
        timeout_seconds: int,
    ) -> ContainerRunResult:
        _ = (container_id, shared_dir, timeout_seconds)
        return run_result

    return wait


def _result(
    *,
    logs_text: str,
    canaries: Mapping[str, CanarySeries],
    state: ContainerState | None = None,
) -> ContainerRunResult:
    result_state = state or ContainerState()
    return ContainerRunResult(
        container_id="abc123",
        exit_code=result_state.exit_code,
        docker_status=result_state.docker_status,
        oom_killed=result_state.oom_killed,
        timed_out=result_state.timed_out,
        process_error=result_state.process_error,
        logs_text=logs_text,
        log_sha256=hashlib.sha256(logs_text.encode()).hexdigest(),
        metrics=FuzzerMetrics(
            coverage_edges=14,
            value_profile_features=33,
            executions_per_second=200,
            executed_units=9001,
        ),
        canaries=canaries,
        findings=(),
    )
