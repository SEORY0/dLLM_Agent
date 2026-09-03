"""One-candidate Magma micro-probe execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

from dllm_agent.experiments.multitarget.candidate_bank import Candidate, ProbeObservation
from dllm_agent.experiments.multitarget.container_monitor import (
    ContainerLaunch,
    ContainerRunResult,
    launch_container,
    supervisor_exited_cleanly,
    wait_for_container,
)
from dllm_agent.experiments.multitarget.docker_runtime import (
    ContainerRequest,
    RuntimeProfile,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_ASAN_MARKER = re.compile(r"ERROR:\s*AddressSanitizer")


class LaunchProbe(Protocol):
    """Launch seam for tests and real Docker execution."""

    def __call__(self, request: ContainerRequest) -> ContainerLaunch:
        """Launch one detached container."""
        ...


class WaitProbe(Protocol):
    """Wait seam for tests and real Docker execution."""

    def __call__(
        self,
        container_id: str,
        *,
        shared_dir: Path,
        timeout_seconds: int,
    ) -> ContainerRunResult:
        """Wait for one container and collect full evidence."""
        ...


@dataclass(slots=True)
class ProbeRunnerError(Exception):
    """Raised when a probe cannot be staged or did not complete cleanly."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ProbeRunRequest:
    """Boundary object for one candidate micro-probe."""

    name: str
    profile: RuntimeProfile
    image_id: str
    work_dir: Path
    candidate: Candidate
    duration_seconds: int
    fuzzer_seed: int
    cpu_index: int
    poll_seconds: int


@dataclass(frozen=True, slots=True)
class ProbeRunResult:
    """Probe observation with persisted staging and log evidence."""

    observation: ProbeObservation
    run_result: ContainerRunResult
    corpus_dir: Path
    shared_dir: Path
    snapshot_path: Path
    log_path: Path


def run_candidate_probe(
    request: ProbeRunRequest,
    *,
    launch: LaunchProbe = launch_container,
    wait: WaitProbe = wait_for_container,
) -> ProbeRunResult:
    """Stage exactly one candidate, run Magma, and return search observation."""
    corpus_dir = request.work_dir / "corpus"
    shared_dir = request.work_dir / "shared"
    log_path = request.work_dir / "probe.log"
    _prepare_probe_dirs(request.work_dir, corpus_dir, shared_dir, log_path)
    snapshot_path = corpus_dir / f"{request.candidate.snapshot.sha256}.bin"
    _ = snapshot_path.write_bytes(request.candidate.snapshot.content)
    container_request = ContainerRequest(
        name=request.name,
        profile=request.profile,
        image_id=request.image_id,
        shared_dir=shared_dir,
        corpus_dir=corpus_dir,
        duration_seconds=request.duration_seconds,
        fuzzer_seed=request.fuzzer_seed,
        cpu_index=request.cpu_index,
        poll_seconds=request.poll_seconds,
    )
    launch_result = launch(container_request)
    run_result = wait(
        launch_result.container_id,
        shared_dir=shared_dir,
        timeout_seconds=request.duration_seconds + 60,
    )
    _ = log_path.write_text(run_result.logs_text, encoding="utf-8")
    _require_clean_result(run_result)
    return ProbeRunResult(
        observation=_observation(request.candidate, run_result),
        run_result=run_result,
        corpus_dir=corpus_dir,
        shared_dir=shared_dir,
        snapshot_path=snapshot_path,
        log_path=log_path,
    )


def _prepare_probe_dirs(work_dir: Path, corpus_dir: Path, shared_dir: Path, log_path: Path) -> None:
    for path in (corpus_dir, shared_dir):
        if path.exists():
            raise ProbeRunnerError(reason=f"{path} must not already exist")
    if log_path.exists():
        raise ProbeRunnerError(reason=f"{log_path} must not already exist")
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir()
    shared_dir.mkdir()


def _require_clean_result(result: ContainerRunResult) -> None:
    if not supervisor_exited_cleanly(result):
        raise ProbeRunnerError(reason="container did not exit cleanly")


def _observation(candidate: Candidate, result: ContainerRunResult) -> ProbeObservation:
    return ProbeObservation(
        candidate_sha256=candidate.snapshot.sha256,
        feedback_source_sha256=candidate.snapshot.sha256,
        canary_triggered=_canary_triggered(result.canaries),
        sanitizer_observed=_ASAN_MARKER.search(result.logs_text) is not None,
        coverage_edges=result.metrics.coverage_edges,
        value_profile_features=result.metrics.value_profile_features,
        executions_per_second=result.metrics.executions_per_second,
        log_sha256=result.log_sha256,
    )


def _canary_triggered(canaries: Mapping[str, object]) -> bool:
    for series in canaries.values():
        triggered = getattr(series, "triggered", ())
        if triggered and triggered[-1][1] > 0:
            return True
    return False
