"""Shell-free Docker container monitoring for Magma multitarget campaigns."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast, override

from pydantic import TypeAdapter, ValidationError

from dllm_agent.experiments.multitarget.docker_runtime import (
    ContainerRequest,
    container_argv,
)
from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence, artifact_evidence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

type StateValue = str | int | bool | None

_FUZZER_PROGRESS = re.compile(r"cov:\s*(\d+).*?ft:\s*(\d+).*?exec/s:?\s*(\d+)")
_PROGRESS_UNITS = re.compile(r"#(\d+):.*?cov:")
_EXECUTED_UNITS = re.compile(r"stat::number_of_executed_units:\s*(\d+)")
_STATE_ADAPTER: Final = TypeAdapter(dict[str, StateValue])
_SUBPROCESS_RUN: Final = cast("ProcessRunner", subprocess.run)


class ProcessRunner(Protocol):
    """subprocess.run-compatible seam used by tests and real campaign runners."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float | None,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run a process with captured text streams and no shell."""
        ...


@dataclass(slots=True)
class ContainerMonitorError(Exception):
    """Raised when launch output is ambiguous or a boundary request is invalid."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ContainerLaunch:
    """Detached Docker container evidence from a launch call."""

    container_id: str


@dataclass(frozen=True, slots=True)
class FuzzerMetrics:
    """Last observed libFuzzer progress metrics from timestamped logs."""

    coverage_edges: int
    value_profile_features: int
    executions_per_second: int
    executed_units: int


@dataclass(frozen=True, slots=True)
class CanarySeries:
    """Reached and triggered counters for one Magma bug canary over time."""

    reached: tuple[tuple[int, int], ...]
    triggered: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ContainerRunResult:
    """Fail-closed result for a finished or interrupted Magma container."""

    container_id: str
    exit_code: int | None
    docker_status: str | None
    oom_killed: bool | None
    timed_out: bool
    process_error: str | None
    logs_text: str
    log_sha256: str
    metrics: FuzzerMetrics
    canaries: Mapping[str, CanarySeries]
    findings: tuple[ArtifactEvidence, ...]


def supervisor_exited_cleanly(result: ContainerRunResult) -> bool:
    """Return whether the campaign supervisor completed despite child findings."""
    return (
        not result.timed_out
        and result.process_error is None
        and result.exit_code == 0
        and result.docker_status == "exited"
    )


def launch_container(
    request: ContainerRequest,
    *,
    runner: ProcessRunner = _SUBPROCESS_RUN,
    timeout_seconds: int = 30,
) -> ContainerLaunch:
    """Launch a detached Docker container and return its exact single stdout id."""
    process = _run(runner, container_argv(request), timeout_seconds)
    if process.returncode != 0:
        raise ContainerMonitorError(reason=_process_failure("docker run", process))
    lines = tuple(line.strip() for line in process.stdout.splitlines() if line.strip())
    if len(lines) != 1:
        raise ContainerMonitorError(reason="docker run must print exactly one container id")
    return ContainerLaunch(container_id=lines[0])


def wait_for_container(
    container_id: str,
    *,
    shared_dir: Path,
    runner: ProcessRunner = _SUBPROCESS_RUN,
    timeout_seconds: int,
) -> ContainerRunResult:
    """Wait for a container, collect Docker state, and parse Magma/libFuzzer evidence."""
    timed_out = False
    process_errors: list[str] = []
    wait_exit_code: int | None = None
    try:
        wait_process = _run(runner, ("docker", "wait", container_id), timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process_errors.append(f"docker wait timed out after {timeout_seconds}s")
    except OSError as error:
        process_errors.append(f"docker wait failed: {error}")
    else:
        if wait_process.returncode != 0:
            process_errors.append(_process_failure("docker wait", wait_process))
        wait_exit_code = _parse_exit_code(wait_process.stdout)

    logs_text = _collect_logs(container_id, runner, process_errors)
    state = _inspect_state(container_id, runner, process_errors)
    state_exit_code = state.get("ExitCode")
    exit_code = state_exit_code if isinstance(state_exit_code, int) else wait_exit_code
    docker_status = state.get("Status")
    oom_killed = state.get("OOMKilled")
    return ContainerRunResult(
        container_id=container_id,
        exit_code=exit_code,
        docker_status=docker_status if isinstance(docker_status, str) else None,
        oom_killed=oom_killed if isinstance(oom_killed, bool) else None,
        timed_out=timed_out,
        process_error="; ".join(process_errors) if process_errors else None,
        logs_text=logs_text,
        log_sha256=hashlib.sha256(logs_text.encode()).hexdigest(),
        metrics=_parse_fuzzer_metrics(logs_text),
        canaries=_parse_monitor_rows(shared_dir / "monitor"),
        findings=_collect_findings(shared_dir / "findings"),
    )


def _run(
    runner: ProcessRunner,
    argv: Sequence[str],
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    return runner(
        tuple(argv),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _collect_logs(container_id: str, runner: ProcessRunner, errors: list[str]) -> str:
    try:
        process = _run(runner, ("docker", "logs", "--timestamps", container_id), None)
    except OSError as error:
        errors.append(f"docker logs failed: {error}")
        return ""
    if process.returncode != 0:
        errors.append(_process_failure("docker logs", process))
    return process.stdout + process.stderr


def _inspect_state(
    container_id: str,
    runner: ProcessRunner,
    errors: list[str],
) -> Mapping[str, object]:
    try:
        process = _run(
            runner,
            ("docker", "inspect", "--format", "{{json .State}}", container_id),
            None,
        )
    except OSError as error:
        errors.append(f"docker inspect failed: {error}")
        return {}
    if process.returncode != 0:
        errors.append(_process_failure("docker inspect", process))
        return {}
    try:
        return _STATE_ADAPTER.validate_json(process.stdout)
    except ValidationError:
        errors.append("docker inspect returned invalid state JSON")
        return {}


def _process_failure(name: str, process: subprocess.CompletedProcess[str]) -> str:
    detail = (process.stderr or process.stdout).strip()
    if detail:
        return f"{name} exited {process.returncode}: {detail}"
    return f"{name} exited {process.returncode}"


def _parse_exit_code(stdout: str) -> int | None:
    stripped = stdout.strip()
    return int(stripped) if stripped.isdecimal() else None


def _parse_fuzzer_metrics(logs_text: str) -> FuzzerMetrics:
    coverage_edges = 0
    value_profile_features = 0
    executions_per_second = 0
    for match in _FUZZER_PROGRESS.finditer(logs_text):
        coverage_edges = int(match.group(1))
        value_profile_features = int(match.group(2))
        executions_per_second = int(match.group(3))
    executed_units = 0
    for match in _EXECUTED_UNITS.finditer(logs_text):
        executed_units = int(match.group(1))
    if executed_units == 0:
        for match in _PROGRESS_UNITS.finditer(logs_text):
            executed_units = int(match.group(1))
    return FuzzerMetrics(
        coverage_edges=coverage_edges,
        value_profile_features=value_profile_features,
        executions_per_second=executions_per_second,
        executed_units=executed_units,
    )


def _parse_monitor_rows(monitor_dir: Path) -> Mapping[str, CanarySeries]:
    rows: dict[str, dict[str, list[tuple[int, int]]]] = {}
    if not monitor_dir.exists():
        return {}
    numeric_paths = (item for item in monitor_dir.iterdir() if item.name.isdecimal())
    for path in sorted(numeric_paths, key=lambda item: int(item.name)):
        timestamp = int(path.name)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                _record_monitor_row(rows, timestamp, row)
                break
    return {
        bug: CanarySeries(
            reached=tuple(series.get("R", ())),
            triggered=tuple(series.get("T", ())),
        )
        for bug, series in sorted(rows.items())
    }


def _record_monitor_row(
    rows: dict[str, dict[str, list[tuple[int, int]]]],
    timestamp: int,
    row: Mapping[str, str],
) -> None:
    for key, value in row.items():
        if not key.endswith(("_R", "_T")):
            continue
        bug, metric = key.rsplit("_", 1)
        if value.strip().isdecimal():
            rows.setdefault(bug, {}).setdefault(metric, []).append((timestamp, int(value)))


def _collect_findings(findings_dir: Path) -> tuple[ArtifactEvidence, ...]:
    if not findings_dir.exists():
        return ()
    return tuple(
        artifact_evidence(path) for path in sorted(findings_dir.rglob("*")) if path.is_file()
    )
