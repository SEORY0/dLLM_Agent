"""Narrow deterministic fakes for the OpenEXR one-day Task 4 tests."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, final

from dllm_agent.experiments.openexr_one_day.coverage_series import CoverageStepRequest

if TYPE_CHECKING:
    from dllm_agent.experiments.openexr_one_day.cell_runner import (
        SearchAttemptResult,
        SearchLaunchRequest,
    )
    from dllm_agent.experiments.openexr_one_day.coverage_series import CoverageStepResult
    from dllm_agent.experiments.openexr_one_day.proof_batch import (
        ExactProofRequestModel,
        ProofOutcome,
    )


@final
class ManualClock:
    """Mutable clock whose state advances only through deterministic fakes."""

    def __init__(self, now: float, cpu: float = 0.0) -> None:
        self.now = now
        self.cpu = cpu

    def monotonic(self) -> float:
        """Return the configured monotonic time."""
        return self.now

    def process_time(self) -> float:
        """Return the configured process CPU time."""
        return self.cpu


@dataclass(frozen=True, slots=True)
class ProcessScript:
    """One process exit point plus its normal and forced outcomes."""

    exit_at: float
    result: SearchAttemptResult
    terminated_result: SearchAttemptResult


@final
class ScriptedSearchProcess:
    """Process fake that exits before a requested deadline or stays resident."""

    def __init__(self, clock: ManualClock, script: ProcessScript) -> None:
        self._clock = clock
        self._script = script
        self.terminate_count = 0

    def wait_until(self, deadline_monotonic_seconds: float) -> SearchAttemptResult | None:
        """Return the result only when its scripted exit is within the wait window."""
        if self._script.exit_at <= deadline_monotonic_seconds:
            self._clock.now = self._script.exit_at
            return self._script.result
        self._clock.now = deadline_monotonic_seconds
        return None

    def terminate(self) -> SearchAttemptResult:
        """Record deadline supervision and return the forced outcome."""
        self.terminate_count += 1
        return self._script.terminated_result


@final
class ScriptedSearchLauncher:
    """Launcher fake that records immutable per-attempt requests."""

    def __init__(self, clock: ManualClock, scripts: tuple[ProcessScript, ...]) -> None:
        self._clock = clock
        self._scripts = list(scripts)
        self.requests: list[SearchLaunchRequest] = []
        self.processes: list[ScriptedSearchProcess] = []

    def launch(self, request: SearchLaunchRequest) -> ScriptedSearchProcess:
        """Launch the next scripted process."""
        self.requests.append(request)
        process = ScriptedSearchProcess(self._clock, self._scripts.pop(0))
        self.processes.append(process)
        return process


@final
class FakeSearchPopen:
    """Controllable binary subprocess used at the production launcher boundary."""

    def __init__(self, output: bytes, exit_code: int, *, timeouts: int = 0) -> None:
        self._output = output
        self._exit_code = exit_code
        self._timeouts = timeouts
        self.returncode: int | None = None
        self.terminate_count = 0
        self.kill_count = 0

    def communicate(
        self,
        input: bytes | None = None,  # noqa: A002
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        """Time out as scripted, then expose the terminal output and exit code."""
        _ = input
        if self._timeouts:
            self._timeouts -= 1
            raise subprocess.TimeoutExpired(
                cmd=("docker",), timeout=0.0 if timeout is None else timeout
            )
        self.returncode = self._exit_code
        return self._output, b""

    def terminate(self) -> None:
        """Record a supervisor termination request."""
        self.terminate_count += 1

    def kill(self) -> None:
        """Record a supervisor kill fallback."""
        self.kill_count += 1


@final
class ScriptedCoverageReplayer:
    """Coverage replay fake with literal cumulative profile results."""

    def __init__(self, results: tuple[CoverageStepResult, ...]) -> None:
        self._results = list(results)
        self.requests: list[CoverageStepRequest] = []

    def replay(self, request: CoverageStepRequest) -> CoverageStepResult:
        """Record one incremental replay request and return its result."""
        self.requests.append(request)
        return self._results.pop(0)


def coverage_step(scheduled: int, path: Path, prior: str | None) -> CoverageStepRequest:
    """Build an exact-hash coverage step for one test input."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return CoverageStepRequest(
        scheduled_elapsed_seconds=scheduled,
        corpus_set_sha256=digest,
        new_input_paths=(path,),
        new_input_sha256=(digest,),
        prior_profdata_sha256=prior,
    )


def host_path(artifact_root: Path, container_path: str) -> Path:
    """Map the fixed test Docker mount back to its host artifact path."""
    return artifact_root / Path(container_path).relative_to("/work")


@dataclass(frozen=True, slots=True)
class FakeProofOutcome:
    """Minimal structural proof result returned by the deferred batch fake."""

    oracle_status: Literal["proved", "not_reproduced", "crash_without_sanitizer"]
    input_sha256: str
    proof_log_sha256: str
    proof_log: bytes
    sanitizer_class: str | None
    root_signature: str | None
    run_argv_sha256: tuple[str, ...]
    run_argv_bytes: tuple[bytes, ...]
    result_bytes: bytes


@final
class ScriptedProofRunner:
    """Proof fake that advances independent wall and CPU clocks."""

    def __init__(
        self,
        clock: ManualClock,
        outcomes: tuple[ProofOutcome, ...],
        wall_seconds: float,
        cpu_seconds: float,
    ) -> None:
        self._clock = clock
        self._outcomes = list(outcomes)
        self._wall_seconds = wall_seconds
        self._cpu_seconds = cpu_seconds
        self.requests: list[ExactProofRequestModel] = []

    def run(self, request: ExactProofRequestModel) -> ProofOutcome:
        """Record one unique-input proof and advance only proof accounting."""
        self.requests.append(request)
        self._clock.now += self._wall_seconds
        self._clock.cpu += self._cpu_seconds
        return self._outcomes.pop(0)
