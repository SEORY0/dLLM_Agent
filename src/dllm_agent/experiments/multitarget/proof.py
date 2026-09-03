"""Target-generic independent ASan proof replay for Magma targets."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - request objects expose Path at runtime.
from subprocess import CompletedProcess, TimeoutExpired
from typing import TYPE_CHECKING, Final, Literal, Protocol, override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dllm_agent.experiments.multitarget.docker_runtime import RuntimeProfile

type ProofStatus = Literal["proved", "not_reproduced", "crash_without_sanitizer", "failed"]

_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ASAN_OPTIONS: Final = "detect_leaks=0:abort_on_error=1:symbolize=1"
_ASAN_RE: Final = re.compile(rb"ERROR:\s*AddressSanitizer:\s*([A-Za-z0-9_-]+)")
_FRAME_RE: Final = re.compile(rb"^\s*#\d+\s+0x[0-9a-fA-F]+\s+in\s+(.+)$", re.MULTILINE)
_HEX_RE: Final = re.compile(rb"0x[0-9a-fA-F]+")
_CRASH_RE: Final = re.compile(rb"(Segmentation fault|SIGSEGV|Address not mapped|core dumped)")
_TIMEOUT_EXIT_CODE: Final = 124
_DOCKER_SETUP_EXIT_CODE: Final = 125
_EXPECTED_PROOF_RUNS: Final = 2


class ProofProcessRunner(Protocol):
    """Shell-free subprocess boundary for one proof replay."""

    def __call__(
        self, argv: Sequence[str], *, capture_output: bool, check: bool, timeout: float
    ) -> CompletedProcess[bytes]:
        """Execute one bounded Docker proof command."""
        ...


@dataclass(slots=True)
class ProofReplayError(Exception):
    """Raised when proof inputs violate the replay boundary."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ProofReplayRequest:
    """Pinned inputs for two fresh exact-byte ASan replays."""

    profile: RuntimeProfile
    image_id: str
    input_path: Path
    input_sha256: str
    first_seed: int
    second_seed: int
    first_container_name: str
    second_container_name: str
    first_cpu: int
    second_cpu: int
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class AsanSignature:
    """Normalized sanitizer class and first target-owned frame."""

    sanitizer_class: str
    first_target_frame: str


@dataclass(frozen=True, slots=True)
class ProofRunEvidence:
    """Hash-bound evidence returned by one fresh replay."""

    container_name: str
    proof_seed: int
    return_code: int
    log: bytes
    log_sha256: str
    argv_sha256: str
    signature: AsanSignature | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ProofReplayResult:
    """Two-replay proof verdict and its complete evidence."""

    status: ProofStatus
    profile: RuntimeProfile
    image_id: str
    input_sha256: str
    runs: tuple[ProofRunEvidence, ...]
    signature: AsanSignature | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _RunAttempt:
    container_name: str
    seed: int
    cpu: int


def _run_subprocess(
    argv: Sequence[str], *, capture_output: bool, check: bool, timeout: float
) -> CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603 - fixed argv, shell disabled, Docker only.
        argv,
        capture_output=capture_output,
        check=check,
        timeout=timeout,
    )


def prove_input(
    request: ProofReplayRequest, *, runner: ProofProcessRunner = _run_subprocess
) -> ProofReplayResult:
    """Replay one exact input twice and require a stable ASan signature."""
    _validate_request(request)
    first_attempt, second_attempt = _attempts(request)
    first = _run_once(request, first_attempt, runner)
    second = _run_once(request, second_attempt, runner)
    runs = (first, second)
    classified = _classify_runs(runs)
    return _result(classified.status, request, runs, classified.signature, classified.error)


def proof_result_validation_error(
    request: ProofReplayRequest,
    result: ProofReplayResult,
) -> str | None:
    """Return why persisted proof evidence does not match its executable request."""
    binding_error = _result_binding_error(request, result)
    if binding_error is not None:
        return binding_error
    runs = _two_proof_runs(result.runs)
    if isinstance(runs, str):
        return runs
    run_error = _runs_validation_error(request, runs)
    if run_error is not None:
        return run_error
    return _classification_validation_error(result, runs)


def _result_binding_error(
    request: ProofReplayRequest,
    result: ProofReplayResult,
) -> str | None:
    if result.profile != request.profile:
        return "result profile does not match request"
    if result.image_id != request.image_id:
        return "result image does not match request"
    if result.input_sha256 != request.input_sha256:
        return "result input sha256 does not match request"
    return None


def _two_proof_runs(
    runs: tuple[ProofRunEvidence, ...],
) -> tuple[ProofRunEvidence, ProofRunEvidence] | str:
    if len(runs) != _EXPECTED_PROOF_RUNS:
        return "proof result must contain exactly two runs"
    return (runs[0], runs[1])


def _runs_validation_error(
    request: ProofReplayRequest,
    runs: tuple[ProofRunEvidence, ProofRunEvidence],
) -> str | None:
    for run, attempt in zip(runs, _attempts(request), strict=True):
        if run.container_name != attempt.container_name:
            return "run container name does not match request"
        if run.proof_seed != attempt.seed:
            return "run proof seed does not match request"
        if run.argv_sha256 != _argv_sha256(_docker_argv(request, attempt)):
            return "run argv sha256 does not match request"
        if hashlib.sha256(run.log).hexdigest() != run.log_sha256:
            return "run log sha256 mismatch"
        if run.signature != asan_signature(run.log):
            return "run ASan signature does not match log"
    return None


def _classification_validation_error(
    result: ProofReplayResult,
    runs: tuple[ProofRunEvidence, ProofRunEvidence],
) -> str | None:
    classified = _classify_runs(runs)
    if result.status != classified.status:
        return "proof status does not match replay evidence"
    if result.signature != classified.signature:
        return "proof signature does not match replay evidence"
    if result.error != classified.error:
        return "proof error does not match replay evidence"
    return None


@dataclass(frozen=True, slots=True)
class _ClassifiedRuns:
    status: ProofStatus
    signature: AsanSignature | None
    error: str | None


def _classify_runs(runs: tuple[ProofRunEvidence, ProofRunEvidence]) -> _ClassifiedRuns:
    errors = tuple(run.error for run in runs if run.error is not None)
    if errors:
        return _ClassifiedRuns("failed", None, errors[0])
    signatures = tuple(run.signature for run in runs)
    if signatures[0] is not None and signatures[0] == signatures[1]:
        return _ClassifiedRuns("proved", signatures[0], None)
    if all(_crash_without_asan(run) for run in runs):
        return _ClassifiedRuns("crash_without_sanitizer", None, None)
    return _ClassifiedRuns("not_reproduced", None, None)


def _run_once(
    request: ProofReplayRequest, attempt: _RunAttempt, runner: ProofProcessRunner
) -> ProofRunEvidence:
    argv = _docker_argv(request, attempt)
    try:
        process = runner(
            argv,
            capture_output=True,
            check=False,
            timeout=float(request.timeout_seconds),
        )
    except TimeoutExpired as error:
        log = _timeout_log(error)
        timeout_text = _format_timeout(error.timeout)
        return _run_evidence(attempt, argv, _TIMEOUT_EXIT_CODE, log, timeout_text)
    except OSError as error:
        detail = error.strerror or str(error) or "docker run failed"
        return _run_evidence(attempt, argv, _DOCKER_SETUP_EXIT_CODE, detail.encode(), detail)
    log = process.stdout + process.stderr
    return _run_evidence(attempt, argv, process.returncode, log, None)


def _run_evidence(
    attempt: _RunAttempt, argv: tuple[str, ...], return_code: int, log: bytes, error: str | None
) -> ProofRunEvidence:
    return ProofRunEvidence(
        container_name=attempt.container_name,
        proof_seed=attempt.seed,
        return_code=return_code,
        log=log,
        log_sha256=hashlib.sha256(log).hexdigest(),
        argv_sha256=_argv_sha256(argv),
        signature=asan_signature(log),
        error=error,
    )


def asan_signature(log: bytes) -> AsanSignature | None:
    """Extract a normalized ASan signature from a replay log."""
    marker = _ASAN_RE.search(log)
    if marker is None:
        return None
    frames = tuple(_normalize_frame(match.group(1)) for match in _FRAME_RE.finditer(log))
    first_target = next((frame for frame in frames if _is_target_frame(frame)), "")
    return AsanSignature(
        sanitizer_class=marker.group(1).decode(),
        first_target_frame=first_target,
    )


def _docker_argv(request: ProofReplayRequest, attempt: _RunAttempt) -> tuple[str, ...]:
    return (
        "docker",
        "run",
        "--rm",
        "--name",
        attempt.container_name,
        "--network",
        "none",
        "--cpuset-cpus",
        str(attempt.cpu),
        "--pids-limit",
        "128",
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--env",
        f"ASAN_OPTIONS={_ASAN_OPTIONS}",
        "--volume",
        f"{request.input_path}:/proof/input:ro",
        "--entrypoint",
        f"/magma_out/{request.profile.program}",
        "--pull",
        "never",
        request.image_id,
        f"-seed={attempt.seed}",
        "-runs=1",
        "/proof/input",
    )


def _validate_request(request: ProofReplayRequest) -> None:
    if _IMAGE_DIGEST.fullmatch(request.image_id) is None:
        raise ProofReplayError(reason="image_id must be a sha256 digest")
    if not request.input_path.is_absolute():
        raise ProofReplayError(reason="input_path must be absolute")
    if not request.input_path.is_file():
        raise ProofReplayError(reason="input_path must be a regular file")
    actual_sha256 = hashlib.sha256(request.input_path.read_bytes()).hexdigest()
    if actual_sha256 != request.input_sha256:
        raise ProofReplayError(reason="input sha256 mismatch")
    if request.first_seed == request.second_seed:
        raise ProofReplayError(reason="proof seeds must be distinct")
    if request.first_container_name == request.second_container_name:
        raise ProofReplayError(reason="container names must be distinct")
    if request.first_cpu == request.second_cpu:
        raise ProofReplayError(reason="proof CPUs must be distinct")
    _validate_run_boundary(request.first_seed, request.first_container_name, request.first_cpu)
    _validate_run_boundary(request.second_seed, request.second_container_name, request.second_cpu)
    if request.timeout_seconds <= 0:
        raise ProofReplayError(reason="timeout_seconds must be positive")


def _validate_run_boundary(seed: int, container_name: str, cpu: int) -> None:
    if seed < 0:
        raise ProofReplayError(reason="proof seed must be nonnegative")
    if _CONTAINER_NAME.fullmatch(container_name) is None:
        raise ProofReplayError(reason="container name is not docker-safe")
    if cpu < 0:
        raise ProofReplayError(reason="proof CPU must be nonnegative")


def _attempts(request: ProofReplayRequest) -> tuple[_RunAttempt, _RunAttempt]:
    return (
        _RunAttempt(request.first_container_name, request.first_seed, request.first_cpu),
        _RunAttempt(request.second_container_name, request.second_seed, request.second_cpu),
    )


def _crash_without_asan(run: ProofRunEvidence) -> bool:
    return run.signature is None and (
        run.return_code in {134, 136, 139} or _CRASH_RE.search(run.log) is not None
    )


def _result(
    status: ProofStatus,
    request: ProofReplayRequest,
    runs: tuple[ProofRunEvidence, ...],
    signature: AsanSignature | None,
    error: str | None,
) -> ProofReplayResult:
    return ProofReplayResult(
        status=status,
        profile=request.profile,
        image_id=request.image_id,
        input_sha256=request.input_sha256,
        runs=runs,
        signature=signature,
        error=error,
    )


def _is_target_frame(frame: str) -> bool:
    lower = frame.lower()
    return "llvmfuzzer" not in lower and "sanitizer" not in lower and "fuzzer" not in lower


def _normalize_frame(frame: bytes) -> str:
    return _HEX_RE.sub(b"0xADDR", frame).decode(errors="replace")


def _argv_sha256(argv: tuple[str, ...]) -> str:
    return hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()


def _format_timeout(timeout: float | None) -> str:
    if timeout is None:
        return "docker run timed out"
    return f"docker run timed out after {timeout:g}s"


def _timeout_log(error: TimeoutExpired) -> bytes:
    output = error.stdout if isinstance(error.stdout, bytes) else b""
    stderr = error.stderr if isinstance(error.stderr, bytes) else b""
    return output + stderr
